#!/usr/bin/env python3
"""Probe EP32 AsyncLL with the K3 packed-MXFP4 to FP8 AITER MoE path."""

import os
import time

import torch
import torch.distributed as dist

import mori
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from sglang.kernels.ops.moe.rocm_moe_utils import upscale_mxfp4


WORLD_SIZE = 32
HIDDEN_SIZE = 3584
INTERMEDIATE_SIZE = 3072
TOPK = 16
NUM_EXPERTS = 896
EXPERTS_PER_RANK = NUM_EXPERTS // WORLD_SIZE
ITERATIONS = int(os.environ.get("PROBE_ITERATIONS", "16"))


def sync(label: str, rank: int) -> None:
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(f"sync_ok label={label}", flush=True)


def make_shuffled_weights(device: torch.device):
    w1_raw = torch.zeros(
        (EXPERTS_PER_RANK, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 2),
        dtype=torch.uint8,
        device=device,
    ).view(dtypes.fp4x2)
    w2_raw = torch.zeros(
        (EXPERTS_PER_RANK, HIDDEN_SIZE, INTERMEDIATE_SIZE // 2),
        dtype=torch.uint8,
        device=device,
    ).view(dtypes.fp4x2)
    w1_scale_raw = torch.ones(
        (EXPERTS_PER_RANK, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 32),
        dtype=dtypes.fp8_e8m0,
        device=device,
    )
    w2_scale_raw = torch.ones(
        (EXPERTS_PER_RANK, HIDDEN_SIZE, INTERMEDIATE_SIZE // 32),
        dtype=dtypes.fp8_e8m0,
        device=device,
    )

    w1 = shuffle_weight_a16w4(w1_raw, 16, True)
    w2 = shuffle_weight_a16w4(w2_raw, 16, False)
    w1_scale = shuffle_scale_a16w4(
        w1_scale_raw.view(-1, w1_scale_raw.shape[-1]),
        EXPERTS_PER_RANK,
        True,
    )
    w2_scale = shuffle_scale_a16w4(
        w2_scale_raw.view(-1, w2_scale_raw.shape[-1]),
        EXPERTS_PER_RANK,
        False,
    )
    w1.is_shuffled = True
    w2.is_shuffled = True
    return w1, w2, w1_scale, w2_scale


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"expected world_size={WORLD_SIZE}, got {world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    torch._C._distributed_c10d._register_process_group(
        "default", torch.distributed.group.WORLD
    )
    mori.shmem.shmem_torch_process_group_init("default")

    config = mori.ops.EpDispatchCombineConfig(
        data_type=torch.float4_e2m1fn_x2,
        rank=rank,
        world_size=world_size,
        hidden_dim=HIDDEN_SIZE,
        scale_dim=HIDDEN_SIZE // 32,
        scale_type_size=torch.float8_e8m0fnu.itemsize,
        max_token_type_size=torch.bfloat16.itemsize,
        max_num_inp_token_per_rank=1,
        num_experts_per_rank=EXPERTS_PER_RANK,
        num_experts_per_token=TOPK,
        warp_num_per_block=8,
        block_num=64,
        rdma_block_num=32,
        max_total_recv_tokens=32,
        use_external_inp_buf=True,
        kernel_type=mori.ops.EpDispatchCombineKernelType.AsyncLL,
        gpu_per_node=8,
        num_qp_per_pe=2,
        quant_type="fp8_blockwise",
    )
    op = mori.ops.EpDispatchCombineOp(config)
    max_recv = op.max_num_tokens_to_recv()
    w1, w2, w1_scale, w2_scale = make_shuffled_weights(device)
    expert_mask = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)
    expert_begin = rank * EXPERTS_PER_RANK
    expert_mask[expert_begin : expert_begin + EXPERTS_PER_RANK] = 1

    if rank == 0:
        props = torch.cuda.get_device_properties(local_rank)
        print(
            "probe_contract "
            f"gpu={props.name} arch={props.gcnArchName} world_size={world_size} "
            f"active_ranks_per_iteration=8 idle_ranks_per_iteration=24 "
            f"topk={TOPK} local_experts={EXPERTS_PER_RANK} "
            f"dispatch=packed_mxfp4 compute_input=fp8 "
            f"expert_weight_storage=mxfp4 combine=fp8_blockwise output=bf16 "
            f"max_recv={max_recv}",
            flush=True,
        )

    sync("initialized", rank)
    started = time.monotonic()
    for iteration in range(ITERATIONS):
        active_dp = iteration % 4
        active_begin = active_dp * 8
        active = active_begin <= rank < active_begin + 8
        num_tokens = 1 if active else 0

        raw = torch.full(
            (num_tokens, HIDDEN_SIZE // 2),
            fill_value=(0x11 + iteration) & 0xFF,
            dtype=torch.uint8,
            device=device,
        )
        dispatch_input = raw.view(torch.float4_e2m1fn_x2)
        dispatch_scales = torch.ones(
            (num_tokens, HIDDEN_SIZE // 32),
            dtype=torch.float8_e8m0fnu,
            device=device,
        )
        dispatch_weights = torch.full(
            (num_tokens, TOPK),
            fill_value=1.0 / TOPK,
            dtype=torch.float32,
            device=device,
        )
        if num_tokens:
            expert_ids = [
                (rank * TOPK + iteration + offset * EXPERTS_PER_RANK) % NUM_EXPERTS
                for offset in range(TOPK)
            ]
            dispatch_indices = torch.tensor(
                [expert_ids], dtype=torch.int32, device=device
            )
        else:
            dispatch_indices = torch.empty(
                (0, TOPK), dtype=torch.int32, device=device
            )

        (
            dispatch_output,
            dispatch_output_weights,
            dispatch_output_scales,
            dispatch_output_indices,
            dispatch_recv_count,
        ) = op.dispatch_send(
            dispatch_input,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
        )
        op.dispatch_recv()

        compute_input = upscale_mxfp4(
            dispatch_output,
            dispatch_output_scales,
            dispatch_recv_count,
            torch.bfloat16,
        )
        expert_output = fused_moe(
            hidden_states=compute_input,
            w1=w1,
            w2=w2,
            topk_weight=dispatch_output_weights,
            topk_ids=dispatch_output_indices,
            expert_mask=expert_mask,
            activation=ActivationType.Situv2,
            quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            num_local_tokens=dispatch_recv_count,
            dtype=torch.bfloat16,
            q_dtype_a=dtypes.fp8,
            beta=4.0,
            linear_beta=25.0,
            gate_mode=GateMode.INTERLEAVE.value,
            fake_topk_slots=0,
        )
        combined_output, _ = op.combine_send(
            expert_output,
            None,
            dispatch_indices,
        )
        op.combine_recv()
        sync(f"iteration_{iteration}", rank)

        recv_count = int(dispatch_recv_count.item())
        if recv_count > max_recv:
            raise RuntimeError(
                f"rank={rank} recv_count={recv_count} exceeds max_recv={max_recv}"
            )
        if recv_count and not torch.isfinite(expert_output[:recv_count]).all():
            raise RuntimeError(f"rank={rank} non-finite expert output")
        if num_tokens and not torch.isfinite(combined_output[:num_tokens]).all():
            raise RuntimeError(f"rank={rank} non-finite combine output")
        if rank == 0:
            print(
                f"iteration_ok iteration={iteration} active_dp={active_dp} "
                f"rank0_source_tokens={num_tokens} rank0_recv_tokens={recv_count}",
                flush=True,
            )

    elapsed = time.monotonic() - started
    sync("completed", rank)
    if rank == 0:
        print(
            f"MORI_AITER_EP32_MIXED_IDLE_PASS=1 iterations={ITERATIONS} "
            f"elapsed_seconds={elapsed:.6f} "
            f"allocated_gib={torch.cuda.memory_allocated() / 2**30:.6f}",
            flush=True,
        )

    mori.shmem.shmem_finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
