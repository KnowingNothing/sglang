#!/usr/bin/env python3
"""Probe K3 MORI normal-prefill to AsyncLL-decode transition on EP32."""

import os
import time

import torch
import torch.distributed as dist

import mori
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4


WORLD_SIZE = 32
HIDDEN_SIZE = 3584
INTERMEDIATE_SIZE = 3072
TOPK = 16
NUM_EXPERTS = 896
EXPERTS_PER_RANK = NUM_EXPERTS // WORLD_SIZE
PREFILL_TOKENS_PER_ACTIVE_RANK = 394
PREFILL_LAYERS = int(os.environ.get("PROBE_PREFILL_LAYERS", "1"))
PREFILL_SYNC_INTERVAL = int(os.environ.get("PROBE_PREFILL_SYNC_INTERVAL", "0"))
DECODE_ITERATIONS = int(os.environ.get("PROBE_DECODE_ITERATIONS", "128"))


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


def make_expert_mask(rank: int, device: torch.device) -> torch.Tensor:
    expert_mask = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)
    expert_begin = rank * EXPERTS_PER_RANK
    expert_mask[expert_begin : expert_begin + EXPERTS_PER_RANK] = 1
    return expert_mask


def make_inputs(
    *,
    rank: int,
    num_tokens: int,
    iteration: int,
    device: torch.device,
):
    hidden = torch.full(
        (num_tokens, HIDDEN_SIZE),
        fill_value=(iteration % 7 + 1) / 16.0,
        dtype=torch.bfloat16,
        device=device,
    )
    weights = torch.full(
        (num_tokens, TOPK),
        fill_value=1.0 / TOPK,
        dtype=torch.float32,
        device=device,
    )
    if num_tokens:
        token_offsets = torch.arange(num_tokens, dtype=torch.int32, device=device)[
            :, None
        ]
        slot_offsets = torch.arange(TOPK, dtype=torch.int32, device=device)[None, :]
        indices = (
            rank * TOPK
            + iteration
            + token_offsets
            + slot_offsets * EXPERTS_PER_RANK
        ) % NUM_EXPERTS
    else:
        indices = torch.empty((0, TOPK), dtype=torch.int32, device=device)
    return hidden, weights, indices


def run_aiter(
    *,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    recv_count: torch.Tensor,
    expert_mask: torch.Tensor,
    weights,
):
    w1, w2, w1_scale, w2_scale = weights
    expert_output = fused_moe(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weight=topk_weights,
        topk_ids=topk_ids,
        expert_mask=expert_mask,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        num_local_tokens=recv_count,
        dtype=torch.bfloat16,
        q_dtype_a=dtypes.fp8,
        beta=4.0,
        linear_beta=25.0,
        gate_mode=GateMode.INTERLEAVE.value,
        fake_topk_slots=0,
    )
    return expert_output


def make_normal_op(rank: int):
    config = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=WORLD_SIZE,
        hidden_dim=HIDDEN_SIZE,
        scale_dim=0,
        scale_type_size=torch.float32.itemsize,
        max_token_type_size=torch.bfloat16.itemsize,
        max_num_inp_token_per_rank=16384,
        num_experts_per_rank=EXPERTS_PER_RANK,
        num_experts_per_token=TOPK,
        warp_num_per_block=8,
        block_num=64,
        rdma_block_num=32,
        max_total_recv_tokens=524288,
        use_external_inp_buf=True,
        kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV1,
        gpu_per_node=8,
        num_qp_per_pe=2,
        quant_type="none",
    )
    return mori.ops.EpDispatchCombineOp(config)


def make_asyncll_op(rank: int):
    config = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=WORLD_SIZE,
        hidden_dim=HIDDEN_SIZE,
        scale_dim=0,
        scale_type_size=torch.float32.itemsize,
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
        quant_type="none",
    )
    return mori.ops.EpDispatchCombineOp(config)


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

    weights = make_shuffled_weights(device)
    expert_mask = make_expert_mask(rank, device)
    if rank == 0:
        props = torch.cuda.get_device_properties(local_rank)
        print(
            "probe_contract "
            f"gpu={props.name} arch={props.gcnArchName} world_size={world_size} "
            f"prefill_active_ranks=8 prefill_tokens_per_active_rank="
            f"{PREFILL_TOKENS_PER_ACTIVE_RANK} prefill_layers={PREFILL_LAYERS} "
            f"prefill_sync_interval={PREFILL_SYNC_INTERVAL} decode_active_ranks=8 "
            f"decode_idle_ranks=24 decode_iterations={DECODE_ITERATIONS} "
            f"topk={TOPK} local_experts={EXPERTS_PER_RANK} "
            f"normal_max_input=16384 normal_max_recv=524288 "
            f"decode_max_input=1 decode_max_recv=32 "
            f"dispatch=bf16 combine=bf16 expert_weight_storage=mxfp4 "
            f"activation_compute=fp8",
            flush=True,
        )

    normal_op = make_normal_op(rank)
    sync("normal_initialized", rank)
    prefill_tokens = PREFILL_TOKENS_PER_ACTIVE_RANK if rank < 8 else 0
    combined = None
    recv_count = None
    for layer in range(PREFILL_LAYERS):
        hidden, topk_weights, topk_ids = make_inputs(
            rank=rank,
            num_tokens=prefill_tokens,
            iteration=layer,
            device=device,
        )
        (
            recv_hidden,
            recv_topk_weights,
            _recv_scales,
            recv_topk_ids,
            recv_count,
        ) = normal_op.dispatch(hidden, topk_weights, None, topk_ids)
        expert_output = run_aiter(
            hidden_states=recv_hidden,
            topk_weights=recv_topk_weights,
            topk_ids=recv_topk_ids,
            recv_count=recv_count,
            expert_mask=expert_mask,
            weights=weights,
        )
        combined, _ = normal_op.combine(expert_output, None, topk_ids)
        if PREFILL_SYNC_INTERVAL and (layer + 1) % PREFILL_SYNC_INTERVAL == 0:
            sync(f"normal_layer_{layer}", rank)
        if rank == 0 and (layer < 4 or (layer + 1) % 16 == 0):
            print(f"normal_layer_enqueued layer={layer}", flush=True)
        del recv_hidden, recv_topk_weights, recv_topk_ids, expert_output

    sync("normal_layers_completed", rank)
    assert combined is not None
    assert recv_count is not None
    if prefill_tokens and not torch.isfinite(combined[:prefill_tokens]).all():
        raise RuntimeError(f"rank={rank} non-finite normal combine output")
    if rank == 0:
        print(
            f"normal_ok layers={PREFILL_LAYERS} rank0_source_tokens={prefill_tokens} "
            f"rank0_recv_tokens={int(recv_count.item())}",
            flush=True,
        )
    del combined

    asyncll_op = make_asyncll_op(rank)
    sync("asyncll_initialized_after_normal", rank)
    started = time.monotonic()
    for iteration in range(DECODE_ITERATIONS):
        active_dp = iteration % 4
        active_begin = active_dp * 8
        num_tokens = 1 if active_begin <= rank < active_begin + 8 else 0
        hidden, topk_weights, topk_ids = make_inputs(
            rank=rank,
            num_tokens=num_tokens,
            iteration=iteration + 1,
            device=device,
        )
        (
            recv_hidden,
            recv_topk_weights,
            _recv_scales,
            recv_topk_ids,
            recv_count,
        ) = asyncll_op.dispatch_send(hidden, topk_weights, None, topk_ids)
        asyncll_op.dispatch_recv()
        expert_output = run_aiter(
            hidden_states=recv_hidden,
            topk_weights=recv_topk_weights,
            topk_ids=recv_topk_ids,
            recv_count=recv_count,
            expert_mask=expert_mask,
            weights=weights,
        )
        combined, _ = asyncll_op.combine_send(expert_output, None, topk_ids)
        asyncll_op.combine_recv()
        sync(f"decode_iteration_{iteration}", rank)
        if num_tokens and not torch.isfinite(combined[:num_tokens]).all():
            raise RuntimeError(f"rank={rank} non-finite decode combine output")
        if rank == 0 and (iteration < 4 or (iteration + 1) % 16 == 0):
            print(
                f"decode_iteration_ok iteration={iteration} active_dp={active_dp} "
                f"rank0_source_tokens={num_tokens} "
                f"rank0_recv_tokens={int(recv_count.item())}",
                flush=True,
            )

    elapsed = time.monotonic() - started
    sync("completed", rank)
    if rank == 0:
        print(
            "MORI_AITER_EP32_NORMAL_TO_ASYNCLL_PASS=1 "
            f"decode_iterations={DECODE_ITERATIONS} "
            f"decode_elapsed_seconds={elapsed:.6f} "
            f"allocated_gib={torch.cuda.memory_allocated() / 2**30:.6f} "
            f"reserved_gib={torch.cuda.memory_reserved() / 2**30:.6f}",
            flush=True,
        )

    mori.shmem.shmem_finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
