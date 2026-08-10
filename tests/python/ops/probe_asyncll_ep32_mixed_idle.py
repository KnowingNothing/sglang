#!/usr/bin/env python3
"""Probe DP4/TP8-to-EP32 AsyncLL collectives with mixed active/idle ranks."""

import os
import time

import torch
import torch.distributed as dist

import mori


WORLD_SIZE = 32
HIDDEN_SIZE = 3584
TOPK = 16
NUM_EXPERTS = 896
EXPERTS_PER_RANK = NUM_EXPERTS // WORLD_SIZE
ITERATIONS = 16


def sync(label: str, rank: int) -> None:
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(f"sync_ok label={label}", flush=True)


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
        max_token_type_size=4,
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
    if rank == 0:
        props = torch.cuda.get_device_properties(local_rank)
        print(
            "probe_contract "
            f"gpu={props.name} arch={props.gcnArchName} world_size={world_size} "
            f"active_ranks_per_iteration=8 idle_ranks_per_iteration=24 "
            f"dispatch=packed_mxfp4 combine=fp8_blockwise output=bf16 "
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
        if dispatch_input.shape != (num_tokens, HIDDEN_SIZE // 2):
            raise RuntimeError(
                f"rank={rank} unexpected packed FP4 logical shape "
                f"{tuple(dispatch_input.shape)}"
            )
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
            _dispatch_output,
            _dispatch_output_weights,
            _dispatch_output_scales,
            _dispatch_output_indices,
            dispatch_recv_count,
        ) = op.dispatch_send(
            dispatch_input,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
        )
        op.dispatch_recv()
        sync(f"iteration_{iteration}_dispatch", rank)

        recv_count = int(dispatch_recv_count.item())
        if recv_count > max_recv:
            raise RuntimeError(
                f"rank={rank} recv_count={recv_count} exceeds max_recv={max_recv}"
            )
        expert_output = torch.zeros(
            (max_recv, HIDDEN_SIZE), dtype=torch.bfloat16, device=device
        )
        if recv_count:
            expert_output[:recv_count].fill_(float(iteration + 1))
        combined_output, _ = op.combine_send(
            expert_output,
            None,
            dispatch_indices,
        )
        op.combine_recv()
        sync(f"iteration_{iteration}_combine", rank)

        if num_tokens:
            sample = combined_output[:num_tokens].float()
            if not torch.isfinite(sample).all():
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
            f"MORI_ASYNCLL_EP32_MIXED_IDLE_PASS=1 iterations={ITERATIONS} "
            f"elapsed_seconds={elapsed:.6f}",
            flush=True,
        )

    mori.shmem.shmem_finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
