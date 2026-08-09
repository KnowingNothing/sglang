"""MI308X/gfx942 native MXFP4-storage, FP8-compute MoE regressions."""

import functools

import pytest
import torch

import aiter.fused_moe as fused_moe_module
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
from aiter.ops.quant import (
    mxfp4_moe_sort_fwd,
    per_1x32_f8_scale_f8_quant,
    per_1x32_mx_quant_hip,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4


pytestmark = pytest.mark.skipif(get_gfx() != "gfx942", reason="gfx942-only path")


def _nested_kernel_name(call):
    current = call
    while isinstance(current, functools.partial):
        if "kernelName" in (current.keywords or {}):
            return current.keywords["kernelName"]
        current = current.func
    return None


def test_public_a8w4_and_mori_live_prefix(monkeypatch):
    """Public q_dtype override and MORI global-id/live-prefix stay exact."""
    monkeypatch.delenv("AITER_SITUV2_A8W4", raising=False)
    torch.set_default_device("cuda")
    torch.manual_seed(20260808)

    tokens, model_dim, inter_dim, experts, topk, block_m = 16, 256, 256, 8, 1, 16
    hidden = torch.randn((tokens, model_dim), dtype=torch.bfloat16) / 10
    score = torch.randn((tokens, experts), dtype=torch.bfloat16)
    topk_weights, topk_ids = fused_topk(hidden, score, topk, True)

    w1 = torch.randint(
        0,
        256,
        (experts, 2 * inter_dim, model_dim // 2),
        dtype=torch.uint8,
    ).view(dtypes.fp4x2)
    w2 = torch.randint(
        0,
        256,
        (experts, model_dim, inter_dim // 2),
        dtype=torch.uint8,
    ).view(dtypes.fp4x2)
    w1_scale = torch.randint(
        116,
        124,
        (experts * 2 * inter_dim, model_dim // 32),
        dtype=torch.uint8,
    )
    w2_scale = torch.randint(
        116,
        124,
        (experts * model_dim, inter_dim // 32),
        dtype=torch.uint8,
    )
    w1 = shuffle_weight_a16w4(w1, 16, True)
    w2 = shuffle_weight_a16w4(w2, 16, False)
    w1_scale = shuffle_scale_a16w4(w1_scale, experts, True)
    w2_scale = shuffle_scale_a16w4(w2_scale, experts, False)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids,
        topk_weights,
        experts,
        model_dim,
        torch.bfloat16,
        block_m,
    )
    x1, x1_scale = per_1x32_mx_quant_hip(
        hidden,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    x1_scale = mxfp4_moe_sort_fwd(
        x1_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=tokens,
        cols=model_dim,
    )
    stage1 = flydsl_moe_stage1(
        a=x1,
        w1=w1,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk,
        tile_m=16,
        tile_n=256,
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        w1_scale=w1_scale,
        a1_scale=x1_scale,
        act="situv2",
        situ_beta=4.0,
        situ_linear_beta=25.0,
        gate_mode="interleave",
        waves_per_eu=3,
    )
    x2, x2_scale = per_1x32_f8_scale_f8_quant(
        stage1,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
    )
    x2_scale = mxfp4_moe_sort_fwd(
        x2_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=tokens,
        cols=inter_dim,
    )
    reference = flydsl_moe_stage2(
        inter_states=x2,
        w2=w2,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk,
        tile_m=16,
        tile_n=128,
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        mode="atomic",
        w2_scale=w2_scale,
        a2_scale=x2_scale,
        sorted_weights=sorted_weights,
    )

    fused_moe_module.kernel_bench_callable = []
    actual = fused_moe(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        q_dtype_a=dtypes.fp8,
        beta=4.0,
        linear_beta=25.0,
        gate_mode=GateMode.INTERLEAVE.value,
    )
    torch.cuda.synchronize()
    kernel_names = [
        (label, _nested_kernel_name(call))
        for label, call in fused_moe_module.kernel_bench_callable
    ]
    fused_moe_module.kernel_bench_callable = None

    arena_rows = 32
    global_experts = 128
    local_start = 5 * experts
    arena_hidden = torch.randn(
        (arena_rows, model_dim), dtype=torch.bfloat16
    ) / 10
    arena_hidden[:tokens] = hidden
    arena_ids = torch.zeros((arena_rows, topk), dtype=torch.int32)
    arena_ids[:tokens] = topk_ids.to(torch.int32) + local_start
    arena_weights = torch.zeros((arena_rows, topk), dtype=torch.float32)
    arena_weights[:tokens] = topk_weights
    expert_mask = torch.zeros((global_experts,), dtype=torch.int32)
    expert_mask[local_start : local_start + experts] = 1
    num_local_tokens = torch.tensor([tokens], dtype=torch.int32)

    original_mxfp8_quant_moe_sort = (
        fused_moe_module.fused_dynamic_mxfp8_quant_moe_sort
    )
    observed_num_rows = []

    def record_mxfp8_quant_moe_sort(*args, num_rows=None, **kwargs):
        observed_num_rows.append(num_rows)
        return original_mxfp8_quant_moe_sort(
            *args, num_rows=num_rows, **kwargs
        )

    monkeypatch.setattr(
        fused_moe_module,
        "fused_dynamic_mxfp8_quant_moe_sort",
        record_mxfp8_quant_moe_sort,
    )
    actual_mori = fused_moe(
        arena_hidden,
        w1,
        w2,
        arena_weights,
        arena_ids,
        expert_mask=expert_mask,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        num_local_tokens=num_local_tokens,
        q_dtype_a=dtypes.fp8,
        beta=4.0,
        linear_beta=25.0,
        gate_mode=GateMode.INTERLEAVE.value,
    )
    torch.cuda.synchronize()

    assert w1.dtype == dtypes.fp4x2 and w1.element_size() == 1
    assert kernel_names == [
        ("stage1", "flydsl_moe1_afp8_wfp4_bf16_t16x256x256_w3_bnt0_gui"),
        ("stage2", "flydsl_moe2_afp8_wfp4_bf16_t16x128x256_atomic"),
    ]
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(actual_mori[:tokens], actual, atol=0, rtol=0)
    assert torch.isfinite(actual_mori[:tokens]).all()
    assert len(observed_num_rows) == 2
    assert all(num_rows is num_local_tokens for num_rows in observed_num_rows)
