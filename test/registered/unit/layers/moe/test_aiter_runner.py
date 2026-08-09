import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import sglang.srt.layers.moe.moe_runner.aiter as aiter_runner
from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    AiterRunnerCore,
    AiterRunnerInput,
    _pre_permute_deepep_to_aiter,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.moriep import (
    MoriEPLLDispatchOutput,
    MoriEPNormalDispatchOutput,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def _runner_input():
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    return AiterRunnerInput(
        hidden_states=torch.zeros((1, 4), dtype=torch.bfloat16),
        topk_ids=topk_ids,
        topk_weights=torch.ones(topk_ids.shape, dtype=torch.float32),
        quant_type=AiterQuantType.PER_1X32,
    )


def _quant_info(**overrides):
    kwargs = {
        "w13_weight": torch.empty((2, 8, 2)),
        "w2_weight": torch.empty((2, 4, 2)),
        "quant_type": AiterQuantType.PER_1X32,
    }
    kwargs.update(overrides)
    return AiterMoeQuantInfo(**kwargs)


def _install_fake_aiter(monkeypatch, fused_moe):
    fake_aiter = ModuleType("aiter")
    fake_aiter.__path__ = []
    fake_aiter.ActivationType = SimpleNamespace(Silu="Silu", Situv2="Situv2")
    fake_aiter.QuantType = SimpleNamespace(per_1x32="per_1x32")
    fake_aiter.dtypes = SimpleNamespace(fp8="fp8", bf16="bf16")

    fake_fused_moe = ModuleType("aiter.fused_moe")
    fake_fused_moe.fused_moe = fused_moe

    fake_ops = ModuleType("aiter.ops")
    fake_ops.__path__ = []
    fake_flydsl = ModuleType("aiter.ops.flydsl")
    fake_flydsl.__path__ = []
    fake_moe_common = ModuleType("aiter.ops.flydsl.moe_common")
    fake_moe_common.GateMode = SimpleNamespace(
        INTERLEAVE=SimpleNamespace(value="interleave"),
        SEPARATED=SimpleNamespace(value="separated"),
    )

    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", fake_fused_moe)
    monkeypatch.setitem(sys.modules, "aiter.ops", fake_ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", fake_flydsl)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.moe_common", fake_moe_common)


def test_aiter_runner_forwards_no_combine_and_extra_fused_moe_kwargs(monkeypatch):
    captured = {}

    def fused_moe(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fused_moe)
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: True
    )

    runner = AiterRunnerCore(MoeRunnerConfig(activation="silu", no_combine=True))

    runner.run(
        _runner_input(),
        _quant_info(fused_moe_kwargs={"custom_fused_moe_kwarg": "enabled"}),
        running_state={},
    )

    assert captured["activation"] == "Silu"
    assert captured["quant_type"] == "per_1x32"
    assert captured["no_combine"] is True
    assert captured["custom_fused_moe_kwarg"] == "enabled"


def test_aiter_runner_rejects_no_combine_when_fused_moe_does_not_support_it(
    monkeypatch,
):
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: False
    )
    runner = AiterRunnerCore(MoeRunnerConfig(no_combine=True))

    with pytest.raises(NotImplementedError, match="no_combine=True"):
        runner.run(_runner_input(), _quant_info(), running_state={})


def test_aiter_runner_preserves_no_combine_rank_for_empty_input(monkeypatch):
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: True
    )
    runner = AiterRunnerCore(MoeRunnerConfig(no_combine=True))
    runner_input = _runner_input()
    runner_input.hidden_states = torch.zeros((0, 4), dtype=torch.bfloat16)
    runner_input.topk_ids = torch.zeros((0, 2), dtype=torch.int32)
    runner_input.topk_weights = torch.zeros((0, 2), dtype=torch.float32)

    output = runner.run(runner_input, _quant_info(), running_state={})

    assert output.hidden_states.shape == (0, 2, 4)


def test_mori_fp4_situ_restores_bf16_before_selected_mxfp4_compute(monkeypatch):
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is None:
        pytest.skip("torch build does not expose packed FP4")

    hidden_states = torch.empty((2, 2), dtype=fp4_dtype)
    hidden_states_scale = torch.ones((2, 1), dtype=torch.uint8)
    num_local_tokens = torch.tensor([2], dtype=torch.int32)
    topk_ids = torch.zeros((2, 1), dtype=torch.int32)
    topk_weights = torch.ones((2, 1), dtype=torch.float32)
    dispatch_output = SimpleNamespace(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        num_recv_tokens_per_expert=num_local_tokens,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        origin_topk_ids=topk_ids,
        origin_topk_weights=topk_weights,
        out_dtype=torch.bfloat16,
    )

    from sglang.kernels.ops.moe import rocm_moe_utils

    calls = []

    def fake_upscale_mxfp4(hidden, scale, rows, output_dtype):
        calls.append((hidden, scale, rows, output_dtype))
        return torch.zeros((2, 4), dtype=output_dtype)

    monkeypatch.setattr(rocm_moe_utils, "upscale_mxfp4", fake_upscale_mxfp4)
    runner_input = _pre_permute_deepep_to_aiter(
        dispatch_output,
        _quant_info(),
        MoeRunnerConfig(activation="situ", num_local_experts=1),
        running_state={},
    )

    assert len(calls) == 1
    assert calls[0][0] is hidden_states
    assert calls[0][1] is hidden_states_scale
    assert calls[0][2] is num_local_tokens
    assert calls[0][3] is torch.bfloat16
    assert runner_input.hidden_states.dtype is torch.bfloat16
    assert runner_input.a1_scale is None
    assert runner_input.quant_type is AiterQuantType.PER_1X32
    assert runner_input.num_local_tokens is num_local_tokens


def _mori_dispatch_output(cls, *, capacity=8, live_tokens=2):
    topk_ids = torch.arange(capacity, dtype=torch.int32).view(capacity, 1)
    topk_weights = torch.arange(capacity, dtype=torch.float32).view(capacity, 1)
    return cls(
        hidden_states=torch.arange(
            capacity * 4, dtype=torch.bfloat16
        ).view(capacity, 4),
        hidden_states_scale=None,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        num_recv_tokens_per_expert=torch.tensor([live_tokens], dtype=torch.int32),
        origin_topk_ids=topk_ids,
        origin_topk_weights=topk_weights,
        out_dtype=torch.bfloat16,
    )


def test_mori_normal_aiter_slices_fixed_arena_to_live_prefix():
    dispatch_output = _mori_dispatch_output(MoriEPNormalDispatchOutput)

    runner_input = _pre_permute_deepep_to_aiter(
        dispatch_output,
        _quant_info(),
        MoeRunnerConfig(activation="situ", num_local_experts=1),
        running_state={},
    )

    assert runner_input.hidden_states.shape == (2, 4)
    assert runner_input.topk_ids.shape == (2, 1)
    assert runner_input.topk_weights.shape == (2, 1)
    assert runner_input.num_local_tokens is dispatch_output.num_recv_tokens_per_expert


def test_mori_low_latency_aiter_keeps_small_static_arena():
    dispatch_output = _mori_dispatch_output(MoriEPLLDispatchOutput)

    runner_input = _pre_permute_deepep_to_aiter(
        dispatch_output,
        _quant_info(),
        MoeRunnerConfig(activation="situ", num_local_experts=1),
        running_state={},
    )

    assert runner_input.hidden_states.shape == (8, 4)
    assert runner_input.topk_ids.shape == (8, 1)
    assert runner_input.topk_weights.shape == (8, 1)
    assert runner_input.num_local_tokens is dispatch_output.num_recv_tokens_per_expert


@pytest.mark.parametrize(
    ("compute_dtype", "expected_gate_mode", "expected_q_dtype_a"),
    [("fp8", "interleave", "fp8"), ("bf16", "separated", "bf16")],
)
def test_situ_gate_mode_matches_selected_mxfp4_weight_layout(
    monkeypatch, compute_dtype, expected_gate_mode, expected_q_dtype_a
):
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_server_args",
        lambda: SimpleNamespace(mxfp4_moe_compute_dtype=compute_dtype),
    )

    captured = {}

    def fake_fused_moe(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fake_fused_moe)

    core = AiterRunnerCore(
        MoeRunnerConfig(activation="situ", num_local_experts=1)
    )
    hidden_states = torch.zeros((1, 4), dtype=torch.bfloat16)
    runner_input = AiterRunnerInput(
        hidden_states=hidden_states,
        topk_ids=torch.zeros((1, 1), dtype=torch.int32),
        topk_weights=torch.ones((1, 1), dtype=torch.float32),
        quant_type=AiterQuantType.PER_1X32,
    )

    core.run(runner_input, _quant_info(), running_state={})
    assert captured["gate_mode"] == expected_gate_mode
    assert captured["q_dtype_a"] == expected_q_dtype_a


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
