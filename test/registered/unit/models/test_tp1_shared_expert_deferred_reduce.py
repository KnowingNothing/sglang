from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_v2 import DeepseekV2MoE


def _moe(tp_size: int = 8) -> DeepseekV2MoE:
    moe = DeepseekV2MoE.__new__(DeepseekV2MoE)
    moe.tp_size = tp_size
    return moe


def _forward_flags(*, reduce_scatter: bool = False, fused: bool = False):
    return SimpleNamespace(
        mlp_reduce_scatter=reduce_scatter,
        fuse_mlp_allreduce=fused,
    )


def test_tp1_shared_expert_is_prescaled_before_dp_reduce_scatterv():
    final_hidden = torch.full((2, 4), 3.0)
    shared = torch.full((2, 4), 8.0)

    with (
        patch(
            "sglang.srt.models.deepseek_v2.should_use_dp_reduce_scatterv",
            return_value=True,
        ),
        patch(
            "sglang.srt.models.deepseek_v2.get_forward",
            return_value=_forward_flags(),
        ),
    ):
        output = _moe()._add_tp1_shared_expert_output(final_hidden, shared)

    assert torch.equal(output, torch.full((2, 4), 4.0))


def test_tp1_shared_expert_is_prescaled_for_other_deferred_tp_reductions():
    final_hidden = torch.zeros(1, 2)
    shared = torch.full((1, 2), 8.0)

    for flags in (
        _forward_flags(reduce_scatter=True),
        _forward_flags(fused=True),
    ):
        with (
            patch(
                "sglang.srt.models.deepseek_v2.should_use_dp_reduce_scatterv",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.deepseek_v2.get_forward",
                return_value=flags,
            ),
        ):
            output = _moe()._add_tp1_shared_expert_output(final_hidden, shared)

        assert torch.equal(output, torch.ones(1, 2))


def test_tp1_shared_expert_keeps_full_value_without_deferred_reduction():
    final_hidden = torch.full((2, 3), 3.0)
    shared = torch.full((2, 3), 8.0)

    with (
        patch(
            "sglang.srt.models.deepseek_v2.should_use_dp_reduce_scatterv",
            return_value=False,
        ),
        patch(
            "sglang.srt.models.deepseek_v2.get_forward",
            return_value=_forward_flags(),
        ),
    ):
        output = _moe()._add_tp1_shared_expert_output(final_hidden, shared)

    assert torch.equal(output, torch.full((2, 3), 11.0))
