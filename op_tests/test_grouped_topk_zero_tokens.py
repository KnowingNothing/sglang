# SPDX-License-Identifier: MIT

import pytest
import torch

from aiter.ops.topk import (
    biased_grouped_topk,
    biased_grouped_topk_hip,
    grouped_topk,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grouped top-k requires a GPU"
)


def _empty_inputs():
    logits = torch.empty((0, 896), dtype=torch.bfloat16, device="cuda")
    bias = torch.empty((896,), dtype=torch.bfloat16, device="cuda")
    weights = torch.empty((0, 16), dtype=torch.float32, device="cuda")
    ids = torch.empty((0, 16), dtype=torch.int32, device="cuda")
    return logits, bias, weights, ids


def _assert_no_sticky_hip_error():
    # A zero-grid launch is asynchronous on some ROCm stacks.  Exercise a later
    # allocation as well as synchronize so the regression cannot hide at the
    # next unrelated operator.
    torch.zeros((1,), dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()


def test_biased_grouped_topk_high_level_skips_empty_batch():
    logits, bias, weights, ids = _empty_inputs()
    assert (
        biased_grouped_topk(
            logits,
            bias,
            weights,
            ids,
            num_expert_group=1,
            topk_group=1,
            need_renorm=True,
        )
        is None
    )
    _assert_no_sticky_hip_error()


def test_biased_grouped_topk_kernel_accepts_empty_batch():
    logits, bias, weights, ids = _empty_inputs()
    biased_grouped_topk_hip(
        logits,
        bias,
        weights,
        ids,
        num_expert_group=1,
        topk_grp=1,
        need_renorm=True,
    )
    _assert_no_sticky_hip_error()


def test_grouped_topk_kernel_accepts_empty_batch():
    logits, _bias, weights, ids = _empty_inputs()
    grouped_topk(
        logits,
        weights,
        ids,
        num_expert_group=1,
        topk_group=1,
        need_renorm=True,
        is_softmax=False,
    )
    _assert_no_sticky_hip_error()
