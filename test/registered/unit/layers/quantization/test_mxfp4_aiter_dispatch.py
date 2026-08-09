from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
)
from sglang.srt.layers.moe.token_dispatcher.moriep import (
    MoriEPNormalDispatchOutput,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class _CapturingAiterRunner:
    runner_backend = MoeRunnerBackend.AITER

    def __init__(self):
        self.dispatch_output = None
        self.quant_info = None

    def run(self, dispatch_output, quant_info):
        self.dispatch_output = dispatch_output
        self.quant_info = quant_info
        return "aiter-result"


def test_mxfp4_aiter_accepts_mori_dispatch_without_standard_topk_output():
    runner = _CapturingAiterRunner()
    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method.use_deep_gemm = False
    method.runner = runner
    method.hidden_size = 4
    method.hidden_pad = 0
    method.intermediate_pad = 0
    method.with_bias = False
    method.moe_runner_config = SimpleNamespace(
        apply_router_weight_on_input=False,
        activation="situ",
        gemm1_clamp_limit=1.0,
        swiglu_limit=0.0,
    )

    layer = SimpleNamespace(
        w13_weight=torch.zeros((1, 8, 2), dtype=torch.uint8),
        w2_weight=torch.zeros((1, 4, 2), dtype=torch.uint8),
        w13_weight_scale=torch.ones((1, 8, 1), dtype=torch.uint8),
        w2_weight_scale=torch.ones((1, 4, 1), dtype=torch.uint8),
        dispatcher=SimpleNamespace(
            expert_mask_gpu=torch.ones((1,), dtype=torch.bool)
        ),
    )
    topk_ids = torch.zeros((2, 1), dtype=torch.int32)
    topk_weights = torch.ones((2, 1), dtype=torch.float32)
    dispatch_output = MoriEPNormalDispatchOutput(
        hidden_states=torch.zeros((2, 4), dtype=torch.bfloat16),
        hidden_states_scale=None,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        num_recv_tokens_per_expert=torch.tensor([2], dtype=torch.int32),
        origin_topk_ids=topk_ids,
        origin_topk_weights=topk_weights,
        out_dtype=torch.bfloat16,
    )

    result = method.apply(layer, dispatch_output)

    assert result == "aiter-result"
    assert runner.dispatch_output.hidden_states is dispatch_output.hidden_states
    assert runner.dispatch_output.topk_ids is topk_ids
    assert runner.dispatch_output.topk_weights is topk_weights
    assert isinstance(runner.quant_info, AiterMoeQuantInfo)
    assert runner.quant_info.quant_type is AiterQuantType.PER_1X32
    assert runner.quant_info.swiglu_limit == 0.0
