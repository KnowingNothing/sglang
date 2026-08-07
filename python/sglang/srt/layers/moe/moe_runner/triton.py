from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda, is_gfx95_supported, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class TritonRunnerInput(RunnerInput):

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonRunnerOutput(RunnerOutput):

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    use_mxfp8: bool = False
    use_fp8_w8a8: bool = False
    use_int8_w8a8: bool = False
    use_int8_w8a16: bool = False
    use_int4_w4a16: bool = False
    per_channel_quant: bool = False
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    w13_zp: Optional[torch.Tensor] = None
    w2_zp: Optional[torch.Tensor] = None
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None


class TritonRunnerCore(MoeRunnerCore):

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

    def run(
        self,
        runner_input: TritonRunnerInput,
        quant_info: TritonMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> TritonRunnerOutput:
        if runner_input.hidden_states.shape[0] == 0:
            if self.config.no_combine:
                topk = runner_input.topk_ids.shape[-1]
                hidden_size = runner_input.hidden_states.shape[-1]
                return TritonRunnerOutput(
                    hidden_states=runner_input.hidden_states.new_empty(
                        (0, topk, hidden_size)
                    )
                )
            return TritonRunnerOutput(hidden_states=runner_input.hidden_states)

        if quant_info.use_mxfp8 and is_hip() and is_gfx95_supported():
            from sglang.kernels.ops.moe.mxfp8_moe_amd_gfx95 import (
                fused_experts_mxfp8,
            )

            out = fused_experts_mxfp8(
                runner_input.hidden_states,
                quant_info.w13_weight,
                quant_info.w2_weight,
                runner_input.topk_weights,
                runner_input.topk_ids,
                quant_info.w13_scale,
                quant_info.w2_scale,
                b1=quant_info.b13,
                b2=quant_info.b2,
                activation=self.config.activation,
                is_gated=self.config.is_gated,
                no_combine=self.config.no_combine,
                inplace=self.config.inplace,
                apply_router_weight_on_input=self.config.apply_router_weight_on_input,
                routed_scaling_factor=self.config.routed_scaling_factor,
                gemm1_alpha=self.config.gemm1_alpha,
                gemm1_limit=self.config.gemm1_clamp_limit,
                swiglu_limit=self.config.swiglu_limit,
                gate_up_interleaved=self.config.gate_up_interleaved,
            )
            return TritonRunnerOutput(hidden_states=out)

        if quant_info.use_mxfp8 and is_cuda():
            raise NotImplementedError(
                "Triton MoE runner does not support NVIDIA MXFP8; use "
                "--moe-runner-backend deep_gemm (or flashinfer_trtllm/cutlass)."
            )

        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            _fused_moe_kernel_sequence,
        )

        filter_expert = (
            self.config.num_experts is None
            or self.config.num_experts != self.config.num_local_experts
        )

        out = _fused_moe_kernel_sequence(
            runner_input.hidden_states,
            quant_info.w13_weight,
            quant_info.w2_weight,
            runner_input.topk_weights,
            runner_input.topk_ids,
            runner_input.sorted_token_ids,
            runner_input.expert_ids,
            runner_input.num_tokens_post_padded,
            running_state["config"],
            running_state.get("down_config"),
            running_state.get("down_moe_use_tma", False),
            b1=quant_info.b13,
            b2=quant_info.b2,
            use_fp8_w8a8=quant_info.use_fp8_w8a8,
            use_int8_w8a8=quant_info.use_int8_w8a8,
            use_int8_w8a16=quant_info.use_int8_w8a16,
            use_int4_w4a16=quant_info.use_int4_w4a16,
            per_channel_quant=quant_info.per_channel_quant,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            w1_zp=quant_info.w13_zp,
            w2_zp=quant_info.w2_zp,
            a1_scale=quant_info.a13_scale,
            a2_scale=quant_info.a2_scale,
            block_shape=quant_info.block_shape,
            activation=self.config.activation,
            is_gated=self.config.is_gated,
            no_combine=self.config.no_combine,
            inplace=self.config.inplace,
            apply_router_weight_on_input=self.config.apply_router_weight_on_input,
            routed_scaling_factor=self.config.routed_scaling_factor,
            gemm1_alpha=self.config.gemm1_alpha,
            gemm1_limit=self.config.gemm1_clamp_limit,
            filter_expert=filter_expert,
            hooks=hooks,
            swiglu_limit=self.config.swiglu_limit,
        )

        return TritonRunnerOutput(hidden_states=out)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@register_fused_func("none", "triton")
def fused_experts_none_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    if quant_info.use_mxfp8 and is_hip() and is_gfx95_supported():
        from sglang.kernels.ops.moe.mxfp8_moe_amd_gfx95 import (
            fused_experts_mxfp8,
        )

        topk_weights, topk_ids, _ = dispatch_output.topk_output
        output = fused_experts_mxfp8(
            hidden_states=dispatch_output.hidden_states,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            b1=quant_info.b13,
            b2=quant_info.b2,
            activation=runner_config.activation,
            is_gated=runner_config.is_gated,
            no_combine=runner_config.no_combine,
            inplace=runner_config.inplace,
            apply_router_weight_on_input=runner_config.apply_router_weight_on_input,
            routed_scaling_factor=runner_config.routed_scaling_factor,
            gemm1_alpha=runner_config.gemm1_alpha,
            gemm1_limit=runner_config.gemm1_clamp_limit,
            swiglu_limit=runner_config.swiglu_limit,
            gate_up_interleaved=runner_config.gate_up_interleaved,
        )
    else:
        if quant_info.use_mxfp8 and is_cuda():
            raise NotImplementedError(
                "Triton MoE runner does not support NVIDIA MXFP8; use "
                "--moe-runner-backend deep_gemm (or flashinfer_trtllm/cutlass)."
            )
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )

        output = fused_experts(
            hidden_states=dispatch_output.hidden_states,
            w1=quant_info.w13_weight,
            w2=quant_info.w2_weight,
            topk_output=dispatch_output.topk_output,
            moe_runner_config=runner_config,
            b1=quant_info.b13,
            b2=quant_info.b2,
            use_fp8_w8a8=quant_info.use_fp8_w8a8,
            use_int8_w8a8=quant_info.use_int8_w8a8,
            use_int8_w8a16=quant_info.use_int8_w8a16,
            use_int4_w4a16=quant_info.use_int4_w4a16,
            per_channel_quant=quant_info.per_channel_quant,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            w1_zp=quant_info.w13_zp,
            w2_zp=quant_info.w2_zp,
            a1_scale=quant_info.a13_scale,
            a2_scale=quant_info.a2_scale,
            block_shape=quant_info.block_shape,
        )

    return StandardCombineInput(
        hidden_states=output,
    )


@register_pre_permute("standard", "triton")
def pre_permute_standard_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:

    # Registered fallback for format-conversion tests and examples.

    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _prepare_fused_moe_run,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )

    assert TopKOutputChecker.format_is_standard(topk_output)

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        hidden_states,
        quant_info.w13_weight,
        quant_info.w2_weight,
        topk_output.topk_ids,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        block_shape=quant_info.block_shape,
    )

    running_state["config"] = config
    running_state["down_config"] = down_config
    running_state["down_moe_use_tma"] = down_moe_use_tma

    return TritonRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_post_permute("triton", "standard")
def post_permute_triton_to_standard(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:

    # Registered fallback for format-conversion tests and examples.

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(
        hidden_states=runner_output.hidden_states,
    )


def _compact_mori_routes(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    global_topk_ids: torch.Tensor,
    *,
    expert_offset: int,
    num_local_experts: int,
):
    """Turn MORI's received global routes into compact local top-k=1 rows."""

    local_topk_ids = global_topk_ids.to(torch.int64) - expert_offset
    valid = (local_topk_ids >= 0) & (local_topk_ids < num_local_experts)
    local_topk_ids = torch.where(
        valid, local_topk_ids, torch.full_like(local_topk_ids, -1)
    ).to(torch.int32)

    valid_positions = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
    token_indices = torch.div(
        valid_positions, global_topk_ids.shape[1], rounding_mode="floor"
    )

    compact_hidden_states = hidden_states.index_select(0, token_indices)
    compact_topk_ids = (
        local_topk_ids.reshape(-1).index_select(0, valid_positions).reshape(-1, 1)
    )
    compact_topk_weights = torch.ones(
        (valid_positions.numel(), 1),
        dtype=topk_weights.dtype,
        device=topk_weights.device,
    )

    output_index = torch.full_like(local_topk_ids, -1, dtype=torch.int32)
    if valid_positions.numel() > 0:
        output_index.reshape(-1).index_copy_(
            0,
            valid_positions,
            torch.arange(
                valid_positions.numel(),
                dtype=torch.int32,
                device=valid_positions.device,
            ),
        )

    return (
        compact_hidden_states,
        compact_topk_weights,
        compact_topk_ids,
        local_topk_ids,
        output_index,
    )


@register_pre_permute("deepep_ll", "triton")
@register_pre_permute("deepep_normal", "triton")
def pre_permute_mori_to_triton(
    dispatch_output,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:
    """Adapt MORI normal or low-latency dispatch to the Triton MoE runner."""

    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _prepare_fused_moe_run,
    )

    if not hasattr(dispatch_output, "origin_topk_ids"):
        raise NotImplementedError(
            "Triton DeepEP-format pre-permute currently supports MORI only"
        )
    if runner_config.no_combine:
        raise NotImplementedError("MORI + Triton does not support no_combine=True")
    if runner_config.apply_router_weight_on_input:
        raise NotImplementedError(
            "MORI + Triton does not support apply_router_weight_on_input=True"
        )

    total_recv_tensor = dispatch_output.num_recv_tokens_per_expert
    if total_recv_tensor.numel() != 1:
        raise ValueError(
            "MORI dispatch must provide a one-element total receive count, got "
            f"shape={tuple(total_recv_tensor.shape)}"
        )
    # MORI allocates a fixed-capacity receive arena. Triton must only process
    # the live prefix. The scalar sync is correctness-first and can later be
    # replaced by a device-count-aware compaction kernel.
    total_recv = int(total_recv_tensor.item())

    hidden_states = dispatch_output.hidden_states[:total_recv]
    hidden_states_scale = dispatch_output.hidden_states_scale
    if hidden_states_scale is not None:
        from sglang.kernels.ops.moe.rocm_moe_utils import upscale, upscale_mxfp4

        hidden_states_scale = hidden_states_scale[:total_recv]
        if hidden_states.dtype == torch.float4_e2m1fn_x2:
            hidden_states = upscale_mxfp4(
                hidden_states,
                hidden_states_scale,
                total_recv_tensor,
                dispatch_output.out_dtype,
            )
        else:
            hidden_states = upscale(
                hidden_states,
                hidden_states_scale,
                total_recv_tensor,
                dispatch_output.out_dtype,
            )

    global_topk_ids = dispatch_output.topk_ids[:total_recv]
    topk_weights = dispatch_output.topk_weights[:total_recv]
    expert_offset = get_parallel().moe_ep_rank * runner_config.num_local_experts

    (
        compact_hidden_states,
        compact_topk_weights,
        compact_topk_ids,
        local_topk_ids,
        output_index,
    ) = _compact_mori_routes(
        hidden_states,
        topk_weights,
        global_topk_ids,
        expert_offset=expert_offset,
        num_local_experts=runner_config.num_local_experts,
    )

    running_state["triton_mori_local_topk_ids"] = local_topk_ids
    running_state["triton_mori_topk_weights"] = topk_weights
    running_state["triton_mori_output_index"] = output_index
    running_state["triton_mori_total_recv"] = total_recv
    running_state["triton_mori_hidden_size"] = hidden_states.shape[1]
    running_state["triton_mori_output_dtype"] = dispatch_output.out_dtype
    running_state["triton_mori_origin_topk_ids"] = dispatch_output.origin_topk_ids
    running_state["triton_mori_origin_topk_weights"] = (
        dispatch_output.origin_topk_weights
    )

    if compact_hidden_states.shape[0] == 0:
        empty_i32 = torch.empty(
            (0,), dtype=torch.int32, device=compact_hidden_states.device
        )
        return TritonRunnerInput(
            hidden_states=compact_hidden_states,
            topk_weights=compact_topk_weights,
            topk_ids=compact_topk_ids,
            sorted_token_ids=empty_i32,
            expert_ids=empty_i32,
            num_tokens_post_padded=torch.zeros(
                (1,), dtype=torch.int32, device=compact_hidden_states.device
            ),
        )

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        compact_hidden_states,
        quant_info.w13_weight,
        quant_info.w2_weight,
        compact_topk_ids,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        block_shape=quant_info.block_shape,
    )
    running_state["config"] = config
    running_state["down_config"] = down_config
    running_state["down_moe_use_tma"] = down_moe_use_tma

    return TritonRunnerInput(
        hidden_states=compact_hidden_states,
        topk_weights=compact_topk_weights,
        topk_ids=compact_topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


def _post_permute_triton_to_mori(
    runner_output: TritonRunnerOutput,
    running_state: dict,
    *,
    is_normal: bool,
):
    from sglang.kernels.ops.moe.ep_moe_kernels import ep_gather
    from sglang.srt.layers.moe.token_dispatcher.moriep import (
        MoriEPLLCombineInput,
        MoriEPNormalCombineInput,
    )

    total_recv = running_state["triton_mori_total_recv"]
    hidden_size = running_state["triton_mori_hidden_size"]
    recv_output = torch.empty(
        (total_recv, hidden_size),
        dtype=running_state["triton_mori_output_dtype"],
        device=runner_output.hidden_states.device,
    )
    if total_recv > 0:
        ep_gather(
            runner_output.hidden_states,
            running_state["triton_mori_local_topk_ids"],
            running_state["triton_mori_topk_weights"],
            running_state["triton_mori_output_index"],
            recv_output,
        )

    combine_cls = MoriEPNormalCombineInput if is_normal else MoriEPLLCombineInput
    return combine_cls(
        hidden_states=recv_output,
        topk_ids=running_state["triton_mori_origin_topk_ids"],
        topk_weights=running_state["triton_mori_origin_topk_weights"],
    )


@register_post_permute("triton", "deepep_normal")
def post_permute_triton_to_mori_normal(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    return _post_permute_triton_to_mori(runner_output, running_state, is_normal=True)


@register_post_permute("triton", "deepep_ll")
def post_permute_triton_to_mori_low_latency(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
):
    return _post_permute_triton_to_mori(runner_output, running_state, is_normal=False)
