import sys
from types import ModuleType, SimpleNamespace

from sglang.srt.layers.moe.token_dispatcher.moriep import (
    _MoriEPDispatcherImplLowLatency,
    _MoriEPDispatcherImplNormal,
    _get_mori_gpu_per_node,
)
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-c-test-cpu")


def _group(rank, ranks, *, local_size=8):
    return SimpleNamespace(
        rank=rank,
        ranks=ranks,
        local_size=local_size,
        world_size=len(ranks),
    )


def test_ep16_moe_dp2_uses_all_eight_local_gpus():
    assert _get_mori_gpu_per_node(_group(3, list(range(16)))) == 8
    assert _get_mori_gpu_per_node(_group(20, list(range(16, 32)))) == 8


def test_interleaved_ep_subgroup_counts_only_local_members():
    ranks = list(range(0, 8, 2)) + list(range(8, 16, 2))
    assert _get_mori_gpu_per_node(_group(2, ranks)) == 4
    assert _get_mori_gpu_per_node(_group(10, ranks)) == 4


def test_auto_mode_builds_distinct_normal_and_low_latency_contracts(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "mori", ModuleType("mori"))
    monkeypatch.setenv("SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "16384")
    monkeypatch.setenv("SGLANG_MORI_PREALLOC_MAX_RECV_TOKENS", "524288")
    monkeypatch.setenv(
        "SGLANG_MORI_LOW_LATENCY_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "1"
    )
    monkeypatch.setenv(
        "SGLANG_MORI_LOW_LATENCY_PREALLOC_MAX_RECV_TOKENS", "32"
    )
    common = dict(
        group=_group(0, list(range(32))),
        router_topk=16,
        permute_fusion=False,
        num_experts=896,
        num_local_experts=28,
        hidden_size=3584,
        params_dtype=None,
        deepep_mode=DeepEPMode.AUTO,
    )

    normal = _MoriEPDispatcherImplNormal(async_finish=False, **common)
    low_latency = _MoriEPDispatcherImplLowLatency(**common)

    assert normal.mori_op_mode is DeepEPMode.NORMAL
    assert normal.num_max_dispatch_tokens_per_rank == 16384
    assert normal.max_total_recv_tokens == 524288
    assert low_latency.mori_op_mode is DeepEPMode.LOW_LATENCY
    assert low_latency.num_max_dispatch_tokens_per_rank == 1
    assert low_latency.max_total_recv_tokens == 32
