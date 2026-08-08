from types import SimpleNamespace

from sglang.srt.layers.moe.token_dispatcher.moriep import _get_mori_gpu_per_node
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
