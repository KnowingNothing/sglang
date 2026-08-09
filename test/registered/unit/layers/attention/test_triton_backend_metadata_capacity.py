from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention.triton_backend import _get_metadata_batch_capacity


def test_triton_metadata_buffers_cover_eager_tp_padding():
    runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(size=1),
        server_args=SimpleNamespace(),
    )

    with patch(
        "sglang.srt.utils.common.get_eager_max_batch_size",
        return_value=8,
    ):
        assert _get_metadata_batch_capacity(runner) == 8


def test_triton_metadata_capacity_never_shrinks_the_request_pool():
    runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(size=16),
        server_args=SimpleNamespace(),
    )

    with patch(
        "sglang.srt.utils.common.get_eager_max_batch_size",
        return_value=8,
    ):
        assert _get_metadata_batch_capacity(runner) == 16
