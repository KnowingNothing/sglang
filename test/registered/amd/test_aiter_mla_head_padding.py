from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention import aiter_backend


@pytest.mark.parametrize("num_heads", [3, 5, 12, 15])
def test_aiter_mla_pads_sub16_heads_without_changing_real_heads(num_heads):
    backend = object.__new__(aiter_backend.AiterAttnBackend)
    backend.num_head = num_heads
    backend.num_head_padded = 16
    backend.head_repeat_factor = 1
    backend.input_dtype = torch.float32

    q = torch.arange(2 * num_heads * 4, dtype=torch.float32).reshape(
        2, num_heads, 4
    )
    layer = SimpleNamespace(tp_q_head_num=num_heads, v_head_dim=3)
    seen = {}

    def fake_mla_decode_fwd(q_in, _k_buffer_flat, o, **kwargs):
        seen["q"] = q_in.clone()
        seen["kwargs"] = kwargs
        o.copy_(q_in[:, :, : layer.v_head_dim])

    with patch.object(aiter_backend, "mla_decode_fwd", fake_mla_decode_fwd):
        actual = backend._mla_decode_fwd_with_head_pad(
            q,
            torch.empty(0),
            layer,
            marker="forwarded",
        )

    torch.testing.assert_close(actual, q[:, :, : layer.v_head_dim])
    torch.testing.assert_close(seen["q"][:, :num_heads], q)
    num_pad_heads = 16 - num_heads
    repeats = (num_pad_heads + num_heads - 1) // num_heads
    expected_padding = q.repeat(1, repeats, 1)[:, :num_pad_heads, :]
    torch.testing.assert_close(seen["q"][:, num_heads:], expected_padding)
    assert seen["kwargs"]["marker"] == "forwarded"


@pytest.mark.parametrize("num_heads,repeat_factor", [(4, 4), (8, 2)])
def test_aiter_mla_preserves_existing_repeat_padding(num_heads, repeat_factor):
    backend = object.__new__(aiter_backend.AiterAttnBackend)
    backend.num_head = num_heads
    backend.num_head_padded = 16
    backend.head_repeat_factor = repeat_factor
    backend.input_dtype = torch.float32

    q = torch.arange(num_heads * 4, dtype=torch.float32).reshape(1, num_heads, 4)
    layer = SimpleNamespace(tp_q_head_num=num_heads, v_head_dim=3)

    def fake_mla_decode_fwd(q_in, _k_buffer_flat, o, **_kwargs):
        o.copy_(q_in[:, :, : layer.v_head_dim])

    with patch.object(aiter_backend, "mla_decode_fwd", fake_mla_decode_fwd):
        actual = backend._mla_decode_fwd_with_head_pad(
            q,
            torch.empty(0),
            layer,
        )

    torch.testing.assert_close(actual, q[:, :, : layer.v_head_dim])
