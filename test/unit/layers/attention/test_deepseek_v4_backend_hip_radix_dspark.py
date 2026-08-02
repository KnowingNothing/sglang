from unittest.mock import Mock

import torch

from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
)


def test_dspark_draft_block_uses_gamma_not_verify_width():
    backend = object.__new__(DeepseekV4HipRadixBackend)
    backend._move_to_device = lambda values: torch.tensor(values, dtype=torch.int32)
    backend.init_forward_metadata_prefill = Mock(return_value="metadata")

    seq_lens = torch.tensor([100, 200], dtype=torch.int32)
    out_cache_loc = torch.arange(14, dtype=torch.int64)
    result = backend.init_forward_metadata_dspark_draft_block(
        max_seq_len=4096,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        seq_lens=seq_lens,
        out_cache_loc=out_cache_loc,
        block_size=7,
    )

    assert result == "metadata"
    kwargs = backend.init_forward_metadata_prefill.call_args.kwargs
    torch.testing.assert_close(
        kwargs["seq_lens"], torch.tensor([107, 207], dtype=torch.int32)
    )
    assert kwargs["seq_lens_cpu"] == [107, 207]
    assert kwargs["extend_seq_lens_cpu"] == [7, 7]
    assert kwargs["num_tokens"] == 14
    assert kwargs["out_cache_loc"].numel() == 14
    assert kwargs["need_compress"] is False
    assert kwargs["dspark_block_size"] == 7
