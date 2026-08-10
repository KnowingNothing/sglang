from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)


class _DecodeMode:
    @staticmethod
    def is_decode_or_idle():
        return True

    @staticmethod
    def is_extend(include_draft_extend_v2=False):
        return False


class _MaskThatRejectsHostReads:
    def any(self):
        raise AssertionError("decode metadata must not read a GPU mask scalar")


class _ReqToTokenPool:
    mamba_pool = None

    @staticmethod
    def get_mamba_indices(req_pool_indices):
        return req_pool_indices.to(torch.int64)

    @staticmethod
    def translate_mamba_indices(indices):
        return indices


def test_decode_metadata_does_not_read_mamba_track_mask_scalar():
    backend = object.__new__(MambaAttnBackendBase)
    backend.device = torch.device("cpu")
    backend.req_to_token_pool = _ReqToTokenPool()

    forward_batch = SimpleNamespace(
        batch_size=1,
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        _original_batch_size=None,
        forward_mode=_DecodeMode(),
        mamba_track_indices=torch.tensor([0], dtype=torch.int64),
        mamba_track_mask=_MaskThatRejectsHostReads(),
    )

    metadata = backend._forward_metadata(forward_batch)

    assert metadata.has_mamba_track_mask is False
    assert torch.equal(
        metadata.query_start_loc, torch.tensor([0, 1], dtype=torch.int32)
    )
