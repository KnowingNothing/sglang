import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    _dsa_sparse_topk_capacity,
)


class TestDSAAiterKPoolTail(unittest.TestCase):
    def test_sparse_topk_capacity_includes_live_tail(self):
        self.assertEqual(_dsa_sparse_topk_capacity(2048, 1), 2048)
        self.assertEqual(_dsa_sparse_topk_capacity(2048, 16), 2063)

    def test_aiter_is_allowed_for_kpool_tail(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.dsa_index_kpool = 16

        backend._check_kpool_tail_backend(object(), "aiter", "decode")
        with self.assertRaises(NotImplementedError):
            backend._check_kpool_tail_backend(
                object(), "flashmla_sparse", "decode"
            )

    def test_aiter_metadata_uses_tail_extended_width(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.num_head_padded = 256
        backend.dsa_index_topk = 2048
        backend.aiter_dsa_max_split_per_batch = 64
        backend.aiter_dsa_kv_last_page_lens = torch.ones(4, dtype=torch.int32)
        backend.aiter_dsa_work_metadata = torch.empty(1)
        backend.aiter_dsa_work_info_set = torch.empty(1)
        backend.aiter_dsa_work_indptr = torch.empty(1)
        backend.aiter_dsa_reduce_indptr = torch.empty(1)
        backend.aiter_dsa_reduce_final_map = torch.empty(1)
        backend.aiter_dsa_reduce_partial_map = torch.empty(1)
        backend._ensure_aiter_dsa_decode_metadata_buffer = MagicMock()

        qo_indptr = torch.tensor([0, 1, 2], dtype=torch.int32)
        kv_indptr = torch.tensor([0, 2049, 4112], dtype=torch.int32)
        with patch(
            "sglang.srt.layers.attention.dsa_backend.get_mla_metadata_v1"
        ) as get_metadata:
            backend._prepare_aiter_dsa_decode_metadata(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                bs=2,
                max_seqlen_q=1,
                q_dtype=torch.bfloat16,
                kv_dtype=torch.float8_e4m3fn,
                sparse_topk=2063,
            )

        self.assertEqual(get_metadata.call_args.kwargs["topk"], 2063)


if __name__ == "__main__":
    unittest.main()
