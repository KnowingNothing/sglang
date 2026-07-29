import unittest
from types import SimpleNamespace

import torch

from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.dflash_utils import build_dflash_verify_target_probs
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(is_hip(), "AITER top-k renormalization requires ROCm")
class TestDsparkMixedTopKAMD(CustomTestCase):
    def test_default_and_finite_top_k_share_one_batch(self):
        torch.manual_seed(7)
        batch_size = 2
        draft_token_num = 3
        vocab_size = 512
        logits = torch.randn(
            batch_size * draft_token_num,
            vocab_size,
            device="cuda",
            dtype=torch.float32,
        )
        sampling_info = SimpleNamespace(
            need_top_k_sampling=True,
            need_top_p_sampling=False,
            temperatures=torch.ones(batch_size, device="cuda"),
            top_ks=torch.tensor(
                [TOP_K_ALL, 8], device="cuda", dtype=torch.int32
            ),
        )

        actual = build_dflash_verify_target_probs(
            next_token_logits=logits,
            sampling_info=sampling_info,
            draft_token_num=draft_token_num,
            bs=batch_size,
            max_top_k=TOP_K_ALL,
            uniform_top_k_value=None,
            use_sparse_topk=True,
        )

        expected = torch.empty_like(actual)
        expected[0] = torch.softmax(logits[:draft_token_num], dim=-1)
        finite_logits = logits[draft_token_num:]
        top_values, top_indices = torch.topk(finite_logits, k=8, dim=-1)
        finite_probs = torch.softmax(top_values, dim=-1)
        expected[1].zero_().scatter_(1, top_indices, finite_probs)

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        self.assertTrue(
            torch.equal(
                (actual > 0).sum(dim=-1).cpu(),
                torch.tensor(
                    [[vocab_size] * draft_token_num, [8] * draft_token_num]
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
