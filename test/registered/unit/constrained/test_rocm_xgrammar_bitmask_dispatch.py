import pytest
import torch

from sglang.srt.constrained import xgrammar_backend


@pytest.mark.skipif(not torch.version.hip, reason="ROCm-only dispatch contract")
def test_xgrammar_uses_triton_bitmask_on_rocm():
    logits = torch.zeros((4, 64), device="cuda")
    bitmask = torch.full((4, 2), -1, dtype=torch.int32, device="cuda")
    bitmask[:, 0] = 0

    grammar = object.__new__(xgrammar_backend.XGrammarGrammar)
    grammar.apply_vocab_mask(logits, bitmask)
    torch.cuda.synchronize()
    assert torch.isneginf(logits[:, :32]).all()
    assert torch.isfinite(logits[:, 32:]).all()

    logits.zero_()
    xgrammar_backend.XGrammarGrammarBackend.apply_vocab_mask(logits, bitmask)
    torch.cuda.synchronize()
    assert torch.isneginf(logits[:, :32]).all()
    assert torch.isfinite(logits[:, 32:]).all()
