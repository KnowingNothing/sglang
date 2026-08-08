import os

import pytest

from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-c-test-cpu")


def test_explicit_dp4_tp8_moe_dp2_ep16_topology_parses_and_validates():
    args = prepare_server_args(
        [
            "--model-path",
            "dummy",
            "--model-parallel-size",
            "32",
            "--attention-data-parallel-size",
            "4",
            "--attention-tensor-parallel-size",
            "8",
            "--attention-context-parallel-size",
            "1",
            "--expert-parallel-size",
            "16",
            "--moe-data-parallel-size",
            "2",
            "--moe-tensor-parallel-size",
            "1",
            "--per-dp-chunked-prefill-size",
            "16384",
        ]
    )

    assert args.tp_size == 32
    assert args.dp_size == 4
    assert args.attention_tp_size == 8
    assert args.attn_cp_size == 1
    assert args.ep_size == 16
    assert args.moe_dp_size == 2
    assert args.moe_tp_size == 1
    assert args.dp_local_chunked_prefill_size == 16384
    assert args.chunked_prefill_size is None
    args._handle_dp_local_chunked_prefill_size()
    assert args.chunked_prefill_size == 65536
    args._handle_explicit_parallel_topology()


def test_explicit_parallel_topology_rejects_a_misleading_attention_tp():
    args = ServerArgs(
        model_path="dummy",
        tp_size=32,
        dp_size=4,
        attention_tp_size=32,
        ep_size=16,
        moe_dp_size=2,
        moe_tp_size=1,
    )

    with pytest.raises(ValueError, match="resolved 8"):
        args._handle_explicit_parallel_topology()


def test_mxfp4_fp8_compute_is_command_line_owned(monkeypatch):
    monkeypatch.delenv("AITER_SITUV2_A8W4", raising=False)
    monkeypatch.delenv("SGLANG_MXFP4_DEQUANT_TO_FP8", raising=False)
    args = ServerArgs(
        model_path="dummy",
        moe_runner_backend="aiter",
        mxfp4_moe_compute_dtype="fp8",
    )

    args._handle_mxfp4_moe_compute_dtype()
    assert "AITER_SITUV2_A8W4" not in os.environ


def test_mxfp4_compute_rejects_conflicting_legacy_environment(monkeypatch):
    monkeypatch.setenv("AITER_SITUV2_A8W4", "0")
    monkeypatch.delenv("SGLANG_MXFP4_DEQUANT_TO_FP8", raising=False)
    args = ServerArgs(
        model_path="dummy",
        moe_runner_backend="aiter",
        mxfp4_moe_compute_dtype="fp8",
    )

    with pytest.raises(ValueError, match="must be unset"):
        args._handle_mxfp4_moe_compute_dtype()


def test_mxfp4_fp8_compute_rejects_unvalidated_triton_backend(monkeypatch):
    monkeypatch.delenv("AITER_SITUV2_A8W4", raising=False)
    monkeypatch.delenv("SGLANG_USE_AITER", raising=False)
    monkeypatch.delenv("SGLANG_MXFP4_DEQUANT_TO_FP8", raising=False)
    args = ServerArgs(
        model_path="dummy",
        moe_runner_backend="triton",
        mxfp4_moe_compute_dtype="fp8",
    )

    with pytest.raises(ValueError, match="requires --moe-runner-backend aiter"):
        args._handle_mxfp4_moe_compute_dtype()
