# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License

import mori.ops.dispatch_combine as dispatch_combine


def test_jit_prepare_is_node_local_and_has_no_world_barrier(monkeypatch):
    ensure_calls = []
    kernel_type = object()

    monkeypatch.setattr(
        dispatch_combine,
        "_ensure_jit_kernels",
        lambda value: ensure_calls.append(value),
    )
    monkeypatch.setattr(
        dispatch_combine.dist,
        "barrier",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    dispatch_combine._prepare_jit_kernels(kernel_type)

    assert ensure_calls == [kernel_type]


def test_jit_prepare_does_not_query_distributed_state(monkeypatch):
    monkeypatch.setattr(dispatch_combine, "_ensure_jit_kernels", lambda _: None)
    monkeypatch.setattr(
        dispatch_combine.dist,
        "is_initialized",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected dist query")),
    )

    dispatch_combine._prepare_jit_kernels(object())
