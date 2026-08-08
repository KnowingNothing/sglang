# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License

import mori.ops.dispatch_combine as dispatch_combine


def test_barrier_uses_current_local_device(monkeypatch):
    barrier_calls = []

    monkeypatch.setattr(dispatch_combine.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dispatch_combine.torch.cuda, "current_device", lambda: 5)
    monkeypatch.setattr(
        dispatch_combine.dist,
        "barrier",
        lambda **kwargs: barrier_calls.append(kwargs),
    )

    dispatch_combine._barrier_on_current_device()

    assert barrier_calls == [{"device_ids": [5]}]


def test_barrier_is_skipped_before_distributed_init(monkeypatch):
    monkeypatch.setattr(dispatch_combine.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        dispatch_combine.dist,
        "barrier",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    dispatch_combine._barrier_on_current_device()
