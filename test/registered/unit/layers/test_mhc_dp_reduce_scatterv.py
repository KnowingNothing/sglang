from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.communicator_mhc import (
    MHCCommunicateSummableTensorPairFn,
)


def test_mhc_scatter_hidden_states_uses_variable_length_reduce_scatter():
    global_hidden = torch.empty(6, 4)
    local_hidden = torch.empty(2, 4)
    residual = torch.empty(2, 4)
    sizes = [2, 1, 3]

    group = MagicMock()
    mhc = SimpleNamespace(mlp_combine=lambda hidden, _: hidden)
    forward_batch = SimpleNamespace(
        dp_padding_mode=SimpleNamespace(is_max_len=lambda: False)
    )

    with (
        patch(
            "sglang.srt.layers.communicator_mhc.should_use_dp_reduce_scatterv",
            return_value=True,
        ),
        patch(
            "sglang.srt.layers.communicator_mhc.get_tp_group", return_value=group
        ),
        patch(
            "sglang.srt.layers.communicator_mhc.get_local_dp_buffer_mhc",
            return_value=local_hidden,
        ),
        patch(
            "sglang.srt.layers.communicator_mhc.get_dp_global_num_tokens",
            return_value=sizes,
        ),
        patch("sglang.srt.layers.communicator_mhc.dp_scatter") as dp_scatter,
        patch(
            "sglang.srt.layers.communicator_mhc.dp_reduce_scatter_tensor"
        ) as fixed_reduce_scatter,
    ):
        output, output_residual = (
            MHCCommunicateSummableTensorPairFn._scatter_hidden_states(
                hidden_states=global_hidden,
                residual=residual,
                forward_batch=forward_batch,
                context=SimpleNamespace(),
                allow_reduce_scatter=True,
                mhc=mhc,
                is_last_layer=False,
            )
        )

    group.reduce_scatterv.assert_called_once_with(
        global_hidden,
        output=local_hidden,
        sizes=sizes,
    )
    dp_scatter.assert_not_called()
    fixed_reduce_scatter.assert_not_called()
    assert output is local_hidden
    assert output_residual is None
