import unittest

import torch
from torch import nn

from sglang.srt.layers.quantization.fp8 import (
    _convert_mxfp4_moe_weights_to_block_fp8,
)


class TestMxfp4ToBlockFp8(unittest.TestCase):
    def test_tp_shard_padding_and_conversion(self):
        layer = nn.Module()
        num_experts = 1
        hidden_size = 128
        intermediate_size = 96

        layer.w13_weight = nn.Parameter(
            torch.full(
                (num_experts, 2 * intermediate_size, hidden_size // 2),
                0x11,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.w2_weight = nn.Parameter(
            torch.full(
                (num_experts, hidden_size, intermediate_size // 2),
                0x11,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        # compressed-tensors serializes UE8M0 scales as raw uint8 bytes.
        layer.w13_weight_scale_inv = nn.Parameter(
            torch.full(
                (num_experts, 2 * intermediate_size, hidden_size // 32),
                127,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.w2_weight_scale_inv = nn.Parameter(
            torch.full(
                (num_experts, hidden_size, intermediate_size // 32),
                127,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )

        _convert_mxfp4_moe_weights_to_block_fp8(layer)

        self.assertEqual(layer.intermediate_pad, 32)
        self.assertEqual(layer.hidden_pad, 0)
        self.assertEqual(layer.w13_weight.shape, (1, 256, 128))
        self.assertEqual(layer.w2_weight.shape, (1, 128, 128))
        self.assertEqual(layer.w13_weight_scale_inv.shape, (1, 2, 1))
        self.assertEqual(layer.w2_weight_scale_inv.shape, (1, 1, 1))
        self.assertEqual(layer.w13_weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.w2_weight.dtype, torch.float8_e4m3fn)
        self.assertGreater(
            torch.count_nonzero(layer.w13_weight[:, :96].float()).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(layer.w13_weight[:, 96:128].float()).item(), 0
        )
        self.assertGreater(
            torch.count_nonzero(layer.w13_weight[:, 128:224].float()).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(layer.w13_weight[:, 224:].float()).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(layer.w2_weight[:, :, 96:].float()).item(), 0
        )


if __name__ == "__main__":
    unittest.main()
