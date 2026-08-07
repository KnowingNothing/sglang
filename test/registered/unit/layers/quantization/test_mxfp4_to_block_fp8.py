import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.quantization.fp8 import (
    Fp8MoEMethod,
    _convert_mxfp4_moe_weights_to_block_fp8,
)
from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config
from sglang.srt.models.kimi_k3 import _get_kimi_k3_moe_quant_config


class TestMxfp4ToBlockFp8(unittest.TestCase):
    def test_tp32_fp4_allocation_defers_block_shape_check_until_conversion(self):
        layer = nn.Module()
        quant_config = SimpleNamespace(
            weight_block_size=[128, 128],
            activation_scheme="dynamic",
            is_checkpoint_fp8_serialized=True,
            dequant_fp4_to_fp8=True,
        )

        with patch(
            "sglang.srt.layers.quantization.fp8.get_parallel",
            return_value=SimpleNamespace(tp_size=32),
        ):
            Fp8MoEMethod.create_fp8_moe_weight_(
                layer=layer,
                num_experts=1,
                hidden_size=128,
                intermediate_size_per_partition=96,
                block_quant=True,
                quant_config=quant_config,
                use_mxfp8=False,
                is_checkpoint_fp8_serialized=True,
                is_fp4_expert=True,
                params_dtype=torch.bfloat16,
                fp4_scale_dtype=torch.uint8,
            )

        self.assertEqual(layer.w13_weight.shape, (1, 192, 64))
        self.assertEqual(layer.w2_weight.shape, (1, 128, 48))
        self.assertEqual(layer.w13_weight_scale_inv.shape, (1, 192, 4))
        self.assertEqual(layer.w2_weight_scale_inv.shape, (1, 128, 3))

    def test_kimi_k3_preserves_compressed_config_for_fallback(self):
        class CompressedConfig:
            quant_format = "mxfp4-pack-quantized"

        config = CompressedConfig()
        with patch.dict("os.environ", {"SGLANG_MXFP4_DEQUANT_TO_FP8": "1"}):
            self.assertIs(_get_kimi_k3_moe_quant_config(config), config)

    def test_kimi_k3_uses_native_mxfp4_by_default(self):
        class CompressedConfig:
            quant_format = "mxfp4-pack-quantized"

        with patch.dict("os.environ", {"SGLANG_MXFP4_DEQUANT_TO_FP8": "0"}):
            config = _get_kimi_k3_moe_quant_config(CompressedConfig())

        self.assertIsInstance(config, Mxfp4Config)
        self.assertTrue(config.is_checkpoint_mxfp4_serialized)

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
