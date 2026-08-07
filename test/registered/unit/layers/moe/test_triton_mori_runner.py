import importlib.util
import unittest
from pathlib import Path

import torch


_UTILS_PATH = (
    Path(__file__).resolve().parents[5]
    / "python/sglang/srt/layers/moe/moe_runner/triton_mori_utils.py"
)
_SPEC = importlib.util.spec_from_file_location("triton_mori_utils", _UTILS_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
compact_mori_routes = _MODULE.compact_mori_routes


class TestCompactMoriRoutes(unittest.TestCase):
    def test_maps_global_experts_and_preserves_route_order(self):
        hidden_states = torch.tensor([[1.0], [2.0], [3.0]])
        topk_weights = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )
        global_topk_ids = torch.tensor(
            [[4, 8, -1], [5, 7, 10], [6, 9, 11]], dtype=torch.int32
        )

        (
            compact_hidden_states,
            compact_topk_weights,
            compact_topk_ids,
            local_topk_ids,
            output_index,
        ) = compact_mori_routes(
            hidden_states,
            topk_weights,
            global_topk_ids,
            expert_offset=4,
            num_local_experts=4,
        )

        torch.testing.assert_close(
            compact_hidden_states, torch.tensor([[1.0], [2.0], [2.0], [3.0]])
        )
        torch.testing.assert_close(compact_topk_weights, torch.ones((4, 1)))
        self.assertEqual(compact_topk_ids.tolist(), [[0], [1], [3], [2]])
        self.assertEqual(
            local_topk_ids.tolist(), [[0, -1, -1], [1, 3, -1], [2, -1, -1]]
        )
        self.assertEqual(
            output_index.tolist(), [[0, -1, -1], [1, 2, -1], [3, -1, -1]]
        )

        route_outputs = compact_hidden_states * 10
        reconstructed = torch.zeros_like(hidden_states)
        for token in range(output_index.shape[0]):
            for slot in range(output_index.shape[1]):
                route = int(output_index[token, slot])
                if route >= 0:
                    reconstructed[token] += (
                        route_outputs[route] * topk_weights[token, slot]
                    )

        torch.testing.assert_close(
            reconstructed, torch.tensor([[1.0], [18.0], [21.0]])
        )

    def test_handles_no_local_routes(self):
        hidden_states = torch.tensor([[1.0], [2.0]])
        topk_weights = torch.ones((2, 2))
        global_topk_ids = torch.tensor([[0, 1], [8, 9]], dtype=torch.int32)

        compact = compact_mori_routes(
            hidden_states,
            topk_weights,
            global_topk_ids,
            expert_offset=4,
            num_local_experts=4,
        )

        self.assertEqual(compact[0].shape, (0, 1))
        self.assertEqual(compact[2].shape, (0, 1))
        self.assertTrue(torch.all(compact[3] == -1))
        self.assertTrue(torch.all(compact[4] == -1))


if __name__ == "__main__":
    unittest.main()
