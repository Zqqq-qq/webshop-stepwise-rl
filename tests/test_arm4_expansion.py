from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class Arm4ExpansionTests(unittest.TestCase):
    def test_update_weights_each_trajectory_equally(self):
        try:
            import torch
            import train_arm4_expansion as expansion
        except ImportError:
            self.skipTest("Requires torch")

        class SmallScorer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.2))

            def forward(self):
                return {"step_scores": self.value.reshape(1, 1)}

        model = SmallScorer()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
        with patch.object(expansion, "tensor_batch", return_value={}):
            result = expansion.train_update(model, tokenizer, "cpu",
                                            [(1, 1.0, [None] * 3), (2, 0.0, [None])], optimizer)
        self.assertAlmostEqual(result["loss"], (0.8 ** 2 + 0.2 ** 2) / 2)
        self.assertAlmostEqual(float(model.value.detach()), 0.26, places=6)
