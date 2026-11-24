"""Small CPU contracts for the Issue #48 cloud-only fixed-feature diagnosis."""

import hashlib
import unittest

from scripts.prm_cpu_head_diagnostic import score_predictions, train_head

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None


class MetricsContract(unittest.TestCase):
    def test_all_eight_and_exact_metrics(self):
        self.assertEqual(score_predictions([0.0] * 8, [0, 0, 0, 0, 0, 0, 1, 1]),
                         (0.25, 1.0))
        with self.assertRaises(ValueError):
            score_predictions([0.0] * 7, [0.0] * 8)
        with self.assertRaises(ValueError):
            score_predictions([float("nan")] * 8, [0.0] * 8)

    @unittest.skipIf(torch is None, "NumPy and PyTorch required for CPU numerical suite")
    def test_matched_exposures_distinct_step_counts(self):
        x = np.zeros((8, 6), dtype=np.float32)
        y = np.zeros(8, dtype=np.float32)
        single = train_head(x, y, 1e-7, 1, max_exposures=64)
        batch = train_head(x, y, 1e-7, 8, max_exposures=64)
        for run, steps in ((single, 64), (batch, 8)):
            self.assertEqual(run["optimizer_steps"], steps)
            self.assertEqual(run["sample_exposures"], 64)
            self.assertEqual(run["clip_events"], 0)
            self.assertEqual(run["stop_reason"], "memory_gate_passed")
            self.assertEqual(run["history"][0]["mse"], 0.0)

    @unittest.skipIf(torch is None, "NumPy and PyTorch required for CPU numerical suite")
    def test_clip_counts_and_finite_updates(self):
        x = np.ones((8, 6), dtype=np.float32) * 100
        y = np.ones(8, dtype=np.float32)
        run = train_head(x, y, 1e-7, 1, max_exposures=64)
        self.assertEqual(run["clip_events"], 64)
        self.assertEqual(len(run["history"]), 1)
        self.assertEqual(run["history"][0]["sample_exposures"], 64)


if __name__ == "__main__":
    unittest.main()
