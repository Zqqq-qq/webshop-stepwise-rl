"""Regression tests for scripts/evaluate_prm57_independent_dev.py."""
import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate_prm57_independent_dev.py'


def _load_module():
    spec = importlib.util.spec_from_file_location('evaluate_prm57_independent_dev', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class IndependentDevEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load_module()

    def test_evaluates_valid_pairs_ties_too_long_and_fit_only_baselines(self):
        frozen = [101, 102, 103, 104]
        records = [
            {'task_id': 101, 'status': 'valid_pair', 'env_high': 1.0, 'env_low': 0.0},
            {'task_id': 102, 'status': 'environment_tie', 'env_high': 1.0, 'env_low': 1.0},
            {'task_id': 103, 'status': 'too_long'},
            {'task_id': 104, 'status': 'valid_pair', 'env_high': 1.0, 'env_low': 0.5},
        ]

        def mk(s0, s1, s2):
            return {'turn_end_indices': [100, 200, 300], 'step_scores': [s0, s1, s2], 'total': s0 + s1 + s2}

        preds = {
            'stage126': {
                (101, 'high'): mk(0.2, 0.25, 0.1), (101, 'low'): mk(0.2, 0.26, 0.05),
                (104, 'high'): mk(0.2, 0.27, 0.1), (104, 'low'): mk(0.2, 0.26, 0.05),
            },
            'terminal': {
                (101, 'high'): mk(0.2, 0.26, 0.1), (101, 'low'): mk(0.2, 0.25, 0.05),
                (104, 'high'): mk(0.2, 0.27, 0.1), (104, 'low'): mk(0.2, 0.26, 0.05),
            },
            'ranking': {
                (101, 'high'): mk(0.2, 0.28, 0.1), (101, 'low'): mk(0.2, 0.24, 0.05),
                (104, 'high'): mk(0.2, 0.29, 0.1), (104, 'low'): mk(0.2, 0.25, 0.05),
            },
        }
        res = self.m.evaluate_fork_pairs(frozen, records, preds)
        self.assertEqual(res['counts']['valid_contrast_pairs'], 2)
        self.assertEqual(res['counts']['environment_ties'], 1)
        self.assertEqual(res['counts']['too_long_rejected'], 1)
        self.assertEqual(res['models']['stage126']['correct_pairs'], 1)
        self.assertEqual(res['models']['terminal']['correct_pairs'], 2)
        self.assertEqual(res['models']['ranking']['correct_pairs'], 2)
        self.assertGreater(res['ranking_minus_terminal_fork_delta']['mean'], 0)

        fit_rows = [{'terminal_reward': 1.0, 'num_turns': 3}, {'terminal_reward': 0.0, 'num_turns': 5}]
        base = self.m.fit_only_baselines(fit_rows)
        dev_trajs = [{'task_id': 101, 'terminal_reward': 1.0, 'num_turns': 3},
                     {'task_id': 104, 'terminal_reward': 0.5, 'num_turns': 4}]
        total_preds = {
            'stage126': {101: 0.7, 104: 0.6},
            'terminal': {101: 0.8, 104: 0.55},
            'ranking': {101: 0.85, 104: 0.52},
        }
        mse_res = self.m.evaluate_total_score_mse(frozen, dev_trajs, total_preds, base)
        self.assertEqual(mse_res['n_trajectories'], 2)
        self.assertLess(mse_res['models']['ranking']['mse'], mse_res['models']['stage126']['mse'])

    def test_rejects_unfrozen_task_id_or_prefix_drift_or_equal_reward_valid_pair(self):
        frozen = [101]
        mk = lambda s0, s1: {'turn_end_indices': [100, 200, 300], 'step_scores': [s0, s1, 0.1], 'total': s0 + s1 + 0.1}
        preds = {m: {(101, 'high'): mk(0.2, 0.3), (101, 'low'): mk(0.2, 0.2)} for m in self.m.MODELS}
        # Reject non-frozen task (e.g. 7544 accidentally pooled into independent dev)
        with self.assertRaisesRegex(ValueError, 'not in frozen independent dev list'):
            self.m.evaluate_fork_pairs(frozen, [{'task_id': 7544, 'status': 'valid_pair', 'env_high': 1.0, 'env_low': 0.0}], preds)
        # Reject environment tie mislabelled as valid_pair
        with self.assertRaisesRegex(ValueError, 'env_high > env_low'):
            self.m.evaluate_fork_pairs(frozen, [{'task_id': 101, 'status': 'valid_pair', 'env_high': 1.0, 'env_low': 1.0}], preds)
        # Reject shared prefix drift > 1e-5
        preds['ranking'][(101, 'low')] = mk(0.201, 0.2)
        with self.assertRaisesRegex(ValueError, 'shared prefix drift'):
            self.m.evaluate_fork_pairs(frozen, [{'task_id': 101, 'status': 'valid_pair', 'env_high': 1.0, 'env_low': 0.0}], preds)


if __name__ == '__main__':
    unittest.main()
