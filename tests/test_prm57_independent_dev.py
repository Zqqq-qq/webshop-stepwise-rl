"""Regression tests for scripts/freeze_prm57_independent_dev.py."""
import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'freeze_prm57_independent_dev.py'


def _load_module():
    spec = importlib.util.spec_from_file_location('freeze_prm57_independent_dev', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_world(mod):
    # 10 clean tasks: first 4 viewed, next 4 selected (2 primary + 2 reserve), 2 leftover
    clean = list(range(100, 110))
    ranked = mod.rank_clean_pool(clean)
    viewed = ranked[:4]
    train32_ids = [1, 2]
    fit501_ids = {1, 2, 3}
    val_ids = {200, 201}
    test_ids = {300, 301}
    splits = {
        'train': set(clean) | fit501_ids,
        'val': val_ids,
        'test': test_ids,
    }
    goals = {}
    for tid in splits['train'] | val_ids | test_ids:
        goals[tid] = f'Instruction: unique item number {tid}, and price lower than 50.00 dollars'
    train_cands = [
        {'task_id': tid, 'instruction_sha256': mod.norm_instruction_hash(goals[tid])}
        for tid in train32_ids
    ]
    return clean, viewed, train_cands, fit501_ids, splits, goals


class IndependentDevFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load_module()

    def test_deterministic_selection_and_zero_overlaps(self):
        clean, viewed, train_cands, fit501, splits, goals = _synthetic_world(self.m)
        selected, audit = self.m.build_independent_dev_freeze(
            clean, viewed, train_cands, fit501, splits, goals, count=4, primary_count=2
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(audit['primary_32_ids'], selected[:2])
        self.assertEqual(audit['reserve_32_ids'], selected[2:])
        self.assertFalse(audit['test_authorization'])
        self.assertEqual(audit['regression_only_task_ids'], [7544])

    def test_rejects_id_overlap_with_train_candidates(self):
        clean, viewed, train_cands, fit501, splits, goals = _synthetic_world(self.m)
        ranked = self.m.rank_clean_pool(clean)
        leak_id = ranked[4]
        train_cands.append({'task_id': leak_id, 'instruction_sha256': self.m.norm_instruction_hash(goals[leak_id])})
        with self.assertRaisesRegex(ValueError, 'Task ID overlap with pr57_train_candidates_32'):
            self.m.build_independent_dev_freeze(
                clean, viewed, train_cands, fit501, splits, goals, count=4, primary_count=2
            )

    def test_rejects_budget_stripped_instruction_overlap_with_test(self):
        clean, viewed, train_cands, fit501, splits, goals = _synthetic_world(self.m)
        ranked = self.m.rank_clean_pool(clean)
        leak_id = ranked[5]
        # Same item text, different price cap -> budget-stripped collision with test!
        goals[300] = f'Instruction: unique item number {leak_id}, and price lower than 99.00 dollars'
        with self.assertRaisesRegex(ValueError, 'Budget-stripped instruction overlap with mapped_test_200'):
            self.m.build_independent_dev_freeze(
                clean, viewed, train_cands, fit501, splits, goals, count=4, primary_count=2
            )

    def test_rejects_cross_goal_twin_anywhere_in_goal_map(self):
        clean, viewed, train_cands, fit501, splits, goals = _synthetic_world(self.m)
        ranked = self.m.rank_clean_pool(clean)
        leak_id = ranked[4]
        goals[9999] = goals[leak_id]
        with self.assertRaisesRegex(ValueError, 'Cross-goal normalized instruction twins'):
            self.m.build_independent_dev_freeze(
                clean, viewed, train_cands, fit501, splits, goals, count=4, primary_count=2
            )


if __name__ == '__main__':
    unittest.main()
