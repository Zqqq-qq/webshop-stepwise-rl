"""Regression tests for scripts/freeze_prm59_reverify_dev.py."""
import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'freeze_prm59_reverify_dev.py'


def _load_module():
    spec = importlib.util.spec_from_file_location('freeze_prm59_reverify_dev', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PR59ReverifyDevFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load_module()

    def test_excludes_pr55_pr57_twins_and_budget_stripped_collisions(self):
        clean = list(range(100, 114))  # 14 clean tasks
        ranked = self.m.rank_pool(clean)
        v55 = ranked[:4]
        v57 = ranked[4:6]
        # Remaining 8 tasks: make ranked[6] a cross-goal twin and ranked[7] a budget-stripped collision with fit501
        twin_id = ranked[6]
        ch_collide_id = ranked[7]
        fit501 = {1, 2}
        train32 = [{'task_id': 1, 'instruction_sha256': 'x'}]
        splits = {'train': set(clean) | fit501, 'val': {200}, 'test': {300}}
        goals = {
            t: f'Instruction: distinct product {t}, and price lower than 50.00 dollars'
            for t in (splits['train'] | splits['val'] | splits['test'])
        }
        goals[9999] = goals[twin_id]  # external twin
        goals[2] = f'Instruction: distinct product {ch_collide_id}, and price lower than 99.00 dollars'

        selected, primary, reserve, audit = self.m.build_pr59_reverify_dev(
            clean_pool_444=clean,
            viewed_pr55_160=v55,
            viewed_pr57_64=v57,
            train_candidates_32=train32,
            fit501_ids=fit501,
            splits=splits,
            goals=goals,
            count=4,
            primary_count=2,
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(primary, selected[:2])
        self.assertEqual(reserve, selected[2:])
        self.assertNotIn(twin_id, selected)
        self.assertNotIn(ch_collide_id, selected)
        self.assertEqual(audit['pool_accounting']['excluded_cross_goal_twins'], {str(twin_id): [9999]})
        self.assertEqual(audit['pool_accounting']['excluded_budget_stripped_collisions'], [ch_collide_id])
        self.assertEqual(audit['isolation_checks']['cross_goal_twins_in_12087'], 0)
        self.assertFalse(audit['test_authorization'])


if __name__ == '__main__':
    unittest.main()
