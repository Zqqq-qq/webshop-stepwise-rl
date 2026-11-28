"""Regression tests for scripts/audit_prm55_option_pair.py."""
import copy
import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'audit_prm55_option_pair.py'


def _load_module():
    spec = importlib.util.spec_from_file_location('audit_prm55_option_pair', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_fixture(mod):
    manifest = {
        'identity': {
            'source_commit': '559a7791b49332375c640bc132af05e079676a83',
            'base_revision': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
            'score_position': 'assistant_im_end',
        }
    }
    archived_ep = {
        'schema_version': 1,
        'task_id': mod.TASK_ID,
        'instruction': 'Instruction: find me some highly pigmented eye shadow in color b, and price lower than 30.00 dollars',
        'goal_fingerprint': 'fp-7084',
        'done': True,
        'termination': 'purchase',
        'final_reward': 1.0,
        'success': True,
        'steps': [
            {'step': 0, 'observation': 'obs0', 'available': {'clickables': ['search']},
             'response': 'Thought: s\nAction: search[eye shadow]', 'action': 'search[eye shadow]',
             'grounding': 1, 'error': None, 'environment_reward': 0.9},
            {'step': 1, 'observation': 'obs1', 'available': {'clickables': ['b09g6n33k9']},
             'response': 'Thought: p\nAction: click[b09g6n33k9]', 'action': 'click[b09g6n33k9]',
             'grounding': 1, 'error': None, 'environment_reward': 1.0},
            {'step': 2, 'observation': 'obs2-product', 'available': {'clickables': ['a', 'b', 'buy now']},
             'response': 'Thought: b\nAction: click[b]', 'action': 'click[b]',
             'grounding': 1, 'error': None, 'environment_reward': 1.0},
            {'step': 3, 'observation': 'obs3-b', 'available': {'clickables': ['buy now']},
             'response': 'Thought: buy\nAction: click[buy now]', 'action': 'click[buy now]',
             'grounding': 1, 'error': None, 'environment_reward': 1.0},
        ],
    }
    archived_pred = {
        'task_id': mod.TASK_ID,
        'turn_end_indices': [179, 914, 1081, 1246],
        'step_scores': [0.285, 0.266, 0.081, 0.054],
        'total': 0.686,
    }

    def make_rollout(option, reward):
        steps = copy.deepcopy(archived_ep['steps'][:2])
        steps.append({
            'step': 2, 'observation': 'obs2-product', 'available': {'clickables': ['a', 'b', 'buy now']},
            'response': f'Action: click[{option}]', 'action': f'click[{option}]',
            'grounding': 1, 'error': None, 'environment_reward': reward,
        })
        steps.append({
            'step': 3, 'observation': f'obs3-{option}', 'available': {'clickables': ['buy now']},
            'response': 'Action: click[buy now]', 'action': 'click[buy now]',
            'grounding': 1, 'error': None, 'environment_reward': reward,
        })
        return {
            'schema_version': 1, 'task_id': mod.TASK_ID, 'repeat': 0,
            'instruction': archived_ep['instruction'], 'goal_fingerprint': archived_ep['goal_fingerprint'],
            'done': True, 'termination': 'purchase', 'final_reward': reward,
            'success': reward >= 1 - 1e-6, 'steps': steps,
        }

    def make_pred(scores, indices):
        return {
            'identity': copy.deepcopy(manifest['identity']),
            'input_schema': 'rollout', 'total_samples': 1, 'rejected': [],
            'test_authorization': False,
            'predictions': [{
                'task_id': mod.TASK_ID,
                'turn_end_indices': indices,
                'step_scores': scores,
                'total': sum(scores),
            }],
        }

    b_scores = [0.285000, 0.266000, 0.090000, 0.060000]
    a_scores = [0.285001, 0.265999, 0.040000, 0.050000]
    delta = [b - a for b, a in zip(b_scores, a_scores)]
    artifacts = {
        'environment': {
            'task_id': mod.TASK_ID,
            'source_archive_sha256': mod.ARCHIVE_SHA,
            'health': {'ready': True, 'goal_fingerprint': 'fp-7084',
                       'alignment_mismatches': 0, 'split_status': {'verified': True}},
            'terminal_scores': {'b': 1.0, 'a': 0.5},
            'test_authorization': False,
        },
        'b_rollout': make_rollout('b', 1.0),
        'a_rollout': make_rollout('a', 0.5),
        'b_predictions': make_pred(b_scores, [179, 914, 1020, 1140]),
        'a_predictions': make_pred(a_scores, [179, 914, 1020, 1141]),
        'pair_comparison': {
            'environment_b_minus_a': 0.5,
            'valid_environment_contrast': True,
            'prm_b_minus_a_by_step': delta,
            'prm_b_minus_a_total': sum(b_scores) - sum(a_scores),
            'shared_prefix_within_1e_5': True,
            'test_authorization': False,
        },
    }
    return manifest, archived_ep, archived_pred, artifacts


class OptionPairAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load_module()

    def test_valid_pair_passes_and_computes_exact_deltas(self):
        manifest, ep, pred, art = _synthetic_fixture(self.m)
        res = self.m.audit_pair_artifacts(ep, pred, manifest, art)
        self.assertTrue(res['prefix_verified_against_archive'])
        self.assertTrue(res['controlled_post_fork_wording_verified'])
        self.assertTrue(res['shared_prefix_within_1e_5'])
        self.assertTrue(res['valid_environment_contrast'])
        self.assertTrue(res['pair_comparison_json_verified'])
        self.assertAlmostEqual(res['option_step_b_minus_a'], 0.05, places=9)
        self.assertAlmostEqual(res['buy_step_b_minus_a'], 0.01, places=9)
        self.assertFalse(res['test_authorization'])

    def test_rejects_prefix_drift_from_archived_7084_even_when_arms_match(self):
        manifest, ep, pred, art = _synthetic_fixture(self.m)
        art['b_rollout']['steps'][0]['response'] = 'Action: search[different]'
        art['a_rollout']['steps'][0]['response'] = 'Action: search[different]'
        with self.assertRaisesRegex(ValueError, 'differ from archived task 7084'):
            self.m.audit_pair_artifacts(ep, pred, manifest, art)

    def test_rejects_uncontrolled_post_fork_thought_wording(self):
        manifest, ep, pred, art = _synthetic_fixture(self.m)
        art['a_rollout']['steps'][2]['response'] = 'Thought: pick a\nAction: click[a]'
        with self.assertRaisesRegex(ValueError, 'wording drifted'):
            self.m.audit_pair_artifacts(ep, pred, manifest, art)

    def test_rejects_shared_turn_end_index_drift_vs_archive(self):
        manifest, ep, pred, art = _synthetic_fixture(self.m)
        art['b_predictions']['predictions'][0]['turn_end_indices'][1] = 915
        art['a_predictions']['predictions'][0]['turn_end_indices'][1] = 915
        with self.assertRaisesRegex(ValueError, 'differ from archived task 7084'):
            self.m.audit_pair_artifacts(ep, pred, manifest, art)

    def test_rejects_tampered_pair_comparison_or_test_authorization(self):
        manifest, ep, pred, art = _synthetic_fixture(self.m)
        art['pair_comparison']['prm_b_minus_a_total'] += 0.1
        with self.assertRaisesRegex(ValueError, 'prm_b_minus_a_total mismatch'):
            self.m.audit_pair_artifacts(ep, pred, manifest, art)

        manifest, ep, pred, art = _synthetic_fixture(self.m)
        art['b_predictions']['test_authorization'] = True
        with self.assertRaisesRegex(ValueError, 'test_authorization=False'):
            self.m.audit_pair_artifacts(ep, pred, manifest, art)

    def test_equal_environment_reward_marks_invalid_contrast(self):
        manifest, ep, pred, art = _synthetic_fixture(self.m)
        art['a_rollout']['final_reward'] = 1.0
        art['environment']['terminal_scores']['a'] = 1.0
        art['pair_comparison']['environment_b_minus_a'] = 0.0
        art['pair_comparison']['valid_environment_contrast'] = False
        res = self.m.audit_pair_artifacts(ep, pred, manifest, art)
        self.assertFalse(res['valid_environment_contrast'])
        self.assertEqual(res['environment_b_minus_a'], 0.0)


if __name__ == '__main__':
    unittest.main()
