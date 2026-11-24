"""CPU tests for PRM data conversion, split gating and fit/dev isolation."""
import json
import tempfile
import unittest
from pathlib import Path

from webshop_lab.prm import data


def episode(task, instruction, observations=('o',), done=True, final=1.0):
    steps = [{'observation': obs, 'response': 'Thought: t\nAction: search[x]'}
             for obs in observations]
    return {'task_id': task, 'repeat': 0, 'instruction': instruction, 'steps': steps,
            'done': done, 'final_reward': final,
            'success': bool(done and final >= 1.0 - 1e-6),
            'termination': 'purchase' if done else 'max_steps'}


class RewardTests(unittest.TestCase):
    def test_purchase_keeps_recorded_score(self):
        self.assertAlmostEqual(data.terminal_reward(episode(1, 'buy socks', final=0.75)), 0.75)

    def test_non_purchase_is_zero_even_if_page_score_is_high(self):
        self.assertEqual(data.terminal_reward(episode(1, 'x', done=False, final=0.9)), 0.0)

    def test_missing_done_is_rejected(self):
        row = episode(1, 'x')
        row.pop('done')
        with self.assertRaises(ValueError):
            data.terminal_reward(row)

    def test_nan_and_inf_are_rejected_not_zeroed(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                data.terminal_reward(episode(1, 'x', done=False, final=value))

    def test_boolean_score_is_rejected(self):
        with self.assertRaises(ValueError):
            data.terminal_reward(episode(1, 'x', final=True))

    def test_environment_error_is_a_technical_interruption(self):
        row = episode(1, 'x', done=False, final=0.0)
        row['termination'] = 'environment_error'
        with self.assertRaises(data.TechnicalInterruption):
            data.terminal_reward(row)

    def test_done_and_termination_must_agree(self):
        row = episode(1, 'x', done=False, final=0.0)
        row['termination'] = 'purchase'
        with self.assertRaises(ValueError):
            data.terminal_reward(row)

    def test_success_must_match_purchase(self):
        row = episode(1, 'x', done=True, final=0.5)
        row['success'] = True
        with self.assertRaises(ValueError):
            data.terminal_reward(row)


class MessageTests(unittest.TestCase):
    def test_messages_alternate_and_start_with_system(self):
        messages = data.episode_messages(episode(1, 'buy socks', ('o1', 'o2')))
        self.assertEqual([m['role'] for m in messages],
                         ['system', 'user', 'assistant', 'user', 'assistant'])
        self.assertIn('Shopping instruction: buy socks', messages[0]['content'])

    def test_inference_messages_do_not_require_labels(self):
        row = {'task_id': 1, 'instruction': 'buy socks',
               'steps': [{'observation': 'o', 'response': 'Thought: t\nAction: search[x]'}]}
        messages = data.episode_messages(row, require_labels=False)
        self.assertEqual(messages[-1]['role'], 'assistant')

    def test_training_messages_require_labels(self):
        row = {'task_id': 1, 'instruction': 'buy socks',
               'steps': [{'observation': 'o', 'response': 'Thought: t\nAction: search[x]'}]}
        with self.assertRaises(ValueError):
            data.episode_messages(row)

    def test_empty_assistant_action_is_rejected(self):
        row = episode(1, 'x')
        row['steps'][0]['response'] = '   '
        with self.assertRaises(ValueError):
            data.episode_messages(row)


class GateTests(unittest.TestCase):
    def build(self, tmp, train, val, test, goal):
        d = Path(tmp)
        for name, ids in (('train', train), ('val', val), ('test', test)):
            (d / (name + '_ids.json')).write_text(json.dumps(ids), encoding='utf-8')
        map_path = d / 'goal.json'
        map_path.write_text(json.dumps(goal), encoding='utf-8')
        splits, digests = data.load_split(d)
        mapping, sha = data.load_goal_map(map_path)
        return splits, mapping, digests, sha

    def test_val_tasks_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            goal = [{'task_id': i, 'instruction': 'inst %d' % i} for i in range(1, 7)]
            splits, mapping, _, _ = self.build(tmp, [1, 2, 3], [4, 5], [6], goal)
            with self.assertRaises(data.SourceNotAllowed):
                data.gate_train_pool([episode(4, 'inst 4', final=0.5)], splits, mapping)

    def test_train_tasks_pass_with_matching_goal_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            goal = [{'task_id': i, 'instruction': 'inst %d' % i} for i in range(1, 7)]
            splits, mapping, _, _ = self.build(tmp, [1, 2, 3], [4], [5], goal)
            info = data.gate_train_pool([episode(1, 'inst 1', final=0.5)], splits, mapping)
            self.assertEqual(info['instruction_overlap_with_eval'], 0)
            self.assertEqual(info['forbidden_instructions'], 2)

    def test_instruction_mismatch_against_goal_map_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            goal = [{'task_id': i, 'instruction': 'inst %d' % i} for i in range(1, 7)]
            splits, mapping, _, _ = self.build(tmp, [1, 2, 3], [4], [5], goal)
            with self.assertRaises(data.SourceNotAllowed):
                data.gate_train_pool([episode(1, 'different text', final=0.5)], splits, mapping)

    def test_task_missing_from_goal_map_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            goal = [{'task_id': i, 'instruction': 'inst %d' % i} for i in range(1, 3)]
            splits, mapping, _, _ = self.build(tmp, [1, 2, 3], [4], [5], goal)
            with self.assertRaises(data.SourceNotAllowed):
                data.gate_train_pool([episode(3, 'inst 3', final=0.5)], splits, mapping)

    def test_partial_map_covering_only_input_is_rejected(self):
        """Zqqq 复现的绕过：map 只覆盖输入 train 任务、漏掉 val/test 时不能报零重叠。"""
        with tempfile.TemporaryDirectory() as tmp:
            splits, mapping, _, _ = self.build(tmp, [1, 2], [3], [4],
                                               [{'task_id': 1, 'instruction': 'buy socks'}])
            with self.assertRaises(data.SourceNotAllowed) as ctx:
                data.gate_train_pool([episode(1, 'buy socks', final=0.5)], splits, mapping)
            self.assertIn('goal-map', str(ctx.exception))

    def test_complete_map_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            goal = [{'task_id': i, 'instruction': 'inst %d' % i} for i in range(1, 5)]
            splits, mapping, _, _ = self.build(tmp, [1, 2], [3], [4], goal)
            info = data.gate_train_pool([episode(1, 'inst 1', final=0.5)], splits, mapping)
            self.assertEqual(info['forbidden_tasks'], 2)
            self.assertEqual(info['forbidden_instructions'], 2)

    def test_environment_instruction_prefix_matches_goal_map(self):
        """环境回传的指令带 `Instruction:` 前缀，身份比对必须先去掉。"""
        with tempfile.TemporaryDirectory() as tmp:
            goal = [{'task_id': i, 'instruction': 'inst %d' % i} for i in range(1, 5)]
            splits, mapping, _, _ = self.build(tmp, [1, 2], [3], [4], goal)
            info = data.gate_train_pool([episode(1, 'Instruction: inst 1', final=0.5)],
                                        splits, mapping)
            self.assertEqual(info['instruction_overlap_with_eval'], 0)

    def test_val_instruction_leak_is_rejected_even_for_train_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            goal = [{'task_id': 1, 'instruction': 'shared text'},
                    {'task_id': 4, 'instruction': 'Shared  Text'},
                    {'task_id': 2, 'instruction': 'inst 2'},
                    {'task_id': 5, 'instruction': 'inst 5'}]
            splits, mapping, _, _ = self.build(tmp, [1, 2], [4], [5], goal)
            with self.assertRaises(data.SourceNotAllowed):
                data.gate_train_pool([episode(1, 'shared text', final=0.5)], splits, mapping)

    def test_missing_goal_map_raises(self):
        with self.assertRaises(ValueError):
            data.load_goal_map('does-not-exist.json')

    def test_overlapping_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            goal = [{'task_id': i, 'instruction': 'x'} for i in range(1, 4)]
            with self.assertRaises(ValueError):
                self.build(tmp, [1, 2], [2], [3], goal)


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        rows = [episode(i, 'instruction %d' % i, final=0.5) for i in range(10)]
        fit_a, dev_a, summary_a = data.split_by_task(rows, 0.7, seed=7)
        fit_b, dev_b, _ = data.split_by_task(rows, 0.7, seed=7)
        self.assertEqual([r['task_id'] for r in fit_a], [r['task_id'] for r in fit_b])
        self.assertEqual(set(r['task_id'] for r in fit_a) & set(r['task_id'] for r in dev_a), set())
        self.assertEqual(summary_a['instruction_overlap'], 0)

    def test_duplicate_normalized_instructions_across_split_are_rejected(self):
        rows = [episode(1, 'Buy  Socks', final=0.5),
                episode(2, 'buy socks', final=0.5),
                episode(3, 'other', final=0.5)]
        with self.assertRaises(ValueError):
            data.split_by_task(rows, 0.34, seed=1)


class BuildTests(unittest.TestCase):
    def test_overlong_trajectory_is_counted_not_truncated(self):
        rows = [episode(1, 'a')]
        original = data.tokenize_trajectory

        def boom(*args, **kwargs):
            raise data.TrajectoryTooLong('too long')

        data.tokenize_trajectory = boom
        try:
            samples, rejected, dropped = data.build_samples(rows, object(), 8)
        finally:
            data.tokenize_trajectory = original
        self.assertEqual(samples, [])
        self.assertEqual(rejected['too_long'], 1)
        self.assertEqual(dropped[0]['task_id'], 1)
        self.assertEqual(dropped[0]['reason'], 'too_long')

    def test_technical_interruption_is_counted_separately(self):
        rows = [episode(1, 'a', done=False, final=0.0)]
        rows[0]['termination'] = 'environment_error'
        samples, rejected, dropped = data.build_samples(rows, object(), 2048)
        self.assertEqual(samples, [])
        self.assertEqual(rejected['technical_interruption'], 1)

    def test_empty_action_is_counted(self):
        rows = [episode(1, 'a')]
        rows[0]['steps'][0]['response'] = '  '
        samples, rejected, _ = data.build_samples(rows, object(), 2048)
        self.assertEqual(samples, [])
        self.assertEqual(rejected.get('empty_action'), 1)


class ManifestTests(unittest.TestCase):
    def samples(self):
        return [{'task_id': 1, 'repeat': 0, 'instruction_sha256': 'a' * 64,
                 'terminal_reward': 1.0, 'num_turns': 2, 'num_tokens': 10,
                 'template_sha256': 'b' * 64, 'source': {}, 'encoded': {}}]

    def test_manifest_records_identity_and_drops(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = data.write_dataset(
                Path(tmp) / 'ds', 'f' * 64, [{'path': 'r.jsonl', 'sha256': 'c' * 64}],
                {'seed': 42, 'fit_fraction': 0.8, 'fit_tasks': 1, 'dev_tasks': 1,
                 'fit_episodes': 1, 'dev_episodes': 1, 'instruction_overlap': 0},
                self.samples(), self.samples(),
                {'fit': {'too_long': 1}, 'dev': {}},
                [{'task_id': 9, 'instruction_sha256': 'e' * 64, 'reason': 'too_long'}],
                {'path': 'tok', 'revision': 'd' * 40, 'template_sha256': 'b' * 64},
                '1' * 64, {'train_ids': '2' * 64, 'val_ids': '3' * 64, 'test_ids': '4' * 64})
            self.assertEqual(manifest['goal_map_sha256'], '1' * 64)
            self.assertEqual(manifest['split_sha256']['val_ids'], '3' * 64)
            self.assertEqual(manifest['dropped_tasks'], [9])
            self.assertEqual(manifest['files']['fit']['label_buckets'], {'full': 1})

    def test_empty_split_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                data.write_dataset(
                    Path(tmp) / 'ds', 'f' * 64, [], {},
                    self.samples(), [], {'fit': {}, 'dev': {}}, [],
                    {'path': 'tok', 'revision': 'd' * 40, 'template_sha256': 'b' * 64},
                    '1' * 64, {})


if __name__ == '__main__':
    unittest.main()
