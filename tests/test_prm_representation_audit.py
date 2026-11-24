"""Synthetic contract tests; never presented as a real Qwen extraction."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'representation_audit', Path(__file__).resolve().parents[1] /
    'scripts/audit_prm_representation.py')
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)
try:
    import numpy as np
except ImportError:
    np = None


class SyntheticTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [0, 1, 9, 2]


@unittest.skipIf(np is None, 'Run the numerical audit suite with NumPy explicitly installed')
class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [{'task_id': i, 'messages': [
            {'role': 'system', 'content': 'synthetic'},
            {'role': 'user', 'content': 'synthetic observation'},
            {'role': 'assistant', 'content': 'synthetic action'}],
            'terminal_reward': float(i % 2)} for i in range(8)]
        self.positions = [{'task_id': i, 'input_ids': [0, 1, 9, 2],
                           'turn_end_indices': [2]} for i in range(8)]
        self.z = np.eye(8, 10, dtype=np.float32)
        self.F = np.c_[self.z, np.ones(8)].astype(np.float32)
        self.y = np.array([r['terminal_reward'] for r in self.rows])
        self.meta = {'schema_version': 'prm-representation-export-v1',
                     'vocab_size': 10, 'im_end_token_id': 9,
                     'feature_definition': 'concat(sum_step_raw_logits,turn_count)',
                     'score_position': 'assistant_im_end', 'eval_mode': True,
                     'dropout_enabled': False, 'labels_in_forward': False,
                     'parameter_dtype': 'float32', 'model_identity': {'scope': 'synthetic'}}
        self.save()

    def save(self, steps=True):
        arrays = {'F': self.F, 'y': self.y}
        if steps:
            arrays.update(step_logits=self.z, offsets=np.arange(9, dtype=np.int64))
        np.savez(self.root/'features.npz', **arrays)
        (self.root/'samples.jsonl').write_text(
            ''.join(json.dumps(r)+'\n' for r in self.rows), encoding='utf-8')
        (self.root/'positions.json').write_text(json.dumps(self.positions), encoding='utf-8')
        self.meta['files_sha256'] = {name: audit_module.sha256(self.root/name)
                                    for name in ('features.npz','samples.jsonl','positions.json')}
        (self.root/'manifest.json').write_text(json.dumps(self.meta), encoding='utf-8')

    def run_audit(self, **kwargs):
        return audit_module.audit(*(self.root/p for p in (
            'features.npz','samples.jsonl','positions.json','manifest.json')), **kwargs)

    def test_full_row_rank_fit_and_step_reconstruction(self):
        r = self.run_audit(tokenizer=SyntheticTokenizer())
        self.assertEqual(r['rank'], 8)
        self.assertLess(r['residual_max_abs'], 1e-12)
        self.assertLess(r['fp32_step_then_sum_max_abs_label_error'], 1e-6)
        self.assertEqual(r['status'], 'representation_inputs_checked')
        self.assertIsNone(r['generalization_passed'])
        self.assertIsNone(r['memory_training_passed'])
        self.assertFalse(r['extraction_runtime_reproduced'])

    def test_F_only_is_partial_not_step_verification(self):
        self.save(steps=False)
        r = self.run_audit(tokenizer=SyntheticTokenizer())
        self.assertFalse(r['F_reconstructed_from_raw_steps'])
        self.assertIn('partial', r['status'])

    def test_token_ids_alone_do_not_verify_real_template(self):
        self.assertIn('partial', self.run_audit()['status'])

    def test_added_trajectory_intercept_is_rejected(self):
        self.F = np.c_[self.F, np.ones(8)]
        self.save()
        with self.assertRaises(ValueError): self.run_audit()

    def test_wrong_bias_coefficient_is_rejected(self):
        self.F[0,-1] = 2
        self.save()
        with self.assertRaises(ValueError): self.run_audit()

    def test_aggregates_must_match_raw_steps(self):
        self.F[0,0] = 2
        self.save()
        with self.assertRaises(ValueError): self.run_audit()

    def test_changed_label_is_rejected(self):
        self.y[0] = 1
        self.save()
        with self.assertRaises(ValueError): self.run_audit()

    def test_nonfinite_feature_is_rejected(self):
        self.F[0,0] = np.nan
        self.save()
        with self.assertRaises(ValueError): self.run_audit()

    def test_reordered_tasks_are_rejected(self):
        self.positions.reverse()
        self.save()
        with self.assertRaises(ValueError): self.run_audit()

    def test_duplicate_task_is_rejected(self):
        self.rows[-1]['task_id'] = self.rows[0]['task_id']
        self.positions[-1]['task_id'] = self.rows[0]['task_id']
        self.save()
        with self.assertRaises(ValueError): self.run_audit()

    def test_nonassistant_position_is_rejected(self):
        self.positions[0]['turn_end_indices'] = [0]
        self.save()
        with self.assertRaises(ValueError): self.run_audit()

    def test_wrong_real_tokenizer_input_is_rejected(self):
        class Wrong(SyntheticTokenizer):
            def apply_chat_template(self, *a, **kw): return [0,9]
        with self.assertRaises(ValueError): self.run_audit(tokenizer=Wrong())

    def test_tampered_file_hash_is_rejected(self):
        with (self.root/'samples.jsonl').open('a') as out: out.write('\n')
        with self.assertRaises(ValueError): self.run_audit()

    def test_training_mode_or_reward_input_declarations_are_rejected(self):
        for name, value in [('eval_mode',False), ('labels_in_forward',True),
                            ('dropout_enabled',True), ('parameter_dtype','bfloat16')]:
            old = self.meta[name]
            self.meta[name] = value
            self.save()
            with self.assertRaises(ValueError): self.run_audit()
            self.meta[name] = old

    def test_invalid_rcond_is_rejected(self):
        for rcond in (0, -1, 1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): self.run_audit(rcond=rcond)

    def test_cli_never_overwrites_existing_report(self):
        report = self.root/'output.json'
        report.write_text('preserve')
        argv=[]
        for name in ('features','samples','positions','manifest'):
            extension = '.npz' if name=='features' else ('.jsonl' if name=='samples' else '.json')
            argv += ['--'+name, str(self.root/(name+extension))]
        with self.assertRaises(SystemExit):
            audit_module.main(argv+['--output',str(report)])
        self.assertEqual(report.read_text(),'preserve')


if __name__ == '__main__': unittest.main()
