"""Exercise both real reload entry points with a lightweight CPU fake runtime.

These tests validate orchestration, not GPU learning or checkpoint identity.
"""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from prm_reload_validation import compare_predictions


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixtures():
    return [{'task_id': i, 'total': float(i % 2),
             'turn_end_indices': [5, 10] if i < 4 else [5, 10, 15],
             'step_scores': ([0.0, float(i % 2)] if i < 4
                             else [0.0, 0.0, float(i % 2)])} for i in range(8)]


def module(name, **attributes):
    result = types.ModuleType(name)
    result.__dict__.update(attributes)
    return result


class Scores:
    def __init__(self, values): self.values = values
    def cpu(self): return self
    def tolist(self): return self.values


class ReloadEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diagnostic = load_script('run_prm_diagnostic')
        cls.memory = load_script('run_prm_memory_probe')

    def exercise(self, entry, *, reference=None, current=None):
        reference = copy.deepcopy(fixtures() if reference is None else reference)
        current = copy.deepcopy(fixtures() if current is None else current)
        by_id = {r['task_id']: r for r in current}
        encoded = [(r['task_id'], {'task_id': r['task_id'],
                    'terminal_reward': float(r['task_id'] % 2),
                    'turn_end_indices': r['turn_end_indices']}) for r in current]

        class Model:
            def __call__(self, task_id):
                row = by_id[task_id]
                return {'totals': [row['total']], 'step_scores': [Scores(row['step_scores'])]}

        fake = {
            'torch': module('torch', inference_mode=contextlib.nullcontext),
            'webshop_lab': module('webshop_lab'),
            'webshop_lab.prm': module('webshop_lab.prm'),
            'webshop_lab.prm.config': module('config', load_config=lambda _: types.SimpleNamespace(
                max_length=2048, sft_adapter_files_sha256={'adapter_model.safetensors': 'sha'})),
            'webshop_lab.prm.runtime': module('runtime',
                build_prm=lambda *a, **k: (Model(), types.SimpleNamespace(pad_token_id=0), None, None),
                tensor_batch=lambda rows, *a: {'task_id': rows[0]['task_id']}),
            'webshop_lab.prm.train': module('train',
                load_samples=lambda *a: (encoded, [], len(encoded)),
                predict=lambda *a: current),
        }
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, fake), contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            (root / 'ids.json').write_text(json.dumps(list(range(8))))
            for split in ('fit', 'dev'):
                (root / (split + '.jsonl')).write_text(''.join(
                    json.dumps({'task_id': i}) + '\n' for i in range(8)))
            (root / 'reference-predictions.json').write_text(json.dumps(reference))
            if entry == 'diagnostic':
                self.diagnostic.assess({'train_ids': str(root / 'ids.json'),
                                       'fit': str(root / 'fit.jsonl'), 'dev': str(root / 'dev.jsonl')},
                                       {'config': 'config', 'output': str(root)})
                return json.loads((root / 'diagnostic-evaluation.json').read_text())['metrics']
            self.memory.evaluate({'train_ids': str(root / 'ids.json'), 'samples': 'samples',
                                  'labels': {str(i): float(i % 2) for i in range(8)}}, root)
            return json.loads((root / 'memory-independent-reload.json').read_text())

    def test_complete_identical_records_pass_both_real_entries(self):
        diagnostic = self.exercise('diagnostic')
        memory = self.exercise('memory')
        self.assertEqual(diagnostic['independent_reload_comparison']['tasks_compared'], 8)
        self.assertEqual(diagnostic['independent_reload_comparison']['steps_compared'], 20)
        self.assertEqual(memory['reload_comparison']['max_total_difference'], 0.0)
        self.assertTrue(memory['metrics']['passed'])

    def test_reordered_reference_is_aligned_by_task_id(self):
        for entry in ('diagnostic', 'memory'):
            with self.subTest(entry=entry): self.exercise(entry, reference=list(reversed(fixtures())))

    def test_reference_coverage_errors_are_rejected_by_both_entries(self):
        cases = {'empty': [], 'missing': fixtures()[:-1],
                 'duplicate': fixtures() + [fixtures()[0]],
                 'unexpected': fixtures() + [{**fixtures()[0], 'task_id': 99}]}
        for entry in ('diagnostic', 'memory'):
            for name, rows in cases.items():
                with self.subTest(entry=entry, case=name), self.assertRaises(ValueError):
                    self.exercise(entry, reference=rows)

    def test_step_alignment_errors_rejected_by_both_entries(self):
        for entry in ('diagnostic', 'memory'):
            for name in ('short', 'long', 'empty', 'shifted', 'duplicate-position', 'unsorted'):
                rows = fixtures()
                if name == 'short': rows[0]['step_scores'].pop()
                if name == 'long': rows[0]['step_scores'].append(0.0)
                if name == 'empty': rows[0].update(step_scores=[], turn_end_indices=[])
                if name == 'shifted': rows[0]['turn_end_indices'] = [6, 11]
                if name == 'duplicate-position': rows[0]['turn_end_indices'] = [5, 5]
                if name == 'unsorted': rows[0]['turn_end_indices'] = [10, 5]
                with self.subTest(entry=entry, case=name), self.assertRaises(ValueError):
                    self.exercise(entry, reference=rows)

    def test_nonfinite_scores_and_totals_rejected_on_both_sides(self):
        for entry in ('diagnostic', 'memory'):
            for side in ('reference', 'current'):
                for field in ('step_scores', 'total'):
                    for bad in (float('nan'), float('inf'), -float('inf')):
                        rows = fixtures()
                        if field == 'step_scores': rows[0][field][0] = bad
                        else: rows[0][field] = bad
                        with self.subTest(entry=entry, side=side, field=field, bad=bad), self.assertRaises(ValueError):
                            self.exercise(entry, **{side: rows})

    def test_total_disagrees_with_own_steps_is_rejected(self):
        for entry in ('diagnostic', 'memory'):
            for side in ('reference', 'current'):
                rows = fixtures(); rows[0]['total'] = 0.1
                with self.subTest(entry=entry, side=side), self.assertRaises(ValueError):
                    self.exercise(entry, **{side: rows})

    def test_each_step_must_match_even_when_totals_match(self):
        rows = fixtures(); rows[0]['step_scores'] = [0.1, -0.1]
        for entry in ('diagnostic', 'memory'):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                self.exercise(entry, reference=rows)

    def test_total_comparison_catches_accumulated_small_step_differences(self):
        rows = fixtures(); rows[0].update(step_scores=[6e-6, 6e-6], total=12e-6)
        for entry in ('diagnostic', 'memory'):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                self.exercise(entry, reference=rows)

    def test_small_finite_differences_pass_with_explicit_tolerance(self):
        rows = fixtures(); rows[0].update(step_scores=[0.0, 5e-6], total=5e-6)
        for entry in ('diagnostic', 'memory'):
            with self.subTest(entry=entry): self.exercise(entry, reference=rows)

    def test_current_coverage_errors_rejected_by_both_entries(self):
        for entry in ('diagnostic', 'memory'):
            for name in ('duplicate', 'missing', 'unexpected'):
                rows = fixtures()
                if name == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
                if name == 'missing': rows.pop()
                if name == 'unexpected': rows[-1]['task_id'] = 99
                with self.subTest(entry=entry, case=name), self.assertRaises(ValueError):
                    self.exercise(entry, current=rows)

    def test_shared_validator_rejects_jointly_missing_tasks(self):
        with self.assertRaises(ValueError):
            compare_predictions(fixtures()[:-1], fixtures()[:-1], range(8))

    def test_invalid_tolerances_are_rejected(self):
        for bad in (-1.0, float('nan'), float('inf')):
            with self.subTest(atol=bad), self.assertRaises(ValueError):
                compare_predictions(fixtures(), fixtures(), range(8), atol=bad)


if __name__ == '__main__':
    unittest.main()
