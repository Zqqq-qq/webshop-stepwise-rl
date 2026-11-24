"""CPU subprocess regression tests; these are not GPU training evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1]/'scripts'/'prm_checked_pipeline.py'


class PipelineTests(unittest.TestCase):
    def run_case(self, ids, rows, exit_code=0, meta_ids=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'ids.json').write_text(json.dumps(ids))
            marker = root/'TRAINING_STARTED'
            rollout = root/'rollout.jsonl'
            meta = root/'rollout.meta.json'
            sample = [sys.executable, '-c',
                      'import pathlib,sys; '
                      f'pathlib.Path({str(rollout)!r}).write_text({"".join(json.dumps(r)+chr(10) for r in rows)!r}); '
                      f'pathlib.Path({str(meta)!r}).write_text({json.dumps({"run_id":"sample-run", "selected_ids": ids if meta_ids is None else meta_ids, "repeats":1})!r}); '
                      f'sys.exit({exit_code})']
            plan = {'cwd': str(root), 'expected_ids': str(root/'ids.json'),
                    'rollout': str(rollout), 'metadata': str(meta),
                    'stages': [{'name':'sample','command':sample},
                               {'name':'train','command':[sys.executable,'-c',
                                   f'from pathlib import Path; Path({str(marker)!r}).touch()']}]}
            p = root/'plan.json'; p.write_text(json.dumps(plan))
            result = subprocess.run([sys.executable,str(SCRIPT),'--plan',str(p),
                                     '--output',str(root/'attempt')],capture_output=True,text=True)
            status_path=root/'attempt/status.json'
            self.assertTrue(status_path.exists(),result.stderr)
            return result.returncode, marker.exists(), json.loads(status_path.read_text())

    def row(self, i, **overrides):
        return {'task_id':i,'repeat':0,'run_id':'sample-run','done':False,
                'termination':'max_steps', **overrides}

    def test_sampler_failure_stops_before_training(self):
        code, started, status=self.run_case([1,2],[self.row(1)],7)
        self.assertNotEqual(code,0)
        self.assertFalse(started)
        self.assertEqual(status['status'],'failed')
        self.assertEqual(status['stage'],'sample')

    def test_success_exit_with_partial_coverage_still_stops(self):
        code, started, status=self.run_case([1,2],[self.row(1)])
        self.assertNotEqual(code,0)
        self.assertFalse(started)
        self.assertIn('coverage',status['error'])

    def test_duplicate_cannot_hide_missing_task(self):
        code, started, _=self.run_case([1,2],[self.row(1),self.row(1)])
        self.assertNotEqual(code,0)
        self.assertFalse(started)

    def test_exact_coverage_allows_next_stage(self):
        code, started, status=self.run_case([1,2],[self.row(1),self.row(2)])
        self.assertEqual(code,0,status)
        self.assertTrue(started)
        self.assertEqual(status['status'],'complete')
        self.assertEqual(status['coverage']['completed_tasks'],2)

    def test_technical_error_does_not_count_as_completed(self):
        code, started, _=self.run_case([1],[self.row(1,termination='environment_error')])
        self.assertNotEqual(code,0)
        self.assertFalse(started)

    def test_metadata_or_run_identity_mismatch_stops(self):
        for rows, meta_ids in [([self.row(1)],[2]),
                               ([self.row(1,run_id='other')],[1])]:
            code, started, _=self.run_case([1],rows,meta_ids=meta_ids)
            self.assertNotEqual(code,0)
            self.assertFalse(started)


if __name__ == '__main__':
    unittest.main()
