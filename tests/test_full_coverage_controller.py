"""CPU-only scheduling tests; the child writes synthetic summaries, never loads a model."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


STUB = '''import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser()
for k in ('stage','config','ids','split-dir','goal-map','output','resume'):
 p.add_argument('--'+k)
a=p.parse_args()
stage=Path(a.stage);out=Path(a.output);index=int(out.name.split('-')[-1])
if (stage/'fail-second').exists() and index==2:raise SystemExit(7)
if index==1:
 assert a.resume is None
 before,critic,parent='initial-actor','initial-critic',None
else:
 assert Path(a.resume)==out.parent/('round-'+str(index-1))
 previous=json.loads((Path(a.resume)/'summary.json').read_text())
 before,critic,parent=previous['actor_after'],previous['critic_after'],previous['run_id']
ids=json.loads(Path(a.ids).read_text())
summary={'updates':1,'cumulative_updates':index,'same_process_reload':True,
 'run_id':str(out),'parent_run_id':parent,'actor_before':before,'critic_before':critic,
 'actor_after':str(out)+'actor','critic_after':str(out)+'critic',
 'loss':.1,'gradient_norm':1.,'replay_ratio_min':1.,'replay_ratio_max':1.,
 'turns':len(ids),'environment_calls':len(ids),'generated_tokens':len(ids)*2}
out.mkdir(parents=True)
(out/'summary.json').write_text(json.dumps(summary))
(out/'run.meta.json').write_text(json.dumps({'ids':ids,'update_index':index}))
out.with_name(out.name+'.audit.json').write_text(json.dumps({'status':'complete'}))
'''


@unittest.skipUnless(os.name == 'posix', 'The real supervisor uses Linux/WSL file locks')
class FullCoverageControllerTests(unittest.TestCase):
    def prepare(self, folder):
        stage = Path(folder)
        (stage / 'run-update.py').write_text(STUB, encoding='utf-8')
        ids = []
        for i, batch in enumerate([[1, 2, 3, 4], [5, 6, 7]], 1):
            path = stage / ('ids-' + str(i) + '.json')
            path.write_text(json.dumps(batch), encoding='utf-8')
            ids.append({'path': str(path), 'tasks': batch})
        config = stage / 'config.json'
        config.write_text('{}', encoding='utf-8')
        native = {'schedule': ids,
                  'arms': {arm: {'config': str(config)} for arm in ['terminal', 'candidate']},
                  'equal_budget_caps': {'policy_turn_cap': 56, 'environment_call_cap': 56,
                                        'generated_token_cap': 7168}}
        native_path = stage / 'native.json'
        native_path.write_text(json.dumps(native), encoding='utf-8')
        jobs = [{'label': label, 'arm': arm, 'preflight_plan': str(native_path),
                 'output': str(stage / label)}
                for label, arm in [('tg42', 'terminal'), ('op42', 'candidate'),
                                   ('tg43', 'terminal'), ('op43', 'candidate')]]
        plan = {'execution_source_dir': str(stage), 'python': sys.executable,
                'lock_dir': str(stage / 'locks'), 'jobs': jobs, 'split_dir': str(stage),
                'goal_map': str(config), 'minimum_free_disk_bytes': 0,
                'total_updates': 8, 'total_episodes': 28, 'train_task_count': 7,
                'initial_actor_tensor_digest': 'initial-actor',
                'initial_critic_tensor_digest': 'initial-critic'}
        (stage / 'experiment-plan.json').write_text(json.dumps(plan), encoding='utf-8')
        controller = Path(__file__).with_name('run_full_coverage_training.py')
        if not controller.exists():
            controller = Path(__file__).resolve().parents[1] / 'scripts/run_full_coverage_training.py'
        return stage, controller

    def test_all_four_chains_start_fresh_and_keep_the_short_final_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, controller = self.prepare(folder)
            result = subprocess.run([sys.executable, str(controller), '--stage', str(stage)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            state = json.loads((stage / 'progress.json').read_text())
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['completed_updates'], 8)
            self.assertEqual(state['completed_episodes'], 28)
            self.assertEqual(set(state['chains']), {'tg42', 'op42', 'tg43', 'op43'})
            self.assertTrue(all(c['completed_episodes'] == 7 for c in state['chains'].values()))

    def test_failure_preserves_finished_update_and_stops_before_other_chains(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, controller = self.prepare(folder)
            (stage / 'fail-second').touch()
            result = subprocess.run([sys.executable, str(controller), '--stage', str(stage)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            state = json.loads((stage / 'progress.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['completed_updates'], 1)
            self.assertEqual(state['completed_episodes'], 4)
            self.assertTrue((stage / 'tg42/round-1/summary.json').exists())
            self.assertFalse((stage / 'op42').exists())


if __name__ == '__main__':
    unittest.main()
