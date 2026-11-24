"""Execute the frozen full-coverage schedule serially; stop on the first failure."""
import argparse
import datetime
import fcntl
import json
import math
import os
import shutil
import subprocess
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage
    plan = read(stage / 'experiment-plan.json')
    source = Path(plan['execution_source_dir'])
    python = plan['python']
    lock_dir = Path(plan.get('lock_dir', '/root/webshop-main/artifacts'))
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = open(lock_dir / 'gpu-training.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    # Also exclude the validation supervisor used by the preceding experiment.
    validation_lock = open(lock_dir / 'fp32-validation.lock', 'a')
    fcntl.flock(validation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {**os.environ, 'PYTHONPATH': str(source / 'src'),
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    state = {'status': 'running', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'completed_updates': 0, 'completed_episodes': 0, 'chains': {},
             'target_updates': plan['total_updates'], 'target_episodes': plan['total_episodes'],
             'test_authorization': False}

    def save():
        temporary = stage / 'progress.tmp.json'
        temporary.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        temporary.replace(stage / 'progress.json')

    save()
    try:
        for job in plan['jobs']:
            label = job['label']
            native = read(Path(job['preflight_plan']))
            arm = job['arm']
            config = native['arms'][arm]['config']
            state['chains'][label] = {'completed_updates': 0, 'completed_episodes': 0,
                                     'turns': 0, 'environment_calls': 0, 'generated_tokens': 0,
                                     'status': 'running'}
            previous, parent = None, None
            for index, batch in enumerate(native['schedule'], 1):
                minimum = plan['minimum_free_disk_bytes']
                assert shutil.disk_usage(stage).free > minimum, 'Low Linux filesystem space'
                host_path = Path(plan.get('host_filesystem_path', stage))
                assert shutil.disk_usage(host_path).free > minimum, 'Low host filesystem space'
                output = Path(job['output']) / ('round-' + str(index))
                state['current'] = {'label': label, 'update': index,
                                    'updates_in_chain': len(native['schedule']),
                                    'task_ids': batch['tasks'], 'output': str(output)}
                save()
                cmd = [python, '-u', str(stage / 'run-update.py'), '--stage', str(stage),
                       '--config', config, '--ids', batch['path'],
                       '--split-dir', plan['split_dir'], '--goal-map', plan['goal_map'],
                       '--output', str(output)]
                if previous is not None:
                    cmd += ['--resume', str(previous)]
                log_path = stage / 'logs' / (label + '-round-' + str(index) + '.log')
                log_path.parent.mkdir(exist_ok=True)
                log_path.with_suffix('.command.json').write_text(
                    json.dumps({'command': cmd, 'cwd': str(source)}, indent=2) + '\n',
                    encoding='utf-8')
                print('UPDATE_START', label, index, batch['tasks'], flush=True)
                with log_path.open('xb') as log:
                    result = subprocess.run(cmd, cwd=source, env=env, stdout=log,
                                            stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'{label} update {index} failed, exit={result.returncode}; see {log_path}')
                summary = read(output / 'summary.json')
                meta = read(output / 'run.meta.json')
                audit = read(output.with_name(output.name + '.audit.json'))
                assert audit['status'] == 'complete' and summary['same_process_reload']
                assert summary['updates'] == 1 and summary['cumulative_updates'] == index
                assert meta['ids'] == batch['tasks'] and meta['update_index'] == index
                assert all(math.isfinite(summary[k]) for k in
                           ['loss', 'gradient_norm', 'replay_ratio_min', 'replay_ratio_max'])
                if parent:
                    assert summary['parent_run_id'] == parent['run_id']
                    assert summary['actor_before'] == parent['actor_after']
                    assert summary['critic_before'] == parent['critic_after']
                else:
                    assert summary['parent_run_id'] is None
                    assert summary['actor_before'] == plan['initial_actor_tensor_digest']
                    assert summary['critic_before'] == plan['initial_critic_tensor_digest']
                chain = state['chains'][label]
                chain['completed_updates'] += 1
                chain['completed_episodes'] += len(batch['tasks'])
                chain['latest_checkpoint'] = str(output)
                for key in ('turns', 'environment_calls', 'generated_tokens'):
                    chain[key] += summary[key]
                caps = native['equal_budget_caps']
                assert chain['turns'] <= caps['policy_turn_cap']
                assert chain['environment_calls'] <= caps['environment_call_cap']
                assert chain['generated_tokens'] <= caps['generated_token_cap']
                state['completed_updates'] += 1
                state['completed_episodes'] += len(batch['tasks'])
                save()
                print('UPDATE_COMPLETE', label, index, 'episodes', chain['completed_episodes'], flush=True)
                previous, parent = output, summary
            assert state['chains'][label]['completed_episodes'] == plan['train_task_count']
            state['chains'][label]['status'] = 'complete'
            save()
        assert state['completed_updates'] == state['target_updates']
        assert state['completed_episodes'] == state['target_episodes']
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        state['last_status_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        print('TRAINING_STATUS', state['status'], flush=True)


if __name__ == '__main__':
    main()
