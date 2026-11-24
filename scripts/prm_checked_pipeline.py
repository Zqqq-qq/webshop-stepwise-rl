"""Fail closed between PRM sampling and downstream conversion/training.

Each command is an argv list executed without a shell. Plans use absolute paths.
The caller owns GPU locking. Existing attempts and sampling output are never reused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_coverage(ids_path, rollout_path, metadata_path):
    ids = json.loads(Path(ids_path).read_text(encoding='utf-8'))
    if (not isinstance(ids, list) or not ids or
            any(type(i) is not int for i in ids) or len(set(ids)) != len(ids)):
        raise ValueError('Expected task IDs must be unique integers')
    meta = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    if meta.get('selected_ids') != ids or meta.get('repeats') != 1 or not meta.get('run_id'):
        raise ValueError('Sampling metadata does not match the requested tasks/run')
    rows = [json.loads(line) for line in Path(rollout_path).read_text(encoding='utf-8').splitlines()
            if line.strip()]
    actual = [row.get('task_id') for row in rows]
    if any(type(i) is not int for i in actual) or actual != ids:
        raise ValueError(f'Sampling coverage incomplete or duplicated: {len(actual)}/{len(ids)} rows')
    for row in rows:
        if row.get('run_id') != meta['run_id'] or type(row.get('repeat')) is not int or row['repeat'] != 0:
            raise ValueError('Trajectory run/repeat identity mismatch')
        if (type(row.get('done')) is not bool or
                row.get('termination') not in ('purchase', 'max_steps') or
                row['done'] != (row['termination'] == 'purchase')):
            raise ValueError('Technical failure or inconsistent trajectory termination')
    return {'completed_tasks': len(rows), 'expected_tasks': len(ids),
            'run_id': meta['run_id'], 'ids_sha256': sha(ids_path),
            'rollout_sha256': sha(rollout_path), 'metadata_sha256': sha(metadata_path)}


def run(plan_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan_bytes = Path(plan_path).read_bytes()
    (output/'plan.json').write_bytes(plan_bytes)
    plan = json.loads(plan_bytes)
    state = {'status': 'running', 'completed_stages': [],
             'plan_sha256': hashlib.sha256(plan_bytes).hexdigest()}

    def save():
        temp = output/'status.tmp.json'
        temp.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
        temp.replace(output/'status.json')

    try:
        stages = plan['stages']
        if not stages or stages[0]['name'] != 'sample' or sum(s['name']=='sample' for s in stages) != 1:
            raise ValueError('Exactly one sampling stage must run first')
        for key in ('rollout', 'metadata'):
            if Path(plan[key]).exists():
                raise ValueError(f'Refusing existing sampling output: {plan[key]}')
        ids_hash = sha(plan['expected_ids'])
        for index, stage in enumerate(stages):
            state['stage'] = stage['name']
            save()
            command = stage['command']
            if not isinstance(command,list) or not command or not all(isinstance(v,str) and v for v in command):
                raise ValueError('Commands must be nonempty argv lists')
            # Recheck sampling inputs before every downstream command.
            if index:
                if verify_coverage(plan['expected_ids'],plan['rollout'],plan['metadata']) != state['coverage']:
                    raise ValueError('Sampling evidence changed between stages')
            with (output/f'{index:02d}.log').open('xb') as log:
                subprocess.run(command, cwd=plan['cwd'], check=True, stdout=log, stderr=subprocess.STDOUT)
            if sha(plan['expected_ids']) != ids_hash:
                raise ValueError('Requested task list changed during execution')
            if index == 0:
                state['coverage'] = verify_coverage(plan['expected_ids'],plan['rollout'],plan['metadata'])
            state['completed_stages'].append(stage['name'])
            save()
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        save()
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.output)
