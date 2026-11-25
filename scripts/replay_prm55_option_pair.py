"""Replay task 7084 with B and A eyeshadow options and compare frozen PRM scores."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from zipfile import ZipFile


ARCHIVE_SHA = '0834998fe986f450f40721f0864b51351c68743d5b2e5979334cffde28ca44b4'
TASK_ID = 7084
OPTIONS = ('b', 'a')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_archive(path):
    require(hashlib.sha256(path.read_bytes()).hexdigest() == ARCHIVE_SHA, 'Wrong PR #55 evidence ZIP')
    with ZipFile(path) as archive:
        episodes = [json.loads(line) for line in archive.read('rollout/dev160.jsonl').splitlines()]
        (episode,) = [row for row in episodes if row['task_id'] == TASK_ID]
        require(episode['success'] and len(episode['steps']) == 4, 'Unexpected archived task')
        require([s['action'] for s in episode['steps'][2:]] == ['click[b]', 'click[buy now]'],
                'Unexpected archived suffix')
        return episode, json.loads(archive.read('training/stage126/prm-manifest.json'))


def request(server, endpoint, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = Request(server + endpoint, data=data, headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=180) as response:
        return json.load(response)


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def replay_arm(server, old, option):
    state = request(server, '/reset', {'task_id': TASK_ID})
    require(state['instruction'] == old['instruction']
            and state['goal_fingerprint'] == old['goal_fingerprint'], 'Reset identity mismatch')
    actions = [s['action'] for s in old['steps'][:2]] + [f'click[{option}]', 'click[buy now]']
    responses = [s['response'] for s in old['steps'][:2]] + [f'Action: {a}' for a in actions[2:]]
    steps = []
    for index, (action, response) in enumerate(zip(actions, responses)):
        if index < 3:
            require(state['observation'] == old['steps'][index]['observation'],
                    f'Arm {option} differs from archived environment at step {index + 1}')
        if index == 2:
            require(set(OPTIONS) <= set(state['available']['clickables']), 'Option is unavailable')
        if index == 3:
            selection = state['available'].get('selected_options')
            if selection is not None:
                require(option in selection.values(), f'Environment did not select {option}')
        prior = state
        state = request(server, '/step', {'action': action})
        require(state['done'] is (index == 3), f'Unexpected terminal step in arm {option}')
        reward = state['reward']
        require(type(reward) in (float, int) and 0 <= reward <= 1, 'Invalid environment reward')
        steps.append({'step': index, 'observation': prior['observation'], 'available': prior['available'],
                      'response': response, 'action': action, 'grounding': 1, 'error': None,
                      'environment_reward': reward})
    return {'schema_version': old['schema_version'], 'task_id': TASK_ID, 'repeat': 0,
            'instruction': old['instruction'], 'goal_fingerprint': old['goal_fingerprint'],
            'done': True, 'termination': 'purchase', 'final_reward': reward,
            'success': reward >= 1 - 1e-6, 'steps': steps}


def check_pair(arms):
    left, right = (arms[option] for option in OPTIONS)
    for key in ('task_id', 'instruction', 'goal_fingerprint'):
        require(left[key] == right[key], f'Unequal pair field {key}')
    require(left['steps'][:2] == right['steps'][:2], 'Shared prefix differs before selection')
    for option, arm in arms.items():
        require(len(arm['steps']) == 4 and arm['done'] is True, 'Expected complete four-step purchase')
        require(arm['steps'][2]['action'] == f'click[{option}]', 'Selection is inconsistent')
    for index in (2, 3):
        field = 'observation' if index == 2 else 'response'
        require(left['steps'][index][field] == right['steps'][index][field], 'Pair changed a controlled field')


def collect(args, archived):
    server = args.server.rstrip('/')
    address = urlsplit(server)
    require(address.scheme == 'http' and address.hostname == '127.0.0.1' and address.port
            and not (address.path or address.query or address.fragment or address.username),
            'Use the local WebShop bridge')
    health = request(server, '/health')
    require(health['ready'] and health['goal_fingerprint'] == archived['goal_fingerprint']
            and health['alignment_mismatches'] == 0 and health['split_status']['verified'],
            'Wrong or unhealthy WebShop environment')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    arms = {}
    for option in OPTIONS:
        arms[option] = replay_arm(server, archived, option)
        write_json(args.run_dir / f'{option}.rollout.jsonl', arms[option])
    check_pair(arms)
    write_json(args.run_dir / 'environment.json', {
        'task_id': TASK_ID, 'health': health, 'source_archive_sha256': ARCHIVE_SHA,
        'terminal_scores': {k: arm['final_reward'] for k, arm in arms.items()},
        'test_authorization': False,
    })


def compare(args, manifest):
    arms = {option: json.loads((args.run_dir / f'{option}.rollout.jsonl').read_text()) for option in OPTIONS}
    check_pair(arms)
    summary = {}
    for option in OPTIONS:
        result = json.loads((args.run_dir / f'{option}.predictions.json').read_text())
        require(result['identity'] == manifest['identity'] and result['input_schema'] == 'rollout'
                and result['total_samples'] == 1 and not result['rejected']
                and result['test_authorization'] is False, 'Scorer identity or coverage mismatch')
        (row,) = result['predictions']
        scores = row['step_scores']
        require(row['task_id'] == TASK_ID and len(scores) == len(row['turn_end_indices']) == 4
                and all(math.isfinite(x) for x in scores)
                and math.isclose(math.fsum(scores), row['total'], abs_tol=1e-6),
                'Invalid scoring result')
        summary[option] = {'environment_terminal_score': arms[option]['final_reward'],
                           'scores': scores, 'turn_end_indices': row['turn_end_indices'],
                           'prm_total': row['total']}
    delta = [b - a for b, a in zip(summary['b']['scores'], summary['a']['scores'])]
    env_delta = summary['b']['environment_terminal_score'] - summary['a']['environment_terminal_score']
    require(summary['b']['turn_end_indices'][:2] == summary['a']['turn_end_indices'][:2],
            'Shared scoring indices differ')
    report = {'arms': summary, 'environment_b_minus_a': env_delta,
              'valid_environment_contrast': env_delta > 0,
              'prm_b_minus_a_by_step': delta,
              'prm_b_minus_a_total': summary['b']['prm_total'] - summary['a']['prm_total'],
              'shared_prefix_within_1e_5': all(abs(x) <= 1e-5 for x in delta[:2]),
              'test_authorization': False}
    write_json(args.run_dir / 'pair-comparison.json', report)


def verify_checkpoint(checkpoint, manifest):
    local = json.loads((checkpoint / 'prm-manifest.json').read_text())
    require(local == manifest, 'Checkpoint manifest is not the archived stage126')
    for name in ('head.pt', 'adapter/adapter_model.safetensors', 'adapter/adapter_config.json'):
        digest = hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
        require(digest == manifest['files_sha256'][name], 'Stage126 payload mismatch: ' + name)


def score(args, manifest):
    require(args.config is not None and args.checkpoint is not None,
            'Scoring requires the stage126 config and checkpoint')
    checkpoint = args.checkpoint.resolve()
    verify_checkpoint(checkpoint, manifest)
    for option in OPTIONS:
        subprocess.run([
            sys.executable, '-m', 'webshop_lab.prm.infer', '--device', 'cuda',
            '--source-commit', manifest['identity']['source_commit'],
            '--config', str(args.config.resolve()), '--checkpoint', str(checkpoint),
            '--rollout', str(args.run_dir.resolve() / f'{option}.rollout.jsonl'),
            '--output', str(args.run_dir.resolve() / f'{option}.predictions.json'),
        ], check=True)
    compare(args, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('replay', 'score'))
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--server', default='http://127.0.0.1:8766')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    episode, manifest = read_archive(args.archive)
    if args.mode == 'replay':
        collect(args, episode)
    else:
        score(args, manifest)


if __name__ == '__main__':
    main()
