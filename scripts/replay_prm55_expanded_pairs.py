"""Replay three option pairs and one product pair on the frozen #55 dev environment."""

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

from replay_prm55_option_pair import (
    ARCHIVE_SHA, read_system_prompts, request, require, scoring_sample, verify_checkpoint, write_json,
)


OPTION_CASES = {
    # Task: (zero-based branch step, archived option, counterfactual option)
    7402: (2, 'dusty pink', 'black'),
    7113: (3, '3x-large', 'large'),
    7489: (2, 'yubikey 5c', 'yubikey 5c nano'),
}
PRODUCT_TASK = 7544
# All running-shoe alternatives on the archived result page, in page order.
RUNNING_SHOES = (
    'b09kgmgvv9', 'b097db499g', 'b0957cd5w8', 'b088tcl16p', 'b09dct8kh6',
    'b0833kzbnb', 'b09pzxkc7p', 'b09ntfzj45', 'b088tyck9m',
)
ARMS = ('target', 'alternative')
SCORE_TOLERANCE = 1e-5


def archive_inputs(path):
    require(hashlib.sha256(path.read_bytes()).hexdigest() == ARCHIVE_SHA, 'Wrong #55 evidence archive')
    task_ids = set(OPTION_CASES) | {PRODUCT_TASK}
    with ZipFile(path) as archive:
        rows = [json.loads(line) for line in archive.read('rollout/dev160.jsonl').splitlines()]
        selected = [row for row in rows if row['task_id'] in task_ids]
        require(len(selected) == len({r['task_id'] for r in selected}) == len(task_ids),
                'Wrong task coverage in raw archive')
        return ({row['task_id']: row for row in selected},
                read_system_prompts(archive, task_ids),
                json.loads(archive.read('training/stage126/prm-manifest.json')))


def replay_arm(server, old, branch, choice, continuation):
    require(continuation in ('buy', 'size8', 'product'), 'Unknown replay continuation')
    state = request(server, '/reset', {'task_id': old['task_id']})
    require(state['instruction'] == old['instruction']
            and state['goal_fingerprint'] == old['goal_fingerprint'], 'Reset identity mismatch')
    steps = []

    def advance(action, response, terminal=False):
        nonlocal state
        clickables = state['available']['clickables']
        if action.startswith('click['):
            require(action[6:-1] in clickables, f'Unavailable action {action}')
        else:
            require(state['available']['has_search_bar'], 'Search is unavailable')
        after = request(server, '/step', {'action': action})
        reward = after['reward']
        require(type(reward) in (float, int) and 0 <= reward <= 1, 'Invalid environment reward')
        require(after['done'] is terminal, f'Unexpected end state after {action}')
        steps.append({'step': len(steps), 'observation': state['observation'],
                      'available': state['available'], 'action': action, 'response': response,
                      'environment_reward': reward})
        state = after

    for index in range(branch + 1):
        archived = old['steps'][index]
        require(state['observation'] == archived['observation']
                and ('available' not in archived
                     or state['available']['clickables'] == archived['available']['clickables']),
                f"Task {old['task_id']} differs from archived page at turn {index + 1}")
        if index < branch:
            advance(archived['action'], archived['response'])
    advance(f'click[{choice}]', f'Action: click[{choice}]')
    if continuation == 'size8':
        size = next((value for value in state['available']['clickables']
                     if re.fullmatch(r'8(?:\.0)?(?: (?:us|women|wide|wide women \| 8 wide men))?',
                                     value)), None)
        if size is not None:
            advance(f'click[{size}]', f'Action: click[{size}]')
            selection = state['available'].get('selected_options')
            require(selection is None or size in selection.values(), f'Size click did not select {size}')
    elif continuation == 'buy':
        selection = state['available'].get('selected_options')
        require(selection is None or choice in selection.values(), f'Option click did not select {choice}')
    advance('click[buy now]', 'Action: click[buy now]', terminal=True)
    return {'task_id': old['task_id'], 'instruction': old['instruction'],
            'goal_fingerprint': old['goal_fingerprint'], 'final_reward': state['reward'],
            'done': True, 'termination': 'purchase', 'steps': steps}


def check_pair(target, alternative, branch):
    for field in ('task_id', 'instruction', 'goal_fingerprint'):
        require(target[field] == alternative[field], f'Different {field} in pair')
    left, right = target['steps'], alternative['steps']
    require(left[:branch] == right[:branch], 'Shared trajectory prefix differs')
    require(left[branch]['observation'] == right[branch]['observation']
            and left[branch]['available'] == right[branch]['available']
            and left[branch]['action'] != right[branch]['action'], 'Invalid decision pair')
    require(left[-1]['action'] == right[-1]['action'] == 'click[buy now]', 'Missing purchase')
    require(target['done'] is True and alternative['done'] is True, 'Missing terminal state')


def collect(args, episodes, prompts):
    server = args.server.rstrip('/')
    address = urlsplit(server)
    require(address.scheme == 'http' and address.hostname == '127.0.0.1' and address.port
            and not (address.path or address.query or address.fragment or address.username),
            'Use the original local WebShop bridge')
    health = request(server, '/health')
    require(health['ready'] and health['alignment_mismatches'] == 0
            and health['split_status']['verified']
            and all(row['goal_fingerprint'] == health['goal_fingerprint'] for row in episodes.values()),
            'Wrong or unhealthy WebShop environment')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    cases = {}
    for task_id, (branch, requested, other) in OPTION_CASES.items():
        old = episodes[task_id]
        require(old['steps'][branch]['action'] == f'click[{requested}]'
                and [s['action'] for s in old['steps'][branch + 1:]] == ['click[buy now]'],
                f'Archived suffix changed for {task_id}')
        target, alternative = (replay_arm(server, old, branch, option, 'buy')
                               for option in (requested, other))
        require(target['final_reward'] == old['final_reward'], f'Archived reward differs for {task_id}')
        cases[task_id] = (branch, target, alternative)

    old = episodes[PRODUCT_TASK]
    wrong = replay_arm(server, old, 1, 'b092f97bw4', 'size8')
    require(wrong['final_reward'] == old['final_reward']
            and [s['action'] for s in wrong['steps']] == [s['action'] for s in old['steps']],
            'Archived wrong-product path or score differs')
    probes = [replay_arm(server, old, 1, asin, 'size8') for asin in RUNNING_SHOES]
    best = max(probes, key=lambda arm: arm['final_reward'])
    cases[PRODUCT_TASK] = (1, best, wrong)
    write_json(args.run_dir / 'product-candidates.json', probes)

    pairs = {}
    for task_id, (branch, target, alternative) in cases.items():
        check_pair(target, alternative, branch)
        pairs[task_id] = {'branch_step': branch + 1}
        for name, arm in zip(ARMS, (target, alternative)):
            write_json(args.run_dir / f'{task_id}.{name}.rollout.jsonl', arm)
            pairs[task_id][name] = {
                'branch_action': arm['steps'][branch]['action'],
                'continuation': [s['action'] for s in arm['steps'][branch + 1:]],
                'environment_terminal_score': arm['final_reward'],
            }

    for name in ARMS:
        with (args.run_dir / f'{name}.samples.jsonl').open('x', encoding='utf-8') as stream:
            for task_id in cases:
                row = json.loads((args.run_dir / f'{task_id}.{name}.rollout.jsonl').read_text())
                # Environment labels stay in their own artifact.
                sample = scoring_sample(row, prompts[task_id])
                stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + '\n')
    write_json(args.run_dir / 'environment.json', {
        'health': health, 'source_archive_sha256': ARCHIVE_SHA,
        'pairs': pairs, 'test_authorization': False,
    })


def compare(run_dir, manifest, device):
    env = json.loads((run_dir / 'environment.json').read_text())
    require(env['source_archive_sha256'] == ARCHIVE_SHA and env['test_authorization'] is False
            and {int(key) for key in env['pairs']} == set(OPTION_CASES) | {PRODUCT_TASK},
            'Wrong environment evidence')
    predictions = {}
    for name in ARMS:
        output = json.loads((run_dir / f'{name}.predictions.json').read_text())
        require(output['identity'] == manifest['identity'] and output['input_schema'] == 'samples'
                and output['device'] == device and output['total_samples'] == len(env['pairs'])
                and not output['rejected'] and output['test_authorization'] is False,
                'Scorer identity or coverage mismatch')
        predictions[name] = {str(row['task_id']): row for row in output['predictions']}
        require(len(output['predictions']) == len(env['pairs'])
                and predictions[name].keys() == env['pairs'].keys(), 'Missing or duplicate prediction')

    for task_id, pair in env['pairs'].items():
        branch = pair['branch_step'] - 1
        rolls = [json.loads((run_dir / f'{task_id}.{name}.rollout.jsonl').read_text()) for name in ARMS]
        check_pair(*rolls, branch)
        for name, roll in zip(ARMS, rolls):
            row = predictions[name][task_id]
            require(roll['task_id'] == int(task_id)
                    and roll['final_reward'] == pair[name]['environment_terminal_score']
                    and roll['steps'][branch]['action'] == pair[name]['branch_action']
                    and len(row['step_scores']) == len(row['turn_end_indices']) == len(roll['steps'])
                    and all(math.isfinite(x) for x in row['step_scores'])
                    and math.isclose(math.fsum(row['step_scores']), row['total'], abs_tol=1e-6),
                    f'Invalid scoring result for {task_id}/{name}')
            pair[name].update(row)

        left, right = (pair[name] for name in ARMS)
        env_delta = left['environment_terminal_score'] - right['environment_terminal_score']
        branch_delta = left['step_scores'][branch] - right['step_scores'][branch]
        require(left['turn_end_indices'][:branch] == right['turn_end_indices'][:branch],
                f'Shared prefix token positions differ for {task_id}')
        prefix_delta = [x-y for x, y in zip(left['step_scores'][:branch], right['step_scores'][:branch])]
        require(all(abs(delta) <= SCORE_TOLERANCE for delta in prefix_delta),
                f'Shared prefix scores differ for {task_id}')
        same_env = math.isclose(env_delta, 0, abs_tol=1e-9)
        pair.update(environment_delta=env_delta,
                    environment_order='tied' if same_env else ('target_higher' if env_delta > 0 else 'alternative_higher'),
                    branch_delta=branch_delta,
                    branch_agrees_with_environment=None if same_env else
                    (abs(branch_delta) > SCORE_TOLERANCE and branch_delta * env_delta > 0),
                    purchase_delta=left['step_scores'][-1] - right['step_scores'][-1],
                    total_delta=left['total'] - right['total'], shared_prefix_deltas=prefix_delta)
    env['scorer_device'] = device
    env['scorer_identity'] = manifest['identity']
    write_json(run_dir / 'pair-comparison.json', env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('replay', 'score', 'compare'))
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--server', default='http://127.0.0.1:8766')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    episodes, prompts, manifest = archive_inputs(args.archive)
    if args.mode == 'replay':
        collect(args, episodes, prompts)
        return
    if args.mode == 'score':
        require(args.config is not None and args.checkpoint is not None,
                'Scoring needs the frozen config and checkpoint')
        verify_checkpoint(args.checkpoint, manifest)
        for name in ARMS:
            subprocess.run([
                sys.executable, '-m', 'webshop_lab.prm.infer', '--device', args.device,
                '--source-commit', manifest['identity']['source_commit'],
                '--config', str(args.config.resolve()), '--checkpoint', str(args.checkpoint.resolve()),
                '--samples', str((args.run_dir / f'{name}.samples.jsonl').resolve()),
                '--output', str((args.run_dir / f'{name}.predictions.json').resolve()),
            ], check=True)
    compare(args.run_dir, manifest, args.device)


if __name__ == '__main__':
    main()
