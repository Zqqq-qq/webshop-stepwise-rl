"""Plan and collect same-result-page purchases from the archived fit501 train pool."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

from replay_prm55_expanded_pairs import check_pair, replay_arm
from replay_prm55_option_pair import ARCHIVE_SHA, request, require, write_json

ASIN = re.compile(r'[a-z0-9]{10}')
SCHEMA = 'prm55-product-pairs-v1'
TASK_COUNT = 32
BRANCH = 1  # zero-based, after search


def read_archive(path):
    require(hashlib.sha256(path.read_bytes()).hexdigest() == ARCHIVE_SHA, 'Wrong #55 evidence archive')
    with ZipFile(path) as archive:
        fit = [json.loads(line) for line in archive.read('data/fit501.training.jsonl').splitlines()]
        dev = [json.loads(line) for line in archive.read('data/dev160/dev160.training.jsonl').splitlines()]
        rollout_dev = [json.loads(line) for line in archive.read('rollout/dev160.jsonl').splitlines()]
    require(len(fit) == len({row['task_id'] for row in fit}) == 501, 'Wrong fit501 coverage')
    require(not ({row['task_id'] for row in fit} & {row['task_id'] for row in rollout_dev})
            and not ({row['instruction_sha256'] for row in fit} & {row['instruction_sha256'] for row in dev}),
            'Training overlap with examined dev')
    fingerprints = {row['goal_fingerprint'] for row in rollout_dev}
    require(len(fingerprints) == 1, 'Mixed archived environments')
    return fit, fingerprints.pop()


def action_from(message):
    response = message['content'].strip()
    action = response.splitlines()[-1]
    require(action.startswith('Action: '), 'Missing canonical action line')
    return action[len('Action: '):].lower()


def make_plan(fit, fingerprint):
    eligible = []
    for row in fit:
        messages = row['messages']
        if row['terminal_reward'] != 1 or row['termination'] != 'purchase' or len(messages) != 7:
            continue
        steps = [
            {'observation': page['content'], 'response': reply['content'], 'action': action_from(reply)}
            for page, reply in zip(messages[1::2], messages[2::2])
        ]
        search, selected, purchase = (step['action'] for step in steps)
        if not search.startswith('search[') or not purchase == 'click[buy now]' or not selected.startswith('click['):
            continue
        cells = [cell.strip().lower() for cell in steps[BRANCH]['observation'].split('[SEP]')]
        if not any(cell.startswith('page 1 (total results:') for cell in cells):
            continue
        products = list(dict.fromkeys(cell for cell in cells if ASIN.fullmatch(cell)))
        target = selected[6:-1]
        if target not in products or len(products) < 3:
            continue
        alternatives = [product for product in reversed(products) if product != target][:2]
        eligible.append({
            'task_id': row['task_id'], 'instruction_sha256': row['instruction_sha256'],
            'search': search, 'reference_product': target, 'alternative_products': alternatives,
            'result_page_sha256': hashlib.sha256(steps[BRANCH]['observation'].encode()).hexdigest(),
        })
    eligible.sort(key=lambda row: hashlib.sha256(f"prm-fork-v1:{row['task_id']}".encode()).digest())
    return {
        'schema_version': SCHEMA, 'archive_sha256': ARCHIVE_SHA, 'goal_fingerprint': fingerprint,
        'source_split': 'fit501', 'eligible_tasks': len(eligible), 'tasks': eligible[:TASK_COUNT],
        'test_authorization': False,
    }


def collect(server, fit, plan, output, limit):
    server = server.rstrip('/')
    address = urlsplit(server)
    require(address.scheme == 'http' and address.hostname == '127.0.0.1' and address.port
            and not (address.path or address.query or address.fragment or address.username),
            'Use the original local WebShop bridge')
    health = request(server, '/health')
    require(health['ready'] and health['alignment_mismatches'] == 0 and health['split_status']['verified']
            and health['goal_fingerprint'] == plan['goal_fingerprint'], 'Wrong or unhealthy WebShop environment')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)
    write_json(output / 'health.json', health)
    rows = {row['task_id']: row for row in fit}
    selected = 0
    with (output / 'training-pairs.jsonl').open('x', encoding='utf-8') as pairs, (
            output / 'collection.jsonl').open('x', encoding='utf-8') as audit:
        for case in plan['tasks'][:limit]:
            task_id = case['task_id']
            messages = rows[task_id]['messages']
            system = messages[0]
            instruction = system['content'].split('\nShopping instruction: ', 1)[1]
            old = {'task_id': task_id, 'instruction': instruction, 'goal_fingerprint': plan['goal_fingerprint'],
                   'steps': [{'observation': page['content'], 'response': reply['content'],
                              'action': action_from(reply)}
                             for page, reply in zip(messages[1::2], messages[2::2])]}
            probes = []
            target = None
            try:
                target = replay_arm(server, old, BRANCH, case['reference_product'], 'product')
                require(target['final_reward'] == 1, 'Archived full-reward reference did not reproduce')
                for product in case['alternative_products']:
                    candidate = replay_arm(server, old, BRANCH, product, 'product')
                    check_pair(target, candidate, BRANCH)
                    probes.append(candidate)
                    if candidate['final_reward'] < target['final_reward']:
                        row = {
                            'schema_version': SCHEMA, 'task_id': task_id, 'branch': BRANCH,
                            'target': target, 'alternative': candidate,
                        }
                        pairs.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                        pairs.flush()
                        selected += 1
                        break
                evidence = {'task_id': task_id, 'target': target, 'probes': probes}
            except ValueError as error:
                evidence = {'task_id': task_id, 'status': 'rejected', 'reason': str(error),
                            'target': target, 'probes': probes}
            audit.write(json.dumps(evidence, ensure_ascii=False, allow_nan=False) + '\n')
            audit.flush()
    write_json(output / 'summary.json', {'planned_tasks': limit, 'usable_pairs': selected,
                                        'test_authorization': False})
    print(f'Recorded {selected} valid pairs from at most {limit} candidates', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'collect'))
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--server', default='http://127.0.0.1:8766')
    parser.add_argument('--max-tasks', type=int)
    args = parser.parse_args()
    fit, fingerprint = read_archive(args.archive)
    plan = make_plan(fit, fingerprint)
    if args.mode == 'plan':
        with args.plan.open('x', encoding='utf-8') as stream:
            json.dump(plan, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    else:
        require(args.out is not None and json.loads(args.plan.read_text()) == plan,
                'Use the exact frozen plan and a new --out directory')
        limit = len(plan['tasks']) if args.max_tasks is None else args.max_tasks
        require(0 < limit <= len(plan['tasks']), 'Invalid number of tasks')
        collect(args.server, fit, plan, args.out, limit)


if __name__ == '__main__':
    main()
