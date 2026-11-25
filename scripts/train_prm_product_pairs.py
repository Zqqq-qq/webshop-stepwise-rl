"""Run a paired terminal-MSE or terminal-MSE + fork ranking arm on frozen #55 code."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

import torch
from torch.nn import functional as F

from collect_prm_product_pairs import (
    ASIN, BRANCH, SCHEMA, action_from, make_plan, read_archive,
)
from replay_prm55_expanded_pairs import check_pair
from replay_prm55_option_pair import require, scoring_sample, verify_checkpoint
from webshop_lab.prm.config import load_config
from webshop_lab.prm.model_core import save_prm
from webshop_lab.prm.runtime import (
    build_prm, finalize_manifest, sha, tensor_batch, verify_runtime_inputs, write_json,
)
from webshop_lab.prm.tokenize import TrajectoryTooLong, tokenize_trajectory
from webshop_lab.prm.train import attempt_record, predict


def load_protocol(path):
    protocol = json.loads(path.read_text())
    for name in ('updates', 'pairs_per_update'):
        require(type(protocol[name]) is int and protocol[name] > 0, f'Invalid {name}')
    require(type(protocol['seed']) is int and 0 <= protocol['seed'] < 2**32, 'Invalid seed')
    for name in ('lora_learning_rate', 'head_learning_rate', 'weight_decay'):
        value = protocol[name]
        require(type(value) in (int, float) and math.isfinite(value)
                and (value >= 0 if name == 'weight_decay' else value > 0), f'Invalid {name}')
    weights = protocol['rank_weight']
    require(set(weights) == {'terminal', 'ranking'} and weights['terminal'] == 0
            and type(weights['ranking']) in (int, float)
            and math.isfinite(weights['ranking']) and weights['ranking'] > 0, 'Invalid rank weights')
    return protocol


def load_pair_samples(path, plan, fit):
    rows = {row['task_id']: row for row in fit}
    cases = {case['task_id']: case for case in plan['tasks']}
    pairs = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row['task_id']
        require(row['schema_version'] == SCHEMA and row['branch'] == BRANCH
                and type(task_id) is int and task_id in cases and task_id not in pairs,
                'Invalid or duplicate training pair')
        high, low = row['target'], row['alternative']
        case = cases[task_id]
        original = rows[task_id]['messages']
        instruction = original[0]['content'].split('\nShopping instruction: ', 1)[1]
        check_pair(high, low, BRANCH)
        samples = []
        for episode in (high, low):
            steps = episode['steps']
            reward = episode['final_reward']
            require(episode['task_id'] == task_id and episode['goal_fingerprint'] == plan['goal_fingerprint']
                    and episode['instruction'] == instruction and episode['termination'] == 'purchase'
                    and len(steps) == 3 and type(reward) in (int, float) and 0 <= reward <= 1
                    and reward == steps[-1]['environment_reward'], 'Inconsistent recorded purchase')
            require(steps[0]['observation'] == original[1]['content']
                    and steps[0]['response'] == original[2]['content']
                    and steps[1]['observation'] == original[3]['content']
                    and steps[0]['action'] == case['search'], 'Changed archived search prefix')
            action = steps[BRANCH]['action']
            require(action == action_from({'content': steps[BRANCH]['response']})
                    and steps[-1]['response'] == 'Action: click[buy now]'
                    and action[6:-1] in steps[BRANCH]['available']['clickables']
                    and action.startswith('click[') and action.endswith(']')
                    and ASIN.fullmatch(action[6:-1]), 'Invalid fork action')
            sample = scoring_sample(episode, original[0])
            sample['terminal_reward'] = reward
            samples.append(sample)
        require(high['final_reward'] == 1 and high['final_reward'] > low['final_reward']
                and high['steps'][BRANCH]['action'] == f"click[{case['reference_product']}]"
                and low['steps'][BRANCH]['action'][6:-1] in case['alternative_products'],
                'Only verified, unequal and planned purchases may train the rank term')
        pairs[task_id] = tuple(samples)
    require(pairs, 'No usable training pairs')
    return [(task_id, pairs[task_id]) for task_id in cases if task_id in pairs]


def encode_pairs(tokenizer, max_length, pairs):
    retained, rejected = [], []
    for task_id, samples in pairs:
        try:
            encoded = [tokenize_trajectory(tokenizer, s['messages'], max_length) for s in samples]
        except TrajectoryTooLong as error:
            rejected.append({'task_id': task_id, 'reason': str(error)})
            continue
        prefixes = [row['input_ids'][:row['assistant_spans'][BRANCH][0]] for row in encoded]
        require(prefixes[0] == prefixes[1], f'Changed pre-fork token prefix for {task_id}')
        for row, sample in zip(encoded, samples):
            row['terminal_reward'] = sample['terminal_reward']
        retained.append((task_id, encoded))
    require(retained, 'All pairs exceed the frozen context limit')
    return retained, rejected


def optimize_update(model, optimizer, pairs, pad_token_id, device, rank_weight):
    """Each microbatch contains a verified pair and accumulates one loss per pair."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    records = []
    for task_id, encoded in pairs:
        outputs = [model(**tensor_batch([row], pad_token_id, device, True)) for row in encoded]
        terminal_mse = (outputs[0]['loss'] + outputs[1]['loss']) / 2
        gap = outputs[0]['step_scores'][0, BRANCH] - outputs[1]['step_scores'][0, BRANCH]
        ranking_loss = F.softplus(-gap)
        loss = terminal_mse + rank_weight * ranking_loss
        require(bool(torch.isfinite(loss)), f'Nonfinite pair loss for {task_id}')
        (loss / len(pairs)).backward()
        records.append({'task_id': task_id, 'terminal_mse': float(terminal_mse.detach()),
                        'fork_gap': float(gap.detach()), 'rank_loss': float(ranking_loss.detach()),
                        'loss': float(loss.detach())})
    trainable = [p for p in model.parameters() if p.requires_grad]
    require(any(p.grad is not None for p in trainable), 'Missing training gradients')
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {'pairs': records, 'gradient_norm': float(norm)}


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(args):
    script_dir = Path(__file__).resolve().parent
    source_paths = [script_dir / name for name in (
        'train_prm_product_pairs.py', 'collect_prm_product_pairs.py',
        'replay_prm55_expanded_pairs.py', 'replay_prm55_option_pair.py',
    )]
    input_paths = [args.pairs, args.plan, args.protocol, args.config, *source_paths]
    input_hashes = {str(path): sha(path) for path in input_paths}
    protocol = load_protocol(args.protocol)
    fit, fingerprint = read_archive(args.archive)
    plan = make_plan(fit, fingerprint)
    require(json.loads(args.plan.read_text()) == plan, 'Wrong frozen train plan')
    raw = load_pair_samples(args.pairs, plan, fit)
    with ZipFile(args.archive) as archive:
        manifest = json.loads(archive.read('training/stage126/prm-manifest.json'))
    verify_checkpoint(args.checkpoint, manifest)
    config = load_config(args.config)
    device = args.device
    torch.manual_seed(protocol['seed'])
    if device == 'cuda':
        torch.cuda.manual_seed_all(protocol['seed'])
        torch.cuda.reset_peak_memory_stats()
    model, tokenizer, identity, runtime_inputs = build_prm(config, args.checkpoint, device, True)
    require(identity == manifest['identity'], 'Scoring runtime differs from frozen stage126')
    pairs, rejected = encode_pairs(tokenizer, config.max_length, raw)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    head = [p for name, p in trainable if name.startswith('score_head.')]
    lora = [p for name, p in trainable if 'lora_' in name]
    require(head and lora and len(head) + len(lora) == len(trainable), 'Unexpected trainable parameters')
    optimizer = torch.optim.AdamW([
        {'params': head, 'lr': protocol['head_learning_rate']},
        {'params': lora, 'lr': protocol['lora_learning_rate']},
    ], weight_decay=protocol['weight_decay'])
    rank_weight = protocol['rank_weight'][args.arm]
    updates = 1 if args.smoke else protocol['updates']
    per_update = protocol['pairs_per_update']
    before = parameter_hash(model)
    records = []
    for step in range(updates):
        batch = [pairs[(step * per_update + i) % len(pairs)] for i in range(per_update)]
        record = optimize_update(model, optimizer, batch, tokenizer.pad_token_id, device, rank_weight)
        record['update'] = step + 1
        records.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
    after = parameter_hash(model)
    require(after != before, 'Trainable parameters did not change')
    arms = ('target', 'alternative')
    references = {
        name: predict(model, tokenizer, [(task_id, rows[index]) for task_id, rows in pairs])
        for index, name in enumerate(arms)
    }
    require(verify_runtime_inputs(config) == runtime_inputs
            and {str(path): sha(path) for path in input_paths} == input_hashes,
            'Inputs changed during training')
    save_prm(model, args.out, identity)
    write_json(args.out / 'config.json', config.to_dict())
    retained_ids = [task_id for task_id, _ in pairs]
    retained_set = set(retained_ids)
    for index, name in enumerate(arms):
        write_json(args.out / f'{name}.reference.json', references[name])
        with (args.out / f'{name}.samples.jsonl').open('x', encoding='utf-8') as stream:
            for task_id, samples in raw:
                if task_id in retained_set:
                    sample = {key: value for key, value in samples[index].items() if key != 'terminal_reward'}
                    stream.write(json.dumps(sample, ensure_ascii=False) + '\n')
    write_json(args.out / 'summary.json', {
        'arm': args.arm, 'rank_weight': rank_weight, 'protocol': protocol, 'device': device,
        'smoke': args.smoke, 'parent_checkpoint_sha256': sha(args.checkpoint / 'prm-manifest.json'),
        'optimizer_initialization': 'fresh_adamw_both_arms',
        'before_parameter_sha256': before, 'after_parameter_sha256': after,
        'input_files_sha256': input_hashes, 'retained_task_ids': retained_ids,
        'rejected_pairs': rejected, 'updates': updates, 'pair_presentations': updates * per_update,
        'records': records, 'test_authorization': False,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3 if device == 'cuda' else None,
    })
    finalize_manifest(args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'plan', 'pairs', 'protocol', 'config', 'checkpoint', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=('terminal', 'ranking'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    require(not args.out.exists() and args.checkpoint.resolve() not in args.out.resolve().parents,
            'Use a fresh output directory outside the frozen checkpoint')
    with attempt_record(args.out, args.device):
        run(args)


if __name__ == '__main__':
    main()
