"""Verify the published 8-sample PRM supplement and run the CPU head comparison."""

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import platform
import struct
import zipfile

from audit_prm_representation import fit_fixed_representation, sample_key, sha256


MEMORY_DIGEST = 'b711fc49d9934adc321236a8837932406b6d1f3c8b81f8952d1f2c4a382453c8'
SUPPLEMENT_DIGEST = '2d1c2763783759adb47bf8eeb056cc36b7134cd030a2f6b274ea32df68a86aa4'
LEARNING_RATES = (1e-7, 3e-7, 1e-6)
MAX_UPDATES = 512
CHECK_INTERVAL = 64


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def read_archive(path, expected_digest, prefix):
    require(sha256(path) == expected_digest, f'Archive digest mismatch: {path.name}')
    with zipfile.ZipFile(path) as archive:
        manifest = archive.read(prefix + '/SHA256SUMS').decode()
        payloads = {}
        for line in manifest.splitlines():
            expected, name = line.split(maxsplit=1)
            name = name.lstrip('*')
            payload = archive.read(f'{prefix}/{name}')
            require(name not in payloads and digest(payload) == expected,
                    f'Duplicate or mismatched manifest entry: {name}')
            payloads[name] = payload
    return payloads


def audit_inputs(memory, supplement):
    import numpy as np
    import torch

    sample_bytes = memory['memory-8/samples.jsonl']
    samples = [json.loads(line) for line in sample_bytes.splitlines() if line.strip()]
    keys = [sample_key(row) for row in samples]
    require(len(keys) == len(set(keys)) == 8, 'Expected eight unique samples')
    labels = np.array([row['terminal_reward'] for row in samples], dtype=np.float64)
    require(np.isfinite(labels).all() and ((0 <= labels) & (labels <= 1)).all(),
            'Invalid terminal labels')
    plan = json.loads(memory['memory-8/plan.json'])
    require(len(plan['labels']) == len(keys) and all(
        plan['labels'][str(task)] == label for (task, _), label in zip(keys, labels)
    ), 'Labels differ from the frozen eight-sample plan')
    schedule = []
    for end in range(CHECK_INTERVAL, MAX_UPDATES + 1, CHECK_INTERVAL):
        summary = json.loads(memory[f'memory-8/step-{end}/summary.json'])
        require(summary['optimizer_steps'] == end and len(summary['records']) == CHECK_INTERVAL,
                'Historical update block differs from plan')
        schedule.extend(summary['records'])
    require([r['optimizer_step'] for r in schedule] == list(range(1, MAX_UPDATES + 1))
            and [r['task_id'] for r in schedule] == [task for task, _ in keys] * 64,
            'Historical episode order or update sequence differs')
    require(all(r['terminal_reward'] == labels[i % len(keys)] for i, r in enumerate(schedule)),
            'Historical training labels differ')

    step = 'memory-8/step-512/'
    ref = json.loads(memory[step + 'reference-predictions.json'])
    require([sample_key(row) for row in ref] == keys, 'Reference task order differs')
    positions = [row['turn_end_indices'] for row in ref]
    counts = np.array([len(ends) for ends in positions], dtype=np.int64)
    require(all(ends and ends == sorted(set(ends)) and ends[0] >= 0 for ends in positions),
            'Invalid archived scoring positions')
    require(all(sum(m['role'] == 'assistant' for m in row['messages']) == n
                for row, n in zip(samples, counts)), 'Turn counts differ from source messages')
    archived_error = np.array([row['total'] for row in ref]) - labels
    archived = metrics(archived_error)
    progress = json.loads(memory['memory-8/progress.json'])['history'][-1]
    require(progress['updates'] == MAX_UPDATES and all(
        abs(archived[k] - progress[k]) < 1e-12 for k in ('mse', 'max_absolute_error')
    ) and archived['passed'] == progress['passed'], 'Historical memory metrics differ')

    manifest_bytes = memory[step + 'prm-manifest.json']
    manifest = json.loads(manifest_bytes)
    require(manifest_bytes == supplement['step-512/prm-manifest.json'],
            'Step-512 supplement manifest differs from original')
    rep = 'representation-diagnostic/'
    meta = json.loads(supplement[rep + 'F.meta.json'])
    config = json.loads(supplement[rep + 'extraction.config.json'])
    identity = manifest['identity']
    require(meta['score_position'] == 'assistant im_end (turn_end_indices)'
            and meta['eval_mode'] is True and meta['feature_dtype'] == 'float64'
            and meta['summation_precision'] == 'FP32 logits summed in float32 then cast to float64',
            'Unexpected extraction declarations')
    require(config['revision'] == meta['revision'] == identity['base_revision']
            and config['base_weights_sha256'] == identity['base_sha256']
            and config['tokenizer_revision'] == identity['tokenizer_revision']
            and config['tokenizer_files_sha256'] == identity['tokenizer_sha256']
            and config['sft_adapter_files_sha256'] == meta['sft_adapter_sha256']
            == identity['initialization']['sft_adapter_files_sha256']
            and config['policy_dtype'] == 'float32' and config['load_in_4bit'] is False
            and config['score_position'] == identity['score_position'] == 'assistant_im_end',
            'Extraction configuration and archived model identity disagree')
    require(len(meta['samples_order']) == len(samples) and all(
        row['row'] == i and sample_key(row) == key and row['turns'] == count
        and row['label'] == label
        for i, (row, key, count, label) in enumerate(zip(meta['samples_order'], keys, counts, labels))
    ), 'Feature rows do not align with original samples and scoring positions')
    feature_bytes = supplement[rep + 'F.npy']
    sums = np.load(io.BytesIO(feature_bytes), allow_pickle=False)
    require(sums.shape == (len(keys), manifest['vocab_size']) and sums.dtype == np.float64
            and np.isfinite(sums).all(), 'Expected finite float64 raw-logit sums with no bias column')
    design = np.column_stack([sums, counts])
    _, fit = fit_fixed_representation(design, labels, rcond=1e-10)
    for report in (memory[rep + 'representation-diagnostic.json'],
                   supplement[rep + 'representation-diagnostic.json']):
        reported = json.loads(report)
        require(fit['rank'] == reported['numerical_rank'] and np.allclose(
            fit['singular_values'], reported['singular_values'], rtol=1e-10, atol=1e-10
        ) and np.isclose(fit['solution_l2'], reported['weight_l2_norm'], rtol=1e-10, atol=1e-12),
                'Recomputed spectrum or weight norm differs from report')

    state_bytes = supplement['step-512/training-state.pt']
    state_hash = digest(state_bytes)
    require(state_hash == manifest['files_sha256']['training-state.pt'],
            'Optimizer state digest differs from original checkpoint manifest')
    saved = torch.load(io.BytesIO(state_bytes), map_location='cpu', weights_only=True)
    require(saved['samples_sha256'] == digest(sample_bytes) and saved['completed_steps'] == MAX_UPDATES,
            'Optimizer state has unexpected sample identity or step')
    optimizer = saved['optimizer']
    require(len(optimizer['param_groups']) == 1, 'Expected one historical optimizer group')
    group = optimizer['param_groups'][0]
    slots = optimizer['state']
    require(len(group['params']) == len(slots) and set(group['params']) == set(slots),
            'Optimizer group membership differs from state')
    head = torch.load(io.BytesIO(memory[step + 'head.pt']), map_location='cpu', weights_only=True)
    adapter = memory[step + 'adapter/adapter_model.safetensors']
    header_size = struct.unpack_from('<Q', adapter)[0]
    header = json.loads(adapter[8:8 + header_size])
    shapes = Counter(tuple(x.shape) for x in head.values())
    shapes.update(tuple(x['shape']) for k, x in header.items() if k != '__metadata__')
    require(shapes == Counter(tuple(slot['exp_avg'].shape) for slot in slots.values()),
            'Optimizer moment shapes differ from saved parameter shapes')
    for slot in slots.values():
        avg, sq = slot['exp_avg'], slot['exp_avg_sq']
        require(float(slot['step']) == MAX_UPDATES and avg.shape == sq.shape
                and avg.dtype == sq.dtype == torch.float32
                and bool(torch.isfinite(avg).all() and torch.isfinite(sq).all() and (sq >= 0).all()),
                'Nonfinite, mismatched or non-FP32 AdamW state')
    settings = {key: group[key] for key in ('lr', 'betas', 'eps', 'weight_decay')}
    require(settings['lr'] == plan['learning_rate'] == saved['learning_rate'] == LEARNING_RATES[0],
            'Original learning rate differs from the comparison baseline')
    rng = [saved['cpu_rng'], *saved['cuda_rng']]
    require(all(t.dtype == torch.uint8 and t.ndim == 1 and t.numel() for t in rng),
            'Invalid CPU/CUDA RNG state record')

    report = {
        'memory_archive_sha256': MEMORY_DIGEST, 'supplement_archive_sha256': SUPPLEMENT_DIGEST,
        'memory_payloads_verified': len(memory), 'supplement_payloads_verified': len(supplement),
        'samples_sha256': digest(sample_bytes), 'task_ids_in_order': [task for task, _ in keys],
        'labels_in_order': labels.tolist(), 'turn_counts_in_order': counts.tolist(),
        'historical_memory_run_at_512': archived,
        'representation': {
            'origin': 're-extracted fixed B1 features; first extraction array was not saved',
            'stored_shape': list(sums.shape), 'stored_F_sha256': digest(feature_bytes),
            'solve_shape': list(design.shape), 'bias_column_source': 'archived scoring position counts',
            'config_sha256': digest(supplement[rep + 'extraction.config.json']),
            'extractor_sha256': digest(supplement[rep + 'svd_diagnostic.py']),
            'base_revision': config['revision'],
            'b1_adapter_sha256': config['sft_adapter_files_sha256']['adapter_model.safetensors'],
            'matches_reported_rank_spectrum_and_solution_norm': True,
            'source_binding_of_extraction_independently_verified': False,
            'raw_step_logits_available': False, 'model_forward_repeated': False,
            **fit,
        },
        'optimizer': {
            'checkpoint_manifest_match': True, 'training_state_sha256': state_hash,
            'historical_source_commit': identity['source_commit'],
            'completed_steps': saved['completed_steps'], 'seed': saved['seed'],
            'parameter_states': len(slots), 'moment_dtype': 'float32',
            'parameter_shape_multiset_matches': True,
            'cpu_rng_bytes': rng[0].numel(), 'cuda_rng_states': len(rng) - 1,
            **settings,
        },
    }
    return report, sums, counts, labels, settings


def metrics(errors):
    mse = float((errors ** 2).mean())
    max_error = float(abs(errors).max())
    return {'mse': mse, 'max_absolute_error': max_error,
            'passed': mse < 0.001 and max_error < 0.1}


def train_head(sums, counts, labels, settings, lr, batch_size):
    import torch

    features = torch.tensor(sums, dtype=torch.float32)
    turns = torch.tensor(counts, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.float32)
    head = torch.nn.Linear(features.shape[1], 1, dtype=torch.float32, device='cpu')
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.AdamW(head.parameters(), **{**settings, 'lr': lr})
    history, clips = [], 0
    stop_reason = 'budget_reached'
    updates, presented = 0, 0
    for start in range(0, MAX_UPDATES, batch_size):
        indices = (torch.arange(batch_size) + start) % len(labels)
        optimizer.zero_grad(set_to_none=True)
        gains = torch.nn.functional.linear(features[indices], head.weight).squeeze(-1)
        loss = (gains + turns[indices] * head.bias - targets[indices]).square().mean()
        presented += batch_size
        if not bool(torch.isfinite(loss)):
            stop_reason = 'nonfinite_loss'
            break
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        if not bool(torch.isfinite(norm)):
            stop_reason = 'nonfinite_gradient_norm'
            break
        clips += float(norm) > 1.0
        optimizer.step()
        updates += 1
        if not all(bool(torch.isfinite(p).all()) for p in head.parameters()):
            stop_reason = 'nonfinite_parameters'
            break
        if presented % CHECK_INTERVAL:
            continue
        with torch.inference_mode():
            gains = torch.nn.functional.linear(features, head.weight).squeeze(-1)
            predictions = (gains + turns * head.bias).numpy().astype('float64')
        if not torch.isfinite(torch.from_numpy(predictions)).all():
            stop_reason = 'nonfinite_prediction'
            break
        recorded = {'optimizer_updates': updates, 'samples_presented': presented,
                    'clipped_updates': clips, 'predictions_in_task_order': predictions.tolist(),
                    **metrics(predictions - labels)}
        history.append(recorded)
        if recorded['passed']:
            stop_reason = 'memory_gate_reached'
            break
    return {'learning_rate': lr, 'batch_size': batch_size, 'optimizer_updates': updates,
            'samples_presented': presented, 'clipped_updates': clips, 'stop_reason': stop_reason,
            'checkpoints': history, 'passed': stop_reason == 'memory_gate_reached'}


def compare_heads(audit, sums, counts, labels, settings):
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    arms = [train_head(sums, counts, labels, settings, lr, batch_size=1) for lr in LEARNING_RATES]
    eligible = [arm for arm in arms if arm['checkpoints']
                and arm['stop_reason'] in ('budget_reached', 'memory_gate_reached')]
    batch_comparison = None
    if eligible and not any(arm['passed'] for arm in arms):
        # Pick by last common-budget MSE, with the rate list order breaking a tie.
        selected = min(eligible, key=lambda arm: arm['checkpoints'][-1]['mse'])
        batch_comparison = {
            'learning_rate': selected['learning_rate'],
            'batch_1_control_index': arms.index(selected),
            'batch_8': train_head(sums, counts, labels, settings, selected['learning_rate'], batch_size=8),
        }
    return {
        'scope': 'FP32 CPU AdamW trains only a zero-initialized linear head on fixed B1 logit sums',
        'python': platform.python_version(), 'numpy': np.__version__, 'torch': str(torch.__version__),
        'task_ids_in_order': audit['task_ids_in_order'], 'labels_in_order': labels.tolist(),
        'samples_sha256': audit['samples_sha256'],
        'F_sha256': audit['representation']['stored_F_sha256'],
        'weight_decay': settings['weight_decay'], 'betas': settings['betas'], 'eps': settings['eps'],
        'grad_clip_norm': 1.0, 'max_sample_presentations_per_arm': MAX_UPDATES,
        'evaluate_every_sample_presentations': CHECK_INTERVAL,
        'gate': {'mse_strictly_below': 0.001, 'max_absolute_error_strictly_below': 0.1},
        'learning_rate_arms': arms, 'batch_comparison_if_learning_rate_gate_unmet': batch_comparison,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--supplement', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    outputs = [args.output_dir / name for name in ('supplement-review.json', 'cpu-head-comparison.json')]
    if any(path.exists() for path in outputs):
        parser.error('Output reports already exist; use a new output directory')
    memory = read_archive(args.bundle, MEMORY_DIGEST, 'prm-evidence-20261124')
    supplement = read_archive(args.supplement, SUPPLEMENT_DIGEST, 'prm-evidence-supplement-20261124')
    audited, sums, counts, labels, settings = audit_inputs(memory, supplement)
    comparison = compare_heads(audited, sums, counts, labels, settings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, result in zip(outputs, (audited, comparison)):
        with path.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()
