"""Audit and summarize all six completed fixed-budget project validation runs."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, p):
    values = sorted(values)
    at = (len(values) - 1) * p
    lo = math.floor(at)
    return values[lo] + (values[math.ceil(at)] - values[lo]) * (at - lo)


def paired(a, b, ids, field):
    delta = [b[i][field] - a[i][field] for i in ids]
    rng = random.Random(42)
    draws = [sum(rng.choices(delta, k=len(ids))) / len(ids) for _ in range(10000)]
    return {'difference': sum(delta) / len(ids),
            'ci95': [quantile(draws, .025), quantile(draws, .975)]}


def summarize(stage):
    ids = read(stage / 'val_ids.json')
    assert len(ids) == len(set(ids)) == 160 and 6700 in ids
    assert sha(stage / 'val_ids.json') == '45440d724b33e69e13916d0adbe243cc63ce3bd67677b8220feff9f30b866c36'
    identity = read(stage / 'identity-preflight.json')
    labels = ['b0', 'b1', 'tg42', 'op42', 'tg43', 'op43']
    result = {'scope': 'complete fixed-budget project FP32 validation; registration pending',
              'scheduled_tasks': 160, 'policies': {}, 'comparisons': {},
              'bootstrap': {'resamples': 10000, 'seed': 42, 'unit': 'paired task',
                            'implementation': 'Python random.Random.choices, linear quantiles',
                            'order': 'frozen val_ids.json'},
              'test_authorization': False, 'formal_registration': False}
    aligned = {}
    shared_config, shared_backend, shared_source = None, None, None
    for label in labels:
        folder = stage / label
        config = read(stage / 'configs' / (label + '.json'))
        meta = read(folder / 'rollout.meta.json')
        audit = read(folder / 'execution-audit.json')
        evaluation = read(folder / 'evaluation.json')
        rows = [json.loads(line) for line in (folder / 'rollout.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [r['task_id'] for r in rows] == ids
        assert all(r['run_id'] == meta['run_id'] and r['repeat'] == 0 for r in rows)
        assert all(r['success'] == (r['done'] and r['final_reward'] >= 1 - 1e-6) for r in rows)
        assert audit['status'] == 'complete' and audit['optimizer_updates'] == 0
        assert audit['jsonl_sha256'] == sha(folder / 'rollout.jsonl')
        assert audit['meta_sha256'] == sha(folder / 'rollout.meta.json')
        assert meta['config'] == config
        assert meta['config_sha256'] == sha(stage / 'configs' / (label + '.json'))
        assert meta['ids_sha256'] == sha(stage / 'val_ids.json')
        assert meta['selected_ids'] == ids
        assert meta['precision']['selected_load_dtype'] == 'torch.float32'
        assert set(meta['precision']['parameter_numel_by_dtype']) == {'torch.float32'}
        assert meta['precision']['input_identity'] == identity['policies'][label]
        assert meta['precision']['base_weights_sha256'] == identity['base_weights_sha256']
        assert evaluation['usable_as_result'] and evaluation['completeness']['complete']
        assert evaluation['completeness']['provenance_ok']
        assert evaluation['run']['commit'] == audit['source_commit'] == meta['precision']['source_commit']
        comparable = {k: v for k, v in config.items() if k != 'adapter'}
        backend = meta['precision']['backend']
        source = meta['execution_source_sha256']
        if shared_config is None:
            shared_config, shared_backend, shared_source = comparable, backend, source
        assert comparable == shared_config and backend == shared_backend and source == shared_source
        assert backend['float32_matmul_precision'] == 'highest'
        assert not backend['cuda_matmul_allow_tf32'] and not backend['cudnn_allow_tf32']
        assert not backend['autocast_enabled']
        assert meta['health']['type_reward_cache'] == 'page-local-v1'
        assert all(r['goal_fingerprint'] == config['goal_fingerprint'] for r in rows)
        metrics = evaluation['metrics']
        data = {r['task_id']: {'success': int(r['done'] and r['final_reward'] >= 1 - 1e-6),
                               'terminal_reward': r['final_reward'] if r['done'] else 0.}
                for r in rows}
        assert sum(v['success'] for v in data.values()) == metrics['full_reward_successes']
        assert abs(sum(v['terminal_reward'] for v in data.values()) / 160 - metrics['mean_terminal_reward']) < 1e-12
        aligned[label] = data
        result['policies'][label] = {
            'metrics': metrics, 'run_id': meta['run_id'],
            'config_sha256': audit['config_sha256'], 'jsonl_sha256': audit['jsonl_sha256'],
            'meta_sha256': audit['meta_sha256'], 'elapsed_seconds': audit['elapsed_seconds'],
            'task6700': data[6700], 'source_commit': audit['source_commit']}
    for left, right in [('b0', 'b1'), ('b1', 'tg42'), ('b1', 'tg43'),
                        ('b1', 'op42'), ('b1', 'op43'), ('tg42', 'op42'), ('tg43', 'op43')]:
        a, b = aligned[left], aligned[right]
        result['comparisons'][right + '_minus_' + left] = {
            'success': paired(a, b, ids, 'success'),
            'terminal_reward': paired(a, b, ids, 'terminal_reward'),
            'only_left_success': sum(a[i]['success'] and not b[i]['success'] for i in ids),
            'only_right_success': sum(b[i]['success'] and not a[i]['success'] for i in ids)}
    result['shared_config'] = shared_config
    result['shared_backend'] = shared_backend
    result['shared_source_sha256'] = shared_source
    result['training'] = read(stage / 'training-completion-ledger.json')['totals']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', type=Path)
    args = parser.parse_args()
    result = summarize(args.stage)
    (args.stage / 'six-policy-comparison.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for label, data in result['policies'].items():
        print(label, data['metrics']['full_reward_successes'], '/160', data['metrics']['mean_terminal_reward'])
