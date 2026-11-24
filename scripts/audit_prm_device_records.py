"""Audit PR46 summaries on CPU; no GPU, weight reload or dev reproduction."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, source=None):
    names = ['smoke-lr5e6-summary.json', 'smoke-lr1e6-summary.json',
             'smoke-lr1e7-summary.json', 'train-run-001-summary.json']
    result = {'scope': 'committed-record consistency only; no GPU/reload/dev reproduction',
              'summaries': {}, 'input_sha256': {}}
    all_source_hashes = None
    for name in names:
        path = root / name
        data = json.loads(path.read_text())
        records = data['records']
        if len(records) != data['optimizer_steps']:
            raise ValueError('Record/update count differs: ' + name)
        errors = []
        for index, record in enumerate(records, 1):
            if record['optimizer_step'] != index or len(record['totals_before_update']) != 1:
                raise ValueError('Unexpected record layout: ' + name)
            p, y, loss = (record['totals_before_update'][0],
                          record['terminal_reward'], record['loss'])
            if not all(math.isfinite(v) for v in (p, y, loss)):
                raise ValueError('Nonfinite training record: ' + name)
            errors.append(abs((p - y) ** 2 - loss))
        if max(errors) > 1e-5:
            raise ValueError('Recorded MSE differs from prediction/label: ' + name)
        losses = [r['loss'] for r in records]
        lengths = [(r['task_id'], int(m.group(1))) for r in data['rejected']
                   if (m := re.search(r'Complete trajectory has (\d+) tokens', r['reason']))]
        result['summaries'][name] = {
            'updates': len(records), 'mse_recompute_max_absolute_difference': max(errors),
            'loss_min': min(losses), 'loss_max': max(losses), 'losses': losses,
            'post_first_loss_min': min(losses[1:]), 'post_first_loss_max': max(losses[1:]),
            'maximum_observed_rejected_tokens': max((n for _, n in lengths), default=None),
            'maximum_is_full_dataset_maximum': False,
        }
        expected = data['identity']['source_sha256']
        if all_source_hashes is not None and expected != all_source_hashes:
            raise ValueError('Historical source identities differ')
        all_source_hashes = expected
        result['input_sha256'][name] = sha(path)
    manifest_path = root / 'train-run-001-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if result['input_sha256']['train-run-001-summary.json'] != manifest['files_sha256']['summary.json']:
        raise ValueError('20-step summary bytes differ from checkpoint manifest')
    result['input_sha256'][manifest_path.name] = sha(manifest_path)
    result['summary_manifest_hash_matches'] = True
    if source is not None:
        actual = {name: sha(source / name) for name in all_source_hashes}
        if actual != all_source_hashes:
            raise ValueError('Recovered historical source differs from summary identity')
        result['recovered_source_sha256'] = actual
        result['historical_source_files_match'] = True
    result['original_smoke_reload_status'] = 'self-reported; original checkpoint/reload evidence pending'
    result['original_29_dev_status'] = 'self-reported; per-task predictions and baseline evidence pending'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'results/prm-device-20261124')
    parser.add_argument('--source', type=Path,
                        help='Optional recovered a67db6a src/webshop_lab/prm directory')
    args = parser.parse_args()
    print(json.dumps(audit(args.records, args.source), indent=2, allow_nan=False))
