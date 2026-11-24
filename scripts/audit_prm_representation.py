"""CPU-only, read-only audit of a PRM fixed-representation export.

NumPy is an analysis dependency, not a new training dependency. Never loads a
checkpoint, runs a model, or authorizes training. See docs/PRM_EVALUATION_AUDIT.md.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read_json(path):
    def invalid(value):
        raise ValueError('Non-finite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_key(row):
    task, repeat = row['task_id'], row.get('repeat', 0)
    if type(task) is not int or type(repeat) is not int or task < 0 or repeat < 0:
        raise ValueError('task_id/repeat must be nonnegative integers')
    return task, repeat


def finite_number(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError('Expected a finite numeric label')
    return float(value)


def fit_fixed_representation(features, labels, rcond):
    """Solve the fixed linear problem and check its FP32 CPU reconstruction."""
    import numpy as np

    F64, y64 = features.astype(np.float64), labels.astype(np.float64)
    theta, _, rank, singular = np.linalg.lstsq(F64, y64, rcond=rcond)
    residual = F64 @ theta - y64
    fp32 = F64.astype(np.float32) @ theta.astype(np.float32)
    return theta, {
        'rcond': rcond, 'rank': int(rank), 'singular_values': singular.tolist(),
        'rank_cutoff': float(rcond * singular[0]),
        'nonzero_spectrum_condition': float(singular[0] / singular[rank - 1]) if rank else None,
        'full_row_rank': int(rank) == len(labels),
        'solution_l2': float(np.linalg.norm(theta)),
        'solution_max_abs': float(np.abs(theta).max()),
        'residual_max_abs': float(np.abs(residual).max()),
        'residual_rms': float(np.sqrt(np.mean(residual ** 2))),
        'fp32_aggregated_max_abs_label_error': float(np.abs(fp32 - y64).max()),
    }


def audit(features, samples, positions, manifest, rcond=1e-10, tokenizer=None,
          expected_samples=8):
    import numpy as np
    if not math.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError('rcond must be finite and strictly between zero and one')
    meta = read_json(manifest)
    if meta.get('schema_version') != 'prm-representation-export-v1':
        raise ValueError('Unsupported export manifest schema')
    v = meta.get('vocab_size')
    if type(v) is not int or v < 1:
        raise ValueError('Invalid vocab_size')
    if meta.get('feature_definition') != 'concat(sum_step_raw_logits,turn_count)':
        raise ValueError('Expected sum of raw logits and per-turn bias coefficient')
    if meta.get('score_position') != 'assistant_im_end':
        raise ValueError('Wrong scoring position definition')
    required = {'eval_mode': True, 'dropout_enabled': False,
                'labels_in_forward': False, 'parameter_dtype': 'float32'}
    if any(type(meta.get(k)) is not type(value) or meta[k] != value
           for k, value in required.items()):
        raise ValueError('Extraction declarations violate the frozen contract')
    for path in (features, samples, positions):
        path = Path(path)
        try:
            relative = path.resolve().relative_to(Path(manifest).resolve().parent).as_posix()
        except ValueError as exc:
            raise ValueError('Export inputs must be inside the manifest directory') from exc
        if meta.get('files_sha256', {}).get(relative) != sha256(path):
            raise ValueError('Missing or mismatched SHA256: ' + relative)
    rows = []
    for line in Path(samples).read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(
                ValueError('Non-finite JSON constant: ' + x))))
    pos = read_json(positions)
    if not isinstance(pos, list) or not rows or len(rows) != len(pos):
        raise ValueError('Samples and positions must have the same nonzero length')
    if len(rows) != expected_samples:
        raise ValueError('Unexpected sample count; do not silently audit a subset')
    keys = [sample_key(row) for row in rows]
    if len(set(keys)) != len(keys) or keys != [sample_key(row) for row in pos]:
        raise ValueError('Duplicated, reordered, or mismatched task/repeat keys')
    counts, rewards = [], []
    for row, p in zip(rows, pos):
        messages = row['messages']
        if (not isinstance(messages, list) or not messages or
                messages[0].get('role') != 'system' or
                messages[-1].get('role') != 'assistant'):
            raise ValueError('Expected complete messages ending at the last action')
        n = sum(m.get('role') == 'assistant' for m in messages)
        ends, ids = p['turn_end_indices'], p['input_ids']
        if (not n or len(ends) != n or not ids or len(ids) > meta.get('max_length', 2048) or
                any(type(x) is not int or x < 0 or x >= v for x in ids) or
                any(type(x) is not int or x < 0 or x >= len(ids) for x in ends) or
                ends != sorted(set(ends))):
            raise ValueError('Invalid token IDs, turn count, or scoring indices')
        eos = meta.get('im_end_token_id')
        if type(eos) is not int or not 0 <= eos < v or any(ids[x] != eos for x in ends):
            raise ValueError('Scoring position is not the declared im_end token')
        if tokenizer is not None:
            ids2 = tokenizer.apply_chat_template(messages, tokenize=True,
                                                  add_generation_prompt=False)
            if list(ids2) != ids:
                raise ValueError('Real tokenizer IDs differ from the exported input')
            # Prefix-stability checks avoid counting user EOS as an action boundary.
            expected = []
            for i, message in enumerate(messages):
                if message.get('role') != 'assistant':
                    continue
                prefix = list(tokenizer.apply_chat_template(
                    messages[:i + 1], tokenize=True, add_generation_prompt=False))
                if ids[:len(prefix)] != prefix:
                    raise ValueError('Chat template is not prefix stable')
                at = len(prefix) - 1
                while at >= 0 and prefix[at] != eos:
                    at -= 1
                expected.append(at)
            if expected != ends:
                raise ValueError('Assistant turn boundaries differ from real tokenization')
        reward = finite_number(row['terminal_reward'])
        if not 0 <= reward <= 1:
            raise ValueError('Terminal reward outside [0,1]')
        rewards.append(reward)
        counts.append(n)
    with np.load(features, allow_pickle=False) as archive:
        F, y = archive['F'], archive['y']
        if (F.dtype not in (np.dtype('float32'), np.dtype('float64')) or
                y.dtype.kind not in 'fiu'):
            raise ValueError('F must be floating point and y must be numeric, not boolean/object')
        if F.shape != (len(rows), v + 1) or y.shape != (len(rows),):
            raise ValueError('Expected F=[sum logits,N], with no trajectory intercept')
        if (not np.isfinite(F).all() or not np.isfinite(y).all() or
                not np.allclose(y, rewards, atol=1e-7, rtol=0) or
                not np.array_equal(F[:, -1], counts)):
            raise ValueError('Non-finite data, changed labels, or wrong bias coefficients')
        step_verified = 'step_logits' in archive and 'offsets' in archive
        z = offsets = None
        if ('step_logits' in archive) != ('offsets' in archive):
            raise ValueError('step_logits and offsets must be provided together')
        if step_verified:
            z, offsets = archive['step_logits'], archive['offsets']
            expected_offsets = np.r_[0, np.cumsum(counts)]
            if (z.dtype != np.float32 or z.shape != (sum(counts), v) or
                    offsets.dtype.kind not in 'iu' or
                    not np.array_equal(offsets, expected_offsets) or not np.isfinite(z).all()):
                raise ValueError('Wrong raw-step feature shape/dtype/offsets')
            aggregation_dtype = meta.get('aggregation_dtype', 'float32')
            if aggregation_dtype not in ('float32', 'float64'):
                raise ValueError('Unsupported aggregation_dtype')
            sums = np.stack([z[a:b].sum(axis=0, dtype=aggregation_dtype)
                             for a, b in zip(offsets[:-1], offsets[1:])])
            if not np.allclose(sums, F[:, :-1], atol=1e-5, rtol=1e-6):
                raise ValueError('F does not reconstruct from per-step raw logits')
        theta, fit = fit_fixed_representation(F, y, rcond)
        result = {
            'schema_version': 'prm-representation-audit-v1',
            'scope': 'fixed-representation fit only; no model forward or GPU training',
            'sample_count': len(rows), 'features': int(F.shape[1]),
            'input_sha256': {Path(p).name: sha256(p)
                             for p in (features, samples, positions, manifest)},
            'extraction_identity_declared': meta.get('model_identity', {}),
            'extraction_runtime_reproduced': False,
            'tokenizer_boundaries_recomputed': tokenizer is not None,
            'F_reconstructed_from_raw_steps': step_verified,
            **fit,
            'memory_training_passed': None, 'generalization_passed': None,
            'step_attribution_passed': None,
            'budget_inference': 'No step-count lower bound follows from this solution or lr',
        }
        if step_verified:
            result['aggregation_dtype'] = aggregation_dtype
            result['raw_step_sum_max_abs_difference'] = float(np.abs(sums-F[:, :-1]).max())
            result['raw_step_sum_tolerance'] = {'atol': 1e-5, 'rtol': 1e-6}
            per_step = z @ theta[:-1].astype(np.float32) + np.float32(theta[-1])
            totals = np.array([per_step[a:b].sum(dtype=np.float32)
                               for a, b in zip(offsets[:-1], offsets[1:])])
            result['fp32_step_then_sum_max_abs_label_error'] = float(
                np.abs(totals - y.astype(np.float64)).max())
        result['status'] = ('representation_inputs_checked' if step_verified and tokenizer is not None
                            else 'partial_requires_raw_steps_or_real_tokenizer')
        result['limitations'] = [
            'Full row rank is not full column rank of the LM output matrix.',
            'A minimum-L2 interpolant does not minimize maximum coordinate magnitude.',
            'FP32 results use NumPy CPU arithmetic, not the original GPU kernel.',
            'Identity and eval/no-dropout declarations require extractor/source review.',
        ]
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'samples', 'positions', 'manifest', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--rcond', type=float, default=1e-10)
    parser.add_argument('--expected-samples', type=int, default=8)
    parser.add_argument('--tokenizer', help='Optional local tokenizer; never downloads files')
    args = parser.parse_args(argv)
    output = Path(args.output)
    inputs = [Path(getattr(args, name)).resolve()
              for name in ('features', 'samples', 'positions', 'manifest')]
    if output.exists() or output.resolve() in inputs:
        parser.error('Output must be new and must not overwrite any input')
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        meta = read_json(args.manifest)
        tokenizer_root = Path(args.tokenizer).resolve()
        hashes = meta.get('model_identity', {}).get('tokenizer_files_sha256', {})
        if not hashes:
            parser.error('Real tokenizer verification requires recorded tokenizer file hashes')
        for name, digest in hashes.items():
            path = (tokenizer_root/name).resolve()
            if (not path.is_relative_to(tokenizer_root) or not path.is_file() or
                    sha256(path) != digest):
                parser.error('Tokenizer file identity mismatch: ' + name)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True,
                                                  trust_remote_code=False)
        template_sha = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        if template_sha != meta.get('model_identity', {}).get('template_sha256'):
            parser.error('Tokenizer template identity mismatch')
        eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
        if eos != meta.get('im_end_token_id') or tokenizer.decode([eos]) != '<|im_end|>':
            parser.error('Tokenizer im_end identity mismatch')
    try:
        result = audit(args.features, args.samples, args.positions, args.manifest,
                       args.rcond, tokenizer, args.expected_samples)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, 'Audit rejected: ' + str(exc) + '\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(output)}))
    return 0 if result['status'] == 'representation_inputs_checked' else 2


if __name__ == '__main__':
    raise SystemExit(main())
