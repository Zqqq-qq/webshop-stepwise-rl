"""Independent read-only evaluator for PR #57 unviewed independent dev acceptance.

Computes from How's returned device artifacts (without running any model/rollout/test):
1. Pair coverage breakdown: attempted tasks, valid environment-contrast pairs (env_high > env_low
   and both arms <= 2048 tokens), environment-tie pairs (env_high == env_low, logged separately),
   too_long rejections, and non-fork/technical rejections.
2. Fork-step (open-product, step index 1) ranking accuracy and mean B-A (high-minus-low) margin
   for stage126, terminal, and ranking on the exact same frozen valid pairs, plus paired delta
   between ranking and terminal.
3. Total-score MSE on the independent dev trajectories against the three fit501-only baselines
   (constant mean, turn-count mean, OLS on turn count) with deterministic paired bootstrap 95% CIs.
4. Separate regression-only summary for examined task 7544 (never pooled into independent dev).

Enforces test_authorization=false and strict ID/input consistency across all 3 checkpoints.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path


MODELS = ('stage126', 'terminal', 'ranking')
FORK_STEP_INDEX = 1
PREFIX_TOL = 1e-5
SUM_TOL = 1e-6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fit_only_baselines(fit_rows: list[dict]) -> dict:
    """Fit constant, turn-count mean, and OLS-on-turn-count baselines strictly on fit501."""
    require(len(fit_rows) > 0, 'Empty fit_rows')
    ys = [float(r['terminal_reward']) for r in fit_rows]
    turns = [int(r['num_turns']) for r in fit_rows]
    n = len(ys)
    const_mean = math.fsum(ys) / n

    by_turn: dict[int, list[float]] = {}
    for t, y in zip(turns, ys):
        by_turn.setdefault(t, []).append(y)
    turn_means = {t: math.fsum(vals) / len(vals) for t, vals in by_turn.items()}

    mean_t = math.fsum(turns) / n
    var_t = math.fsum((t - mean_t) ** 2 for t in turns)
    cov_ty = math.fsum((t - mean_t) * (y - const_mean) for t, y in zip(turns, ys))
    slope = cov_ty / var_t if var_t > 0 else 0.0
    intercept = const_mean - slope * mean_t
    return {
        'fit_count': n,
        'constant': const_mean,
        'turn_means': turn_means,
        'ols_intercept': intercept,
        'ols_slope': slope,
    }


def predict_baselines(baselines: dict, num_turns: int) -> dict[str, float]:
    const = baselines['constant']
    tmean = baselines['turn_means'].get(num_turns, const)
    ols = baselines['ols_intercept'] + baselines['ols_slope'] * num_turns
    return {'constant': const, 'turn_count_mean': tmean, 'ols_turn_count': ols}


def paired_bootstrap_ci(diffs: list[float], n_boot: int = 2000, seed: int = 42) -> dict:
    require(len(diffs) > 0, 'Empty diffs for bootstrap')
    n = len(diffs)
    mean_val = math.fsum(diffs) / n
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        s = math.fsum(diffs[rng.randrange(n)] for _ in range(n)) / n
        samples.append(s)
    samples.sort()
    lo = samples[int(0.025 * n_boot)]
    hi = samples[min(n_boot - 1, int(0.975 * n_boot))]
    return {'mean': mean_val, 'ci95': [lo, hi], 'n': n, 'n_boot': n_boot, 'seed': seed}


def evaluate_fork_pairs(
    frozen_dev_ids: list[int],
    pair_records: list[dict],
    predictions_by_model: dict[str, dict[tuple[int, str], dict]],
) -> dict:
    """Evaluate fork-step ranking on strictly positive environment-contrast pairs."""
    frozen_set = set(frozen_dev_ids)
    counts = {
        'attempted_tasks': len(pair_records),
        'valid_contrast_pairs': 0,
        'environment_ties': 0,
        'too_long_rejected': 0,
        'non_fork_or_other_rejected': 0,
    }
    per_task = []
    margins: dict[str, list[float]] = {m: [] for m in MODELS}
    correct: dict[str, int] = {m: 0 for m in MODELS}
    max_prefix_drift: dict[str, float] = {m: 0.0 for m in MODELS}

    for rec in pair_records:
        tid = rec['task_id']
        require(tid in frozen_set, f'Task {tid} is not in frozen independent dev list')
        status = rec['status']
        if status == 'environment_tie':
            counts['environment_ties'] += 1
            per_task.append({'task_id': tid, 'status': status, 'env_high': rec.get('env_high'), 'env_low': rec.get('env_low')})
            continue
        if status == 'too_long':
            counts['too_long_rejected'] += 1
            per_task.append({'task_id': tid, 'status': status})
            continue
        if status != 'valid_pair':
            counts['non_fork_or_other_rejected'] += 1
            per_task.append({'task_id': tid, 'status': status})
            continue

        env_high = float(rec['env_high'])
        env_low = float(rec['env_low'])
        require(env_high > env_low, f'Task {tid} marked valid_pair without env_high > env_low')
        counts['valid_contrast_pairs'] += 1

        row_summary = {
            'task_id': tid,
            'status': 'valid_pair',
            'env_high': env_high,
            'env_low': env_low,
            'env_delta': env_high - env_low,
            'models': {},
        }
        ref_turn_indices = None
        for m in MODELS:
            hi = predictions_by_model[m][(tid, 'high')]
            lo = predictions_by_model[m][(tid, 'low')]
            require(hi['turn_end_indices'][0] == lo['turn_end_indices'][0],
                    f'Task {tid} model {m} shared step-0 turn_end_index mismatch')
            if ref_turn_indices is None:
                ref_turn_indices = (hi['turn_end_indices'][FORK_STEP_INDEX], lo['turn_end_indices'][FORK_STEP_INDEX])
            else:
                require(
                    (hi['turn_end_indices'][FORK_STEP_INDEX], lo['turn_end_indices'][FORK_STEP_INDEX]) == ref_turn_indices,
                    f'Task {tid} model {m} fork turn_end_indices differ across checkpoints',
                )
            for arm_name, pred in (('high', hi), ('low', lo)):
                require(all(math.isfinite(x) for x in pred['step_scores']), f'Non-finite score in {tid}/{m}/{arm_name}')
                require(math.isclose(math.fsum(pred['step_scores']), pred['total'], abs_tol=SUM_TOL),
                        f'Sum mismatch in {tid}/{m}/{arm_name}')
            prefix_diff = abs(hi['step_scores'][0] - lo['step_scores'][0])
            require(prefix_diff <= PREFIX_TOL, f'Task {tid} model {m} shared prefix drift {prefix_diff} > {PREFIX_TOL}')
            max_prefix_drift[m] = max(max_prefix_drift[m], prefix_diff)

            fork_delta = hi['step_scores'][FORK_STEP_INDEX] - lo['step_scores'][FORK_STEP_INDEX]
            total_delta = hi['total'] - lo['total']
            margins[m].append(fork_delta)
            if fork_delta > 0:
                correct[m] += 1
            row_summary['models'][m] = {
                'fork_high': hi['step_scores'][FORK_STEP_INDEX],
                'fork_low': lo['step_scores'][FORK_STEP_INDEX],
                'fork_delta': fork_delta,
                'total_delta': total_delta,
                'prefix_diff_step0': prefix_diff,
            }
        per_task.append(row_summary)

    n_valid = counts['valid_contrast_pairs']
    model_metrics = {}
    for m in MODELS:
        model_metrics[m] = {
            'correct_pairs': correct[m],
            'valid_pairs': n_valid,
            'fork_accuracy': (correct[m] / n_valid) if n_valid > 0 else None,
            'mean_fork_delta': (math.fsum(margins[m]) / n_valid) if n_valid > 0 else None,
            'max_shared_prefix_drift': max_prefix_drift[m],
        }
    paired_rt = (
        paired_bootstrap_ci([r - t for r, t in zip(margins['ranking'], margins['terminal'])])
        if n_valid > 0
        else None
    )
    return {
        'counts': counts,
        'models': model_metrics,
        'ranking_minus_terminal_fork_delta': paired_rt,
        'per_task': per_task,
    }


def evaluate_total_score_mse(
    frozen_dev_ids: list[int],
    dev_trajectories: list[dict],
    predictions_by_model: dict[str, dict[int, float]],
    baselines: dict,
) -> dict:
    """Compute total-score MSE and paired bootstrap CIs vs 3 fit-only baselines."""
    frozen_set = set(frozen_dev_ids)
    require(len(dev_trajectories) > 0, 'Empty dev_trajectories')
    sq_err_base: dict[str, list[float]] = {'constant': [], 'turn_count_mean': [], 'ols_turn_count': []}
    sq_err_model: dict[str, list[float]] = {m: [] for m in MODELS}

    for row in dev_trajectories:
        tid = row['task_id']
        require(tid in frozen_set, f'Trajectory task {tid} is not in frozen independent dev list')
        y = float(row['terminal_reward'])
        bp = predict_baselines(baselines, int(row['num_turns']))
        for bname, pred in bp.items():
            sq_err_base[bname].append((pred - y) ** 2)
        for m in MODELS:
            p = float(predictions_by_model[m][tid])
            require(math.isfinite(p), f'Non-finite total prediction for {tid}/{m}')
            sq_err_model[m].append((p - y) ** 2)

    n = len(dev_trajectories)
    base_mse = {b: math.fsum(v) / n for b, v in sq_err_base.items()}
    model_mse = {}
    for m in MODELS:
        mse = math.fsum(sq_err_model[m]) / n
        vs_base = {
            b: paired_bootstrap_ci([em - eb for em, eb in zip(sq_err_model[m], sq_err_base[b])])
            for b in sq_err_base
        }
        model_mse[m] = {'mse': mse, 'n': n, 'minus_baseline_mse': vs_base}

    ranking_minus_terminal = paired_bootstrap_ci(
        [er - et for er, et in zip(sq_err_model['ranking'], sq_err_model['terminal'])]
    )
    return {
        'n_trajectories': n,
        'fit_only_baselines_mse': base_mse,
        'models': model_mse,
        'ranking_minus_terminal_mse': ranking_minus_terminal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-ids', type=Path, required=True)
    parser.add_argument('--input-bundle-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    frozen_ids = json.loads(args.frozen_ids.read_text(encoding='utf-8'))
    bundle = json.loads(args.input_bundle_json.read_text(encoding='utf-8'))
    require(bundle.get('test_authorization') is False, 'Bundle must preserve test_authorization=False')

    baselines = fit_only_baselines(bundle['fit501_rows'])
    pair_preds = {
        m: {(r['task_id'], r['arm']): r for r in bundle['pair_predictions'][m]}
        for m in MODELS
    }
    pair_res = evaluate_fork_pairs(frozen_ids, bundle['pair_records'], pair_preds)
    total_preds = {
        m: {r['task_id']: r['total'] for r in bundle['dev_total_predictions'][m]}
        for m in MODELS
    }
    mse_res = evaluate_total_score_mse(frozen_ids, bundle['dev_trajectories'], total_preds, baselines)

    result = {
        'frozen_ids_sha256': sha256_bytes(args.frozen_ids.read_bytes()),
        'fork_pair_evaluation': pair_res,
        'total_score_evaluation': mse_res,
        'regression_7544_separate': bundle.get('regression_7544'),
        'test_authorization': False,
    }
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
