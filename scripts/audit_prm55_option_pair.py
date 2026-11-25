"""Independent read-only verifier for PR #55/#56 task 7084 option-pair diagnostic.

Two modes:
- ``baseline``: verifies the archived PR #55 evidence bundle (SHA-256 + 61/61 SHA256SUMS),
  extracts the frozen task 7084 reference prefix/scores, verifies Zqqq's zero-reward purchase
  and 3-step full-reward reference numbers, and records fallback task 7402.
- ``pair``: independently verifies How's device replay and step-126 scoring outputs
  (environment.json, b/a.rollout.jsonl, b/a.predictions.json, pair-comparison.json)
  against the archived task 7084 prefix, stage-126 manifest identity, controlled wording,
  shared scoring indices, and step-by-step B-A score pairing.

Never executes WebShop rollouts, GPU inference, or reserved test (test_authorization=false).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile


ARCHIVE_SHA = '0834998fe986f450f40721f0864b51351c68743d5b2e5979334cffde28ca44b4'
TASK_ID = 7084
FALLBACK_TASK_ID = 7402
OPTIONS = ('b', 'a')
ZERO_PURCHASE_IDS = (7544, 8669, 9984)
PREFIX_TOL = 1e-5
SUM_TOL = 1e-6


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_archived_bundle(archive_path: Path, verify_all_sums: bool = True) -> dict:
    raw = archive_path.read_bytes()
    actual_sha = sha256_bytes(raw)
    require(actual_sha == ARCHIVE_SHA, f'PR #55 archive SHA mismatch: {actual_sha}')
    with ZipFile(archive_path) as zf:
        sums_lines = [line.strip() for line in zf.read('SHA256SUMS').decode('utf-8').splitlines() if line.strip()]
        verified_count = 0
        if verify_all_sums:
            for line in sums_lines:
                expected, name = line.split(None, 1)
                require(sha256_bytes(zf.read(name)) == expected, f'SHA256SUMS mismatch: {name}')
                verified_count += 1
        episodes = {row['task_id']: row for row in map(json.loads, zf.read('rollout/dev160.jsonl').splitlines())}
        pred_doc = json.loads(zf.read('eval/new-dev-predictions.json'))
        preds = {row['task_id']: row for row in pred_doc['predictions']}
        manifest = json.loads(zf.read('training/stage126/prm-manifest.json'))
    return {
        'archive_name': archive_path.name,
        'archive_size_bytes': len(raw),
        'archive_sha256': actual_sha,
        'verified_sha256sums_count': verified_count if verify_all_sums else len(sums_lines),
        'episodes': episodes,
        'pred_doc': pred_doc,
        'predictions': preds,
        'stage126_manifest': manifest,
    }


def extract_baseline_summary(bundle: dict) -> dict:
    episodes = bundle['episodes']
    preds = bundle['predictions']
    manifest = bundle['stage126_manifest']

    require(TASK_ID in episodes and TASK_ID in preds, f'Missing archived task {TASK_ID}')
    ep = episodes[TASK_ID]
    pr = preds[TASK_ID]
    require(ep['success'] is True and len(ep['steps']) == 4, 'Archived task 7084 must be a 4-step success')
    require([s['action'] for s in ep['steps'][2:]] == ['click[b]', 'click[buy now]'],
            'Unexpected archived suffix for task 7084')
    require(set(OPTIONS) <= set(ep['steps'][2]['available']['clickables']),
            'Archived task 7084 step 2 missing a/b clickables')
    require(len(pr['step_scores']) == len(pr['turn_end_indices']) == 4, 'Invalid archived 7084 prediction shape')
    require(math.isclose(math.fsum(pr['step_scores']), pr['total'], abs_tol=SUM_TOL),
            'Archived 7084 step_scores sum mismatch')

    zero_purchases = {}
    for tid in ZERO_PURCHASE_IDS:
        require(tid in episodes and tid in preds, f'Missing zero-purchase reference task {tid}')
        zero_purchases[str(tid)] = {
            'final_reward': episodes[tid]['final_reward'],
            'actions': [s['action'] for s in episodes[tid]['steps']],
            'open_product_step_score': preds[tid]['step_scores'][1],
            'step_scores': preds[tid]['step_scores'],
            'prm_total': preds[tid]['total'],
        }

    three_step_open_scores = [
        preds[tid]['step_scores'][1]
        for tid, row in sorted(episodes.items())
        if tid in preds
        and row['success'] is True
        and len(row['steps']) == 3
        and row['steps'][0]['action'].startswith('search[')
        and row['steps'][2]['action'] == 'click[buy now]'
    ]
    require(len(three_step_open_scores) > 0, 'No 3-step full-reward trajectories found')

    fb_ep = episodes[FALLBACK_TASK_ID]
    fb_pr = preds[FALLBACK_TASK_ID]

    return {
        'mode': 'baseline',
        'archive': {
            'name': bundle['archive_name'],
            'size_bytes': bundle['archive_size_bytes'],
            'sha256': bundle['archive_sha256'],
            'verified_sha256sums_count': bundle['verified_sha256sums_count'],
        },
        'stage126_identity': {
            'source_commit': manifest['identity']['source_commit'],
            'base_revision': manifest['identity']['base_revision'],
            'template_sha256': manifest['identity']['template_sha256'],
            'head_sha256': manifest['files_sha256']['head.pt'],
            'adapter_model_sha256': manifest['files_sha256']['adapter/adapter_model.safetensors'],
            'adapter_config_sha256': manifest['files_sha256']['adapter/adapter_config.json'],
            'manifest_sha256': canonical_sha256(manifest),
        },
        'task_7084_reference': {
            'task_id': TASK_ID,
            'instruction': ep['instruction'],
            'goal_fingerprint': ep['goal_fingerprint'],
            'final_reward': ep['final_reward'],
            'prefix_steps_0_1_sha256': canonical_sha256(ep['steps'][:2]),
            'step_2_observation_sha256': canonical_sha256(ep['steps'][2]['observation']),
            'step_2_available_sha256': canonical_sha256(ep['steps'][2]['available']),
            'step_2_clickables': ep['steps'][2]['available']['clickables'],
            'actions': [s['action'] for s in ep['steps']],
            'archived_turn_end_indices': pr['turn_end_indices'],
            'archived_step_scores': pr['step_scores'],
            'archived_prm_total': pr['total'],
        },
        'zero_reward_purchase_crosscheck': {
            'cases': zero_purchases,
            'three_step_full_reward_count': len(three_step_open_scores),
            'three_step_full_reward_open_product_mean': math.fsum(three_step_open_scores) / len(three_step_open_scores),
        },
        'fallback_task_7402': {
            'task_id': FALLBACK_TASK_ID,
            'instruction': fb_ep['instruction'],
            'final_reward': fb_ep['final_reward'],
            'actions': [s['action'] for s in fb_ep['steps']],
            'archived_turn_end_indices': fb_pr['turn_end_indices'],
            'archived_step_scores': fb_pr['step_scores'],
            'archived_prm_total': fb_pr['total'],
        },
        'live_pair_replay_verified': False,
        'test_authorization': False,
    }


def audit_pair_artifacts(archived_episode: dict, archived_pred: dict, manifest: dict, artifacts: dict) -> dict:
    """Independently verify prefix consistency and score pairing from saved replay/score JSON dicts.

    ``artifacts`` must contain keys:
      ``environment``, ``b_rollout``, ``a_rollout``, ``b_predictions``, ``a_predictions``,
      and optionally ``pair_comparison``.
    """
    env_doc = artifacts['environment']
    arms = {'b': artifacts['b_rollout'], 'a': artifacts['a_rollout']}
    preds = {'b': artifacts['b_predictions'], 'a': artifacts['a_predictions']}

    # 1. Verify environment.json
    require(env_doc.get('task_id') == TASK_ID, 'environment.json task_id mismatch')
    require(env_doc.get('source_archive_sha256') == ARCHIVE_SHA, 'environment.json source_archive_sha256 mismatch')
    require(env_doc.get('test_authorization') is False, 'environment.json must have test_authorization=False')
    health = env_doc.get('health', {})
    require(
        health.get('ready') is True
        and health.get('goal_fingerprint') == archived_episode['goal_fingerprint']
        and health.get('alignment_mismatches') == 0
        and health.get('split_status', {}).get('verified') is True,
        'environment.json health or goal_fingerprint check failed',
    )

    # 2. Verify both rollouts against archived task 7084 AND against each other
    for option in OPTIONS:
        arm = arms[option]
        require(arm.get('task_id') == TASK_ID, f'Arm {option} task_id mismatch')
        require(arm.get('instruction') == archived_episode['instruction'], f'Arm {option} instruction mismatch')
        require(arm.get('goal_fingerprint') == archived_episode['goal_fingerprint'],
                f'Arm {option} goal_fingerprint mismatch')
        require(arm.get('done') is True and arm.get('termination') == 'purchase',
                f'Arm {option} must terminate via purchase')
        require(len(arm.get('steps', [])) == 4, f'Arm {option} must have exactly 4 steps')
        reward = arm.get('final_reward')
        require(type(reward) in (float, int) and math.isfinite(reward) and 0 <= reward <= 1,
                f'Arm {option} invalid final_reward')
        require(env_doc.get('terminal_scores', {}).get(option) == reward,
                f'environment.json terminal_scores[{option}] mismatch')

        # Prefix steps 0-1 must match archived 7084 verbatim (observations, available, responses, actions, rewards)
        require(arm['steps'][:2] == archived_episode['steps'][:2],
                f'Arm {option} shared prefix steps 0-1 differ from archived task 7084')
        # Pre-fork step 2 observation and available clickables must match archived 7084 verbatim
        require(arm['steps'][2]['observation'] == archived_episode['steps'][2]['observation'],
                f'Arm {option} pre-fork step 2 observation differs from archived task 7084')
        require(arm['steps'][2]['available'] == archived_episode['steps'][2]['available'],
                f'Arm {option} pre-fork step 2 available differs from archived task 7084')
        # Controlled post-fork action & response wording (no extra Thought text)
        require(arm['steps'][2]['action'] == f'click[{option}]', f'Arm {option} step 2 action mismatch')
        require(arm['steps'][2]['response'] == f'Action: click[{option}]',
                f'Arm {option} step 2 response wording drifted from controlled template')
        require(arm['steps'][3]['action'] == 'click[buy now]', f'Arm {option} step 3 action mismatch')
        require(arm['steps'][3]['response'] == 'Action: click[buy now]',
                f'Arm {option} step 3 response wording drifted from controlled template')

    require(arms['b']['steps'][:2] == arms['a']['steps'][:2], 'Arms differ on shared prefix steps 0-1')
    require(arms['b']['steps'][2]['observation'] == arms['a']['steps'][2]['observation'],
            'Arms differ on pre-fork step 2 observation')
    require(arms['b']['steps'][3]['response'] == arms['a']['steps'][3]['response'],
            'Arms differ on controlled step 3 response')

    # 3. Verify step-126 predictions & prefix/score pairing
    summary = {}
    for option in OPTIONS:
        res = preds[option]
        require(res.get('identity') == manifest['identity'], f'Arm {option} scorer identity mismatch')
        require(res.get('input_schema') == 'rollout', f'Arm {option} input_schema must be rollout')
        require(res.get('total_samples') == 1 and not res.get('rejected'), f'Arm {option} coverage mismatch')
        require(res.get('test_authorization') is False, f'Arm {option} predictions must have test_authorization=False')
        rows = res.get('predictions', [])
        require(len(rows) == 1, f'Arm {option} expected exactly 1 prediction row')
        row = rows[0]
        scores = row['step_scores']
        indices = row['turn_end_indices']
        require(row.get('task_id') == TASK_ID, f'Arm {option} prediction task_id mismatch')
        require(len(scores) == len(indices) == 4, f'Arm {option} expected 4 step_scores and 4 turn_end_indices')
        require(all(type(x) in (float, int) and math.isfinite(x) for x in scores),
                f'Arm {option} non-finite step_score')
        require(type(row['total']) in (float, int) and math.isfinite(row['total']),
                f'Arm {option} non-finite total')
        require(math.isclose(math.fsum(scores), row['total'], abs_tol=SUM_TOL),
                f'Arm {option} step_scores sum disagrees with total')
        summary[option] = {
            'environment_terminal_score': arms[option]['final_reward'],
            'scores': scores,
            'turn_end_indices': indices,
            'prm_total': row['total'],
        }

    # Shared scoring indices must match both between arms AND with archived task 7084
    require(summary['b']['turn_end_indices'][:2] == summary['a']['turn_end_indices'][:2],
            'Shared prefix turn_end_indices differ between arms')
    require(summary['b']['turn_end_indices'][:2] == archived_pred['turn_end_indices'][:2],
            'Shared prefix turn_end_indices differ from archived task 7084')

    delta_by_step = [b - a for b, a in zip(summary['b']['scores'], summary['a']['scores'])]
    env_delta = summary['b']['environment_terminal_score'] - summary['a']['environment_terminal_score']
    prm_total_delta = summary['b']['prm_total'] - summary['a']['prm_total']
    shared_prefix_ok = all(abs(x) <= PREFIX_TOL for x in delta_by_step[:2])
    valid_contrast = env_delta > 0

    b_vs_archived_prefix = [
        summary['b']['scores'][i] - archived_pred['step_scores'][i] for i in range(2)
    ]
    a_vs_archived_prefix = [
        summary['a']['scores'][i] - archived_pred['step_scores'][i] for i in range(2)
    ]

    # 4. Cross-check pair-comparison.json if provided
    pair_doc = artifacts.get('pair_comparison')
    pair_doc_verified = False
    if pair_doc is not None:
        require(pair_doc.get('test_authorization') is False, 'pair-comparison.json test_authorization must be False')
        require(math.isclose(pair_doc['environment_b_minus_a'], env_delta, abs_tol=1e-9),
                'pair-comparison.json environment_b_minus_a mismatch')
        require(pair_doc['valid_environment_contrast'] is valid_contrast,
                'pair-comparison.json valid_environment_contrast mismatch')
        require(len(pair_doc['prm_b_minus_a_by_step']) == 4 and all(
            math.isclose(x, y, abs_tol=1e-9) for x, y in zip(pair_doc['prm_b_minus_a_by_step'], delta_by_step)
        ), 'pair-comparison.json prm_b_minus_a_by_step mismatch')
        require(math.isclose(pair_doc['prm_b_minus_a_total'], prm_total_delta, abs_tol=1e-9),
                'pair-comparison.json prm_b_minus_a_total mismatch')
        require(pair_doc['shared_prefix_within_1e_5'] is shared_prefix_ok,
                'pair-comparison.json shared_prefix_within_1e_5 mismatch')
        pair_doc_verified = True

    return {
        'mode': 'pair',
        'task_id': TASK_ID,
        'prefix_verified_against_archive': True,
        'controlled_post_fork_wording_verified': True,
        'shared_turn_end_indices_0_1': summary['b']['turn_end_indices'][:2],
        'shared_prefix_b_minus_a': delta_by_step[:2],
        'shared_prefix_within_1e_5': shared_prefix_ok,
        'b_prefix_minus_archived': b_vs_archived_prefix,
        'a_prefix_minus_archived': a_vs_archived_prefix,
        'arms': summary,
        'environment_b_minus_a': env_delta,
        'valid_environment_contrast': valid_contrast,
        'option_step_b_minus_a': delta_by_step[2],
        'buy_step_b_minus_a': delta_by_step[3],
        'prm_b_minus_a_by_step': delta_by_step,
        'prm_b_minus_a_total': prm_total_delta,
        'pair_comparison_json_verified': pair_doc_verified,
        'live_pair_replay_verified': True,
        'test_authorization': False,
    }


def load_pair_dir(run_dir: Path) -> dict:
    def read_json(name: str) -> dict:
        return json.loads((run_dir / name).read_text(encoding='utf-8'))

    artifacts = {
        'environment': read_json('environment.json'),
        'b_rollout': read_json('b.rollout.jsonl'),
        'a_rollout': read_json('a.rollout.jsonl'),
        'b_predictions': read_json('b.predictions.json'),
        'a_predictions': read_json('a.predictions.json'),
    }
    pair_path = run_dir / 'pair-comparison.json'
    if pair_path.exists():
        artifacts['pair_comparison'] = read_json('pair-comparison.json')
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('baseline', 'pair'), required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    bundle = load_archived_bundle(args.archive, verify_all_sums=True)
    if args.mode == 'baseline':
        result = extract_baseline_summary(bundle)
    else:
        if args.run_dir is None:
            parser.error('--run-dir is required for --mode pair')
        artifacts = load_pair_dir(args.run_dir)
        result = audit_pair_artifacts(
            bundle['episodes'][TASK_ID],
            bundle['predictions'][TASK_ID],
            bundle['stage126_manifest'],
            artifacts,
        )

    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps({
        'mode': result['mode'],
        'live_pair_replay_verified': result['live_pair_replay_verified'],
        'test_authorization': result['test_authorization'],
        'output': str(args.output),
    }, indent=2))


if __name__ == '__main__':
    main()
