"""Freeze and audit the brand-new unviewed dev list for PR #59 (Arm 4 vs stage126) re-verification.

Excludes ALL training sets (fit501, PR #57 train-candidates 32), ALL previously viewed dev sets
(PR #55 dev160, PR #57 independent-dev 64 verified against prm57-independent-dev-evaluation-20261126.zip),
mapped val160, mapped test200, device-exposed 85, twin-excluded 6755, 6426, the 3 remaining cross-goal
normalized-instruction twins (7610, 9085, 11501), AND any budget-stripped instruction collision with
training/viewed/val/test sets.

Deterministically freezes 64 unviewed tasks (primary_32 + reserve_32) under SHA256(ASCII("42:" + ID)),
preserving test_authorization=false.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from zipfile import ZipFile


PR55_REFREEZE_REF = 'a601999aa42f67d68fdc041f4b84a8bf5d682593'
PR57_REF = 'f34c880bb8f2e7dabd3e845103849090579f36e2'
PR55_ARCHIVE_SHA = '0834998fe986f450f40721f0864b51351c68743d5b2e5979334cffde28ca44b4'
PR57_DEV_ZIP_SHA = '07a36cbf4b26ff0f66faf3aff919f07d15e65d7890c8d2eea33828bc14bfbb9e'
GOAL_MAP_SHA = '6b5d9dfceee913828871b8dff62fd59720a40306738a92a496d1b0fb6838fa88'
REGRESSION_ONLY_IDS = (7544,)
DEV_TOTAL_COUNT = 64
PRIMARY_COUNT = 32


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def norm_instruction_hash(text: str) -> str:
    cleaned = text.strip()
    if cleaned.lower().startswith('instruction:'):
        cleaned = cleaned[len('instruction:'):]
    cleaned = ' '.join(cleaned.lower().split())
    return sha256_bytes(cleaned.encode('utf-8'))


def common_instruction_hash(text: str) -> str:
    cleaned = ' '.join(text.lower().split()).rstrip('.')
    cleaned = re.sub(r',?\s*and price lower than [\d.]+ dollars\.?$', '', cleaned).strip().rstrip('.')
    return sha256_bytes(cleaned.encode('utf-8'))


def canonical_ids_bytes(ids) -> bytes:
    return (json.dumps(list(ids), indent=2) + '\n').encode('utf-8')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def rank_pool(task_ids) -> list[int]:
    return sorted(
        task_ids,
        key=lambda tid: (hashlib.sha256(f'42:{tid}'.encode('ascii')).hexdigest(), tid),
    )


def build_pr59_reverify_dev(
    clean_pool_444: list[int],
    viewed_pr55_160: list[int],
    viewed_pr57_64: list[int],
    train_candidates_32: list[dict],
    fit501_ids: set[int],
    splits: dict[str, set[int]],
    goals: dict[int, str],
    count: int = DEV_TOTAL_COUNT,
    primary_count: int = PRIMARY_COUNT,
) -> tuple[list[int], list[int], list[int], dict]:
    require(len(clean_pool_444) == len(set(clean_pool_444)), 'Duplicate IDs in clean_pool_444')
    ranked = rank_pool(clean_pool_444)
    n55 = len(viewed_pr55_160)
    n57 = len(viewed_pr57_64)
    require(ranked[:n55] == list(viewed_pr55_160), 'PR #55 dev160 prefix mismatch')
    require(ranked[n55:n55 + n57] == list(viewed_pr57_64), 'PR #57 dev64 slice mismatch')

    remaining = ranked[n55 + n57:]
    train32_ids = {row['task_id'] for row in train_candidates_32}

    # Build full-universe normalized instruction twin index
    by_nh: dict[str, list[int]] = {}
    for tid, txt in goals.items():
        by_nh.setdefault(norm_instruction_hash(txt), []).append(tid)

    excluded_sets = {
        'fit501': fit501_ids,
        'pr57_train_candidates_32': train32_ids,
        'pr55_viewed_dev160': set(viewed_pr55_160),
        'pr57_viewed_dev64': set(viewed_pr57_64),
        'mapped_val_160': splits['val'],
        'mapped_test_200': splits['test'],
    }
    excluded_nh = {norm_instruction_hash(goals[t]) for s in excluded_sets.values() for t in s}
    excluded_ch = {common_instruction_hash(goals[t]) for s in excluded_sets.values() for t in s}

    removed_twins: dict[str, list[int]] = {}
    removed_ch_collisions: list[int] = []
    removed_internal_ch_dups: list[int] = []
    seen_ch: set[str] = set()
    strict_pool: list[int] = []

    for tid in remaining:
        nh_val = norm_instruction_hash(goals[tid])
        ch_val = common_instruction_hash(goals[tid])
        twins = sorted(x for x in by_nh[nh_val] if x != tid)
        if twins:
            removed_twins[str(tid)] = twins
            continue
        if nh_val in excluded_nh or ch_val in excluded_ch:
            removed_ch_collisions.append(tid)
            continue
        if ch_val in seen_ch:
            removed_internal_ch_dups.append(tid)
            continue
        seen_ch.add(ch_val)
        strict_pool.append(tid)

    require(len(strict_pool) >= count, f'Insufficient strictly isolated unviewed tasks: {len(strict_pool)} < {count}')
    selected = strict_pool[:count]
    primary = selected[:primary_count]
    reserve = selected[primary_count:]

    selected_set = set(selected)
    require(selected_set <= splits['train'], 'Selected tasks must belong to mapped train')
    for label, other in excluded_sets.items():
        require(not (selected_set & other), f'Task ID overlap with {label}')
        other_nh = {norm_instruction_hash(goals[t]) for t in other}
        other_ch = {common_instruction_hash(goals[t]) for t in other}
        require(not ({norm_instruction_hash(goals[t]) for t in selected} & other_nh), f'nh overlap with {label}')
        require(not ({common_instruction_hash(goals[t]) for t in selected} & other_ch), f'ch overlap with {label}')

    audit = {
        'schema_version': 'prm59-reverify-dev-v1',
        'selection_rule': (
            'Deterministic SHA256(ASCII("42:" + ID)) ascending over PR #55 a601999 clean 444 pool, '
            'excluding PR #55 dev160, PR #57 dev64, cross-goal twins in all 12,087 goals, and any '
            'budget-stripped instruction collisions with fit501/train32/dev160/dev64/val160/test200.'
        ),
        'pool_accounting': {
            'pr55_clean_pool_444': len(clean_pool_444),
            'excluded_pr55_viewed_dev160': n55,
            'excluded_pr57_viewed_dev64': n57,
            'remaining_unviewed_raw': len(remaining),
            'excluded_cross_goal_twins': removed_twins,
            'excluded_budget_stripped_collisions': removed_ch_collisions,
            'excluded_internal_budget_stripped_dups': removed_internal_ch_dups,
            'strictly_clean_unviewed_pool': len(strict_pool),
            'selected_total': len(selected),
            'primary_32_count': len(primary),
            'reserve_32_count': len(reserve),
        },
        'selected_ids_sha256': sha256_bytes(canonical_ids_bytes(selected)),
        'primary_32_ids_sha256': sha256_bytes(canonical_ids_bytes(primary)),
        'reserve_32_ids_sha256': sha256_bytes(canonical_ids_bytes(reserve)),
        'primary_32_ids': primary,
        'reserve_32_ids': reserve,
        'selected_instruction_sha256': {str(tid): norm_instruction_hash(goals[tid]) for tid in selected},
        'isolation_checks': {
            'all_in_mapped_train': True,
            'id_overlap_all_excluded_sets': 0,
            'norm_instruction_overlap_all_excluded_sets': 0,
            'budget_stripped_instruction_overlap_all_excluded_sets': 0,
            'cross_goal_twins_in_12087': 0,
            'unique_norm_instructions': len({norm_instruction_hash(goals[t]) for t in selected}),
            'unique_budget_stripped_instructions': len({common_instruction_hash(goals[t]) for t in selected}),
        },
        'regression_only_task_ids': list(REGRESSION_ONLY_IDS),
        'caveats': [
            'B1 SFT inherited exposure is outside PRM holdout isolation and remains disclosed separately.',
            'Task 7544 is an examined diagnostic case and is reported strictly in a separate regression section.',
            'Pre-click visibility labels (V+ / U / V-) must be logged blindly before viewing model scores and kept separate from environment terminal reward contrast.',
            'Environment ties (env_high == env_low), too_long (>2048), no_replayable_fork, and technical_error must be tallied separately with exact count conservation.',
        ],
        'test_authorization': False,
    }
    return selected, primary, reserve, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--goal-map', type=Path, required=True)
    parser.add_argument('--pr55-archive', type=Path, required=True)
    parser.add_argument('--pr57-dev-zip', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()

    def git_json(ref: str, path: str):
        raw = subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=args.repo)
        return json.loads(raw), sha256_bytes(raw)

    goal_raw = args.goal_map.read_bytes()
    require(sha256_bytes(goal_raw) == GOAL_MAP_SHA, 'goal-instructions.json SHA mismatch')
    goals = {row['task_id']: row['instruction'] for row in json.loads(goal_raw)}

    pr55_raw = args.pr55_archive.read_bytes()
    require(sha256_bytes(pr55_raw) == PR55_ARCHIVE_SHA, 'PR #55 archive SHA mismatch')
    with ZipFile(args.pr55_archive) as zf:
        fit501_ids = {json.loads(line)['task_id'] for line in zf.read('data/fit501.training.jsonl').splitlines()}

    pr57_dev_raw = args.pr57_dev_zip.read_bytes()
    require(sha256_bytes(pr57_dev_raw) == PR57_DEV_ZIP_SHA, 'PR #57 dev ZIP SHA mismatch')
    with ZipFile(args.pr57_dev_zip) as zf:
        pr57_dev_ids_bytes = zf.read('frozen/independent-dev-ids.json')
        viewed_pr57_64 = json.loads(pr57_dev_ids_bytes)

    audit55, audit55_sha = git_json(PR55_REFREEZE_REF, 'results/prm-expanded-fit-20261125/holdout-audit.json')
    holdout160, holdout160_sha = git_json(PR55_REFREEZE_REF, 'results/prm-expanded-fit-20261125/holdout_ids.json')
    cand57, cand57_sha = git_json(PR57_REF, 'results/prm-fork-ranking-20261126/train-candidates.json')

    splits = {
        name: set(json.loads((args.repo / 'results' / 'env-2026-11-17' / f'{name}_ids.json').read_text(encoding='utf-8')))
        for name in ('train', 'val', 'test')
    }

    selected, primary, reserve, audit = build_pr59_reverify_dev(
        clean_pool_444=audit55['eligible_task_ids'],
        viewed_pr55_160=holdout160,
        viewed_pr57_64=viewed_pr57_64,
        train_candidates_32=cand57['tasks'],
        fit501_ids=fit501_ids,
        splits=splits,
        goals=goals,
    )
    audit['provenance'] = {
        'pr55_refreeze_commit': PR55_REFREEZE_REF,
        'pr55_holdout_audit_sha256': audit55_sha,
        'pr55_holdout_ids_sha256': holdout160_sha,
        'pr57_commit': PR57_REF,
        'pr57_train_candidates_sha256': cand57_sha,
        'pr57_dev_zip_sha256': PR57_DEV_ZIP_SHA,
        'pr57_dev_ids_sha256': sha256_bytes(pr57_dev_ids_bytes),
        'pr55_archive_sha256': PR55_ARCHIVE_SHA,
        'goal_map_sha256': GOAL_MAP_SHA,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fname, ids in (
        ('prm59-reverify-dev-ids.json', selected),
        ('prm59-reverify-dev-primary_32-ids.json', primary),
        ('prm59-reverify-dev-reserve_32-ids.json', reserve),
    ):
        with (args.out_dir / fname).open('xb') as handle:
            handle.write(canonical_ids_bytes(ids))
    with (args.out_dir / 'prm59-reverify-dev-audit.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')

    print(json.dumps({
        'strictly_clean_unviewed_pool': audit['pool_accounting']['strictly_clean_unviewed_pool'],
        'excluded_cross_goal_twins': list(audit['pool_accounting']['excluded_cross_goal_twins'].keys()),
        'excluded_budget_stripped_collisions': audit['pool_accounting']['excluded_budget_stripped_collisions'],
        'selected_ids_sha256': audit['selected_ids_sha256'],
        'primary_32_ids_sha256': audit['primary_32_ids_sha256'],
        'reserve_32_ids_sha256': audit['reserve_32_ids_sha256'],
        'test_authorization': audit['test_authorization'],
    }, indent=2))


if __name__ == '__main__':
    main()
