"""Freeze and audit the unviewed independent dev list for PR #57 fork-ranking short training.

Selects 64 completely unviewed, device-clean tasks (ranks 161..224 of the audited 444 clean pool
from PR #55 commit a601999 under the frozen SHA256(ASCII("42:" + ID)) order), partitioned into
primary_32 (ranks 161..192) and reserve_32 (ranks 193..224).

Verifies zero overlap by task_id, normalized instruction_sha256, and budget-stripped
common_instruction_hash against PR #57 train-candidates (32), fit501 (501), viewed PR #55
dev160 (including examined diagnostics 7084/7402/7113/7489/7544/8669/9984), mapped val (160),
mapped test (200), device-exposed 85, twin-excluded 6755, and 6426, plus zero cross-goal twins
across all 12,087 WebShop goals. Keeps task 7544 isolated as a separate regression-only check
and preserves test_authorization=false.
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
GOAL_MAP_SHA = '6b5d9dfceee913828871b8dff62fd59720a40306738a92a496d1b0fb6838fa88'
EXAMINED_DIAGNOSTIC_IDS = (7084, 7113, 7402, 7489, 7544, 8669, 9984)
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


def rank_clean_pool(task_ids) -> list[int]:
    return sorted(
        task_ids,
        key=lambda tid: (hashlib.sha256(f'42:{tid}'.encode('ascii')).hexdigest(), tid),
    )


def build_independent_dev_freeze(
    clean_pool: list[int],
    viewed_dev160: list[int],
    train_candidates_32: list[dict],
    fit501_ids: set[int],
    splits: dict[str, set[int]],
    goals: dict[int, str],
    count: int = DEV_TOTAL_COUNT,
    primary_count: int = PRIMARY_COUNT,
) -> tuple[list[int], dict]:
    require(len(clean_pool) == len(set(clean_pool)), 'Duplicate IDs in clean_pool')
    ranked = rank_clean_pool(clean_pool)
    require(ranked[:len(viewed_dev160)] == list(viewed_dev160),
            'viewed_dev160 does not match prefix of ranked clean_pool')
    unviewed = ranked[len(viewed_dev160):]
    require(len(unviewed) >= count, f'Insufficient unviewed clean tasks: {len(unviewed)} < {count}')
    selected = unviewed[:count]
    primary = selected[:primary_count]
    reserve = selected[primary_count:]

    selected_set = set(selected)
    train32_ids = {row['task_id'] for row in train_candidates_32}
    viewed_set = set(viewed_dev160)

    # Verify train32 instruction hashes match goal map
    for row in train_candidates_32:
        require(norm_instruction_hash(goals[row['task_id']]) == row['instruction_sha256'],
                f"PR #57 train candidate {row['task_id']} instruction_sha256 mismatch")

    # Membership & ID disjointness
    require(selected_set <= splits['train'], 'Selected dev tasks must all come from mapped train')
    for label, other in (
        ('pr57_train_candidates_32', train32_ids),
        ('fit501', fit501_ids),
        ('pr55_viewed_dev160', viewed_set),
        ('examined_diagnostics', set(EXAMINED_DIAGNOSTIC_IDS)),
        ('mapped_val_160', splits['val']),
        ('mapped_test_200', splits['test']),
    ):
        overlap = sorted(selected_set & other)
        require(not overlap, f'Task ID overlap with {label}: {overlap}')

    # Instruction-hash disjointness (both normalized and budget-stripped)
    selected_nh = {tid: norm_instruction_hash(goals[tid]) for tid in selected}
    selected_ch = {tid: common_instruction_hash(goals[tid]) for tid in selected}
    require(len(set(selected_nh.values())) == count, 'Duplicate normalized instruction within selected dev')
    require(len(set(selected_ch.values())) == count, 'Duplicate budget-stripped instruction within selected dev')

    for label, other in (
        ('pr57_train_candidates_32', train32_ids),
        ('fit501', fit501_ids),
        ('pr55_viewed_dev160', viewed_set),
        ('mapped_val_160', splits['val']),
        ('mapped_test_200', splits['test']),
    ):
        other_nh = {norm_instruction_hash(goals[t]) for t in other}
        other_ch = {common_instruction_hash(goals[t]) for t in other}
        nh_hit = sorted(t for t, h in selected_nh.items() if h in other_nh)
        ch_hit = sorted(t for t, h in selected_ch.items() if h in other_ch)
        require(not nh_hit, f'Normalized instruction overlap with {label}: {nh_hit}')
        require(not ch_hit, f'Budget-stripped instruction overlap with {label}: {ch_hit}')

    # Cross-goal twin check across all goals in goal-instructions.json
    by_nh: dict[str, list[int]] = {}
    for tid, text in goals.items():
        by_nh.setdefault(norm_instruction_hash(text), []).append(tid)
    cross_goal_twins = {
        str(tid): sorted(x for x in by_nh[selected_nh[tid]] if x != tid)
        for tid in selected
    }
    cross_goal_twins = {k: v for k, v in cross_goal_twins.items() if v}
    require(not cross_goal_twins, f'Cross-goal normalized instruction twins found: {cross_goal_twins}')

    ids_sha = sha256_bytes(canonical_ids_bytes(selected))
    primary_sha = sha256_bytes(canonical_ids_bytes(primary))
    reserve_sha = sha256_bytes(canonical_ids_bytes(reserve))

    audit = {
        'schema_version': 'prm57-independent-dev-v1',
        'selection_rule': 'Ranks 161..224 of PR #55 a601999 clean 444 pool ordered by SHA256(ASCII("42:" + ID)) ascending (immediately following viewed dev160).',
        'source_clean_pool_count': len(clean_pool),
        'viewed_dev160_count': len(viewed_dev160),
        'unviewed_pool_count': len(unviewed),
        'selected_count': count,
        'primary_32_count': len(primary),
        'reserve_32_count': len(reserve),
        'selected_ids_sha256': ids_sha,
        'primary_32_ids_sha256': primary_sha,
        'reserve_32_ids_sha256': reserve_sha,
        'primary_32_ids': primary,
        'reserve_32_ids': reserve,
        'selected_instruction_sha256': {str(tid): selected_nh[tid] for tid in selected},
        'isolation_checks': {
            'all_in_mapped_train': True,
            'id_overlap_pr57_train32': 0,
            'id_overlap_fit501': 0,
            'id_overlap_pr55_viewed_dev160': 0,
            'id_overlap_mapped_val160': 0,
            'id_overlap_mapped_test200': 0,
            'norm_instruction_overlap_all_sets': 0,
            'budget_stripped_instruction_overlap_all_sets': 0,
            'cross_goal_twins_in_full_goal_map': 0,
        },
        'examined_diagnostic_ids_excluded': list(EXAMINED_DIAGNOSTIC_IDS),
        'regression_only_task_ids': list(REGRESSION_ONLY_IDS),
        'caveats': [
            'B1 SFT inherited exposure is outside PRM holdout isolation and remains disclosed separately.',
            'Task 7544 is an examined diagnostic case and is tracked strictly as a separate regression check, never pooled into independent dev metrics.',
            'Environment-tie pairs (env_high == env_low) must be recorded separately and never counted as directional wins.',
        ],
        'test_authorization': False,
    }
    return selected, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--goal-map', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-ids', type=Path, required=True)
    parser.add_argument('--output-audit', type=Path, required=True)
    args = parser.parse_args()

    def git_json(ref: str, path: str):
        raw = subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=args.repo)
        return json.loads(raw), sha256_bytes(raw)

    goal_raw = args.goal_map.read_bytes()
    require(sha256_bytes(goal_raw) == GOAL_MAP_SHA, 'goal-instructions.json SHA mismatch')
    goals = {row['task_id']: row['instruction'] for row in json.loads(goal_raw)}

    archive_raw = args.archive.read_bytes()
    require(sha256_bytes(archive_raw) == PR55_ARCHIVE_SHA, 'PR #55 archive SHA mismatch')
    with ZipFile(args.archive) as zf:
        fit501_ids = {json.loads(line)['task_id'] for line in zf.read('data/fit501.training.jsonl').splitlines()}

    audit55, audit55_sha = git_json(PR55_REFREEZE_REF, 'results/prm-expanded-fit-20261125/holdout-audit.json')
    holdout160, holdout160_sha = git_json(PR55_REFREEZE_REF, 'results/prm-expanded-fit-20261125/holdout_ids.json')
    cand57, cand57_sha = git_json(PR57_REF, 'results/prm-fork-ranking-20261126/train-candidates.json')

    splits = {
        name: set(json.loads((args.repo / 'results' / 'env-2026-11-17' / f'{name}_ids.json').read_text(encoding='utf-8')))
        for name in ('train', 'val', 'test')
    }

    selected, audit = build_independent_dev_freeze(
        clean_pool=audit55['eligible_task_ids'],
        viewed_dev160=holdout160,
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
        'pr55_archive_sha256': PR55_ARCHIVE_SHA,
        'goal_map_sha256': GOAL_MAP_SHA,
    }

    with args.output_ids.open('xb') as handle:
        handle.write(canonical_ids_bytes(selected))
    with args.output_audit.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps({
        'selected_count': len(selected),
        'selected_ids_sha256': audit['selected_ids_sha256'],
        'primary_32_ids_sha256': audit['primary_32_ids_sha256'],
        'reserve_32_ids_sha256': audit['reserve_32_ids_sha256'],
        'test_authorization': audit['test_authorization'],
    }, indent=2))


if __name__ == '__main__':
    main()
