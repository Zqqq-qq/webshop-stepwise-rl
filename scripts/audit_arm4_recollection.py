#!/usr/bin/env python3
"""Check frozen collection coverage and emit an ordered continuation list."""
import argparse
import json
from pathlib import Path

from webshop_lab.common import file_hash
from webshop_lab.evaluation import inspect_run, load_episodes


PLAN_FILES = {
    "train": "train-recollection-candidates-ids.json",
    "dev-primary": "fresh-dev-primary-64-ids.json",
    "dev-reserve": "fresh-dev-reserve-32-ids.json",
}


def audit(inventory, group, shards, config):
    frozen = json.loads(inventory.read_text(encoding="utf-8"))
    if frozen["schema"] != "arm4-prm-expansion-v1" or frozen["test_authorization"]:
        raise ValueError("Use the frozen PRM inventory with test held back")
    plan = inventory.with_name(PLAN_FILES[group])
    plan_sha = file_hash(plan)
    if plan_sha != frozen["files_sha256"][plan.name]:
        raise ValueError("Frozen collection plan SHA changed")
    expected = json.loads(plan.read_text(encoding="utf-8"))
    if len(set(expected)) != len(expected):
        raise ValueError("Frozen plan has duplicate IDs")
    order = {task: i for i, task in enumerate(expected)}
    completed = set()
    reports = []
    fingerprint = None
    for rollout, metadata, shard_ids in shards:
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        ids = json.loads(shard_ids.read_text(encoding="utf-8"))
        if (not ids or any(type(task) is not int or task not in order for task in ids)
                or len(set(ids)) != len(ids) or sorted(ids, key=order.__getitem__) != ids):
            raise ValueError(f"Shard plan is outside {group} or changes its frozen order")
        if (meta["selected_ids"] != ids or meta["ids_sha256"] != file_hash(shard_ids)
                or meta["repeats"] != 1):
            raise ValueError("Shard does not identify the supplied ID file with one repeat")
        result = inspect_run(rollout, metadata, config)
        if not result["provenance_ok"] or result["status"] not in ("complete", "partial"):
            raise ValueError(f"Invalid rollout identity or coverage: {result['summary']}")
        if fingerprint is not None and result["goal_fingerprint"] != fingerprint:
            raise ValueError("Collection shards use different environment goal fingerprints")
        fingerprint = result["goal_fingerprint"]
        actual = [row["task_id"] for row in load_episodes(rollout)]
        if actual != ids[:len(actual)] or completed.intersection(actual):
            raise ValueError("Shard is not an ordered prefix or reuses previously completed IDs")
        completed.update(actual)
        reports.append({"rollout_sha256": file_hash(rollout), "meta_sha256": file_hash(metadata),
                        "ids_sha256": file_hash(shard_ids), "status": result["status"],
                        "tasks": len(actual)})

    missing = [task for task in expected if task not in completed]
    return {"group": group, "plan_sha256": plan_sha, "expected": len(expected),
            "completed": len(completed), "complete": not missing, "missing_ids": missing,
            "goal_fingerprint": fingerprint, "shards": reports}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--group", choices=PLAN_FILES, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--shard", nargs=3, type=Path, metavar=("JSONL", "META", "IDS"),
                   action="append", default=[])
    p.add_argument("--remaining-ids", type=Path)
    args = p.parse_args()
    report = audit(args.inventory, args.group, args.shard, args.config)
    if args.remaining_ids:
        with args.remaining_ids.open("x", encoding="utf-8") as f:
            json.dump(report["missing_ids"], f, indent=2)
            f.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
