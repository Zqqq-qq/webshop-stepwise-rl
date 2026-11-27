#!/usr/bin/env python3
"""Verify the frozen orders and write per-update ID files for a full-pool run."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(output, frozen=None):
    frozen = Path(__file__).resolve().parent if frozen is None else frozen.resolve()
    repo = frozen.parents[1]
    manifest = json.loads((frozen / "manifest.json").read_text(encoding="utf-8"))
    if manifest["test_authorization"]:
        raise ValueError("This schedule only permits the frozen training and validation splits")
    splits = {}
    for split in ("train", "val", "test"):
        entry = manifest["sources"][split]
        contents = (repo / entry["path"]).read_bytes()
        if sha(contents) != entry["sha256"]:
            raise ValueError(f"Frozen {split} split changed")
        values = json.loads(contents)
        if len(values) != entry["count"] or len(set(values)) != len(values):
            raise ValueError(f"Frozen {split} split has invalid coverage")
        splits[split] = set(values)

    holdout = set()
    for entry in manifest.get("additional_holdouts", []):
        contents = (frozen / entry["path"]).read_bytes()
        ids = json.loads(contents)
        if sha(contents) != entry["sha256"] or len(ids) != entry["count"] or len(set(ids)) != len(ids):
            raise ValueError(f"Frozen holdout changed: {entry['path']}")
        if holdout.intersection(ids):
            raise ValueError("Additional holdouts overlap")
        holdout.update(ids)

    schedules = {}
    common_pool = None
    budget = manifest["budget"]
    limit = budget["maximum_episodes_per_update"]
    for seed in manifest["seeds"]:
        entry = manifest["schedules"][str(seed)]
        contents = (frozen / entry["path"]).read_bytes()
        if sha(contents) != entry["sha256"]:
            raise ValueError(f"Frozen order for seed {seed} changed")
        ids = json.loads(contents)
        pool = set(ids)
        if len(ids) != len(pool) or len(pool) != budget["pool_size"] or len(ids) != entry["episodes"]:
            raise ValueError(f"Seed {seed} has incomplete or duplicate IDs")
        if not pool <= splits["train"] or pool & (splits["val"] | splits["test"]):
            raise ValueError(f"Seed {seed} contains out-of-split IDs")
        if pool & holdout:
            raise ValueError(f"Seed {seed} overlaps with an additional frozen holdout")
        if common_pool is not None and pool != common_pool:
            raise ValueError("Seeds must traverse the same task pool")
        common_pool = pool
        batches = [ids[i:i + limit] for i in range(0, len(ids), limit)]
        if len(batches) != entry["updates"] or len(batches[-1]) != entry["last_batch_size"]:
            raise ValueError(f"Seed {seed} update budget changed")
        schedules[seed] = batches

    output.mkdir(parents=True, exist_ok=False)
    index = {}
    for seed, batches in schedules.items():
        records = []
        for step, ids in enumerate(batches, 1):
            relative = Path(f"seed-{seed}") / f"update-{step:03d}-ids.json"
            target = output / relative
            target.parent.mkdir(exist_ok=True)
            contents = (json.dumps(ids, indent=2) + "\n").encode()
            target.write_bytes(contents)
            records.append({"update": step, "path": str(relative), "sha256": sha(contents)})
        index[str(seed)] = records
    (output / "batch-index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path)
    args = parser.parse_args()
    prepare(args.output, args.frozen_dir)
