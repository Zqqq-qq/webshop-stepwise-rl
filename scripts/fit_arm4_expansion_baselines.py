#!/usr/bin/env python3
"""Fit the three frozen Arm 4 comparison baselines using training rollouts only."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fit(raw_path, inventory_path):
    inventory_bytes = inventory_path.read_bytes()
    inventory = json.loads(inventory_bytes)
    if inventory["schema"] != "arm4-prm-expansion-v1" or inventory["test_authorization"] is not False:
        raise ValueError("Use the frozen expansion inventory with test held back")
    order_file = inventory_path.with_name("train-verified-ids.json")
    order_bytes = order_file.read_bytes()
    raw_bytes = raw_path.read_bytes()
    order_sha, raw_sha = sha256(order_bytes), sha256(raw_bytes)
    if order_sha != inventory["files_sha256"][order_file.name] or raw_sha != inventory["sources"]["raw_sha256"]:
        raise ValueError("Frozen episode order or rollout archive hash mismatch")

    order = json.loads(order_bytes)
    episodes = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
    raw_by_id = {row["task_id"]: row for row in episodes}
    if (len(order) != inventory["verified_train_trajectories"] or len(set(order)) != len(order)
            or len(raw_by_id) != len(episodes) or not set(order) <= raw_by_id.keys()):
        raise ValueError("Invalid training episode coverage")

    by_turn_count = defaultdict(list)
    labels = []
    turns = []
    for task in order:
        row = raw_by_id[task]
        count = len(row["steps"])
        target = float(row["final_reward"]) if row["done"] else 0.0
        status = inventory["task_status"][str(task)]
        if (status["status"] != "eligible" or count != status["turns"] or count < 1
                or not math.isfinite(target) or not 0 <= target <= 1):
            raise ValueError(f"Invalid training target or turn count for task {task}")
        labels.append(target)
        turns.append(count)
        by_turn_count[count].append(target)

    constant, mean_turns = mean(labels), mean(turns)
    variance = math.fsum((n - mean_turns) ** 2 for n in turns)
    covariance = math.fsum((n - mean_turns) * (y - constant) for n, y in zip(turns, labels))
    slope = covariance / variance if variance else 0.0
    return {
        "schema": "arm4-expansion-fit-baselines-v1",
        "fit_trajectories": len(order),
        "feature": "number of logged agent turns, including invalid actions",
        "target": "final_reward if done, zero otherwise",
        "sources_sha256": {"inventory": sha256(inventory_bytes), "train_order": order_sha, "raw_rollouts": raw_sha},
        "constant": constant,
        "turn_count_mean": {str(n): {"n": len(scores), "prediction": mean(scores)}
                            for n, scores in sorted(by_turn_count.items())},
        "turn_count_mean_fallback": constant,
        "turn_count_ols": {"intercept": constant - slope * mean_turns, "slope": slope, "clip": False},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-rollouts", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = fit(args.raw_rollouts, args.inventory)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(baseline, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
