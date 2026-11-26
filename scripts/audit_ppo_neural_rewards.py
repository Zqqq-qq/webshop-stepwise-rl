#!/usr/bin/env python3
"""Audit high neural rewards on abandoned shopping episodes and wrong purchases."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Reward evidence must contain finite numeric scores")
    return value


def audit(rows, threshold):
    if not rows:
        raise ValueError("No trajectories to audit")
    counts, flagged = Counter(), []
    for row in rows:
        if type(row["task_id"]) is not int or type(row["purchased"]) is not bool:
            raise ValueError("Missing canonical task ID or purchase status")
        terminal = finite(row["environment_terminal"])
        if not 0 <= terminal <= 1 or (not row["purchased"] and terminal != 0):
            raise ValueError("Inconsistent environment terminal reward")
        rewards = row["neural_rewards"]
        if not rewards:
            raise ValueError("No per-turn neural rewards")
        total = math.fsum(map(finite, rewards))
        if row["purchased"] and terminal > 0:
            counts["positive_purchase"] += 1
            continue
        category = "zero_purchase" if row["purchased"] else "no_purchase"
        counts[category] += 1
        if total >= threshold:
            counts[category + "_high_neural_reward"] += 1
            flagged.append({**row, "category": category, "cumulative_neural_reward": total})
    return {"trajectory_count": len(rows), "threshold": threshold, "counts": dict(counts), "flagged": flagged}


def training_rows(path):
    rows = []
    for episode in json.loads(path.read_text(encoding="utf-8")):
        turns = episode["turns"]
        previous, rewards = 0.0, []
        for turn in turns:
            value = finite(turn["prm_prefix_value"])
            reward = finite(turn["reward"])
            if not math.isclose(value - previous, reward, abs_tol=1e-7):
                raise ValueError(f"Task {episode['task_id']}: actual PPO reward differs from prefix increment")
            rewards.append(reward)
            previous = value
        rows.append({"source": str(path), "task_id": episode["task_id"],
                     "purchased": episode["purchase_completed"],
                     "environment_terminal": episode["terminal_reward"], "neural_rewards": rewards,
                     "skip_reasons": [t.get("prm_skip_reason") for t in turns]})
    return rows


def evaluation_rows(rollout, scores):
    payload = json.loads(scores.read_text(encoding="utf-8"))
    if payload["scoring_mode"] != "prefix_value_current_prefix_v1":
        raise ValueError("Validation audit requires the current-prefix scorer")
    if payload.get("input_schema") != "rollout" or payload.get("input_sha256") != digest(rollout):
        raise ValueError("Predictions do not identify this exact validation rollout file")
    episodes = [json.loads(line) for line in rollout.read_text(encoding="utf-8").splitlines() if line.strip()]
    predictions = payload["predictions"]
    ids = [p["task_id"] for p in predictions]
    if ids != [e["task_id"] for e in episodes] or len(set(ids)) != len(ids):
        raise ValueError("Validation rollout/prediction coverage or order differs")
    rows = []
    for episode, prediction in zip(episodes, predictions):
        rewards = prediction["step_scores"]
        records = prediction["turn_scoring"]
        if len(rewards) != len(records) or len(rewards) != len(episode["steps"]):
            raise ValueError("Validation reward length differs from the rollout")
        if not math.isclose(math.fsum(map(finite, rewards)), finite(prediction["total"]), abs_tol=1e-7):
            raise ValueError("Validation rewards do not sum to the prediction total")
        rows.append({"source": str(rollout), "task_id": episode["task_id"], "purchased": episode["done"],
                     "environment_terminal": episode["final_reward"] if episode["done"] else 0.0,
                     "neural_rewards": rewards, "skip_reasons": [r["skip_reason"] for r in records]})
    return rows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-episodes", type=Path, action="append", default=[])
    parser.add_argument("--validation-rollout", type=Path)
    parser.add_argument("--validation-scores", type=Path)
    parser.add_argument("--threshold", type=float, default=0.30)
    args = parser.parse_args()
    if not math.isfinite(args.threshold):
        parser.error("Threshold must be finite")
    if bool(args.validation_rollout) != bool(args.validation_scores):
        parser.error("Supply both validation files")
    if not args.training_episodes and not args.validation_rollout:
        parser.error("Supply training episodes or a validation rollout with its predictions")
    result = {"input_sha256": {}}
    if args.training_episodes:
        rows = [row for path in args.training_episodes for row in training_rows(path)]
        result["training_rewards"] = audit(rows, args.threshold)
    if args.validation_rollout:
        rows = evaluation_rows(args.validation_rollout, args.validation_scores)
        result["validation_readonly_scores"] = audit(rows, args.threshold)
    inputs = args.training_episodes + (
        [args.validation_rollout, args.validation_scores] if args.validation_rollout else []
    )
    result["input_sha256"] = {str(path): digest(path) for path in inputs}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
