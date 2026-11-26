#!/usr/bin/env python3
"""Standalone reload and inference CLI for Prefix-Value Telescoping PRM (PR #59 Arm 4 / Arm 2).

Use the pinned checkpoint runtime on PYTHONPATH:
  - Uses shared `webshop_lab.prm.runtime.build_prm` for strict `prm-manifest.json` & `checkpoint_identity` verification;
  - Accepts unmodified raw `--samples` (or `--rollout`) identical to `stage126`, never reading labels;
  - For Arm 4 (default `--attach-asin-context`), constructs the PRM scoring view strictly from the
    immediately preceding user search-results message `messages[msg_idx - 1]`, recording exact character
    spans and scored `<|im_end|>` token indices in `asin_context_provenance`;
  - Outputs `v_prefix` and telescoping `step_scores` (`s_0 = V_0, s_t = V_t - V_{t-1}, total = V_{T-1}`).
  - Scores current prefixes by default. Use --historical-full-trajectory only to reproduce archived scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

import torch

from webshop_lab.prm.config import load_config
from webshop_lab.prm.predictions import compare_predictions, validate_predictions
from webshop_lab.model import SYSTEM, fit_history
from webshop_lab.prm.runtime import build_prm, tensor_batch, verify_runtime_inputs, write_json
from webshop_lab.prm.tokenize import TrajectoryTooLong

# The checkpoint loader must come from the pinned runtime. Only the scoring
# policy comes from this checkout, which may have modules the runtime lacks.
_PREFIX = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                             "src/webshop_lab/prm/prefix_reward.py"))
encode_trajectory_for_pv = _PREFIX["encode_trajectory_for_pv"]
extract_preclick_asin_snippet = _PREFIX["extract_preclick_asin_snippet"]
CURRENT_MODE = "prefix_value_current_prefix_v1"
HISTORICAL_MODE = "prefix_value_historical_full_trajectory_v1"


def load_pv_samples(path: Path):
    raw_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    seen = set()
    for row in raw_rows:
        task_id = row.get("task_id")
        if type(task_id) is not int or not isinstance(row.get("messages"), list):
            raise ValueError("每条样本需显式整数 task_id 和完整 messages 列表；推理不读取任何标签")
        if task_id in seen:
            raise ValueError(f"输入样本存在重复 task_id={task_id}")
        seen.add(task_id)
        rows.append((task_id, row["messages"]))
    return rows


def rollout_messages(row):
    instruction = row["instruction"]
    messages = [{"role": "system", "content": SYSTEM + "\nShopping instruction: " + instruction}]
    observation = row["steps"][0]["observation"] if row.get("steps") else None
    if observation is None:
        raise ValueError("Rollout has no initial observation")
    for i, step in enumerate(row["steps"]):
        if step["step"] != i:
            raise ValueError("Rollout steps must be consecutive and in execution order")
        messages.extend((
            {"role": "user", "content": observation},
            {"role": "assistant", "content": step["response"]},
        ))
        observation = None
        if not step["grounding"]:
            observation = "Observation: Invalid action or format. Reply with one valid Action: line."
        elif i + 1 < len(row["steps"]):
            observation = row["steps"][i + 1]["observation"]
    return messages


def load_pv_rollouts(path: Path, task_id=None, repeat=None):
    raw_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [
        r for r in raw_rows
        if (task_id is None or r.get("task_id") == task_id)
        and (repeat is None or r.get("repeat", 0) == repeat)
    ]
    if not selected:
        raise ValueError("原始 rollout 中没有匹配的任务/repeat")
    rows = []
    seen = set()
    for row in selected:
        tid = row.get("task_id")
        if type(tid) is not int or tid in seen:
            raise ValueError("原始轨迹任务重复或 ID 无效；请用 --task-id/--repeat 选择唯一轨迹")
        seen.add(tid)
        rows.append((tid, rollout_messages(row)))
    return rows


def score_current_rows(model, tokenizer, rows, max_length, prompt_budget, attach_asin_context):
    predictions = []
    for task_id, messages in rows:
        previous, values, rewards, records, provenance = 0.0, [], [], [], []
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("Supply a trajectory ending at an assistant turn")
        for msg_index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            fitted, _ = fit_history(tokenizer, messages[:msg_index], prompt_budget, generation=True)
            value, index, digest, reason = previous, None, None, "empty_response"
            details = []
            if message["content"].strip():
                encoded, details = _PREFIX["encode_current_prefix_details"](
                    tokenizer, fitted + [message], max_length, attach_asin_context,
                )
                reason = "prefix_too_long" if encoded is None else None
                if encoded is not None:
                    value = _PREFIX["score_last_prefix"](model, tokenizer, encoded)
                    index = encoded["turn_end_indices"][-1]
                    digest = hashlib.sha256(json.dumps(encoded["input_ids"]).encode()).hexdigest()
            turn = len(values)
            provenance.extend(
                {**record, "turn_index": turn, "assistant_message_index": msg_index,
                 "source_search_message_index": msg_index - 1}
                for record in details if record["scored_turn_end_token_index"] == index
            )
            records.append({"turn_index": turn, "prefix_ids_sha256": digest, "score_index": index,
                            "skip_reason": reason, "asin_length_fallback": any(
                                r["fallback_to_raw_due_to_max_length"] for r in details)})
            values.append(value)
            rewards.append(value - previous)
            previous = value
        predictions.append({
            "task_id": task_id, "turn_end_indices": [r["score_index"] for r in records],
            "v_prefix": values, "step_scores": rewards, "total": previous,
            "turn_scoring": records, "asin_context_provenance": provenance,
        })
    return predictions


def compare_scores(actual, expected, atol, rtol, historical):
    if historical:
        compare_predictions(actual, expected, atol, rtol)
        return
    if len(actual) != len(expected) or any(
        a["turn_scoring"] != b.get("turn_scoring") for a, b in zip(actual, expected)
    ):
        raise ValueError("Current-prefix reference has different inputs or scoring positions")
    # Local token coordinates can reset after history truncation. Compare numeric
    # predictions by turn and verify local coordinates in turn_scoring above.
    indexed = lambda rows: [
        {**row, "turn_end_indices": list(range(len(row["step_scores"])))} for row in rows
    ]
    compare_predictions(indexed(actual), indexed(expected), atol, rtol)


@torch.no_grad()
def score_rows_telescoping(model, tokenizer, rows):
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    for task_id, enc, prov in rows:
        batch = tensor_batch([enc], tokenizer.pad_token_id, device, False)
        out = model(**batch)
        mask = batch["turn_mask"][0]
        v_list = out["step_scores"][0][mask].float().cpu().tolist()
        step_scores = []
        prev = 0.0
        for v in v_list:
            step_scores.append(float(v - prev))
            prev = float(v)
        total = float(v_list[-1])
        predictions.append({
            "task_id": task_id,
            "turn_end_indices": enc["turn_end_indices"],
            "v_prefix": v_list,
            "step_scores": step_scores,
            "total": total,
            "asin_context_provenance": prov,
        })
    return predictions


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "checkpoint", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--samples", type=Path)
    src.add_argument("--rollout", type=Path)
    p.add_argument("--task-id", type=int)
    p.add_argument("--repeat", type=int)
    p.add_argument("--source-commit", help="可选断言实际 runtime 源码提交；不改写 checkpoint 身份校验")
    p.add_argument("--reference", type=Path)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--contract", type=Path, default=_PREFIX["CONTRACT_PATH"])
    p.add_argument("--policy-context-length", type=int, default=2048)
    p.add_argument("--policy-max-new-tokens", type=int, default=96)
    p.add_argument("--historical-full-trajectory", action="store_true",
                   help="Only for archiving/reproducing the old whole-trajectory scores.")
    p.add_argument(
        "--attach-asin-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default True (Arm 4 template); pass --no-attach-asin-context for raw-text Arm 2.",
    )
    args = p.parse_args(argv)

    if args.samples and any(v is not None for v in (args.task_id, args.repeat)):
        raise ValueError("--task-id/--repeat 仅用于原始 rollout 输入")
    if args.output.exists() or args.checkpoint.resolve() in args.output.resolve().parents:
        raise ValueError("新输出须位于检查点目录之外，禁止覆盖已有证据")

    if not 0 < args.policy_max_new_tokens < args.policy_context_length:
        raise ValueError("Policy context must reserve room for generation")
    contract = _PREFIX["prefix_value_checkpoint"](args.checkpoint, args.attach_asin_context, args.contract)
    cfg = load_config(args.config)
    model, tok, identity, inputs = build_prm(cfg, args.checkpoint, args.device, False)
    if args.source_commit is not None and args.source_commit != identity["source_commit"]:
        raise ValueError("声明的源码提交与当前 runtime 检出不符")

    if args.samples:
        rows = load_pv_samples(args.samples)
    else:
        rows = load_pv_rollouts(args.rollout, args.task_id, args.repeat)
    if not rows:
        raise ValueError("无可评分的完整轨迹")

    total, rejected = len(rows), []
    if args.historical_full_trajectory:
        encoded = []
        for task_id, messages in rows:
            try:
                enc, prov = encode_trajectory_for_pv(tok, messages, cfg.max_length, args.attach_asin_context)
            except TrajectoryTooLong as error:
                rejected.append({"task_id": task_id, "reason": str(error)})
            else:
                encoded.append((task_id, enc, prov))
        actual = validate_predictions(score_rows_telescoping(model, tok, encoded))
    else:
        actual = score_current_rows(model, tok, rows, cfg.max_length,
                                    args.policy_context_length - args.policy_max_new_tokens,
                                    args.attach_asin_context)
    mode = HISTORICAL_MODE if args.historical_full_trajectory else CURRENT_MODE
    if args.reference:
        if rejected:
            raise ValueError("重载验收输入含超长剔除项，不能仅比较保留子集")
        expected_ref = json.loads(args.reference.read_text(encoding="utf-8"))
        if isinstance(expected_ref, dict):
            mode_expected = expected_ref.get("scoring_mode")
            if mode_expected != mode and not (
                args.historical_full_trajectory and mode_expected in (None, "prefix_value_telescoping")
            ):
                raise ValueError("Reference was scored using a different prefix policy")
            expected_ref = expected_ref["predictions"]
        compare_scores(actual, expected_ref, args.atol, args.rtol, args.historical_full_trajectory)
    else:
        compare_scores(actual, actual, args.atol, args.rtol, args.historical_full_trajectory)

    if verify_runtime_inputs(cfg) != inputs:
        raise ValueError("推理过程中初始化输入文件发生变化")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output,
        {
            "predictions": actual,
            "scoring_mode": mode,
            "scoring_contract": contract,
            "policy_context_length": args.policy_context_length,
            "policy_max_new_tokens": args.policy_max_new_tokens,
            "attach_asin_context": args.attach_asin_context,
            "identity": identity,
            "device": args.device,
            "total_samples": total,
            "rejected": rejected,
            "input_schema": "samples" if args.samples else "rollout",
            "input_sha256": hashlib.sha256((args.samples or args.rollout).read_bytes()).hexdigest(),
            "reference_match": True if args.reference else None,
            "atol": args.atol,
            "rtol": args.rtol,
            "test_authorization": False,
        },
    )


if __name__ == "__main__":
    main()
