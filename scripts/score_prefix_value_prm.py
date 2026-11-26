#!/usr/bin/env python3
"""固定旧 runtime 下的因果前缀价值评分；必须指定外部语义契约及其冻结哈希。"""
import argparse
import json
from pathlib import Path

import torch

from prm_prefix_value_support import (
    compare_pv_predictions, encode_current_prefix, extract_preclick_asin_snippet, file_sha, prepare_prefixes,
    rollout_messages, validate_pv_predictions, verify_contract,
)
from webshop_lab.prm.config import load_config
from webshop_lab.prm.runtime import build_prm, tensor_batch, verify_runtime_inputs, write_json
from webshop_lab.model import fit_history


def prepare_scoring_inputs(tokenizer, messages, max_length, prompt_budget):
    if prompt_budget is None:
        return prepare_prefixes(tokenizer, messages, max_length)
    if not messages or len(messages) < 3 or len(messages) % 2 != 1:
        raise ValueError("Policy replay requires complete system/user/assistant turns")
    prefixes = []
    for index in range(2, len(messages), 2):
        history, _ = fit_history(tokenizer, messages[:index], prompt_budget, generation=True)
        prefixes.append(encode_current_prefix(tokenizer, history + [messages[index]], max_length))
    return prefixes


def load_pv_samples(path, tokenizer, max_length, prompt_budget=None):
    raw = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, seen = [], set()
    for row in raw:
        task = row.get("task_id")
        if type(task) is not int or task in seen:
            raise ValueError("任务 ID 无效或重复；请先按 repeat 选择唯一轨迹")
        seen.add(task)
        rows.append((task, prepare_scoring_inputs(tokenizer, row.get("messages"), max_length, prompt_budget)))
    return rows, [], len(raw)


def load_pv_rollouts(path, tokenizer, max_length, task_id=None, repeat=None, prompt_budget=None):
    raw = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in raw if (task_id is None or row.get("task_id") == task_id)
                and (repeat is None or row.get("repeat", 0) == repeat)]
    if not selected:
        raise ValueError("原始轨迹中无匹配任务/repeat")
    rows, seen = [], set()
    for row in selected:
        task = row.get("task_id")
        if type(task) is not int or task in seen:
            raise ValueError("原始轨迹 ID 无效或重复")
        seen.add(task)
        messages = rollout_messages(row)
        if prompt_budget is not None:
            # rollout.jsonl records the simulator state. The policy saw the error
            # feedback instead on the turn immediately after an invalid action.
            for i, step in enumerate(row["steps"][:-1]):
                if step.get("grounding") not in (0, 1):
                    raise ValueError("Policy replay needs the rollout grounding field")
                if not step["grounding"]:
                    messages[2 * i + 3]["content"] = (
                        "Observation: Invalid action or format. Reply with one valid Action: line."
                    )
        rows.append((task, prepare_scoring_inputs(tokenizer, messages, max_length, prompt_budget)))
    return rows, [], len(selected)


@torch.no_grad()
def score_rows_telescoping(model, tokenizer, rows):
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    for task, prefixes in rows:
        values, rewards, traces, provenance = [], [], [], []
        previous = 0.0
        for turn, (encoded, trace) in enumerate(prefixes):
            value = previous
            if encoded is not None:
                batch = tensor_batch([encoded], tokenizer.pad_token_id, device, False)
                value = float(model(**batch)["step_scores"][0, -1])
            for record in trace.get("asin_context_provenance", []):
                if record["assistant_message_index"] == trace["current_assistant_message_index"]:
                    provenance.append({**record, "turn_index": turn,
                                       "scored_turn_end_token_index": trace["score_token_index"],
                                       "scored_turn_end_token": tokenizer.decode([encoded["input_ids"][trace["score_token_index"]]])})
            values.append(value)
            rewards.append(value - previous)
            traces.append(trace)
            previous = value
        predictions.append({"task_id": task, "v_prefix": values, "step_scores": rewards,
                            "total": previous, "prefix_traces": traces,
                            "position_convention": "per_prefix_local",
                            "turn_end_indices": [trace.get("score_token_index") for trace in traces],
                            "asin_context_provenance": provenance})
    return validate_pv_predictions(predictions)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "checkpoint", "output", "scoring-contract"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--scoring-contract-sha256", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--samples", type=Path)
    source.add_argument("--rollout", type=Path)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--source-commit")
    parser.add_argument("--policy-context-length", type=int)
    parser.add_argument("--policy-max-new-tokens", type=int)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    if args.samples and any(x is not None for x in (args.task_id, args.repeat)):
        raise ValueError("task-id/repeat 仅用于原始轨迹")
    if args.output.exists() or args.checkpoint.resolve() in args.output.resolve().parents:
        raise ValueError("输出须在检查点外，不能覆盖旧证据")
    context, max_new = args.policy_context_length, args.policy_max_new_tokens
    if (context is None) != (max_new is None) or (context is not None and not 0 < max_new < context):
        raise ValueError("Supply both policy limits and leave room for generation")
    prompt_budget = None if context is None else context - max_new
    input_path = args.samples or args.rollout
    input_sha256 = file_sha(input_path)
    contract = verify_contract(args.checkpoint, args.scoring_contract, args.scoring_contract_sha256)
    cfg = load_config(args.config)
    if cfg.max_length != contract["max_length"]:
        raise ValueError("配置窗口与评分契约不符")
    model, tokenizer, identity, inputs = build_prm(cfg, args.checkpoint, args.device, False)
    if args.source_commit is not None and args.source_commit != identity["source_commit"]:
        raise ValueError("声明的 runtime 提交与实际身份不符")
    if args.samples:
        rows, rejected, total = load_pv_samples(args.samples, tokenizer, cfg.max_length, prompt_budget)
    else:
        rows, rejected, total = load_pv_rollouts(args.rollout, tokenizer, cfg.max_length,
                                                task_id=args.task_id, repeat=args.repeat, prompt_budget=prompt_budget)
    actual = score_rows_telescoping(model, tokenizer, rows)
    if args.reference:
        expected = json.loads(args.reference.read_text(encoding="utf-8"))
        if (not isinstance(expected, dict)
                or expected.get("scoring_contract_sha256") != args.scoring_contract_sha256
                or expected.get("scoring_mode") != "causal_prefix_value_v1"
                or expected.get("policy_context_length") != context
                or expected.get("policy_max_new_tokens") != max_new
                or expected.get("input_sha256") != input_sha256):
            raise ValueError("reference 缺少相同版本的评分契约，不能复用历史全轨迹参考")
        compare_pv_predictions(actual, expected["predictions"], args.atol, args.rtol)
    else:
        compare_pv_predictions(actual, actual, args.atol, args.rtol)
    if verify_runtime_inputs(cfg) != inputs:
        raise ValueError("推理中初始化输入文件发生变化")
    verify_contract(args.checkpoint, args.scoring_contract, args.scoring_contract_sha256)
    if file_sha(input_path) != input_sha256:
        raise ValueError("Scoring input changed during the run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"predictions": actual, "identity": identity,
                            "scoring_mode": "causal_prefix_value_v1",
                            "input_schema": "samples" if args.samples else "rollout",
                            "input_sha256": input_sha256,
                            "policy_context_length": context, "policy_max_new_tokens": max_new,
                            "scoring_contract_sha256": args.scoring_contract_sha256,
                            "scoring_contract": contract, "device": args.device,
                            "total_samples": total, "rejected": rejected,
                            "reference_match": True if args.reference else None,
                            "atol": args.atol, "rtol": args.rtol, "test_authorization": False})


if __name__ == "__main__":
    main()
