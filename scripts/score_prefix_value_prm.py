#!/usr/bin/env python3
"""Standalone reload and inference CLI for Prefix-Value Telescoping PRM (PR #59 Arm 4 / Arm 2).

Drop-in compatible with `python -m webshop_lab.prm.infer`:
  - Uses shared `webshop_lab.prm.runtime.build_prm` for strict `prm-manifest.json` & `checkpoint_identity` verification;
  - Accepts unmodified raw `--samples` (or `--rollout`) identical to `stage126`, never reading labels;
  - For Arm 4 (default `--attach-asin-context`), constructs the PRM scoring view strictly from the
    immediately preceding user search-results message `messages[msg_idx - 1]`, recording exact character
    spans and scored `<|im_end|>` token indices in `asin_context_provenance`;
  - Outputs `v_prefix` and telescoping `step_scores` (`s_0 = V_0, s_t = V_t - V_{t-1}, total = V_{T-1}`).
"""
import argparse
import json
import re
from pathlib import Path

import torch

from webshop_lab.prm.config import load_config
from webshop_lab.prm.predictions import compare_predictions, validate_predictions
from webshop_lab.prm.runtime import build_prm, tensor_batch, verify_runtime_inputs, write_json
from webshop_lab.prm.tokenize import TrajectoryTooLong, tokenize_trajectory

ASIN_CLICK_RE = re.compile(r"click\[(b[0-9a-z]{9})\]", flags=re.IGNORECASE)


def extract_preclick_asin_snippet(messages: list[dict], msg_idx: int) -> dict | None:
    """Inspect `messages[msg_idx]` and read product title + price strictly from `messages[msg_idx - 1]`."""
    if msg_idx <= 0 or messages[msg_idx].get("role") != "assistant":
        return None
    if messages[msg_idx - 1].get("role") != "user":
        return None
    m_click = ASIN_CLICK_RE.search(messages[msg_idx].get("content", ""))
    if not m_click:
        return None
    asin_raw = m_click.group(1)
    asin_upper = asin_raw.upper()
    prev_page = messages[msg_idx - 1].get("content", "")
    pat = re.compile(
        r"\[SEP\]\s*" + re.escape(asin_upper) + r"\s*\[SEP\]\s*(.*?)\s*\[SEP\]\s*(\$[^\[]+?)(?:\s*\[SEP\]|$)",
        flags=re.DOTALL,
    )
    hit = pat.search(prev_page)
    if not hit:
        return None
    title = " ".join(hit.group(1).split())[:180]
    price = " ".join(hit.group(2).split())[:32]
    augmented_content = f"{messages[msg_idx]['content']} [Item {asin_upper}: {title} | {price}]"
    return {
        "assistant_message_index": msg_idx,
        "source_search_message_index": msg_idx - 1,
        "clicked_asin_raw": asin_raw,
        "matched_asin_upper": asin_upper,
        "source_char_span": [hit.start(), hit.end()],
        "source_verbatim_record": prev_page[hit.start():hit.end()],
        "extracted_title": title,
        "extracted_price": price,
        "original_assistant_content": messages[msg_idx]["content"],
        "augmented_assistant_content": augmented_content,
    }


def encode_trajectory_for_pv(tokenizer, messages: list[dict], max_length: int, attach_asin_context: bool):
    """Encode raw trajectory messages, optionally attaching deterministic pre-click ASIN context."""
    base_enc = tokenize_trajectory(tokenizer, messages, max_length)
    provenance = []
    chosen_enc = base_enc

    if attach_asin_context:
        aug_messages = [dict(m) for m in messages]
        assistant_turn_counter = 0
        extracted_records = []
        for msg_idx, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                info = extract_preclick_asin_snippet(messages, msg_idx)
                if info is not None:
                    info["turn_index"] = assistant_turn_counter
                    aug_messages[msg_idx]["content"] = info["augmented_assistant_content"]
                    extracted_records.append(info)
                assistant_turn_counter += 1

        if extracted_records:
            try:
                cand_enc = tokenize_trajectory(tokenizer, aug_messages, max_length)
                if len(cand_enc["turn_end_indices"]) == len(base_enc["turn_end_indices"]):
                    chosen_enc = cand_enc
                    for rec in extracted_records:
                        t_idx = rec["turn_index"]
                        tok_pos = chosen_enc["turn_end_indices"][t_idx]
                        tok_id = chosen_enc["input_ids"][tok_pos]
                        rec["fallback_to_raw_due_to_max_length"] = False
                        rec["raw_turn_end_token_index"] = base_enc["turn_end_indices"][t_idx]
                        rec["scored_turn_end_token_index"] = tok_pos
                        rec["scored_turn_end_token"] = tokenizer.decode([tok_id])
                        provenance.append(rec)
            except TrajectoryTooLong:
                chosen_enc = base_enc
                for rec in extracted_records:
                    t_idx = rec["turn_index"]
                    tok_pos = base_enc["turn_end_indices"][t_idx]
                    tok_id = base_enc["input_ids"][tok_pos]
                    rec["fallback_to_raw_due_to_max_length"] = True
                    rec["raw_turn_end_token_index"] = tok_pos
                    rec["scored_turn_end_token_index"] = tok_pos
                    rec["scored_turn_end_token"] = tokenizer.decode([tok_id])
                    provenance.append(rec)

    return chosen_enc, provenance


def load_pv_samples(path: Path, tokenizer, max_length: int, attach_asin_context: bool):
    raw_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    rejected = []
    seen = set()
    for idx, row in enumerate(raw_rows):
        task_id = row.get("task_id")
        if type(task_id) is not int or not isinstance(row.get("messages"), list):
            raise ValueError("每条样本需显式整数 task_id 和完整 messages 列表；推理不读取任何标签")
        if task_id in seen:
            raise ValueError(f"输入样本存在重复 task_id={task_id}")
        seen.add(task_id)
        try:
            enc, prov = encode_trajectory_for_pv(tokenizer, row["messages"], max_length, attach_asin_context)
        except TrajectoryTooLong as exc:
            rejected.append({"row": idx, "task_id": task_id, "reason": str(exc)})
            continue
        rows.append((task_id, enc, prov))
    return rows, rejected, len(raw_rows)


def load_pv_rollouts(path: Path, tokenizer, max_length: int, attach_asin_context: bool, task_id=None, repeat=None):
    from webshop_lab.prm.data import episode_messages

    raw_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [
        r for r in raw_rows
        if (task_id is None or r.get("task_id") == task_id)
        and (repeat is None or r.get("repeat", 0) == repeat)
    ]
    if not selected:
        raise ValueError("原始 rollout 中没有匹配的任务/repeat")
    rows = []
    rejected = []
    seen = set()
    for row in selected:
        tid = row.get("task_id")
        if type(tid) is not int or tid in seen:
            raise ValueError("原始轨迹任务重复或 ID 无效；请用 --task-id/--repeat 选择唯一轨迹")
        seen.add(tid)
        messages = episode_messages(row, require_labels=False)
        try:
            enc, prov = encode_trajectory_for_pv(tokenizer, messages, max_length, attach_asin_context)
        except TrajectoryTooLong as exc:
            rejected.append({"task_id": tid, "repeat": row.get("repeat", 0), "reason": str(exc)})
            continue
        rows.append((tid, enc, prov))
    return rows, rejected, len(selected)


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

    cfg = load_config(args.config)
    model, tok, identity, inputs = build_prm(cfg, args.checkpoint, args.device, False)
    if args.source_commit is not None and args.source_commit != identity["source_commit"]:
        raise ValueError("声明的源码提交与当前 runtime 检出不符")

    if args.samples:
        rows, rejected, total = load_pv_samples(args.samples, tok, cfg.max_length, args.attach_asin_context)
    else:
        rows, rejected, total = load_pv_rollouts(
            args.rollout, tok, cfg.max_length, args.attach_asin_context, args.task_id, args.repeat
        )
    if not rows:
        raise ValueError("无可评分的完整轨迹")

    actual = validate_predictions(score_rows_telescoping(model, tok, rows))
    if args.reference:
        if rejected:
            raise ValueError("重载验收输入含超长剔除项，不能仅比较保留子集")
        expected_ref = json.loads(args.reference.read_text(encoding="utf-8"))
        if isinstance(expected_ref, dict) and "predictions" in expected_ref:
            expected_ref = expected_ref["predictions"]
        compare_predictions(actual, expected_ref, args.atol, args.rtol)
    else:
        compare_predictions(actual, actual, args.atol, args.rtol)

    if verify_runtime_inputs(cfg) != inputs:
        raise ValueError("推理过程中初始化输入文件发生变化")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output,
        {
            "predictions": actual,
            "scoring_mode": "prefix_value_telescoping",
            "attach_asin_context": args.attach_asin_context,
            "identity": identity,
            "device": args.device,
            "total_samples": total,
            "rejected": rejected,
            "input_schema": "samples" if args.samples else "rollout",
            "reference_match": True if args.reference else None,
            "atol": args.atol,
            "rtol": args.rtol,
            "test_authorization": False,
        },
    )


if __name__ == "__main__":
    main()
