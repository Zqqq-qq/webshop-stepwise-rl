#!/usr/bin/env python3
"""Exploratory controlled ablation for Prefix-Value Telescoping PRM (100% terminal-reward only).

Tests solving the two root causes of flat click-step PRM scores WITHOUT privileged step labels:
  1. Prefix-Value Monte Carlo MSE + Telescoping Step Scores:
        Head predicts V_t = E[R | h_t] at each assistant turn-end t with loss mean_t (V_t - R)^2;
       Step progress score is s_0 = V_0, s_t = V_t - V_{t-1} (so sum_t s_t == V_{T-1} == total).
       Eliminates the collinear sum(s_t)=R late-step shortcut while automatically handling both
       pre-click observable (V+) and post-click observable (U) cases via conditional expectation.
  2. Cross-case ASIN lookup bottleneck (search page 'B07YNGTFK3' vs action 'click[b07yngtfk3]'):
       Tested both via expanded LoRA target modules (q,k,v,o,gate,up,down) on raw text AND via
       deterministic zero-leakage snippet lookup strictly from the immediately preceding user page.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from webshop_lab.prm.config import PRMConfig, load_config
from webshop_lab.prm.model_core import save_prm
from webshop_lab.prm.runtime import (
    build_prm,
    finalize_manifest,
    sha,
    tensor_batch,
    verify_runtime_inputs,
    write_json,
)
from webshop_lab.prm.tokenize import TrajectoryTooLong, tokenize_trajectory
from webshop_lab.prm.train import cosine_lr_multiplier

ALL_LINEAR_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

ASIN_CLICK_RE = re.compile(r"click\[(b[0-9a-z]{9})\]", flags=re.IGNORECASE)


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_public_asin_from_prev_page(messages, index: int) -> str | None:
    """Strictly extract verbatim product title + price of clicked ASIN from messages[index - 1].

    Uses ONLY the public observation already shown to the agent before this action (zero leakage).
    """
    if index <= 0 or messages[index]["role"] != "assistant" or messages[index - 1]["role"] != "user":
        return None
    m = ASIN_CLICK_RE.search(messages[index]["content"])
    if not m:
        return None
    asin_upper = m.group(1).upper()
    page = messages[index - 1]["content"]
    # WebShop search page format: [SEP] B07YNGTFK3 [SEP] <Title> [SEP] $<Price>
    pat = re.compile(
        r"\[SEP\]\s*" + re.escape(asin_upper) + r"\s*\[SEP\]\s*(.*?)\s*\[SEP\]\s*(\$[^\[]+?)(?:\s*\[SEP\]|$)",
        flags=re.DOTALL,
    )
    hit = pat.search(page)
    if not hit:
        return None
    title = " ".join(hit.group(1).split())[:180]
    price = " ".join(hit.group(2).split())[:32]
    return f"{messages[index]['content']} [Item {asin_upper}: {title} | {price}]"


def encode_rows(tokenizer, raw_rows, max_length: int, need_reward: bool, attach_asin_context: bool = False):
    kept = []
    rejected = []
    for idx, row in enumerate(raw_rows):
        orig_messages = row["messages"]
        try:
            base_enc = tokenize_trajectory(tokenizer, orig_messages, max_length)
        except TrajectoryTooLong as exc:
            rejected.append({"row": idx, "task_id": row["task_id"], "reason": str(exc)})
            continue

        chosen_enc = base_enc
        if attach_asin_context:
            aug_messages = [dict(m) for m in orig_messages]
            changed = False
            for i in range(len(aug_messages)):
                repl = resolve_public_asin_from_prev_page(aug_messages, i)
                if repl is not None:
                    aug_messages[i]["content"] = repl
                    changed = True
            if changed:
                try:
                    cand_enc = tokenize_trajectory(tokenizer, aug_messages, max_length)
                    if len(cand_enc["turn_end_indices"]) == len(base_enc["turn_end_indices"]):
                        chosen_enc = cand_enc
                except TrajectoryTooLong:
                    # Keep exact same retained sample set as raw baseline if snippet pushes > max_length
                    chosen_enc = base_enc

        item = dict(chosen_enc)
        if need_reward:
            reward = float(row["terminal_reward"])
            if not (math.isfinite(reward) and 0.0 <= reward <= 1.0):
                raise ValueError(f"Invalid terminal_reward on task {row['task_id']}: {reward}")
            item["terminal_reward"] = reward
        kept.append((int(row["task_id"]), item, row))
    return kept, rejected


def prefix_value_step(model, batches, optimizer, max_grad_norm: float = 1.0):
    """Optimize per-turn Monte Carlo MSE: mean_t (V_t - R)^2 using ONLY terminal scalar reward R."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    n = len(batches)
    mean_loss = 0.0
    mean_terminal_mse = 0.0
    for batch in batches:
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            turn_end_indices=batch["turn_end_indices"],
            turn_mask=batch["turn_mask"],
            labels=None,
        )
        mask = batch["turn_mask"][0]
        v_preds = out["step_scores"][0][mask].float()
        target_r = batch["labels"][0].float()
        loss = (v_preds - target_r).square().mean()
        term_mse = (v_preds[-1] - target_r).square()
        (loss / n).backward()
        mean_loss += float(loss.detach()) / n
        mean_terminal_mse += float(term_mse.detach()) / n
        del out, loss
    params = [p for p in model.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return {
        "pv_loss": mean_loss,
        "last_turn_mse": mean_terminal_mse,
        "gradient_norm": float(grad_norm),
    }


def prefix_value_pair_step(model, pair_batches, optimizer, diff_weight: float = 1.0, max_grad_norm: float = 1.0):
    """Optimize PV-MSE + terminal-difference matching at branch turn on frozen 20 fork pairs (pure terminal R)."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    m = len(pair_batches)
    total_loss_val = 0.0
    mean_gap_val = 0.0
    for branch_idx, high_batch, low_batch in pair_batches:
        out_h = model(
            input_ids=high_batch["input_ids"],
            attention_mask=high_batch["attention_mask"],
            turn_end_indices=high_batch["turn_end_indices"],
            turn_mask=high_batch["turn_mask"],
            labels=None,
        )
        out_l = model(
            input_ids=low_batch["input_ids"],
            attention_mask=low_batch["attention_mask"],
            turn_end_indices=low_batch["turn_end_indices"],
            turn_mask=low_batch["turn_mask"],
            labels=None,
        )
        vh = out_h["step_scores"][0][high_batch["turn_mask"][0]].float()
        vl = out_l["step_scores"][0][low_batch["turn_mask"][0]].float()
        rh = high_batch["labels"][0].float()
        rl = low_batch["labels"][0].float()
        pv_term = 0.5 * ((vh - rh).square().mean() + (vl - rl).square().mean())
        pred_gap = vh[branch_idx] - vl[branch_idx]
        target_gap = rh - rl
        diff_term = (pred_gap - target_gap).square()
        loss = pv_term + diff_weight * diff_term
        (loss / m).backward()
        total_loss_val += float(loss.detach()) / m
        mean_gap_val += float(pred_gap.detach()) / m
        del out_h, out_l, loss
    params = [p for p in model.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return {
        "pair_loss": total_loss_val,
        "mean_train_gap": mean_gap_val,
        "gradient_norm": float(grad_norm),
    }


@torch.no_grad()
def predict_telescoping(model, pad_token_id: int, encoded_rows, device: str):
    """Predict V_t and convert to telescoping step_scores: s_0 = V_0, s_t = V_t - V_{t-1}, total = V_{T-1}."""
    model.eval()
    results = {}
    for task_id, enc, _ in encoded_rows:
        batch = tensor_batch([enc], pad_token_id, device, False)
        out = model(**batch)
        mask = batch["turn_mask"][0]
        v_list = out["step_scores"][0][mask].float().cpu().tolist()
        step_scores = []
        prev = 0.0
        for v in v_list:
            step_scores.append(float(v - prev))
            prev = float(v)
        total = float(v_list[-1])
        results[task_id] = {
            "task_id": task_id,
            "turn_end_indices": enc["turn_end_indices"],
            "v_prefix": v_list,
            "step_scores": step_scores,
            "total": total,
        }
    return results


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    result = np.empty(len(values), dtype=float)
    idx = 0
    while idx < len(values):
        end = idx + 1
        while end < len(values) and sorted_values[end] == sorted_values[idx]:
            end += 1
        result[order[idx:end]] = (idx + end - 1) / 2.0 + 1
        idx = end
    return result


def calc_metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    err = p - y
    pearson = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and np.any(y != y[0]) and np.any(p != p[0]) else None
    spearman = float(np.corrcoef(ranks(y), ranks(p))[0, 1]) if len(y) > 1 and np.any(y != y[0]) and np.any(p != p[0]) else None
    return {
        "n": int(len(y)),
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
        "max_abs_error": float(np.max(np.abs(err))),
        "pearson_r": pearson,
        "spearman_rho": spearman,
    }


def evaluate_all(model, tokenizer, data_dir: Path, device: str, attach_asin_context: bool, max_length: int):
    pad_id = tokenizer.pad_token_id
    fit_raw = load_jsonl(data_dir / "fit501.training.jsonl")
    fit_labels = np.array([float(r["terminal_reward"]) for r in fit_raw], dtype=float)
    fit_turns = np.array([sum(m["role"] == "assistant" for m in r["messages"]) for r in fit_raw], dtype=float)
    const_base = float(fit_labels.mean())
    group_map = {}
    for t, y in zip(fit_turns, fit_labels):
        group_map.setdefault(float(t), []).append(float(y))
    group_mean = {t: float(np.mean(vs)) for t, vs in group_map.items()}
    ols_beta = np.linalg.lstsq(np.column_stack([np.ones_like(fit_turns), fit_turns]), fit_labels, rcond=None)[0]

    # 1. Independent dev160 evaluation
    dev160_raw = load_jsonl(data_dir / "dev160.training.jsonl")
    dev160_enc, dev160_rej = encode_rows(tokenizer, dev160_raw, max_length, True, attach_asin_context)
    dev160_preds = predict_telescoping(model, pad_id, dev160_enc, device)
    y160 = np.array([enc["terminal_reward"] for _, enc, _ in dev160_enc], dtype=float)
    p160 = np.array([dev160_preds[tid]["total"] for tid, _, _ in dev160_enc], dtype=float)
    t160 = np.array([len(enc["turn_end_indices"]) for _, enc, _ in dev160_enc], dtype=float)
    terms160 = [row["termination"] for _, _, row in dev160_enc]

    zero_purchase_mask = np.array([y == 0.0 and term == "purchase" for y, term in zip(y160, terms160)])
    max_steps_mask = np.array([term == "max_steps" for term in terms160])
    pos_purchase_mask = np.array([y > 0.0 and term == "purchase" for y, term in zip(y160, terms160)])

    dev160_report = {
        "total_raw": len(dev160_raw),
        "retained": len(dev160_enc),
        "rejected": len(dev160_rej),
        "model": calc_metrics(y160, p160),
        "baselines": {
            "constant": float(np.mean((const_base - y160) ** 2)),
            "turn_count_mean": float(np.mean((np.array([group_mean.get(float(t), const_base) for t in t160]) - y160) ** 2)),
            "turn_count_ols": float(np.mean(((ols_beta[0] + ols_beta[1] * t160) - y160) ** 2)),
        },
        "subgroups": {
            "positive_purchase": {"n": int(pos_purchase_mask.sum()), "mse": float(np.mean((p160[pos_purchase_mask] - y160[pos_purchase_mask]) ** 2)) if pos_purchase_mask.any() else None},
            "zero_purchase": {"n": int(zero_purchase_mask.sum()), "mse": float(np.mean((p160[zero_purchase_mask] - y160[zero_purchase_mask]) ** 2)) if zero_purchase_mask.any() else None},
            "max_steps_zero": {"n": int(max_steps_mask.sum()), "mse": float(np.mean((p160[max_steps_mask] - y160[max_steps_mask]) ** 2)) if max_steps_mask.any() else None},
        },
    }

    # 2. #57 Independent Dev 38 Pairs (32 directional + 6 tied)
    pairs_meta = json.loads((data_dir / "dev_pairs_meta.json").read_text(encoding="utf-8"))
    tgt_raw = load_jsonl(data_dir / "dev_pairs.target.samples.jsonl")
    alt_raw = load_jsonl(data_dir / "dev_pairs.alternative.samples.jsonl")
    tgt_enc, _ = encode_rows(tokenizer, tgt_raw, max_length, False, attach_asin_context)
    alt_enc, _ = encode_rows(tokenizer, alt_raw, max_length, False, attach_asin_context)
    tgt_preds = predict_telescoping(model, pad_id, tgt_enc, device)
    alt_preds = predict_telescoping(model, pad_id, alt_enc, device)

    correct = incorrect = unresolved = 0
    gaps = []
    by_vis = {"V+": [0, 0], "U": [0, 0], "V-": [0, 0]}
    pair_y, pair_p = [], []
    per_pair = []
    for item in pairs_meta:
        tid = int(item["task_id"])
        br = int(item["branch"])
        rt = float(item["target_reward"])
        ra = float(item["alternative_reward"])
        vis = item.get("visibility_class")
        st = float(tgt_preds[tid]["step_scores"][br])
        sa = float(alt_preds[tid]["step_scores"][br])
        pair_y.extend([rt, ra])
        pair_p.extend([tgt_preds[tid]["total"], alt_preds[tid]["total"]])
        if rt != ra:
            high_s, low_s = (st, sa) if rt > ra else (sa, st)
            gap = high_s - low_s
            gaps.append(gap)
            is_ok = gap > 1e-5
            if is_ok:
                correct += 1
            elif gap < -1e-5:
                incorrect += 1
            else:
                unresolved += 1
            if vis in by_vis:
                by_vis[vis][1] += 1
                if is_ok:
                    by_vis[vis][0] += 1
            per_pair.append({
                "task_id": tid,
                "visibility": vis,
                "env_gap": abs(rt - ra),
                "model_click_gap": gap,
                "correct": is_ok,
                "target_step_scores": tgt_preds[tid]["step_scores"],
                "alt_step_scores": alt_preds[tid]["step_scores"],
            })

    dev_pairs_report = {
        "directional_n": len(gaps),
        "correct": correct,
        "incorrect": incorrect,
        "unresolved": unresolved,
        "accuracy": correct / len(gaps) if gaps else None,
        "mean_high_minus_low_gap": float(np.mean(gaps)) if gaps else None,
        "median_high_minus_low_gap": float(np.median(gaps)) if gaps else None,
        "by_visibility": {k: f"{v[0]}/{v[1]}" for k, v in by_vis.items() if v[1] > 0},
        "pair_total_metrics": calc_metrics(pair_y, pair_p),
        "per_pair": per_pair,
    }

    # 3. Regression 7544 (Running Shoes 1.0 vs Nordways Clogs 0.0)
    r7544_t_enc, _ = encode_rows(tokenizer, load_jsonl(data_dir / "r7544.target.samples.jsonl"), max_length, False, attach_asin_context)
    r7544_a_enc, _ = encode_rows(tokenizer, load_jsonl(data_dir / "r7544.alternative.samples.jsonl"), max_length, False, attach_asin_context)
    p7544_t = predict_telescoping(model, pad_id, r7544_t_enc, device)[7544]
    p7544_a = predict_telescoping(model, pad_id, r7544_a_enc, device)[7544]
    reg7544_report = {
        "running_shoes_1.0": p7544_t,
        "clogs_0.0": p7544_a,
        "click_step_gap_shoes_minus_clogs": float(p7544_t["step_scores"][1] - p7544_a["step_scores"][1]),
        "total_gap_shoes_minus_clogs": float(p7544_t["total"] - p7544_a["total"]),
    }

    return {
        "dev32_pairs": dev_pairs_report,
        "regression_7544": reg7544_report,
        "dev160": dev160_report,
    }


def train_fit_arm(
    arm_name: str,
    base_cfg_dict: dict,
    data_dir: Path,
    out_root: Path,
    device: str,
    target_modules: tuple,
    attach_asin_context: bool,
    steps: int,
    accumulation: int,
    lora_lr: float,
    head_lr: float,
    seed: int = 42,
):
    t0 = time.monotonic()
    ckpt_dir = out_root / arm_name
    cfg_dict = copy.deepcopy(base_cfg_dict)
    cfg_dict["target_modules"] = list(target_modules)
    cfg_dict["output_dir"] = str(ckpt_dir)
    cfg = PRMConfig.from_dict(cfg_dict)

    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer, identity, _ = build_prm(cfg, None, device, True)
    fit_raw = load_jsonl(data_dir / "fit501.training.jsonl")
    encoded, rejected = encode_rows(tokenizer, fit_raw, cfg.max_length, True, attach_asin_context)

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    head_params = [p for n, p in trainable if n.startswith("score_head.")]
    lora_params = [p for n, p in trainable if "lora_" in n]
    base_lrs = [head_lr, lora_lr]
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": head_lr}, {"params": lora_params, "lr": lora_lr}],
        weight_decay=0.01,
    )

    pad_id = tokenizer.pad_token_id
    consumed = 0
    records = []
    for step_idx in range(steps):
        mult = cosine_lr_multiplier(step_idx, steps)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * mult
        selected = [encoded[(consumed + i) % len(encoded)] for i in range(accumulation)]
        batches = [tensor_batch([enc], pad_id, device, True) for _, enc, _ in selected]
        rep = prefix_value_step(model, batches, optimizer, max_grad_norm=1.0)
        consumed += len(selected)
        rep["step"] = step_idx + 1
        rep["lora_lr"] = lora_lr * mult
        rep["head_lr"] = head_lr * mult
        records.append(rep)
        if (step_idx + 1) % max(1, steps // 6) == 0 or step_idx == 0 or step_idx + 1 == steps:
            print(
                f"[{arm_name}] step {step_idx+1}/{steps} | pv_loss={rep['pv_loss']:.5f} "
                f"| last_mse={rep['last_turn_mse']:.5f} | gnorm={rep['gradient_norm']:.2f} "
                f"| elapsed={time.monotonic()-t0:.1f}s",
                flush=True,
            )

    save_prm(model, ckpt_dir, identity)
    write_json(ckpt_dir / "config.json", cfg.to_dict())
    finalize_manifest(ckpt_dir)

    eval_res = evaluate_all(model, tokenizer, data_dir, device, attach_asin_context, cfg.max_length)
    summary = {
        "arm": arm_name,
        "target_modules": list(target_modules),
        "attach_asin_context": attach_asin_context,
        "steps": steps,
        "accumulation": accumulation,
        "lora_lr": lora_lr,
        "head_lr": head_lr,
        "retained_fit": len(encoded),
        "rejected_fit": len(rejected),
        "elapsed_seconds": time.monotonic() - t0,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None,
        "last_5_records": records[-5:],
        "evaluation": eval_res,
    }
    write_json(out_root / f"{arm_name}_result.json", summary)
    p32 = eval_res["dev32_pairs"]
    r7544 = eval_res["regression_7544"]
    d160 = eval_res["dev160"]
    print(
        f"==> [{arm_name} DONE in {summary['elapsed_seconds']:.1f}s] "
        f"dev32_acc={p32['correct']}/{p32['directional_n']} ({p32['accuracy']*100:.1f}%) "
        f"by_vis={p32['by_visibility']} mean_gap={p32['mean_high_minus_low_gap']:.5f} | "
        f"7544_click_gap={r7544['click_step_gap_shoes_minus_clogs']:+.4f} | "
        f"dev160_mse={d160['model']['mse']:.5f} (const={d160['baselines']['constant']:.5f})",
        flush=True,
    )
    return cfg, ckpt_dir, summary


def train_pairs20_continuation(
    arm_name: str,
    parent_cfg: PRMConfig,
    parent_ckpt: Path,
    data_dir: Path,
    out_root: Path,
    device: str,
    attach_asin_context: bool,
    steps: int = 24,
    pairs_per_step: int = 4,
    lora_lr: float = 1e-4,
    head_lr: float = 2e-5,
    diff_weight: float = 1.0,
    seed: int = 42,
):
    t0 = time.monotonic()
    ckpt_dir = out_root / arm_name
    cfg_dict = parent_cfg.to_dict()
    cfg_dict["output_dir"] = str(ckpt_dir)
    cfg = PRMConfig.from_dict(cfg_dict)

    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer, identity, _ = build_prm(cfg, parent_ckpt, device, True)
    raw_pairs = load_jsonl(data_dir / "training_pairs20.jsonl")
    pad_id = tokenizer.pad_token_id
    encoded_pairs = []
    for row in raw_pairs:
        h_enc, _ = encode_rows(tokenizer, [row["target"]], cfg.max_length, True, attach_asin_context)
        l_enc, _ = encode_rows(tokenizer, [row["alternative"]], cfg.max_length, True, attach_asin_context)
        if h_enc and l_enc:
            encoded_pairs.append((int(row["branch"]), h_enc[0][1], l_enc[0][1]))

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    head_params = [p for n, p in trainable if n.startswith("score_head.")]
    lora_params = [p for n, p in trainable if "lora_" in n]
    base_lrs = [head_lr, lora_lr]
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": head_lr}, {"params": lora_params, "lr": lora_lr}],
        weight_decay=0.01,
    )

    records = []
    for step_idx in range(steps):
        mult = cosine_lr_multiplier(step_idx, steps)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * mult
        batch = []
        for i in range(pairs_per_step):
            br, he, le = encoded_pairs[(step_idx * pairs_per_step + i) % len(encoded_pairs)]
            batch.append((br, tensor_batch([he], pad_id, device, True), tensor_batch([le], pad_id, device, True)))
        rep = prefix_value_pair_step(model, batch, optimizer, diff_weight=diff_weight, max_grad_norm=1.0)
        rep["step"] = step_idx + 1
        records.append(rep)
        if (step_idx + 1) % max(1, steps // 4) == 0 or step_idx == 0 or step_idx + 1 == steps:
            print(
                f"[{arm_name}] step {step_idx+1}/{steps} | pair_loss={rep['pair_loss']:.5f} "
                f"| train_gap={rep['mean_train_gap']:+.4f} | gnorm={rep['gradient_norm']:.2f}",
                flush=True,
            )

    save_prm(model, ckpt_dir, identity)
    write_json(ckpt_dir / "config.json", cfg.to_dict())
    finalize_manifest(ckpt_dir)

    eval_res = evaluate_all(model, tokenizer, data_dir, device, attach_asin_context, cfg.max_length)
    summary = {
        "arm": arm_name,
        "parent": str(parent_ckpt.name),
        "attach_asin_context": attach_asin_context,
        "steps": steps,
        "pairs_per_step": pairs_per_step,
        "diff_weight": diff_weight,
        "elapsed_seconds": time.monotonic() - t0,
        "last_5_records": records[-5:],
        "evaluation": eval_res,
    }
    write_json(out_root / f"{arm_name}_result.json", summary)
    p32 = eval_res["dev32_pairs"]
    r7544 = eval_res["regression_7544"]
    d160 = eval_res["dev160"]
    print(
        f"==> [{arm_name} DONE in {summary['elapsed_seconds']:.1f}s] "
        f"dev32_acc={p32['correct']}/{p32['directional_n']} ({p32['accuracy']*100:.1f}%) "
        f"by_vis={p32['by_visibility']} mean_gap={p32['mean_high_minus_low_gap']:.5f} | "
        f"7544_click_gap={r7544['click_step_gap_shoes_minus_clogs']:+.4f} | "
        f"dev160_mse={d160['model']['mse']:.5f}",
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--steps", type=int, default=126)
    parser.add_argument("--pair-steps", type=int, default=24)
    parser.add_argument("--accumulation", type=int, default=8)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    base_cfg_dict = load_config(args.base_config).to_dict()
    base_cfg_dict["schema_version"] = "prm-qwen-config-v2"
    base_cfg_dict["representation"] = "final_hidden_state"

    all_summaries = {}

    # Arm 1: Pure single-factor loss change (Prefix-Value MSE + Telescoping) on fit501, q_proj/v_proj only
    cfg1, ckpt1, sum1 = train_fit_arm(
        arm_name="arm1_pv_qv_fit501",
        base_cfg_dict=base_cfg_dict,
        data_dir=args.data_dir,
        out_root=args.out_root,
        device=args.device,
        target_modules=("q_proj", "v_proj"),
        attach_asin_context=False,
        steps=args.steps,
        accumulation=args.accumulation,
        lora_lr=2e-4,
        head_lr=2e-5,
    )
    all_summaries["arm1_pv_qv_fit501"] = sum1

    # Arm 2: Prefix-Value MSE + Full-linear LoRA (q,k,v,o,gate,up,down) on fit501 (raw text)
    cfg2, ckpt2, sum2 = train_fit_arm(
        arm_name="arm2_pv_alllinear_fit501",
        base_cfg_dict=base_cfg_dict,
        data_dir=args.data_dir,
        out_root=args.out_root,
        device=args.device,
        target_modules=ALL_LINEAR_MODULES,
        attach_asin_context=False,
        steps=args.steps,
        accumulation=args.accumulation,
        lora_lr=2e-4,
        head_lr=2e-5,
    )
    all_summaries["arm2_pv_alllinear_fit501"] = sum2

    # Arm 3: +24-step continuation from Arm 2 on frozen #57 20 training pairs (still pure terminal R only)
    sum3 = train_pairs20_continuation(
        arm_name="arm3_pv_alllinear_pairs20",
        parent_cfg=cfg2,
        parent_ckpt=ckpt2,
        data_dir=args.data_dir,
        out_root=args.out_root,
        device=args.device,
        attach_asin_context=False,
        steps=args.pair_steps,
    )
    all_summaries["arm3_pv_alllinear_pairs20"] = sum3

    # Arm 4: Prefix-Value MSE + deterministic pre-click public ASIN snippet from messages[i-1] (fit501 only)
    cfg4, ckpt4, sum4 = train_fit_arm(
        arm_name="arm4_pv_asin_context_fit501",
        base_cfg_dict=base_cfg_dict,
        data_dir=args.data_dir,
        out_root=args.out_root,
        device=args.device,
        target_modules=ALL_LINEAR_MODULES,
        attach_asin_context=True,
        steps=args.steps,
        accumulation=args.accumulation,
        lora_lr=2e-4,
        head_lr=2e-5,
    )
    all_summaries["arm4_pv_asin_context_fit501"] = sum4

    # Arm 5: +24-step continuation from Arm 4 on frozen #57 20 training pairs
    sum5 = train_pairs20_continuation(
        arm_name="arm5_pv_asin_context_pairs20",
        parent_cfg=cfg4,
        parent_ckpt=ckpt4,
        data_dir=args.data_dir,
        out_root=args.out_root,
        device=args.device,
        attach_asin_context=True,
        steps=args.pair_steps,
    )
    all_summaries["arm5_pv_asin_context_pairs20"] = sum5

    write_json(args.out_root / "all_arms_summary.json", all_summaries)
    print("=== ALL ARMS COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
