"""Offline verifier and metric recomputation for PR #54 three-arm PRM evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_ARMS_DIR = HERE / "rtx3090-fp32-three-arms"

EXPECTED_SAMPLES_SHA = "9c88beffaef277f519712ab4a07c45766212579770d6292572c7f189bca4d8fe"
EXPECTED_TRAIN_IDS_SHA = "8c85e56b3ac2e1f0a09b54560ad0f9bf77eedddb241d73617c7ff32416522e20"
EXPECTED_BASE_SHA = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
EXPECTED_B1_SHA = "c39e2a78e7c5973328e9d2c23fb6320e566352e63cc089562488f3dd46ae5f9c"
EXPECTED_TRAIN_PY_SHA = "d71a30626cb4a747837a09a9e3814abc4bcccb2fbad2d2da6fe98febf4ab0ac8"
EXPECTED_BUNDLE_SHA = "6ffeaba47f6586e54efd3ab13fef9890d2d3bd1bad6386bbe597aa8b349bb6f5"
BUNDLE_PREFIX = "prm-split-cosine-rtx3090-20261124"

# Locked ground-truth labels and assistant turn counts in canonical file order of samples.jsonl (9c88beff...)
CANONICAL_EIGHT = [
    (351, 3, 2.0 / 3.0),
    (95, 3, 0.0),
    (5478, 10, 0.0),
    (3542, 10, 0.0),
    (359, 3, 2.0 / 3.0),
    (5031, 3, 0.0),
    (0, 3, 1.0),
    (6225, 3, 1.0),
]
ARMS = ("hidden-lr-low-acc8", "hidden-split-cos-acc8", "hidden-split-cos-acc1")
BINARY_PAYLOADS = {"adapter/adapter_model.safetensors", "head.pt", "training-state.pt"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def cosine_mult(step_idx: int, total_steps: int) -> float:
    return 0.5 * (1.0 + math.cos(math.pi * step_idx / total_steps))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_turn_end_indices(indices: object, expected_turns: int, max_length: int, label: str) -> list[int]:
    _require(isinstance(indices, list), f"{label}: turn_end_indices must be a list")
    _require(expected_turns >= 1 and len(indices) == expected_turns,
             f"{label}: turn_end_indices length {len(indices)} != expected_turns {expected_turns}")
    for pos in indices:
        _require(type(pos) is int and 0 <= pos < max_length,
                 f"{label}: invalid score position {pos!r} for max_length={max_length}")
    _require(all(indices[i + 1] > indices[i] for i in range(len(indices) - 1)),
             f"{label}: turn_end_indices must be strictly increasing")
    return indices


def validate_scores(scores: object, total: object, expected_turns: int, label: str) -> tuple[list[float], float]:
    _require(isinstance(scores, list) and len(scores) == expected_turns,
             f"{label}: step_scores length mismatch")
    for val in scores:
        _require(type(val) in (int, float) and math.isfinite(val), f"{label}: non-finite step score")
    _require(type(total) in (int, float) and math.isfinite(total), f"{label}: non-finite total")
    float_scores = [float(v) for v in scores]
    float_total = float(total)
    _require(abs(math.fsum(float_scores) - float_total) <= 1e-5,
             f"{label}: sum(step_scores) does not match total")
    return float_scores, float_total


def reconstruct_expected_positions(
    samples_path: Path,
    tokenizer_path: Path,
    expected_tokenizer_sha: dict[str, str],
    max_length: int,
) -> dict[int, list[int]]:
    for name, exp_sha in expected_tokenizer_sha.items():
        fpath = tokenizer_path / name
        _require(fpath.is_file() and sha256(fpath) == exp_sha, f"Tokenizer file hash mismatch: {name}")
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from transformers import AutoTokenizer
    from webshop_lab.prm.tokenize import tokenize_trajectory

    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False, use_fast=True)
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    _require(type(im_end_id) is int and im_end_id >= 0, "Invalid <|im_end|> token id")

    expected_map: dict[int, list[int]] = {}
    raw_rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(len(raw_rows) == 8, "Expected exactly 8 trajectories in samples.jsonl")
    for row in raw_rows:
        enc = tokenize_trajectory(tok, row["messages"], max_length)
        positions = enc["turn_end_indices"]
        _require(all(enc["input_ids"][p] == im_end_id for p in positions),
                 f"task {row['task_id']}: reconstructed position is not <|im_end|>")
        expected_map[int(row["task_id"])] = list(positions)
    return expected_map


def verify_arm(
    arm: str,
    arms_dir: Path,
    repo_root: Path,
    labels_by_task: list[tuple[int, int, float]],
    reconstructed_positions: dict[int, list[int]] | None = None,
    bundle_zf: zipfile.ZipFile | None = None,
) -> dict:
    ckpt = arms_dir / arm
    manifest = json.loads((ckpt / "prm-manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((ckpt / "summary.json").read_text(encoding="utf-8"))
    ref_preds = json.loads((ckpt / "reference-predictions.json").read_text(encoding="utf-8"))
    reload_report = json.loads((arms_dir / f"{arm}-reload.json").read_text(encoding="utf-8"))
    attempt = json.loads((arms_dir / f"{arm}.attempt.json").read_text(encoding="utf-8"))

    _require(attempt.get("status") == "complete" and attempt.get("test_authorization") is False,
             f"{arm}: invalid attempt status")
    _require(manifest.get("schema_version") == 2, f"{arm}: expected schema_version=2")
    _require(manifest.get("architecture", {}).get("representation") == "final_hidden_state",
             f"{arm}: expected final_hidden_state")
    _require(manifest.get("architecture", {}).get("input_dim") == 1536, f"{arm}: expected input_dim=1536")

    ident = manifest["identity"]
    max_length = int(ident["max_length"])
    _require(max_length == 2048, f"{arm}: unexpected max_length={max_length}")
    _require(ident["base_sha256"] == EXPECTED_BASE_SHA, f"{arm}: base_sha256 mismatch")
    _require(ident["initialization"]["sft_adapter_files_sha256"]["adapter_model.safetensors"] == EXPECTED_B1_SHA,
             f"{arm}: B1 adapter SHA mismatch")
    _require(summary["samples_sha256"] == EXPECTED_SAMPLES_SHA, f"{arm}: samples_sha256 mismatch")
    _require(summary["train_ids_sha256"] == EXPECTED_TRAIN_IDS_SHA, f"{arm}: train_ids_sha256 mismatch")
    _require(summary["retained_samples"] == 8 and summary["rejected"] == [], f"{arm}: expected 8/8 retained")

    for src_name, src_digest in ident["source_sha256"].items():
        actual = sha256(repo_root / "src/webshop_lab/prm" / src_name)
        _require(actual == src_digest, f"{arm}: source mismatch for {src_name}: {actual} != {src_digest}")

    sums = {}
    for line in (ckpt / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            sums[rel] = digest
    _require(sums.get("prm-manifest.json") == sha256(ckpt / "prm-manifest.json"),
             f"{arm}: prm-manifest.json hash mismatch in SHA256SUMS")
    for rel, expected_digest in manifest["files_sha256"].items():
        _require(sums.get(rel) == expected_digest, f"{arm}: SHA256SUMS mismatch on {rel}")
        local_file = ckpt / rel
        if rel not in BINARY_PAYLOADS:
            _require(local_file.is_file(), f"{arm}: missing tracked evidence file {local_file}")
            _require(sha256(local_file) == expected_digest, f"{arm}: SHA mismatch on {local_file}")
        elif bundle_zf is not None:
            arcname = f"{BUNDLE_PREFIX}/{arm}/{rel}"
            _require(arcname in bundle_zf.namelist(), f"{arm}: missing binary {arcname} in bundle")
            actual_bin_sha = sha256_bytes(bundle_zf.read(arcname))
            _require(actual_bin_sha == expected_digest,
                     f"{arm}: binary SHA mismatch on {rel}: {actual_bin_sha} != {expected_digest}")

    _require(reload_report.get("reference_match") is True, f"{arm}: reload reference_match is not True")
    reload_preds = reload_report.get("predictions")
    _require(isinstance(ref_preds, list) and isinstance(reload_preds, list), f"{arm}: predictions must be lists")
    _require(len(ref_preds) == len(reload_preds) == len(labels_by_task) == 8, f"{arm}: expected 8 predictions")

    max_reload_step_diff = 0.0
    max_reload_total_diff = 0.0
    per_sample = []
    sq_errors = []
    abs_errors = []

    for (expected_tid, expected_turns, label), ref_row, rel_row in zip(labels_by_task, ref_preds, reload_preds):
        _require(isinstance(ref_row, dict) and isinstance(rel_row, dict), f"{arm}: prediction row must be dict")
        _require(ref_row.get("task_id") == expected_tid and rel_row.get("task_id") == expected_tid,
                 f"{arm}: task_id mismatch for {expected_tid}")
        ref_pos = validate_turn_end_indices(
            ref_row.get("turn_end_indices"), expected_turns, max_length, f"{arm}/ref/task-{expected_tid}"
        )
        rel_pos = validate_turn_end_indices(
            rel_row.get("turn_end_indices"), expected_turns, max_length, f"{arm}/reload/task-{expected_tid}"
        )
        _require(ref_pos == rel_pos, f"{arm}/task-{expected_tid}: ref vs reload turn_end_indices mismatch")
        if reconstructed_positions is not None:
            _require(ref_pos == reconstructed_positions[expected_tid],
                     f"{arm}/task-{expected_tid}: turn_end_indices do not match tokenizer <|im_end|> positions")

        ref_scores, ref_total = validate_scores(
            ref_row.get("step_scores"), ref_row.get("total"), expected_turns, f"{arm}/ref/task-{expected_tid}"
        )
        rel_scores, rel_total = validate_scores(
            rel_row.get("step_scores"), rel_row.get("total"), expected_turns, f"{arm}/reload/task-{expected_tid}"
        )
        for a, b in zip(ref_scores, rel_scores):
            d = abs(a - b)
            max_reload_step_diff = max(max_reload_step_diff, d)
            _require(d <= 1e-5 + 1e-5 * abs(a), f"{arm}/task-{expected_tid}: step reload tolerance exceeded")
        td = abs(ref_total - rel_total)
        max_reload_total_diff = max(max_reload_total_diff, td)
        _require(td <= 1e-5 + 1e-5 * abs(ref_total), f"{arm}/task-{expected_tid}: total reload tolerance exceeded")

        err = abs(ref_total - label)
        abs_errors.append(err)
        sq_errors.append((ref_total - label) ** 2)
        per_sample.append({
            "task_id": expected_tid,
            "turns": expected_turns,
            "turn_end_indices": ref_pos,
            "label": label,
            "pred_total": ref_total,
            "abs_error": err,
            "step_scores": ref_scores,
        })

    opt = summary["optimization"]
    records = summary["records"]
    _require(len(records) == summary["optimizer_steps"], f"{arm}: record count mismatch")
    _require(summary["samples_consumed"] == summary["optimizer_steps"] * opt["gradient_accumulation_steps"],
             f"{arm}: samples_consumed mismatch")
    for idx, rec in enumerate(records):
        _require(rec["optimizer_step"] == idx + 1, f"{arm}: optimizer_step index mismatch")
        mult = cosine_mult(idx, opt["cosine_total_steps"]) if opt["lr_scheduler"] == "cosine" else 1.0
        _require(math.isclose(rec["learning_rate"], opt["learning_rate"] * mult, rel_tol=1e-12, abs_tol=0.0),
                 f"{arm}: LoRA lr schedule mismatch at step {idx + 1}")
        _require(math.isclose(rec["head_learning_rate"], opt["head_learning_rate"] * mult, rel_tol=1e-12, abs_tol=0.0),
                 f"{arm}: head lr schedule mismatch at step {idx + 1}")

    mse = math.fsum(sq_errors) / len(sq_errors)
    max_err = max(abs_errors)
    by_tid = {r["task_id"]: r["pred_total"] for r in per_sample}
    full_3turn_mean = 0.5 * (by_tid[0] + by_tid[6225])
    zero_3turn_mean = 0.5 * (by_tid[95] + by_tid[5031])

    return {
        "arm": arm,
        "optimizer_steps": summary["optimizer_steps"],
        "samples_consumed": summary["samples_consumed"],
        "optimization": opt,
        "clipped_updates": sum(1 for r in records if r["gradient_clipped"]),
        "elapsed_seconds": summary["elapsed_seconds"],
        "peak_allocated_gib": summary["peak_allocated_gib"],
        "peak_reserved_gib": summary["peak_reserved_gib"],
        "max_reload_step_diff": max_reload_step_diff,
        "max_reload_total_diff": max_reload_total_diff,
        "tokenizer_im_end_verified": reconstructed_positions is not None,
        "binary_payload_verified": bundle_zf is not None,
        "mse": mse,
        "max_abs_error": max_err,
        "memory_gate_passed": mse < 0.001 and max_err < 0.1,
        "three_turn_full_mean_0_6225": full_3turn_mean,
        "three_turn_zero_mean_95_5031": zero_3turn_mean,
        "three_turn_full_minus_zero": full_3turn_mean - zero_3turn_mean,
        "binary_payload_sha256": {k: manifest["files_sha256"][k] for k in sorted(BINARY_PAYLOADS)},
        "per_sample": per_sample,
    }


def verify_evidence(
    arms_dir: Path = DEFAULT_ARMS_DIR,
    repo_root: Path = REPO_ROOT,
    samples_path: Path | None = None,
    tokenizer_path: Path | None = None,
    bundle_path: Path | None = None,
) -> dict:
    actual_train_py = sha256(repo_root / "src/webshop_lab/prm/train.py")
    _require(actual_train_py == EXPECTED_TRAIN_PY_SHA, f"train.py SHA mismatch: {actual_train_py}")
    _require(sha256(repo_root / "results/env-2026-11-17/train_ids.json") == EXPECTED_TRAIN_IDS_SHA,
             "train_ids.json SHA mismatch")

    labels = list(CANONICAL_EIGHT)
    reconstructed: dict[int, list[int]] | None = None
    if samples_path is not None:
        _require(sha256(samples_path) == EXPECTED_SAMPLES_SHA, "samples.jsonl SHA-256 mismatch")
        raw = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        _require(
            [(r["task_id"], sum(1 for m in r["messages"] if m["role"] == "assistant"), float(r["terminal_reward"])) for r in raw]
            == [(tid, turns, lbl) for tid, turns, lbl in labels],
            "samples.jsonl task_id/turn_count/label sequence mismatch",
        )
        if tokenizer_path is not None:
            sample_manifest = json.loads((arms_dir / ARMS[0] / "prm-manifest.json").read_text(encoding="utf-8"))
            reconstructed = reconstruct_expected_positions(
                samples_path, tokenizer_path, sample_manifest["identity"]["tokenizer_sha256"], 2048
            )
    elif tokenizer_path is not None:
        raise ValueError("--tokenizer requires --samples so positions can be reconstructed from raw messages")

    if bundle_path is not None:
        actual_bundle_sha = sha256(bundle_path)
        _require(actual_bundle_sha == EXPECTED_BUNDLE_SHA,
                 f"Bundle ZIP SHA-256 mismatch: {actual_bundle_sha} != {EXPECTED_BUNDLE_SHA}")
        with zipfile.ZipFile(bundle_path, "r") as zf:
            arms_report = [verify_arm(arm, arms_dir, repo_root, labels, reconstructed, zf) for arm in ARMS]
    else:
        arms_report = [verify_arm(arm, arms_dir, repo_root, labels, reconstructed, None) for arm in ARMS]

    return {
        "verified_train_py_sha256": actual_train_py,
        "samples_sha256": EXPECTED_SAMPLES_SHA,
        "train_ids_sha256": EXPECTED_TRAIN_IDS_SHA,
        "bundle_sha256": EXPECTED_BUNDLE_SHA if bundle_path is not None else None,
        "tokenizer_im_end_verified": reconstructed is not None,
        "arms": arms_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, help="Optional path to samples.jsonl (9c88beff...) to cross-check raw labels")
    parser.add_argument("--tokenizer", type=Path, help="Optional path to pinned Qwen tokenizer snapshot to verify <|im_end|> indices")
    parser.add_argument("--bundle", type=Path, help="Optional path to prm-split-cosine-rtx3090-20261124.zip to verify binary weights")
    args = parser.parse_args()

    report = verify_evidence(
        samples_path=args.samples,
        tokenizer_path=args.tokenizer,
        bundle_path=args.bundle,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
