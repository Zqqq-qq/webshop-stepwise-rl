"""Reproducible, CPU-only PRM head diagnostic for Issue #48.

Consumes How's *re-extracted* fixed B1 logits, never a live model, LoRA,
dev/test data or GPU. The historical neural-training gate is NOT inferred
from this fixed-feature diagnostic. Only a verified supplementary ZIP is used.
"""

import argparse
import hashlib
import io
import json
import math
import platform
from pathlib import Path
import zipfile


ARCHIVE_SHA256 = "2d1c2763783759adb47bf8eeb056cc36b7134cd030a2f6b274ea32df68a86aa4"
PREFIX = "prm-evidence-supplement-20261124/"
EXPECTED_TASKS = [351, 95, 5478, 3542, 359, 5031, 0, 6225]
EXPECTED_TURNS = [3, 3, 10, 10, 3, 3, 3, 3]
EXPECTED_STATE_SHA256 = "c19461e3b22b822e5162cc3fb93e0825c5e398fddd26c55aa6a1326b87895c3c"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def get_archive_files(path):
    data = Path(path).read_bytes()
    if digest(data) != ARCHIVE_SHA256:
        raise ValueError("Supplement ZIP SHA256 mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            not name.startswith(PREFIX) or ".." in Path(name).parts or name.endswith("/")
            for name in names
        ):
            raise ValueError("Unexpected or unsafe ZIP member")
        files = {name[len(PREFIX):]: archive.read(name) for name in names}
    manifest = files["SHA256SUMS"].decode("utf-8")
    expected = {}
    for line in manifest.splitlines():
        sha, filename = line.split(None, 1)
        filename = filename.strip()
        if filename in expected or len(sha) != 64:
            raise ValueError("Duplicate or malformed checksum entry")
        expected[filename] = sha
    if set(expected) != set(files) - {"SHA256SUMS"}:
        raise ValueError("Supplement members do not match checksum manifest")
    for name, sha in expected.items():
        if digest(files[name]) != sha:
            raise ValueError("Failed per-file SHA256: " + name)
    if expected["step-512/training-state.pt"] != EXPECTED_STATE_SHA256:
        raise ValueError("Unexpected original step-512 optimizer checkpoint")
    return files, expected


def load_fixed_features(files):
    import numpy as np

    meta = json.loads(files["representation-diagnostic/F.meta.json"])
    rows = meta["samples_order"]
    if ([r["task_id"] for r in rows] != EXPECTED_TASKS or
            [r["turns"] for r in rows] != EXPECTED_TURNS or
            [r["row"] for r in rows] != list(range(8)) or
            meta["summation_precision"] != "FP32 logits summed in float32 then cast to float64" or
            meta["eval_mode"] is not True):
        raise ValueError("Feature extraction identity/order is incompatible")
    stored = np.load(io.BytesIO(files["representation-diagnostic/F.npy"]), allow_pickle=False)
    if stored.shape != (8, 151936) or stored.dtype != np.float64 or not np.isfinite(stored).all():
        raise ValueError("Expected stored (8, 151936) float64 summed logits, without N")
    labels64 = np.asarray([r["label"] for r in rows], dtype=np.float64)
    if not np.isfinite(labels64).all() or not np.all((0 <= labels64) & (labels64 <= 1)):
        raise ValueError("Labels must be finite [0,1] terminal rewards")
    # Supplement meta incorrectly lists this final N column as *already* stored.
    # F.npy really holds only 151936 summed logits columns, verified above.
    augmented64 = np.column_stack((stored, np.asarray(EXPECTED_TURNS, dtype=np.float64)))
    return augmented64.astype(np.float32), labels64.astype(np.float32), augmented64, labels64


def audit_svd(augmented64, labels64):
    import numpy as np

    u, singular, vt = np.linalg.svd(augmented64, full_matrices=False)
    y = labels64.astype(np.float64)
    solution = vt.T @ ((u.T @ y) / singular)
    residual = augmented64 @ solution - y
    fp32_error = augmented64.astype(np.float32) @ solution.astype(np.float32) - y
    return {
        "stored_F_shape": [8, 151936],
        "audited_A_shape": [8, 151937],
        "bias_column_turn_counts": EXPECTED_TURNS,
        "rcond": 1e-10,
        "rank": int((singular > singular[0] * 1e-10).sum()),
        "nonzero_spectrum_condition": float(singular[0] / singular[-1]),
        "solution_l2": float(np.linalg.norm(solution)),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "max_abs_fp32_prediction_error": float(np.max(np.abs(fp32_error))),
        "source": "re-extracted fixed B1 representation, NOT contemporaneous F",
        "meaning": "eight-row linear representability only; no training/generalization acceptance",
    }


def audit_optimizer_state(blob):
    import torch

    # Explicit weights_only eliminates arbitrary pickle globals; map all GPU
    # storages to CPU, with no checkpoint restore into a trainable model.
    root = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    state = root["optimizer"]
    groups = state["param_groups"]
    if len(groups) != 1:
        raise ValueError("Expected one saved AdamW parameter group")
    group = groups[0]
    if (group["lr"] != 1e-7 or group["weight_decay"] != 0.01 or
            tuple(group["betas"]) != (0.9, 0.999) or group["eps"] != 1e-8):
        raise ValueError("Historical hyperparameters do not match saved state")
    entries = state["state"]
    if len(entries) != 114 or int(root["completed_steps"]) != 512:
        raise ValueError("Unexpected historical optimizer states/step count")
    if len(group["params"]) != 114 or set(group["params"]) != set(entries):
        raise ValueError("Parameter-state ID set mismatch")
    if root["samples_sha256"] != "9c88beffaef277f519712ab4a07c45766212579770d6292572c7f189bca4d8fe":
        raise ValueError("Training task evidence digest mismatch")
    for value in entries.values():
        if int(value["step"]) != 512:
            raise ValueError("Inconsistent saved Adam step counter")
        avg, sq = value["exp_avg"], value["exp_avg_sq"]
        if avg.dtype != torch.float32 or sq.dtype != torch.float32:
            raise ValueError("Adam moments are not FP32")
        if not torch.isfinite(avg).all() or not torch.isfinite(sq).all() or bool((sq < 0).any()):
            raise ValueError("Non-finite or negative saved Adam variance")
    return {
        "sha256": digest(blob),
        "state_entry_count": len(entries),
        "completed_steps": int(root["completed_steps"]),
        "moment_dtype": "float32",
        "moment_values_finite": True,
        "second_moments_nonnegative": True,
        "learning_rate": group["lr"],
        "weight_decay": group["weight_decay"],
        "betas": list(group["betas"]),
        "eps": group["eps"],
        "identity_limitation": "no independently persisted optimizer-ID to model-name mapping; no full-model restore",
    }


def score_predictions(predictions, labels):
    if len(predictions) != len(labels) or len(labels) != 8:
        raise ValueError("Must score all eight fixed training examples")
    errors = [float(p) - float(y) for p, y in zip(predictions, labels)]
    if not all(math.isfinite(e) for e in errors):
        raise ValueError("Non-finite prediction")
    return sum(e * e for e in errors) / 8, max(abs(e) for e in errors)


def train_head(features32, labels32, lr, batch_size, max_exposures=512):
    import torch

    if batch_size not in (1, 8) or max_exposures % (64 * batch_size // 8):
        raise ValueError("Unsupported batch size/exposure budget")
    x = torch.from_numpy(features32.copy())
    y = torch.from_numpy(labels32.copy())
    w = torch.nn.Parameter(torch.zeros(x.shape[1], dtype=torch.float32))
    optimizer = torch.optim.AdamW([w], lr=lr, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.01)
    history, clip_events = [], 0
    steps = exposures = 0
    stop_reason = "exposure_cap"
    while exposures < max_exposures:
        idx = torch.arange(exposures, exposures + batch_size) % 8
        optimizer.zero_grad(set_to_none=True)
        loss = ((x[idx] @ w - y[idx]) ** 2).mean()
        if not bool(torch.isfinite(loss)):
            stop_reason = "nonfinite_loss"
            break
        loss.backward()
        try:
            norm = torch.nn.utils.clip_grad_norm_([w], 1.0, error_if_nonfinite=True)
        except RuntimeError:
            stop_reason = "nonfinite_gradient"
            break
        if float(norm) > 1.0:
            clip_events += 1
        optimizer.step()
        steps += 1
        exposures += batch_size
        if not bool(torch.isfinite(w).all()):
            stop_reason = "nonfinite_parameter"
            break
        if exposures % 64 == 0:
            with torch.no_grad():
                predictions = (x @ w).tolist()
            try:
                mse, max_error = score_predictions(predictions, labels32.tolist())
            except ValueError:
                stop_reason = "nonfinite_prediction"
                break
            history.append({
                "sample_exposures": exposures,
                "optimizer_steps": steps,
                "clip_events": clip_events,
                "mse": mse,
                "max_abs_error": max_error,
                "predictions": predictions,
            })
            if mse < 0.001 and max_error < 0.1:
                stop_reason = "memory_gate_passed"
                break
    return {
        "learning_rate": lr,
        "batch_size": batch_size,
        "sample_exposures": exposures,
        "optimizer_steps": steps,
        "clip_events": clip_events,
        "stop_reason": stop_reason,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True,
                        help="Locally downloaded pinned supplementary ZIP")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.archive.resolve():
        parser.error("Output must be a new path distinct from the evidence")

    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    files, hashes = get_archive_files(args.archive)
    x, y, a64, y64 = load_fixed_features(files)
    svd = audit_svd(a64, y64)
    optimizer_state = audit_optimizer_state(files["step-512/training-state.pt"])
    runs = [train_head(x, y, lr, 1) for lr in (1e-7, 3e-7, 1e-6)]
    if any(not run["history"] for run in runs):
        raise ValueError("A non-finite run stopped before the first diagnostic interval")
    # No dev/test selection: pick the lowest *final training-fit* MSE.
    chosen = min(runs, key=lambda r: r["history"][-1]["mse"])
    if not any(r["stop_reason"] == "memory_gate_passed" for r in runs):
        runs.append(train_head(x, y, chosen["learning_rate"], 8))
    for run in runs:
        for record in run["history"]:
            mse, err = score_predictions(record["predictions"], y.tolist())
            if abs(mse - record["mse"]) > 1e-10 or abs(err - record["max_abs_error"]) > 1e-10:
                raise ValueError("Independent prediction/label metric replay failed")
    baseline = runs[0]["history"][-1]
    if abs(baseline["mse"] - 0.13999) > 0.002 or abs(baseline["max_abs_error"] - 0.64871) > 0.02:
        raise ValueError("PyTorch CPU reference diverges from published clip=1 diagnostic")
    output = {
        "schema_version": "prm-cpu-fixed-feature-head-v1",
        "scope": "fixed B1 feature / head-only CPU; no model forward, LoRA, GPU, dev/test",
        "source_provenance": "F.npy is re-extracted, not saved from historical training run",
        "archive_sha256": ARCHIVE_SHA256,
        "verified_member_sha256": hashes,
        "task_order": EXPECTED_TASKS,
        "labels_float32": y.tolist(),
        "train_features_shape": list(x.shape),
        "svd": svd,
        "historical_step512_optimizer_audit": optimizer_state,
        "optimizer": {"name": "torch.optim.AdamW", "initialization": "zeros", "dtype": "float32",
                      "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01,
                      "clip_norm": 1.0, "sample_order": "8 historical examples in listed order"},
        "memory_gate": {"mse_strictly_less_than": 0.001,
                        "max_abs_error_strictly_less_than": 0.1},
        "selection": {"criterion": "minimum final train MSE, no dev/test",
                      "learning_rate": chosen["learning_rate"]},
        "runs": runs,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": torch.__version__},
        "limitations": [
            "Fixed features are re-extracted from B1, not an original same-run F snapshot.",
            "A full-rank eight-row linear fit is not proof of successful gradient memorization.",
            "Batch comparisons match sample exposures but not optimizer step counts.",
            "No GPU neural training, generalization or step attribution was performed.",
            "Independent reviewer must sign off before any GPU execution order.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    for run in runs:
        print(json.dumps({k: run[k] for k in ("learning_rate", "batch_size", "stop_reason",
                                                "optimizer_steps", "clip_events")}
                         | {"final": run["history"][-1]}))
    print("Validated report:", args.output)


if __name__ == "__main__":
    main()
