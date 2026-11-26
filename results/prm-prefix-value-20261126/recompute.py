#!/usr/bin/env python3
"""Zero-dependency offline verification and metric recomputation for Prefix-Value PRM 5-arm ablation."""
import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

EXPECTED_SCRIPT_SHA = "c1de7a9e1fcf1c6fcee0a0e1064253f9c132eb00824096bbd3d2516edc24a7a8"
EXPECTED_BUNDLE_SHA = "3b53a6615d217f8c97b285a9a148bfa686762b81bcc9aa8e371684b8c6f0b43f"
ARMS = (
    "arm1_pv_qv_fit501",
    "arm2_pv_alllinear_fit501",
    "arm3_pv_alllinear_pairs20",
    "arm4_pv_asin_context_fit501",
    "arm5_pv_asin_context_pairs20",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_all(repo_root: Path, bundle_path: Path | None = None) -> dict:
    script_path = repo_root / "scripts/run_prefix_value_prm_ablation.py"
    script_sha = sha256_file(script_path)
    if script_sha != EXPECTED_SCRIPT_SHA:
        raise ValueError(f"Script SHA mismatch: {script_sha} != {EXPECTED_SCRIPT_SHA}")

    ev_root = repo_root / "results/prm-prefix-value-20261126/rtx3090-five-arms"
    all_sum = json.loads((ev_root / "all_arms_summary.json").read_text(encoding="utf-8"))

    bundle_hashes = {}
    bundle_sha = None
    if bundle_path is not None:
        bundle_sha = sha256_file(bundle_path)
        if bundle_sha != EXPECTED_BUNDLE_SHA:
            raise ValueError(f"Bundle outer SHA mismatch: {bundle_sha} != {EXPECTED_BUNDLE_SHA}")
        with zipfile.ZipFile(bundle_path, "r") as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    bundle_hashes[info.filename] = sha256_bytes(zf.read(info.filename))

    arm_reports = {}
    for arm in ARMS:
        ckpt = ev_root / arm
        manifest = json.loads((ckpt / "prm-manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 2:
            raise ValueError(f"{arm}: expected schema_version=2")
        if manifest.get("architecture", {}).get("representation") != "final_hidden_state":
            raise ValueError(f"{arm}: expected final_hidden_state")

        sums = {}
        for line in (ckpt / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, rel = line.strip().split("  ", 1)
                sums[rel] = digest

        if sums.get("prm-manifest.json") != sha256_file(ckpt / "prm-manifest.json"):
            raise ValueError(f"{arm}: prm-manifest.json SHA mismatch against SHA256SUMS")

        files_sha = manifest["files_sha256"]
        for rel, expected in files_sha.items():
            if sums.get(rel) != expected:
                raise ValueError(f"{arm}: SHA256SUMS disagrees with manifest on {rel}")
            local_file = ckpt / rel
            if local_file.exists():
                if sha256_file(local_file) != expected:
                    raise ValueError(f"{arm}: tracked file {rel} SHA mismatch")
            elif bundle_hashes:
                zip_key = f"{arm}/{rel}"
                if bundle_hashes.get(zip_key) != expected:
                    raise ValueError(f"{arm}: binary payload {zip_key} SHA mismatch in bundle")

        arm_res = json.loads((ev_root / f"{arm}_result.json").read_text(encoding="utf-8"))
        if arm_res != all_sum.get(arm):
            raise ValueError(f"{arm}: {arm}_result.json differs from all_arms_summary.json")

        p32 = arm_res["evaluation"]["dev32_pairs"]
        per_pair = p32["per_pair"]
        if len(per_pair) != 32:
            raise ValueError(f"{arm}: expected 32 directional pairs, got {len(per_pair)}")

        recomp_correct = 0
        gaps = []
        vis_counts = {"V+": [0, 0], "U": [0, 0], "V-": [0, 0]}
        for row in per_pair:
            ts = row["target_step_scores"]
            als = row["alt_step_scores"]
            if not ts or not als or any(not math.isfinite(x) for x in ts + als):
                raise ValueError(f"{arm}: non-finite or empty step_scores on task {row['task_id']}")
            gap = float(row["model_click_gap"])
            gaps.append(gap)
            ok = gap > 1e-5
            if ok != bool(row["correct"]):
                raise ValueError(f"{arm}: inconsistent correct flag on task {row['task_id']}")
            if ok:
                recomp_correct += 1
            vis = row.get("visibility")
            if vis in vis_counts:
                vis_counts[vis][1] += 1
                if ok:
                    vis_counts[vis][0] += 1

        if recomp_correct != p32["correct"]:
            raise ValueError(f"{arm}: recomputed correct {recomp_correct} != {p32['correct']}")
        mean_gap = sum(gaps) / len(gaps)
        if not math.isclose(mean_gap, p32["mean_high_minus_low_gap"], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{arm}: mean_gap mismatch {mean_gap} != {p32['mean_high_minus_low_gap']}")

        # Verify exact telescoping identity sum(step_scores) == v_prefix[-1] == total on 7544
        r7544 = arm_res["evaluation"]["regression_7544"]
        for side in ("running_shoes_1.0", "clogs_0.0"):
            pred = r7544[side]
            vp = pred["v_prefix"]
            ss = pred["step_scores"]
            if len(vp) != len(ss) or len(vp) != len(pred["turn_end_indices"]):
                raise ValueError(f"{arm}: 7544 {side} length mismatch")
            prev = 0.0
            for v, s in zip(vp, ss):
                if not math.isclose(v - prev, s, rel_tol=0.0, abs_tol=1e-6):
                    raise ValueError(f"{arm}: 7544 {side} telescoping step mismatch")
                prev = v
            if not math.isclose(sum(ss), pred["total"], rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"{arm}: 7544 {side} sum(step_scores) != total")

        click_gap_7544 = r7544["running_shoes_1.0"]["step_scores"][1] - r7544["clogs_0.0"]["step_scores"][1]
        if not math.isclose(click_gap_7544, r7544["click_step_gap_shoes_minus_clogs"], rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{arm}: 7544 click gap mismatch")

        d160 = arm_res["evaluation"]["dev160"]
        arm_reports[arm] = {
            "dev32_correct": f"{recomp_correct}/32",
            "dev32_accuracy": recomp_correct / 32.0,
            "by_visibility": {k: f"{v[0]}/{v[1]}" for k, v in vis_counts.items()},
            "mean_click_gap": mean_gap,
            "r7544_click_gap": click_gap_7544,
            "dev160_mse": d160["model"]["mse"],
            "dev160_pearson": d160["model"]["pearson_r"],
            "dev160_spearman": d160["model"]["spearman_rho"],
            "binary_verified": bool(bundle_hashes),
        }

    return {
        "verified_script_sha256": script_sha,
        "bundle_sha256": bundle_sha,
        "arms": arm_reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=None, help="Optional Release ZIP path to verify binary payloads")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    report = verify_all(repo_root, args.bundle)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
