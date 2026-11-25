"""Regression tests for PR #54 offline evidence verifier (turn_end_indices & tamper rejection)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
RECOMPUTE_PATH = REPO_ROOT / "results/prm-split-cosine-20261124/recompute.py"

spec = importlib.util.spec_from_file_location("prm_split_cosine_recompute", RECOMPUTE_PATH)
recompute = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(recompute)


def _refresh_arm_hashes(ckpt: Path) -> None:
    manifest_path = ckpt / "prm-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel in list(manifest["files_sha256"]):
        local = ckpt / rel
        if local.is_file():
            manifest["files_sha256"][rel] = hashlib.sha256(local.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_hashes = dict(manifest["files_sha256"])
    all_hashes["prm-manifest.json"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (ckpt / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(all_hashes.items())),
        encoding="utf-8",
    )


class PRMSplitCosineRecomputeTests(unittest.TestCase):
    def test_committed_three_arms_pass_and_reproduce_exact_metrics(self):
        report = recompute.verify_evidence()
        by_arm = {row["arm"]: row for row in report["arms"]}
        self.assertFalse(by_arm["hidden-lr-low-acc8"]["memory_gate_passed"])
        self.assertAlmostEqual(by_arm["hidden-lr-low-acc8"]["mse"], 0.303530243, places=8)
        self.assertTrue(by_arm["hidden-split-cos-acc8"]["memory_gate_passed"])
        self.assertAlmostEqual(by_arm["hidden-split-cos-acc8"]["mse"], 7.1533606543e-7, places=12)
        self.assertAlmostEqual(by_arm["hidden-split-cos-acc8"]["max_abs_error"], 0.0014405996, places=9)
        self.assertTrue(by_arm["hidden-split-cos-acc1"]["memory_gate_passed"])
        self.assertAlmostEqual(by_arm["hidden-split-cos-acc1"]["mse"], 1.7329243922e-7, places=12)
        self.assertAlmostEqual(by_arm["hidden-split-cos-acc1"]["max_abs_error"], 0.0007626377, places=9)

    def test_rejects_simultaneous_missing_or_malformed_turn_end_indices(self):
        bad_cases = [
            ("both_empty", []),
            ("short_length", [10, 20]),
            ("long_length", [10, 20, 30, 40]),
            ("negative_index", [-1, 20, 30]),
            ("duplicate_index", [10, 10, 30]),
            ("decreasing_index", [20, 10, 30]),
            ("out_of_bounds", [10, 20, 2048]),
            ("boolean_index", [False, 10, 20]),
            ("missing_key", None),
        ]
        for case_name, bad_indices in bad_cases:
            with self.subTest(case=case_name), tempfile.TemporaryDirectory() as tmp:
                copied_arms = Path(tmp) / "arms"
                shutil.copytree(recompute.DEFAULT_ARMS_DIR, copied_arms)
                arm = "hidden-split-cos-acc8"
                ckpt = copied_arms / arm
                ref_path = ckpt / "reference-predictions.json"
                rel_path = copied_arms / f"{arm}-reload.json"

                ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
                rel_data = json.loads(rel_path.read_text(encoding="utf-8"))
                if bad_indices is None:
                    ref_data[0].pop("turn_end_indices", None)
                    rel_data["predictions"][0].pop("turn_end_indices", None)
                else:
                    ref_data[0]["turn_end_indices"] = bad_indices
                    rel_data["predictions"][0]["turn_end_indices"] = bad_indices
                ref_path.write_text(json.dumps(ref_data, indent=2) + "\n", encoding="utf-8")
                rel_path.write_text(json.dumps(rel_data, indent=2) + "\n", encoding="utf-8")
                _refresh_arm_hashes(ckpt)

                with self.assertRaises(ValueError):
                    recompute.verify_evidence(arms_dir=copied_arms)

    def test_reconstruct_positions_rejects_shifted_non_im_end_index(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("Transformers not installed")
        rev = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
        try:
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", revision=rev, local_files_only=True)
        except OSError:
            self.skipTest("Pinned Qwen tokenizer not cached locally")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tok_dir = root / "tok"
            tok.save_pretrained(tok_dir)
            tok_hashes = {
                name: hashlib.sha256((tok_dir / name).read_bytes()).hexdigest()
                for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
                if (tok_dir / name).is_file()
            }
            samples = root / "samples.jsonl"
            rows = []
            for tid, turns, reward in recompute.CANONICAL_EIGHT:
                msgs = [{"role": "system", "content": "Shop."}]
                for step in range(turns):
                    msgs.append({"role": "user", "content": f"Task {tid} obs {step}"})
                    msgs.append({"role": "assistant", "content": f"Action: search[item {tid} {step}]"})
                rows.append({"task_id": tid, "messages": msgs, "terminal_reward": reward})
            samples.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            positions = recompute.reconstruct_expected_positions(samples, tok_dir, tok_hashes, 2048)
            self.assertEqual(len(positions[351]), 3)
            self.assertEqual(len(positions[5478]), 10)


if __name__ == "__main__":
    unittest.main()
