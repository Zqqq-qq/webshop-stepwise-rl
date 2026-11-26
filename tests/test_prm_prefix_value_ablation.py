"""Unit tests for Prefix-Value Telescoping PRM helper logic, standalone CLI scorer, and archived 5-arm recompute verification."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PrefixValuePRMAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recompute = _load_module(
            "prm_pv_recompute",
            REPO_ROOT / "results/prm-prefix-value-20261126/recompute.py",
        )
        try:
            import torch  # noqa: F401
            cls.ablation = _load_module(
                "run_prefix_value_prm_ablation",
                REPO_ROOT / "scripts/run_prefix_value_prm_ablation.py",
            )
            cls.scorer = _load_module(
                "score_prefix_value_prm",
                REPO_ROOT / "scripts/score_prefix_value_prm.py",
            )
        except ImportError:
            cls.ablation = None
            cls.scorer = None

    def test_committed_five_arm_evidence_recomputes_cleanly(self):
        report = self.recompute.verify_all(REPO_ROOT, bundle_path=None)
        arms = report["arms"]
        self.assertEqual(arms["arm1_pv_qv_fit501"]["dev32_correct"], "22/32")
        self.assertEqual(arms["arm2_pv_alllinear_fit501"]["dev32_correct"], "22/32")
        self.assertEqual(arms["arm2_pv_alllinear_fit501"]["by_visibility"]["V+"], "19/21")
        self.assertGreater(arms["arm2_pv_alllinear_fit501"]["r7544_click_gap"], 0.02)
        self.assertEqual(arms["arm4_pv_asin_context_fit501"]["dev32_correct"], "26/32")
        self.assertEqual(arms["arm4_pv_asin_context_fit501"]["by_visibility"]["V+"], "20/21")
        self.assertGreater(arms["arm4_pv_asin_context_fit501"]["r7544_click_gap"], 0.11)
        self.assertLess(arms["arm4_pv_asin_context_fit501"]["dev160_mse"], 0.082)

    def test_public_asin_lookup_uses_only_immediately_preceding_user_page(self):
        if self.ablation is None or self.scorer is None:
            self.skipTest("Requires torch environment")
        messages = [
            {"role": "system", "content": "Shop carefully."},
            {
                "role": "user",
                "content": "Instruction: [SEP] running shoes [SEP] B07YNGTFK3 [SEP] Women Running Shoes [SEP] $49.99 [SEP] B08CLOGS00 [SEP] Nordways Clogs [SEP] $19.99",
            },
            {"role": "assistant", "content": "Action: click[b07yngtfk3]"},
            {"role": "user", "content": "Product page [SEP] B09FUTURE1 [SEP] Secret Future Item [SEP] $9.99"},
            {"role": "assistant", "content": "Action: click[buy now]"},
        ]
        resolved = self.ablation.resolve_public_asin_from_prev_page(messages, 2)
        self.assertIn("B07YNGTFK3", resolved)
        self.assertIn("Women Running Shoes", resolved)
        self.assertIn("$49.99", resolved)
        self.assertNotIn("Nordways Clogs", resolved)
        self.assertIsNone(self.ablation.resolve_public_asin_from_prev_page(messages, 4))

        info = self.scorer.extract_preclick_asin_snippet(messages, 2)
        self.assertEqual(info["assistant_message_index"], 2)
        self.assertEqual(info["source_search_message_index"], 1)
        self.assertEqual(info["extracted_title"], "Women Running Shoes")
        self.assertEqual(info["extracted_price"], "$49.99")
        span = info["source_char_span"]
        self.assertEqual(messages[1]["content"][span[0]:span[1]], info["source_verbatim_record"])

        messages_future_only = [
            messages[0],
            messages[1],
            {"role": "assistant", "content": "Action: click[b09future1]"},
            messages[3],
        ]
        self.assertIsNone(self.ablation.resolve_public_asin_from_prev_page(messages_future_only, 2))
        self.assertIsNone(self.scorer.extract_preclick_asin_snippet(messages_future_only, 2))

    def test_standalone_score_cli_reload_reference_and_provenance(self):
        if self.scorer is None:
            self.skipTest("Requires torch environment")
        import torch
        from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
        from peft import LoraConfig, get_peft_model
        from webshop_lab.prm.config import PRMConfig
        from webshop_lab.prm.model_core import save_prm
        from webshop_lab.prm.runtime import build_prm, finalize_manifest, sha

        rev = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
        try:
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", revision=rev, local_files_only=True)
        except OSError:
            self.skipTest("Requires cached Qwen tokenizer")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, tok_dir, b1, ckpt = root / "base", root / "tok", root / "b1", root / "ckpt"
            tok.save_pretrained(tok_dir)
            torch.manual_seed(42)
            tiny = Qwen2ForCausalLM(Qwen2Config(
                vocab_size=151936, hidden_size=8, intermediate_size=16,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                max_position_embeddings=512, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
            ))
            tiny.save_pretrained(base)
            sft = get_peft_model(tiny, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", bias="none"))
            sft.save_pretrained(b1)
            cfg = PRMConfig(
                model=str(base), revision=rev, base_weights_sha256=sha(base / "model.safetensors"),
                tokenizer=str(tok_dir), tokenizer_revision=rev,
                tokenizer_files_sha256={p.name: sha(p) for p in tok_dir.iterdir() if p.is_file()},
                initialization="sft_b1", sft_adapter=str(b1),
                sft_adapter_files_sha256={n: sha(b1 / n) for n in ("adapter_model.safetensors", "adapter_config.json")},
                output_dir=str(ckpt), max_length=512, lora_r=2, lora_alpha=4,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
                schema_version="prm-qwen-config-v2", representation="final_hidden_state",
            )
            model, _, identity, _ = build_prm(cfg, None, "cpu", True)
            with torch.no_grad():
                model.score_head.weight.fill_(0.05)
                model.score_head.bias.fill_(0.1)
            save_prm(model, ckpt, identity)
            cfg_path = root / "cfg.json"
            cfg_path.write_text(json.dumps(cfg.to_dict()))
            finalize_manifest(ckpt)
            from webshop_lab.prm.prefix_reward import CONTRACT_PATH, checkpoint_manifest_sha

            contract = json.loads(CONTRACT_PATH.read_text())
            contract["checkpoints"] = {checkpoint_manifest_sha(ckpt): "preclick_asin"}
            contract_path = root / "fixture-contract.json"
            contract_path.write_text(json.dumps(contract))

            samples = root / "samples.jsonl"
            samples.write_text(json.dumps({
                "task_id": 7544,
                "messages": [
                    {"role": "system", "content": "Shop carefully."},
                    {"role": "user", "content": "Instruction: [SEP] shoes [SEP] Search"},
                    {"role": "assistant", "content": "Action: search[running shoes]"},
                    {"role": "user", "content": "Instruction: [SEP] shoes [SEP] B09KGMGVV9 [SEP] Women Running Shoes [SEP] $41.99 [SEP]"},
                    {"role": "assistant", "content": "Action: click[b09kgmgvv9]"},
                    {"role": "user", "content": "Detail page"},
                    {"role": "assistant", "content": "Action: click[buy now]"},
                ],
            }) + "\n")

            out1 = root / "out1.json"
            out2 = root / "out2.json"
            cmd1 = [
                sys.executable, str(REPO_ROOT / "scripts/score_prefix_value_prm.py"),
                "--config", str(cfg_path), "--checkpoint", str(ckpt),
                "--contract", str(contract_path),
                "--samples", str(samples), "--output", str(out1), "--device", "cpu",
            ]
            res1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=90)
            self.assertEqual(res1.returncode, 0, res1.stderr)
            payload1 = json.loads(out1.read_text())
            pred = payload1["predictions"][0]
            self.assertEqual(len(pred["step_scores"]), 3)
            self.assertAlmostEqual(sum(pred["step_scores"]), pred["total"], places=6)
            self.assertEqual(len(pred["asin_context_provenance"]), 1)
            prov = pred["asin_context_provenance"][0]
            self.assertEqual(prov["turn_index"], 1)
            self.assertEqual(prov["assistant_message_index"], 4)
            self.assertEqual(prov["source_search_message_index"], 3)
            self.assertEqual(prov["scored_turn_end_token"], "<|im_end|>")
            self.assertEqual(prov["scored_turn_end_token_index"], pred["turn_end_indices"][1])

            # Cross-process --reference reload check
            cmd2 = cmd1[:cmd1.index("--output")] + ["--reference", str(out1), "--output", str(out2), "--device", "cpu"]
            res2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=90)
            self.assertEqual(res2.returncode, 0, res2.stderr)
            self.assertTrue(json.loads(out2.read_text())["reference_match"])


if __name__ == "__main__":
    unittest.main()
