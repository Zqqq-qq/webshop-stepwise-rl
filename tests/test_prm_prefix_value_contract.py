"""三项接口修复的因果性、契约拒载与固定 runtime 启动回归。"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pv_support_test", ROOT / "scripts/prm_prefix_value_support.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


class ContractTests(unittest.TestCase):
    def fixture(self, root):
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "head.pt").write_bytes(b"fixture head")
        manifest = {"schema_version": 2, "architecture": {"representation": "final_hidden_state"},
                    "identity": {"max_length": 128},
                    "files_sha256": {"head.pt": support.file_sha(checkpoint / "head.pt")}}
        (checkpoint / "prm-manifest.json").write_text(json.dumps(manifest))
        path = root / "contract.json"
        path.write_text(json.dumps(support.make_contract(checkpoint)))
        return checkpoint, path, support.file_sha(path)

    def test_contract_binds_manifest_payload_and_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt, path, digest = self.fixture(Path(tmp))
            actual = support.verify_contract(ckpt, path, digest)
            self.assertEqual(actual["total"], "last_prefix_value")
            self.assertEqual(actual["step_reward"], "value_minus_previous_value_initial_zero")
            (ckpt / "head.pt").write_bytes(b"changed head")
            with self.assertRaisesRegex(ValueError, "载荷"):
                support.verify_contract(ckpt, path, digest)

    def test_changed_contract_and_wrong_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt, path, digest = self.fixture(Path(tmp))
            data = json.loads(path.read_text()); data["total"] = "sum"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "SHA"):
                support.verify_contract(ckpt, path, digest)
            with self.assertRaisesRegex(ValueError, "语义"):
                support.verify_contract(ckpt, path, support.file_sha(path))
            data["total"] = "last_prefix_value"; path.write_text(json.dumps(data))
            (ckpt / "prm-manifest.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "manifest"):
                support.verify_contract(ckpt, path, support.file_sha(path))

    def test_helper_mismatch_rejects_before_model_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt, path, _ = self.fixture(Path(tmp))
            data = json.loads(path.read_text()); data["helper_sha256"] = "0" * 64
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "实现哈希"):
                support.verify_contract(ckpt, path, support.file_sha(path))

    def test_old_sum_entry_rejects_registered_pv_even_after_reformat(self):
        from webshop_lab.prm.scoring_guard import reject_registered_pv_checkpoint
        source = ROOT / 'results/prm-prefix-value-20261126/rtx3090-five-arms/arm4_pv_asin_context_fit501/prm-manifest.json'
        reject_path = source.parent
        with self.assertRaisesRegex(ValueError, "旧 sum"):
            reject_registered_pv_checkpoint(reject_path)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "prm-manifest.json"
            p.write_text(json.dumps(json.loads(source.read_text()), sort_keys=True))
            with self.assertRaisesRegex(ValueError, "旧 sum"):
                reject_registered_pv_checkpoint(p.parent)
            p.write_text(json.dumps({"score_definition": "assistant_turn_end_vocab_logits_linear_sum_mse"}))
            reject_registered_pv_checkpoint(p.parent)

    def test_cli_imports_without_new_runtime_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            shutil.copytree(ROOT / "src", runtime / "src")
            for name in ("prefix_reward.py", "scoring_guard.py", "data.py"):
                (runtime / "src/webshop_lab/prm" / name).unlink(missing_ok=True)
            env = {**os.environ, "PYTHONPATH": str(runtime / "src")}
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/score_prefix_value_prm.py'), '--help'],
                                    env=env, cwd=runtime, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--scoring-contract-sha256', result.stdout)

    def test_new_reference_rejects_same_values_from_other_input(self):
        trace = {"skip_reason": None, "input_ids_sha256": "a" * 64, "score_token_index": 1}
        row = {"task_id": 1, "step_scores": [0.5], "v_prefix": [0.5], "total": 0.5,
               "prefix_traces": [trace], "turn_end_indices": [1], "position_convention": "per_prefix_local"}
        other = copy.deepcopy(row); other['prefix_traces'][0]['input_ids_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, "输入身份"):
            support.compare_pv_predictions([row], [other], 1e-5, 1e-5)


class CausalTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',
                revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306', local_files_only=True)
        except OSError:
            raise unittest.SkipTest('缺少固定 Qwen tokenizer 缓存')

    def messages(self):
        return [{'role':'system','content':'Find running shoes below 70 dollars.'},
                {'role':'user','content':'Search'},
                {'role':'assistant','content':'Action: search[running shoes]'},
                {'role':'user','content':'Results [SEP] B09KGMGVV9 [SEP] Women Running Shoes Lightweight Breathable Walking Shoes [SEP] $41.99 [SEP]'},
                {'role':'assistant','content':'Action: click[b09kgmgvv9]'},
                {'role':'user','content':'Detail page'},
                {'role':'assistant','content':'Action: click[buy now]'}]

    def test_future_overflow_cannot_change_earlier_click(self):
        from webshop_lab.prm.prefix_reward import SUPPORT as online
        messages = self.messages(); original = copy.deepcopy(messages)
        short = support.prepare_prefixes(self.tokenizer, messages, 2048)
        messages[5]['content'] += ' detail' * 1897
        long = support.prepare_prefixes(self.tokenizer, messages, 2048)
        self.assertEqual(short[1], long[1])
        direct = online.encode_current_prefix(self.tokenizer, messages[:5], 2048)
        self.assertEqual(direct, long[1])
        self.assertEqual(short[1][1]['score_token_index'], 113)
        self.assertNotEqual(messages[5], original[5])

    def test_empty_reply_and_later_observation_are_preserved(self):
        messages = self.messages(); messages[4]['content'] = ''
        prefixes = support.prepare_prefixes(self.tokenizer, messages, 2048)
        self.assertIsNone(prefixes[1][0])
        self.assertEqual(prefixes[1][1]['skip_reason'], 'empty_response')
        self.assertIsNotNone(prefixes[2][0])

    def test_rollout_reconstruction_ignores_labels(self):
        row = {'instruction':'Shoes', 'steps':[{'observation':'Page','response':'Action: search[shoe]'}]}
        before = support.rollout_messages(row)
        row.update(final_reward=999, success='untrusted label')
        self.assertEqual(before, support.rollout_messages(row))


if __name__ == '__main__':
    unittest.main()
