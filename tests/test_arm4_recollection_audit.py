import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_arm4_recollection import audit


class RecollectionAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        plan = self.write("train-recollection-candidates-ids.json", [9, 2, 7])
        self.inventory = self.write("inventory.json", {
            "schema": "arm4-prm-expansion-v1", "test_authorization": False,
            "files_sha256": {plan.name: self.sha(plan)},
        })
        self.config = self.write("config.json", {"temperature": 0.0})

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value) + "\n")
        return path

    def shard(self, label, selected, recorded):
        ids = self.write(label + "-ids.json", selected)
        meta = self.write(label + "-meta.json", {
            "run_id": label, "config": json.loads(self.config.read_text()),
            "selected_ids": selected, "repeats": 1, "health": {"goal_fingerprint": "catalog"},
            "config_sha256": self.sha(self.config), "ids_sha256": self.sha(ids),
        })
        rollout = self.root / (label + ".jsonl")
        rollout.write_text("".join(json.dumps({
            "task_id": task, "run_id": label, "repeat": 0, "goal_fingerprint": "catalog",
            "final_reward": 0.0, "steps": [{}],
        }) + "\n" for task in recorded))
        return rollout, meta, ids

    def test_partial_run_resumes_in_frozen_order_without_duplicate_episodes(self):
        first = self.shard("first", [9, 2, 7], [9])
        result = audit(self.inventory, "train", [first], self.config)
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing_ids"], [2, 7])
        resumed = self.shard("resumed", [2, 7], [2, 7])
        completed = audit(self.inventory, "train", [first, resumed], self.config)
        self.assertTrue(completed["complete"])
        self.assertEqual(completed["completed"], 3)
        with self.assertRaisesRegex(ValueError, "reuses"):
            audit(self.inventory, "train", [first, resumed, resumed], self.config)

    def test_reordered_and_out_of_plan_ids_are_rejected(self):
        reordered = self.shard("reordered", [2, 9], [2, 9])
        with self.assertRaisesRegex(ValueError, "frozen order"):
            audit(self.inventory, "train", [reordered], self.config)
        wrong = self.shard("wrong", [100], [100])
        with self.assertRaisesRegex(ValueError, "outside"):
            audit(self.inventory, "train", [wrong], self.config)
