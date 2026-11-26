import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_ppo_neural_rewards import audit, digest, evaluation_rows, training_rows


class RewardAuditTests(unittest.TestCase):
    def test_reports_failures_and_uses_the_rewards_passed_to_ppo(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "episodes.json"
            rows = [
                {"task_id": 89, "purchase_completed": False, "terminal_reward": 0,
                 "turns": [{"prm_prefix_value": 0.75, "reward": 0.75}]},
                {"task_id": 100, "purchase_completed": True, "terminal_reward": 0,
                 "turns": [{"prm_prefix_value": 0.29, "reward": 0.29}]},
                {"task_id": 101, "purchase_completed": True, "terminal_reward": 1,
                 "turns": [{"prm_prefix_value": 0.8, "reward": 0.8}]},
            ]
            path.write_text(json.dumps(rows))
            result = audit(training_rows(path), 0.30)
            self.assertEqual(result["trajectory_count"], 3)
            self.assertEqual(result["counts"]["zero_purchase"], 1)
            self.assertEqual([row["task_id"] for row in result["flagged"]], [89])
            rows[0]["turns"][0]["reward"] = 0
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "actual PPO reward"):
                training_rows(path)

    def test_validation_treats_timeout_page_score_as_zero_and_checks_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            rollout, predictions = (Path(folder) / filename for filename in ("val.jsonl", "scores.json"))
            rollout.write_text(json.dumps({"task_id": 1, "done": False, "final_reward": 0.9, "steps": [{}]}))
            payload = {"scoring_mode": "prefix_value_current_prefix_v1",
                       "input_schema": "rollout", "input_sha256": digest(rollout),
                       "predictions": [{"task_id": 1, "step_scores": [0.30], "total": 0.30,
                                        "turn_scoring": [{"skip_reason": None}]}]}
            predictions.write_text(json.dumps(payload))
            result = audit(evaluation_rows(rollout, predictions), 0.30)
            self.assertEqual(result["flagged"][0]["environment_terminal"], 0)
            rollout.write_text(rollout.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "exact validation rollout"):
                evaluation_rows(rollout, predictions)
