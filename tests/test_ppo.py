import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from webshop_lab import ppo
from webshop_lab.ppo import assign_episode_rewards, episode_targets, replay_ratio


class PPOMathTests(unittest.TestCase):
    def test_online_collection_continues_after_empty_and_unscorable_replies(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PPO collection requires torch")

        state = {"instruction": "Instruction: shop", "goal_fingerprint": "catalog",
                 "observation": "Home", "available": {}}
        replies = [1, 0, 2, 2]
        logits = torch.zeros(1, 3)
        model = Mock(device="cpu")
        model.generate.side_effect = [
            SimpleNamespace(sequences=torch.tensor([[1, reply]]), scores=[logits]) for reply in replies
        ]
        tokenizer = Mock()
        tokenizer.decode.side_effect = lambda tokens, **_: {
            0: "", 1: "Action: search[shoe]", 2: "Action: click[buy now]",
        }[tokens[0]]
        generation = SimpleNamespace(eos_token_id=0, max_new_tokens=1)
        progress = Mock(side_effect=[(0.5, False), (None, False), (0.75, False)])
        config = {"goal_fingerprint": "catalog", "reward_mode": "neural_progress",
                  "max_steps": 4, "context_length": 10}
        request = Mock(side_effect=[
            state, {**state, "done": False, "reward": 0.0},
            {**state, "done": False, "reward": 0.0},
            {**state, "done": True, "reward": 1.0},
        ])

        def fitted_history(_, messages, *args, **kwargs):
            return [messages[0], messages[-1]], [1]

        with patch.object(ppo, "request", request), patch.object(ppo, "fit_history", fitted_history), \
                patch.object(ppo, "replay", return_value=(logits[0].log_softmax(-1)[:1],
                                                         torch.tensor(0.0))), \
                patch.object(ppo, "is_executable", return_value=True):
            episode = ppo.collect(model, tokenizer, None, generation, "http://mock", 1, config, "shop", progress)
        turns = episode["turns"]
        self.assertEqual(request.call_count, 4)
        self.assertEqual([t["prm_skip_reason"] for t in turns],
                         [None, "empty_response", "prefix_too_long", None])
        self.assertEqual([t["prm_prefix_value"] for t in turns], [0.5, 0.5, 0.5, 0.75])
        self.assertEqual([t["reward"] for t in turns], [0.5, 0.0, 0.0, 0.25])
        self.assertFalse(turns[1]["environment_step_applied"])
        self.assertEqual(turns[1]["response_mask"], [1])
        self.assertEqual(progress.call_count, 3)
        for call in progress.call_args_list:
            self.assertEqual([m["role"] for m in call.args[0]], ["system", "user", "assistant"])

    def test_terminal_return_does_not_include_page_scores(self):
        self.assertEqual(episode_targets([0.125, 0.25], False, 0.75),
                         (0.0, [-0.125, -0.25], [0.0, 0.0]))
        self.assertEqual(episode_targets([0.125, 0.25], True, 0.75),
                         (0.75, [0.625, 0.5], [0.75, 0.75]))

    def test_replay_rejects_accumulating_token_error(self):
        self.assertAlmostEqual(replay_ratio([-1.0], [-0.99], 0.05), math.exp(0.01))
        with self.assertRaisesRegex(ValueError, "joint"):
            replay_ratio([-1.0] * 128, [-0.999] * 128, 0.05)
        for invalid in ([], [float("nan")], [float("inf")]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replay_ratio([-1.0], invalid, 0.05)


    def test_progress_rewards_are_paid_once_per_turn_and_use_suffix_returns(self):
        turns = [
            {"observation": "home", "response_text_parsed": "search", "grounding": 1, "old_value": 0.125,
             "response_token_ids": [1, 2, 3]},
            {"observation": "item", "response_text_parsed": "invalid", "grounding": 0, "old_value": 0.25,
             "response_token_ids": [7]},
        ]
        model = {"dimensions": 2, "weights": [0.25, 0.0]}
        self.assertEqual(assign_episode_rewards(turns, False, 0.9, model), 0.0)
        self.assertEqual([t["reward"] for t in turns], [0.25, 0.25])
        self.assertEqual([t["return_target"] for t in turns], [0.5, 0.25])
        self.assertEqual([t["advantage"] for t in turns], [0.375, 0.0])
        self.assertEqual([t["continuation_mask"] for t in turns], [1, 0])

    def test_terminal_mode_preserves_purchase_reward(self):
        turns = [{"old_value": 0.125}, {"old_value": 0.25}]
        self.assertEqual(assign_episode_rewards(turns, True, 0.75), 0.75)
        self.assertEqual([t["reward"] for t in turns], [0.0, 0.75])
        self.assertEqual([t["return_target"] for t in turns], [0.75, 0.75])
