import unittest
from unittest.mock import Mock, patch

from webshop_lab.prm import prefix_reward
from webshop_lab.prm.tokenize import TrajectoryTooLong


class OnlinePrefixTests(unittest.TestCase):
    def test_overlong_raw_prefix_drops_only_complete_old_exchanges(self):
        messages = [
            {"role": "system", "content": "Shop"},
            {"role": "user", "content": "Old page"},
            {"role": "assistant", "content": "Action: search[shoe]"},
            {"role": "user", "content": "Current page"},
            {"role": "assistant", "content": "Action: click[buy now]"},
        ]
        original = [dict(message) for message in messages]
        with patch.object(prefix_reward.SUPPORT, "tokenize_trajectory",
                          side_effect=[TrajectoryTooLong(), {"input_ids": [1], "turn_end_indices": [0]}]) as encode:
            self.assertEqual(prefix_reward.encode_current_prefix(None, messages, 10),
                             ({"input_ids": [1], "turn_end_indices": [0]}, False))
        self.assertEqual(encode.call_args.args[1], [messages[0], *messages[-2:]])
        self.assertEqual(messages, original)

    def test_unscorable_current_exchange_returns_no_value(self):
        messages = [
            {"role": "system", "content": "Shop"},
            {"role": "user", "content": "Huge current page"},
            {"role": "assistant", "content": "Action: click[buy now]"},
        ]
        with patch.object(prefix_reward.SUPPORT, "tokenize_trajectory",
                          side_effect=TrajectoryTooLong()) as encode:
            self.assertEqual(prefix_reward.encode_current_prefix(None, messages, 10), (None, False))
        encode.assert_called_once_with(None, messages, 10)

    def test_blank_past_reply_retains_the_following_observation(self):
        messages = [
            {"role": "system", "content": "Shop"},
            {"role": "user", "content": "Home"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Observation: Invalid action"},
            {"role": "assistant", "content": "Action: search[shoe]"},
        ]
        with patch.object(prefix_reward.SUPPORT, "tokenize_trajectory", return_value={"input_ids": [1], "turn_end_indices": [0]}) as encode:
            prefix_reward.encode_current_prefix(None, messages, 100)
        history = encode.call_args.args[1]
        self.assertTrue(history[2]["content"])
        self.assertEqual(history[3:], messages[3:])
        self.assertEqual(messages[2]["content"], "")

    def test_later_overflow_does_not_remove_earlier_asin_context(self):
        def encode(_tokenizer, messages, limit):
            content = " ".join(message["content"] for message in messages)
            if len(content) > limit:
                raise TrajectoryTooLong()
            ends = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
            return {"content": content, "turn_end_indices": ends, "input_ids": [0] * len(messages)}

        click = {"role": "assistant", "content": "Action: click[b012345678]"}
        initial = [
            {"role": "system", "content": "Shop"},
            {"role": "user", "content": "[SEP] B012345678 [SEP] Running shoe [SEP] $35"},
            click,
        ]
        later = initial + [
            {"role": "user", "content": "x" * 45},
            {"role": "assistant", "content": "Action: click[Buy Now]"},
        ]
        tokenizer = Mock()
        with patch.object(prefix_reward.SUPPORT, "tokenize_trajectory", encode):
            early, early_fallback = prefix_reward.encode_current_prefix(tokenizer, initial, 149)
            late, late_fallback = prefix_reward.encode_current_prefix(tokenizer, later, 149)
            repeated, repeated_fallback = prefix_reward.encode_current_prefix(tokenizer, later[:3], 149)
        assert early == repeated
        assert not early_fallback and not repeated_fallback
        assert late_fallback
        assert "[Item B012345678: Running shoe | $35]" in early["content"]
        assert "[Item " not in late["content"]
        assert click["content"] == "Action: click[b012345678]"
