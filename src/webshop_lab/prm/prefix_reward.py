"""Arm 4 ASIN context and current-prefix value scoring."""
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from webshop_lab.prm.tokenize import TrajectoryTooLong, tokenize_trajectory


ASIN_CLICK_RE = re.compile(r"click\[(b[0-9a-z]{9})\]", flags=re.IGNORECASE)


def extract_preclick_asin_snippet(messages: list[dict], msg_idx: int) -> dict | None:
    """Inspect `messages[msg_idx]` and read product title + price strictly from `messages[msg_idx - 1]`."""
    if msg_idx <= 0 or messages[msg_idx].get("role") != "assistant":
        return None
    if messages[msg_idx - 1].get("role") != "user":
        return None
    m_click = ASIN_CLICK_RE.search(messages[msg_idx].get("content", ""))
    if not m_click:
        return None
    asin_raw = m_click.group(1)
    asin_upper = asin_raw.upper()
    prev_page = messages[msg_idx - 1].get("content", "")
    pat = re.compile(
        r"\[SEP\]\s*" + re.escape(asin_upper) + r"\s*\[SEP\]\s*(.*?)\s*\[SEP\]\s*(\$[^\[]+?)(?:\s*\[SEP\]|$)",
        flags=re.DOTALL,
    )
    hit = pat.search(prev_page)
    if not hit:
        return None
    title = " ".join(hit.group(1).split())[:180]
    price = " ".join(hit.group(2).split())[:32]
    augmented_content = f"{messages[msg_idx]['content']} [Item {asin_upper}: {title} | {price}]"
    return {
        "assistant_message_index": msg_idx,
        "source_search_message_index": msg_idx - 1,
        "clicked_asin_raw": asin_raw,
        "matched_asin_upper": asin_upper,
        "source_char_span": [hit.start(), hit.end()],
        "source_verbatim_record": prev_page[hit.start():hit.end()],
        "extracted_title": title,
        "extracted_price": price,
        "original_assistant_content": messages[msg_idx]["content"],
        "augmented_assistant_content": augmented_content,
    }


def encode_trajectory_for_pv(tokenizer, messages: list[dict], max_length: int, attach_asin_context: bool):
    """Encode raw trajectory messages, optionally attaching deterministic pre-click ASIN context."""
    base_enc = tokenize_trajectory(tokenizer, messages, max_length)
    provenance = []
    chosen_enc = base_enc

    if attach_asin_context:
        aug_messages = [dict(m) for m in messages]
        assistant_turn_counter = 0
        extracted_records = []
        for msg_idx, msg in enumerate(messages):
            if msg.get("role") == "assistant":
                info = extract_preclick_asin_snippet(messages, msg_idx)
                if info is not None:
                    info["turn_index"] = assistant_turn_counter
                    aug_messages[msg_idx]["content"] = info["augmented_assistant_content"]
                    extracted_records.append(info)
                assistant_turn_counter += 1

        if extracted_records:
            try:
                cand_enc = tokenize_trajectory(tokenizer, aug_messages, max_length)
                if len(cand_enc["turn_end_indices"]) == len(base_enc["turn_end_indices"]):
                    chosen_enc = cand_enc
                    for rec in extracted_records:
                        t_idx = rec["turn_index"]
                        tok_pos = chosen_enc["turn_end_indices"][t_idx]
                        tok_id = chosen_enc["input_ids"][tok_pos]
                        rec["fallback_to_raw_due_to_max_length"] = False
                        rec["raw_turn_end_token_index"] = base_enc["turn_end_indices"][t_idx]
                        rec["scored_turn_end_token_index"] = tok_pos
                        rec["scored_turn_end_token"] = tokenizer.decode([tok_id])
                        provenance.append(rec)
            except TrajectoryTooLong:
                chosen_enc = base_enc
                for rec in extracted_records:
                    t_idx = rec["turn_index"]
                    tok_pos = base_enc["turn_end_indices"][t_idx]
                    tok_id = base_enc["input_ids"][tok_pos]
                    rec["fallback_to_raw_due_to_max_length"] = True
                    rec["raw_turn_end_token_index"] = tok_pos
                    rec["scored_turn_end_token_index"] = tok_pos
                    rec["scored_turn_end_token"] = tokenizer.decode([tok_id])
                    provenance.append(rec)

    return chosen_enc, provenance



def encode_current_prefix(tokenizer, messages, max_length):
    """Encode the actor's visible history, dropping old exchanges if PRM needs more room."""
    # Earlier EOS-only actions still have an environment error response. Preserve
    # that exchange without passing an empty assistant message to the PRM tokenizer.
    history = [
        {"role": "assistant", "content": "(Empty assistant reply)"}
        if m["role"] == "assistant" and not m["content"].strip() else m
        for m in messages[:-1]
    ] + messages[-1:]
    while True:
        try:
            encoded, provenance = encode_trajectory_for_pv(tokenizer, history, max_length, True)
        except TrajectoryTooLong:
            # Keep the system instruction and the current observation/action pair.
            if len(history) <= 3:
                return None, False
            del history[1:3]
        else:
            return encoded, any(record["fallback_to_raw_due_to_max_length"] for record in provenance)


class OnlinePrefixReward:
    """Send one current prefix at a time to the checkpoint's pinned runtime."""
    def __init__(self, config, checkpoint, runtime):
        runtime = Path(runtime).resolve()
        self.worker = subprocess.Popen(
            [sys.executable, "-c", 'import runpy, sys; sys.argv.pop(0); runpy.run_path(sys.argv[0], run_name="__main__")',
             str(Path(__file__).resolve()), str(config.resolve()), str(checkpoint.resolve())],
            cwd=runtime, env={**os.environ, "PYTHONPATH": str(runtime / "src")},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
        )
        try:
            if self._receive() != {"ready": True}:
                raise RuntimeError("Unexpected Arm 4 worker handshake")
        except BaseException:
            self.close()
            raise

    def _receive(self):
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError("Arm 4 worker stopped before returning a value. See its traceback.")
        return json.loads(line)

    def __call__(self, messages):
        self.worker.stdin.write(json.dumps(messages, ensure_ascii=False) + "\n")
        self.worker.stdin.flush()
        result = self._receive()
        return result["value"], result["asin_length_fallback"]

    def close(self):
        self.worker.stdin.close()
        try:
            self.worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.worker.kill()
            self.worker.wait()
        finally:
            self.worker.stdout.close()


def serve(config, checkpoint):
    import torch
    from webshop_lab.prm.config import load_config
    from webshop_lab.prm.runtime import build_prm, tensor_batch

    cfg = load_config(config)
    model, tokenizer, _, _ = build_prm(cfg, checkpoint, "cuda", False)
    device = next(model.parameters()).device
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        prefix = json.loads(line)
        encoded, fallback = encode_current_prefix(tokenizer, prefix, cfg.max_length)
        if encoded is None:
            print(json.dumps({"value": None, "asin_length_fallback": False}), flush=True)
            continue
        batch = tensor_batch([encoded], tokenizer.pad_token_id, device, False)
        with torch.no_grad():
            value = float(model(**batch)["step_scores"][0, -1])
        if not math.isfinite(value):
            raise ValueError("Arm 4 returned a non-finite prefix value")
        print(json.dumps({"value": value, "asin_length_fallback": fallback}), flush=True)


if __name__ == "__main__":
    serve(Path(sys.argv[1]), Path(sys.argv[2]))
