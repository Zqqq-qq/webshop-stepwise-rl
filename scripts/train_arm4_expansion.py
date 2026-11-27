#!/usr/bin/env python3
"""Train the frozen Arm 4 prefix-value expansion or verify its CPU inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

from prm_prefix_value_support import file_sha, make_contract, rollout_messages
from score_prefix_value_prm import prepare_scoring_inputs
from run_prefix_value_prm_ablation import ALL_LINEAR_MODULES
from webshop_lab.prm.config import load_config
from webshop_lab.prm.model_core import save_prm
from webshop_lab.prm.runtime import build_prm, finalize_manifest, tensor_batch, write_json
from webshop_lab.prm.train import cosine_lr_multiplier


B1_SHA = "c39e2a78e7c5973328e9d2c23fb6320e566352e63cc089562488f3dd46ae5f9c"
BASE_SHA = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def frozen_data(raw, inventory_path):
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory["schema"] != "arm4-prm-expansion-v1" or inventory["test_authorization"]:
        raise ValueError("Unsupported training inventory or test authorization")
    contents = raw.read_bytes()
    if hashlib.sha256(contents).hexdigest() != inventory["sources"]["raw_sha256"]:
        raise ValueError("Raw rollout SHA-256 differs from the frozen inventory")
    ids_file = inventory_path.with_name("train-verified-ids.json")
    if file_sha(ids_file) != inventory["files_sha256"][ids_file.name]:
        raise ValueError("Training order SHA-256 differs from the frozen inventory")
    ids = json.loads(ids_file.read_text(encoding="utf-8"))
    episodes = [json.loads(line) for line in contents.splitlines() if line.strip()]
    rows = {row["task_id"]: row for row in episodes}
    if (len(rows) != len(episodes) or len(ids) != len(set(ids))
            or len(ids) != inventory["verified_train_trajectories"] or not set(ids) <= rows.keys()):
        raise ValueError("Training task coverage or trajectory uniqueness changed")
    return inventory, [(task, rows[task]) for task in ids]


def encode_training(tokenizer, episodes, inventory):
    config = inventory["original_rollout_metadata"]
    prompt_budget = config["policy_context_length"] - config["policy_max_new_tokens"]
    encoded = []
    for task, episode in episodes:
        messages = rollout_messages(episode)
        for i, step in enumerate(episode["steps"][:-1]):
            if not step["grounding"]:
                messages[2 * i + 3]["content"] = (
                    "Observation: Invalid action or format. Reply with one valid Action: line."
                )
        prefixes = prepare_scoring_inputs(tokenizer, messages, 2048, prompt_budget)
        usable = [prefix for prefix, _ in prefixes if prefix is not None]
        expected = inventory["task_status"][str(task)]
        if prefixes[-1][0] is None or len(usable) != expected["scoreable_turns"]:
            raise ValueError(f"Task {task} no longer matches frozen encoding coverage")
        target = float(episode["final_reward"]) if episode["done"] else 0.0
        if not math.isfinite(target) or not 0 <= target <= 1:
            raise ValueError(f"Task {task} has invalid environment target")
        encoded.append((task, target, usable))
    return encoded


def train_update(model, tokenizer, device, episodes, optimizer):
    import torch

    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for _, target, prefixes in episodes:
        weight = 1 / (len(episodes) * len(prefixes))
        for prefix in prefixes:
            batch = tensor_batch([prefix], tokenizer.pad_token_id, device, False)
            value = model(**batch)["step_scores"][0, -1]
            loss = (value - target).square() * weight
            loss.backward()
            loss_sum += float(loss.detach())
    parameters = [p for p in model.parameters() if p.requires_grad]
    grad = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": loss_sum, "gradient_norm": float(grad)}


def main():
    import torch
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-rollouts", type=Path, required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    if not args.preflight and args.output is None:
        p.error("--output is required for training")
    cfg = load_config(args.config)
    if (cfg.schema_version != "prm-qwen-config-v2" or cfg.representation != "final_hidden_state"
            or tuple(cfg.target_modules) != ALL_LINEAR_MODULES or cfg.initialization != "sft_b1"
            or cfg.revision != REVISION or cfg.tokenizer_revision != REVISION
            or cfg.base_weights_sha256 != BASE_SHA or cfg.policy_dtype != "float32" or cfg.load_in_4bit
            or cfg.max_length != 2048 or cfg.lora_r != 8 or cfg.lora_alpha != 16
            or cfg.sft_adapter_files_sha256.get("adapter_model.safetensors") != B1_SHA):
        raise ValueError("Use the frozen Arm 4 FP32 B1/all-linear configuration")
    for name, expected in cfg.tokenizer_files_sha256.items():
        if file_sha(Path(cfg.tokenizer) / name) != expected:
            raise ValueError("Tokenizer differs from the frozen Arm 4 configuration")
    inventory, rows = frozen_data(args.raw_rollouts, args.inventory)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer, local_files_only=True)
    episodes = encode_training(tokenizer, rows, inventory)
    print(json.dumps({"trajectories": len(episodes), "prefixes": sum(len(x[2]) for x in episodes),
                      "inventory_sha256": file_sha(args.inventory), "mode": "preflight" if args.preflight else "train"}),
          flush=True)
    if args.preflight:
        return

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model, _, identity, _ = build_prm(cfg, None, "cuda", True)
    head = [p for n, p in model.named_parameters() if n.startswith("score_head.") and p.requires_grad]
    lora = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    base_lrs = (2e-5, 2e-4)
    optimizer = torch.optim.AdamW([{"params": head, "lr": base_lrs[0]},
                                   {"params": lora, "lr": base_lrs[1]}], weight_decay=0.01)
    updates = 2 * math.ceil(len(episodes) / 8)
    step = 0
    for epoch in (1, 2):
        with (output / f"epoch-{epoch}-updates.jsonl").open("x", encoding="utf-8") as log:
            for start in range(0, len(episodes), 8):
                batch = episodes[start:start + 8]
                scale = cosine_lr_multiplier(step, updates)
                for group, rate in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = rate * scale
                begin = time.monotonic()
                record = train_update(model, tokenizer, "cuda", batch, optimizer)
                step += 1
                record.update(step=step, task_ids=[x[0] for x in batch], seconds=time.monotonic() - begin)
                log.write(json.dumps(record) + "\n")
                log.flush()
        checkpoint = output / f"epoch-{epoch}"
        save_prm(model, checkpoint, identity)
        write_json(checkpoint / "config.json", cfg.to_dict())
        write_json(checkpoint / "training-summary.json",
                   {"epochs": epoch, "updates": step, "inventory_sha256": file_sha(args.inventory),
                    "raw_sha256": inventory["sources"]["raw_sha256"], "test_authorization": False})
        finalize_manifest(checkpoint)
        contract = output / f"epoch-{epoch}-scoring-contract.json"
        write_json(contract, make_contract(checkpoint))
        print(json.dumps({"checkpoint": str(checkpoint), "contract": str(contract),
                          "contract_sha256": file_sha(contract)}), flush=True)


if __name__ == "__main__":
    main()
