"""Single-batch engineering PPO smoke test for the WebShop terminal reward arm."""
import hashlib
import math
from pathlib import Path
import re
import uuid

from .common import file_hash, git_head, instruction_hash, is_executable, parse_action, read_json, write_json
from .model import SYSTEM, fit_history, load
from .progress import GUARDED_KIND, LEGACY_KIND, predict_steps
from .rollout import request

CLIP = 0.2
JOINT_LOGPROB_TOLERANCE = 0.05
TOKEN_LOGPROB_TOLERANCE = 0.02


def episode_targets(old_values, purchased, score, turn_rewards=None):
    """For complete episodes with gamma=lambda=1, GAE is the suffix return minus V."""
    if not old_values or not all(math.isfinite(value) for value in old_values):
        raise ValueError("Invalid old values")
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid environment score")
    reward = score if purchased else 0.0
    if turn_rewards is None:
        return reward, [reward - value for value in old_values], [reward] * len(old_values)
    if len(turn_rewards) != len(old_values) or not all(map(math.isfinite, turn_rewards)):
        raise ValueError("Invalid per-turn progress rewards")
    returns = [math.fsum(turn_rewards[i:]) for i in range(len(turn_rewards))]
    return reward, [ret - value for ret, value in zip(returns, old_values)], returns


def option_potential(instruction, selected_options):
    """Bounded lexical match of the requested item and the active visible selections."""
    if not isinstance(selected_options, dict) or any(not isinstance(v, str) for v in selected_options.values()):
        raise ValueError("Reward bridge must supply selected_options")
    ignore = {"a", "an", "of", "the", "and", "in", "with", "for"}
    goal = set(re.findall(r"[a-z0-9]+", instruction.lower().split(", and price lower than")[0])) - ignore
    option = {word for value in selected_options.values()
              for word in re.findall(r"[a-z0-9]+", value.lower())} - ignore
    return 0.25 * len(option & goal) / len(option) if option else 0.0


def assign_episode_rewards(turns, purchased, score, progress_model=None, *, shaping=False, prefix_rewards=None):
    progress = prefix_rewards
    if progress_model is not None:
        steps = [{"observation": turn["observation"], "response": turn["response_text_parsed"],
                  "grounding": turn["grounding"], "action": turn.get("parsed_action")}
                 for turn in turns]
        progress = predict_steps(progress_model, steps)
    shaping_deltas = None
    turn_rewards = progress
    if shaping:
        final_reward, _, _ = episode_targets([turn["old_value"] for turn in turns], purchased, score)
        shaping_deltas = [
            (0.0 if i == len(turns) - 1 else turn["potential_after"]) - turn["potential_before"]
            for i, turn in enumerate(turns)
        ]
        turn_rewards = shaping_deltas.copy()
        turn_rewards[-1] += final_reward
    outcome, advantages, returns = episode_targets(
        [turn["old_value"] for turn in turns], purchased, score, turn_rewards
    )
    for i, turn in enumerate(turns):
        has_next = i + 1 < len(turns)
        terminal = 0.0 if has_next else outcome
        prediction = None if progress is None else progress[i]
        reward = terminal if turn_rewards is None else turn_rewards[i]
        turn.update(reward=reward, advantage=advantages[i], return_target=returns[i],
                    reward_components={"terminal_outcome": terminal, "progress": prediction,
                                       "potential_delta": 0.0 if shaping_deltas is None else shaping_deltas[i],
                                       "grounding_bonus": 0.0, "combined": reward},
                    bootstrap_mask=int(has_next), continuation_mask=int(has_next),
                    next_old_value=turns[i + 1]["old_value"] if has_next else 0.0)
    return outcome


def load_progress_model(cfg, split_dir):
    mode = cfg.get("reward_mode", "terminal")
    if mode in ("terminal", "option_potential"):
        if "progress_model" in cfg or "progress_model_sha256" in cfg:
            raise ValueError("Only progress mode accepts a progress model")
        return None
    if mode == "neural_progress":
        from .prm.prefix_reward import OnlinePrefixReward

        config = Path(cfg["prm_config"])
        checkpoint = Path(cfg["prm_checkpoint"])
        if (file_hash(config) != cfg["prm_config_sha256"]
                or file_hash(checkpoint / "prm-manifest.json") != cfg["prm_manifest_sha256"]):
            raise ValueError("Arm 4 config or checkpoint manifest hash mismatch")
        return OnlinePrefixReward(config, checkpoint, cfg["prm_runtime"])
    if mode != "progress":
        raise ValueError("Unknown PPO reward mode")
    path = Path(cfg["progress_model"])
    if file_hash(path) != cfg["progress_model_sha256"]:
        raise ValueError("Progress model hash mismatch")
    model = read_json(path)
    dimensions = model.get("dimensions")
    weights = model.get("weights")
    if (model.get("kind") not in (LEGACY_KIND, GUARDED_KIND) or model.get("target") != "terminal_reward"
            or model.get("goal_fingerprint") != cfg["goal_fingerprint"]
            or type(dimensions) is not int or dimensions < 2
            or not isinstance(weights, list) or len(weights) != dimensions
            or not all(type(w) in (int, float) and math.isfinite(w) for w in weights)):
        raise ValueError("Invalid progress model identity, target or weights")
    for split in ("train", "val"):
        ids = model.get(f"{split}_task_ids")
        frozen = read_json(Path(split_dir) / f"{split}_ids.json")
        if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids) or not set(ids) <= set(frozen):
            raise ValueError(f"Progress model {split} tasks differ from the frozen split")
    return model


def replay_ratio(old, replayed, joint_tolerance):
    if not old or len(old) != len(replayed):
        raise ValueError("Generated token and logprob lengths do not match")
    if not all(math.isfinite(value) for value in (*old, *replayed)):
        raise ValueError("Non-finite token logprob")
    differences = [b - a for a, b in zip(old, replayed)]
    joint_delta = math.fsum(differences)
    if abs(joint_delta) > joint_tolerance:
        raise ValueError(f"Old joint logprob replay failed: delta={joint_delta:.6g}")
    max_token_delta = max(map(abs, differences))
    if max_token_delta > TOKEN_LOGPROB_TOLERANCE:
        raise ValueError(f"Old token logprob replay failed: max={max_token_delta:.6g}, "
                         f"joint={joint_delta:.6g}, tokens={len(old)}")
    return math.exp(joint_delta)


def training_tasks(args, health):
    partitions = {}
    hashes = {}
    for name in ("train", "val", "test"):
        path = Path(args.split_dir) / f"{name}_ids.json"
        values = read_json(path)
        if not values or any(type(i) is not int or not 0 <= i < health["n_goals"] for i in values):
            raise ValueError(f"Invalid {name} split IDs")
        partitions[name] = set(values)
        if len(partitions[name]) != len(values):
            raise ValueError(f"Duplicate {name} split IDs")
        hashes[name] = file_hash(path)
    goals = read_json(args.goal_map)
    goal_texts = {row["task_id"]: row["instruction"] for row in goals}
    if set(goal_texts) != set(range(health["n_goals"])) or len(goals) != len(goal_texts):
        raise ValueError("Goal map is incomplete or contains duplicate IDs")
    instructions = {name: {instruction_hash(goal_texts[i]) for i in ids}
                    for name, ids in partitions.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if partitions[a] & partitions[b] or instructions[a] & instructions[b]:
            raise ValueError(f"Split overlap: {a}/{b}")
    ids = read_json(args.ids)
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 8
            or any(type(i) is not int or i not in partitions["train"] for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Select 1–8 unique frozen training task IDs")
    return ids, hashes, goal_texts


def replay(model, critic, turn):
    import torch

    prompt = turn["prompt_input_ids"]
    response = turn["response_token_ids"]
    ids = torch.tensor([prompt + response], device=model.device)
    result = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   output_hidden_states=True, use_cache=False)
    # At position N-1 the causal LM predicts the first sampled token.
    logits = result.logits[0, len(prompt) - 1:-1].float()
    logprobs = logits.log_softmax(-1).gather(1, ids[0, len(prompt):, None]).squeeze(1)
    # The value head has its own gradients. The actor is updated only by the policy loss.
    value = critic(result.hidden_states[-1][0, len(prompt) - 1].detach().float()).squeeze()
    return logprobs, value


def collect(model, tokenizer, critic, generation, base, task, cfg, expected_instruction, progress_model=None):
    import torch

    state = request(base + "/reset", {"task_id": task})
    instruction = state["instruction"].removeprefix("Instruction: ")
    if state["goal_fingerprint"] != cfg["goal_fingerprint"] or instruction != expected_instruction:
        raise ValueError(f"Live goal differs from frozen task {task}")
    messages = [{"role": "system", "content": SYSTEM + "\nShopping instruction: " + state["instruction"]},
                {"role": "user", "content": state["observation"]}]
    turns = []
    shaping = cfg.get("reward_mode") == "option_potential"
    online = cfg.get("reward_mode") == "neural_progress"
    prefix_rewards = [] if online else None
    previous_value = 0.0
    purchased, score = False, 0.0
    eos = generation.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    for step in range(cfg["max_steps"]):
        prompt_messages, prompt = fit_history(
            tokenizer, messages, cfg["context_length"] - generation.max_new_tokens, generation=True,
        )
        ids = torch.tensor([prompt], device=model.device)
        with torch.no_grad():
            result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    generation_config=generation)
        response = result.sequences[0, ids.shape[1]:].tolist()
        if not response or len(result.scores) != len(response):
            raise ValueError("Generated token and sampling score lengths do not match")
        old = [float(logits[0].float().log_softmax(-1)[token])
               for logits, token in zip(result.scores, response)]
        del result
        raw_text = tokenizer.decode(response, skip_special_tokens=False)
        text = tokenizer.decode(response, skip_special_tokens=True).strip()
        turn = {"turn_index": step, "prompt_input_ids": prompt, "response_token_ids": response,
                "response_mask": [1] * len(response), "old_logprobs": old,
                "old_action_logprob": math.fsum(old), "reward_token_index": len(response) - 1,
                "generation_stop": "eos" if response[-1] in eos else "max_new_tokens",
                "observation": state["observation"], "response_text_raw": raw_text,
                "response_text_parsed": text}
        with torch.no_grad():
            lp, value = replay(model, critic, turn)
        turn["initial_replay_ratio"] = replay_ratio(old, lp.tolist(), JOINT_LOGPROB_TOLERANCE)
        turn["old_value"] = float(value)
        if shaping:
            turn["potential_before"] = option_potential(instruction, state["available"].get("selected_options"))
        if online:
            value, fallback, skip_reason = previous_value, False, None
            if text:
                prefix = prompt_messages + [{"role": "assistant", "content": text}]
                prediction, fallback = progress_model(prefix)
                if prediction is None:
                    skip_reason = "prefix_too_long"
                else:
                    value = prediction
            else:
                skip_reason = "empty_response"
            prefix_rewards.append(value - previous_value)
            previous_value = value
            turn.update(prm_prefix_value=value, prm_asin_length_fallback=fallback,
                        prm_skip_reason=skip_reason)
        action, error = None, None
        try:
            action = parse_action(text)
            if not is_executable(action, state["available"]):
                raise ValueError("Action is not available in current state")
        except ValueError as exc:
            error = str(exc)
        if error is None:
            state = request(base + "/step", {"action": action})
            purchased, score = state["done"], state["reward"]
            feedback = state["observation"]
        else:
            feedback = "Observation: Invalid action or format. Reply with one valid Action: line."
        turn.update(parsed_action=action, parse_error=error, grounding=int(error is None),
                    environment_step_applied=error is None, environment_score=score,
                    next_observation=feedback, purchase_completed_after=purchased)
        if shaping:
            turn["potential_after"] = (0.0 if purchased else option_potential(
                instruction, state["available"].get("selected_options")))
        turns.append(turn)
        messages.extend([{"role": "assistant", "content": text}, {"role": "user", "content": feedback}])
        if purchased:
            break
    reward = assign_episode_rewards(turns, purchased, score, None if online else progress_model,
                                    shaping=shaping, prefix_rewards=prefix_rewards)
    return {"task_id": task, "turns": turns, "purchase_completed": purchased,
            "terminal_reward": reward, "environment_final_score": score,
            "end": "purchase" if purchased else "max_steps"}


def update(model, critic, turns, optimizer, parameters):
    import torch

    if not any(turn["advantage"] != 0 for turn in turns):
        raise ValueError("No nonzero policy advantages; collect another batch")
    optimizer.zero_grad(set_to_none=True)
    losses, ratios = [], []
    for turn in turns:
        lp, value = replay(model, critic, turn)
        ratios.append(replay_ratio(turn["old_logprobs"], lp.detach().tolist(), JOINT_LOGPROB_TOLERANCE))
        old_joint = torch.tensor(turn["old_action_logprob"], dtype=torch.float32, device=lp.device)
        ratio = (lp.sum() - old_joint).exp()
        advantage = turn["advantage"]
        actor = -torch.minimum(ratio * advantage, ratio.clamp(1 - CLIP, 1 + CLIP) * advantage)
        loss = actor + 0.5 * (value - turn["return_target"]).square()
        if not torch.isfinite(loss):
            raise ValueError("Non-finite PPO loss")
        (loss / len(turns)).backward()
        losses.append(float(loss.detach()))
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    if not model_has_gradient(model) or not model_has_gradient(critic):
        raise ValueError("Expected nonzero gradients in both adapter and critic")
    optimizer.step()
    return {"loss": math.fsum(losses) / len(losses), "gradient_norm": float(grad_norm),
            "replay_ratio_min": min(ratios), "replay_ratio_max": max(ratios)}


def model_has_gradient(module):
    return any(parameter.grad is not None and bool(parameter.grad.count_nonzero())
               for parameter in module.parameters() if parameter.requires_grad)


def tensor_digest(tensors):
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().float().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite checkpoint tensor: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def resume_metadata(path, config_sha, split_hashes, goal_map_sha):
    meta = read_json(path / "run.meta.json")
    summary = read_json(path / "summary.json")
    if (summary["run_id"] != meta["run_id"] or summary["updates"] != 1
            or not summary["same_process_reload"] or meta["schema_version"] != "rl-smoke-v0.1"):
        raise ValueError("Resume requires a completed and reloaded PPO run")
    if (meta["config_sha256"] != config_sha or meta["split_hashes"] != split_hashes
            or meta["goal_map_sha256"] != goal_map_sha):
        raise ValueError("Resume config or frozen data changed")
    return summary


def run(args):
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from transformers import GenerationConfig, set_seed

    cfg = read_json(args.config)
    base = cfg["environment_url"].rstrip("/")
    if not base.startswith("http://127.0.0.1:"):
        raise ValueError("Use a local WebShop bridge")
    if any(cfg.get(key, value) != value for key, value in
           (("temperature", 1.0), ("top_p", 1.0), ("top_k", 0), ("beta", 0.0))):
        raise ValueError("The smoke test requires unwarped sampling and beta=0")
    for key, upper in (("max_steps", 10), ("max_new_tokens", 128), ("context_length", 2048)):
        if type(cfg.get(key)) is not int or not 1 <= cfg[key] <= upper:
            raise ValueError(f"Invalid smoke-test budget: {key}")
    if cfg["max_new_tokens"] >= cfg["context_length"]:
        raise ValueError("Context budget must leave room for the prompt")
    health = request(base + "/health")
    if (not health["ready"] or not health["alignment_checked"] or health["alignment_mismatches"]
            or not health["split_status"]["verified"] or health["goal_fingerprint"] != cfg["goal_fingerprint"]):
        raise ValueError("WebShop health, split or goal fingerprint mismatch")
    ids, splits, goals = training_tasks(args, health)
    adapter = Path(cfg["adapter"])
    snapshot = Path(cfg["model"])
    if snapshot.name != cfg["revision"] or not snapshot.is_dir():
        raise ValueError("Use the pinned local model snapshot")
    base_hash = file_hash(snapshot / "model.safetensors")
    adapter_hash = file_hash(adapter / "adapter_model.safetensors")
    if base_hash != cfg["base_weights_sha256"] or adapter_hash != cfg["adapter_sha256"]:
        raise ValueError("Base model or adapter hash mismatch")
    config_sha, goal_map_sha = file_hash(args.config), file_hash(args.goal_map)
    resume = Path(args.resume) if getattr(args, "resume", None) else None
    parent = resume_metadata(resume, config_sha, splits, goal_map_sha) if resume else None
    if resume:
        adapter = resume / "adapter"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    run_id = str(uuid.uuid4())
    set_seed(cfg.get("seed", 42))
    policy_dtype = cfg.get("policy_dtype", "auto")
    if policy_dtype not in ("auto", "float32") or (policy_dtype == "float32" and cfg.get("load_in_4bit")):
        raise ValueError("Use auto dtype or an unquantized float32 policy")
    dtype = torch.float32 if policy_dtype == "float32" else None
    base_model, tokenizer, _ = load(cfg, training=True, dtype=dtype)
    model = PeftModel.from_pretrained(base_model, str(adapter), is_trainable=True).eval()
    critic = torch.nn.Linear(model.config.hidden_size, 1, device=model.device, dtype=torch.float32)
    torch.nn.init.zeros_(critic.weight)
    torch.nn.init.zeros_(critic.bias)
    generation = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                  max_new_tokens=cfg["max_new_tokens"], return_dict_in_generate=True,
                                  output_scores=True, use_cache=False, eos_token_id=model.generation_config.eos_token_id,
                                  pad_token_id=tokenizer.pad_token_id)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or any("lora_" not in name for name, p in model.named_parameters() if p.requires_grad):
        raise ValueError("Expected only LoRA policy parameters to be trainable")
    parameters.extend(critic.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.0)
    if resume:
        critic.load_state_dict(torch.load(resume / "critic.pt", weights_only=True, map_location=model.device))
        state = torch.load(resume / "training-state.pt", weights_only=True, map_location="cpu")
        if state["update"] != parent.get("cumulative_updates", 1):
            raise ValueError("Resume optimizer update index mismatch")
        optimizer.load_state_dict(state["optimizer"])
        optimizer_steps = {int(entry["step"]) for entry in optimizer.state.values()}
        if optimizer_steps != {state["update"]}:
            raise ValueError("Resume optimizer steps mismatch")
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    cumulative_updates = parent.get("cumulative_updates", 1) + 1 if parent else 1
    actor_before = tensor_digest(get_peft_model_state_dict(model))
    critic_before = tensor_digest(critic.state_dict())
    if parent and (actor_before != parent["actor_after"] or critic_before != parent["critic_after"]):
        raise ValueError("Resume weights do not match the previous run")
    write_json(out / "run.meta.json", {"schema_version": "rl-smoke-v0.1", "run_id": run_id,
        "scope": "one engineering update", "split": "train", "source_commit": git_head(),
        "config": cfg, "config_sha256": config_sha, "ids": ids,
        "ids_sha256": file_hash(args.ids), "split_hashes": splits, "goal_map_sha256": goal_map_sha,
        "parent_run_id": parent["run_id"] if parent else None, "update_index": cumulative_updates,
        "health": health, "sampling": generation.to_dict(), "torch_version": torch.__version__,
        "reward_mode": cfg.get("reward_mode", "terminal"),
        "algorithm": {"unit": "assistant_turn", "ratio": "joint_response", "gamma": 1.0, "lambda": 1.0,
                      "clip": CLIP, "beta": 0.0, "kl_coefficient": 0.0, "optimizer_updates": 1,
                      "learning_rate": 1e-5, "value_coefficient": 0.5, "max_gradient_norm": 1.0,
                      "joint_logprob_tolerance": JOINT_LOGPROB_TOLERANCE,
                      "token_logprob_tolerance": TOKEN_LOGPROB_TOLERANCE},
        "actor_before": actor_before, "critic_before": critic_before})
    progress_model = load_progress_model(cfg, args.split_dir)
    episodes = []
    try:
        for task in ids:
            episode = collect(model, tokenizer, critic, generation, base, task, cfg, goals[task], progress_model)
            episodes.append(episode)
            print(f"Collected task={task} turns={len(episode['turns'])} R={episode['terminal_reward']}", flush=True)
    finally:
        if cfg.get("reward_mode") == "neural_progress":
            progress_model.close()
    write_json(out / "episodes.json", episodes)
    turns = [turn for ep in episodes for turn in ep["turns"]]
    report = update(model, critic, turns, optimizer, parameters)
    actor_after = tensor_digest(get_peft_model_state_dict(model))
    critic_after = tensor_digest(critic.state_dict())
    if actor_after == actor_before or critic_after == critic_before:
        raise RuntimeError("Expected both LoRA and critic parameters to change")
    checkpoint = out / "adapter"
    model.save_pretrained(checkpoint)
    torch.save(critic.state_dict(), out / "critic.pt")
    torch.save({"optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "update": cumulative_updates}, out / "training-state.pt")
    with torch.no_grad():
        expected, expected_value = replay(model, critic, turns[0])
    model.load_adapter(str(checkpoint), adapter_name="smoke_reload")
    model.set_adapter("smoke_reload")
    model.eval()
    critic.load_state_dict(torch.load(out / "critic.pt", weights_only=True, map_location=model.device))
    with torch.no_grad():
        reloaded, reloaded_value = replay(model, critic, turns[0])
    if tensor_digest(get_peft_model_state_dict(model, adapter_name="smoke_reload")) != actor_after:
        raise RuntimeError("Reloaded adapter hash mismatch")
    if tensor_digest(critic.state_dict()) != critic_after:
        raise RuntimeError("Reloaded critic hash mismatch")
    replay_ratio(expected.tolist(), reloaded.tolist(), JOINT_LOGPROB_TOLERANCE)
    torch.testing.assert_close(expected_value, reloaded_value)
    write_json(out / "summary.json", {**report, "run_id": run_id, "updates": 1,
        "cumulative_updates": cumulative_updates, "parent_run_id": parent["run_id"] if parent else None,
        "turns": len(turns), "environment_calls": sum(t["environment_step_applied"] for t in turns),
        "generated_tokens": sum(len(t["response_token_ids"]) for t in turns),
        "actor_before": actor_before, "actor_after": actor_after,
        "critic_before": critic_before, "critic_after": critic_after, "same_process_reload": True})
    print(f"Completed one PPO update and same-process checkpoint reload: {out}", flush=True)
