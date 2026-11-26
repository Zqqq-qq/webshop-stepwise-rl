# TG and frozen Arm 4 PPO comparison

## Versions and budget

Use one clean commit containing the Arm 4 scorer, online PPO integration and #60 boundary fixes for all runs. Before running, record the exact commit, both configuration hashes, the hash of each ordered task batch, and the initial actor and critic hashes. Arm 4 scoring uses runtime commit `d25e6164894c87fdea796bb73e3ee546807483d8`, the unchanged Arm 4 manifest `826f9cce3a51365ec6bf02fb10574ca4990010a8927e9082b5a0383fd028b01c`, and the [versioned scoring contract](../src/webshop_lab/prm/prefix_value_contract.json). The PPO run metadata records the contract hash.

Start both arms from the same B1 SFT adapter (`c39e2a78e7c5973328e9d2c23fb6320e566352e63cc089562488f3dd46ae5f9c`) and matching initial PPO critic. Use reward mode `terminal` for TG and `neural_progress` for Arm 4. Keep Arm 4 frozen. The proposed fixed budget is seeds 42 and 43, four updates per seed and arm, and four mapped-train tasks per update. Freeze the ordered batches before the first run. Each arm collects its own trajectories, with matched task order, rollout caps, decoding settings and PPO hyperparameters. Record actual environment steps, generated tokens, wall time, peak memory and skipped scores. Resume only from the previous update in the same arm and seed.

Use the same frozen `val160` (`45440d724b33e69e13916d0adbe243cc63ce3bd67677b8220feff9f30b866c36`) and FP32 greedy decoding for both arms. Report paired full-reward success and environment terminal reward, and retain per-seed outcomes, purchase completion, step count and invalid actions. The two arms compare complete reward definitions. Keep `test200` frozen.

## Scoring and audit

Run the scorer from the clean pinned runtime with its `src` on `PYTHONPATH`. Set the policy context and generation budget to the settings used to produce the input rollout. The default CLI scores each action using its current prefix, just as the PPO worker does. Score indices are local to each prefix after truncation. The read-only evaluation output includes each prefix hash, skip reason and local score position.

```sh
PYTHONPATH="$EXEC_REPO/src" python "$EXPERIMENT_REPO/scripts/score_prefix_value_prm.py" \
  --source-commit d25e6164894c87fdea796bb73e3ee546807483d8 \
  --config "$ARM4_CONFIG" --checkpoint "$ARM4_CKPT" \
  --policy-context-length "$VAL_CONTEXT" --policy-max-new-tokens "$VAL_MAX_NEW" \
  --rollout "$VAL_ROLLOUT" --output "$VAL_SCORES"

python "$EXPERIMENT_REPO/scripts/audit_ppo_neural_rewards.py" \
  --training-episodes "$ARM4_UPDATE_DIR/episodes.json" \
  --validation-rollout "$VAL_ROLLOUT" --validation-scores "$VAL_SCORES" \
  > "$REWARD_AUDIT"
```

The audit reports all zero-score purchases and non-purchases in the denominator, and lists those whose cumulative neural reward reaches **0.30**. It distinguishes rewards actually used in PPO from read-only validation scores. Keep the source trajectories, per-turn rewards, skip reasons and evidence-file hashes. Review repeated cases of high neural reward with zero environment reward before allocating more training. Earlier archived scores use the historical whole-trajectory augmentation policy. For an exact archival reproduction only, pass `--historical-full-trajectory` with the original reference. The old pinned runtime is preserved for loading original checkpoint bytes. Use its prefix-value launcher for Arm 4 instead of invoking its historical sum inference module directly.
