# PRM sampling guard and bounded diagnostic

## Problem and measured evidence

The previous requested 1,451-task rollout stopped after 911 unique tasks, an exact
ordered prefix ending at 6395; 6426 was the next requested task. The sampler raised
`Instruction/current exchange exceeds context budget`. The shell chain did not
check its failure and trained on the partial output. Original logs/checkpoints are
retained. The old local chain now delegates to the checked pipeline rather than
running its unsafe commands. No missing task is relabelled as reward zero.

The saved 500-step model has post-training fit MSE 0.323010 and independently
reproduced dev MSE 0.277643 (constant 0.1198; length baseline 0.1099). Prediction vs
turn-count correlation is 0.973682 on fit and 0.960380 on dev. All 500 steps clipped
gradients, median preclip norm 2890.6. These observations do not prove that clipping
caused failure. A 20-input inference diagnostic found mean absolute effect of
disabling PRM LoRA 0.000821 and train/eval dropout difference 0.00000527. Parameter
changes alone do not establish effective representation learning.

Filtering retained 628/911 trajectories (fit 501, dev 127), rejecting 283 overlong
ones without truncation. Rejected examples have mean reward 0.25758 versus
0.61281 for retained examples. 154 of 187 max-step failures were rejected. This
selection bias must be disclosed and addressed before claiming broad coverage.

## Fix

`scripts/prm_checked_pipeline.py` runs argv lists without a shell and stops on the
first nonzero exit. Before conversion/training it requires exact ordered task
coverage, unique requested IDs, matching run/repeat metadata, and normal purchase
or max-step termination. Subsequent stages recheck input hashes and coverage.
Existing attempt and rollout output paths are refused. Logs, frozen commands and
atomic status are retained. The caller must acquire the project GPU locks; this
script itself does not own the device. It does not replace data.py's label,
policy-identity and train/val/test instruction gates.

Plan schema: `cwd`, `expected_ids`, `rollout`, `metadata` are absolute paths;
`stages` is a list of `{name, command: [argv, ...]}`, starting with exactly one
`sample` stage, followed by conversion and training. Invoke:

```text
python scripts/prm_checked_pipeline.py --plan /absolute/plan.json --output /new/attempt
```

Six CPU subprocess regressions pass on Windows and WSL: sampler error; success
exit with partial output; duplicate task; successful exact coverage; technical
termination; metadata/run mismatch. These tests are not GPU evidence.

## Frozen diagnostic design

This follows the user's instruction to start the next step after diagnosis. It is
an explicitly partial, exploratory diagnostic, not the repaired full rollout or
the final learned-progress PPO experiment. PRM structure, B1 initialization, new
PRM LoRA, FP32, terminal-sum MSE, gradient clipping and weight decay remain the
same. No final test tasks or policy-val trajectories are used.

- Fixed 64 retained fit examples: 16 full, 16 partial, 16 purchased-zero, and 16
  max-step-zero trajectories. Selection RNG 20261124 within the old fit partition.
- Fixed 28 retained train-pool dev examples: 8 full, 8 partial, 8 purchased-zero,
  and all 4 available max-step-zero. Enriched diagnostic distribution is not an
  estimate of population performance; the dev set has already been inspected.
- Three independent fresh initializations from original B1 and zero score head:
  sorted input at lr 1e-7; the same input shuffled once with seed42 at lr 1e-7;
  that same shuffled input at lr 1e-8.
- 128 actual optimizer steps each (two passes over 64 examples), 384 total. Each
  arm saves its own LoRA/head/optimizer/RNG. No continuation from the failed-fit
  500-step model, no adaptive extension or checkpoint picking.
- Independent process reload and per-step comparison at tolerance 1e-5; evaluate
  both fit and dev, report MSE, constant/length baselines, length correlation and
  raw aligned step scores. Failures stop the serial supervisor.

The runtime is the unchanged clean integrated local commit
`a67db6abefc73c6f6c7f1efb78645086782b1ade`, assembled from PRs #43/#44/#45 plus
local integration. It is not claimed to be merged main. The protocol records the
actual source hashes, full command/config and frozen selection. This PR adds only
guard/diagnostic orchestration and evidence; it does not merge teammates' PRs.

Comparisons can identify order/rate sensitivity on this diagnostic sample. They
cannot alone establish correct stepwise attribution or choose final training
hyperparameters. Long-trajectory handling remains unresolved: do not silently
truncate a full-R target, increase context, skip the next task, or restart over
old output. Such changes need an explicit new sampling design and capacity check.

Zqqq retains team scheduling. Test is not released. Current GPU progress and
results must come from the device records, not this design document.
