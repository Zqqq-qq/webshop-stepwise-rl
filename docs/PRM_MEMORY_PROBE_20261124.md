# Eight-example PRM memorization probe

The user explicitly authorized this next diagnostic after the three-arm study.
This is fit-only: the same eight examples are used for learning and acceptance.
No dev, policy validation, reserved test or learned-progress PPO is used.

## Fixed design

- From the previous fixed64 fit partition, select the two shortest complete
  retained trajectories in each bucket: full, partial, purchased-zero,
  max-step-zero. This deliberate easy/short selection is appropriate for a
  memorization sanity check, not a generalization estimate. Shuffle once seed42
  and repeat that fixed order. There are no identical input conversations.
- Task IDs in actual training order: 351, 95, 5478, 3542, 359, 5031, 0, 6225.
  Labels are respectively 2/3, 0, 0, 0, 2/3, 0, 1, 1. Both max-step failures
  contain 10 assistant turns. Raw purchase and termination semantics were
  already checked against the source911 rollout.
- Native tokenizer preflight retained all eight without truncation. Complete
  lengths are 1007, 902, 903, 914, 944, 952, 1037 and 1052 tokens. Every assistant
  score position decodes to `<|im_end|>`; task-specific future rewards are labels
  only, never added to token input.
- Fresh original B1 plus independent new PRM LoRA and zero head. Unchanged local
  clean core `a67db6abefc73c6f6c7f1efb78645086782b1ade`; no architecture, loss,
  precision, optimizer or sampling change hidden in the probe driver.
- FP32/highest precision, TF32/autocast off. Existing lr1e-7, AdamW weight decay
  0.01, clip norm1, LoRA dropout0.05 for training and eval mode for prediction.
  Each update uses one complete trajectory; eight updates constitute one pass.
- At most512 updates (64 passes). Save/evaluate all eight in eval mode every64
  updates, restoring this run's actor/head/optimizer/CPU/CUDA RNG in the next
  process. Check parameter digest continuity and actual cumulative updates.
- Predefined acceptance: MSE strictly below0.001 AND every sample absolute error
  strictly below0.1. Stop at the first saved checkpoint satisfying both, or at
  the fixed512 cap. No automatic extension or hyperparameter search.
- Independently reload the final full adapter/head in a fresh process. Compare
  aligned per-step scores against saved predictions at tolerance1e-5. Record
  per-sample errors, losses, preclip norms, head/LoRA gradients, hashes, time and
  memory. First-step zero LoRA gradient is expected from a zero head; inspect
  subsequent updates rather than treating that first zero as a broken path.

## Interpretation and implementation

`scripts/run_prm_memory_probe.py` orchestrates the existing trainer; it never
changes the running core. GPU/validation locks exclude other project jobs.
Atomic progress and distinct output directories preserve every checkpoint.
The CPU suite includes both metric-only tests and direct calls to the real
`evaluate` and diagnostic `assess` entry points using a lightweight fake runtime.
It checks missing/extra/duplicate tasks, short/long/empty score vectors, shifted
positions, NaN/Inf on both sides, total/step inconsistency, matching totals with
different steps, and accumulated total drift. These tests do not prove GPU
learning. The real token boundary preflight is recorded separately.

A passing test establishes only the ability to fit these eight trajectories.
A failure establishes failure of this particular bounded optimizer setup, not
that the architecture or the paper cannot learn. In that case prioritize an
optimizer/implementation investigation before more all-data training; do not
label the failure as overfitting or silently change the model.

The previously completed384 updates and their results remain separate. Zqqq
continues to own scheduling. Check device progress/results for actual completion.

## Reload-gate correction and next device handoff

The original controllers could accept malformed reload predictions. Both now use
`scripts/prm_reload_validation.py`: exact nonempty task coverage, unique integer
IDs, nonempty strictly increasing score positions, equal step counts, finite
steps/totals, total equal to step sum, and aligned step **and total** comparisons.
The explicit tolerance is `atol=1e-5, rtol=0`. The final parent status uses the
independently reloaded fit metrics, not just the pre-reload reference metrics.

The historical plan in `results/prm-memory-20261124/plan.json` is an immutable
design record pointing to the original controller SHA. Do not change that plan
or existing progress/checkpoints in place. On the device, prepare a new attempt
directory and plan; retain the same eight samples, labels, B1 initialization,
hyperparameters, 64-update interval and 512-update cap. Copy the revised controller
**and its sibling validator** together, and freeze their absolute paths and SHA256
values in the new plan's `input_sha256`. Freeze the new initial config/output
paths as well. The controller rejects a plan without the validator hash before
training starts. Record the new controller commit and `supersedes_plan` in the
new plan. No change to the frozen model core or loss is needed for this fix.

Run `python3 -m unittest discover -s tests -p 'test_prm*.py' -v` in the revised
checkout first, then use the device's existing environment with
`python -u /absolute/new-attempt/run_prm_memory_probe.py --plan
/absolute/new-attempt/plan.json` (one shell command). Preserve the command, logs,
all checkpoints/reference predictions, independent-reload JSON, SHA manifest and
actual stop reason in a private evidence package. If an old attempt already ran,
preserve it and evaluate its original checkpoint with the corrected validator in
a separate output location; do not overwrite its historical evaluation file.

This repair and its CPU tests are not a new GPU execution report. Until device
results are actually archived and reviewed, memory-test and generalization
acceptance remain pending. A bad old gate does not by itself invalidate the
separately audited three-arm predictions, nor require rerunning those 384 updates.
