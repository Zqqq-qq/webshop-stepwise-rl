# Issue #48: opt-in hidden-state PRM implementation candidate

Owner for this follow-up: Yuxiao. **Foundation and attribution:** Jun's
`docs/PRM_ISSUE48_STRUCTURE_AND_VARIANT_PLAN.md`, `variant_contract.py`,
paired proposal envelopes, and CPU Qwen2/PEFT probes at commit `6fe97a0`.
This patch extends that plan; it does not claim to replace or reauthor it.

## Scope and status

This is a **reviewable implementation candidate**, not an approved training
protocol, finished GPU acceptance, an improved reward model, or permission to
use the reserved test split. Existing v1 logits checkpoints remain untouched.
No checkpoint is silently converted, and no optimizer settings, sampling
budget, rating labels, score positions, reduction, or backbone are changed.

The original code path stays the default:

- v1 config: `schema_version=prm-qwen-config-v1`, implicit
  `representation=vocab_logits`;
- v1 model: assistant `im_end` position -> raw vocabulary logits ->
  `Linear(V,1)` -> sum -> terminal MSE;
- v1 checkpoints: original manifest schema 1 and score definition.

For a new, opt-in paired configuration, start from a **separately copied**
validated original B1 configuration, give each arm a unique output directory,
and explicitly set:

| Arm | Configuration additions | Scoring head |
|---|---|---|
| logits control | `schema_version: "prm-qwen-config-v2"`, `representation: "vocab_logits"` | `Linear(151936,1)` |
| hidden candidate | `schema_version: "prm-qwen-config-v2"`, `representation: "final_hidden_state"` | `Linear(1536,1)` |

These are changes to the **runtime config object**, not to the outer
`configs/prm/proposals/*-v1.json` envelopes. The latter are Jun's inert
proposals and must not be passed to the training CLI. A v2 config must explicitly
name its representation; v1 cannot opt into hidden scoring.

## Feature and gradient path

At the exact same assistant `im_end` indices, hidden mode calls the underlying
Qwen decoder (`PEFT.get_base_model().model`) and consumes its final-normalized
`last_hidden_state`. It neither requests every intermediate hidden layer nor
constructs all vocabulary logits. The decoder's LoRA modules remain on the
gradient path: features are **not detached**. It keeps zero-initialized head,
FP32, the same sum/terminal MSE, and existing masking/padding validation.

The new definition is
`assistant_im_end_final_norm_hidden_linear_sum_mse`, separately registered
from `assistant_turn_end_vocab_logits_linear_sum_mse`. This is a variant of the
public code, not a claim that the original published model trained successfully.

## Checkpoint contract

- v2 contains an `architecture` object from Jun's
  `RepresentationContract` with representation, input dimension, head type,
  score position, reduction, loss, dtype, initialization and a distinct score
  definition; source/base/B1/tokenizer identity and every file SHA remain
  mandatory.
- On reload the runtime validates expected config representation/schema,
  model dimensions, identity and file hashes **before** full backbone load.
  The head is then loaded with `weights_only=True` and its keys, shape,
  FP32 dtype and finite values checked. Missing or changed files fail closed.
- A v2 hidden head is not loadable as v2 logits, nor the reverse. Old v1 logits
  checkpoints cannot load as hidden. Historical v1 strict source identities
  stay pinned; this patch does not promise to remap them to a newer runtime.
- Separate directories/checkpoints are mandatory for the paired experiment.
  No run-time architecture toggle is allowed inside one existing checkpoint.

The training CLI still records a separate optimizer/RNG state for any new
checkpoint. Actual CPU/GPU resume, especially optimizer parameter ID mappings,
remains a required integration acceptance test.

## Verification performed versus pending

Performed in an environment without Torch:

- Python syntax compilation of modified sources and tests;
- `PYTHONPATH=src python -m unittest -v tests.test_prm_variant_contract`:
  **6/6 passed**, including v1 serialization stability and v2 opt-in validation.

**Not performed here:** PyTorch model-core tests, actual Qwen2/PEFT integration,
real 1.5B device training, new-process v2 save/reload, resume, VRAM profiling,
or any fixed/held-out task evaluation. Added/extended Torch-based tests exercise
hidden feature use, LoRA gradients after zero-head initialization, and strict
v1/v2 compatibility; they **must actually run** in Jun/How's compatible isolated
CPU environment before this candidate is eligible for GPU smoke testing.

Before any execution, Zqq must freeze sample SHA, source commit, per-arm
optimizer settings, matching update budget, evaluation cadence and stop gates.
Use new evidence directories, record OOM and validation failures, keep the
reserved test isolated, and do not call a fixed-feature SVD fit an end-to-end
neural PRM result.
