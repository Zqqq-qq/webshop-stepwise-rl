# Completed PRM diagnostic (2026-11-24)

All three arms completed, 128 actual updates each, 384 total. Fresh common B1/zero
head initialization and saved manifest hashes verified. Independent-process
per-step reload difference is zero (see JSON for exact values).

Fixed balanced 64-task fit / 28-task already-viewed train-pool dev; no policy-val
or reserved test. These are exploratory comparisons, not final generalization
acceptance or evidence of correct attribution.

| arm | updates | fit MSE | dev MSE | dev prediction/turn correlation |
|---|---:|---:|---:|---:|
| sorted-lr1e7 | 128 | 0.2661 | 0.3038 | -0.5285 |
| shuffled-lr1e7 | 128 | 0.4327 | 0.2982 | 0.9874 |
| shuffled-lr1e8 | 128 | 0.3183 | 0.3654 | 0.9147 |

Common constant baseline MSE 0.1942; common length baseline
MSE 0.1445. Compare only within this fixed diagnostic, not
against the earlier 127-task dev numbers. Full collection and long trajectories
remain unresolved; no automatic full training, test, or learned-progress PPO.

Private complete device evidence: [prm-diagnostic-evidence-20261124.zip](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm-diagnostic-evidence-20261124.zip).
SHA256 `e1578d582bdc1deaa5954e0b33e869d137eba0cb8d3efd7c57ecd7fcdadf2ec7`; 70 payloads individually rehashed after ZIP
extraction, GitHub asset digest matched. Contains actual drivers/plan, selected
data, all three checkpoints/optimizer/RNG, summaries, independent predictions,
logs, source snapshot and partial911 source/audits. No base weights or credentials.
