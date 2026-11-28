# PR #60 TG vs 冻结 Arm 4 配对短训任务锁定与验证口径

2026-11-26，按 [PR #60 评论 #7195224609](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/60#issuecomment-7195224609) 分工，预先锁定配对短训任务 ID、执行顺序与 `val160` 验证口径。**`test200` 继续冻结（`test_authorization=false`）。**

## 1. 冻结短训任务清单与训练顺序（Seed 42，1 次 PPO 更新）
- **冻结任务顺序（4 题，每题 1 次在线采样）**：`[5034, 8742, 89, 5211]`
- **任务清单**：[pr60-ppo-shortrun-seed42-ids.json](pr60-ppo-shortrun-seed42-ids.json)（SHA-256 `07459cd86227b3624298126b8d27f2ddd60f005e7943ecbca2c4289034f4cad6`）
- **隔离审计**：[pr60-ppo-shortrun-seed42-audit.json](pr60-ppo-shortrun-seed42-audit.json)（SHA-256 `d6ed8019f48aa86f7b88dd38bf788ac7e6fd426e12303dc40d59134b8c661b52`）
- **抽样与隔离**：从 `train_ids.json`（1,451 题）扣除 `fit501`（501）、`#55 dev160`（160）、`#57 dev64`（64）、`#59 reverify dev64`（64）及 `{2581, 6285, 7544}` 后余 661 题，按 `SHA256(ASCII("pr60-ppo-shortrun-seed42:" + ID))` 升序取前 4 题；与上述全部训练/评估集及 `val160`/`test200` 交集全为 0。

## 2. 双臂同起点与同预算约束
- 开工前锁定共同 `source_commit`（含 `#59` 评分器 + How 在线前缀 PPO 接入 + `#60` 两处边界修复）、同一冻结 B1 SFT 起点及冻结 Arm 4 检查点（`prm_manifest_sha256 = 826f9cce…`）。
- TG 臂（`terminal`）与 Arm 4 臂（`neural_progress`）共用 `seed=42`、相同顺序 `[5034, 8742, 89, 5211]`、`updates=1` 与相同采样/步数/优化配置，仅更新策略 LoRA 与 PPO 价值头；同步记录实际步数、生成 token、耗时、峰值显存、`prm_skip_reason` 与检查点重载。

## 3. 冻结 `val160` 验证与高神经奖励异常单列口径
- **验证集**：冻结 `results/env-2026-11-17/val_ids.json`（160 题，SHA-256 `45440d724b33e69e13916d0adbe243cc63ce3bd67677b8220feff9f30b866c36`），B1 / TG / Arm 4 共用同一 FP32 贪心解码与步数上限。
- **指标**：主指标为满分购物成功率（`env_score == 1.0`），辅指标为平均环境终局分、购买完成率、平均步数与非法动作率（附配对 95% Bootstrap CI 与逐题对照）。
- **异常单列（强制）**：在训练 4 题与 `val160` 逐题记录中，单列所有「零分购买（`purchase_completed=true` 且 `env_score=0`）」和「未完成购买（如冒烟 `6285` 终局分 0 但神经奖励合计约 `+0.36`）」却获较高神经奖励（如 `sum(prm_reward) >= 0.30`）的轨迹。
- **边界**：本轮短训用于校准交互预算与奖励行为，为后续 `42/43` 多种子正式对照定预算，`test200` 保持冻结（`test_authorization=false`）。
