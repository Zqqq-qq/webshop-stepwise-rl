# #55 设备预检证据包与 160 替代清单独立核验

2026-11-25，针对 [#55 固定提交 `3fa1b9d8`](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/55) 及 How 上传的私有证据包 [`prm55-preflight-evidence-20261125.zip`](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm55-preflight-evidence-20261125.zip) 做只读独立核验。机器记录：[pr55-preflight-handoff-audit.json](pr55-preflight-handoff-audit.json)。未修改原冻结文件，未执行 rollout/训练，未放行 test（`test_authorization=false`）。

## 1. 证据包完整性与设备/配置身份

- 外层 ZIP 16,573 字节，SHA-256 `a6034a1b192e834d492d52c1d32d82973a5e561250771e2407941f6924657022`，与 `data-v1` Release digest 一致。包内 `SHA256SUMS` 全部 10 项载荷逐字节匹配，4 份声明文件 SHA 均吻合：
  - `preflight-summary.json`：`c1ed10885a1d0b5c60eb2e7f302ff28cd5282f008fb708849c7f587261fa894f`
  - `full-history-audit.json`：`a3cad07fd70d7bb1183ac71e5f1a922c720f4a1106785bef348c8e3703ee4d2a`
  - `eligible-history-audit.json`：`66a9d5d0542d6cf21976a00c7a6892f622e504098d82c2ff439e5048aa0278ee`
  - `holdout-replacement-candidate.json`：`15b96f015ed8a2b804633523144011eb2fdf092903ccded17395020b925eb5c1`
- 4 项冻结输入（`holdout_ids.json`、`holdout-audit.json`、`dev-rollout.json`、`b1-pr18.json`）SHA 均与 `3fa1b9d8` 匹配；`dev-rollout.local.json` 与冻结配置仅差本地 `adapter` 路径（`/root/autodl-tmp/inputs/b1-adapter`），其余采样、种子、上下文、revision 与 `goal_fingerprint` 均一致。
- `model-load-preflight.json` 记录基模、4 份 tokenizer 文件、B1 adapter、`generation_config.json`（SHA `e558847a…`、`top_k=20`、`repetition_penalty=1.1`）及环境 `/health`（`alignment_mismatches=0`、`split_status.verified=true`），且 `rollout_executed=false`、`training_executed=false`。注：`finalize_preflight.py` 中 `cuda_kernel_passed` 与 `fit501_retained_bytes_sha256` 为前置检查后写入的常量，不冒充在汇总脚本内重新计算。

## 2. 85 个排除项的逐文件追溯与分类核实

- `full-history-audit.json`（46 文件、27 个唯一 ID：PRM 23 / RL 19 / other 3）与 `eligible-history-audit.json`（87 文件、85 个唯一 ID：PRM 72 / RL 56 / other 11）内部逐类并集完全自洽，且前者恰为后者在原冻结 160 上的限制；两份均记录 `scanned_json_files=3583`、`skipped_json_files=[]`。
- **实际来源拆解**：
  - **84 个**确有真实本地 PRM / PPO-RL / 进度估计器记录：`prm-data-20261123` / `prm-run-001`（PR #46 200 题 PRM 数据集）71 个、`rl-full-coverage-20261123`（rounds 1–34）50 个、`fp32-full160` / `ppo-compare` 3 个（`7793, 9014, 9258`）、`nov21` PPO 2 个（`9818, 10518`）、`progress-estimator` / `progress-explore` / `progress-action-pairs` 10 个（类别有重叠）。
  - **1 个（`9067`）仅出现在 7 份复制工作树的 `handoffs/sft-smoke/evidence/sft-label-length-audit.json` 中**（该文件是 `train` 1451 离线 SFT 长度审计的 `skipped_details`，并非 PRM/PPO/进度估计器运行），因父目录名含 `prm-`/`fp32-full160`/`rl-full-coverage` 而被同时归入三类。去除 `9067` 后真实非 SFT 暴露为 **84 个**（PRM 71 / RL 55 / other 10）；不排除 `9067` 时其排序位于 `181/446`，**不影响替代 160 名单**，按保守口径一并排除无害但应如实注明来源。
- 对照仓库及 `data-v1` 已公开发布产物（PR #46、`webshop-nov21-b1-ppo-six-update-evidence.zip`、`webshop-nov22-seed42-43-ppo-four-update-evidence.zip`、`webshop-project-fp32-full160-training-evidence-20261123.zip`、`prm-dev-eval-evidence-20261124.zip`）：已独立交叉印证 **15/85**（含原 160 中 **5/27**：`7033, 9014, 9142, 9807, 9818`），剩余 445 中按 `task_id` 命中 **0**。

## 3. 替代 160 复算与关键阻塞项：候选外同指令核查

- 从 530 排除 85 后余 445，按 `SHA256(ASCII("42:" + ID))` 升序重选 160：保留原 160 中 **133** 个且相对顺序不变、移除 27 个、新增 27 个，标准 JSON 字节 SHA-256 为 **`15b96f015ed8a2b804633523144011eb2fdf092903ccded17395020b925eb5c1`**，与 `holdout-replacement-candidate.json` 逐字节一致。
- 用锁定 `goal-instructions.json`（SHA `6b5d9dfc…`）核验归一化指令哈希：新 160 内部 160/160 唯一，与 `used(911)`、`val(160)`、`test(200)`、`6426`、原指令重合 6 题及本轮 85 题的指令交集均为 0。
- **为什么尚不能正式签收重冻结**：审阅包内源码发现 `full_history_audit.py:23` 仅匹配 `task_id in holdout`（160），`eligible_history_audit.py:24` 仅匹配 `task_id in targets`（530），**两者均丢弃了 530 以外的全部实际 `task_id`**，因此无法单凭本包完成 Jun 与 Qiu 要求的“候选外实际使用任务与新 160 的归一化指令隔离核对”。
- **已查实一例真实的候选外同指令暴露**：在全量 12,087 题 `goal-instructions.json` 中，剩余 445 题共有 **6 题**与库内其他任务 ID 共享完全相同的归一化指令：
  - **在替代 160 内的 2 题**：`8547`（排第 11）↔ **`8000`**，`8778`（排第 17）↔ **`2525`**（两者均在 train/val/test 外）；
  - **在剩余 445 但不在替代 160 内的 4 题**：`6755`（排第 226）↔ **`7900`**（`train` 内，**已实证出现在 PR #46 `dataset-200-manifest.json` 的 `dropped` 记录中**，故 `6755` 须从洁净候选池移出，余 444 题，不影响前 160）、`9085` ↔ **`9381`**（`train`）、`11501` ↔ **`11798`**（`train`）、`7610` ↔ **`6045`**（split 外）。

## 4. 结论与最小闭环步骤

- **排除口径认可**：按全部真实 PRM / PPO-RL / 进度估计器历史记录排除，且叠加“按全部实际使用 ID 的归一化指令哈希排除”；B1 既有 SFT 暴露仍须单列保留。
- **当前暂不签收正式重冻结**：请 How 在同一 7 个目录（3,583 份 JSON/JSONL）上做一次**不按 530 过滤**的只读扫描（同步覆盖 `task_id`、`task_ids`、`sample_keys`），至少回传 **`8000`、`2525`**（以及 `9381`、`11798`、`6045`、`7900`）的实际命中与指令哈希核验记录及 SHA：
  - 若 `8000` 与 `2525` 均 0 命中，则替代 160 名单（`15b96f01…`）即可正式签收重冻结（洁净候选池记 444 扣除 `6755` 及任何其余同指令命中项）；
  - 若 `8000` 或 `2525` 有实际记录，须同步排除对应 `8547` / `8778` 后按原 SHA-256 规则递补再冻结。

## 可复核入口

```bash
python scripts/audit_prm_holdout_protocol.py --mode replacement-handoff \
  --repo /local/repository --goal-map /evidence/goal-instructions.json \
  --handoff-zip /evidence/prm55-preflight-evidence-20261125.zip \
  --git-ref origin/results/prm-device-20261124 \
  --archive /evidence/webshop-nov21-b1-ppo-six-update-evidence.zip \
  --archive /evidence/webshop-project-fp32-full160-training-evidence-20261123.zip \
  --output /new/pr55-handoff-audit.json
```
