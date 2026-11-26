# #57 神经 PRM 选品分叉短训：未查看独立 dev 冻结清单与验收协议

2026-11-26，按 [#55 11/26 分工](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/55#issuecomment-7169543431) 与 [#57 提交 `f34c880`](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/57)，由 Qiu 侧独立冻结本轮选品分叉短训（`terminal` vs `ranking`）的**未查看独立 dev 清单**并明确验收口径。

- 冻结 64 题总清单：[prm57-independent-dev-ids.json](prm57-independent-dev-ids.json)（SHA-256 `4e0e86abd884cd01b80388e271f7a0c96d183b11d4690fefd8bd89b9a002177b`）
- 独立交付主评 32 题清单：[prm57-independent-dev-primary_32-ids.json](prm57-independent-dev-primary_32-ids.json)（SHA-256 `dcb9495a2d10b38512b51f8d3fafd286adf74a0c5bfaa0e4a18d24ed454c2305`）
- 独立交付备选 32 题清单：[prm57-independent-dev-reserve_32-ids.json](prm57-independent-dev-reserve_32-ids.json)（SHA-256 `8982bbeec537a147972dca9b367b81b23280cdb83910c59a176dfb7e2d5a7d11`）
- 完整隔离审计记录：[prm57-independent-dev-audit.json](prm57-independent-dev-audit.json)
- 冻结/隔离与验收脚本：[freeze_prm57_independent_dev.py](../../scripts/freeze_prm57_independent_dev.py) · [evaluate_prm57_independent_dev.py](../../scripts/evaluate_prm57_independent_dev.py)

## 1. 抽样来源与分层结构

- **来源池**：直接承接 [#55 `a601999`](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/55) 已完成设备历史补查与同指令去重的 **444 题洁净候选池**（全部来自 mapped `train`，已排除 `used(911)`、`6426`、设备暴露 85 题、`7900` 同指令项 `6755` 及与 `val/test/used` 同指令项）。
- **无偏确定性抽取**：沿用既有 `SHA256(ASCII("42:" + ID))` 升序排列，跳过 #55 已查看的前 160 题（`dev160`），取紧随其后的 **第 161–224 位（共 64 题）**，分为两段（按顺序优先执行前 32 题，若因超长或环境同分导致有效对比不足，再按顺序启用后 32 题，不挑题）：
  - **主评 32 题（`primary_32`，第 161–192 位）**：SHA-256 `dcb9495a2d10b38512b51f8d3fafd286adf74a0c5bfaa0e4a18d24ed454c2305`。
  - **备选 32 题（`reserve_32`，第 193–224 位）**：SHA-256 `8982bbeec537a147972dca9b367b81b23280cdb83910c59a176dfb7e2d5a7d11`。

## 2. 隔离核验（全部 0 重叠）

基于锁定 `goal-instructions.json`（SHA-256 `6b5d9dfc…`）、#55 归档包（`0834998f…`）与 #57 `train-candidates.json`（32 题，`f34c880`），已独立确认这 64 题满足：

- **任务 ID 交集均为 0**：与 #57 `train-candidates`（32 题）、`fit501`（501 题）、#55 已查看 `dev160`（含 `7084, 7113, 7402, 7489, 7544, 8669, 9984`）、mapped `val`（160 题）、mapped `test`（200 题）交集全为 **0**。
- **指令哈希交集均为 0**：64 题内部归一化指令与去价格预算指令均 **64/64 唯一**，与上述全部集合的归一化指令哈希及去价格预算指令哈希交集全为 **0**。
- **全库 12,087 题无跨 ID 同指令孪生**：在全量 12,087 条 WebShop 指令中，这 64 题**不存在任何跨任务 ID 的同指令孪生项**（0 例）。

## 3. 交给 How 的采集口径与“启用备选 32 题”盲态门控（查看模型预测前决定）

1. **主评 32 题采集（按冻结顺序，不挑题）**：
   - 先用与 #55 相同的冻结 B1 配置（`dev-rollout.json` SHA `4e9f7ccd…`，4-bit NF4、seed 42、temp 1、top_p 1、top_k 20、rep 1.1、max_new_tokens 128、context 2048、max_steps 10）在 `primary_32`（32 题）上各跑 1 条自然轨迹，全部保存（用于总分 MSE 基线评估，超 2048 token 整条剔除且不补样）。
   - 对其中第 0 步执行 `search[...]`、结果页含 `≥2` 个不同 ASIN 且第 1 步打开某商品 `click[asin_0]` 的任务，按与 #57 相同的同页选品分叉规则重放：保持搜索步与分叉前结果页完全一致，一臂走 `Action: click[asin_0]` → `Action: click[buy now]`，另一臂按结果页末尾两个不同备选 ASIN 依次尝试 `Action: click[alt_asin]` → `Action: click[buy now]`。记录每次尝试的真实环境终局分，遇到**首个与首臂环境终局分严格不同（`env_high > env_low`）的备选**即保留该对（按真实终局分确定高/低臂）；若两备选均同分，登记为 `environment_tie`。
2. **启用备选 32 题的前置规则（必须在运行或查看任何 PRM 预测之前决定）**：
   - 仅统计 `primary_32` 中同时满足 **（a）环境终局严格有分差（`env_high > env_low`）** 且 **（b）两臂均在 2048 token 上限内可评分** 的有效对数 `N_valid`。
   - 若 **`N_valid >= 20`**（与训练 20 对同量级）：**停在主评 32 题**，不再采集 `reserve_32`，直接冻结该批 dev 输入进入三份检查点只读推理。
   - 若 **`N_valid < 20`**：在**不看任何模型预测**的前提下，按冻结顺序整批启用 `reserve_32`（固定分母 64 题，避免数据依赖早停），完成同样采集与 2048 上下文筛查后再统一进入三份检查点只读推理。
3. **三份检查点只读推理与回传清单**：
   - 在 `559a7791` 干净 checkout 上，对同一份冻结 dev 输入（含有效选品对与全部保留自然轨迹）分别用 `stage126`、`terminal`、`ranking` 做只读推理；另对 #56 归档的 `7544` 两条轨迹分别推理作独立回归。
   - 回传：尝试清单（有效对数 / 同分 / 超长 / 非分叉分类计数）、每臂轨迹与真实终局分、实际评分样本、三份预测 JSON（含 `7544`）、三份检查点 manifest/SHA 及 `SHA256SUMS`。全程 `test_authorization=false`。

## 4. 短训两组（`terminal` vs `ranking`）预登记验收口径

待 How 回传真实训练配对、两组 `--smoke` 与 12 步短训检查点及独立重载原件后，按以下同一标准验收：

1. **起点与预算一致性**：核对两组 `before_parameter_sha256`（须同为 stage126）、输入配对文件 SHA、顺序、种子、12 次更新预算、独立重载 `--reference` 误差与 `test_authorization=false`。
2. **独立 dev 分叉步排序（核心指标）**：在同一批未查看 dev 任务的真实同页选品分叉上，**仅在环境终局严格有分差（`env_high > env_low`）的对上**比较 `ranking` 与 `terminal` 的打开商品步排序准确率与分差均值；**环境同分对（`env_high == env_low`）单独登记，绝不计入方向胜负**。
3. **总分基线对照**：在同一批未查看 dev 轨迹上分别汇报 `ranking`、`terminal` 及 stage126 相对 fit-only 常数、回合数均值与 OLS 基线的总分 MSE 与配对区间，检查加入排序损失是否牺牲总分校准。
4. **回归单列与边界**：`7544` 仅作为已查看回归项单独汇报，不并入独立 dev 统计；B1 既有 SFT 暴露继续单列保留；正式 PPO 接入视独立 dev 验收结果决定，预留 test 继续冻结（`test_authorization=false`）。

## 可复核入口

```bash
python scripts/freeze_prm57_independent_dev.py \
  --repo /local/repository \
  --goal-map /evidence/goal-instructions.json \
  --archive /evidence/prm55-expanded-fit-20261125.zip \
  --output-ids /new/prm57-independent-dev-ids.json \
  --output-audit /new/prm57-independent-dev-audit.json
```
