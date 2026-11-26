# #59 神经进度模型（Arm 4 vs stage126）新冻结 dev 清单与验收协议

2026-11-26，按 [#59 复验分工](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/59#issuecomment-7189640161)，在查看任何新 dev 预测之前，由 Qiu 侧独立冻结全新未查看 dev 清单、去重审计与预登记验收表。**11/27 pre 仍使用已有可追溯证据，#59 标注为待新 dev 复验；预留 `test200` 继续冻结（`test_authorization=false`）。**

## 1. 冻结任务清单与 SHA-256

- **64 题总清单**：[prm59-reverify-dev-ids.json](prm59-reverify-dev-ids.json) — SHA-256 `a033b312440952d6325734b4b9bebc9f99ecc1db9b865939a31cc3e54e57f2e8`
- **主评 32 题（优先执行）**：[prm59-reverify-dev-primary_32-ids.json](prm59-reverify-dev-primary_32-ids.json) — SHA-256 `a04c338502a50db274af6eedfc6f98d54eac86834ef390ddd6e4aca4b5030721`
- **备选 32 题（盲态按序补齐）**：[prm59-reverify-dev-reserve_32-ids.json](prm59-reverify-dev-reserve_32-ids.json) — SHA-256 `3ad6b162301b3b6a45e1c1d2f63ed928f19d9d698bd0da1bb039f4a81d442cd4`
- **完整去重与隔离审计 JSON**：[prm59-reverify-dev-audit.json](prm59-reverify-dev-audit.json)
- **可复现脚本**：[freeze_prm59_reverify_dev.py](../../scripts/freeze_prm59_reverify_dev.py)（含单元测试）

## 2. 去重与隔离结果（全项 0 重叠）

沿用 [#55 `a601999`](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/55) 已核验的 444 题洁净候选池（`SHA256(ASCII("42:" + ID))` 升序），逐级扣除后严格隔离：

| 过滤阶段 | 数量 | 说明 |
| --- | ---: | --- |
| #55 `a601999` 洁净池 | 444 | 已排除 `used(911)`、`6426`、设备暴露 85 题、`6755` 及 `val/test` |
| 扣除 #55 已查看 `dev160` | −160 | `holdout_ids.json`（SHA `15b96f01…`） |
| 扣除 #57 已查看 `dev64` | −64 | 逐字节核对 `prm57-independent-dev-evaluation-20261126.zip:frozen/independent-dev-ids.json`（SHA `4e0e86ab…`） |
| 扣除全库 12,087 题跨 ID 同指令孪生 | −3 | `7610`（↔`6045`）、`9085`（↔`9381`）、`11501`（↔`11798`） |
| 扣除与训练/已看 dev/val/test 去价格同指令冲突 | −3 | `8685`、`7330`、`10009` |
| **剩余严格未查看洁净池** | **214** | 内部归一化与去价格指令均唯一 |
| **本次冻结（前 64 题：`primary_32` + `reserve_32`）** | **64** | ID / 归一化指令 / 去价格指令与 `fit501`、#57 `train32`、#55 `dev160`、#57 `dev64`、`val160`、`test200` 交集全为 **0**；全库 12,087 题孪生 **0 例** |

> 披露：B1 既有 SFT 暴露仍独立单列披露，不笼统宣称“策略从未见过”；`7544` 单列为已查看回归案例，不并入新 dev。

## 3. 采集、盲态启备与异常处理登记规则

1. **盲态启用备选 32 题规则（查看任何模型预测前决定）**：
   - 按 `primary_32` 冻结顺序采集同页选品分叉（与 #57 相同规则：B1 自然轨迹 + 同搜索页结果末尾备选 ASIN 实买）。
   - 在运行任何 PRM 推理前统计 `primary_32` 中**环境终局严格有分差（`env_high > env_low`）且两臂均 `≤2048` token 可评分**的有效对数 `N_valid`。若 `N_valid < 20`，按 `reserve_32` 冻结顺序整批（或补足至预登记目标 32 对）补采，完成后锁定共同原始轨迹输入，再统一对 `stage126` 与 `#59 Arm 4` 做只读推理（各模型按自身评分模板构造输入；Arm 4 的标题与价格须经核实仅取自点击前搜索页 `messages[i-1]`）。
2. **计数守恒与异常分类登记（总尝试数严格守恒）**：
   $$\text{attempted} = N_{\text{valid\_contrast}} + N_{\text{environment\_tie}} + N_{\text{too\_long}} + N_{\text{no\_replayable\_fork}} + N_{\text{technical\_error}}$$
   - **`environment_tie`（环境同分）**：两臂 `env_high == env_low`；**单独登记原始分差，绝不计入方向排序胜负**，可按任务纳入总分 MSE（并同步报告仅有效对与全保留轨迹口径）。
   - **`too_long`（超长）**：任一臂完整 PRM 输入超过 `2048` token（Arm 4 回退原文仍超 `2048` 时亦记 `too_long`）；整对剔除不补样，确保 `stage126` 与 `Arm 4` 在**完全相同的有效任务集合**上对比。
   - **`no_replayable_fork`（无可重放分叉）**：自然轨迹未出现“搜索 → 打开商品”或结果页不足 2 个不同 ASIN；单列登记，自然轨迹若 `≤2048` 仍可保留用于整条总分 MSE 评估（单列分母）。
   - **`technical_error`（技术异常）**：环境/桥接报错；保留失败原件，不混作零分负例。
3. **点击前可见证据与环境终局分差严格分开记录**：
   - 在揭示模型预测分之前，对全部 `valid_contrast` 对仅凭点击前搜索页（标题 + 价格）标注点击前可见性：`V+`（点击前有明确证据支持环境高分臂）、`U`（点击前证据不足以区分，如 P02/P03）、`V-`（点击前表面信息与环境隐藏判定相反）。
   - 严禁把“环境终局分差 > 0”等同于“点击前证据可见”，两列在验收表中始终分开汇报。

## 4. 预登记验收表（待 How 回传原件后逐项填入）

### 4.1 覆盖与去重计数表

| 统计项 | 数值（待实测） | 核验要求 |
| --- | --- | --- |
| 冻结尝试任务数（32 或 64） | TBD | 与冻结清单 SHA 逐字节一致 |
| 有效环境分差对 `N_valid_contrast`（`env_high > env_low` 且 `≤2048`） | TBD | `stage126` 与 `Arm 4` 分母完全一致 |
| 环境同分对 `N_environment_tie`（`env_high == env_low`） | TBD | 单列，不计方向胜负 |
| 超长剔除 `N_too_long`（`>2048`） | TBD | 两检查点共同有效集合一致 |
| 无可重放分叉 `N_no_replayable_fork` | TBD | 附逐题原因 |
| 技术异常 `N_technical_error` | TBD | 应为 0（否则保留原件隔离） |
| 点击前可见性拆分（`V+` / `U` / `V-`） | TBD | 在看模型分前锁定，与环境分差分列 |

### 4.2 主指标对比表（同原始轨迹、同有效分母）

| 检查点 | 全部有效对选品步正确率 | `V+`（点击前可见）正确率 | `U`（点击前难判）正确率 / 平均分差 | 选品步平均分差（高−低） | 整条轨迹总分 MSE ↓ | 相对 fit-only 三基线（常数 / 回合数 / OLS）配对 95% CI | `7544` 单列回归（跑鞋−洞洞鞋） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `stage126` | TBD | TBD | TBD | TBD | TBD | TBD | TBD（基线约 `−0.0034`） |
| `#59 Arm 4` | TBD | TBD | TBD | TBD | TBD | TBD | TBD（单列，不并入新 dev） |
| `Arm 4 − stage126` 配对差 | TBD | TBD | TBD | TBD（附配对 95% CI） | TBD（附配对 95% CI） | — | 单列 |

- **进入 PPO 冒烟的门槛**：新冻结 dev 上，**选品步排序**（`Arm 4` 在全部有效对及 `V+` 子集上相对 `stage126` 稳定提升且分差为正）与**整条轨迹总分校准**（MSE 不劣于 `stage126` 且对照三条 fit-only 基线）两项均需过关并经独立复算；未过关则按负结果归档，`test200` 保持冻结。

## 可复核入口

```bash
python scripts/freeze_prm59_reverify_dev.py \
  --repo /local/repository \
  --goal-map /evidence/goal-instructions.json \
  --pr55-archive /evidence/prm55-expanded-fit-20261125.zip \
  --pr57-dev-zip /evidence/prm57-independent-dev-evaluation-20261126.zip \
  --out-dir /new/prm59-dev-freeze
```
