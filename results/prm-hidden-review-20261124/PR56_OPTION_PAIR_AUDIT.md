# #56 任务 7084 同页选色诊断：归档基线与独立前缀/配对核验入口

2026-11-26，针对 [#55 反馈](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/55#issuecomment-7167924678) 与 [#56 固定提交 `faf8b319`](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/56) 完成两项前置独立核验：**（1）归档原件与任务 7084 基线核实**；**（2）重放/评分脚本审阅与独立前缀/配对复核工具落地**。机器记录：[pr55-option-pair-baseline-7084.json](pr55-option-pair-baseline-7084.json)。本机未运行 WebShop bridge 重放、GPU 推理或预留 test（`test_authorization=false`）。

## 1. #55 归档包与任务 7084 / 错购参照独立核实

- [`prm55-expanded-fit-20261125.zip`](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm55-expanded-fit-20261125.zip)（25,530,124 字节，SHA-256 `0834998fe986f450f40721f0864b51351c68743d5b2e5979334cffde28ca44b4`）包内 `SHA256SUMS` **61/61 全部逐字节匹配**。
- **任务 `7084` 归档基线**（4 步满分，`final_reward=1.0`）：
  - 指令：`Instruction: find me some highly pigmented eye shadow in color b, and price lower than 30.00 dollars`。
  - 动作链：`search[highly pigmented eye shadow in color b under \$30.00]` → `click[b09g6n33k9]` → `click[b]`（同页 `clickables` 含 `a, b, c, d`）→ `click[buy now]`。
  - 第 126 步归档评分：`turn_end_indices = [179, 914, 1081, 1246]`，`step_scores = [0.284884, 0.266004, 0.080918, 0.053752]`，总分 `0.685559`（与逐步之和误差 `<1e-6`）。
  - 冻结第 126 步检查点身份：`source_commit = 559a7791b49332375c640bc132af05e079676a83`，`head.pt` SHA `c2168784…`，`adapter_model.safetensors` SHA `abcdd508…`，`adapter_config.json` SHA `22740a3a…`。
- **错购对照与后备任务独立复算**（与 #55 评论逐项一致）：
  - 3 条零分购买打开商品步得分 / 总分：`7544`（`+0.265678` / `0.700135`）、`8669`（`+0.263719` / `0.674103`）、`9984`（`+0.264477` / `0.714821`）。
  - 23 条三步满分轨迹打开商品步均值：`+0.264142`。
  - 后备任务 `7402`（4 步满分，`click[dusty pink]`）：`turn_end_indices = [188, 848, 1059, 1298]`，总分 `0.688078`。

## 2. #56 重放/评分脚本审阅与独立复核补强

审阅 `faf8b319` 中的 `scripts/replay_prm55_option_pair.py`，其核心流程（归档 ZIP SHA 核验、本地 bridge `/health` 与前两步观测校验、`head.pt`/adapter SHA 校验、`turn_end_indices[:2]` 与 `±1e-5` 前缀容差、环境同分/反向标记 `valid_environment_contrast=false`、`test_authorization=false`）整体清晰严谨。为确保独立核验闭环，Qiu 侧新增独立只读核验器 [audit_prm55_option_pair.py](../../scripts/audit_prm55_option_pair.py)（含 6 项回归/反例测试），在 How 回传设备原件后额外强制核对三项：

1. **评分阶段复核归档 7084 原始前缀**：即使 B/A 两臂彼此一致，若前两步（含 observation/available/response/action/reward）或分叉前第 3 步页面与归档 7084 发生漂移，立即拒绝。
2. **分叉后受控措辞检查**：强制校验选色步 `response == "Action: click[b|a]"` 与购买步 `response == "Action: click[buy now]"`，防止额外 Thought 文本混入两臂。
3. **归档评分位点与 `environment.json` 交叉核验**：强制核对两臂 `turn_end_indices[:2]` 与归档 `[179, 914]` 一致、记录两臂共享前缀相对归档 7084 的数值漂移，并复核 `environment.json` 健康状态、终局分与 `pair-comparison.json` 全部差值字段。

## 3. 当前状态与待回传核验命令

目前 How 尚未上传任务 7084 的设备重放与第 126 步评分原件，因此实测前缀一致性与 B−A 分数配对结果为 **待设备原件回传**（`live_pair_replay_verified=false`、`test_authorization=false`）。回传后直接运行下述命令完成最终独立核验：

```bash
python scripts/audit_prm55_option_pair.py --mode pair \
  --archive /evidence/prm55-expanded-fit-20261125.zip \
  --run-dir /evidence/prm55-option-pair-7084 \
  --output /new/pr56-option-pair-7084-audit.json
```
