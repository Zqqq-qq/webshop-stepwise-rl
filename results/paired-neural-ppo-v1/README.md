# TG 与 Arm 4 配对 PPO 训练和验证

## 冻结批次

[manifest.json](manifest.json) 记录每批任务顺序、文件 SHA-256、抽样规则、排除集来源以及隔离结果。训练代码在 `01def67412aecd0cc6571a84b4b1d974aece0b40`；本目录是独立的数据交付，执行代码与批次文件应分别按提交固定。两臂均从同一个冻结 B1 初始化策略和 PPO 价值头。Arm 4 的检查点、语义契约与运行时沿用 [组合协议](../../docs/NEURAL_PPO_COMPARISON.md)。

| Seed | Update 1 | Update 2 | Update 3 | Update 4 |
| --- | --- | --- | --- | --- |
| 42 | [158, 2441, 9807, 7616](seed-42/update-1-ids.json) | [1328, 1635, 11034, 7887](seed-42/update-2-ids.json) | [103, 7750, 6066, 6349](seed-42/update-3-ids.json) | [3304, 5080, 6091, 10050](seed-42/update-4-ids.json) |
| 43 | [5874, 1710, 9284, 8028](seed-43/update-1-ids.json) | [5439, 1017, 1854, 1457](seed-43/update-2-ids.json) | [183, 1717, 1748, 6954](seed-43/update-3-ids.json) | [7849, 6589, 7128, 323](seed-43/update-4-ids.json) |

从原 `mapped_train` 1,451 题扣除清单中的已用训练、开发集、四题短训、冒烟和回归任务以及 `val160`/`test200`，剩 657 题。用冻结 goal map 校验全库任务文本；剔除精确归一化指令孪生项及与排除集的指令碰撞后剩 620 题。按 `SHA256(ASCII("pr61-formal-ppo-v1:<seed>:<id>"))` 升序取题，先 42 后 43，按四题切批。指令归一化使用小写、合并空格、去首部 `instruction:` 和尾句点。去预算检查额外删除末尾 `and price lower than … dollars`。两个 seed 之间和已列排除集的任务 ID、归一化指令与去预算指令均零交集。B1 既有 SFT 暴露沿用原登记口径。

## 执行顺序和预算

每个 seed 跑终局奖励 `terminal` 和冻结 Arm 4 `neural_progress` 两臂，均按该 seed 的四个批次顺序运行。每批用 `ppo-smoke --ids` 指定一份清单，做一次更新。第一批从 B1 启动，第二至第四批用 `--resume` 接续同一 seed、同一臂上一批的输出。两臂配置保留相同底模、SFT adapter、种子、采样和优化参数。两臂各自在线采样，实际步数和 token 数可以不同，逐批登记。

沿用四题短训的训练采样预算。每题一次，最多 3 个策略回合，每回合最多 96 个新 token，context 2048，FP32，无量化，temperature 1、top-p 1、top-k 0、beta 0。每种子每臂 4 次更新、16 条轨迹，全实验 16 次更新、64 条轨迹。三回合上限可能使多步骤购物在训练中提前结束，报告时记录超时和未购买。冻结 Arm 4 权重，不在本实验中更新评分器。

## 验证与复算

- 两臂各用第 4 次更新后的策略在固定 `val160` 上评一次。B1 如作为参照，使用同一 `val160` 与解码设置。真实环境 FP32 贪心解码，temperature 0、context 2048、每回合最多 128 个新 token、每题最多 10 步、每题一次。留存逐题 JSONL 和原生成元数据。预留 `test200` 保持冻结。
- 主指标为满分购物成功率，沿用仓库的 `is_success`。辅指标为平均环境终局分、购买完成率、平均交互步数、按步计非法动作率。未完成购买的环境终局奖励按 0 计，页面分另存。
- 每个 seed 在完整的 160 个任务上逐题做 `Arm 4 − TG`，用仓库 `compare` 同时提供两臂 metadata 检查来源和成功率差。报告仅 TG 成功、仅 Arm 4 成功、两臂均成功、两臂均失败。成功率差与终局均分差均按任务配对 bootstrap 2,000 次，bootstrap seed 为 42。均分差用仓库 `terminal_reward` 逐题计算，并用 `paired_task_bootstrap` 生成区间。
- 两个训练 seed 分开报告，并给等权平均。计算两 seed 平均差的验证任务区间时，先对每个 task 取该 task 跨 seed 的平均差，再整题配对重采样；该区间只反映这两个已跑 seed 下的验证题不确定性。报告 seed 间方向差异，保留负结果。
- 训练和验证均单列**未购买或零分购买且累计神经奖励 ≥0.30** 的轨迹，分别标明实际 PPO 奖励和事后只读评分，使用 [奖励审计命令](../../docs/NEURAL_PPO_COMPARISON.md#scoring-and-audit)。同时报告两类失败各自分母及跳过原因；验证回算从原 rollout 元数据读取 128 的生成上限，不沿用训练的 96。

训练后交付四条训练链的各批次原件和配置哈希、各终点检查点、四份验证 JSONL 及 metadata、两个 seed 的逐题对照和奖励异常审计。正式结果须从这些原件复算。
