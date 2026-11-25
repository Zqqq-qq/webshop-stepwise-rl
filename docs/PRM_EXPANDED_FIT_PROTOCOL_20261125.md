# 神经进度模型扩大训练与独立 dev 协议

2026-11-25 冻结。本轮由 HowseaonLink 使用 RTX 5060 Ti 执行，QiuQiu-xuan 复核数据隔离、样本覆盖和配对指标。训练沿用 #54 的 hidden-state 评分及分组学习率配置。第 32 组的课程展示为 11/27，GPU 采集与训练安排在本组展示结束且新 dev 清单重冻结、设备预检通过后。PPO 与预留 test 另行验收，`test_authorization=false`。[#54 执行草案](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/54#issuecomment-7161569916) · [评估复核](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/54#issuecomment-7161673743) · [第 32 组展示日期更新](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/55#issuecomment-7163790426)

## 冻结数据与采集

- **训练输入。** 使用 #47 归档中的原 `fit1451.training.jsonl`，SHA-256 为 `03a6370bc2048ea0119331aa06969ae6bde18e4e7ba5db3decde108153792500`。固定 Qwen tokenizer 和 2048 上限整条过滤后应为 **501/729**，依原始顺序保留 **446 条正分购买、26 条购买零分、29 条步数耗尽零分**。保留行按原文件字节顺序拼接后的 SHA-256 为 `b1f1eeda3a762337d0ce99b9f98451a9fb0d444e6ed0a3cc5d302bd383e7cd71`。训练 CLI 的 `--train-ids` 仍用冻结的 1451 个策略训练任务清单，实际拟合范围由这份固定输入锁定。[历史输入归档](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/47) · [冻结 train 清单](../results/env-2026-11-17/train_ids.json)
- **新 dev 清单。** 1451 个计划任务中有 911 条完整旧 B1 轨迹，恰为原 fit729 与历史 dev182 的并集。其余 540 个候选与已取得的旧 500 步 dev127、64/28 诊断和 8 条记忆探针任务无交集。另排除第一条未完成任务 6426、与已登记 PRM/策略 val/test 指令重合的 6 个 ID，以及候选内部同指令的 3 个重复 ID（同组保留最小 ID）。共 **530 个候选**。指令身份按历史转换器小写、压缩空白、去开头的 `Instruction:` 再做 SHA-256，已与原 911 条逐条核对。[任务与指令排除记录](../results/prm-expanded-fit-20261125/holdout-audit.json)
- **固定抽样。** 对 530 个候选按 `SHA256(ASCII("42:" + 十进制任务 ID))` 升序排序，摘要相同按 ID 升序，取前 160 个，采集沿用文件顺序。清单 SHA-256 为 `0980e0da7728b644fa34ca07c64e308746fdc50707e8742c9aa219b6b5328db1`。它对本轮 PRM 拟合和方案选择隔离。B1 沿用既有 SFT 身份。启动前核对设备侧历史使用记录；若发现遗漏的历史 PRM 使用证据，先登记并在看新标签前按同法更新清单。[固定目标 ID](../results/prm-expanded-fit-20261125/holdout_ids.json)
- **采集。** 使用 `559a7791b49332375c640bc132af05e079676a83` 的 `webshop-lab rollout`，原 B1 adapter、固定 Qwen `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` 和原环境 fingerprint。采集复用历史 B1 的 4-bit NF4 路径、生成 seed 42、temperature 1、top-p 1、top-k 20（来自固定的 Qwen `generation_config.json`）、max_new_tokens 128、context 2048、max_steps 10、每题一次。基模、adapter、tokenizer 和生成配置在设备按文件 SHA 核验，记录精度与依赖版本。配置副本只按需调整本机路径与本机环境端口，并记录生效配置 SHA。[采集配置](../results/prm-expanded-fit-20261125/dev-rollout.json) · [B1 身份](../configs/identities/b1-pr18.json) · [固定生成配置](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/generation_config.json)
- **覆盖。** 首先核对 160 个任务各有一个完整 rollout，匹配采集顺序、run ID、配置、环境和锁定指令哈希。异常时保留失败原件，修复后用相同清单与 seed 在新目录重跑。正常步数耗尽作为零分；超过 2048 的完整 PRM 序列整条剔除且不补样。记录尝试、完成、保留各自的分母和分类计数。转换采用 #45 的 `load_episodes`、`gate_train_pool`、`write_training_jsonl`，所有目标作为独立 dev 输入，保持同一标签定义。收集新 dev 时固定训练预算和最终检查点，预测与评分等第 126 步产物锁定后进行。[完整性守卫](https://github.com/Zqqq-qq/webshop-stepwise-rl/blob/ba6ae4f16310db61955461b12cb77bc518088cfc/scripts/prm_checked_pipeline.py) · [转换入口](https://github.com/Zqqq-qq/webshop-stepwise-rl/blob/0e1a15e46d2419e4451e6863934a6eafdd1ad3f7/src/webshop_lab/prm/data.py)

## 训练与重载

- 在干净的 `559a7791b49332375c640bc132af05e079676a83` checkout，从原 B1 初始化新的 hidden-state `Linear(1536,1)` 评分头和 PRM LoRA。PRM 训练、重载和打分采用非量化 FP32、2048、seed 42，保留每回合评分求和及终局 MSE。
- 总预算固定 **126 次 AdamW 更新、1008 次轨迹呈现**。每次累积 8 条轨迹，取平均后裁剪至 1.0。score head LR 为 `1e-5`，LoRA LR 为 `2e-4`，余弦总周期 126，weight decay 0.01。固定顺序循环，原保留顺序的前 6 个 ID **0、9、11、13、28、31** 各呈现三次，其余各两次。
- 第一段 `--steps 63`，同 checkout 从第 63 步 `--resume` 再跑 `--steps 63`，两段均显式 `--cosine-total-steps 126` 并写各自的新目录。最终只用第 126 步进行新 dev 验收。第 63 步只保留诊断记录。OOM、非有限值、身份不一致或重载失败时保存日志与已完成产物。
- 独立进程用保留后的 fit501（字节与顺序按上文 SHA）和最终检查点自带的 `reference-predictions.json` 做 `--reference` 核对。原 729 条仍供长度与分项审计。两段记录实际组学习率、裁剪前范数、loss、参数摘要、显存、耗时，复核第 63→126 步状态衔接。[训练与续训参数](https://github.com/Zqqq-qq/webshop-stepwise-rl/blob/559a7791b49332375c640bc132af05e079676a83/docs/PRM_HIDDEN_ACCUMULATION_EXPERIMENT.md)

## 验收与报告

三条基线均只在保留的 fit501 上拟合，分别为常数、回合数分组均值、回合数 OLS。在新 dev 的同一批保留任务上报告 MSE、MAE、最大误差、Pearson、Spearman（常数相关为 NA）。按任务做 10,000 次配对 bootstrap（NumPy `default_rng(42)`，同一组采样索引）。观测差定义为「模型平方误差减基线平方误差」在原任务上的平均值，并另记重采样均值和 percentile 95% 区间。正分购买、错买零分、耗尽零分分别给样本数、覆盖及误差。

- **总分通过。** 模型的新 dev MSE 低于三条基线，且相对常数、OLS 的 95% 配对区间上界均小于 0。回合数分组均值的区间单列，只有其上界也小于 0 时才声称明确优于该对照。常数与 OLS 达线、分组均值仍更低或打平时登记为部分改善。
- **后续验证。** 原 8 条记忆与历史 dev127 保留作连续性诊断。逐步归因另用同页面中对/错规格的分叉验证，待总分和逐步评分分别验收再决定接入 PPO。本轮的新 dev 一旦被用于后续方案选择，登记为已查看的开发集。

交付采集与推理的完整命令行参数、运行配置和源码身份、任务/指令和输入 SHA、原始轨迹/转换/拒绝清单、两段检查点和独立重载证据、逐任务预测及复算表。原始轨迹和权重沿用共享 Release 交付。
