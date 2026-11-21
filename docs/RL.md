# 进度估计、奖励和 PPO 的具体边界

接入前先看 [PPO 接入清单](PPO_INTEGRATION_CHECKLIST.md) 和 [训练 rollout 字段设计](TRAINING_ROLLOUT_SPEC.md)。两份文件是待实现的设计与验收合同，不代表 PPO 已跑通。

## 论文思路

轨迹有 T 个步骤。估计每步贡献 `p_t`，使 `sum(p_t)` 接近最终奖励 `R`。策略训练使用 `alpha * p_t + beta * grounding_t`。仅有总分监督并不唯一确定每步贡献，因此低总分预测误差不等于因果正确归因。

原神经估计器在 `upstream/prm/`，PPO 在 `upstream/ppo/`。默认新工具提供一个无依赖的可加线性基线，帮助在有限算力下先检查数据、监督信号和奖励接口。

## 低成本线性基线：已实现

对当前观测与模型回复做哈希词频特征 `phi_t`，使用线性权重输出 `p_t = w · phi_t`。以轨迹特征之和预测 `R`，优化平方误差并以独立验证集选择 epoch。特征不包含最终奖励、目标商品答案或未来步骤。模型在观测上可能学到表面线索，不能据此宣称理解了真实进展。

```bash
webshop-lab progress-train --train artifacts/explore-train-small.jsonl --val artifacts/explore-val-small.jsonl --output artifacts/progress-linear.json
webshop-lab progress-predict --model artifacts/progress-linear.json --input artifacts/explore-train-small.jsonl --output artifacts/explore-train-annotated.jsonl
```

输出 JSON 模型：权重、维度、最佳 epoch、val MSE、输入 SHA、train/val ID、goal fingerprint。预测将 `progress` 加入每步。训练/验证任务或底层指令重叠时拒绝拟合；训练奖励完全没有变化也拒绝拟合。

注意：annotate 训练轨迹的结果是训练集预测，不应拿其误差当泛化表现。要报告预测误差，使用独立 val 轨迹；如需更严格的奖励标注，考虑按任务交叉拟合，不能把自己训练过的轨迹效果当成 out-of-sample 证据。

## 奖励导出：已实现

```bash
webshop-lab reward-export --input artifacts/explore-train-annotated.jsonl --output artifacts/rewards-spa.json --mode spa
webshop-lab reward-export --input artifacts/explore-train-small.jsonl --output artifacts/rewards-terminal.json --mode terminal
webshop-lab reward-export --input artifacts/explore-train-small.jsonl --output artifacts/rewards-uniform.json --mode uniform
```

- `terminal`：前 T−1 步为 0，最后一步为实际 R。
- `uniform`：每步为 R/T，是简单奖励分配对照。
- `spa`：每步为 alpha × progress + beta × grounding，默认 alpha=1、beta=0.5；这里指接口公式，估计器类型必须另报。

SPA 模式缺少任意一步预测即报错；不允许把全 0 或最终奖励偷偷当作进度模型预测。grounding 是 0/1，而 beta 控制其权重。加入 grounding 后奖励和不一定等于最终 R，这是融合奖励的设计结果。

导出字段：`task_id`、`repeat`、`instruction`、`observations`、`responses`、`rewards`、`progress_sum`、`final_reward`、`goal_fingerprint`。

## 原神经进度模型：代码保留，目标机未验证

| 文件 | 作用 |
|---|---|
| `upstream/prm/data_org.py` | 将旧格式探索输出转换为进度模型输入 |
| `train_our_progress_model.py` | 全参数/原始版本参考 |
| `train_our_progress_model_lora.py` | LoRA 版本参考 |
| `train_our_progress_model_lora_fp16.py` | 带 fp16 调整的本地版本 |
| `inference_prm.py` | 预测各回合贡献 |
| `rl_data_org.py` | 融合奖励并构造旧 PPO 输入 |

这些脚本不是新 CLI 格式的直接消费者。原始探索 JSON 字段是 `agent_conversations` 等，新工具是逐步 JSONL。移植时必须显式转换，并测试 turn 数、assistant mask、终止奖励、ID 和奖励长度是否一致。

原脚本会使用相同/包含关系的 train/val 数据，不能照运行后直接报告验证结果。必须按本项目划分收集独立 train/val 轨迹，替换路径并检查 token 截断对每步分数的影响。

## 原 PPO：可用参考源码，不是已完成的新主线

入口：`upstream/ppo/train_ppo.sh` → `step_ppo.py` → `step_ppotrainer.py`；配置 `upstream/config/StepTool_ppo.json`；合并 `upstream/ppo/merge.py`。

原包提供 `upstream/ppo/requirements.txt`，指定 `torch==2.2.0`、`transformers==4.43.1`、`trl==0.10.1`、`peft==0.5.0`；根 requirements 又指定 `trl==0.7.7`，两个环境不能混装。这些旧版 torch 配置不能直接用于 5060 Ti，PPO 接口也不能无条件替换成最新版 TRL。现代 GPU 环境与旧 PPO 需要单独适配，不能用原 requirements 覆盖已经验证的 SFT 环境。

原 PPO 从 JSON 加载固定 prompt/response/reward，再对这些响应更新；这不自动保证它们来自当前待更新策略。正式实验要检查采样策略、保存的行为概率、旧策略概率和 PPO 多轮更新之间的一致性。仅把离线数据送入名为 PPO 的类不足以证明是正确的 on-policy PPO。

同时检查：

1. tokenizer 与实际模型一致；原 `step_ppo.py` tokenizer 路径硬编码为 Llama。
2. 上下文截断后，fragment mask 与逐步奖励仍一一对应；原默认上下文 512 很容易截断长轨迹。
3. 相同初始化、采样预算与更新步数下比较 terminal / stepwise。
4. value head、reference policy、adapter、KL、优势估计与 EOS mask 均经过小样本检查。
5. batch=1、16 GB 显存能跑过一次完整 forward/backward/save/reload，记录真实峰值。
6. 训练后的真实 WebShop 成功率和行为变化在独立 val 上评估。

**当前仓库不提供“现代环境下一键完整 PPO 成功”的虚假承诺。** Zqqq-qq 需要完成上述集成验收后，才把 R0/R1 状态改成可运行；相关任务在团队计划中有明确截止和备选路线。

## 最新学习与验收安排

以 [现行逐日计划](DAILY_PLAN.md) 为准：11/21 整理差距，11/22 选择奖励路线/预算并验收核心多轮链，11/23 两臂训练及测试冻结检查，11/24 完成必要训练/统一评估和原件归档，11/25 项目签收及展示冻结，11/26 pre。原 11/25 首试、11/28 缓冲、12/01 验收与 12 月主实验安排已停用。

#23–#28 是待按具体提交核验的工程候选，单次 smoke 不等于完整研究验收。两臂必须共享初始化/算法和辅助奖励设置；当前工程候选 beta=0，不自动等于采用非零 grounding 的正式对照。#25/#26 学习进度模型的验证问题必须披露，#27 规则型规格势函数不能冒充学习进度或神经 SPA。选择替代路线时同步 proposal 和实验命名。

主路线、预算、种子与 test 放行规则由 Zqqq/Qiu 在看 test 前冻结。门槛未过就如实记录未完成，不因压缩日程跳过 on-policy、来源、恢复或独立评估检查。若转用简单 on-policy 策略梯度，明确改算法名，不称为 PPO。
