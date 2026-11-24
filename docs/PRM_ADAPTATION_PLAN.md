# Qwen 适配计划（神经 PRM）

依 Zqqq 11/23 12:24 与 12:35 的要求。范围：只做接线，不改模型结构与 loss；`upstream/` 原样不动，新入口放本项目代码。

## 非目标

不改 `upstream/`；不改核心方法（神经 backbone + PRM LoRA + 每回合末位词表 logits → 标量评分头，`L=(Σp_t − R)²`，R 仅作标签）；首版不做成功分类、hidden-state 头、规则蒸馏；不覆盖 B1 与策略 adapter；不使用 test。

## 新增文件（`src/webshop_lab/prm/` 下）

| 文件 | 职责 |
|---|---|
| `config.py` | 显式记录模型路径/revision、tokenizer 身份、初始化来源、dtype/量化、LoRA 模块与超参、context、输出目录 |
| `tokenize.py` | 用真实 Qwen tokenizer/chat template 构造序列；输出 `assistant_spans`、`turn_end_indices`；attention_mask 与 turn mask 分离；EOS/角色分隔归属明确 |
| `data.py` | 现有 rollout JSONL → PRM 样本；R 语义校验（购买完成取任务分，未购买=0，页面匹配分不得当 R）；按 task_id + 归一化指令做 fit/dev 隔离；超长样本显式计数/拒绝，不静默截断 |
| `model.py` | 加载锁定 Qwen；可选冻结/合并 B1 adapter 作为初始化；挂 PRM 专用 LoRA；评分头输入维度取实际 `model.config.vocab_size` 并与 logits 末维核对，禁止写死 32000 |
| `train.py` | 每样本先求和再算 MSE；padding 步骤不参与；保存 checkpoint 与训练元数据 |
| `infer.py` | 独立进程加载、对单条轨迹逐步输出 p_t |
| `tests/test_prm_*.py` | CPU 边界、形状、隔离、因果无泄漏测试 |

## 分阶段

**阶段 0 · 数据与切分（CPU）**
PRM fit/dev **只能取冻结 `train_ids` 池**，不能用 160 条 val 开发评估轨迹（那是开发诊断，且多策略同一批 160 任务不等于 960 个独立任务）。

- 现有 train 池轨迹：`#42` 全量覆盖跑至今约 22 次更新×4 任务≈88 条，加上 seed42/43 两臂 PPO 训练 64 条，合计约 150 条；但它们来自逐步演化的策略，**不是纯 B1 采样**，复用需显式 schema 适配与身份/任务校验。
- 因此需要新采 B1 探索轨迹：建议 B1 策略温度=1 在 train 任务上采 200 条（约 50 分钟 GPU），按 task_id + 归一化指令切 fit 160 / dev 40，并报告奖励分布。
- 产出：清单 + SHA + 分布统计；不碰 `test_ids`。

**阶段 1 · 模板与回合边界（CPU）**
Qwen chat template 构造 system/user/assistant 序列；确认 assistant 末位与 turn_end_index；测多轮连续、中文、特殊字符、EOS/pad 相同、空动作、超长轨迹；人工解码真实样本逐字核对评分位。

**阶段 2 · 模型与评分头（CPU toy）**
init = 锁定 Qwen 快照（必要时合并并冻结 B1 adapter 二进制，保留其身份）；PRM LoRA 独立；首版完整logits参考路径在每回合末位取logits再进`V→1`头；完整返回[B,L,V]后切片不降低生成该张量的峰值。等价优化须在原LM head之前先选末位hidden states，再经原LM head及V→1头，核对输出与梯度；不替换为hidden-state到1的新结构。CPU测试不证明真实16GB容量。

**阶段 3 · GPU 真实验收**
forward/backward、梯度非零、参数变化、保存、独立进程重载、逐步预测；记录峰值显存与耗时；量化与 gradient checkpointing 是否够 16GB 需实测，不预先承诺。

**阶段 4 · 正式 PRM 训练**
fit/dev 训练与选择；报总分 MSE vs 常数/长度基线、动作分诊断、规格分叉与循环探针。通过后才进入 TG vs learned-progress 的 PPO 对照（两臂各自从 B1 新采样、同预算）。

## GPU 排期

按 Zqqq 指示：等 #42 当前批次完成（update + save + audit）后停止启动后续批次，记录已完成批次/最后父 run/模型与 optimizer/RNG/日志/剩余任务清单，移交 GPU 给 PRM。#42 无自动恢复入口，续跑另记方案。

历史执行方反馈：22/1452更新（12:42Z）、GPU 99%；这是当时快照，当前交接须补最后完整run及停派发后的实际状态，不能预写已腾卡。

## 交付

一个 draft PR，内含配置、最小训练/推理入口、数据方案与 CPU 测试；Jun 在同一 PR 复核模型结构、模板对齐、初始化身份与保存重载契约。

## Jun模型与重载交接

已接入#44固定版本的配置/tokenizer，新增完整Qwen/B1初始化、短训、独立进程推理与续训入口，见[模型交接](PRM_MODEL_RELOAD_CONTRACT.md)和[运行说明](PRM_RUNTIME_HANDOFF.md)。24项CPU测试实际通过，含真实tokenizer及小型真实Qwen/PEFT训练/重载，不代表项目1.5B/GPU验收。真实数据转换与采集、fit/dev核验、GPU容量及正式PRM泛化仍待完成。执行遵守固定FP32与依赖；head零初始化首步LoRA梯度为0属预期，须检查后续变化。不改upstream，不覆盖原B1。
