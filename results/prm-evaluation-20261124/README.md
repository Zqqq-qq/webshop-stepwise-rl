# PRM 原件、保留集与 CPU 评分头对照（2026-11-24）

QiuQiu-xuan 提交了首轮 CPU 算术、源码和文件身份核验，以及真实 tokenizer 复编码。后续由 Tutor 完成补包复算和 CPU 评分头对照。[Issue #48 分工](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/48#issuecomment-7129106643)

## 补包复算与 CPU 对照

复算核对主包 85 项和补包 10 项载荷的 SHA256。补包 `F.npy` 是重新提取的 **8×151936** 特征；按归档评分位置数追加回合数列后得到 **8×151937**，数值秩 **8**、条件数 **205.17497**、最大残差 **5.13e-14**，FP32 聚合误差 **7.75e-7**。秩、奇异值和最小 L2 解范数与报告一致。设备的特征提取与逐步 logits 重建仍属于单独的验证范围。

补传的原 `step-512/training-state.pt` 与主包原检查点 manifest 的 SHA256 一致。114 份 AdamW 参数状态均记录第 **512** 步，动量张量均为有限 FP32。实际 **lr=1e-7、weight decay=0.01**，CPU RNG 和一份 CUDA RNG 在包中。此处验证了参数形状集合，参数名称映射和 CUDA 续训仍需设备验证。原 8 样本模型训练的最终 MSE **0.138531**，最大绝对误差 **0.648720**，未达到原记忆门槛。

[补包核验报告](supplement-review.json) · [补包发布说明](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/48#issuecomment-7128613974)

同一批固定 B1 特征上，仅更新评分头。各组使用 FP32、相同零初始化、AdamW（weight decay=0.01）、clip=1。每 64 条样本处理后评估原 8 条样本。下表对照共同终点 **512 条样本处理量**，历史 GPU 训练另列在上文。

| 学习率 | 每次用几条样本 | 优化器更新 | 最终 MSE ↓ | 最大绝对误差 ↓ |
|---|---:|---:|---:|---:|
| 1e-7 | 1 | 512 | 0.139987 | 0.648712 |
| 3e-7 | 1 | 512 | 0.181568 | 0.610330 |
| 1e-6 | 1 | 512 | 0.368613 | 0.918559 |
| 1e-7 | 8 | 64 | 0.123179 | 0.547732 |

四组均未达到 MSE < 0.001 且每条绝对误差 < 0.1 的门槛，所有更新均触发裁剪。逐条对照中，`3e-7` 在 448 步曾达到 MSE 0.099351，512 步回升至 0.181568。整批更新在相同样本处理量下改善了 `1e-7` 的终点误差，两组实际优化器更新次数不同。对照支持继续排查评分头的优化表现，尚未选出达标的 GPU 候选。固定表示拟合已复算，梯度记忆仍未达标，泛化和逐步归因待专门实验。

[每组预测、检查点指标和停止原因](cpu-head-comparison.json) · [下载和复跑命令](../../docs/PRM_EVALUATION_AUDIT.md#复跑已发布的补包和-cpu-对照)

## 首轮 8 样本原件核验

[How发布的新包](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm-evidence-20261124.zip)，SHA `b711fc49d9934adc321236a8837932406b6d1f3c8b81f8952d1f2c4a382453c8`。94个ZIP文件，顶层清单列出的85个payload逐一核SHA通过；另外9个是顶层/各检查点的校验清单本身。

完整机器结果：[memory-review.json](memory-review.json)。

- 原8条与前一归档64-fit的选中记录逐字段一致，全部属于冻结train。样本SHA `9c88beff…`与计划一致；标签顺序一致。
- 固定真实Qwen tokenizer四文件SHA与checkpoint一致。独立重编码8/8成功，长度902–1052、回合数3或10；所有8个检查点的每个评分位均与重新编码的assistant im_end一致。
- 合并各summary后更新记录恰为1–512，无缺失/重复。每64步的8条预测、MSE、最大误差均独立复算一致。
- 512步MSE **0.1385312484**、最大绝对误差 **0.6487203240**，未过MSE<0.001且每条绝对误差<0.1。8个检查点全部未过。最低已保存MSE在256步，约0.13330；这条曲线不能解释为已证明只需更多预算。
- 原件中的reference与另存独立重载预测逐步完全相同，差0；这是核对既有保存记录，本机未独立运行模型重载。
- 8个checkpoint的LoRA safetensors头均为F32；head.pt 指令检查确认 FloatStorage。补包进一步确认了第 512 步的优化器状态。
- 核心5文件SHA与旧包中a67db6a源码一致。首轮主包缺少各检查点的 training-state.pt，补包现已提供并核验 512 步文件。

## 首轮表示报告及来源边界

首轮核验确认报告的任务、标签、token 和回合数与原 8 条一致。当时仅能检查报告内部算术，真实聚合特征的独立复算由上方补包报告补齐。

归档提取脚本加载合并 B1 的 base，在 eval/inference_mode 下取原回合末位 raw logits，以 FP32 求和后转 float64。求解时追加回合数作为偏置系数。特征来自固定 B1，原模型梯度训练还更新了 PRM LoRA。补包提供了重提配置和提取脚本，首次未保存的数组和提取时实际源码绑定仍未取得。

报告中的 `max_abs_step_budget_512_lr1e7` 不是AdamW通用位移上限；`ratio_weight_norm_to_step_budget` 用的是L2范数/单坐标步长量，更不能直接作为必需更新数。最小L2解最大坐标也不是所有可拟合解的必要下界。因此不签收“至少860步”或“1500步失败即可排除预算”的推断。

通用数值复算入口和输入契约见 [CPU 复算入口](../../scripts/audit_prm_representation.py) 和 [原件格式与判据](../../docs/PRM_EVALUATION_AUDIT.md)。

## #45 保留集：可验收的范围已完成

固定head `0e1a15e46d2419e4451e6863934a6eafdd1ad3f7`，31项原数据/CLI回归独立执行通过。另用真实tokenizer对原911条逐条重编码，复现旧manifest的501 fit/127 dev；超长分别228/55，没有静默截断。实际新保留集经retained_episodes/write_training_jsonl导出后，编码与训练文件键序、标签、步骤数、来源SHA与行为策略声明一致；完整CLI保留集控制流由上述合成回归覆盖。没有将真实重编码说成完整模型跨PR验收。

完整机器结果：[data-review.json](data-review.json)。

| 项 | fit | dev |
|---|---:|---:|
| 过滤前 | 729 | 182 |
| 保留 | 501 | 127 |
| 超长拒绝 | 228 | 55 |
| 满奖励/部分/零 | 154/292/55 | 36/77/14 |

任务/指令fit-dev交叉0，val/test指令交叉0；train/val/test文件与goal-map SHA均和原manifest一致。仅实际911/1451任务，整体保留628/911。行为策略SHA是轨迹来源声明与文件身份核对，不是从文本证明采样实际权重。新代码目标GPU与模型重载仍不在本次签收范围。

## 复核命令与环境

首轮复核用 Python 3.12.10、Transformers 4.56.2、Tokenizers 0.22.2、Jinja2 3.1.6 和 NumPy 2.5.4。补包复算与 CPU 对照使用 PyTorch 2.14.1 和 NumPy 2.5.4。具体版本见机器报告。

```bash
python scripts/audit_prm_memory_evidence.py \
  --bundle /evidence/prm-evidence-20261124.zip \
  --previous-bundle /evidence/prm-diagnostic-evidence-20261124.zip \
  --tokenizer /local/pinned-qwen-tokenizer \
  --train-ids results/env-2026-11-17/train_ids.json \
  --output /new/memory-review.json
python -m unittest discover -s tests -v
python -m compileall -q src scripts
python scripts/check_repository.py
webshop-lab audit --output artifacts/prm-evaluation-source-audit.json
```
