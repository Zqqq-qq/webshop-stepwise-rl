# PRM 跨PR加载与训练导出契约（2026-11-24）

本次协同修复 #43 与 #45 的三个集成阻塞，不修改 main、原 upstream、原 B1 或历史设备产物。#44 的配置、Qwen 模板、im_end 评分位与显式超长拒绝定义保持不变。#47 编排器的重载门禁另由该PR修复；本次没有在主设备启动训练。

## 唯一模型加载与身份

训练与推理都调用 `webshop_lab.prm.runtime.build_prm`，身份只由 `checkpoint_identity` 从已验证的本地输入、真实tokenizer模板与当前源码计算。绑定底模revision/文件SHA、原B1两文件、全部配置的tokenizer文件及辅助输入、LoRA/评分位/长度/精度与源码身份。先核 manifest/文件，再加载 adapter/head。禁止另一套 identity_from、手改manifest或用命令参数伪造训练源码提交。

`infer.py` 在两个PR中使用完全相同内容。#45 复用 #43 的 runtime、predictions、train 和 model_core，不能只取单个infer.py而缺少依赖。集成时须将两个固定head的变更合入同一个干净检出，核同路径文件一致，再在该检出完成训练→进程退出→推理。不同Git提交或不同源码的检查点仍严格拒绝：本修复不提供旧 a67db6a 或旧 #43 checkpoint 的自动迁移，也不通过删身份校验让旧记录看起来通过；旧证据应在它自己的原始bundle里复核。

## 统一命令与输入

```bash
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 配置、样本和IDs替换为已核对的实际路径，输出目录须全新。
python -m webshop_lab.prm.train --config /path/to/config.json \
  --samples /path/to/fit.training.jsonl --train-ids /path/to/train_ids.json \
  --steps 2 --device cuda

# 训练进程退出后，在相同检出/环境中完整重载。
python -m webshop_lab.prm.infer --config /path/to/config.json \
  --checkpoint /path/to/new-checkpoint --samples /path/to/fit.training.jsonl \
  --reference /path/to/new-checkpoint/reference-predictions.json \
  --output /path/to/new-reload.json --atol 1e-5 --rtol 1e-5 --device cuda

# 原始rollout仅做输入适配，使用同一完整加载与评分核心。
python -m webshop_lab.prm.infer --config /path/to/config.json \
  --checkpoint /path/to/new-checkpoint --rollout /path/to/rollout.jsonl \
  --task-id 1001 --repeat 0 --output /path/to/one-task-score.json --device cuda
```

`samples` 使用现有 task_id/messages schema；rollout 使用 task_id/instruction/steps（observation/response），可含 repeat。两种输入互斥；原始轨迹重复任务须选择唯一repeat，不能默取第一条。推理不读终局标签。两种输入都输出 predictions 列表，每项为 task_id、turn_end_indices、step_scores、total，并附同一 identity。旧 `turn_scores` 单任务输出改为该统一schema；下游须相应接入。`--source-commit` 只可断言实际源码提交，不能覆盖身份。

只有当 reference 含同一完整输入集合时才传入比较；如从完整rollout选择一个任务，须提供该任务独立reference，不拿全fit参考文件与单任务比较。空集/重复任务、评分位置/步骤数不一致、任意NaN/Inf、总分或逐步值超容差均拒绝；验收输入有超长剔除时也拒绝只比较剩余子集。无reference输出不是“重载一致通过”。

## 同一保留集导出

#45 仍由How原 data.py 完成转换/门禁，Jun在原实现上修复导出：`--training-jsonl`按 build_samples 返回的 `(task_id, repeat)` 保留键取回原轨迹，禁止重新导出超长/技术中断等剔除项。training schema新增保留 repeat；manifest.training_files记录每侧文件SHA、数量及sample_keys，与编码文件逐项一致，原 rejections/dropped 保留。拒绝覆盖输入、已有导出和dataset保留文件路径。

这些导出SHA只证明生成文件的身份，不能替代采样策略实际身份、完整数据来源和fit/dev隔离审计。

## 已验证范围与来源

- How原 #45（42486afb）data实现及原始轨迹→messages定义保留；#43 为Jun的完整Qwen/B1/PRM加载核心；#44固定配置/tokenizer来源保留。
- 合成隔离CPU检出61项测试全部实际执行，无skip。使用固定真实Qwen tokenizer，小型随机Qwen2/PEFT结构（非项目1.5B）。Torch 2.15.0、Transformers 4.56.2、PEFT 0.17.1。
- 新增跨模块测试真实训练3步→退出→独立进程samples重载→独立进程无标签rollout重载，两者逐步/总分对照一致，并核B1/tokenizer篡改及head破坏拒绝、续训累计4步。
- 新增真实data.main控制流测试：合法+超长、合法+技术中断两例均完成转换；编码与训练文件ID/数量一致、剔除项不回流，manifest SHA可复算。tokenizer在这两项CLI控制流测试中用显式替身，不冒充真实tokenizer测试。
- 共用比较器有空/缺失/多余/重复任务、NaN/Inf、短/长步骤、位置/总分不一致与非法容差的拒绝测试。
- 普通无模型依赖CI应明确skip模型集成测试；不能把绿色CI当1.5B/CUDA、效果、历史重载或正式泛化验收。

测试命令：在相同合成源码树、已缓存固定tokenizer的依赖环境设置 `PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，运行 `python -m unittest discover -s tests -v`。目标设备仍须用torch2.8.0+cu128重新验证实际源码、参数精度/变化、显存、完整独立重载和数据原件。本次不改变设备学习率/weight_decay方案、不扩预算或放行test。
