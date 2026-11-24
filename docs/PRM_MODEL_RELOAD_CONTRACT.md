# Jun：神经PRM模型、损失与重载交接

本交接对应草稿PR #43。原始参考为upstream/prm/train_our_progress_model_lora.py的prm_model.forward及零初始化LN，不修改upstream。现已接入PR #44固定版本的配置/tokenizer模块，并提供本地Qwen/B1初始化、真实短训、推理与续训入口；How负责真实数据转换/采集、目标设备集成与GPU执行，目标1.5B/GPU实训及正式PRM泛化仍待验收。

## 已实现及验证范围

- src/webshop_lab/prm/model_core.py：读取真实backbone.config.vocab_size，在明确的assistant回合末位取词表logits，V到1线性头，按轨迹求和后MSE；标签只进入loss。
- src/webshop_lab/prm/train_core.py：真实PyTorch backward、有限梯度检查与optimizer.step。调用方提供已隔离、已分词、未截断的完整轨迹。
- 保存PRM专用adapter与独立head.pt，并记录初始化、tokenizer、模板、源码和文件清单SHA。加载先verify_prm，再恢复相同基础初始化和PRM adapter，最后restore_head；仅调用restore_head不构成完整重载。
- 早期核心测试使用小型toy backbone和明确的adapter文件占位；新增接线测试使用真实Qwen2结构、真实PEFT及固定Qwen tokenizer，验证B1合并、模型训练、adapter/head独立进程重载和续训。原占位测试只覆盖底层文件契约，不当作实际模型证据；新增小型模型也不代表1.5B/GPU容量验收。

运行模型测试需PyTorch/Transformers/PEFT和固定tokenizer缓存。没有依赖的通用CI会显式skip，不把skip后的绿色状态当模型验收。本轮安装依赖的独立测试实际执行24项，无skip。

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*prm*.py' -v
```

## 与How适配入口的张量契约

input_ids与attention_mask：[B,L]。turn_end_indices：[B,T]，long；turn_mask：[B,T]，bool。有效评分位置严格递增、在非padding范围内，无效位置为-1。labels：[B]，有限且位于[0,1]；未购买正常终止的R为0，技术中断不混为正常失败。输出step_scores：[B,T]，无效步骤严格为0；totals：[B]，loss为按轨迹平均MSE。

评分位置采用包含当前动作的因果末位定义，不向前移位套用PPO概率切片。真实模板必须保证assistant之外的用户/网页/角色/padding位置不被标记。模型核心只能核位置形状和padding，无法从任意token ID推断真实角色，必须由模板模块独立验证。

## 两个容易误验收的点

1. 原评分头零初始化时，首个反向步骤的backbone/LoRA梯度为0属于数学预期，head应有梯度并更新。随后在非零head和非零标签数据上检查PRM LoRA的梯度/参数变化；不能因为第一步LoRA为0就更改原初始化或宣称断梯度，也不能只测第一步就宣布LoRA训练通过。LoRA自身的零初始化也会影响各子矩阵何时出现梯度，按模块和实际更新记录判断。
2. 本核心使用完整[B,L,V]参考路径，再取位；不宣称减少其峰值显存。等价节省路径须在LM head之前先选末位hidden states，再经过原LM head和V到1评分头；核对相同输出与梯度后才能采用。直接hidden-state到1是架构变体。GPU显存不足应保留尺寸、日志和峰值记录，不静默改结构。

## GPU最小实训与独立进程交接

1. runtime.build_prm已提供Qwen构建入口：核基础/B1/tokenizer文件身份，B1在内存中合并后冻结，再挂独立PRM LoRA，不覆盖原文件。小型真实Qwen/PEFT已验证此路径，How仍须对项目1.5B和原B1执行真实GPU检查。
2. 用策略train范围的真实完整轨迹，人工解码每个评分位置。不得把旧六策略160项val记录切作fit/dev。记录原件SHA、任务/规范化指令组、成功/部分/失败分布、超长拒绝数量。
3. 按执行方设备契约固定FP32、关闭TF32/autocast、离线本地快照、batch size 1；记录上下文、依赖、短训批次/步数和容差；两步只是起始探针，若不足以产生可观察LoRA更新须明确追加诊断范围，不冒称正式PRM训练。保存实际loss、head/LoRA梯度、参数前后摘要、峰值显存和耗时。
4. train入口除底层adapter/head/manifest外，还保存config/summary/清单、optimizer/CPU与CUDA RNG及步数、固定输入参考分数。当前恒定学习率，无scheduler；CPU已验证续训累计步数。How仍需验证目标设备CUDA RNG恢复与继续训练。
5. 原进程eval、关闭dropout，对固定无标签输入保存逐步分数及总分；退出后由新进程重新加载基础/B1初始化、PRM adapter和head，逐项比较步骤对齐、分数及总分，采用运行前冻结的容差。不能复用内存模型，不能拿策略critic替代PRM head。
6. GPU未执行、来源未完整或重载不一致就保持待验收，不把CPU toy、文件保存成功、低MSE当完整模块完成。后续常数/长度基线与循环/规格分叉诊断由Qiu独立评估。

GPU仍需由How在#42完成当前批次的保存与audit后交接，并提供实际最后完整run和停派发状态。本交接没有操作GPU进程，也不声称已经腾卡。神经PRM通过后再冻结learned-progress PPO的新协议；旧规则OP结果单独保留。

目标执行环境按[设备契约](PRM_DEVICE_CONTRACT.md)：Python3.10.22/torch2.8.0+cu128/transformers4.56.2/peft0.17.1，不新增依赖；本轮CPU测试在torch2.15.0执行，目标torch/CUDA与项目1.5B仍待How实测，不视为目标设备兼容验收。save_prm仅提供底层子契约，完整产物由train入口补齐。

完整短训/独立进程运行方法及输入schema见[接线说明](PRM_RUNTIME_HANDOFF.md)。save_prm/restore_head仍是底层接口；runtime.build_prm负责完整基础初始化、独立PRM adapter与head恢复，train/infer是可执行入口。前述仅核心、缺入口的历史限制现已关闭，GPU容量/泛化限制继续保留。
