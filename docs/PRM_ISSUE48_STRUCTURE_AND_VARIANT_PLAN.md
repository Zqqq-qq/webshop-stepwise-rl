# Issue48：结构查证与隐藏状态变体方案

日期：2026-11-24。Jun模型侧交付。此文件是查证和待冻结设计；不是新GPU执行单，也没有在现行runtime中打开hidden评分。沿用现有#43分支，不修改main、原upstream、原B1或历史检查点。

## 1. 查证结论

| 问题 | 判断 | 证据 |
|---|---|---|
| 上游LoRA代码是否取assistant连续区间末位的raw logits，经线性头逐回合求和并MSE对齐R？ | 是 | 固定main快照的LoRA文件93–112、118–140行；推理42–98行 |
| 所有训练分支是否使用同样的评分链？ | 是，但训练条件不同 | 非LoRA/fp16/LoRA-fp16仍是logits→linear→sum→MSE；只有bf16 LoRA版本74–76行显式将head权重和偏置清零 |
| 论文是否规定末token的next-token logits作为特征？ | 论文未写明 | §3.2式(5)描述最后隐藏层接MLP；式(6)(7)才规定贡献求和与终局MSE |
| 论文是否证明总分监督唯一识别真实逐步贡献？ | 未给出该唯一性证明 | 附录A式(13)–(20)讨论势函数差分及理想回报/梯度关系，不构成逐步语义唯一性证明 |
| 高余弦是否证明动作信息不存在、常数解不可避免？ | 不是 | 见下节独立数学分析和CPU反例 |

固定代码来源：[本项目upstream LoRA](https://github.com/Zqqq-qq/webshop-stepwise-rl/blob/01cb439643b8017afcabe9a9e7d2f4b6eb24c2a7/upstream/prm/train_our_progress_model_lora.py#L74-L140)、[推理](https://github.com/Zqqq-qq/webshop-stepwise-rl/blob/01cb439643b8017afcabe9a9e7d2f4b6eb24c2a7/upstream/prm/inference_prm.py#L42-L98)。五份文件均通过连接器按该固定commit读取查证；原包不能冒充作者所有公开版本。
论文：[SPA-RL §3.2](https://arxiv.org/html/2505.20732v1#S3.SS2)、[附录A](https://arxiv.org/html/2505.20732v1#A1)。本地78条样例不代表论文全量探索数据。

## 2. 当前模型结构和数学判断

#43固定f28540b中model_core.py 17–20行创建V→1零初始化head，45–60行取完整logits、按turn_end_indices选位、屏蔽padding、求和及MSE。没有softmax，raw logits不是归一化概率。当前Qwen取完整动作结束符im_end自己的位置，不套PPO next-token logprob的前移索引；该位置的因果隐藏状态已看过当前动作和先前页面。

设post-final-norm隐藏向量h∈R^d，LM输出矩阵W∈R^(V×d)，z=Wh，评分c=wᵀz+b=(Wᵀw)ᵀh+b。如果W满列秩，raw-logits线性头能够表示任意hidden线性评分；实际W的秩/条件数尚未测量，不能当作实测结论。两种参数化的学习动态不同：例如普通梯度下，有效hidden权重更新涉及WᵀW，Adam、裁剪与正则化又增加差异；表达能力相同也不意味着相同lr有相同收敛速度。

合成CPU反例z0=(100,-1)、z1=(100,1)余弦0.99980002，仍可用w=(0,0.5)、b=0.5完全预测0/1。该反例已执行，不是真实Qwen特征。总分监督对每步贡献仅给同一误差导数2(sum(c)-R)，允许多种分解；但不推出所有轨迹总分只能为常数。每步近似常数往往形成步数相关的总分偏差。

Qwen官方配置：V=151936、d=1536、28层、tie_word_embeddings=true。[配置](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json)。本次网页取得的是官方main配置，固定项目权重/辅助配置身份仍以既有锁定SHA为准，没有将网页main冒充固定快照校验。
Llama3.2-3B：V=128256、d=3072、28层，共享embedding，见[Meta结构](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py#L608-L620)、[Meta模型卡](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md)、[官方Transformers转换器](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/convert_llama_weights_to_hf.py#L361-L375)。没有“Llama末位天然有动作分、Qwen没有”的结构保证。

本地Llama SFT脚本为1epoch/lr2e-4/r8/alpha16/2048；本组B1记录1epoch/656步/固定adapter。未取得原作者WebShop合并Llama权重及实际训练日志，无法确认两者相同示范/有效更新，也未执行Llama同40位置余弦、熵或尖锐度对照。不下载/替用通用Llama冒充原权重。

## 3. FP32、BF16与当前归因边界

#43 runtime 106–108、140–141行按FP32加载并拒绝非FP32参数，NeuralPRM head沿backbone dtype构造；train.py AdamW直接接收这些FP32参数，没有显式BF16原地参数写回或单独master参数实现。优化器状态的实际dtype仍应从对应运行产物读取，不能由源码推定所有历史实验。

本次合成CPU AdamW探针：初值约0.02、恒定梯度1、lr5e-6、wd0、512步，BF16参数变化0，FP32变化约-0.00255966；本机BF16 exp_avg/exp_avg_sq也为BF16，step为FP32。此构造说明舍入风险存在，不解释已归档FP32实验，更不复现8样本512步那次训练。结果见results/prm-structure-20261124/cpu-probes.json。

因此暂不能将0.1385/最大误差0.649定因于BF16、模型太小、裁剪或表达能力。该8样本结果及40位置余弦仍是How报告，须核其自身checkpoint/optimizer/日志，不用另外三臂的原件替代。

## 4. 先做一次固定特征诊断

How提供原8条样本和SHA、原评分位/终止信息、明确B1/初始PRM身份。使用eval、无dropout，标签单列不传入backbone；逐步提取raw logits，保留task、位置、特征SHA。Jun核位置、因果前缀及输入无R。

对每轨迹构造F_i=[sum_t z_it, N_i]，求Fθ≈R，最后一列对应每步bias；不另加轨迹级截距。建议一次CPU float64 SVD，full_matrices=False、显式rcond=1e-10，保留全部奇异值、截断秩、奇异值比、残差、最小范数权重及FP32重算误差，报告阈值敏感性限制，不循环搜索rcond。8行系统欠定，拟合好只证明这批标签可表示；巨大权重/FP32误差说明数值风险，冻结表示拟合差也不证明可训练LoRA无解。

先测F比直接分解151936×1536的W更便宜、更贴当前失败。若要声称W满列秩，再独立报告实际W谱与数值容差；不把F满行秩当成W满列秩。对W用WᵀW会平方条件数，不能不加说明把Gram小特征值当精确秩证据。本次没有真实特征或完整项目权重，尚未执行这项诊断。

## 5. 变体配置：同模块可选、不同检查点

两个提案JSON见configs/prm/proposals/，都含有效的旧runtime_config子对象和独立architecture描述。外层是proposal，execution_authorized=false且数据SHA=null；不能直接交给现行train CLI，旧load_config会拒绝。这避免将未实施的设计伪装成可执行训练配置。

| 项目 | logits对照 | hidden变体 |
|---|---|---|
| 表示 | 同一im_end处raw logits | 同一im_end处post-final-norm hidden |
| 头 | Linear(151936,1)，151937参数 | Linear(1536,1)，1537参数 |
| 聚合/监督 | sum + trajectory MSE | 相同 |
| 初始head | weight/bias零 | 相同 |
| 原B1/PRM LoRA | 同一固定B1、安全内存合并后独立q/v LoRA，r8/alpha16/dropout.05 | 相同 |
| 精度/长度 | 非量化FP32、TF32/autocast关、2048、batch1 | 相同 |

实现方案：在现有NeuralPRM/runtime/config中增加显式representation开关，同一模块负责两条路径。hidden从带当前LoRA的Qwen decoder输出last_hidden_state（最后RMSNorm之后）取位，不detach；不能改为内容末token/平均池化，也不先生成全部logits再假称省显存。直接调用decoder须验证PEFT注入、active adapter、梯度检查点和input-require-grad仍生效；本次小型真实Qwen2/PEFT CPU探针已验证对应末位输出与后续LoRA梯度，未验证目标1.5B完整训练。

直接hidden线性头是相对代码的架构变体，且不等于论文所有MLP实现；不宣称逐字复现。若只在选位hidden后再算原LM head和旧V→1，才是保持原数学定义的等价优化，需输出/梯度对照。两者不可混名。

[Transformers4.56.2 Qwen源码](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen2/modeling_qwen2.py#L357-L420)显示最终norm、last_hidden_state和无bias lm_head的关系。

## 6. 保存/重载方案与兼容矩阵

新实现应另用manifest schema_version=2，在既有base/B1/tokenizer/source身份之外记录architecture中的representation、head_type、input_dim、score_position、reduction、loss、dtype、初始化及唯一score_definition。adapter、head.pt、config、reference、optimizer/RNG、source与文件SHA继续完整保存，输出全新目录；每个checkpoint固定一种头，不是一个head.pt随时切开关。

| 文件类型 | 预期加载行为 |
|---|---|
| 原v1 logits检查点 | 保持原始文件/manifest不变，在原固定源码环境按v1加载 |
| 原v1检查点请求hidden | 加载权重前拒绝；不能截断/reshape旧head |
| v2 logits→v2 hidden，或反向 | 在加载adapter/head之前拒绝类型/维度/定义不符 |
| 同类型v2，base/B1/tokenizer/source不符 | 拒绝，不删身份检查 |
| 同类型v2，head文件损坏/形状/dtype/有限性异常 | SHA/安全权重读取及实际tensor检查拒绝 |
| v2合法完整检查点 | 恢复原初始化+PRM adapter+head；新进程无标签输入比逐步和总分 |

顺序：读schema→核预期结构→核严格初始化/source身份→核全部文件SHA与安全路径→建立对应backbone/head→安全加载实际tensor并核[1,input_dim]/[1]/FP32/有限性→完整恢复adapter/head→eval逐步/总分比较→如续训，再核匹配的optimizer参数映射/超参/数据/CPU-CUDA RNG及累计步数。现行verify_prm只支持v1；未来必须显式分派v1/v2，不能放宽成“忽略未知结构”。

本次variant_contract.py只提供待实施头部检查器和拒绝测试，不是完整v2保存/重载器；明确不能绕过现行verify_prm。小型CPU head.pt回读也不是完整变体checkpoint验收。原v1完整保存/重载测试仍保留。

## 7. 有界优化建议与设备计划（尚未授权执行）

先等固定特征诊断，不同时改lr/初始化/裁剪/顺序。提案为两臂共同AdamW lr1e-7、betas(.9,.999)、eps1e-8、wd.01、clip1、head零初始化，原8条固定shuffle42循环，最多128更新/臂、每64步评估，不搜索、不追加预算。这是与设备a67db6a超参对齐的候选，现行#43 trainer仍固定wd0，不能直接声称CLI已经支持.01；实际实施必须另冻结代码/命令及优化器状态。按同样预算执行才有可比性，不承诺同lr最优。

具体诊断顺序：记录每步preclip norm、裁剪比例、head/LoRA实际更新L2和最大值/未变元素比例、参数/梯度/Adam状态dtype。若线性解可拟合而Adam失败，先在固定特征上做head-only受控对照。若需warmup，建议另登记8更新线性warmup后常数lr；若改随机/最小二乘初始化、clip阈值或每epoch重排，每次只变一项，并与原对照等额限制预算。不凭尖峰直接改clip，更不能把首步LoRA零梯度当断图。

设备执行前：数据原件及SHA齐全、特征诊断已交、Zqqq冻结新协议、完整v2实现/兼容测试通过、#47实际门禁修复独立复审。先新目录1–2步GPU冒烟，保留所有OOM/数值失败，不静默降精度/截轨迹；再按冻结执行单走配对预算。16GB可能更容易容纳hidden路径，但不能预先保证。

单张量估算：完整[1,2048,151936] FP32 logits约1.159GiB，最终[1,2048,1536] hidden约12MiB；这是张量体积，不是整个训练峰值。避免output_hidden_states=True收集所有层；hidden仍需Transformer激活，tied LM矩阵与输入embedding共享，不能因为不调用LM head就声称整块权重可删除。目标峰值/耗时/梯度必须实测。

## 8. 文献与适用边界

- [PURE / Stop Summation §3、§4.1](https://arxiv.org/html/2504.15275v1)：改使用既有PRM时的RL归因，并以逐步标注另训PRM；不是sum-MSE拟合失败的直接修复。需新PPO协议，数学变换便宜，但要先有可靠步骤分，购物可恢复性假设需另验。
- [Jia等，ICML2025](https://proceedings.mlr.press/v267/jia25f.html)：覆盖等假设下结果监督未必统计更难，rollout优势可作过程奖励；不保证当前优化成功或语义分解唯一。固定继续策略MC优势需新数据/标签协议，16GB可串行，交互成本另计。
- [Lessons of Developing PRMs §3.1–3.2](https://arxiv.org/html/2501.07301v1)：MC终局成功率可能混淆步骤正确性与恢复能力，结果与步骤评价应结合。WebShop可借鉴分叉/循环/恢复探针，不能直接搬数学正确性标签。CPU人工/环境诊断可行，变更监督需新协议。

## 9. 已执行与未执行

本次完成五份固定upstream文件查证、论文/官方结构阅读、CPU高余弦反例与BF16风险探针、5项方案契约测试及3项小型真实Qwen2/PEFT形状/梯度/head回读测试。无真实8条特征、原作者Llama权重或新GPU运行，不报告根因已定位，也不报告hidden模型已训练。完整测试结果另存results/prm-structure-20261124。
