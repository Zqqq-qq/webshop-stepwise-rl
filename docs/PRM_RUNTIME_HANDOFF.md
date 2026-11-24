# Qwen神经PRM训练与完整重载接线

本次在#43集成#44固定提交d95493279808c1a16279af3498218e71a98edb6d的配置/tokenizer代码；其原文件保持不改。不是合并main或宣布GPU训练完成。以下是Jun负责的模型/训练入口；How仍负责真实rollout转换、数据采集、GPU执行与正式PRM推理/PPO集成，Qiu负责fit/dev和泛化验收。

## 当前可执行入口

- runtime.build_prm：只用本地离线快照，核对基础/B1/tokenizer SHA；支持base或sft_b1初始化。后者先加载原B1并在内存中安全合并、冻结基础模型，再创建独立PRM LoRA。原文件不改变。只允许LoRA和评分头训练。
- tensor_batch：接#44的list批处理，转换long位置/bool步骤mask/float32 labels；terminal_reward映射为labels，assistant_mask不误传forward。推理不给labels。
- train：batch size固定1，每条轨迹完整求和后MSE；记录真实梯度和参数变化，保存优化器/CPU及CUDA RNG/累计步数及数据身份。短训不代替正式fit/dev选择与基线评估。
- infer：新进程重新构建相同基础/B1初始化，加载PRM adapter和head；对固定无标签输入核逐步分数/总分与评分位置，默认预设atol=rtol=1e-5。没有可评分样本会拒绝空比较。

## 配置分工

模型配置直接采用#44的configs/prm/qwen25-b1-interface-example.json对应schema，填真实本地model/tokenizer/B1和新output_dir。PRMConfig会拒绝未知字段；不要把旧configs/prm-qwen-smoke-contract.json直接传给load_config。旧文件只作教学契约，保留draft标签。训练步数、学习率、种子和续训路径走CLI；固定FP32、关闭TF32/autocast、上下文不超过2048。不引入主设备新依赖，按设备契约torch2.8.0+cu128/transformers4.56.2/peft0.17.1执行。

真实GPU运行必须使用包含全部PRM源码的干净Git检出；保存实际source commit、源码SHA、tokenizer模板及初始化身份。独立重载在同一冻结代码/输入环境执行。CPU结构测试可在没有Git的隔离目录运行，但输出明确标注范围，不当作目标设备验收。

## 数据入口

train不直接读取旧rollout或PPO episodes。How先转换为一行一条完整PRM轨迹的JSONL，字段如下：

```json
{
  "task_id": 123,
  "messages": [
    {"role": "system", "content": "真实指令及系统提示"},
    {"role": "user", "content": "动作前网页"},
    {"role": "assistant", "content": "真实模型动作"}
  ],
  "terminal_reward": 0.0,
  "done": false,
  "termination": "max_steps",
  "source_sha256": "真实原始轨迹文件的64位SHA",
  "policy_adapter_sha256": "实际行为策略原B1权重的64位SHA"
}
```

示例不是可运行数据，文本及SHA须由真实原件填充。购买终止使用done=true/termination=purchase与实际任务分；未购买正常步数结束R=0；timeout等技术中断拒绝进入拟合。messages包含每轮动作前网页和过去动作/反馈，结束于最后assistant；终局反馈和R不拼入输入。规范化指令组、行为run身份等由数据旁路manifest保存，核心只核任务/策略SHA与输入结构，不能代替原始数据审计。

GPU短训额外核--train-ids文件SHA为当前冻结策略train清单8c85e56b…，并逐条检查成员关系。sft_b1初始化时要求样本policy_adapter_sha256匹配原B1锁；这核对的是提交字段，原件与实际采样来源仍由How/Qiu核验。fit/dev必须在策略train池内按任务和规范化指令分组；任何多策略/repeat不能跨组。旧六策略160项val及保留test不用于PRM拟合或模型选择。

超长整条拒绝，summary保留数量、任务和标签分布所需拒绝清单；数据负责人分别统计fit/dev保留和拒绝率，不能把一半轨迹配完整R。推理可不提供terminal_reward/done/termination，标签完全不参与预测。短训默认两步，数据需覆盖非零标签；若零初始化head配全零标签，不更新是数据/初始化预期，不能据低MSE宣告成功。

## 主设备短训与新进程重载

How先按#43记录GPU安全交接，准备隔离fit样本及实际模型配置。使用configs/prm/qwen25-b1-smoke.json的主设备本地路径，核对并为每次尝试设置新output_dir，例如artifacts/prm/prm-smoke-001，每次尝试均用新目录。以下命令是运行入口，不表示已在主设备执行：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src
python -m webshop_lab.prm.train --config configs/prm/qwen25-b1-smoke.json --samples /path/to/prm-fit.jsonl --train-ids data/processed/train_ids.json --steps 2
python -m webshop_lab.prm.infer --config configs/prm/qwen25-b1-smoke.json --checkpoint artifacts/prm/prm-smoke-001 --samples /path/to/prm-fit.jsonl --output artifacts/prm/prm-smoke-001-reload.json --reference artifacts/prm/prm-smoke-001/reference-predictions.json --atol 1e-5 --rtol 1e-5
```

第一条进程退出后再执行第二条，避免两份GPU模型并存。保存head/LoRA前后摘要、逐步预测和原始stdout/stderr。零head首步LoRA梯度0正常，但必须检查后续PRM LoRA梯度/参数变化；必要诊断增步需明确记录。当前参考路径生成完整logits，尚未证明16GB容量；OOM或非有限失败保留attempt记录、日志与峰值，不静默换dtype或评分结构。

续训用新output_dir配置和--resume <旧检查点>。恢复完整初始化、PRM adapter/head/optimizer、CPU与CUDA RNG和累计步数；数据文件、train清单、种子和学习率须保持一致。学习率恒定，当前无scheduler，不声称已恢复不存在的调度器。

输出至少有adapter/、head.pt、config.json、summary.json、prm-manifest.json、SHA256SUMS、training-state.pt及reference-predictions.json。目录外另有<run>.attempt.json，保存本次状态/失败原因及可取得的GPU峰值，不覆盖旧尝试。GPU容量与历史状态未验证的字段不补造。

## 本次验证范围

已用固定真实Qwen tokenizer测试#44的EOS/评分位/padding接口，并以小型真实Qwen2结构、PEFT LoRA执行CPU训练、B1合并初始化保持、完整adapter/head独立子进程重载及续训。小型结构随机权重只作集成测试，不是项目1.5B权重或正式神经训练。CPU环境torch2.15.0、transformers4.56.2、peft0.17.1，目标torch2.8/CUDA/1.5B仍待How实测。

不改变原每回合词表logits到标量、贡献求和MSE定义。低总分误差不代表逐步归因正确；正式PRM仍需独立dev及常数/长度基线、规格分叉/循环/无效动作诊断，通过后再冻结learned-progress PPO方案。test仍未放行。
