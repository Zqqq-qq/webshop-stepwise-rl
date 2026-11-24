# 神经 PRM 配置与 Qwen 回合接口

关联 [适配计划 #43](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/43)。本次实现配置校验、实际 Qwen 模板/回合定位和 Python list 批处理；不包含神经模型、GPU 训练或 PRM 训练数据构造。模型/训练由 Jun、数据/推理由 How 接入。原 `upstream/` 不变。

## 配置

`webshop_lab.prm.config.PRMConfig` 支持 `from_dict`、`to_dict`、`fingerprint`；`load_config(path)` 读 JSON。未知字段、非固定 revision、无效 SHA、B1 身份缺失、输出覆盖输入目录及矛盾精度标记会拒绝。

[配置样例](../configs/prm/qwen25-b1-interface-example.json) 的底模、分词器、原 B1 身份来自现有身份锁。它是接口样例，不是经过 GPU 容量验证/冻结的训练命令；`NEW_RUN` 必须换成新目录。训练超参由训练模块单独管理，不能不加说明塞进共享配置。

`verify_input_files(config)` 流式核对本地单文件底模、记录的 tokenizer 文件和原 B1 文件 SHA。它不导入 Torch，也不加载或合并 adapter。加载前和运行结束仍需实际调用核验；仅 JSON 中填写 SHA 不代表已核验载荷。Hub 标识可作为配置记录，但这个核验函数要求已经下载到本地的路径。

`sft_b1` 表示必须保留锁定 B1 作为初始化组成，建立独立 PRM 参数/LoRA/head；如何合并原 B1 由模型模块实现并核对身份。本模块不实施合并、不训练模型、不覆盖原 B1。

## 单轨迹编码

```python
from webshop_lab.prm.tokenize import tokenize_trajectory, collate_trajectories
encoded = tokenize_trajectory(tokenizer, messages, max_length=2048)
batch = collate_trajectories([encoded], pad_token_id=tokenizer.pad_token_id)
```

`tokenizer` 必须是固定版本的 Qwen fast tokenizer，加载者核对实际 tokenizer 文件与配置身份。只支持纯文本 system/user/assistant；显式初始 system、最后一个消息必须是最后一次 assistant 动作。最后动作后的环境反馈/终局信息不能拼入该动作评分；由数据模块从原轨迹转换。中间失败动作与反馈保留。连续 assistant 消息支持独立回合，工具调用/图片等格式不在首版范围。

每条消息只含 `role`/`content`。空白 assistant、正文中嵌入保留特殊 token、非前缀稳定模板均报错。HTML/普通尖括号、中文和换行可正常编码；普通网页原文不得凭空改写为角色标记。

通过真实 `apply_chat_template` 渲染完整序列，使用这一次 fast encoding 的字符 offset 找回合，不拼接逐段独立分词结果。完整编码同时与模板 `tokenize=True` 的 token ID 比对。每个回合的前缀渲染必须与完整序列一致。

### 评分位约定

`score_position = assistant_im_end`：assistant 内容加 `<|im_end|>` 为其半开 token span，**在该 im_end 自己的位置评分**；其后的换行、角色名、网页和 padding 不入动作 mask。若内容前导空白与 header 末换行被 BPE 合成一个 token，该空白边界 token 归 assistant，其他角色文字不允许跨界。

这是读完动作后的神经评分，不是 PPO 的 next-token logprob，不执行向前移一位。模型必须采用因果 attention；这里验证早期评分位的输入前缀不随未来页面变化，不声称已验证 GPU 模型的因果前向。

训练与线上推理必须遵循相同模板和 EOS 约定。若 PPO 原始回复因 max_new_tokens 停止而无 EOS，这个接口会按模板加 im_end 来定义该回复的 PRM 评分边界；这不是 PPO 原始采样序列，不可拿 PRM 编码代替 PPO 的旧 token/logprob。

| 字段 | 单样本 | 批处理 | 含义 |
|---|---|---|---|
| input_ids | [L] | [B,Lmax] | 完整模板 token |
| attention_mask | [L] | [B,Lmax] | 真实 token 为1，右 padding 为0 |
| assistant_mask | [L] | [B,Lmax] | assistant 内容/EOS 为1；不充当MSE标签 |
| assistant_spans | [T,2] | 编码样本保留 | 半开 [start,end) |
| turn_end_indices | [T] | [B,Tmax] | 有效回合的 im_end 位置 |
| turn_mask | [T] | [B,Tmax] | 有效回合1，补齐回合0 |
| terminal_reward | 无；数据模块可附加 | 可选 [B] | 仅作训练标签，不进入输入 |

无效回合 index 为 -1。**必须先按 turn_mask 过滤/替换安全索引，再 gather**，不能直接将 -1 用作末 token。token padding 不通过比较 token_id 判断，因此 pad==eos 也不会掩掉真实回合EOS。模型所有回合求和和损失必须使用 turn_mask；本模块返回 list，由训练方显式转换为 long/bool/float 等设备张量。

batch中若有标签必须全部有标签，范围有限且在[0,1]；推理不需要标签。task_id、来源身份、样本分组由数据层保留为旁路元数据，不自动输入模型。批内混合模板或被改坏的span/mask会报错。

## 超长与数据边界

不做训练序列截断。超过 max_length 抛 `TrajectoryTooLong`，包含实际长度；数据方要分别报告fit/dev拒绝数量、奖励/任务分布及最终保留集。不能把截断后的部分步骤和整条R硬配对，也不能靠删除长失败轨迹隐瞒偏差。保留测试和策略val不能用于拟合PRM。

## 验证

CPU配置测试无需模型依赖；实际模板测试需要 Transformers 和本地已缓存的固定 Qwen tokenizer，没有缓存时会显式skip而不下载权重。验收时须确认模板测试确实执行，而非只看到skip后的绿色状态。

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_prm*.py' -v
PYTHONPATH=src python scripts/check_prm_tokenizer.py \
  --sft-json /path/to/webshop_sft.json \
  --output /tmp/prm-tokenizer-check.json
```

[实际 CPU tokenizer 验证记录](../results/prm-qwen-token-mask-20261123/tokenizer-check.json)：Transformers 4.56.2，固定Qwen revision，检查原始SFT前16条完整对话，12条编码、4条显式超长拒绝。只用于编码诊断，没有用于PRM拟合/选择；无原始对话复制进Git、无Torch/GPU/真实模型训练。真实 GPU 梯度、head/LoRA更新与独立重载仍由后续模块验收。
