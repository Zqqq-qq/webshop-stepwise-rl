# 训练 rollout 字段设计 v0.1

状态：2026-11-20 **设计稿，采集器与校验器尚未实现**。与现有 `schema_version=1` 评估轨迹分开，拟用 `schema_version="rl-rollout-v0.1"`。关联 [PPO 接入清单](PPO_INTEGRATION_CHECKLIST.md) 与 [Issue #4](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/4)。

## 1. 一个 rollout 是什么

一次购物 episode 包含多个决策 turn：模型看到实际 prompt，生成回复，解析动作，环境给反馈。训练数据还必须保留当时的 token 和概率，才能知道“哪个策略、在什么输入下、以多大概率生成了什么”。

存储建议：每个 run 一个 `run.meta.json`，每个完整 episode 一行 `episodes.jsonl`。大张量可以放带哈希的旁文件，但 JSON 中必须有文件、数组键、shape、dtype 和 SHA256，不能只给本机绝对路径。打包共享时保证所有引用可解析。写完生成文件清单与哈希。

首版不跨 episode packing，不在采集中间切断 episode；仍预留 fragment 字段，后续实现前拒绝以 fragment 为训练输入。

## 2. Run 元信息

| 字段 | 类型 / 含义 |
|---|---|
| `schema_version`, `run_id`, `created_utc` | 字符串；格式版本、唯一 run 和时间 |
| `purpose`, `split` | 首版训练必须为 `rl_train`, `train`；不能接受 val/test |
| `code` | commit、dirty 标记、补丁文件及其 SHA256；环境 bridge 的代码与补丁单独登记 |
| `behavior_policy` | base model ID/revision、adapter SHA256、checkpoint 标识、update index、量化/推理 dtype；checkpoint 标识本身不能代替内容哈希 |
| `old_critic` | critic checkpoint/hash、结构和初始化；首次 critic 未训练也要记录 |
| `reference_policy` | 固定 SFT 的同类来源字段；若未启用 reference，显式 null 并声明 KL 关闭 |
| `tokenizer` | ID/revision、tokenizer 文件/template 哈希、特殊 token IDs；PAD 与 EOS 可同 ID，mask 不得由 ID 相等推断 |
| `sampling` | 完整 resolved config、seed、RNG 状态旁文件、dropout/模型模式、context budget、截断方法 |
| `environment` | health、goal fingerprint、服务实现标识和配置、max_steps、任务 horizon 定义 |
| `data` | train ID 清单/hash、split manifest/hash、指令映射/hash |
| `algorithm` | 名称、optimization/ratio 单位、gamma/lambda、clip、KL、loss 归一化、value loss、优化器、更新预算 |
| `reward_config` | TG/PG、alpha/beta、可选 KL 设置、progress 模型/hash/训练划分、grounding 判定实现 |
| `runtime` | 依赖、设备、启动命令、开始/结束时间、运行日志和资源记录 |

采样首选 temperature=1、top_p=1、top_k=0；同时检查 top_k、repetition penalty、min length、强制 token 等实际默认项。否则 `generate` 的行为分布可能不同于原始 softmax，不能直接把 raw logits 的 logp 当采样 logp。首版校验器应拒绝未支持的分布变换。

## 3. Episode 字段

| 字段 | 类型 / 规则 |
|---|---|
| `run_id`, `episode_id` | 字符串；episode ID 全局唯一 |
| `task_id`, `repeat` | 整数；重复采样用 repeat 区分 |
| `instruction`, `instruction_sha256` | 真实购物指令与约定 UTF-8 原文哈希；与冻结映射校验 |
| `behavior_policy_id`, `critic_id` | 指向 run 中不可变来源；一个批次不得混入未知版本 |
| `goal_fingerprint`, `split` | 与元信息一致，训练必须 train |
| `turns` | 有序 turn 数组，至少一个；见下表 |
| `environment_final_score` | 原环境最后返回的分数，只作诊断，不能凭这个字段判成功 |
| `terminal_reward` | 购买结束为购买评分，未购买结束为 0 |
| `purchase_completed`, `success` | bool；success = purchase_completed 且 terminal_reward≥1−1e−6 |
| `end` | reason、episode_terminal、collection_cut、bootstrap_allowed、bootstrap_value；见结束规则 |
| `seconds`, `environment_steps`, `generated_tokens` | 实际资源计数；非法回复属于策略 turn，不一定有成功环境调用 |

## 4. Turn 字段与张量对齐

以下 N 为实际 prompt token 数，L 为实际生成 token 数；首版保存未 padding 的列表。

| 字段 | shape / 含义 |
|---|---|
| `turn_index` | 从 0 递增 |
| `observation`, `available`, `messages` | 生成前页面、可点击项、完整逻辑会话，供人工检查；不替代实际 token |
| `prompt_input_ids` | `[N]`，经过 chat template 与历史截断后真正输入的 IDs，含实际 generation prefix |
| `context_trace` | 原始/保留消息索引、截断 token 数与规则；不能凭完整 messages 假定全部历史都入模 |
| `response_token_ids` | `[L]`，生成原始 IDs，包括实际 EOS；不含 batch padding |
| `response_text_raw`, `response_text_parsed` | 原始解码和 parser 使用文本分开；strip 只影响 parser，不能反向修改训练 IDs |
| `response_mask` | `[L]`，存储记录中全为 1；batch padding 后相应位置为 0 |
| `generation_stop` | eos / max_new_tokens / supported_stop_rule；实际结束 token 与配置可核对 |
| `old_logprobs` | `[L]`，每个实际生成 token 在行为分布下的条件 logp，未求和、有限浮点 |
| `old_action_logprob` | 标量，等于 masked sum(old_logprobs)；首版 PPO 的分母 |
| `old_value` | 标量，生成此回复前 V(s_t)，不读当前回复与未来反馈 |
| `reference_logprobs` | `[L]` 或 null；若启用 KL 则必须存在并明确分布定义 |
| `parsed_action`, `parse_error` | 结构化动作或 null，失败原因或 null |
| `grounding`, `environment_step_applied` | 0/1 与 bool；按已冻结实现定义，保留校验结果与环境调用情况 |
| `next_observation`, `next_available` | 执行动作后的真实反馈；非法动作保留生成的错误反馈 |
| `environment_score`, `purchase_completed_after` | 诊断环境分和本步之后是否购买完成 |
| `reward_components` | terminal、progress（TG 可 null）、grounding_bonus、kl_penalty、combined；同时保留原始 grounding |
| `reward_token_index` | L−1；为空回复拒收，不能写 −1；EOS 若实际生成则为落点 |
| `bootstrap_mask`, `continuation_mask`, `next_old_value` | 0/1、0/1、标量；分别控制 delta 的 bootstrap 与 GAE 递推，不混用 |

观测不是 action，不进入直接策略损失；但它仍作为上下文参与 forward，可以影响回复概率与梯度。非法回复仍是模型自己采到的输出，不能通过 mask 删除。

## 5. 组 batch 时生成什么

对每个 turn 拼 `prompt_input_ids + response_token_ids`，右侧补 PAD 至本批最大长度：

- `input_ids`：`[B,S]`，原 token 加 PAD。
- `attention_mask`：真实 prompt/response 为 1，padding 为 0。
- `action_token_mask`：仅本轮实际生成 token 为 1，prompt 中历史 assistant 回复也为 0。
- `old_logprobs`：按回复 token 对齐；padding 区域不参与求和。
- `old_values`, `advantages`, `returns`：首版每个 turn 一个数，shape `[B]`，不伪装成 token 价值序列。

causal shift 示例：拼接 `[p0,p1,r0,r1]`，r0 的 logp 来自 logits 位置 1，r1 来自位置 2；不要直接拿位置 2 预测 r0。

一个 minibatch 可以打乱 turn，但 GAE 必须先按完整 episode 顺序算好并固定，再取 minibatch；不能在打乱的 turn 上递推优势。

## 6. 奖励落点与 PPO 的精确定义

首版候选为 assistant turn 级 PPO，不是 token ratio PPO：

```text
old_lp_t = sum_j old_logprobs[t,j]
new_lp_t = sum_j new_logprobs[t,j]   # 同一个实际 prompt 与 response token 序列
ratio_t = exp(new_lp_t - old_lp_t)
delta_t = reward_t + gamma * bootstrap_mask_t * next_old_value_t - old_value_t
A_t = delta_t + gamma * lambda * continuation_mask_t * A_(t+1)
return_t = A_t + old_value_t
actor_loss = -mean_t min(ratio_t*A_t, clip(ratio_t,1-eps,1+eps)*A_t)
```

每步 combined reward 只计一次。`reward_token_index` 是显式对齐锚点，首版不消费 token reward 张量；未来切换 token 级算法时需另定 value/GAE 时间轴。不能给每个生成 token 都复制同一份 step reward，导致长回复获得更大奖励。

## 7. 三种结束情况

| 情况 | episode_terminal | collection_cut | bootstrap_allowed | 最后两个 mask / value |
|---|---|---|---|---|
| 已购买 | true | false | false | bootstrap=0、continuation=0、next_value=0 |
| 任务 max_steps 用完，有限任务结束 | true | false | false | 同上；未购买 terminal_reward=0 |
| 只是批次/采集长度切断，任务可继续 | false | true | true | bootstrap=1、continuation=0、next_value=真实下一状态的 old V；首版拒收此类训练片段 |

普通非末步：bootstrap=1、continuation=1，next_old_value 与下一 turn 的 old_value 相同。环境故障记录为 error artifact，首版拒收受影响 episode；不要把缺失反馈当正常 horizon。

## 8. 教学示例：字段如何对应

**下面是假设的数值示例，不是 How 的真实 rollout，也不是可直接上传训练的完整记录。** token IDs 只是示意；真实 IDs 由 tokenizer 决定。

```json
{
  "turn_index": 0,
  "prompt_input_ids": [101, 102, 103],
  "response_token_ids": [201, 202, 203, 2],
  "response_text_raw": "Action: search[red shoes]",
  "response_mask": [1, 1, 1, 1],
  "old_logprobs": [-0.1, -0.2, -0.3, -0.4],
  "old_action_logprob": -1.0,
  "old_value": 0.2,
  "reference_logprobs": null,
  "generation_stop": "eos",
  "grounding": 1,
  "environment_step_applied": true,
  "environment_score": 0.0,
  "purchase_completed_after": false,
  "reward_components": {
    "terminal": 0.0,
    "progress": null,
    "grounding_bonus": 0.1,
    "kl_penalty": 0.0,
    "combined": 0.1
  },
  "reward_token_index": 3,
  "bootstrap_mask": 1,
  "continuation_mask": 1,
  "next_old_value": 0.3
}
```

这里是 TG 臂 beta=0.1 的**示例值**，不是冻结超参。搜索合法所以辅助奖励为 0.1，但没有购买，terminal 部分为 0。取 gamma=1，此步 TD residual 为 `0.1+0.3−0.2=0.2`；最终优势还取决于后续 residual。更新前 new action logp 也应接近 −1.0，ratio 接近 1；若更新后为 −0.9，ratio≈1.1052。

## 9. 校验器待实现规则

1. 必填字段、类型、非空 turn、ID 唯一、来源文件和哈希完整；拒收 NaN/Inf。
2. N/L、mask、logp、特殊 token 与 stop 原因一致，验证 old_action_logprob 的求和容差。
3. 固定旧模型重算概率/value，与保存值按事先记录的 dtype 容差对齐；不能只验证 shape。
4. 每步奖励各分量、TG/PG 公式与末步奖励落点一致；环境分与终局分分开。
5. 结束规则、next value、相邻 turn、bootstrap/continuation mask 一致；拒收未支持的 fragment。
6. train/val/test 隔离、behavior checkpoint 匹配、未知 sampling 变换拒收。
7. 测试夹具至少含成功、部分得分、未购买、非法回复、实际 EOS、token cap、padding EOS 同 ID、丢失字段和来源不匹配。

当前旧 JSONL 可用于人工分析，但不能无损补回当时的实际 token、采样概率或 critic 值。新采集器应在产生轨迹时记录这些字段，不承诺把历史评估数据自动转换为合格 PPO 样本。
