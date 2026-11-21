# PPO 接入清单：从 SFT 候选到真实 RL 闭环

日期：2026-11-20。主责 Zqqq-qq；模型/概率复核 Jun040504；GPU 执行 HowseaonLink；独立验收 QiuQiu-xuan。

关联：[Issue #4](https://github.com/Zqqq-qq/webshop-stepwise-rl/issues/4)、[训练 rollout 字段设计](TRAINING_ROLLOUT_SPEC.md)、[RL 边界](RL.md)、[逐日计划](DAILY_PLAN.md)。

**本文件是待实现的设计与验收清单。** 新字段、训练采集器、TG 路径和现代 PPO 更新入口均不能因本文件存在而算已完成。方框只有在附上 commit、配置和原始证据后才能勾选。

## 1. 已有基础与缺口

以 `main@aa09199` 为检查基线：线性进度估计器和奖励导出已经实现，#19/#20 已修正为购买终局奖励。现有 rollout 保存文字、动作、环境分、结束标记，但没有保存用于 PPO 的实际 token、旧概率和价值预测。原 `upstream/ppo/` 保留作参考，尚未完成现代环境接入。

How 的 #18 提供可接入检查的 SFT 候选与验证证据；正式模型接纳和来源登记仍需完成。160 个 val 任务用于评估，不能拿来更新策略。SFT 专家数据不是当前策略的新采样。

## 2. 先冻结算法合同

为避免“步级奖励、token 级 PPO”混用，首个实现建议采用下列**候选方案**，由 Zqqq-qq/Jun 在执行前签收，写入配置；它不是已经实测通过的算法。

| 项目 | 首次接入候选 | 必须记录 |
|---|---|---|
| 决策单位 | 一次完整 assistant 回复作为一个动作，包括格式错误的回复 | `optimization_unit=assistant_turn` |
| 概率 | 对实际生成 token 的条件 log probability 求和，得到联合动作 log probability | `ratio_unit=joint_response`；不是 token ratio 的均值 |
| 价值 | 回复生成前、完整实际 prompt 下的 V(s_t)，需要独立 value head/critic | critic 初始化、可训练参数、保存格式 |
| 优势 | 在环境交互步上计算 GAE；首轮候选 gamma=1、lambda=1，避免未解释的长度折扣 | gamma、lambda；后续修改必须两臂同步 |
| PPO | action ratio = exp(sum(new_logp) − sum(old_logp))，使用 clipped surrogate | clip epsilon、epochs、minibatch、停止条件 |
| KL | 固定 SFT reference 可用于监控；首次候选训练 KL 系数为 0 | 若启用 KL 惩罚，另定估计式、落点与系数，两臂一致 |
| 生成分布 | 随机采样，首轮 temperature=1、top_p=1、top_k=0，禁用其他 logits 变换 | 实际 resolved generation config，不能仅存用户传参 |

联合回复 ratio 可能随回复长度出现较大波动或大量 clipping，需要观察再决定是否切换 token 级 PPO。若切换，必须另写 token 时间轴、value、GAE、奖励落点和归一化定义；不能只把 action ratio 换成 token ratio 后仍称同一实现。

- [ ] 写出并冻结算法配置，包括优势是否标准化、loss 按有效动作平均、value loss、梯度裁剪、学习率、seed 和更新次数。
- [ ] 检查旧版 PPO 依赖与当前 GPU 环境；不覆盖已验证的 SFT 环境。
- [ ] 若采用 REINFORCE 备选，改算法标识与验收项，不能留下 PPO 名称冒充完成。

## 3. 初始化与来源

- [ ] 固定基础模型 revision、tokenizer/chat template、adapter 文件 SHA256、有效上下文长度和实际量化设置；成功加载候选 SFT。
- [ ] 确认训练 task ID 属于冻结 train 划分，与 val/test 按 ID 和真实指令都隔离；保存清单与 manifest 哈希。
- [ ] R-TG、R-PG 从同一个 SFT adapter 分别初始化；不能先训练 TG 再接着训练 PG。
- [ ] 固定 reference 为该 SFT；behavior 是本轮采样策略；PPO old policy 是本批更新前策略，更新 epochs 内分母保持不变。三者不能混称。
- [ ] 保存代码 commit、未提交补丁（如有）、环境服务 commit/补丁、health、goal fingerprint 和运行依赖。

## 4. 训练 rollout 与概率对齐

- [ ] 按 [字段合同](TRAINING_ROLLOUT_SPEC.md) 输出独立的训练 schema，不悄悄把现有评估 JSONL 改名复用。
- [ ] 在生成时保留截断后的实际 prompt IDs、原始生成 IDs；不得 decode/strip 后重新 tokenize 代替。
- [ ] token 级 old log probability 对齐生成 token，包含实际采到的 EOS；没有生成的 EOS 不补造。记录 token cap、stop 原因。
- [ ] 同一个旧策略重算 logp，与采样分布 logp 在预设数值容差内一致；差异不通过则禁止更新。
- [ ] 采样和重算使用一致 dropout/模型模式；`eval()` 不等于禁止梯度，更新仍需启用梯度。
- [ ] 第一个回复 token 由 prompt 最后一个位置预测，检查 causal shift；环境文本/历史/padding 的直接策略 loss mask 为 0。
- [ ] Thought、Action 前缀与回复正文若由策略生成，首版都进入联合动作概率；不得按“动作是否合法”决定 mask。
- [ ] value 读 prompt 最后一个有效位置，不能偷看本轮回复或环境反馈。

## 5. 奖励落点：必须逐条验收

令 R 为购买完成后的评分；未购买、任务步数耗尽时 R=0。页面匹配分只存诊断字段。

| 奖励臂 | 第 t 个动作的训练奖励 |
|---|---|
| R-TG | `1[t为轨迹最后动作] * R + beta * grounding_t` |
| R-PG | `alpha * progress_t + beta * grounding_t` |

两臂 beta 相同；alpha、beta 需根据真实分布冻结，不能把现有 CLI 默认值当已选参数。辅助项可使奖励总和超过 R；额外动作可能获益，要检查循环和奖励投机。

- [ ] 实现显式 terminal+grounding 导出/训练路径；当前 terminal 模式不包含 grounding。
- [ ] 每个回复只有一个 action reward。首版 action 级 PPO 直接消费该标量。
- [ ] 同时记录 `reward_token_index`：该回复最后一个实际生成 token 的索引（有 EOS 则为 EOS），用于检查步与 token 的映射；**首版不把它广播成每个 token 一份奖励**。
- [ ] 保存每步 terminal、progress、grounding、KL（若启用）及合成后奖励；progress 预测器版本、训练任务和哈希可追踪。
- [ ] 手工检查成功购买、部分得分购买、未购买、非法动作、循环五种情形；核对奖励数量等于动作数。
- [ ] 独立 train/val 探索轨迹上验证进度模型；报告 val 总分误差和归因案例，不用训练误差代替泛化。不将低 MSE 称为因果归因正确。

## 6. 结束、GAE 与 bootstrap

- [ ] 购买完成：episode terminal，最后 next value=0。
- [ ] 项目定义的 max_steps 是有限购物任务的结束：未购买 R=0，最后 next value=0；不能因为名称叫 truncation 就自动 bootstrap。
- [ ] 仅采集批次切断、任务仍可继续：不是任务结束，保存真实下一状态及 old critic 的 bootstrap value；不能把它混成购物失败。
- [ ] 环境/网络异常：记录失败原因，停止/隔离该训练片段；不伪造为普通 agent 失败。是否重试须留证据。
- [ ] 手算 `delta_t = r_t + gamma * bootstrap_mask_t * V(next_t) - V(s_t)` 和反向 GAE；另用 continuation mask 防止优势递推跨 episode/批次。
- [ ] GAE return target = advantage + old value；targets 和 old logp 固定、无梯度。更新期间不能用新 critic 反复改本批 target。

## 7. 单批 GPU 验收与连续闭环

- [ ] 小批真实训练任务采样，核对有效动作数、长度、奖励、old logp 与 value 全部有限。
- [ ] 更新前 ratio≈1；单元数值例验证正/负优势和 clip 方向。记录 action joint ratio、clip fraction、熵定义、KL 和梯度范数。
- [ ] 记录非零有效梯度、可训练 adapter/critic 参数变化、冻结权重未改变；零优势批次不能宣称学到策略。
- [ ] 保存策略 adapter、critic、优化器、scheduler、RNG、更新号与配置；重载后复核 logp/value，并继续训练。
- [ ] 使用更新后策略采新轨迹，不能反复读取同一份旧 JSONL 冒充新一轮采样。
- [ ] 12/01 验收至少三轮当前策略采样→更新→新采样，并由 Jun/Qiu 独立复核其中一轮。
- [ ] 两奖励臂使用相同初始化协议、算法、辅助项和可比预算；统计实际环境步数、有效动作、生成 token、GPU 时间。性能结论另需独立 val/test。

## 8. 交付与签收

11/20：本清单及 rollout 字段设计；11/24：采集器、TG 路径、单批数值检查；11/25：首次真实 GPU 更新尝试；11/28 缓冲；12/01：多轮闭环。日期是验收计划，不是完成记录。

每次签收附：负责人、commit、配置与哈希、原始轨迹/训练日志路径、检查脚本结果、已验证范围及未解决项。代码能启动、一次 loss 下降、SFT 效果提升，都不能单独替代这些验收。
