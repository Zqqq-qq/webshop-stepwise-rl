# Arm 4 因果评分与重载交接（按三项评审修订）

本说明对应 PR #60 的组合修复。历史 #59 权重、manifest、五臂结果和旧 reference 均保留原字节。修复不重新训练评分器，也不宣告新的 GPU 作业或策略效果验收。

## 两份源码各自负责什么

- `EXEC_REPO`：干净的历史 runtime `d25e6164894c87fdea796bb73e3ee546807483d8`，用于原检查点的底模、B1、adapter、评分头及初始化身份校验。
- `FIX_REPO`：本次修复提交的代码。`scripts/score_prefix_value_prm.py` 从同目录导入 `prm_prefix_value_support.py`；在线 worker 从修复目录显式加载同一辅助文件。两者不要求旧 runtime 含 `prefix_reward` 或 `data` 模块。
- 不把 `PYTHONPATH` 换成新 checkout 来绕过历史 `source_commit`，也不修改旧 manifest 中的身份或哈希。

历史 #59 的全轨迹编码器只保留用于原五臂探索记录复算。当前生产评分入口逐次处理截至当前动作的前缀，不能再用“未来整条轨迹是否超长”决定早先点击的 ASIN 增强。

## 固定载荷与外部评分契约

历史权重包仍为：

- [五臂权重 Release](https://github.com/Zenon-nnnnnn/webshop-stepwise-rl/releases/tag/prm-prefix-value-20261126)
- ZIP：141,085,800 字节，SHA-256 `3b53a6615d217f8c97b285a9a148bfa686762b81bcc9aa8e371684b8c6f0b43f`。
- Arm 4 manifest SHA：`826f9cce3a51365ec6bf02fb10574ca4990010a8927e9082b5a0383fd028b01c`。
- Arm 4 adapter SHA：`fc78e9c77c6d0722bc2926b337ce06f155d1f9a94d954e7778541f5d0027bf92`。
- Arm 4 head SHA：`a234fd6d1a4d89a2214064f2e17ac7f85896126f7fa018df18eec8b5afe2a23e`。
- 底模 revision：`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；B1 adapter SHA：`c39e2a78e7c5973328e9d2c23fb6320e566352e63cc089562488f3dd46ae5f9c`。

新增的 [arm4-scoring-contract-v1.json](arm4-scoring-contract-v1.json) 位于检查点目录之外：

- 契约 SHA-256：`0cd317837cb2db5073e560e870d86c23563bafa7f67f620d9c801bf182eb844d`。
- 明确头输出为终局奖励前缀价值；训练为逐回合终局标签 MSE；步奖为相邻价值差；总分为最后价值。
- 明确 ASIN 信息来自紧邻的已公开搜索页，按当前前缀进行长度回退、完整旧回合裁剪及空回复/不可评分时零奖励跳过。
- 绑定历史 manifest、全部载荷哈希及外置辅助模块哈希，加载模型前验证。不带契约、契约错配、实现变化或载荷篡改均拒绝。

这是针对已知 PV 训练来源的显式语义迁移声明，**不是根据任意权重自动推断其训练方式**。使用者不能自行给旧 sum 模型生成契约并更新哈希来绕过检查。更新编码规则需发布新契约版本及新的参考预测。

当前版本的旧 `webshop_lab.prm.infer` 会拒绝已登记的五臂 PV manifest/评分头；PV 入口则必须绑定正确契约。未修改的历史 runtime 本身不认识该迁移登记，因此禁止再用它的旧 `prm.infer` 对 PV 权重做正式评分；只能按下述外置入口加载它的底层 runtime。

## 离线命令

本机配置保持原有底模、tokenizer、B1、2048 窗口与 all-linear LoRA 身份；可调整已允许的本地路径，不改检查点目录内原文件。以下环境变量需指向实际目录：

```bash
export PYTHONPATH="$EXEC_REPO/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 首先验证旧 runtime 环境下可导入外置辅助模块。
python "$FIX_REPO/scripts/score_prefix_value_prm.py" --help

python "$FIX_REPO/scripts/score_prefix_value_prm.py" \
  --source-commit d25e6164894c87fdea796bb73e3ee546807483d8 \
  --config "$ARM4_CFG" --checkpoint "$ARM4_CKPT" \
  --scoring-contract "$FIX_REPO/results/prm-prefix-value-20261126/arm4-scoring-contract-v1.json" \
  --scoring-contract-sha256 0cd317837cb2db5073e560e870d86c23563bafa7f67f620d9c801bf182eb844d \
  --samples /path/to/frozen.samples.jsonl \
  --output /path/to/new-reference.json --device cuda
```

首次运行生成新版参考；随后在独立进程中对同一输入运行同一命令，增加 `--reference /path/to/new-reference.json`，并换用新的 `--output` 路径。旧 `references/` 中的全轨迹结果不能直接作为新版 reference，入口会拒绝没有相同契约的旧结果。

原始轨迹可用 `--rollout` 替代 `--samples`，格式仍为 instruction 与 steps 中的 observation/response；按需指定 task-id/repeat。转换逻辑不读取终局标签，也不依赖历史 runtime 缺少的 data 模块。

## 输出与可比范围

- `scoring_mode=causal_prefix_value_v1`，包含冻结契约 SHA。
- 每步记录实际前缀 token SHA、局部 assistant 结束位、裁掉的旧回合数、ASIN 回退和跳过原因。
- `turn_end_indices` 明确为每次前缀的局部下标，裁剪后可能变小，跳过时为 null；不是同一完整序列内的递增全局位置。不得送入旧全轨迹 validator 冒充相同模式。
- `v_prefix` 为各前缀预测；`step_scores` 为差分；`total` 为最后保留的价值。跳过时保留上一价值并给零增量。
- 同一份有效历史在线/离线使用相同编码器。原始完整 messages 不必然等于策略当时经过 context_length/max_new_tokens 裁剪后的历史；重放 PPO 时须按真实 prompt 历史核对逐步输入 SHA，不能只凭 task_id 认为输入相同。
- 新入口不再因后续整条轨迹超长而删除早期可评分动作；逐步缺失/跳过必须报告，不能沿用旧版全轨迹保留分母。正式评估需重新冻结有效集合与指标口径。

## 在线 PPO 配置与执行前验收

在原 neural_progress 配置中增加：

```json
{
  "prm_scoring_contract": "/path/to/fix/results/prm-prefix-value-20261126/arm4-scoring-contract-v1.json",
  "prm_scoring_contract_sha256": "0cd317837cb2db5073e560e870d86c23563bafa7f67f620d9c801bf182eb844d"
}
```

同时保留原 prm_config_sha256、prm_manifest_sha256 和 prm_runtime。父进程先核契约；worker 再核契约、载荷与配置窗口，然后调用旧 runtime 的 build_prm(training=False)，评分器保持冻结。PPO 元数据记录新配置；每步 prm_scoring_trace 记录实际编码身份及契约 SHA。

执行前锁定组合提交与两臂配置；用既有训练侧轨迹复跑新 reference 和小规模 GPU 冒烟，核对 actor/critic 变化、重载与跳过情况。本地 CPU/小模型测试不能替代这次新版本的 GPU 验收。旧 #60 冒烟记录保持其原提交边界，不自动覆盖本次提交。预留 test200 继续冻结。
