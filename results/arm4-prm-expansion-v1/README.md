# Arm 4 神经进度模型扩训

本轮训练评分器。购物策略的 618 题 PPO 清单单独保留，等待扩训后的独立验证。预留 `test200` 保持冻结。

## 轨迹盘点

核验 [#47 归档](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/47#issuecomment-7120027532) 的 `partial911-source/b1-train-1451.jsonl` 及原始 metadata。原请求 1,451 个映射训练任务，归档有 911 条真实轨迹，其中旧 fit 729、旧诊断 dev 182、540 个任务未在该归档找到采集原件。[#55](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/55) 对 729 条完整轨迹做 2048-token 门禁，保留 fit501、排除 228 条，全部属于整轨超长，其中 120 条未购买。

用同一个冻结 Qwen tokenizer 基于实际生成时的 2048 context、128 max-new-tokens 重建策略可见前缀，再按 #61 因果评分方式编码。对 729 条旧 fit 去掉 36 个已查看案例、同指令碰撞或去预算重复，余 **693 条、3,656 个可评分前缀**。其中 480 条来自 fit501，213 条来自原超长排除集。没有对有效动作使用点击后信息或终局标签做文本输入。

| 来源 | 满分购买 | 部分分购买 | 零分购买 | 未购买 |
| --- | ---: | ---: | ---: | ---: |
| 历史 fit501 | 154 | 292 | 26 | 29 |
| 扩训可用 693 条 | 167 | 353 | 32 | 141 |
| 其中恢复的 213 条长轨迹 | 20 | 72 | 8 | 113 |

原行为轨迹来自量化 B1 采集，评分器训练用固定 FP32 B1 初始化。低分和未购买的真实轨迹保留；未购买的监督目标为 0，即使最后页面出现非零匹配分。完整任务状态、排除理由与文件哈希见 [inventory.json](inventory.json)。

## 已冻结的扩训和补采

- [693 条已核验训练顺序](train-verified-ids.json) 是这一版能立即执行的完整已验证池。固定 Arm 4 的末层隐藏表示、all-linear LoRA（r=8、alpha=16）、评分头、点击前 ASIN 文本和逐前缀终局回归目标。从同一 B1 重新初始化，seed 42。每更新 8 条轨迹，轨迹内按有效前缀取均值。两遍分别 **87、174 次累计更新**，每遍尾批 5 条。评分头 LR `2e-5`、LoRA LR `2e-4`，沿用余弦调度、AdamW、梯度裁剪 1。第 1 遍和第 2 遍各保存权重、语义契约和更新记录。第 2 遍是预先指定的主评检查点。
- [135 个新训练采集候选](train-recollection-candidates-ids.json) 来自归档没收齐的任务，排除了已知 dev、训练指令族与这轮新留出。采到并校验之前只计候选，下一份数据版本可再纳入；不要在本轮中途换训练清单。
- 新验证单独锁定 [主评 64 题](fresh-dev-primary-64-ids.json) 与 [备选 32 题](fresh-dev-reserve-32-ids.json)。均来自未进入 911 条归档的任务，已排除已知查看记录和训练指令族。先采主评，若可评分且有环境分差的选品配对少于 20，再按固定备选顺序补。补采依据环境和长度，在模型揭分前决定。若环境同分或无有效分叉，单列且不计选品胜负。这 96 个任务与暂存的 PPO 全池清单重叠，后续若启动 PPO 且还要把这个 dev 保持为验证集，需先从 PPO 训练池扣除并重新冻结。现有 PPO 文件保持原样。

### 缺失 540 题对账

[How 补交的采集日志](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/63#issuecomment-7200784113)显示，当前购物页面超出策略上下文上限，采集在 `fit_history` 中退出。原 JSONL 是完整任务清单的前 911 条，最后一条任务为 6395，下一条未落盘任务为 6426。日志输出缓冲会混排，故障 traceback 本身不用于定位具体任务。逐题去向和来源哈希已记录在 [对账清单](missing-540-reconciliation.json)。

| 去向 | 题数 | 原件 |
| --- | ---: | --- |
| 后续 #55 dev | 160 | 后续真实 rollout |
| 后续 #57 dev | 64 | 后续真实 rollout |
| 后续 #59 dev | 64 | 后续评估尝试清单 |
| 新 dev 主评与备选 | 96 | 已冻结 ID，待采集 |
| 新训练补采候选 | 135 | 已冻结 ID，待采集 |
| 隔离排除 | 21 | 10 条去价格指令碰撞、10 条同指令孪生、1 条已用于短训（8742） |

补采时先按 [135 题训练清单](train-recollection-candidates-ids.json) 用冻结 B1 和同一配置采集。新 dev 使用自己的独立清单与目录。每批保存原始 JSONL、生成 metadata、当批 ID、执行日志与失败原因；长页面触发超限时保留日志，按清单导出的缺题另起文件。只对已经落盘的完整 episode 做覆盖验收。若同一题再次超限，保留失败原因，另评估怎样补采，不用页面匹配分填补终局分。

```sh
PYTHONPATH="$EXP_REPO/src" python "$EXP_REPO/scripts/audit_arm4_recollection.py" \
  --inventory "$EXP_REPO/results/arm4-prm-expansion-v1/inventory.json" \
  --group train --config "$COLLECT_CONFIG" \
  --shard "$FIRST_JSONL" "$FIRST_META" "$FIRST_IDS" \
  --remaining-ids "$NEXT_IDS" > "$COVERAGE_JSON"
```

后续每次给验收命令加入所有已有 `--shard JSONL META IDS`；新 dev 用 `dev-primary` 或 `dev-reserve` 分开查。不完整时脚本退出码为 2，并按冻结顺序导出未完成 ID。遇到重复完成、顺序变化、配置或 ID 哈希失配会拒绝验收。补采结果进入下一份训练数据版本。

## 执行

从已校验的 [#47 历史归档](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm-diagnostic-evidence-20261124.zip) 取原始 JSONL。用旧固定 runtime `d25e6164894c87fdea796bb73e3ee546807483d8` 的 `src` 加载模型，外置脚本来自本分支。GPU 用 [#60 固定 runtime 交接](../prm-prefix-value-20261126/ARM4_HANDOFF.md) 中的本机 Arm 4 配置，按下列命令启动。加 `--preflight` 可只用 CPU 重验轨迹和 tokenizer。

```sh
PYTHONPATH="$EXEC_REPO/src" python "$EXP_REPO/scripts/train_arm4_expansion.py" \
  --raw-rollouts "$RAW_ROOT/partial911-source/b1-train-1451.jsonl" \
  --inventory "$EXP_REPO/results/arm4-prm-expansion-v1/inventory.json" \
  --config "$ARM4_CONFIG" --output "$NEW_RUN"
```

同时用固定 B1 采集新 dev 的真实终局分和同页选品分叉。自然轨迹采用 FP32 贪心、2048 context、每回合最多生成 128 token、每题最多 10 步，每题一次。固定原始轨迹后，对原 Arm 4 和两个阶段检查点使用同一 #61 因果评分入口独立重载。对共同覆盖的完整轨迹报告总分 MSE 和配对区间，另报常数、回合数均值和 OLS 三条仅用新 fit 拟合的基线。选品步报告有环境分差对的方向、分差和点击前证据子集。对零分购买和未购买分别报告累计评分达到 0.30 的比例及分母，同时披露跳过和回退覆盖。任务 89、6285、7544 单列作已看回归。冻结的末轮检查点和旧 Arm 4 是主对照；不以这个新 dev 调参或挑最佳轮次。扩训、独立 dev 和 GPU 结果另行交付后复算。
