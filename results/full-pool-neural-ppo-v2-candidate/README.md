# 保留独立 PRM 验证集的 PPO 候选排程

这份排程用于评审后续 TG 与 Arm 4 的全池对照。以 [#62 固定排程](https://github.com/Zqqq-qq/webshop-stepwise-rl/tree/b09d1823191fa1c406fae6f9e6d69944797bf566/results/full-pool-neural-ppo-v1) 为起点，保护 [#63 新验证集](https://github.com/Zqqq-qq/webshop-stepwise-rl/tree/332e11827a7e0227111e173b1afe785cf0da17d6/results/arm4-prm-expansion-v1)。冻结的目标 GPU 工程复验和当前 PRM 扩训各有独立记录。正式 PPO 仍按团队已定安排等待 PRM 独立验证。

## 选择与预算

原 618 题包含主评 64 题、备选 32 题。用原冻结目标映射和仓库的 `instruction_hash` 按去掉价格上限的指令分组，排除这 96 个任务，未发现额外同指令任务。剩余任务仍属于原训练集，与新 PRM 验证、`val160` 和 `test200` 的 ID 和去价格指令均不重叠。两个种子各自保留原相对顺序，同 seed 的两臂共用顺序。

每臂、每 seed 各遍历 522 题一次，共 131 次更新，前 130 批各 4 题、尾批 2 题。两臂、两个 seed 合计 524 次更新、2,088 条训练轨迹。沿用 #62 的单次更新实测线性估算，单卡训练约 33.7 小时，验证、归档和重试另计。

候选池里有 212 题也在 #63 当前 PRM 训练清单，另有 135 题在计划补采清单。这里先按独立验证隔离生成候选。若正式实验还要求 PRM 和 PPO 两个训练池彼此隔离，要按最终评分器训练清单重新取池并更新预算。正式 PPO 启动前还需要记录最终冻结评分器的检查点与契约。

## 生成批次

从仓库根目录运行：

```sh
python results/full-pool-neural-ppo-v1/prepare_batches.py \
  --frozen-dir results/full-pool-neural-ppo-v2-candidate \
  --output /path/to/new-batches
```

生成器检查原拆分、两份新验证清单及种子顺序的哈希，拒绝任何 ID 重叠，并输出逐批任务和哈希索引。[候选清单](manifest.json)保存来源提交、目标映射哈希、预算和隔离核对结果。执行和验证参数沿用原对照。预留 `test200` 仍保持冻结。
