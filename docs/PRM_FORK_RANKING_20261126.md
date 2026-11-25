# 神经 PRM 选品分叉短训

第 126 步模型在 7544 的打开商品步骤把洞洞鞋排在满分跑鞋之前。新实验比较原终局分拟合与加入分叉步排序监督的两组神经模型，使用相同的初始化、训练轨迹、训练顺序和预算。

## 采集训练配对

候选清单来自冻结的 #55 训练池 fit501。筛出原模型三步买得满分、结果页可读出多个商品、无规格操作的 93 条轨迹。用带固定前缀的任务 ID 哈希预先排序，选前 32 条。每条保留归档搜索与原商品，备选固定为同页列表末尾两个不同 ASIN。先实买原商品，再按顺序实买备选，两臂都只走搜索、打开商品、购买三步。原商品必须复现满分；同分照实记录，遇到第一个低分备选即保留一对。正式 test 不采集，7544 已查看，仅用于回归检查。

从本 PR checkout 启动以下命令，evidence_zip 为 [#55 原证据](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm55-expanded-fit-20261125.zip)，使用 How 已运行 #56 的原 bridge。先跑前四条确认环境、归档前缀和奖励，随后移除 `--max-tasks 4`，用新的输出目录采集全部 32 个候选。

```bash
python scripts/collect_prm_product_pairs.py collect \
  --archive "$evidence_zip" --plan results/prm-fork-ranking-20261126/train-candidates.json \
  --server http://127.0.0.1:8766 --max-tasks 4 --out "$fresh_output_dir"
```

将 `plan.json`、`health.json`、`collection.jsonl`、`training-pairs.jsonl` 和 `summary.json` 一起回传。采集器校验真实环境指纹、归档页面和两臂分叉前共同输入。环境分是点击 Buy Now 后的分数。包含错误的尝试保留原因。评估负责人可用清单中的任务 ID 与指令哈希隔离 dev。
