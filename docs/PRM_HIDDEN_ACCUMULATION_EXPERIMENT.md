# Issue #48 的 hidden PRM 梯度累积对照

以 [PR #50](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/50) 的候选实现为起点。使用 B1 初始化的 Qwen2.5-1.5B-Instruct，在每个助手回合的 `<|im_end|>` 位置读取最终归一化后的 1536 维 hidden state，经线性头得到各回合分数。沿用轨迹总分拟合终局标签的 MSE。先测 8 条真实轨迹能否拟合，再由 Issue #48 的独立 dev 流程评估泛化。

## 2 步 GPU 冒烟观测与双尺度优化修订

在基于 `final_hidden_state`（`schema_version=prm-qwen-config-v2`）的 2 步 GPU 冒烟检查中，显存（`6.26 GiB allocated / 6.89 GiB reserved`）、`schema 2` 清单与跨进程重载一致性（`reference_match=true`）均已验证通过；但当 `score_head` 与 `LoRA` 共用恒定学习率 `lr=3e-4`（`accumulation=8`）时，第 2 步即出现 `totals` 由 `0` 跃升至 `3.65–12.04`、`loss` 升至 `43.13`，且两次更新均触发梯度裁剪。

原因在于 `score_head` 与底层 `LoRA` 处于不同的数值尺度：

1. **首步全坐标符号跳步与预训练 RMSNorm 缩放叠加**：`score_head`（`Linear(1536, 1)`）采用全零初始化，第 1 步反向传播至底层 `LoRA` 的梯度严格为 `0`，仅有 `score_head` 更新。AdamW 首步更新幅度与梯度模长无关，对全部 `1,536` 个权重坐标同时施加 `±lr` 位移；乘上预训练 `Qwen2.5-1.5B-Instruct` 末层 `RMSNorm` 的通道缩放权重（单回合 $\|h_t\|_1 \approx 4.05 \times 10^3$）后，`lr=3e-4` 使单回合预测跃升约 `1.21`，在 3 回合与 10 回合轨迹上分别累积为 `3.65` 与 `12.04`。
2. **线性头稳态抖动与 LoRA 特征分离速度冲突**：若将共用恒定学习率统一压低至 `1e-7 ~ 1e-6`，底层 `LoRA` 在 120 次更新预算内位移过小，难以将同为 3 回合的满分购买（`0, 6225`）与错误购买零分轨迹（`5031, 95`）拉开；若对 `score_head` 维持较大恒定学习率，末期每步符号翻转仍会在跨回合求和后产生显著抖动，阻碍训练误差收敛至 `MSE < 0.001`。

为此，`webshop_lab.prm.train` 在不改动 `model_core.py`、`prm-manifest.json` 的 `schema_version=2` 清单契约、评分位置或损失定义的前提下，新增可选参数 `--head-learning-rate`、`--lr-scheduler {constant,cosine}` 与 `--cosine-total-steps`：

- 将 `head_lr` 设为 `1e-5`（较 `3e-4` 缩小 30 倍），使第 2 步初始 `totals` 按比例落入 `0.12–0.40` 区间，消除初始超调；
- 将底层 `LoRA` 初始学习率设为 `2e-4`，保留充足的特征调整步长；
- 采用 `cosine` 余弦退火在总更新步数内平滑衰减两组学习率，消除训练末期线性头的稳态抖动。

默认路径与检查点身份边界：省略上述新参数时，默认保持单参数组 `constant` 学习率，延续旧 CLI 的数值更新行为；但 `training-state.pt` 新增了 `head_learning_rate`、`lr_scheduler`、`cosine_total_steps` 三个调度状态字段（并非训练状态文件逐字节不变），且 `checkpoint_identity` 仍严格绑定检查点生成时的 `source_commit` 与源码 SHA-256。因此，既有检查点（如 PR #52 原链）按设计不会跨源码版本直接互载——PR #52 继续在其原始生成 checkout 上恢复续训，本方案（PR #54）以独立新运行起点执行，保留严格身份门禁不放宽。

## 对照与预算

三组均使用同一份 8 条轨迹、相同文件顺序、种子 42、FP32、AdamW（`weight_decay=0.01`）、梯度裁剪阈值 `1.0`、`max_length=2048` 以及相同的 B1、LoRA、评分位置与零初始化打分头。每次前向包含一条完整轨迹，累积组对 8 条轨迹的损失取均值，整组反传完成后执行一次梯度裁剪和参数更新。

| 组别 | `lora_lr` (`--learning-rate`) | `head_lr` (`--head-learning-rate`) | 调度器 | 每次更新累积轨迹数 | 优化器更新次数 | 轨迹呈现次数 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `hidden-lr-low-acc8` | `1e-7` | `1e-7`（共用） | `constant` | 8 | 120 | 960 |
| `hidden-split-cos-acc8` | `2e-4` | `1e-5` | `cosine` | 8 | 120 | 960 |
| `hidden-split-cos-acc1` | `2e-4` | `1e-5` | `cosine` | 1 | 960 | 960 |

第一组保留原低学习率常数基线；第二组检验“分离头/LoRA 学习率 + 余弦退火”组合在 8 轨迹累积下的 8 样本记忆拟合能力（本组验证组合有效性，未在同一批预算内单独拆解分组学习率与余弦退火各自的单因素贡献）；第三组在相同轨迹呈现预算（960 次）及每消费 `8k` 条轨迹后下一次更新起点的余弦相位对齐（`k/120 = 8k/960`）下，对照单轨迹逐步更新（`acc=1, 960 updates`）与整组均值梯度更新（`acc=8, 120 updates`）。比较相同轨迹呈现次数下的整组 MSE 时，使用对应检查点独立评分并同时报告各组已消耗轨迹数、优化器更新次数与裁剪次数。历史 [#49](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/49) 为固定 `vocab_logits` 特征的独立实验；本组结果须以目标设备实跑为准。

## 在训练设备运行

先在当前 checkout 中核对 [原件归档 PR #47](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/47) 的 `prm-evidence-20261124/memory-8/samples.jsonl`，并在 smoke 配置的副本里填写本机的固定 Qwen snapshot 和原 B1 adapter 路径。路径可改，revision 和文件哈希保持配置内的锁定值。每次尝试使用一个新的 `run_root`。

```bash
export PYTHONPATH=src
samples=/path/to/prm-evidence-20261124/memory-8/samples.jsonl
train_ids=results/env-2026-11-17/train_ids.json
template=/path/to/local-qwen25-b1-smoke.json
run_root=artifacts/prm/issue48-hidden-split-cos-v1
mkdir -p "$run_root"

run_arm() {
  local arm="$1" lora_lr="$2" head_lr="$3" scheduler="$4" accumulation="$5" updates="$6"
  local config="$run_root/$arm.json"
  local checkpoint="$run_root/$arm"
  python - "$template" "$config" "$checkpoint" <<'PY' || return
import json
from pathlib import Path
import sys

template, config, checkpoint = sys.argv[1:]
data = json.loads(Path(template).read_text(encoding="utf-8"))
data.update(
    schema_version="prm-qwen-config-v2",
    representation="final_hidden_state",
    max_length=2048,
    output_dir=checkpoint,
)
with Path(config).open("x", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2)
PY
  local extra_args=(--lr-scheduler "$scheduler")
  if [ "$head_lr" != "same" ]; then
    extra_args+=(--head-learning-rate "$head_lr")
  fi
  python -m webshop_lab.prm.train \
    --config "$config" --samples "$samples" --train-ids "$train_ids" \
    --device cuda --seed 42 --steps "$updates" --learning-rate "$lora_lr" \
    --weight-decay 0.01 --gradient-accumulation-steps "$accumulation" \
    "${extra_args[@]}" &&
  python -m webshop_lab.prm.infer \
    --config "$config" --checkpoint "$checkpoint" --samples "$samples" \
    --reference "$checkpoint/reference-predictions.json" \
    --output "$run_root/$arm-reload.json"
}

run_arm hidden-lr-low-acc8 1e-7 same constant 8 120 &&
run_arm hidden-split-cos-acc8 2e-4 1e-5 cosine 8 120 &&
run_arm hidden-split-cos-acc1 2e-4 1e-5 cosine 1 960
```

若在同一干净源码 checkout 上分阶段执行 `cosine` 断点续训（如先跑 48 次更新、再续训 72 次更新至总计 120 次），两段均须显式传入完整总周期 `--cosine-total-steps 120` 并输出至新目录，`--steps` 仅传当前分段增量步数：

```bash
# 第 1 段（0 -> 48 步）
python -m webshop_lab.prm.train \
  --config "$config_part1" --samples "$samples" --train-ids "$train_ids" \
  --device cuda --seed 42 --steps 48 --learning-rate 2e-4 --head-learning-rate 1e-5 \
  --lr-scheduler cosine --cosine-total-steps 120 \
  --weight-decay 0.01 --gradient-accumulation-steps 8

# 第 2 段（48 -> 120 步，同一源码 checkout 恢复）
python -m webshop_lab.prm.train \
  --config "$config_part2" --samples "$samples" --train-ids "$train_ids" \
  --device cuda --seed 42 --steps 72 --learning-rate 2e-4 --head-learning-rate 1e-5 \
  --lr-scheduler cosine --cosine-total-steps 120 \
  --weight-decay 0.01 --gradient-accumulation-steps 8 \
  --resume "$checkpoint_part1"
```

原始 8 条 `samples.jsonl` 的 SHA-256 是 `9c88beffaef277f519712ab4a07c45766212579770d6292572c7f189bca4d8fe`。冻结 `train_ids.json` 的 SHA-256 是 `8c85e56b3ac2e1f0a09b54560ad0f9bf77eedddb241d73617c7ff32416522e20`。

对照每组的 `summary.json`、`reference-predictions.json`、重载预测和 `.attempt.json`。完整保留 8 条后，按 `task_id` 把预测 `total` 与原标签 `terminal_reward` 一一配对，报告 MSE、最大绝对误差、5031/95 两条错误购买预测、裁剪比例和峰值显存。8 条记忆验收沿用 MSE < 0.001 且最大绝对误差 < 0.1。训练通过可进入独立 dev 评估，逐回合归因和 RL 接入仍各有后续验收。

## 上下文保留率

以固定 tokenizer revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` 和训练入口的 `tokenize_trajectory` 重算 911 条完整轨迹。数据来自此前 911 条 B1 样本的 729 fit / 182 dev 划分，审计发生在长度过滤前。购买轨迹也可能得 0 分，下表单独统计步数耗尽的轨迹。

| token 上限 | fit 保留（其中步数耗尽） | dev 保留（其中步数耗尽） | 共保留 | 步数耗尽保留 |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 501（29） | 127（4） | 628 / 911 | 33 / 187 |
| 3072 | 596（53） | 156（17） | 752 / 911 | 70 / 187 |
| 4096 | 630（64） | 159（18） | 789 / 911 | 82 / 187 |

重算输入 SHA-256 分别为 `03a6370bc2048ea0119331aa06969ae6bde18e4e7ba5db3decde108153792500`（`fit1451.training.jsonl`）和 `df1f71a7843a04a654172c853656ece8f0793f1954557d735a8cd5b200c93d76`（`dev1451.training.jsonl`）。最长轨迹 13,388 tokens。当前训练入口执行 2048 上限。长窗口探针需要单独配置并修改该上限，在实际设备分别用接近 3072 和 4096 的真实轨迹做完整前向、反向、裁剪和更新，记录 allocated、reserved 和耗时，再决定全量实验上限。本次测得的是长度保留率，长窗口 GPU 峰值待测。
