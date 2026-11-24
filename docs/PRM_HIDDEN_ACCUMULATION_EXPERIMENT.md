# Issue #48 的 hidden PRM 梯度累积对照

以 [PR #50](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/50) 的候选实现为起点。使用 B1 初始化的 Qwen2.5-1.5B-Instruct，在每个助手回合的 `<|im_end|>` 位置读取最终归一化后的 1536 维 hidden state，经线性头得到各回合分数。沿用轨迹总分拟合终局标签的 MSE。先测 8 条真实轨迹能否拟合，再由 Issue #48 的独立 dev 流程评估泛化。

## 对照与预算

三组都使用同一份 8 条轨迹、相同文件顺序、种子 42、FP32、AdamW（weight decay 0.01）、梯度裁剪阈值 1.0、`max_length=2048` 和相同的 B1、LoRA、评分位置与打分头初始化。每次前向包含一条完整轨迹，累积组对 8 条轨迹的损失取均值，整组反传完成后做一次裁剪和更新。三组各从头启动，固定学习率。

| 组别 | 学习率 | 每次更新累积轨迹数 | 优化器更新次数 | 轨迹呈现次数 |
| --- | ---: | ---: | ---: | ---: |
| hidden-lr-low-acc8 | `1e-7` | 8 | 120 | 960 |
| hidden-lr-high-acc8 | `3e-4` | 8 | 120 | 960 |
| hidden-lr-high-acc1 | `3e-4` | 1 | 960 | 960 |

前两组的差异是学习率。后两组在相同轨迹呈现次数下比较两种更新方式，优化器更新次数和裁剪频率随累积设置变化。比较相同更新次数下的整组 MSE 时，使用该步的独立检查点评分，并报告各组已用的轨迹数。历史 [#49](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/49) 使用的是固定 vocab logits 的独立实验。截图中 hidden 组报告的误差需要按本实验的模型、数据和训练入口实测。

## 在训练设备运行

先在当前 checkout 中核对 [原件归档 PR #47](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/47) 的 `prm-evidence-20261124/memory-8/samples.jsonl`，并在 smoke 配置的副本里填写本机的固定 Qwen snapshot 和原 B1 adapter 路径。路径可改，revision 和文件哈希保持配置内的锁定值。每次尝试使用一个新的 `run_root`。

```bash
export PYTHONPATH=src
samples=/path/to/prm-evidence-20261124/memory-8/samples.jsonl
train_ids=results/env-2026-11-17/train_ids.json
template=/path/to/local-qwen25-b1-smoke.json
run_root=artifacts/prm/issue48-hidden-accum-v1
mkdir -p "$run_root"

run_arm() {
  local arm="$1" lr="$2" accumulation="$3" updates="$4"
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
  python -m webshop_lab.prm.train \
    --config "$config" --samples "$samples" --train-ids "$train_ids" \
    --device cuda --seed 42 --steps "$updates" --learning-rate "$lr" \
    --weight-decay 0.01 --gradient-accumulation-steps "$accumulation" &&
  python -m webshop_lab.prm.infer \
    --config "$config" --checkpoint "$checkpoint" --samples "$samples" \
    --reference "$checkpoint/reference-predictions.json" \
    --output "$run_root/$arm-reload.json"
}

run_arm hidden-lr-low-acc8 1e-7 8 120 &&
run_arm hidden-lr-high-acc8 3e-4 8 120 &&
run_arm hidden-lr-high-acc1 3e-4 1 960
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
