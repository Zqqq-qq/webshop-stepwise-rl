# PRM 运行契约（执行侧提供，其他人按此实现）

> 由 GPU 执行方维护。所有 PRM 代码必须能在下面这套环境里直接跑通；**不要按自己电脑的环境适配**。
> 如果你有自己的 GPU，可以自行做 smoke 测试，但**不要按自己的卡改 batch size、精度（如 bf16）或依赖版本**；
> 最终验收（forward/backward、峰值显存、保存与独立重载、正式泛化对比）在下面这张卡上做，由执行方回传结果。

## 主机与环境

| 项 | 值 |
|---|---|
| 宿主 | Windows 11 + WSL2 Ubuntu，仓库在 `/root/webshop-main`，以 root 运行 |
| 环境 | conda `webshop-gpu`，`/root/miniconda3/envs/webshop-gpu/bin/python` |
| Python | 3.10.22 |
| torch | 2.8.0+cu128 |
| transformers | 4.56.2 |
| peft | 0.17.1 |
| bitsandbytes | 0.48.2 |
| accelerate | 1.10.1 |
| numpy / safetensors / tokenizers | 1.26.4 / 0.6.2 / 0.22.2 |
| GPU | RTX 5060 Ti，16,311 MiB，驱动 610.88，CUDA 12.8，compute capability (12, 0) |

**不要引入新依赖**；如确需，先在 PR 里写明版本与理由，由执行方在环境里装。

## 模型与精度

- 模型只用本地快照，不走 hub：
  `/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- 必须设 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；配 `revision=989aa798…` 一致。
- dtype **FP32**，并显式关闭 TF32 与 autocast：
  `torch.backends.cuda.matmul.allow_tf32=False`、`torch.backends.cudnn.allow_tf32=False`、`torch.set_float32_matmul_precision("highest")`。
- 当前 B1 adapter：`/root/webshop-stepwise-rl/artifacts/qwen25-one-epoch-final-resume-Ydr3GfjQ/output/adapter`，`adapter_model.safetensors` SHA `c39e2a78…`。合并冻结或旁挂由 PR 说明，禁止覆盖策略权重复用。

## 数据路径

| 用途 | 路径 |
|---|---|
| 划分 | `/root/webshop-main/data/processed/{train_ids.json,val_ids.json,test_ids.json,split-manifest.json}` |
| goal map | `/root/webshop-main/artifacts/goal-instructions.json` |
| 现有 rollout | `/root/webshop-main/artifacts/fp32-full160-*/<policy>/rollout.jsonl`、`artifacts/rl-full-coverage-*/seed*/…` |

rollout JSONL 顶层键：`run_id, task_id, repeat, instruction, goal_fingerprint, final_reward, done, success, termination, steps, seconds`；`steps` 内：`step, observation, available{has_search_bar,clickables,selected_options}, response, action, grounding, error, environment_reward`。

## 输出与 checkpoint 布局

统一放 `artifacts/prm/<run-name>/`，至少包含：

```
adapter/            PRM LoRA（与策略 adapter 分开保存）
head.pt             评分头权重
config.json         实际运行配置（含模型/初始化身份、dtype、LoRA 超参）
summary.json        训练摘要：每样本 sum(p_t)、标签 R、loss、梯度范数、峰值显存
SHA256SUMS          逐文件哈希
```

必须能从 checkpoint 独立进程重载并复现同一组 p_t。

## 资源与串行约束

- 单卡、**同一时间只跑一个 GPU 任务**；不要并行起多个训练进程。
- 显存目标 ≤16GB；batch size 默认 1，梯度检查点等优化需实测后写进配置。
- 记录 `torch.cuda.max_memory_allocated()` 与实际耗时，写入 `summary.json`。

## 提交前的自测（可在自己的 GPU 或 CPU 上做）

- tokenizer/chat template 结果、assistant spans、turn_end_indices、mask 分离、变长 padding
- 数据集构建、fit/dev 隔离、R 语义（购买完成为任务分，未购买为 0）
- 形状/边界/单样本 sum(p_t) 与标签对齐
- 若自有 GPU：可用小 batch 做一次 forward/backward smoke，但需注明所用卡型/精度，且不得据此调整本契约里的 FP32 设置

## 执行方在本卡上的最终验收

- 本契约规格（FP32、关 TF32/autocast、单卡 16GB）下的 forward/backward、梯度非零、参数变化
- 保存与独立进程重载、逐步预测
- 峰值显存与耗时、正式数据上的泛化对比

## 提交方式

各自独立小分支 / 小 PR，链接回 #43，写明依赖与负责人；不在共享分支互相覆盖。
