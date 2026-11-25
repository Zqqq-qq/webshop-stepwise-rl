# PR #54 证据归档：分离评分头/LoRA 学习率与余弦退火三组对照（2026-11-24）

本目录归档 PR #54（训练实现提交 `07bcdcbd75fb289272c2054702afa32059e3c98e`）对应的全部非权重原始 JSON 证据、独立新进程重载报告、初始本地 5 步短测原始日志以及离线校验与指标复算脚本 [recompute.py](recompute.py)；三臂完整二进制检查点压缩包已同步发布至 Fork Release（见第三节第 3 点）。

> **范围与边界声明**：
> 1. `rtx3090-fp32-three-arms/` 是在独立 NVIDIA RTX 3090（24GB）上按 `webshop_lab.prm.train` 与 `webshop_lab.prm.infer` 完整执行的三组 960 次轨迹呈现对照，**与 PR #52 在目标 RTX 5060 Ti 上执行的单一恒定 `1e-5` 续训链完全独立、互不混写**；本记录仅作为该 8 条训练轨迹上的拟合证据，不替代目标 RTX 5060 Ti 实机复核，也不代表独立 `dev` / `test` 泛化、逐回合信用分配或下游 PPO 效果已通过，`test_authorization=false`。
> 2. `local-mac-5step-probe.log` 为首次提交 PR #54 草稿时用于验证首步无超调及第 3–5 步 LoRA 介入无二次发散的 Apple M2 Pro 本地 CPU 5 步探针原始输出（采用 `bfloat16` 常驻权重 + 逐层 `float32` 算子计算，非正式 CLI 产物）；在配齐独立 CUDA 环境后，8 条记忆门槛的完整验证以 `rtx3090-fp32-three-arms/` 三组标准 FP32 CLI 原件及二进制检查点压缩包为准。

---

## 一、 一键离线校验与指标复算

在仓库根目录直接运行（默认仅依赖 Python 标准库，无需 GPU 或模型权重）：

```bash
# 1. 零依赖离线校验 JSON 清单、单侧评分位置合法性、退火曲线与 8 条指标复算
python3 results/prm-split-cosine-20261124/recompute.py

# 2. 同时核对原始 8 条 samples.jsonl 原件（SHA-256: 9c88beff...）与每条消息结构
python3 results/prm-split-cosine-20261124/recompute.py \
  --samples /path/to/samples.jsonl

# 3. 进一步使用本地锁定的 Qwen2.5-1.5B-Instruct tokenizer 重建轨迹并逐位置对齐助手回合末 `<|im_end|>`
python3 results/prm-split-cosine-20261124/recompute.py \
  --samples /path/to/samples.jsonl \
  --tokenizer /path/to/Qwen2.5-1.5B-Instruct-989aa7980e4cf806f80c7fef2b1adb7bc71aa306

# 4. 同时核验 Release 二进制压缩包内全部 9 个 .safetensors / .pt 载荷 SHA-256 与清单逐字节一致
python3 results/prm-split-cosine-20261124/recompute.py \
  --bundle /path/to/prm-split-cosine-rtx3090-20261124.zip
```

`recompute.py` 自动执行以下六项核验：
1. 校验仓库 `src/webshop_lab/prm/` 核心文件 SHA-256 与三组 `prm-manifest.json` 记录完全一致（且 `train.py` SHA-256 为 `d71a30626cb4a747837a09a9e3814abc4bcccb2fbad2d2da6fe98febf4ab0ac8`，与提交 `07bcdcbd` 逐字节相等）；
2. 校验三组检查点目录内所有跟踪文件（`summary.json`、`reference-predictions.json`、`config.json`、`adapter/adapter_config.json`、`adapter/README.md`）及 `prm-manifest.json` 自身与 `SHA256SUMS` 哈希逐字节匹配；
3. **双侧独立校验评分位置与重载一致性**：对 `reference-predictions.json` 与独立新进程 `*-reload.json` **各自独立**检验 `turn_end_indices` 非空、长度与原轨迹预期助手回合数一一对应、全部为 `< max_length` 的非负整数（拒绝 `bool` 或越界索引）且严格递增（修复双侧同缺/同错误放行问题），可选通过固定 tokenizer 重建并核对每个索引确为助手回合末尾 `<|im_end|>`；再比对两侧 `step_scores` 与 `total`（实测三组最大逐步与总分重载差均为 `0.0`）；
4. 校验每组 `summary.json` 中全部优化步（120 / 120 / 960 步）记录的 `learning_rate` 与 `head_learning_rate` 严格符合 `constant` 或 `cosine` 公式；
5. 按原 8 条轨迹标签独立复算 `MSE`、最大绝对误差、3 回合满分组均值（`0, 6225`）与 3 回合零分组均值（`95, 5031`）；
6. 当传入 `--bundle` 时，流式计算压缩包整体 SHA-256 及包内全部 40 个文件（含三臂各自的 `adapter/adapter_model.safetensors`、`head.pt`、`training-state.pt`）SHA-256，与仓库内跟踪的 `prm-manifest.json` 及 `SHA256SUMS` 逐项交叉核验。

---

## 二、 执行身份、源码同步说明与实际命令

- **硬件与软件环境**：NVIDIA GeForce RTX 3090（24576 MiB，驱动 `580.105.08`），Python `3.12.3`、`torch==2.5.1+cu124`（CUDA `12.4`）、`transformers==4.56.2`、`peft==0.17.1`、`safetensors==0.6.2`、`numpy==1.26.4`。
- **后端设置**（见各组 `summary.json` 的 `backend` 字段）：`float32_matmul_precision="highest"`、`cuda_matmul_allow_tf32=false`、`cudnn_allow_tf32=false`、`autocast_enabled=false`，全模型参数 `torch.float32`。
- **源码身份说明**：执行机通过 `git archive 07bcdcbd75fb289272c2054702afa32059e3c98e` 离线同步并在机内执行 `git init && git commit` 以满足工作区干净检查，因此各组 `prm-manifest.json` 中记录的本地根提交号为 `d25e6164894c87fdea796bb73e3ee546807483d8`（`source_tracked_clean=true`）；该检出的 Git tree 对象（`a6da83c56a0ced0cbb6081c94f5eaf0f0bf305f6`）及全部源码文件 SHA-256 与提交 `07bcdcbd` 100% 相同。
- **锁定输入 SHA-256**：
  - `base_weights_sha256`: `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`（revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`）
  - `sft_adapter` (`adapter_model.safetensors`): `c39e2a78e7c5973328e9d2c23fb6320e566352e63cc089562488f3dd46ae5f9c`
  - `sft_adapter` (`adapter_config.json`): `18993c388788ec14de72fd4f6f9ff86ce761ffc3aad5d0923b33f4365be03d97`
  - `samples.jsonl` (8 条训练轨迹): `9c88beffaef277f519712ab4a07c45766212579770d6292572c7f189bca4d8fe`
  - `results/env-2026-11-17/train_ids.json`: `8c85e56b3ac2e1f0a09b54560ad0f9bf77eedddb241d73617c7ff32416522e20`
- **实际执行命令**：
  ```bash
  # Arm 1: hidden-lr-low-acc8
  python -m webshop_lab.prm.train \
    --config /root/autodl-tmp/runs/issue48-verify-v1/hidden-lr-low-acc8.json \
    --samples /root/autodl-tmp/inputs/samples.jsonl \
    --train-ids results/env-2026-11-17/train_ids.json \
    --device cuda --seed 42 --steps 120 \
    --learning-rate 1e-7 --lr-scheduler constant \
    --weight-decay 0.01 --gradient-accumulation-steps 8
  python -m webshop_lab.prm.infer \
    --config /root/autodl-tmp/runs/issue48-verify-v1/hidden-lr-low-acc8.json \
    --checkpoint /root/autodl-tmp/runs/issue48-verify-v1/hidden-lr-low-acc8 \
    --samples /root/autodl-tmp/inputs/samples.jsonl \
    --reference /root/autodl-tmp/runs/issue48-verify-v1/hidden-lr-low-acc8/reference-predictions.json \
    --output /root/autodl-tmp/runs/issue48-verify-v1/hidden-lr-low-acc8-reload.json

  # Arm 2: hidden-split-cos-acc8
  python -m webshop_lab.prm.train \
    --config /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc8.json \
    --samples /root/autodl-tmp/inputs/samples.jsonl \
    --train-ids results/env-2026-11-17/train_ids.json \
    --device cuda --seed 42 --steps 120 \
    --learning-rate 2e-4 --head-learning-rate 1e-5 --lr-scheduler cosine \
    --weight-decay 0.01 --gradient-accumulation-steps 8
  python -m webshop_lab.prm.infer \
    --config /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc8.json \
    --checkpoint /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc8 \
    --samples /root/autodl-tmp/inputs/samples.jsonl \
    --reference /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc8/reference-predictions.json \
    --output /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc8-reload.json

  # Arm 3: hidden-split-cos-acc1
  python -m webshop_lab.prm.train \
    --config /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc1.json \
    --samples /root/autodl-tmp/inputs/samples.jsonl \
    --train-ids results/env-2026-11-17/train_ids.json \
    --device cuda --seed 42 --steps 960 \
    --learning-rate 2e-4 --head-learning-rate 1e-5 --lr-scheduler cosine \
    --weight-decay 0.01 --gradient-accumulation-steps 1
  python -m webshop_lab.prm.infer \
    --config /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc1.json \
    --checkpoint /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc1 \
    --samples /root/autodl-tmp/inputs/samples.jsonl \
    --reference /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc1/reference-predictions.json \
    --output /root/autodl-tmp/runs/issue48-verify-v1/hidden-split-cos-acc1-reload.json
  ```

---

## 三、 对照关系、进度对齐口径与二进制载荷归档

### 1. 同设备三臂与跨记录参考分开陈述

| 记录 | 设备与来源 | LR 与调度器 | `acc` / `optimizer_steps` / `exposures` | `clipped_updates` | 8 条训练 `MSE` | `max_abs_err` | 3 回合满分均值 (`0,6225`) / 零分均值 (`95,5031`) |
| :--- | :--- | :--- | :---: | :---: | ---: | ---: | :---: |
| `#54 hidden-lr-low-acc8` | 独立 RTX 3090 / #54 | shared `1e-7`, `constant` | `8 / 120 / 960` | `120 / 120` | `0.303530243` | `0.873627` | `0.126574 / 0.126795` |
| `#52 共同 LR 历史参考` | RTX 5060 Ti / #52（跨设备独立续训链） | shared `1e-5`, `constant` | `8 / 120 / 960` | — | `0.002859789` | `0.094438` | — |
| `#54 hidden-split-cos-acc8` | 同一 RTX 3090 / #54 | LoRA `2e-4`, head `1e-5`, `cosine` | `8 / 120 / 960` | `114 / 120` | `0.000000715` (`7.15e-7`) | `0.001441` | `0.999441 / 0.000557` |
| `#54 hidden-split-cos-acc1` | 同一 RTX 3090 / #54 | LoRA `2e-4`, head `1e-5`, `cosine` | `1 / 960 / 960` | `818 / 960` | `0.000000173` (`1.73e-7`) | `0.000763` | `0.999517 / 0.000236` |

- **证据归因边界**：`#54` 三臂均在同一台 RTX 3090、同一源码快照与锁定输入下独立运行，支持“在同 8 条训练轨迹、相同 960 次样本呈现预算下，`split + cosine` 组合显著优于同机 `1e-7` 恒定基线，且两支 `split + cosine` 臂的归档预测均通过 8 条记忆门槛（`MSE < 0.001` 且 `max_abs_err < 0.1`）”。
- **跨设备与单因素边界**：`#52` 的 `shared 1e-5 + constant` 来自另一台 GPU（RTX 5060 Ti）及独立源码/续训链，仅列作跨记录背景参考，不并入同设备严格消融；同时，当前 `split + cosine` 组合一并改变了头/LoRA 学习率比例与退火调度器，**未单独隔离“分组学习率”与“余弦退火”各自的独立因果效应**（若后续需拆解单因素贡献，须在同设备同起点补充 `shared 1e-5 + constant`、`split (2e-4/1e-5) + constant` 等控制臂）。

### 2. `acc=8 / 120 更新` 与 `acc=1 / 960 更新`：曝光对齐 ≠ 优化步等价

- **样本呈现预算对齐，更新与裁剪频次不同**：两臂总样本呈现量均为 `exposures = 960`（8 条轨迹各 120 轮），但 `acc=8` 仅执行 `120` 次 AdamW 更新、每 8 条均值梯度裁剪一次（裁剪 `114 / 120`），而 `acc=1` 执行 `960` 次单轨迹梯度更新与裁剪（裁剪 `818 / 960`）。因此两者比较的是**相同样本曝光预算下的收敛表现，而非相同优化步数对照**，末端 `MSE` 差值不单独归因于累积批大小本身。
- **下一次更新起点处的 schedule phase 对齐**：按 `cosine_lr_multiplier(step_index, total_steps)`，`acc=8` 的 `total_steps = 120`，`acc=1` 的 `total_steps = 960`。在累计消费完 `8k` 条样本（`k = 0, 1, ..., 119`）后、**即将开始下一轮首次更新前**，`acc=8` 与 `acc=1` 的归一化余弦相位分别为 `k / 120` 与 `8k / 960 = k / 120`（即**下次更新前的 schedule phase 对齐**）；但刚完成上一轮末次更新时实际使用的相位分别为 `(k - 1) / 120` 与 `(8k - 1) / 960`（并不严格相等），且 `acc=1` 在每轮内部已额外连续更新 7 次、期间学习率逐步变化。

### 3. 二进制权重与状态载荷公开归档（GitHub Release）

按仓库 `.gitignore` 与 `CONTRIBUTING.md` 规范，二进制 `.safetensors` / `.pt` 不纳入 Git 提交历史；三臂完整检查点（含二进制权重、优化器/RNG 状态及全部清单原件）已打包发布至公开 Release 供独立核验：

- **Release 页面**：https://github.com/Zenon-nnnnnn/webshop-stepwise-rl/releases/tag/prm-evidence-pr54-20261124
- **附件直链**：https://github.com/Zenon-nnnnnn/webshop-stepwise-rl/releases/download/prm-evidence-pr54-20261124/prm-split-cosine-rtx3090-20261124.zip
- **字节大小**：`36,760,028` bytes（`40` 个文件，含顶层 `SHA256SUMS`）
- **外层压缩包 SHA-256**：`6ffeaba47f6586e54efd3ab13fef9890d2d3bd1bad6386bbe597aa8b349bb6f5`

| 组别 | `adapter/adapter_model.safetensors` | `head.pt` | `training-state.pt` |
| :--- | :--- | :--- | :--- |
| `hidden-lr-low-acc8` | `0bdba64bf866d57fd1e02f551614ad614fef1f6dd9f4c5650dca0e6ba56cde65` | `10e1eeabe92cba453745674bfea003843f28254ba59eb8052937d594d0cabc77` | `89f8b0ab5ebe8398f96c675a72f429f9183ff713a2e8ffd1d191ab5ee58209b4` |
| `hidden-split-cos-acc8` | `dc21e3967050993b6f1f7439f3e426af2bfc1e4e651477d099c22247f9f8391a` | `1c9d7169faa4f08b929907fb952e6ee05e3e0e63c7682d2e54b1061ff3b28019` | `3d2ffb9757e9029e9ccadb138756a13c9ede1ae69cfea0582d7fbd541e4dcb77` |
| `hidden-split-cos-acc1` | `371421c47f8aa2300c061e699aef3d135387be10a45b84edf4545f8e0dd15484` | `9c4189e1c31600eb7ddbe9a6049e834db417ff2b3fbc66fbdb7d75ec37ba26ed` | `8b722c564dc8a0908007c6d5e086bc34e0919ebe6cd65fc481611b052ed1036b` |
