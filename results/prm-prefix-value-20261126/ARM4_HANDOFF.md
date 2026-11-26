# PR #59 最优组（Arm 4: `arm4_pv_asin_context_fit501`）重载、评分与样例交接单

本文档向 @HowseaonLink（GPU 采集与评分）与 @Yuxiao-Liu233（实现复核）交接 PR #59 最优组 **`arm4_pv_asin_context_fit501`** 的固定提交、检查点载荷、本地配置、跨机重载与新 `dev` 评分命令，以及标明点击前搜索页字符级来源的 `7544` 双臂真实样例。

---

## 一、固定 Commit 与源码检出说明

1. **评分脚本与交接归档分支（本 PR #59）**：
   - 分支：`Zenon-nnnnnn:experiment/prm-prefix-value-telescoping-20261126`
   - 独立重载与评分入口：[scripts/score_prefix_value_prm.py](../../scripts/score_prefix_value_prm.py)（与 `python -m webshop_lab.prm.infer` 命令行参数及输出 JSON 结构兼容，不读取任何标签）
2. **Runtime 严格身份检出（供 `verify_prm` 校验 `source_commit`）**：
   - `arm4_pv_asin_context_fit501/prm-manifest.json` 绑定的执行 `source_commit` 为 **`d25e6164894c87fdea796bb73e3ee546807483d8`**（Git tree `a6da83c56a0ced0cbb6081c94f5eaf0f0bf305f6`，即 @HowseaonLink 已在 RTX 5060 Ti 上从 `pr54-executed.bundle` 克隆并验证通过的干净检出 `evidence/pr54-executed`）。
   - 若需在新目录重新拉取该 runtime 检出：
     ```bash
     git fetch https://github.com/Zenon-nnnnnn/webshop-stepwise-rl.git refs/heads/evidence/pr54-executed:refs/heads/evidence/pr54-executed
     ```
   - 运行 `score_prefix_value_prm.py` 时，将 `PYTHONPATH` 指向该干净 `d25e6164` 检出的 `src` 目录，即可在不改动原 manifest、不放宽身份门禁的前提下 100% 通过 `verify_prm`。

---

## 二、Checkpoint 受控获取方式与精确 SHA-256

- **Fork Release 下载直链**：
  https://github.com/Zenon-nnnnnn/webshop-stepwise-rl/releases/download/prm-prefix-value-20261126/prefix-value-prm-rtx3090-20261126.zip
- **外层 ZIP 大小与 SHA-256**：`141,085,800` bytes，`3b53a6615d217f8c97b285a9a148bfa686762b81bcc9aa8e371684b8c6f0b43f`
- **解压与全量哈希核验**：
  ```bash
  sha256sum prefix-value-prm-rtx3090-20261126.zip
  # 预期：3b53a6615d217f8c97b285a9a148bfa686762b81bcc9aa8e371684b8c6f0b43f
  unzip -q prefix-value-prm-rtx3090-20261126.zip -d /path/to/pv-bundle
  (cd /path/to/pv-bundle && sha256sum -c SHA256SUMS)
  ```
- **`arm4_pv_asin_context_fit501/` 检查点核心载荷 SHA-256**：
  | 文件相对路径 | SHA-256 |
  | :--- | :--- |
  | `prm-manifest.json` | `826f9cce3a51365ec6bf02fb10574ca4990010a8927e9082b5a0383fd028b01c` |
  | `config.json` | `e2d52dcb504300d62ed7dab24e2e0268faa1d144b4dbe399881147e6b96c1a8e` |
  | `head.pt` | `a234fd6d1a4d89a2214064f2e17ac7f85896126f7fa018df18eec8b5afe2a23e` |
  | `adapter/adapter_model.safetensors` | `fc78e9c77c6d0722bc2926b337ce06f155d1f9a94d954e7778541f5d0027bf92` |
  | `adapter/adapter_config.json` | `f0605ffdf903dd4a09e37da5c728cac0e1005fed74d2852772f9490d5954fbb0` |
  | `adapter/README.md` | `4532e7c3696f9b295fee1d451a1db5caad7bbcb8539a64bfac521d24c091a132` |

---

## 三、本地配置适配（`arm4.local.config.json`）

以检查点内的 [config.json](rtx3090-five-arms/arm4_pv_asin_context_fit501/config.json) 为模板复制一份到检查点目录外（**不要修改检查点目录内的原文件**），仅将 `tokenizer`、`sft_adapter`、`output_dir` 改为评估机本地路径；`model` 保持 `"/root/autodl-tmp/qwen25-15b"`（复用 @HowseaonLink 在 PR #54 重载时已建好的指向本机固定 Qwen snapshot `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` 的软链接，以匹配 `auxiliary_input_sha256` 路径键）：

```json
{
  "model": "/root/autodl-tmp/qwen25-15b",
  "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
  "base_weights_sha256": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
  "tokenizer": "/path/to/local/Qwen2.5-1.5B-Instruct-989aa798",
  "tokenizer_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
  "tokenizer_files_sha256": {
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"
  },
  "output_dir": "/path/to/unused-infer-placeholder",
  "initialization": "sft_b1",
  "sft_adapter": "/path/to/local/b1-adapter",
  "sft_adapter_files_sha256": {
    "adapter_model.safetensors": "c39e2a78e7c5973328e9d2c23fb6320e566352e63cc089562488f3dd46ae5f9c",
    "adapter_config.json": "18993c388788ec14de72fd4f6f9ff86ce761ffc3aad5d0923b33f4365be03d97"
  },
  "policy_dtype": "float32",
  "load_in_4bit": false,
  "max_length": 2048,
  "lora_r": 8,
  "lora_alpha": 16,
  "target_modules": [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
  ],
  "schema_version": "prm-qwen-config-v2",
  "score_position": "assistant_im_end",
  "representation": "final_hidden_state"
}
```
可直接通过 `jq` 从检查点内 `config.json` 一键生成本机配置（避免手改哈希）：
```bash
jq --arg tok "/path/to/local/Qwen2.5-1.5B-Instruct-989aa798" \
   --arg b1  "/path/to/local/b1-adapter" \
   --arg out "/path/to/unused-infer-placeholder" \
   '.tokenizer = $tok | .sft_adapter = $b1 | .output_dir = $out' \
   "$ARM4_CKPT/config.json" > /path/to/arm4.local.config.json
```

---

## 四、跨机重载核验与新 `dev` 评分命令

设：
- `EXEC_REPO` 为干净的 `d25e6164894c87fdea796bb73e3ee546807483d8` runtime 检出目录；
- `PR59_REPO` 为本 PR #59 检出目录；
- `ARM4_CKPT` 为解压出的 `/path/to/pv-bundle/arm4_pv_asin_context_fit501`；
- `ARM4_CFG` 为按第三节调整好本机路径的 `arm4.local.config.json`。

### 1. 先跑 `7544` 双臂 `--reference` 重载一致性核验
传入与 `stage126` **完全同一份原始 `regression7544/{target,alternative}.samples.jsonl`**（SHA-256 分别为 `867ddfcd6a83956e35a136c489cef29bd048e0daed6a2f6fecf57c524e7dff9a` 与 `42d578f15dad36775c2f7b28aa042f1ca82bc6cf1d56a267bc0e67a578bdaa0b`）：

```bash
export PYTHONPATH="$EXEC_REPO/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for side in target alternative; do
  python "$PR59_REPO/scripts/score_prefix_value_prm.py" \
    --source-commit d25e6164894c87fdea796bb73e3ee546807483d8 \
    --config "$ARM4_CFG" \
    --checkpoint "$ARM4_CKPT" \
    --samples "/path/to/regression7544/$side.samples.jsonl" \
    --reference "$PR59_REPO/results/prm-prefix-value-20261126/references/arm4-r7544-$side.reference.json" \
    --output "/path/to/out/arm4-r7544-$side.json" \
    --device cuda
done
```
输出 JSON 中 `"reference_match": true` 即确认跨机权重、tokenizer、模板与望远镜差分计算完全对齐。

### 2. 对新冻结 `dev` 分叉（及完整轨迹）评分（与 `stage126` 使用完全同一批原始输入文件）
`stage126` 照常调用 `python -m webshop_lab.prm.infer`；`Arm 4` 直接读取**同一份未修改的原始 `samples.jsonl`（或 `--rollout`）**：

```bash
# 对两臂选品分叉评分（target / alternative）
for side in target alternative; do
  python "$PR59_REPO/scripts/score_prefix_value_prm.py" \
    --source-commit d25e6164894c87fdea796bb73e3ee546807483d8 \
    --config "$ARM4_CFG" \
    --checkpoint "$ARM4_CKPT" \
    --samples "/path/to/new_dev_pairs/$side.samples.jsonl" \
    --output "/path/to/out/arm4-$side.json" \
    --device cuda
done

# 对新 dev 单轨迹集合评分（评估整条总分 MSE）
python "$PR59_REPO/scripts/score_prefix_value_prm.py" \
  --source-commit d25e6164894c87fdea796bb73e3ee546807483d8 \
  --config "$ARM4_CFG" \
  --checkpoint "$ARM4_CKPT" \
  --samples "/path/to/new_dev/samples.jsonl" \
  --output "/path/to/out/arm4-new-dev-total.json" \
  --device cuda
```
输出格式与 `webshop_lab.prm.infer` 完全一致（`predictions[i]["step_scores"][branch]` 即选品打开商品步得分，`predictions[i]["total"]` 即整条轨迹总分预测），且每条预测额外附带 `v_prefix` 与 `asin_context_provenance` 供逐题审计。

- **超长与有效分母守恒说明（对齐 `PR59_REVERIFY_DEV_PROTOCOL.md`）**：`encode_trajectory_for_pv` 先对未改动的原始 `messages` 检查 `max_length = 2048`；若原始轨迹 `≤ 2048` 但拼接 `[Item ...]` 片段后超出 `2048`，自动回退至原始 `base_enc`（并在 `asin_context_provenance` 中标记 `fallback_to_raw_due_to_max_length: true`）；仅当原始轨迹本身 `> 2048` 时才记入 `rejected`。因此 `Arm 4` 与 `stage126` 的 `too_long` 剔除集合与保留有效分母 **100% 严格一致**。

---

## 五、真实样例溯源（`task_id = 7544`，供 @Yuxiao-Liu233 与 @HowseaonLink 核对）

`stage126` 与 `Arm 4` 读取**完全相同的原始输入文件**：
- `target`（满分跑鞋，环境终局分 `1.0`）：`regression7544/target.samples.jsonl`（SHA-256: `867ddfcd6a83956e35a136c489cef29bd048e0daed6a2f6fecf57c524e7dff9a`）
- `alternative`（零分洞洞鞋，环境终局分 `0.0`）：`regression7544/alternative.samples.jsonl`（SHA-256: `42d578f15dad36775c2f7b28aa042f1ca82bc6cf1d56a267bc0e67a578bdaa0b`）

### 1. 原始 `messages` 结构与点击前搜索页来源定位
两条轨迹均含 9 条消息 `messages[0..8]`（共 4 个助手回合 `turn_index = 0, 1, 2, 3`），且 **`messages[0..3]` 逐字节完全相同**：
- `messages[0]` (`system`)：系统提示词
- `messages[1]` (`user`)：初始指令页（`"WebShop [SEP] Instruction: [SEP] ethylene vinyl women's running shoes also choose size 8, and price lower than 70.00 dollars [SEP] Search"`）
- `messages[2]` (`assistant`, `turn_index = 0`)：搜索动作（`Action: search[ethylene vinyl women's running shoes size 8]`）
- **`messages[3]` (`user`) —— 点击前的搜索结果页记录（唯一信息来源！）**：
  - **洞洞鞋记录位于 `messages[3]["content"][178:287]`**：
    ```text
    [SEP] B092F97BW4 [SEP] Nordways Unisex-Adult Clogs Health Care Professional Shoe [SEP] $47.51 to $64.46 [SEP]
    ```
  - **跑鞋记录位于 `messages[3]["content"][282:445]`**：
    ```text
    [SEP] B09KGMGVV9 [SEP] InterestPrint Women's Running Shoes Lightweight Non-Slip Breathable Walking Shoes Gingerbread Men and Stockings [SEP] $41.99 to $43.99 [SEP]
    ```
- **`messages[4]` (`assistant`, `turn_index = 1`) —— 打开商品回复（选品分叉步）**：
  - `target` 原始文本：`'Action: click[b09kgmgvv9]'`
    - `extract_preclick_asin_snippet(messages, msg_idx=4)` **仅读取 `messages[3]`（即 `msg_idx - 1`）** 的字符区间 `[282, 445]`，提取 `title = "InterestPrint Women's Running Shoes Lightweight Non-Slip Breathable Walking Shoes Gingerbread Men and Stockings"`、`price = "$41.99 to $43.99"`，构造 Arm 4 评分输入：
      ```text
      Action: click[b09kgmgvv9] [Item B09KGMGVV9: InterestPrint Women's Running Shoes Lightweight Non-Slip Breathable Walking Shoes Gingerbread Men and Stockings | $41.99 to $43.99]
      ```
  - `alternative` 原始文本：`'Action: click[b092f97bw4]'`
    - 同样**仅读取 `messages[3]`** 的字符区间 `[178, 287]`，提取 `title = "Nordways Unisex-Adult Clogs Health Care Professional Shoe"`、`price = "$47.51 to $64.46"`，构造 Arm 4 评分输入：
      ```text
      Action: click[b092f97bw4] [Item B092F97BW4: Nordways Unisex-Adult Clogs Health Care Professional Shoe | $47.51 to $64.46]
      ```
- `messages[5..8]`：点击后才出现的商品详情页、选尺码回复与 `Action: click[buy now]`（因果注意力下对 `turn_index = 1` 严格不可见，且正则提取器从不访问 `msg_idx > 4`）。

### 2. 评分位置对齐与逐步预测对比

| 模型与分支 | 各助手回合结束评分位置 `turn_end_indices`（全位置解码均为 `<\|im_end\|>`） | 打开商品步位置 `turn_end_indices[1]` | 前缀价值 $\hat{V}_0, \hat{V}_1, \hat{V}_2, \hat{V}_3$ | 逐步得分 `step_scores` (`[turn0, turn1_click, turn2, turn3]`) | 打开商品步得分 `step_scores[1]` | 打开商品步分差 (`target - alt`) |
| :--- | :--- | :---: | :--- | :--- | ---: | ---: |
| `stage126` (`target` 跑鞋) | `[173, 750, 1058, 1367]` | `750` (`<\|im_end\|>`) | — | `[+0.2812, +0.2479, +0.1451, +0.0913]` | `+0.247854` | **`-0.003439`**（排反） |
| `stage126` (`alt` 洞洞鞋) | `[173, 752, 1010, 1262]` | `752` (`<\|im_end\|>`) | — | `[+0.2812, +0.2513, +0.1115, +0.0641]` | `+0.251293` | ↑ |
| **`Arm 4` (`target` 跑鞋)** | `[173, 796, 1104, 1413]` | **`796` (`<\|im_end\|>`)** | `[0.7643, 0.7478, 0.7799, 0.7146]` | `[+0.7643, -0.0165, +0.0321, -0.0652]` | **`-0.016548`** | **`+0.115160`**（正确） |
| **`Arm 4` (`alt` 洞洞鞋)** | `[173, 791, 1049, 1301]` | **`791` (`<\|im_end\|>`)** | `[0.7643, 0.6326, 0.6984, 0.7182]` | `[+0.7643, -0.1317, +0.0658, +0.0198]` | **`-0.131708`** | ↑ |
