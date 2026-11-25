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

## 短训与独立重载

训练入口运行在 #55 的干净 `559a7791b49332375c640bc132af05e079676a83` 源码 checkout，使用原 CUDA venv。另将本 PR checkout 的绝对路径记为 `ranking_repo`。保留原 stage126 配置里的基础模型、分词器与 B1 路径。`stage126_dir` 指原检查点，`stage126_config` 指原配置，`collection_dir` 指真实采集输出。

训练器按候选清单顺序读取环境分差配对。通过真实 Qwen 模板核对分叉前 token 一致性，任一臂超长则整对剔除并登记。两组都使用完整轨迹终局 MSE 与同一对样本、顺序、随机种子和更新预算。排序组额外加入 `0.1 * softplus(score_low - score_high)`，两个 score 均取打开商品回复的 `assistant_im_end`。两组各从同一 stage126 权重重新初始化 AdamW，避免沿用已走完的原余弦调度。首轮使用 12 次更新、每次累积 4 对。

在原 checkout 中首次为两组各做一次真实 GPU 更新，使用下方的 `--smoke`，使用不同且未占用的输出目录。移除该参数并另用两个新目录可运行完整预算。下面的 `output_root` 是已选定的本次产物根目录。

```bash
for arm in terminal ranking; do
  PYTHONPATH=src python "$ranking_repo/scripts/train_prm_product_pairs.py" \
    --archive "$evidence_zip" --plan "$ranking_repo/results/prm-fork-ranking-20261126/train-candidates.json" \
    --protocol "$ranking_repo/results/prm-fork-ranking-20261126/short-train.json" \
    --pairs "$collection_dir/training-pairs.jsonl" --checkpoint "$stage126_dir" --config "$stage126_config" \
    --arm "$arm" --smoke --out "$output_root/$arm" || break
done
```

每个训练目录提供两臂样本与训练完成时的参考分。使用同一个冻结源码独立重载每个检查点。下面展示 ranking 组的 target 样本，另用 `alternative` 文件复核另一臂，terminal 组同样执行。

```bash
PYTHONPATH=src python -m webshop_lab.prm.infer \
  --source-commit 559a7791b49332375c640bc132af05e079676a83 \
  --config "$stage126_config" --checkpoint "$output_root/ranking" \
  --samples "$output_root/ranking/target.samples.jsonl" \
  --reference "$output_root/ranking/target.reference.json" \
  --output "$output_root/ranking-target-reload.json"
```

两份 `summary.json` 的 `before_parameter_sha256`、输入文件哈希、样本与更新预算应一致，且两份模型都应记录参数发生变化。回传采集目录、两组检查点与独立重载输出。Qiu 使用同一批独立 dev 对照两组结果。


## 本地验证

- 从原 #55 归档重建 32 条候选，全部属于 mapped train，和 mapped val/test 任务 ID 的交集均为零。
- 用归档的真实购物指令、页面和动作做模拟 bridge 冒烟，保留同分尝试并在有效分差后生成训练对。独立测试训练端拒收同分记录。
- 用已冻结的 Qwen tokenizer 复核两臂分叉前 token 一致性。终局误差设为 0 时，排序项只推动两个分叉位置，分别提高高分一臂、降低低分一臂。
- 两组都在同初始化的微型 Qwen/LoRA 神经模型上完成一次 CPU 更新，参数发生变化。分别保存评分头与 adapter 后重载，参考预测逐项相同。
- 冻结第 126 步原检查点载荷通过哈希检查，既有 #56 设备报告重新计算一致。项目既有 96 个测试通过。

模拟 bridge 与微型模型验证的是代码路径。真实配对、原始 1.5B 权重的一步 CUDA 冒烟和两组完整短训由 How 回传。
