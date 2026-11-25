# #55 同页选色诊断

固定第 126 步 PRM，重放已查看 dev 的任务 7084。两条轨迹从同一条 B1 记录复用原搜索、原商品和前两轮完整回复。到同一眼影商品页后分别点 `b`、`a`，两边都点 `Buy Now`。选色和购买统一使用 Action 行，控制额外的推理措辞。两条购买的终局分由模拟器分别计算。这个诊断使用已查看任务，后续正式评估仍用独立数据。业务结果与错购分析见 [#55 反馈](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/55#issuecomment-7167924678)。

## 运行

How 在原 WebShop 环境启动同配置的本地 bridge。先在诊断分支运行脚本的绝对路径，传同一个 [#55 原证据 ZIP](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm55-expanded-fit-20261125.zip) 和一个未使用的输出目录。

```bash
python /path/to/scripts/replay_prm55_option_pair.py replay \
  --archive "$evidence_zip" --run-dir "$pair_dir"
```

分别评分时使用真实运行的干净 `559a7791b49332375c640bc132af05e079676a83` checkout、原第 126 步配置和检查点。脚本核对该检查点的 manifest、adapter 和评分头，分别用原推理入口的 `--samples`。系统提示取自已归档的标准转换结果，网页和回复取自实时重放。这个入口不需要旧版本缺失的 `prm/data.py`。

```bash
PYTHONPATH=src python /path/to/scripts/replay_prm55_option_pair.py score \
  --archive "$evidence_zip" --run-dir "$pair_dir" \
  --config "$stage126_config" --checkpoint "$stage126_dir"
```

回传 `environment.json`、两个 rollout、两个预测及 `pair-comparison.json`。先检查环境 B−A 是否为正，再看共同前缀两步是否在容差内，以及选色步和购买步各自的 B−A。总分差与选色步差分别登记。环境同分或反向时记录为无有效规格对照；可以预先指定任务 7402 的椅子颜色作后备，按相同原则重新登记。预留 test 继续冻结。

## 扩展规格与错购对照

使用同一归档和第 126 步检查点。先在真实 bridge 逐臂重置并购买，再对各分叉的固定输入评分。保留实际环境分和各步预测，分叉动作统一用 `Action:` 行，共享动作保留原回复。

| 任务 | 分叉步 | 对照 |
| --- | ---: | --- |
| 7402 | 3 | 椅子颜色 dusty pink / black |
| 7113 | 4 | 选蓝色后，尺码 3x-large / large |
| 7489 | 3 | 钥匙型号 yubikey 5c / yubikey 5c nano |
| 7544 | 2 | 同一结果页的跑鞋 / 已归档零分的洞洞鞋 |

7544 先完整重放错购路径。对同页九个跑鞋候选按展示顺序实买，商品页有预先列出的 8 号尺码就选该尺码，再购买。保存全部候选和实际动作。按模拟器终局分选一个最高的跑鞋候选，并列取原页面顺序第一项。只有终局满分才能称为满足全部条件。不同商品的尺码按钮和购买页可能不同，主要比较输入页面完全相同的打开商品那一步。

```bash
python /path/to/scripts/replay_prm55_expanded_pairs.py replay \
  --archive "$evidence_zip" --run-dir "$expanded_pair_dir"
```

从上文的固定源码 checkout 运行，只读原检查点。执行前清除临时注入的 `sitecustomize.py` 路径。这里用冻结推理已经支持的 `--samples` 输入，样本仅含任务 ID 和完整消息，环境终局分独立保存。

```bash
PYTHONPATH=src python /path/to/scripts/replay_prm55_expanded_pairs.py score \
  --archive "$evidence_zip" --run-dir "$expanded_pair_dir" \
  --config "$stage126_config" --checkpoint "$stage126_dir"
```

回传输出目录原件，至少含 `environment.json`、全部候选、每对 rollout、评分输入、预测输出和 `pair-comparison.json`。环境同分的对照保留原始预测差，方向字段记作空。环境有分差时，方向按实际环境差计算，模型差小于等于 0.00001 记作未分出方向。单独登记购买和总预测差。诊断仅使用已查看的 dev，预留 test 保持冻结。
