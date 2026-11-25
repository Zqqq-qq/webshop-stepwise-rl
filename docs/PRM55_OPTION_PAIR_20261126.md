# #55 同页选色诊断

固定第 126 步 PRM，重放已查看 dev 的任务 7084。两条轨迹从同一条 B1 记录复用原搜索、原商品和前两轮完整回复。到同一眼影商品页后分别点 `b`、`a`，两边都点 `Buy Now`。选色和购买统一使用 Action 行，控制额外的推理措辞。两条购买的终局分由模拟器分别计算。这个诊断使用已查看任务，后续正式评估仍用独立数据。业务结果与错购分析见 [#55 反馈](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/55#issuecomment-7167924678)。

## 运行

How 在原 WebShop 环境启动同配置的本地 bridge。先在诊断分支运行脚本的绝对路径，传同一个 [#55 原证据 ZIP](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm55-expanded-fit-20261125.zip) 和一个未使用的输出目录。

```bash
python /path/to/scripts/replay_prm55_option_pair.py replay \
  --archive "$evidence_zip" --run-dir "$pair_dir"
```

分别评分时使用真实运行的干净 `559a7791b49332375c640bc132af05e079676a83` checkout、原第 126 步配置和检查点。脚本核对该检查点的 manifest、adapter 和评分头，分别调用原推理入口。

```bash
python /path/to/scripts/replay_prm55_option_pair.py score \
  --archive "$evidence_zip" --run-dir "$pair_dir" \
  --config "$stage126_config" --checkpoint "$stage126_dir"
```

回传 `environment.json`、两个 rollout、两个预测及 `pair-comparison.json`。先检查环境 B−A 是否为正，再看共同前缀两步是否在容差内，以及选色步和购买步各自的 B−A。总分差与选色步差分别登记。环境同分或反向时记录为无有效规格对照；可以预先指定任务 7402 的椅子颜色作后备，按相同原则重新登记。预留 test 继续冻结。
