# #54 5060Ti重载及历史dev独立复核

2026-11-25，按 [Zqqq最新分工](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/54#issuecomment-7161372102) 核 [How回传](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/54#issuecomment-7161168885)，补入#53。本次不执行模型或训练，不反序列化Adam/RNG；设备推理成功的来源是归档日志与输出，本机完成原始Git对象恢复、SHA校验、真实tokenizer重编码和独立统计。

## 原件与源码身份

[新复评ZIP](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm-pr54-dev-eval-20261125.zip) SHA `e717e9155f49c8cdde210ec3459cc71dbe69dc7295c9dd82e060959d095f0fd0`，13个载荷全部匹配。包内[原Git bundle](https://github.com/Zenon-nnnnnn/webshop-stepwise-rl/releases/download/prm-evidence-pr54-20261124/pr54-executed.bundle) SHA `4eee0a23ff79a259c3b29c9a9758039dcbec8adcf06114a7eddca7fd6c42916b`，本机实际clone并运行bundle verify通过：

- HEAD `d25e6164894c87fdea796bb73e3ee546807483d8`；tree `a6da83c56a0ced0cbb6081c94f5eaf0f0bf305f6`；受跟踪源码干净。
- 五个核心源码SHA均与manifest一致，train.py SHA `d71a30626cb4a747837a09a9e3814abc4bcccb2fbad2d2da6fe98febf4ab0ac8`。此前“取不到原提交对象”的缺项已解决，不再仅凭作者对tree的声明。
- 新包内checkpoint manifest/config/summary逐字节匹配原3090 `hidden-split-cos-acc8`。原完整ZIP SHA `6ffeaba47f6586e54efd3ab13fef9890d2d3bd1bad6386bbe597aa8b349bb6f5`及其39个载荷匹配，该组8个checkpoint载荷（含adapter/head/training-state）全部逐项校验。
- 新reference/fit/dev输出的完整identity均匹配原manifest，含source_commit和辅助文件路径键；本机没有访问远端物理软链接，仅核归档身份及源码门禁逻辑。

## 跨设备原8条重载

8/8保留，38个真实assistant im_end评分位完整，rejected为空、reference_match=true。与3090原reference的最大逐步差 **1.661479473e-6**、最大总分差 **1.681968570e-6**，均在 `atol=rtol=1e-5` 内；是容差内一致，不是逐bit零差。新fit推理中的这8条与单独reference重载输出也在容差内一致。

5060Ti保存输出的8条MSE **7.161334605e-7**、最大绝对误差 **0.001441612840**，合取记忆门槛通过。与3090的7.153360654e-7差异在重载容差范围内，保留各自设备及运行来源，不覆盖原记录。

## 501/127覆盖及fit-only基线

独立重编码冻结原输入729/182条；输入SHA仍为03a6370b…/df1f71a7…，保留501/127、评分位1980/496；剔除228/55及逐条原因、原标签/回合数/终止类型与CSV匹配。两集合任务/指令哈希交集0，8条记忆样本全在fit。全部预测无缺失/重复，逐回合分数和总分一致。历史dev127任务与#52完全相同。

三条基线独立只在保留fit501上重拟合，再应用到dev127；参数、逐条值、MSE/MAE/max error/Pearson/Spearman与归档一致。常数相关系数正确登记null。机器记录：[pr54-historical-dev-5060ti.json](pr54-historical-dev-5060ti.json)。

| 方法 | fit MSE | 历史dev MSE |
|---|---:|---:|
| #54 split+cosine | 0.139641903258 | 0.154588889754 |
| fit均值常数 | 0.116880678651 | 0.119781809562 |
| fit回合数均值 | 0.092822535309 | 0.109948296241 |
| fit回合数OLS | 0.099374146430 | 0.116788782484 |

模型dev MAE0.312736683786、max1.369680833817、Pearson0.227703326014、Spearman0.173755579732。三条基线仅用fit拟合的定义与#52相同。配对bootstrap按127个任务、10,000次、default_rng(seed42)，各对照复用同一组索引抽样；观测差与重采样均值分开，已修正上轮报告口径的问题。

| 模型−基线 | 观测MSE差 | 95%区间 |
|---|---:|---|
| 常数 | +0.034807080192 | [−0.000199914968, +0.075108504355] |
| 回合数均值 | +0.044640593513 | [+0.011601941090, +0.082503906147] |
| 回合数OLS | +0.037800107270 | [+0.004757245106, +0.077069099572] |

相对常数的区间跨0，不能判定总体优劣或宣称等效；相对两条回合数基线的区间均为正，支持该批历史dev上更差。不能把点估计高于三基线统一说成“三者都显著更差”。

## 与#52同任务配对及分项

按任务ID重新对齐两套127条预测，并核原标签、终止类型和评分位置一致。观测MSE差（#54−#52） **−0.062525409974**，95%区间 **[−0.123234470059, −0.012766361446]**；点估计下降 **28.7984%**。支持该批历史dev上的误差改善，但两条训练链同时改变学习率/调度并有跨设备训练来源，不能归因于某一个超参。

| 分项 | n | #52 MSE | #54 MSE |
|---|---:|---:|---:|
| dev买错0分 | 10 | 0.516487407 | 0.283971780 |
| dev步数耗尽0分 | 4 | 0.961666952 | 0.007963247 |
| fit实际记忆训练 | 8 | 0.002859789 | 0.000000716 |
| fit其余轨迹 | 493 | 0.293478155 | 0.141907886 |

dev购买保留123/144，耗尽只保留4/38。四条耗尽轨迹的改善可描述，不能外推到被拒绝的长轨迹或稳定群体效果。全部结果仍登记**历史dev复评**，不代替新任务正式泛化、逐步归因或PPO验收。扩大fit训练和新dev的预算/划分/选择规则仍由下一轮协议冻结，本次没有启动设备任务。

## 复跑及验证

```bash
python scripts/audit_prm_historical_dev.py --variant pr54 \
  --evaluation-archive /evidence/prm-pr54-dev-eval-20261125.zip \
  --data-archive /evidence/prm-diagnostic-evidence-20261124.zip \
  --split-checkpoint-archive /evidence/prm-split-cosine-rtx3090-20261124.zip \
  --previous-evaluation-archive /evidence/prm-dev-eval-evidence-20261124.zip \
  --source-workdir /new/source-checkout --tokenizer /local/pinned-tokenizer \
  --output /new/pr54-dev-review.json
```

output、source-workdir及其同名.bundle均须新路径，避免覆盖原件。入口保留旧step120模式及其bootstrap更正记录。归档日志提供源码与推理exit0，但未保存reference/fit/dev完整CLI argv；建议执行方另补准确命令，作为复现材料补充，不改原包。本地123项CPU测试通过，compileall、仓库和来源检查通过。新回归覆盖按任务重排配对、缺任务/终止类型变更拒绝及bundle SHA预检；不是新的GPU模型测试，始终 `test_authorization=false`。
