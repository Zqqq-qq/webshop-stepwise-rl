# Qiu 2026-11-25 独立复核

按 #52 [120步包校验分工](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/52#issuecomment-7137043090)、[历史dev复评分工](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/52#issuecomment-7137651251)及 #54 原证据评审项完成。以下均为保存记录的独立算术、真实tokenizer重编码和文件核验；本机没有执行模型前向、训练或反序列化优化器/RNG。历史dev指标、8条记忆与3090独立方案分开签收。

## 恒定1e-5/acc8累计120步

[完整包](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm-evidence-hidden-step120-20261124.zip) SHA `90b1b1f769f21cfe8b44627ebf628f0c3d2f32b506d19fe7d922cc63585aa49f`；16个包内载荷及8个检查点载荷SHA全部匹配，包括adapter/head/training-state。机器结果：[gpu-continuations-through120.json](gpu-continuations-through120.json)。

80→120新增40次更新/320次呈现，累计120/960；8条/38评分位完整。保存参数摘要、身份和设置接上80步；40/40次裁剪，训练loss复算最大差1.90e-9。归档独立重载与reference逐回合/总分差均0。核验的是training-state字节，不冒称已重新恢复了其中Adam/RNG状态。

MSE **0.002859788627**，最大绝对误差 **0.094437996546**。单条误差过线而MSE未过，合取记忆门槛仍未通过。任务0/6225（标签1）预测1.007338524/0.945938408；95/5031（标签0）预测0.049156088/0.048950449。满分−零分均值差 **+0.927585198**。部分得分351/359标签均为2/3，预测0.614607811/0.572228670。

## 固定120步历史dev复评

[复评包](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm-dev-eval-evidence-20261124.zip) SHA `6e8ac493b72af56c9b689e5fdaac8b7d91cdc5586e8cfa48776b21f11a5f0518`，7个载荷SHA匹配。原729/182条输入取自既有diagnostic归档，文件SHA与预登记03a6370b…/df1f71a7…逐字节一致。checkpoint声明的adapter/head/state SHA与完整120步包一致。

独立重编码全部911条，核全部保留/拒绝清单、原标签及终止类型分布，fit/dev任务与指令哈希交集0，8条记忆样本均在fit。保留501/127，评分位1980/496；每条预测与原标签/位置严格配对，无漏样本或重复。dev 127任务与旧500步诊断任务完全相同，登记为**历史dev复评**，不作为新的正式泛化验收。机器结果：[historical-dev-step120.json](historical-dev-step120.json)。

三条基线独立只用保留fit501拟合，逐条CSV与原JSON均已核对。常数0.617040205304；回合数OLS `0.923057836161 − 0.077431733869 × turns`；分组均值及未见回合数回退fit均值按登记口径。模型fit/dev MSE **0.288837543/0.217114300**；dev MAE0.366898353、max1.655515075、Pearson−0.000744864、Spearman0.037929746。三基线dev MSE依次0.119781810、0.109948296、0.116788782，均更低。

**点估计更正**：回传bootstrap数组的第一项实际上是10,000次重采样差值的平均值，不是原127条任务的MSE差。区间逐项复现，但点估计应为下表；不覆盖原附件。

| 模型−基线 | 原样本MSE差 | 原报告第一项（重采样均值） | 95%配对bootstrap区间 |
|---|---:|---:|---|
| 常数 | +0.097332490166 | +0.097214377617 | [+0.048763689945, +0.154308261698] |
| 回合数均值 | +0.107166003487 | +0.107046528578 | [+0.057060087534, +0.169585060261] |
| 回合数OLS | +0.100325517244 | +0.100242436715 | [+0.046806533883, +0.164321889442] |

复现口径：按任务配对、NumPy default_rng(seed42)，每个基线重新使用相同任务索引抽样，10,000次，percentile 2.5/97.5。三个区间仍均大于0，支持该固定检查点在这批历史dev上劣于基线；不外推到全任务总体。常数预测的Pearson/Spearman数学上未定义，本报告以null登记，而不是0或浮点近零。

| 模型分项 | n | MSE | MAE |
|---|---:|---:|---:|
| dev购买0分 | 10 | 0.516487407 | 0.684966692 |
| dev步数耗尽0分 | 4 | 0.961666952 | 0.778230995 |
| fit实际训练8条 | 8 | 0.002859789 | 0.048500637 |
| fit其余493条 | 493 | 0.293478155 | 0.392009484 |

dev购买保留123/144，耗尽只保留4/38，整体指标只适用于127条保留轨迹；耗尽分项仅4条，不能作稳定群体估计。低记忆误差未转化为该批其他轨迹上的好表现，但本次不能判定是哪一优化/表示机制导致。也不能用该结果证明#54的独立检查点如何泛化。

## #54 RTX3090三臂证据

固定审阅 `559a7791b49332375c640bc132af05e079676a83`，训练实现对应07bcdcbd。[Fork完整包](https://github.com/Zenon-nnnnnn/webshop-stepwise-rl/releases/download/prm-evidence-pr54-20261124/prm-split-cosine-rtx3090-20261124.zip) SHA `6ffeaba47f6586e54efd3ab13fef9890d2d3bd1bad6386bbe597aa8b349bb6f5`。39载荷顶层SHA匹配，三组共24个检查点载荷及其中9个二进制载荷SHA匹配，文本与固定PR内容一致。

运行并审阅了作者固定版本recompute.py，使用原8条samples和固定真实tokenizer核两侧评分位置；另用独立审计入口复算全部1,200条更新记录的原标签、顺序、loss、实际LR和裁剪标记。初始参数摘要一致，各960次呈现；三组归档重载差0。机器记录：[pr54-three-arms.json](pr54-three-arms.json)。

| 3090组别 | 更新/呈现 | 裁剪 | MSE | max误差 | 满分−零分均值差 |
|---|---|---:|---:|---:|---:|
| 共同1e-7恒定 acc8 | 120/960 | 120 | 0.303530243101 | 0.873627350 | −0.000220716 |
| split+cosine acc8 | 120/960 | 114 | 7.153360654e-7 | 0.001440600 | +0.999047838 |
| split+cosine acc1 | 960/960 | 818 | 1.732924392e-7 | 0.000762638 | +0.999280528 |

两组split+cosine按归档预测均过8条拟合门槛，支持组合方案的训练样本拟合能力；不签收正式dev泛化、逐步归因或PPO。acc8/acc1是相同呈现预算、不同优化步数；只在8k条消费后下一更新起点相位对齐。未隔离分组LR和退火各自贡献，不把5060Ti恒定1e-5结果当同机单因素对照。执行机重建提交的完整Git树未由本报告独立恢复，只核提供的源码SHA与固定PR。

早期Mac5步文件仅为舍入文本/非标准CLI日志，没有足够原件重算精确的更新后MSE与独立重载。按显示预测和原2/3标签只能近似得到MSE0.276892、max0.8678、满分/零分组差−0.001；它不单独构成门槛签收。三臂完整原件已替代其作为主要证据，原探针来源保留。

## 复核入口与验证

原 `audit_prm_hidden_results.py` 命令追加 `--step120-archive /evidence/step120.zip`（同时提供48/80前驱），新output可生成120步报告。另两入口：

```bash
python scripts/audit_prm_historical_dev.py --evaluation-archive /evidence/dev-eval.zip --data-archive /evidence/prm-diagnostic-evidence-20261124.zip --step120-archive /evidence/step120.zip --tokenizer /local/pinned-tokenizer --output /new/historical-dev.json
python scripts/audit_prm_split_saved.py --repo /local/repo --bundle /evidence/pr54-3090.zip --probe-archive /evidence/prm-evidence-hidden-lrprobe-20261124.zip --output /new/pr54.json
```

dev入口需NumPy，真实位置核验需固定Transformers/Tokenizers；不加载Torch。完整本地120项CPU测试通过，compileall、仓库链接/凭据检查及来源审计通过。新统计测试覆盖rank并列/常数相关未定义、fit基线、配对符号/零差与点估计区别；更新记录测试拒绝错误标签顺序/loss/LR/裁剪标记。保持 `test_authorization=false`，未追加训练或发起设备实验。
