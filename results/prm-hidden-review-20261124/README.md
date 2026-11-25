# Qiu：#52 GPU 短测独立验收与 #51 交接核对

**2026-11-25更新**：[120步、历史dev与#54完整原件复核](NOV25_REVIEW.md)。下方保留此前16/48/80步交付记录；120步和#54不再是“未收到附件”。

**最新交接已闭合**：原d25源码bundle已恢复并独立核验，#54的5060Ti重载与历史dev复评原件已签收；[新复核结果](PR54_5060TI_REVIEW.md)。此前的[源码交接办法](SOURCE_HANDOFF.md)保留为来源记录。

**下一轮执行前复核**：[#55冻结协议的独立隔离/输入核验](PR55_PROTOCOL_REVIEW.md)与[设备预检、候选外无过滤补查及444/160正式重冻结独立核验](PR55_REPLACEMENT_REVIEW.md)均已完成（初版预检记录[pr55-preflight-handoff-audit.json](pr55-preflight-handoff-audit.json)，正式重冻结记录[pr55-refreeze-audit-1c19b2b.json](pr55-refreeze-audit-1c19b2b.json)）。`prm55-unfiltered-history-audit-20261125.zip` 确认 `2525/8000/9381/11798/6045` 均0命中、`7900` 有4条实际使用记录并排除对应候选 `6755`，#55 `a601999`/`1c19b2b` 重冻结清单（SHA `15b96f01…`）与 444 洁净池逐字节吻合，且保留 `test_authorization=false`；新采集覆盖、训练与配对结果待设备执行回传，不预填结果。

2026-11-24。按 [#52 最新分工](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/52#issuecomment-7132933687)，本次核对已保存GPU预测、裁剪记录、覆盖和重载输出，不执行新的模型前向、GPU训练或设备续训。#49补包/CPU后续由Yuxiao承接，原Qiu/Tutor成果保留。

## 已发布三档16步原件

[短测包](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm-evidence-hidden-lrprobe-20261124.zip) SHA `aa33e5ca0b62e06a43801ea6416326256ae0d389f8ee8e6c3d46cbf571e75b08`；[特征/源码包](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm-evidence-hidden-features-20261124.zip) SHA `aa35ec78c4149dc81414e0a44b49740abe5b55103b2fb7fdc57383bc67e5615d`。分别29/16个载荷SHA全部匹配，第16步完整检查点另有8个载荷逐项匹配。

机器结果：[gpu-probes.json](gpu-probes.json)。同8条原输入SHA `9c88beff…`、固定tokenizer四文件及模板SHA一致；独立重编码确认8条、38个assistant im_end评分位。每档16次实际更新/128次轨迹呈现、8/8保留，各16次都触发clip1。每条训练loss由当次8条预测/标签重算，最大差小于2.4e-8。三档reference与另存独立重载的逐步/总分差均为0；这是核保存记录，不是本机再次运行1.5B重载。

| LR / acc8 | 末次8条 MSE | 最大绝对误差 | 满分组−零分组均值 | 记忆门槛 |
|---|---:|---:|---:|---|
| 1e-7 | 0.346186244 | 0.980564242 | +0.000005769 | 未过 |
| 1e-6 | 0.306178403 | 0.849675998 | −0.000186510 | 未过 |
| 1e-5 | 0.206815201 | 0.720792741 | −0.003560647 | 未过 |

均值差定义为 `(p_0+p_6225)/2 − (p_95+p_5031)/2`。四条均为3回合，前两条标签1，后两条标签0。在1e-5下：

| task | 标签 | 总分 |
|---|---:|---:|
| 0 | 1 | 0.287688971 |
| 6225 | 1 | 0.279207259 |
| 95 | 0 | 0.287153423 |
| 5031 | 0 | 0.286864102 |

满分组均值0.283448115，零分组均值0.287008762，当前未形成有效排序。不将该现象或高余弦解释为“特征必然没有信息”。常数MSE **0.187500**、按回合数分组均值MSE **0.129629630**，仅是这8条上的样本内参照，不能当独立dev结果。后续dev基线只用fit拟合。

实际保存身份仍为 `5a529686c2a5a26da8893b67ded98ac63a208d12`；提供的7个核心源码文件逐字节SHA与执行单 `5ae6d83553ced1a74db32d750ada6f9d9b6786b1` 一致，覆盖train/infer/runtime/model_core/train_core/config/tokenize。未独立恢复本地执行的完整Git树，因此只签收这7文件一致，不改历史manifest身份或宣称两个提交完全同树。

## #51 独立核对与交接

固定 #51 `cfb10c9e681ff21090b610504da7f0c6d1424dcf`。已核 [Actions run 45379955682](https://github.com/Zqqq-qq/webshop-stepwise-rl/actions/runs/45379955682) 的实际head、工作流、成功状态；下载artifact14912463941，ZIP SHA `50a10570b28cd2e057328578fff717c81e32ef3eb6c320fb5a21ca66e38e8229`，其results.json与仓库跟踪的JSON逐字段一致。

独立从保存的8条预测/标签复算4组共32个检查点，MSE/max error/轨迹呈现/优化器步数/裁剪计数一致；512呈现的batch1为512更新、batch8为64更新。四组均未通过记忆门槛，不包装为同优化器步数比较或dev调参。本机没有复跑PyTorch训练，也没有反序列化optimizer；安全加载逻辑、114状态等检查通过的证据来自已核验的固定Actions运行，不能写成Qiu本机再次检查了运行中精度。

同时用补包的重提F独立执行NumPy最小二乘：磁盘(8,151936)，显式加N后(8,151937)，rank8、condition205.1749698264、解L2约0.00701406445、FP32最大标签误差约7.56e-7。结果在平台浮点容差内一致；保留“重提特征”来源等级，不改称首次保存原F。完整机器记录：[pr51-replay.json](pr51-replay.json)。

对 #49 `d9d2f6c` 与 #51 做了 `git merge-tree --write-tree` 整合检查：无冲突且两方记录可同时保留。没有执行分支合并、覆盖#49或改main。建议以保留双方源码/结果的方式纳入，避免用新的CPU JSON替换旧记录或将不同数值实现说成逐bit复现。

## 更新：累计48/80步独立核验

已收到并完整下载 [48步包](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm-evidence-hidden-step48-20261124.zip)（SHA `4e115935e87cd3b6577e4a5a56a0fc2a1f4fb22b5eefa9a1064fe7fa0b000666`）及 [80步包](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm-evidence-hidden-step80-20261124.zip)（SHA `4a2daf64b4256d76942b8044d5cade3166c9d4ffa9d40b152cf327931284e46c`）。每包16个载荷SHA、清单中的8个完整检查点载荷SHA全部匹配。机器记录：[gpu-continuations-48-80.json](gpu-continuations-48-80.json)。原16步报告保留，不覆盖。

| 累计更新 | 累计呈现 | 本段更新/裁剪 | 原8条MSE | 最大绝对误差 | 满分组−零分组 |
|---|---:|---|---:|---:|---:|
| 48 | 384 | 32 / 32 | 0.114233857 | 0.535895586 | +0.060077101 |
| 80 | 640 | 32 / 32 | 0.061121196 | 0.402100801 | +0.322291031 |

两段均8/8轨迹、38/38评分位完整，保存的独立重载逐回合/总分差均为0。逐次训练loss复算最大差分别7.98e-9、3.25e-9。16→48→80的保存参数摘要哈希首尾匹配，身份和恒定 `1e-5 / acc8` 优化设置一致、更新号连续；已核验training-state.pt字节SHA，**本机未反序列化优化器/RNG状态或运行模型**，因此不将摘要链接匹配写成重新执行续训一致性测试。

| task | 原标签 | 48步总分 | 80步总分 |
|---|---:|---:|---:|
| 0 | 1 | 0.608224809 | 0.737467587 |
| 6225 | 1 | 0.576199472 | 0.683591604 |
| 95 | 0 | 0.535895586 | 0.402100801 |
| 5031 | 0 | 0.528374493 | 0.374376327 |

48步满分组均值应为 **0.592212141**，不是回传文字的0.590582；其与零分组0.532135040之差为 **0.060077101**。80步满分/零分均值为0.710529596/0.388238564，四条均沿正确方向变化。按回合数做精确误差分解，16→48的MSE下降约82.176%来自组均值误差下降；48→80则约99.601%来自组内中心化误差下降。这支持这8条上同回合成败区分开始改善，仍不能证明独立泛化或逐回合因果归因。两检查点均未过记忆门槛，常数/回合数基线仍仅是样本内参照。

复核入口沿用下方命令，追加 `--step48-archive /evidence/prm-evidence-hidden-step48-20261124.zip --step80-archive /evidence/prm-evidence-hidden-step80-20261124.zip` 并选择新的output。工具现在支持包内嵌套目录及`./`校验路径，并拒绝目录外载荷和规范化后重复清单项。增加续训参数摘要/配置/步号链与MSE分解检查。新增6项反例/恒等式测试后，本地完整 **113项CPU测试通过**；compileall、仓库检查和来源审计通过。

## 下一检查点（原16步记录，现由上节更新）

原16步附件不含累计48/80/120步结果；现已按上节签收48/80的保存证据。累计120步仍待新的原件独立验收，设备执行按Zqqq现行单子，不由本报告追加预算。正式记忆门槛仍为MSE<0.001且每条误差<0.1，泛化/逐步归因/PPO和test不在此层签收。#54另一个分组学习率/余弦方案的正文宣称RTX3090运行通过，但本次可见提交/Release缺相应原始记录，已在该PR留“证据待补”审核，不与本恒定学习率链混合。

## 可复核入口

```bash
python scripts/audit_prm_hidden_results.py \
  --probe-archive /evidence/prm-evidence-hidden-lrprobe-20261124.zip \
  --features-archive /evidence/prm-evidence-hidden-features-20261124.zip \
  --tokenizer /local/pinned-qwen-tokenizer --repo /local/repository \
  --output /new/gpu-probes.json
python -m unittest discover -s tests -p test_prm_hidden_results_audit.py -v
```

需NumPy的旧SVD复算由#49/#51保留；本入口的指标/覆盖检查使用标准库，真实模板复核使用Transformers4.56.2/Tokenizers0.22.2，不加载Torch。`--repo`须已取得固定#52提交以核七个源码文件。审计输出不覆盖输入或已有文件。测试中的11个构造反例分别核缺样本/缺步骤、任务重复、NaN、评分位置、裁剪标记、标签/呈现数/记录loss，以及续训分段计数；不把它们当真实GPU结果。本地完整107项CPU测试通过，compileall、仓库检查和来源审计通过。
