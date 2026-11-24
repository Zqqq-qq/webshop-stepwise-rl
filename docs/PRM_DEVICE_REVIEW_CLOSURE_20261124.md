# PR46：Jun 复审要求的逐项处理与证据索引

本页对应[Jun review 6032526088](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/46#pullrequestreview-6032526088)。原始JSON保持不变；新修订只纠正文档、提供摘要复算工具和证据等级说明，不将后续结果替写为历史结果。最终签收仍由复审者根据原件决定。

## 一、处理状态

| 要求 | 本次完成 | 尚未闭环的原件 |
|---|---|---|
| P1 原5步保存/独立重载 | 正文降为“执行方自报、待独立验收”；保留总分差0的历史自报与其限制 | 原adapter/head、固定输入、checkpoint身份、reference/reload逐步及总分、无标签推理、完整命令/日志；比较须先声明atol/rtol |
| P2 真实源码/命令 | #47已提供可恢复的完整a67db6a源码，5个历史源码SHA匹配；本页补恢复命令和入口差异 | 原采样/转换/训练/独立推理的完整argv、日志、配置映射仍需How归档；不能把省略号命令当可复现命令 |
| P2 原29条dev复算 | 三个MSE均明确标为自报负结果、未独立复算 | 原20步checkpoint/29条同集预测标签、数据SHA、fit-only常数/长度基线参数、脚本和命令；#47的28条dev不能替代 |
| P2 精度因果 | 已撤回bf16导致FP32必须降低lr的结论；保留5步观测及假设边界 | 仅在将来提出因果结论时再设计受控对照；本次无需新增GPU训练 |
| 最长轨迹 | 已区分48条冒烟集的6926和200条集合；200条摘要可确认下界至少9538 | 全200条原始输入与原tokenizer复算精确最大值 |
| loss描述 | 已纠正非单调和范围；新增CPU审计脚本及其实际输出 | 本项不需新增训练 |

缺失原件仅表示本次组内可访问材料尚未提供，不能据此断言设备上已丢失。How应先查找原件；无法恢复时如实注明，新增补验单独命名、保留日期和旧记录。代码修复与文档降级披露均不等于设备、泛化或逐步归因验收通过。

## 二、源码包恢复

私有Release：[prm-diagnostic-evidence-20261124.zip](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/prm-diagnostic-evidence-20261124.zip)，需要仓库权限。大小46,873,644字节，SHA256：

```text
e1578d582bdc1deaa5954e0b33e869d137eba0cb8d3efd7c57ecd7fcdadf2ec7
```

文件索引：

- `MANIFEST.sha256.json`：70个载荷文件的摘要，已在此前审阅中逐文件核对。
- `diagnostic/execution-source.bundle`：完整可恢复Git源码。
- `execution-source.tar`、`diagnostic/execution-source-sha256.json`：源码快照和9个PRM文件摘要。
- `diagnostic/{sorted-lr1e7,shuffled-lr1e7,shuffled-lr1e8}/`：后续三组128更新的权重、reference和诊断；**不是原5步或原20步权重**。
- `previous500-audit/`、`partial911-source/`：此前500步审计、911条部分采样；**不是原29条dev复算**。

在新建空目录，将下载包放入该目录后执行：

```sh
shasum -a 256 prm-diagnostic-evidence-20261124.zip
unzip prm-diagnostic-evidence-20261124.zip -d evidence
git clone evidence/diagnostic/execution-source.bundle recovered-source
git -C recovered-source rev-parse HEAD
# 应为 a67db6abefc73c6f6c7f1efb78645086782b1ade
```

摘要记录的 `runtime.py/model_core.py/train_core.py/tokenize.py/config.py` 五个源码SHA与恢复源码匹配。恢复的 `train.py` 支持`--weight-decay`并传给AdamW；不能把这个未合并的设备集成等同于当前#43/#45分支。这里确认源码可取得和接口差异，没有补造原始执行日志。

原摘要确认训练入口`webshop_lab.prm.train`、配置lr1e-7/weight_decay0.01、20更新和FP32；采样配置指向`webshop_lab.cli rollout`。实际数据转换及独立推理所用入口仍须原日志确认。完整历史命令必须绑定实际config、输入文件SHA、seed、device、optimizer超参；原件缺失前不根据后续运行猜写。

## 三、无需GPU即可复算的内容

在本PR仓库根目录运行：

```sh
python3 scripts/audit_prm_device_records.py
```

若要同时检查恢复源码的五个历史SHA，传入实际绝对路径：

```sh
python3 scripts/audit_prm_device_records.py --source /absolute/path/recovered-source/src/webshop_lab/prm
```

脚本核对三次5步与一次20步共35条记录的MSE、更新序号和有限性，输出loss列表/范围、摘要中观察到的最长超长长度，并验证20步summary原字节SHA与checkpoint manifest一致。带源码校验的本次实际输出位于`results/prm-device-20261124/record-audit.json`，含输入SHA。

该脚本不读取权重，不重跑GPU，不验证历史重载，不重新采样，也不复算原29条dev或全200条精确最大长度。检查摘要一致性不等同于验证真实GPU执行、源码无缺陷或模型能泛化。

## 四、下一步交接

How先按第一节归档原件和完整命令；Jun按证据逐项复核。本次可修的代码/文档已处理，仍缺的原件保持待验收，暂不标记全部完成。#47重载门禁修复与有界8样本记忆测试按其新计划推进；此前计划和证据保持原样，新controller/helper须冻结SHA后另建attempt。尚未提交真实设备结果前，不声称记忆测试或泛化通过，不放行learned-PRM PPO或保留test。
