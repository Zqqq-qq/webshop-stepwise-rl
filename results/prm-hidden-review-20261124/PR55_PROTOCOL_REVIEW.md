# #55 冻结协议：执行前独立核验

2026-11-25，固定审阅 [3fa1b9d8d546316cf237ab931df670798bd0388d](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/55)，落实 [Qiu隔离/覆盖/配对复核分工](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/54#issuecomment-7161844898)。本次只签收可在执行前检查的隔离、冻结输入及统计协议；新轨迹和126步训练尚无结果，设备执行仍按11/26 pre结束后的安排。

机器记录：[pr55-holdout-preflight.json](pr55-holdout-preflight.json)。使用冻结train/val/test **ID与指令身份**做交叉排除，没有读取test结果、执行test评估或选择test参数。

## 从原始记录重新计算

- 冻结train1451；原B1 JSONL确有911条，与fit729/历史dev182并集完全一致，原metadata的前911个计划ID与完成顺序一致。下一未完成ID是6426，单独排除。
- 初始候选补集540。与已归档旧500步dev127、诊断fit64/dev28、原8条探针任务交集均0；历史原件SHA与#55锁定值逐项匹配。
- 用锁定goal map（SHA `6b5d9dfc…`）重新做小写、压缩空白、去开头Instruction前缀的指令哈希；原911条指令哈希逐条吻合。与旧PRM/策略val/test指令重合的6个ID为6554、6877、7035、7205、7853、11489；候选内重复指令按最小ID保留，排除7900、9381、11798。共530个候选，与协议完全一致。
- 对530个按 `SHA256(ASCII("42:"+十进制ID))`、再按ID排序取前160，所得ID顺序、160个指令哈希和原文件SHA均一致。清单SHA **0980e0da7728b644fa34ca07c64e308746fdc50707e8742c9aa219b6b5328db1**。
- 用真实固定Qwen tokenizer逐条重编码fit729，保留501。直接保留原JSONL行字节和顺序（不重新序列化）后SHA为 **b1f1eeda3a762337d0ce99b9f98451a9fb0d444e6ed0a3cc5d302bd383e7cd71**。类别为正分购买446、购买零分26、耗尽零分29。

以上证明在**提供的已归档PRM记录**范围内隔离；B1继承既有SFT身份，不能称为整个策略/项目从未见过的任务。执行前仍需核设备上未归档的PRM使用记录；若发现遗漏，按协议在看新标签前更新清单并保留版本，不能直接用原160声明独立。

## 配置、预算及验收口径

固定559a7791的rollout实际使用2048−128=1920 token输入历史预算及128生成上限，属于总context2048；完整PRM序列另外整条按2048过滤，不混为同一截断规则。NF4采集与FP32训练/评分分开登记。采集配置的temperature1、top-p1、seed42、max_steps10、max_new_tokens128、原B1及环境指纹与协议一致。

rollout显式覆盖temperature/top-p，top-k继承固定Qwen generation_config。独立获取[固定revision生成配置](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/generation_config.json)，SHA `e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6`，top_k=20，另有继承的repetition_penalty=1.1。执行侧应记录生效generation_config及SHA，不能只保存显式覆盖的JSON值；本机未加载GPU模型。

126×8=1008次呈现。真实fit501原顺序前6个ID为0、9、11、13、28、31，各呈现3次，其余495条2次。63+63两段均显式总周期126，最终固定126；第63步只是诊断，不按dev挑选。独立重载用上述**保留后501原字节**与最终reference对应，避免把原729中被剔除的228条带入要求无剔除的reference验收。

总分通过条件已收窄到新dev保留集同任务上的三条fit-only基线，且常数/OLS配对区间上界<0；分组均值只有区间上界<0才声称明确优于。seed42、10,000次任务配对bootstrap，观测差/重采样均值/percentile区间分开。正常耗尽记0，超长剔除不补样，报告尝试160/完成数/保留N及类别分布，不能预写160条都保留。逐步归因、PPO与正式test另行签收。

## 可复核入口

```bash
python scripts/audit_prm_holdout_protocol.py --repo /local/repository \
  --data-archive /evidence/prm-diagnostic-evidence-20261124.zip \
  --probe-archive /evidence/prm-evidence-hidden-lrprobe-20261124.zip \
  --goal-map /evidence/goal-instructions.json --tokenizer /local/pinned-tokenizer \
  --output /new/holdout-review.json
```

该入口拒绝覆盖已有输出，固定#55提交并独立计算排除与排序，不调用rollout/模型/训练。本地完整126项CPU测试通过，compileall、仓库及来源检查通过；新增测试核指令前缀/大小写归一、已用/保留指令和候选重复排除、迭代顺序不影响排序。新采集覆盖、63→126状态衔接和配对指标待真实原件回传，未预先通过。
