# 项目固定预算训练与完整160-task FP32验证（11/23）

执行方原件与统计已核验，正式登记待独立验收。依据[当前项目协议](../../docs/RL_EXECUTION_PROTOCOL_20261121.md)，保留原B1、TG/OP两臂、seed42/43，每臂每种子4更新/16回合，固定第四轮终点。没有把遍历1451任务或原论文模型替换当作新增要求。

## 当前完成范围

- 原采用的1 epoch SFT B1继续使用，不无故重训；历史恢复/同步补丁限制不消除。
- 四条PPO链均完成4次更新，合计16更新、64回合、272策略决策、220次实际step、16128生成token。
- 在设备上重算全部16轮实际actor/critic张量摘要，均与summary相符；父run/参数链、优化器累计step=1/2/3/4及CPU/CUDA RNG载荷检查通过。该核验不是第三方完整GPU重演。
- 六策略各160次验证，共960回合，同一冻结val顺序，含任务6700，无遗漏、重复、技术中断剔除或补批；正常策略失败留在分母。
- 每组实际meta和独立审计均记录FP32参数、输入权重/config/分词器辅助文件SHA、后端和执行源码；完整性与来源检查全部通过。
- test未使用，正式登记与保留测试放行仍由负责人验收。

完整验证实际时间为 **2026-11-23 10:51:52–13:37:19（Asia/Shanghai）**。逐策略elapsed包含加载、身份核对、生成和环境请求，不等于独占GPU kernel时间。当前无训练或评估进程运行。

## 同设置完整开发验证

| 策略 | 满分购买 | 成功率 | Wilson 95% | 购买数 | 平均终局任务奖励 | 非法动作率 |
|---|---:|---:|---|---:|---:|---:|
| B0 底模 | 15/160 | 9.375% | 5.76–14.89% | 50 | 0.1920 | 73.29% |
| B1 SFT | 54/160 | 33.750% | 26.88–41.38% | 156 | 0.6891 | 10.44% |
| TG seed42 | 51/160 | 31.875% | 25.15–39.45% | 155 | 0.6826 | 12.75% |
| OP seed42 | 52/160 | 32.500% | 25.73–40.09% | 155 | 0.6847 | 12.71% |
| TG seed43 | 53/160 | 33.125% | 26.30–40.74% | 152 | 0.6729 | 14.09% |
| OP seed43 | 53/160 | 33.125% | 26.30–40.74% | 151 | 0.6687 | 14.89% |

终局任务奖励在未购买时记0。success要求购买完成且奖励≥1−1e−6；页面最后匹配分不能代替购买奖励。完整数据见[comparison.json](comparison.json)。

## 配对比较

按冻结任务顺序、任务配对、Python random.Random(42).choices、10,000次重采样、线性分位数，各训练种子分别报告，不能把相同160任务视成320个独立任务。

| 比较 | 满分率差（pp） | 95%区间（pp） | 仅左组成功 / 仅右组成功 |
|---|---:|---|---:|
| B1−B0 | 24.375 | 17.500～31.875 | 2 / 41 |
| TG42−B1 | -1.875 | -5.000～1.250 | 5 / 2 |
| TG43−B1 | -0.625 | -4.375～3.125 | 5 / 4 |
| OP42−B1 | -1.250 | -4.375～1.875 | 4 / 2 |
| OP43−B1 | -0.625 | -4.375～3.125 | 5 / 4 |
| OP42−TG42 | 0.625 | 0.000～1.875 | 0 / 1 |
| OP43−TG43 | 0.000 | -1.875～1.875 | 1 / 1 |

B1较B0多39个满分任务，差+24.375pp，区间不含0。四个PPO终点未超过B1的54项成功；与B1的配对差区间均含0。OP42比TG42多1个成功任务，OP43与TG43成功总数相同，两种子OP−TG区间均含0；**本固定预算下未观察到稳定的成功率改善，不能声称等价或方法普遍无效。** 成功总数相同不代表逐任务或行为相同。

旧159-task默认精度结果仅作历史开发诊断，不与本表合并，也不再充当当前FP32结果。

## 执行与模型身份

实际源码[PR39提交a424a890](https://github.com/Zqqq-qq/webshop-stepwise-rl/commit/a424a89087b25618cf6cd19d58b3bbce445f2557)，完整Git blob/tree/commit匹配后在独立干净检出执行。运行时#38/#39在评审分支，结束后两项修复已合并；本记录保留实际执行提交，不改写成后来的main。

六配置除adapter外一致，固定B1或第四轮endpoint，不按val选checkpoint/追加训练。模型参数均为torch.float32，matmul precision=highest、CUDA/cuDNN TF32=false、autocast=false。这里只陈述参数与后端设置，不推及每个算子内部精度。

环境缓存health=page-local-v1、option_state=true；fingerprint冻结值、alignment1624/0、split verified。每策略只生成一份完整160项原始run，6700也按实际政策结果计入，没有将先前诊断拼入。

## 原件与独立复算

原件包地址和SHA见下方发布记录。包内六个根目录b0/b1/tg42/op42/tg43/op43保留JSONL/meta/audit/evaluation/失败案例/图；另含全部16轮训练轨迹、meta、summary、日志、精确执行源码归档和复算脚本。

```bash
sha256sum -c SHA256SUMS
python recompute.py /path/to/extracted-evidence
```

Git中的recompute.py与包内summarize-completed-runs.py相同。已从最终ZIP解压后验完整清单，离线重算输出与归档comparison逐项相等。

模型载荷包另含原B1与四终点的adapter、critic、优化器/RNG、run meta/summary；不重新分发底模。共享仅限私有Release，不进入Git历史。

## 私有发布记录

- [完整训练与160任务验证原件](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/webshop-project-fp32-full160-training-evidence-20261123.zip)，134载荷，SHA-256：`4ea38266297bdc5d70e1da5818851532d092c75503412a75772615b302fe797b`。
- [原B1与四终点模型及恢复载荷](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/webshop-project-b1-and-ppo-endpoints-20261123.zip)，28载荷，SHA-256：`5858c132193979b768ddbc934844988f08c6a6ddae67e6f376bfd037f8473bd0`。

## 尚未完成的项目门槛

本页不代替Qiu的正式结果登记、Jun的模型/恢复验收或Zqqq的test放行。展示稿/已提交proposal的引用版本由负责人协调，不能把当前结果冒充较早提交时已获得。测试模型与配置按最终门禁冻结后才运行保留test，test_authorization=false。
