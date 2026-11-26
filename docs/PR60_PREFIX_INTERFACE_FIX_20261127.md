# 三项评分接口修复与验证记录

日期：2026-11-27。范围：基于 PR #60 的 `3141fab4a31ec712181674cf053c447153016c03` 修复组合入口；不创建远程新分支、不修改 main、不训练或替换权重。

## 问题与结果

| 评审问题 | 本次改动 | 验证 |
|---|---|---|
| 完整轨迹的未来长度影响早期评分 | 离线逐步取当前前缀；线上与离线共用外置因果编码器，逐步记录输入哈希、局部位置、裁剪/回退/跳过 | 固定真实 Qwen tokenizer 下，延长点击后页面不再改变早先点击的输入与 113 位置；在线共享函数输出相同 |
| 归档 manifest 仍标记旧 sum/MSE 语义 | 保留旧归档，通过独立冻结、绑定原 manifest 和全部载荷的评分契约声明 PV 损失、差分、末值与长度规则 | 契约错配、语义篡改、helper 改动和载荷变化拒绝；新旧 reference 不混用；当前旧 sum 入口拒绝已登记的五臂 PV manifest 或 head |
| 固定旧 runtime 中缺新增模块 | 外置 helper 位于脚本旁；worker 显式按路径加载，不从旧 runtime 导入 prefix_reward 或 data | 真实固定旧 runtime 的 PYTHONPATH 下 `--help` 可启动；跨进程测试还移除新版模块，验证无隐藏依赖 |

详细命令、固定身份及新配置见 [Arm 4 交接说明](../results/prm-prefix-value-20261126/ARM4_HANDOFF.md)。契约自身与 helper 的哈希均独立记录；不通过修改原 checkpoint identity 绕过验证。

## 实际检查

- 全套 `python -m unittest discover -s tests -q`：**163 项通过，0 跳过**。
- 其中评分与契约相关 12 项包含：真实 tokenizer 因果边界、辅助模块在旧 runtime 环境下启动、小型真实 Qwen/PEFT 保存与独立进程 reference 重载、五臂历史记录复算、契约与载荷篡改拒绝。
- 原 PPO/在线边界 9 项回归通过，覆盖超长、空回复和后续继续评分。
- `python -m compileall -q src scripts tests`、`python scripts/check_repository.py` 通过。
- register 检查通过；当前 register 只有表头，不能据此声称已登记新的实验。
- 来源审计执行完成，仍报告上游历史任务 ID 重叠 12；本次未修改数据与划分，也没有把该历史审计当成新训练集已通过隔离的结论。
- 本地测试使用固定 PR60 源码 tarball 与只读 Git 元数据夹具完成依赖提交查询，便于运行需要 Git 信息的测试；不是新修复版的干净 GPU 执行检出。正式设备仍须锁定实际组合提交。

复现时使用实际 Git checkout、Python 3.11、已安装的 Torch/Transformers/PEFT，以及离线缓存的固定 Qwen tokenizer；没有固定 tokenizer 的环境可能跳过相关测试，不能以跳过替代本次已执行的结果。

## 保留的证据边界

1. 未改历史权重包、旧 manifest、五臂结果或旧 reference；旧版全轨迹编码规则留在历史实验脚本中，仅用于历史复算，不能替代当前因果入口。
2. 外部契约是针对已确认训练来源的显式发布声明，不是通用的权重训练方式检测器。禁止使用者自行给任意 sum 模型换契约并更新哈希来伪装成 PV。
3. 新输出是 `causal_prefix_value_v1`，记录前缀局部位置；原旧 reference 无相同契约会被拒绝。真实权重的新版参考仍需主设备按交接单重新生成并独立进程核对。
4. 原始 messages 与策略实际裁剪后的 prompt 可能不同；对真实 PPO 重放需核对实际前缀输入哈希。不能仅以同 task_id 声称在线、离线输入相同。
5. 本次没有 GPU 冒烟、策略更新或效果评估。旧 #60 的两题一更新结果仍只覆盖 `3141fab`。组员复审后，在新组合提交上复跑既有训练侧小规模冒烟，核对契约、奖励落点、actor/critic 更新和重载，再启动已冻结的 TG/Arm4 配对短训。
6. 神经奖励仍是前缀价值差，不保证等于真实终局奖励，也不单凭望远镜恒等保证策略不变。test200 继续冻结。
