# 11/23 全训练任务覆盖 RL 协议 v1

用户在查看[全覆盖计划](../configs/rl-full-coverage-20261123/plan.json)后明确指示“开始新训练”。这是新的预算扩展实验，独立于[11/21四更新小预算协议](RL_EXECUTION_PROTOCOL_20261121.md)。不覆盖或改写旧checkpoint/结果，不以扩大数据量保证OP>TG>SFT。Zqqq仍负责团队排期，本文件不新增日程。

## 固定范围

- 模型、奖励和算法沿用当前项目TG/OP变体：原采用的SFT B1初始化；Qwen2.5-1.5B固定revision，原B1权重SHA c39e2a78…，不是旧PPO终点。
- 全部1451个冻结train任务按seed42一次打乱，四组同一顺序，每个任务每组生成一条新on-policy轨迹；与160 val和200 test无交叉。完整363批ID在plan.json，最后一批3任务。
- 每组363次真实优化更新、1451回合。TG/OP×seed42/43合计1452更新、5804回合。每组上限11608策略决策/环境step、1485824生成token；全实验46432决策/step、5943296生成token。相同预算上限不保证实际消耗相同。
- 训练FP32、不量化、上下文2048、每轮生成128 token、最多8个动作；temperature1/top_p1/top_k0；dropout按原实现关闭。gamma=lambda=1、clip0.2、AdamW lr1e-5、value coefficient0.5、梯度裁剪1、不标准化优势、beta/KL均0，与旧实现一致。
- 一批一次optimizer step。每次先用当前策略新采样，再更新、保存；下一批独立进程恢复本组actor/critic/优化器/CPU与CUDA RNG。不得重复读取旧采样文件冒充on-policy训练。

## 顺序、初始化与停止

单GPU、单环境客户端串行 TG42→OP42→TG43→OP43。每组首轮从同一原B1和零critic开始，核对张量摘要；每组后续只接本组前一轮，原source a424a890的PPO核心不改。

固定选择第363次更新的endpoint，不用val挑中间模型或追加更新。训练完再以相同160-task FP32设置统一验证，旧小预算结果另列。test_authorization=false，保留test继续按负责人门禁处理。

非有限loss/概率重放不一致、身份/来源变化、环境失败、磁盘低于预留或子进程失败时停止，保留所有原件。已完成round不覆盖；重启监督器不会自动跳过失败任务，修复与续跑另记实际范围。全部round/checkpoint默认保留。

## 运行与来源

[单次更新审计器](../scripts/run_full_coverage_update.py)调用已验证的ppo.run，不替换损失或奖励；记录加载身份、真实更新时全策略/critic精度、后端、资源和实际summary。[监督器](../scripts/run_full_coverage_training.py)只在真实子进程更新成功及链条核验后增加计数，保存实际argv/日志/状态。

原生两种子[预检](../configs/rl-full-coverage-20261123/preflight.json)均通过；5类输入隔离/初始化/预算条件由原实现核查。两项纯CPU调度测试验证四链独立起点和末批、失败停机；这些不是GPU训练结果，启动后另交真实更新证据。

本次冻结运行路径和解释器指向目标WSL设备。对其它设备迁移应显式重生成路径及预检，不修改旧文件后沿用原SHA。GPU和验证监督器共享锁，避免同时使用。磁盘预留同时检查Linux与宿主C盘。临时缓存服务为已验收单客户端page-local-v1，health/fingerprint持续核查。

## 统计边界与协作

这是在查看旧小预算val后发起的数据覆盖扩展，明确保留探索性质，不称独立预注册复现。OP仍是规则势函数，不改称学习型进度模型。实际总量/耗时、失败、源码、每轮config/meta/轨迹/保存恢复都按真实执行记录。首批完成后给实际吞吐，不用旧文件时间差承诺新实验耗时。

用户已授权执行；全组仍可审查实现与结果。Jun核模型/恢复，Qiu核隔离/计数/统计，Zqqq统筹路线和排期，Yuxiao按实际完成情况更新展示。执行启动不冒充所有组员已审批，训练未完成前不写完成或提升。
