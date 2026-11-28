# Qwen2.5 一轮 SFT 本地预实验

本次在 RTX 5060 Ti 上用固定版本 Qwen2.5-1.5B-Instruct 做 4-bit QLoRA。每次处理 1 个样本，梯度累积 8，上下文上限 2,048 tokens。完整跑完候选训练划分的 5,241 个样本，共 656 次参数更新。605 个验证样本的 loss 为 0.4003。测试集未使用。

首次中断后从 checkpoint-100 续跑。第二次在第 655 步触发完整性校验，当时最终 adapter 尚未保存。排查发现 Trainer 续跑时判断最后一个不满 8 条的累积批次，没有计入已跳过的批次数。仅在本地续跑进程修正同步条件，再从 checkpoint-600 跑到第 656 步，最终 epoch 为 1.0。项目和已安装的 Transformers 源码文件均未修改。

最终 adapter 已保存。另一个进程重载后，LoRA 张量哈希与训练结束时一致，合成购物提示成功生成 8 个 token。

[运行摘要](summary.json) 记录配置、数据指纹、源码、运行时版本、训练和重载哈希。运行记录报告的预留显存峰值约 29.31 GiB，高于设备容量 15.93 GiB。该值原样保留，原因未核实，不能作为实际物理显存占用。

本轮按本地预实验记录。验证 loss 用于检查 SFT 流程，购物表现需要另跑同一批任务比较。本 PR 只提交脱敏证据。

## 续训复现证据与验证集对照

[私有 data-v1 Release 证据包](https://github.com/Zqqq-qq/webshop-stepwise-rl/releases/download/data-v1/webshop-pr18-qwen25-sft-and-validation-evidence.zip)，SHA-256 `c62a1a97ac259a4a0f5ffb7d70e054ca2523c14d9866f53a40b56562c6aaae3b`。解压后运行 `sha256sum -c SHA256SUMS`。包内包含三阶段 wrapper 和实际配置、两次续训 launcher、精确 Trainer 补丁、checkpoint 文件身份、原始训练与重载记录，以及 8 任务和 160 任务的配置、meta 和脱敏逐任务结果。首轮命令按现存记录重建。逐 microbatch 消耗、梯度缩放及恢复状态逐项比对没有运行记录。

160 个验证任务配对结果见 [评估评论](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/18#issuecomment-7026403338)。本地候选满分成功为 50/160，底模为 19/160。正式 B1 配置仍待确认，测试集未使用。
