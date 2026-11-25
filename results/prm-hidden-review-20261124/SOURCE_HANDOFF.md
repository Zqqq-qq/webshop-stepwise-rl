# #54 历史dev复评：源码身份补齐交接

2026-11-25，针对 [How的暂停回报](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/54#issuecomment-7158627109) 做只读核查。此文提供交接办法，不修改身份门禁或批准新训练。

## 已核实

- Fork现有Release仅提供原36,760,028字节检查点ZIP，未见源码Git bundle。独立请求Fork的Git commit API获取 `d25e6164894c87fdea796bb73e3ee546807483d8` 返回404，公开分支tip也未找到该提交。404不证明原执行机没有对象，只说明当前共享渠道不能取得。
- 公开训练实现提交 `07bcdcbd75fb289272c2054702afa32059e3c98e` 的真实tree为 `a6da83c56a0ced0cbb6081c94f5eaf0f0bf305f6`，确实与作者声明的执行树值相同；公开train.py SHA为 `d71a30626cb4a747837a09a9e3814abc4bcccb2fbad2d2da6fe98febf4ab0ac8`。这是公开代码及作者声明的交叉核对，不能替代对不可获取的d25原提交对象的独立检查。
- 固定559a7791的 `runtime.checkpoint_identity` 从实际 `git rev-parse HEAD` 获取source_commit，检查受跟踪源码干净，并记录五个核心文件SHA；`train.py`不在这五文件表中。`infer --source-commit`仅断言实际runtime提交，不能覆盖或改写它。
- 因此仅复制完整src然后git init重新提交，或直接检出559a7791，即使代码内容匹配，也不能让原d25身份校验通过。手动改manifest或传入一个声明提交不能补足来源。

## 建议给Zenon的最小交付

从**仍持有原d25提交对象的执行机仓库**导出一个带可克隆ref的Git bundle，并附SHA256。以下是供对方执行的命令，本次未在对方机器执行：

```bash
git cat-file -t d25e6164894c87fdea796bb73e3ee546807483d8
git rev-parse 'd25e6164894c87fdea796bb73e3ee546807483d8^{tree}'
git branch evidence/pr54-executed d25e6164894c87fdea796bb73e3ee546807483d8
git bundle create /tmp/pr54-executed.bundle refs/heads/evidence/pr54-executed
git bundle verify /tmp/pr54-executed.bundle
sha256sum /tmp/pr54-executed.bundle
```

命名分支已存在时先核其目标，不覆盖现有ref。预期对象类型commit；tree按作者原声明应为a6da83c5…。上传到现有Fork Release，回传bundle SHA和完整命令日志。可另附完整PRM源码SHA清单，但不能用新建提交替代原d25对象。

## How收到后

先核bundle外层SHA，克隆到新目录并核真实HEAD、tree及受跟踪源码状态，均通过后才进入既定输入身份预检：

```bash
git clone --branch evidence/pr54-executed /path/to/pr54-executed.bundle /new/pr54-executed
git -C /new/pr54-executed rev-parse HEAD
git -C /new/pr54-executed rev-parse 'HEAD^{tree}'
git -C /new/pr54-executed status --porcelain --untracked-files=no
```

HEAD必须为d25e6164…，tree与上述公开训练实现对应，最后一条为空。再逐项核原manifest、train.py以及其他PRM源码SHA，保持检查点原件不变。原manifest含路径的auxiliary_input_sha256也需与运行实际一致，不能只核基模/B1权重后忽略config.json/generation_config.json路径身份。然后按Zqqq既定顺序执行原8条reference重载，通过后才做501/127冻结输入的历史dev复评。

如原Git对象确已丢失，请如实登记来源缺口并交Zqqq另定来源恢复/新运行协议；本轮不建议用跳过门禁或换写manifest延续原链。当前暂停的是新设备复评，已完成的3090保存预测/载荷SHA核验和8条拟合结论仍有效；后者不等于已经在5060Ti严格重载成功。
