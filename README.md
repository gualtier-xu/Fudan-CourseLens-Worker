# Fudan CourseLens Worker

> This repository is a signed, CI-generated read-only mirror of the private
> CourseLens monorepo. Do not edit generated files here; changes must originate
> under `worker/` or `shared/protocol/` in the CourseLens source tree and arrive
> through the protected mirror pull request workflow.

> CourseLens 的公开 CPU 计算模板与官方发布仓，也是个人 `Fudan-CourseLens-Worker` 的唯一受信文件来源。
>
> Public CPU worker template for CourseLens. A personal Worker is managed by the desktop client and must remain byte-for-byte aligned with this template.

## 下载与安装

CourseLens 客户端（Windows）从这里发布。顺利的话，一两分钟就能装完进入学习台。

**官方发布页：[gualtier-xu/Fudan-CourseLens-Worker · Releases](https://github.com/gualtier-xu/Fudan-CourseLens-Worker/releases)** ｜ 问题反馈：[官方仓 Issues](https://github.com/gualtier-xu/Fudan-CourseLens-Worker/issues)

1. **下载并校验**：到[官方发布页](https://github.com/gualtier-xu/Fudan-CourseLens-Worker/releases)下载 `CourseLens-0.1.0-setup.exe`，它旁边有一份《校验单》。校验就像核对快递单号：在安装包所在的文件夹，地址栏输入 `powershell` 回车，执行 `certutil -hashfile CourseLens-0.1.0-setup.exe SHA256`，把输出的一串数字和《校验单》对一致再装——确认你拿到的文件和发布的那份逐字节一致。
2. **安装**：双击 setup.exe。如果 Windows 弹出「已保护你的电脑」，点「更多信息」→「仍要运行」——每一步长什么样、为什么要这一步，见[《Windows 安全提示图文指引》](https://github.com/gualtier-xu/Fudan-CourseLens-Worker/releases/download/client-v0.1.0/windows-security-prompt.md)。注意：安装包只负责**第一次安装**，以后的升级都在应用内完成，不要重复运行 setup.exe。
3. **登录选课**：打开 CourseLens，按首跑引导用复旦统一身份认证登录（学号密码只输入客户端本地的受保护表单），然后选择你有权访问的课程就能开始。想用 AI 校对与问答，还可以在设置里填入你自己的 DeepSeek API Key（可选，不填也有高精度非 AI 字幕回退）。

## 如果这是你的个人 Worker

这个仓库由 CourseLens 客户端自动创建，用来临时运行字幕、OCR、摘要、章节和其他派生学习资料任务。日常使用不需要在 GitHub 网页中配置它。

请不要手工修改或删除：

- README、代码和 `.github/workflows/`；
- Environment、Variables 或 Secrets；
- 正在运行任务的 Actions run 或 Artifact。

手工修改会改变仓库 tree，客户端会停止发送任务。需要诊断、修复、取消、清理或撤销凭据时，请回到 CourseLens 客户端的“连接与隐私”或“任务中心”。

## 如果你在查看公开模板

本仓库只提供通用、可审计的 GitHub Actions Worker。私有客户端会固定模板的 commit/tree，将完整文件复制到每名用户自己的 Worker，并在派发任务前重新验证完整性。

```text
CourseLens 客户端
  ├─ 密封任务 → 用户私有 Mailbox
  ├─ 校验并触发 → 用户个人 Worker（本模板的受管副本）
  └─ 验签、解密、事务导入 ← 加密 Artifact
```

开发、协议、workflow、测试和发布说明见 [技术 README](docs/technical/README.md)。

## 它会做什么

- 字幕执行只有一个内部 `automatic` 策略：配置 DeepSeek Key 时做「粗识别 + Paraformer 精修 + 用户授权的 AI 校对」，未配置时自动走非 AI 回退（仅 Paraformer 精修）。平台原生文稿存在且时间覆盖达标时，默认用它替代粗识别腿——它只作校对时的参考文本，绝不直接成为字幕输出；设置 `COURSELENS_ASR_ROUGH_SOURCE=sensevoice` 可随时强制回原双模型链。
- 可选 OCR、摘要、章节、证据问答和云端每日检查。
- 用签名控制消息报告真实阶段；没有可靠总量时不伪造百分比或剩余时间。

## 它不会做什么

- 不提供课程下载、批量抓取、断点归档或公开媒体 API。
- 不在仓库中保存课程账号、Cookie、课程目录或永久媒体地址。
- 不把原视频、PCM、字幕正文或 API Key 写入日志和 Git。
- 不让 Pull Request workflow 读取生产 Environment Secrets。
- 不宣称能够阻止浏览器开发者工具抓取或屏幕录制，也不虚假宣称 DRM。

## 数据和清理

客户端主动提交的任务只通过用户私有 Mailbox 传递密文。结果使用客户端公钥加密并由 Worker 签名；客户端验签、解密、校验哈希并成功导入后，才确认删除临时 Artifact、Mailbox 内容和任务令牌。

云端无人值守（自动学习材料）默认关闭。启用后，凭据只进入个人 Worker 的 GitHub Environment Secrets；是否已配置、是否已验证和是否允许调度是三个不同状态。计划固定为每天 13:00 与 22:00（北京时间）两个 cron 入口，配置哈希绑定 Worker 树、账号与课程规则；凭据被拒立即熔断，结果 Artifact 加密并保留最多 90 天。

## 许可

代码采用 [Apache License 2.0](LICENSE)。模型权重、运行库和外部 API 继续受各自许可证与服务条款约束。
