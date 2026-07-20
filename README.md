# Fudan CourseLens Worker

> CourseLens 的公开 CPU 计算模板，也是个人 `Fudan-CourseLens-Worker` 的唯一受信文件来源。
>
> Public CPU worker template for CourseLens. A personal Worker is managed by the desktop client and must remain byte-for-byte aligned with this template.

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

- `fast`：SenseVoice 快速字幕。
- `no-proofread`：FireRedASR2 CTC 字幕。
- `standard`：SenseVoice 粗识别、FireRedASR2 CTC 和用户授权的 DeepSeek 校对。
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

云端无人值守默认关闭。启用后，凭据只进入个人 Worker 的 GitHub Environment Secrets；是否已配置、是否已验证和是否允许调度是三个不同状态。

## 许可

代码采用 [Apache License 2.0](LICENSE)。模型权重、运行库和外部 API 继续受各自许可证与服务条款约束。
