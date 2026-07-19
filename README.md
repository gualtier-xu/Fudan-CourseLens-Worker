# Fudan CourseLens Worker

这是由私有 Fudan CourseLens 客户端为当前 GitHub 账号自动创建和维护的个人 Actions Worker。

## 用途

- 只运行当前用户已授权的加密字幕、OCR、摘要和章节任务。
- 所有输入通过密封盒传输，媒体只进入有界流式解码，不保存原视频或可下载副本。
- 结果以加密 Artifact 返回；客户端成功导入后删除 Artifact、Issue 密文和临时任务令牌。
- 不要在此仓库手工添加课程 URL、Cookie、复旦账号、字幕正文、API Key 或媒体文件。

## 自动维护

Worker 文件、workflow、协议版本和可信 tree 由私有客户端从公开模板自动校验与修复。发现文件被修改、版本漂移或仓库名称冲突时，客户端会暂停发送媒体授权并提供修复操作。

不要手工修改 `main`、生产 Environment、workflow 或 secrets。需要更新时，先更新公开模板，再由客户端执行可信版本修复。

## 运行资源

生产 Environment 为 `courselens-worker`，包含短期任务令牌、X25519 输入私钥和 Ed25519 签名私钥。Mailbox 仓库名称保存在变量中。Pull Request 只运行合成测试，不访问生产 secrets。

## 数据生命周期

任务 Issue 只保存加密分块控制消息；成功导入后评论被删除并关闭 Issue。Artifact 最长保留七天，正常情况下更早删除。仓库日志只允许出现随机任务 ID、阶段、计数、耗时和固定错误码。

本仓库不是课程目录、播放器、下载器或通用计算服务。