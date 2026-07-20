# CourseLens Worker 技术说明

> 本文面向维护者和安全审计者。个人 Worker 用户不应按本文手工修改受管仓库；客户端会负责创建、校验、修复和清理。
>
> Technical reference for maintainers and auditors. Managed personal Workers must be updated through the CourseLens client.

## 仓库角色与信任模型

公开 `Fudan-CourseLens` 是唯一模板。私有客户端在 `runtime-assets.json` 中固定模板 commit 和 tree；个人 Worker 必须满足仓库归属、公开可见性、默认分支、受管描述和整棵 tree 一致性。任一条件漂移时，客户端禁止下发媒体授权和临时任务令牌。

个人 Worker 只执行当前用户的 GitHub Actions。任务信封来自该用户的私有 Mailbox，结果写入随机命名的加密 Artifact。私有客户端是任务状态、导入结果和清理结论的唯一权威。

## 代码边界

- `courselens_worker/platform_session.py`：唯一允许处理平台会话、授权课程发现和临时媒体会话的模块。
- `courselens_worker/source.py`：HTTPS、公网地址、端口、重定向、响应类型和魔数校验，以及有界流式输入。
- `courselens_worker/asr.py`、`ocr.py`、`llm.py`：派生计算，不持久化原媒体。
- `courselens_worker/protocol.py`、`mailbox.py`：密封、签名、控制消息、分块密文和清理协议。
- `courselens_worker/cloud_automation.py`：默认关闭的云端验证、每日发现、预算、熔断和通知。
- `courselens_worker/runner.py`：受管任务入口、阶段进度、检查点和结果封装。

除唯一平台会话模块外，其他文件和 workflow 不得实现平台登录、课程目录读取或媒体签发逻辑。

## 协议与数据流

当前协议使用 `job.v2`、`control.v2` 和 `result.v2`，并兼容客户端 v3 外层元数据：

1. 客户端为每次任务生成随机 ID、输入哈希和结果密钥对。
2. 任务使用 Worker 的 X25519 公钥密封，分块写入私有 Mailbox。
3. Worker 校验任务、流式处理输入，并用 Ed25519 签名控制消息和结果。
4. 控制消息必须匹配 task ID、input hash 和递增 sequence；重放、乱序和篡改消息被拒绝。
5. 客户端下载 Artifact 后验签、解密、校验 schema 与哈希，再事务导入本地数据库。
6. 只有 GitHub 和客户端都确认后，界面才显示“已取消”“已导入”或“云端数据已清理”。

控制消息可以携带阶段、`completed/total`、观测时间和闭集错误码。只有 Worker 提供可测量总量时才显示百分比；ETA 必须来自后端估算器并带新鲜度。

## Workflow

| Workflow | 用途 | 主要边界 |
| --- | --- | --- |
| `ci.yml` | 单元测试、编译和公共边界检查 | 不读取生产 Secrets |
| `echo.yml` | 加密通道、签名、Artifact 和清理验证 | 不处理课程媒体 |
| `process.yml` | 客户端主动派发的派生计算 | 每任务短期令牌、受管并发 |
| `cloud-verify.yml` | 验证平台登录和 DeepSeek | 不生成课程资料 |
| `cloud-daily.yml` | 默认关闭的云端每日发现与处理 | job-level 时间条件、单用户并发 |
| `synthetic-smoke.yml` | 合成 ASR/OCR 冒烟测试 | 仅合成输入 |

第三方 Action 必须固定完整 commit SHA。PR workflow 不得获得个人 Worker 的生产 Environment Secrets。

## Environment、Secrets 与 Variables

生产 Environment 固定为 `courselens-worker`。长期只保留 Worker 输入私钥和签名私钥；任务令牌只在租约期间存在。Mailbox 仓库名、运行策略和云端调度开关使用受管 Variables。

云端无人值守凭据由私有客户端逐项写入 Environment Secrets。GitHub API 只能确认 Secret 名称和更新时间，不能回读内容，因此“已上传”不等于“已验证”。客户端只有在模板 tree、配置哈希和验证 workflow 均有新鲜正向证据时才允许启用 schedule。

## 媒体与网络安全

- 每次连接和重定向都校验 HTTPS、公网 IP 和允许端口；跨域重定向删除 Cookie、Origin 和 Referer。
- 响应在进入解码器前按 HTTP 类别、Content-Type 和前 64 字节魔数分类；HTML、JSON 和未知媒体安全失败。
- 合法媒体以有界块进入解码管道，不把源容器、PCM、完整响应头或正文写盘。
- 日志仅包含随机任务 ID、阶段、计数、耗时、资源和闭集错误码。
- Artifact、Issue、邮件和诊断输出不得包含课程正文、永久地址、账号或密钥。

## 云端每日自动化

`cloud-daily.yml` 使用固定半小时 cron 集合，并通过 job-level 条件只在用户选定的北京时间分配 runner。新课程默认仅发现；字幕、OCR、摘要和章节必须按课程显式开启。

Worker 维护加密增量状态、每日 runner/token 上限，以及认证、DeepSeek 和平台网络熔断。认证熔断必须回到本地重新验证；达到预算后保留待处理队列，不伪造人民币费用。Worker 不发送邮件，也不接收邮箱账号或 SMTP 授权码。

学习结果使用客户端长期 X25519 公钥加密；云端状态使用独立状态密钥。结果和状态 Artifact 使用不同保留策略，客户端导入后再执行有证据的清理。

## 本地验证

使用 Python 3.10–3.12：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -q
python -m compileall -q courselens_worker scripts tests
python scripts/check_public_boundary.py
python scripts/check_markdown_links.py
python scripts/check_text_encoding.py
git diff --check
```

模型安装和合成冒烟测试由 `scripts/install_models.py`、`synthetic_asr_smoke.py` 和 `synthetic_ocr_smoke.py` 提供。不得用真实课程内容替代 CI 合成数据。

## 发布顺序

1. 从最新 `main` 创建分支，通过 Pull Request、CI、协议测试、UTF-8、链接和公共边界扫描。
2. 合并后记录新的模板 commit/tree，并更新私有客户端固定资产清单。
3. 客户端验证新模板后，使用“一键修复 Worker”同步个人仓库。
4. 运行加密 echo，确认签名、导入和临时数据清理。
5. tree 未受信或清理未确认时禁止派发媒体任务。

不要在 README 中维护易过期的 commit、tree 或 run ID；这些证据应记录在私有客户端的交接和验收报告中。

## 许可证与依赖

仓库代码采用 Apache License 2.0。SenseVoice、FireRedASR2、RapidOCR、ONNX Runtime、FFmpeg、DeepSeek 和其他依赖分别遵循其上游许可证、模型许可与服务条款。发布者必须保留相应声明，不得把 Apache-2.0 误表述为覆盖全部模型和服务。
