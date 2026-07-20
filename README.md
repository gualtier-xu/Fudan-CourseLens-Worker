# Fudan CourseLens 通用 CPU Worker

本仓库是 CourseLens 的公开、通用 GitHub Actions 计算模板。它接收用户客户端创建的短时加密任务，生成字幕、OCR、摘要、章节、证据问答等派生学习资料。

它既是公开模板，也是每名用户个人 `Fudan-CourseLens-Worker` 的唯一受信文件来源。个人 Worker 应与客户端固定的 commit/tree 完全一致；不要直接修改个人仓库中的 README、workflow 或代码。

## 四仓库关系

```text
私有客户端 Fudan-CourseLens-Private
  ├─ 从本仓库固定 commit/tree 修复个人 Worker
  ├─ 把密封任务写入个人私有 Mailbox
  └─ 验签、解密并事务导入个人 Worker 返回的 Artifact

本公开模板 Fudan-CourseLens
  └─ 文件完全复制到个人 Fudan-CourseLens-Worker

个人 Fudan-CourseLens-Mailbox
  └─ 只承载密文任务、控制消息和清理状态
```

## 安全边界

- 平台登录、授权课程发现、媒体会话和凭据读取只允许出现在唯一受审计的 `courselens_worker/platform_session.py`；其他 Worker 文件和 workflow 不得实现平台连接逻辑。
- 不提供原视频下载、断点续传、批量抓取、归档或公开媒体 API。
- 媒体只以 HTTPS 流进入有界解码管道；源容器、PCM、Cookie、URL、字幕正文和 API Key 不写入磁盘、日志或 Artifact。
- 重定向逐跳校验 HTTPS、公网 IP 和端口；跨域不转发 Cookie、Origin 或 Referer。
- 响应在进入 FFmpeg 前按 HTTP 类别、Content-Type 和文件魔数做闭集校验；HTML、JSON 和未知媒体安全失败。
- Pull Request CI 只使用合成数据，无法读取生产 Environment secret。
- Artifact、Mailbox 密文和短期任务令牌在客户端成功导入并确认清理后删除。

浏览器拿到可播放字节后无法绝对阻止开发者工具抓取或屏幕录制。CourseLens 的目标是不提供产品下载能力、不泄露上游授权并显著限制批量抓取，不宣称实现 DRM。

## 处理能力

- `fast`：SenseVoice INT8。
- `no-proofread`：FireRedASR2 CTC INT8。
- `standard`：SenseVoice 粗识别 → FireRedASR2 CTC → 用户授权的 DeepSeek 结合粗识别校对。
- `summary`：可选 RapidOCR、时间戳摘要和章节。
- `learning_pack + answer`：仅依据密封的最小证据回答，并原样保留受控 citation ID。

### 云端无人值守入口

- `cloud-verify.yml` 只验证平台登录、DeepSeek 和已启用的 SMTP，不生成课程资料。
- `cloud-daily.yml` 固定列出 24 个半小时 cron，并通过 job-level 条件只在用户选中的北京时间分配 runner。
- 新课程默认仅发现；每门课程的字幕、OCR、摘要和章节必须由用户显式开启。
- 每日预算、认证/DeepSeek/网络/SMTP 熔断和待处理队列保存在加密状态 Artifact 中；状态保留 90 天且仅保留最新两份。
- 学习结果使用客户端长期 X25519 公钥加密并由 Worker Ed25519 签名，随机命名 Artifact 保留 30 天。
- SMTP 使用 Python 标准库和固定 Gmail、QQ、163、Outlook 预设，不调用第三方邮件 Action。

云端 Secrets 由私有客户端写入个人 Worker 的 `courselens-worker` Environment。Secret 名称存在不表示内容已经通过验证，是否可启用由验证 workflow 和配置哈希共同决定。

协议为 `job.v2` / `control.v2` / `result.v2`，兼容 v3 外层任务元数据：

- 输入使用 X25519 密封；
- 结果使用任务一次性公钥加密并由 Ed25519 签名；
- 客户端校验 task ID、input hash、签名和递增 sequence；
- 运行阶段、可测量 `completed/total` 和闭集错误码通过签名控制消息传递；
- 没有可测量总量时不伪造百分比或 ETA。

## 个人 Worker 运维

个人 Worker 由私有客户端自动创建、校验和修复：

- 生产 Environment 固定为 `courselens-worker`；
- Worker 私钥长期保存在 Environment secret；
- `COURSELENS_JOB_TOKEN` 只在活动任务租约期间存在；
- Mailbox 仓库名保存在受管变量中；
- tree 漂移、workflow 缺失或 Environment 不完整时，客户端停止发送媒体授权。

不要手工修改个人 Worker 的 `main`、Environment、workflow、variables、secrets 或 README。需要升级时先合并本模板的变更，再更新私有客户端固定的 commit/tree，最后使用客户端“一键修复 Worker”。

## 本地验证

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -q
python -m compileall -q courselens_worker scripts tests
python scripts/check_public_boundary.py
python scripts/check_text_encoding.py
```

`scripts/check_public_boundary.py` 会扫描当前文件和 Git 历史，阻止复旦登录、课程发现、Cookie、签名 URL、下载器或真实课程数据进入公开仓库。

## 发布规则

1. 所有第三方 Action 固定完整 commit SHA。
2. 变更通过 Pull Request、公开 CI、协议测试和边界扫描。
3. 合并后记录新的模板 commit 与 tree digest。
4. 私有客户端先更新固定 manifest，再允许用户修复个人 Worker。
5. 未通过 tree 信任校验前，禁止下发短期令牌和媒体授权。

## 许可

本仓库代码采用 Apache License 2.0。模型权重、运行库和外部 API 继续遵循各自上游许可证与使用条款。
