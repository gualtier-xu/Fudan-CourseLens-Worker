"""Encrypted job and control mailbox backed by private GitHub Issues."""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Sequence

import requests

from .protocol import chunk_envelope, join_envelope, validate_task_id

API_ROOT = "https://api.github.com"
ISSUE_LABEL = "courselens-job"
TITLE_PREFIX = "[courselens-job]"
_PART_RE = re.compile(r"^(?:job )?part (\d+)/(\d+)\n([A-Za-z0-9+/=]+)$")

# RR-FIX452-1（DIAG-1 主判修复）：issue 评论创建在持续 ~70 POST/分钟时触发
# GitHub 内容创建限流（secondary rate limit），而同链成功任务实测 5-6 POST/分钟
# 长跑安全；旧实现单发即死（零重试零退避），一发 403/429/瞬时 5xx/网络抖动即
# worker_failed。发布面统一令牌桶节流 + 有界退避重试 + 闭集遥测行。
PUBLISH_RATE_PER_MINUTE = 6.0
PUBLISH_BURST = 3.0
RETRY_WAITS_SECONDS: tuple[float, ...] = (0.5, 1.0, 2.0)
RETRYABLE_STATUS_CODES = frozenset({403, 429, 500, 502, 503, 504})
RETRY_AFTER_MAX_SECONDS = 120.0


class MailboxError(RuntimeError):
    pass


class PublishPacer:
    """Token bucket pacing for issue-comment POSTs (thread-safe).

    clock/sleep 可注入（确定性测试）；缺省=真实单调钟。
    """

    def __init__(
        self,
        rate_per_minute: float = PUBLISH_RATE_PER_MINUTE,
        burst: float = PUBLISH_BURST,
        *,
        clock: Any = time.monotonic,
        sleep: Any = time.sleep,
    ):
        self._rate = max(1e-9, float(rate_per_minute)) / 60.0
        self._burst = max(1.0, float(burst))
        self._tokens = self._burst
        self._updated = clock()
        self._lock = threading.Lock()
        self._clock = clock
        self._sleep = sleep

    def wait(self) -> float:
        """Block until one POST token is available; return seconds waited."""
        while True:
            with self._lock:
                now = self._clock()
                self._tokens = min(self._burst, self._tokens + (now - self._updated) * self._rate)
                self._updated = now
                # 1e-9 就绪容差：浮点累积到 0.999…9 时不再 ε 级补眠空转。
                if self._tokens >= 1.0 - 1e-9:
                    self._tokens = max(0.0, self._tokens - 1.0)
                    return 0.0
                needed = (1.0 - self._tokens) / self._rate
                self._tokens = 0.0
                self._updated = now
            self._sleep(needed)


def _retry_after_seconds(headers: Any) -> float:
    try:
        value = float(str(dict(headers or {}).get("Retry-After") or "").strip() or 0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(value, RETRY_AFTER_MAX_SECONDS))


class IssueMailbox:
    def __init__(
        self,
        repo: str,
        token: str,
        *,
        timeout: int = 30,
        publish_rate_per_minute: float = PUBLISH_RATE_PER_MINUTE,
        publish_burst: float = PUBLISH_BURST,
        retry_waits: Sequence[float] = RETRY_WAITS_SECONDS,
        pacer: PublishPacer | None = None,
    ):
        if not repo or not token:
            raise ValueError("private job repository and token are required")
        self.repo = repo
        self.timeout = timeout
        self.issue_number: int | None = None
        self.status_comment_id: int | None = None
        self.retry_waits = tuple(float(wait) for wait in retry_waits)
        if pacer is not None:
            self.pacer = pacer
        else:
            self.pacer = (
                PublishPacer(publish_rate_per_minute, publish_burst)
                if float(publish_rate_per_minute) > 0
                else None
            )
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2026-03-10",
            "User-Agent": "Fudan-CourseLens-Worker/2",
        })

    def _get(self, path: str, *, params=None):
        try:
            response = self.session.get(f"{API_ROOT}{path}", params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise MailboxError(f"GitHub mailbox request failed: {type(exc).__name__}") from exc
        if response.status_code != 200:
            request_id = response.headers.get("X-GitHub-Request-Id", "unknown")
            raise MailboxError(f"GitHub mailbox returned HTTP {response.status_code} ({request_id})")
        return response.json()

    def _post(self, path: str, *, payload: dict[str, Any], stage: str = "job") -> dict[str, Any]:
        """POST with publish pacing and bounded transient retry.

        403/429/5xx/网络抖动按退避序列重试至多 len(retry_waits) 次；429/403
        （限流/滥用面）绝不进立即重试集——必按 Retry-After（缺省退避）真实等待
        后再发（6fbd077 家规：不 hammer、不越权重试）。401/404/422 等永久类
        立即失败。每次重试与穷尽失败都发闭集遥测行（仅码/计数/阶段词）。
        """
        response = None
        attempt = 0
        while True:
            if self.pacer is not None:
                self.pacer.wait()
            status = 0
            error: str
            try:
                response = self.session.post(
                    f"{API_ROOT}{path}", json=payload, timeout=self.timeout
                )
            except requests.RequestException as exc:
                error = f"GitHub mailbox request failed: {type(exc).__name__}"
            else:
                if response.status_code == 201:
                    return response.json()
                status = response.status_code
                request_id = response.headers.get("X-GitHub-Request-Id", "unknown")
                error = f"GitHub mailbox returned HTTP {status} ({request_id})"
                if status not in RETRYABLE_STATUS_CODES:
                    raise MailboxError(error)
            if attempt >= len(self.retry_waits):
                print(
                    f"mailbox_publish_failed http={status} "
                    f"attempt={attempt}/{len(self.retry_waits)} stage={stage}",
                    flush=True,
                )
                raise MailboxError(error)
            wait_seconds = self.retry_waits[attempt]
            if status in (403, 429) and response is not None:
                wait_seconds = max(
                    wait_seconds, _retry_after_seconds(response.headers)
                )
            print(
                f"mailbox_retry http={status} attempt={attempt + 1}/{len(self.retry_waits)} "
                f"wait_ms={round(wait_seconds * 1000)} stage={stage}",
                flush=True,
            )
            time.sleep(wait_seconds)
            attempt += 1

    def _patch(self, path: str, *, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.session.patch(
                f"{API_ROOT}{path}", json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise MailboxError(f"GitHub mailbox request failed: {type(exc).__name__}") from exc
        if response.status_code != 200:
            request_id = response.headers.get("X-GitHub-Request-Id", "unknown")
            raise MailboxError(f"GitHub mailbox returned HTTP {response.status_code} ({request_id})")
        return response.json()

    def wait(self, task_id: str, *, timeout_seconds: int = 600, poll_seconds: float = 2.0) -> dict[str, Any]:
        task_id = validate_task_id(task_id)
        deadline = time.monotonic() + max(1, timeout_seconds)
        last_error = "encrypted job is not available"
        while time.monotonic() < deadline:
            try:
                return self.read(task_id)
            except MailboxError as exc:
                last_error = str(exc)
                time.sleep(max(0.2, poll_seconds))
        raise MailboxError(last_error)

    def read(self, task_id: str) -> dict[str, Any]:
        task_id = validate_task_id(task_id)
        # N14（2026-10-07）：有界翻页检索。open 积压越过单页 100 时目标信封
        # 会被遮蔽成 "not available" 假死（wait 梯尽整单败）；issue 检索有界
        # 3 页（300 信封上限），评论分页收齐即停（5 页上限，防大载荷假性
        # incomplete）。常规情形目标在首页=与旧单页语义同一次 GET。
        expected_title = f"{TITLE_PREFIX} {task_id}"
        issue = None
        for page in (1, 2, 3):
            issues = self._get(
                f"/repos/{self.repo}/issues",
                params={"state": "open", "labels": ISSUE_LABEL, "per_page": 100, "page": page},
            )
            issue = next(
                (item for item in issues if str(item.get("title") or "") == expected_title),
                None,
            )
            if issue is not None or len(issues) < 100:
                break
        if issue is None:
            raise MailboxError("encrypted job is not available")
        self.issue_number = int(issue["number"])
        pieces: dict[int, str] = {}
        total = 0
        for page in range(1, 6):
            comments = self._get(
                f"/repos/{self.repo}/issues/{self.issue_number}/comments",
                params={"per_page": 100, "page": page},
            )
            for comment in comments:
                match = _PART_RE.fullmatch(str(comment.get("body") or "").strip())
                if not match:
                    continue
                index, candidate_total = int(match.group(1)), int(match.group(2))
                if total and total != candidate_total:
                    raise MailboxError("encrypted job part count mismatch")
                total = candidate_total
                pieces[index] = match.group(3)
            if total and sorted(pieces) == list(range(1, total + 1)):
                break
            if len(comments) < 100:
                break
        if total <= 0 or sorted(pieces) != list(range(1, total + 1)):
            raise MailboxError("encrypted job is incomplete")
        return join_envelope(pieces[index] for index in range(1, total + 1))

    def publish_control(
        self, sequence: int, envelope: dict[str, Any], *, stage: str = "checkpoint"
    ) -> None:
        if self.issue_number is None:
            raise MailboxError("job issue is not available")
        chunks = chunk_envelope(envelope)
        for index, chunk in enumerate(chunks, start=1):
            self._post(
                f"/repos/{self.repo}/issues/{self.issue_number}/comments",
                payload={
                    "body": f"control {int(sequence)} part {index}/{len(chunks)}\n{chunk}"
                },
                stage=stage,
            )

    def publish_status(self, sequence: int, envelope: dict[str, Any]) -> None:
        """Create or replace the single encrypted live-status comment."""
        if self.issue_number is None:
            raise MailboxError("job issue is not available")
        chunks = chunk_envelope(envelope)
        if len(chunks) != 1:
            raise MailboxError("encrypted status is unexpectedly large")
        payload = {"body": f"status {int(sequence)}\n{chunks[0]}"}
        if self.status_comment_id is None:
            created = self._post(
                f"/repos/{self.repo}/issues/{self.issue_number}/comments",
                payload=payload,
                stage="status",
            )
            self.status_comment_id = int(created["id"])
        else:
            self._patch(
                f"/repos/{self.repo}/issues/comments/{self.status_comment_id}",
                payload=payload,
            )
