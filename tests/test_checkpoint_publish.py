"""RR-FIX452-1：检查点评论发布节流/有界退避重试/降级与 cadence 决策。

DIAG-1 主判回归面：452282 长讲每窗全量检查点评论洪泛（死亡前实测 70-75
POST/分钟）触发 GitHub 内容创建限流 × mailbox 单发即死 → worker_failed。
本文件锁定四条不变量：
①评论 POST 走令牌桶（默认 ≤6/分钟、burst 3）；
②403/429/5xx/网络瞬态有界退避重试 ≤3 次，429/403 必按 Retry-After/退避
  真实等待后重发（6fbd077 家规：绝不入立即重试集）；
③检查点评论发布穷尽失败=降级记账继续跑（丢续跑评论点不丢任务）；
④cadence 降频决策闭集可判（跨阶段/收尾必发、阶段内 N 窗一发、90s 兜底）。
"""

from __future__ import annotations

import io
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import requests

from courselens_worker.mailbox import (
    API_ROOT,
    MailboxError,
    IssueMailbox,
    PublishPacer,
    PUBLISH_BURST,
    PUBLISH_RATE_PER_MINUTE,
)
from courselens_worker.runner import (
    _CHECKPOINT_PUBLISH_CHUNKS,
    _CHECKPOINT_PUBLISH_MIN_INTERVAL_SECONDS,
    _CHECKPOINT_PUBLISH_WINDOWS,
    _CheckpointCadence,
    _CheckpointPublisher,
    _checkpoint_progress,
)

COMMENTS_PATH = "/repos/student/jobs/issues/7/comments"


class Response:
    def __init__(self, status_code, payload=None, retry_after=None, request_id="R"):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = {"X-GitHub-Request-Id": request_id}
        if retry_after is not None:
            self.headers["Retry-After"] = str(retry_after)

    def json(self):
        return self._payload


class QueueSession:
    """Per-(method, path) FIFO of responses/exceptions — drives retry sequences."""

    def __init__(self):
        self.calls = []
        self.scripted = {}

    def enqueue(self, method, path, *outcomes):
        self.scripted.setdefault((method, path), []).extend(outcomes)

    def post(self, path, json=None, timeout=None):
        if path.startswith(API_ROOT):
            path = path[len(API_ROOT):]
        self.calls.append(json)
        queue = self.scripted.get(("POST", path), [])
        outcome = queue.pop(0) if queue else Response(201, {"id": len(self.calls)})
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _box(session, **kwargs):
    params = {
        "publish_rate_per_minute": 6e7,
        "publish_burst": 1e6,
        "retry_waits": (0.0, 0.0, 0.0),
    }
    params.update(kwargs)
    box = IssueMailbox("student/jobs", "tok", **params)
    box.session = session
    box.issue_number = 7
    return box


class FakeClock:
    """Deterministic clock: sleep() advances virtual time (sandbox stall-proof)."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class PublishPacerTests(unittest.TestCase):
    def test_default_rate_stays_on_the_proven_safe_line(self):
        # DIAG-1 实测：同链成功 5-6 POST/分钟长跑安全，70-75/分钟触发限流。
        self.assertLessEqual(PUBLISH_RATE_PER_MINUTE, 6.0)
        self.assertGreaterEqual(PUBLISH_BURST, 1.0)

    def test_burst_then_capped_sustained_rate(self):
        # 沙箱环境会间歇冻结进程（实测 44s 停顿），真实钟断言不可靠——
        # 节流语义用确定性假钟锁：burst 1 后每发恰等一个 token（10/s → 0.1s）。
        clock = FakeClock()
        pacer = PublishPacer(600, 1, clock=clock.monotonic, sleep=clock.sleep)
        waits = [pacer.wait() for _ in range(8)]
        self.assertEqual(waits[0], 0.0)
        # burst 1 后 7 发各恰补一个 token 的眠（10/s → 0.1s/发，总量 0.7s）。
        self.assertEqual(clock.sleeps, [0.1] * 7)
        self.assertAlmostEqual(sum(clock.sleeps), 0.7, places=6)

    def test_fast_pacer_never_waits(self):
        clock = FakeClock()
        pacer = PublishPacer(6e7, 1e6, clock=clock.monotonic, sleep=clock.sleep)
        for _ in range(50):
            self.assertEqual(pacer.wait(), 0.0)
        self.assertEqual(clock.sleeps, [])


class PublishRetryTests(unittest.TestCase):
    def test_403_with_retry_after_waits_then_succeeds(self):
        session = QueueSession()
        session.enqueue("POST", COMMENTS_PATH, Response(403, retry_after=1), Response(201, {"id": 1}))
        box = _box(session)
        with patch("courselens_worker.mailbox.time.sleep") as sleep:
            with redirect_stdout(io.StringIO()) as out:
                box.publish_control(1, {"a": "b"})
        self.assertEqual(len(session.calls), 2)
        # 429/403 家规：等待 = max(退避, Retry-After)，绝不立即重发。
        sleep.assert_called_once_with(1.0)
        self.assertIn(
            "mailbox_retry http=403 attempt=1/3 wait_ms=1000 stage=checkpoint",
            out.getvalue(),
        )

    def test_429_without_header_falls_back_to_backoff(self):
        session = QueueSession()
        session.enqueue("POST", COMMENTS_PATH, Response(429), Response(201, {"id": 1}))
        box = _box(session, retry_waits=(0.5, 1.0, 2.0))
        with patch("courselens_worker.mailbox.time.sleep") as sleep:
            with redirect_stdout(io.StringIO()) as out:
                box.publish_control(1, {"a": "b"})
        sleep.assert_called_once_with(0.5)
        self.assertIn("mailbox_retry http=429 attempt=1/3 wait_ms=500", out.getvalue())

    def test_exhausted_retries_emit_closed_set_failure_line(self):
        session = QueueSession()
        session.enqueue(
            "POST", COMMENTS_PATH,
            Response(403, retry_after=0), Response(403, retry_after=0),
            Response(403, retry_after=0), Response(403, retry_after=0),
        )
        box = _box(session)
        with patch("courselens_worker.mailbox.time.sleep"):
            with redirect_stdout(io.StringIO()) as out:
                with self.assertRaises(MailboxError):
                    box.publish_control(1, {"a": "b"})
        self.assertEqual(len(session.calls), 4)
        text = out.getvalue()
        for line in (
            "mailbox_retry http=403 attempt=1/3",
            "mailbox_retry http=403 attempt=2/3",
            "mailbox_retry http=403 attempt=3/3",
            "mailbox_publish_failed http=403 attempt=3/3 stage=checkpoint",
        ):
            self.assertIn(line, text)

    def test_network_exception_retries_then_reports_http_zero(self):
        session = QueueSession()
        session.enqueue(
            "POST", COMMENTS_PATH,
            requests.ConnectionError("reset"), requests.ConnectionError("reset"),
            requests.ConnectionError("reset"), requests.ConnectionError("reset"),
        )
        box = _box(session)
        with patch("courselens_worker.mailbox.time.sleep"):
            with redirect_stdout(io.StringIO()) as out:
                with self.assertRaises(MailboxError):
                    box.publish_control(1, {"a": "b"})
        self.assertIn("mailbox_publish_failed http=0 attempt=3/3 stage=checkpoint", out.getvalue())

    def test_permanent_status_fails_without_retry(self):
        session = QueueSession()
        session.enqueue("POST", COMMENTS_PATH, Response(401))
        box = _box(session)
        with redirect_stdout(io.StringIO()) as out:
            with self.assertRaisesRegex(MailboxError, "HTTP 401"):
                box.publish_control(1, {"a": "b"})
        self.assertEqual(len(session.calls), 1)
        self.assertNotIn("mailbox_retry", out.getvalue())

    def test_server_error_is_retried_then_success(self):
        session = QueueSession()
        session.enqueue("POST", COMMENTS_PATH, Response(503), Response(201, {"id": 1}))
        box = _box(session)
        with patch("courselens_worker.mailbox.time.sleep"):
            with redirect_stdout(io.StringIO()) as out:
                box.publish_control(1, {"a": "b"})
        self.assertEqual(len(session.calls), 2)
        self.assertIn("mailbox_retry http=503 attempt=1/3", out.getvalue())

    def test_pacer_consumes_a_token_per_send_including_retries(self):
        clock = FakeClock()
        session = QueueSession()
        session.enqueue("POST", COMMENTS_PATH, Response(503), Response(201, {"id": 1}))
        box = _box(
            session,
            retry_waits=(0.0, 0.0, 0.0),
            pacer=PublishPacer(600, 1, clock=clock.monotonic, sleep=clock.sleep),
        )
        with redirect_stdout(io.StringIO()):
            box.publish_control(1, {"a": "b"})
        # burst 1：首发免费，重发前恰等一个 token（10/s → 0.1s）。
        self.assertEqual(clock.sleeps, [0.1])

    def test_status_stage_labels_the_telemetry_line(self):
        session = QueueSession()
        session.enqueue("POST", COMMENTS_PATH, Response(429, retry_after=0), Response(201, {"id": 5}))
        box = _box(session)
        with patch("courselens_worker.mailbox.time.sleep"):
            with redirect_stdout(io.StringIO()) as out:
                box.publish_status(1, {"a": "b"})
        self.assertIn("stage=status", out.getvalue())
        self.assertEqual(box.status_comment_id, 5)


def _term_value(window: int, total: int = 57) -> dict:
    return {
        "stage": "term_proofread",
        "term_proofread_completed_windows": window,
        "term_proofread_total_windows": total,
    }


def _asr_value(chunk: int, total: int = 17) -> dict:
    return {"stage": "asr", "completed_chunks": chunk, "total_chunks": total}


class CheckpointProgressTests(unittest.TestCase):
    def test_stage_keyed_counter_lookup(self):
        self.assertEqual(_checkpoint_progress(_asr_value(3)), (3, 17))
        self.assertEqual(_checkpoint_progress(_term_value(5)), (5, 57))
        self.assertEqual(
            _checkpoint_progress({
                "stage": "proofread",
                "proofread_completed_windows": 2,
                "proofread_total_windows": 9,
            }),
            (2, 9),
        )
        # term 值里保留的 proofread 计数不得串阶段（键按 stage 选取）。
        mixed = {**_term_value(5), "proofread_completed_windows": 50, "proofread_total_windows": 57}
        self.assertEqual(_checkpoint_progress(mixed), (5, 57))
        self.assertEqual(_checkpoint_progress({"stage": "unknown"}), (0, 0))
        self.assertEqual(_checkpoint_progress({"stage": "asr", "completed_chunks": "x"}), (0, 0))


class CheckpointCadenceTests(unittest.TestCase):
    def test_stage_boundary_publishes_and_final_stage_publishes(self):
        cadence = _CheckpointCadence()
        now = 100.0
        self.assertTrue(cadence.should_publish(_asr_value(1), now))
        cadence.mark(_asr_value(1), now)
        self.assertFalse(cadence.should_publish(_asr_value(2), now))
        self.assertTrue(cadence.should_publish(_asr_value(17), now))

    def test_stage_window_threshold(self):
        cadence = _CheckpointCadence()
        now = 0.0
        first = _term_value(1)
        self.assertTrue(cadence.should_publish(first, now))
        cadence.mark(first, now)
        # 阈值语义=距上次发布推进 ≥N 窗；首窗已发，下次恰在 1+N 窗。
        for window in range(2, 1 + _CHECKPOINT_PUBLISH_WINDOWS):
            self.assertFalse(cadence.should_publish(_term_value(window), now))
        self.assertTrue(
            cadence.should_publish(_term_value(1 + _CHECKPOINT_PUBLISH_WINDOWS), now)
        )

    def test_freshness_interval_requires_progress(self):
        cadence = _CheckpointCadence()
        cadence.mark(_asr_value(1), 0.0)
        self.assertFalse(cadence.should_publish(_asr_value(2), _CHECKPOINT_PUBLISH_MIN_INTERVAL_SECONDS - 1))
        self.assertTrue(cadence.should_publish(_asr_value(2), _CHECKPOINT_PUBLISH_MIN_INTERVAL_SECONDS + 1))
        # 无进度推进时新鲜度不触发（避免同计数重复刷评论）。
        cadence.mark(_asr_value(2), _CHECKPOINT_PUBLISH_MIN_INTERVAL_SECONDS + 1)
        self.assertFalse(
            cadence.should_publish(
                _asr_value(2), _CHECKPOINT_PUBLISH_MIN_INTERVAL_SECONDS * 3
            )
        )

    def test_summary_uses_chunk_cadence(self):
        cadence = _CheckpointCadence()
        first = {"stage": "summary", "completed_chunks": 1, "total_chunks": 9}
        self.assertTrue(cadence.should_publish(first, 0.0))
        cadence.mark(first, 0.0)
        self.assertFalse(
            cadence.should_publish(
                {"stage": "summary", "completed_chunks": _CHECKPOINT_PUBLISH_CHUNKS, "total_chunks": 9}, 0.0
            )
        )
        self.assertTrue(
            cadence.should_publish(
                {"stage": "summary", "completed_chunks": _CHECKPOINT_PUBLISH_CHUNKS + 1, "total_chunks": 9}, 0.0
            )
        )


class CheckpointPublisherTests(unittest.TestCase):
    def test_drop_stale_keeps_only_the_newest_pending_snapshot(self):
        sent = []
        publisher = _CheckpointPublisher(sent.append, flush_timeout=5.0)
        publisher.submit({"n": 1})
        publisher.submit({"n": 2})
        publisher.submit({"n": 3})
        with redirect_stdout(io.StringIO()) as out:
            publisher.close()
        self.assertEqual(sent, [{"n": 3}])
        self.assertEqual(
            publisher.stats,
            {"published": 1, "degraded": 0, "superseded": 2, "skipped": 0},
        )
        self.assertIn("checkpoint_publish published=1 degraded=0 superseded=2 skipped=0", out.getvalue())

    def test_mailbox_failure_degrades_instead_of_raising(self):
        def boom(_value):
            raise MailboxError("GitHub mailbox returned HTTP 403 (R)")

        publisher = _CheckpointPublisher(boom, flush_timeout=5.0)
        publisher.submit({"n": 1})
        with redirect_stdout(io.StringIO()) as out:
            publisher.close()
        self.assertEqual(publisher.stats["degraded"], 1)
        self.assertIn("checkpoint_publish_degraded stage=checkpoint", out.getvalue())

    def test_zero_flush_abandons_pending_and_stops_promptly(self):
        release = threading.Event()
        started = threading.Event()
        sent = []

        def blocked(value):
            started.set()
            release.wait(0.8)
            sent.append(value)

        publisher = _CheckpointPublisher(blocked, flush_timeout=5.0)
        publisher.submit({"n": 1})
        self.assertTrue(started.wait(2.0))
        publisher.submit({"n": 2})  # 在途发布阻塞时的待发件
        begin = time.monotonic()
        publisher.close(flush_timeout=0.0)  # 不等在途完成、弃待发件
        # join(2) 上界内返回（线程在 0.8s 阻塞后自行收尾）。
        self.assertLess(time.monotonic() - begin, 2.5)
        release.set()
        # 待发的 {"n": 2} 被弃（close 后不再发布）；在途 {"n": 1} 自行完成。
        deadline = time.monotonic() + 4.0
        while sent != [{"n": 1}] and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(sent, [{"n": 1}])

    def test_later_submit_while_publishing_is_drained(self):
        sent = []
        release = threading.Event()
        started = threading.Event()

        def slow_publish(value):
            started.set()
            release.wait(2.0)
            sent.append(value)

        publisher = _CheckpointPublisher(slow_publish, flush_timeout=5.0)
        publisher.submit({"n": 1})
        self.assertTrue(started.wait(2.0))
        publisher.submit({"n": 2})
        release.set()
        with redirect_stdout(io.StringIO()):
            publisher.close()
        self.assertEqual(sent, [{"n": 1}, {"n": 2}])
        self.assertEqual(publisher.stats["published"], 2)


if __name__ == "__main__":
    unittest.main()
