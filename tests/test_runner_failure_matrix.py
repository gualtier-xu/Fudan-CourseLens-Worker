"""夜10-C T6: dispatch/failure/retry 合成矩阵扩充 — 错误码闭集净化矩阵、
source-session 收口排列、进度发布器闭集状态 coercion。

零真实端点、零媒体、零模型依赖；全部走 mock 与纯函数面。
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import Mock, patch

with patch.dict("sys.modules", {"sherpa_onnx": Mock()}):
    from courselens_worker.asr import ASRError
    from courselens_worker.platform_session import PlatformSessionError
from courselens_worker.protocol import (
    JOB_SCHEMA,
    PROTOCOL_VERSION,
)
from courselens_worker.runner import (
    SignedProgressPublisher,
    WorkerError,
    main as runner_main,
    process_job,
    safe_worker_error_detail,
)


class SafeWorkerErrorDetailMatrixTests(unittest.TestCase):
    """错误码闭集净化矩阵：未知码塌缩、registered stage 才允许后缀、
    闭集外异常塌缩成「类型: 净化消息」（N18 最小清晰化；URL 形态只留类型）。"""

    def test_platform_challenge_required_stays_independent_without_suffix(self):
        error = PlatformSessionError(
            "platform_challenge_required", connection_stage="course_context"
        )
        self.assertEqual(safe_worker_error_detail(error), "platform_challenge_required")

    def test_connection_failure_gets_registered_stage_suffix(self):
        error = PlatformSessionError(
            "platform_connection_failed", connection_stage="webvpn_context"
        )
        self.assertEqual(
            safe_worker_error_detail(error),
            "platform_connection_failed_webvpn_context",
        )

    def test_unregistered_stage_never_leaks_into_the_code(self):
        error = PlatformSessionError(
            "platform_connection_failed", connection_stage="totally_bogus_stage"
        )
        self.assertEqual(safe_worker_error_detail(error), "platform_connection_failed")

    def test_unknown_platform_code_collapses_to_session_failed(self):
        error = PlatformSessionError("platform_some_future_code")
        self.assertEqual(safe_worker_error_detail(error), "platform_session_failed")

    def test_asr_error_matrix_maps_known_and_collapses_unknown(self):
        known = ASRError("authorized media request returned HTTP 403")
        self.assertEqual(safe_worker_error_detail(known), "media_http_403")
        unknown = ASRError("some brand new asr failure text")
        self.assertEqual(safe_worker_error_detail(unknown), "asr_error")

    def test_generic_exception_maps_type_and_sanitized_message(self):
        """N18 rider：闭集外异常带 type+净化消息入 reason（H1 活体若有此早
        一行定位——UnboundLocalError 类裸类型不再零帧逃逸）。"""
        self.assertEqual(
            safe_worker_error_detail(RuntimeError("boom")),
            "RuntimeError: boom",
        )

    def test_reason_never_carries_url_bearing_messages(self):
        self.assertEqual(
            safe_worker_error_detail(
                RuntimeError("GET https://api.example.com/v1/x failed")
            ),
            "RuntimeError",
        )

    def test_reason_is_bounded_and_whitespace_flattened(self):
        value = safe_worker_error_detail(RuntimeError("a\nb" + "x" * 200))
        self.assertEqual(value, f"RuntimeError: a b{'x' * 77}")
        self.assertEqual(len(value), 94)

    def test_bare_exception_with_empty_message_returns_type_only(self):
        self.assertEqual(safe_worker_error_detail(RuntimeError()), "RuntimeError")


class SourceSessionCloseMatrixTests(unittest.TestCase):
    """process_job 的 source-session 收口排列：无论检查点/进度回调怎么炸，
    收口回调恰一次；无会话载荷绝不误收口。"""

    def _job(self, close=None):
        payload = {}
        if close is not None:
            payload["_close_source_session"] = close
        return {
            "schema": JOB_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "task_id": "0123456789abcdef0123456789abcdef",
            "job_kind": "echo",
            "input_hash": "0" * 64,
            "pipeline": {"version": "actions-echo-v2"},
            "payload": payload,
        }

    def test_close_runs_once_when_checkpoint_writer_explodes(self):
        close = Mock()
        with patch(
            "courselens_worker.runner._process_materialized_job",
            side_effect=lambda job, **kwargs: kwargs["checkpoint_writer"](
                {"stage": "half", "completed_chunks": 1}
            ),
        ):
            with self.assertRaises(Exception):
                process_job(self._job(close), checkpoint_writer=RuntimeError("boom"))
        close.assert_called_once_with()

    def test_close_runs_once_when_progress_callback_explodes(self):
        close = Mock()
        with patch(
            "courselens_worker.runner._process_materialized_job",
            side_effect=lambda job, **kwargs: kwargs["progress_callback"](
                "asr", 1, 2
            ),
        ):
            with self.assertRaises(Exception):
                process_job(
                    self._job(close), progress_callback=RuntimeError("boom")
                )
        close.assert_called_once_with()

    def test_close_is_skipped_without_session_payload(self):
        close = Mock()
        with patch(
            "courselens_worker.runner._process_materialized_job",
            return_value={"status": "completed", "outputs": {}, "metrics": {}},
        ):
            process_job(self._job(None))
        close.assert_not_called()

    def test_success_result_passes_through_unchanged(self):
        with patch(
            "courselens_worker.runner._process_materialized_job",
            return_value={"status": "completed", "outputs": {"echo": {"ok": True}}},
        ) as inner:
            result = process_job(self._job(None))
        self.assertEqual(result["status"], "completed")
        inner.assert_called_once()


class MainFailureForensicsTests(unittest.TestCase):
    """N18：main() 失败行=type+reason+有界末帧（帧元数据定位，零消息文本）。"""

    def _stderr_of(self, side_effect) -> str:
        buffer = io.StringIO()
        with redirect_stderr(buffer), patch(
            "courselens_worker.runner.run", side_effect=side_effect
        ):
            self.assertEqual(runner_main(), 1)
        return buffer.getvalue()

    def test_bare_type_carries_reason_and_faulting_frame(self):
        def _boom():
            raise TypeError("synthetic boom")

        line = self._stderr_of(_boom)
        self.assertIn("worker_failed type=TypeError", line)
        self.assertIn("reason=TypeError: synthetic boom", line)
        # 帧段=最深抛出点（真 faulting line）：本测试里即 _boom 所在测试文件。
        self.assertRegex(line, r" at=test_runner_failure_matrix\.py:\d+:_boom\b")
        self.assertNotIn("cause=", line)

    def test_chained_exception_reports_cause_frame(self):
        def _boom_chained():
            try:
                raise ValueError("inner cause")
            except ValueError as inner:
                raise TypeError("synthetic boom") from inner

        line = self._stderr_of(_boom_chained)
        self.assertIn("reason=TypeError: synthetic boom", line)
        self.assertRegex(
            line, r" cause=ValueError@test_runner_failure_matrix\.py:\d+:_boom_chained"
        )

    def test_url_bearing_exception_keeps_type_and_frame_without_message(self):
        def _boom_url():
            raise RuntimeError("authorized https://user:secret@example.test/path")

        line = self._stderr_of(_boom_url)
        self.assertIn("worker_failed type=RuntimeError", line)
        self.assertIn("reason=RuntimeError", line)
        self.assertNotIn("example.test", line)
        self.assertRegex(line, r" at=test_runner_failure_matrix\.py:\d+:_boom_url\b")

    def test_closed_set_error_keeps_reason_channel_unchanged(self):
        """闭集异常的 reason 通道仍为闭集码（帧只作 stderr 加性取证）。"""
        error = PlatformSessionError(
            "platform_challenge_required", connection_stage="course_context"
        )
        line = self._stderr_of(error)
        self.assertIn(
            "reason=platform_challenge_required", line,
            "闭集码必须原样上 reason",
        )


class WorkflowProfileContractTests(unittest.TestCase):
    """N20（R3 SEC-13）：媒体 kind 误入 llm.yml 时=合同拒绝+可读文案，
    不再靠 SENSEVOICE_MODEL_DIR env 缺失巧合失败。profile 缺席=本地/测试
    形态放行（老客户端 process.yml 恒带 process-v1，兼容面全覆盖）。"""

    def _job(self, kind: str) -> dict:
        return {
            "job_kind": kind,
            "task_id": "task-profile-" + kind[:6],
            "input_hash": "0" * 64,
            "pipeline": {"version": "test-v2"},
            "payload": {},
            "secrets": {},
        }

    def test_media_kind_on_llm_profile_is_a_contract_rejection(self):
        for kind in ("subtitle", "learning_pack"):
            with patch.dict("os.environ", {"COURSELENS_WORKFLOW_PROFILE": "llm-v1"}):
                with self.assertRaises(WorkerError) as caught:
                    process_job(self._job(kind))
            message = str(caught.exception)
            self.assertIn(f"job kind {kind} requires workflow profile process-v1", message)
            self.assertIn("(got llm-v1)", message)

    def test_missing_profile_keeps_legacy_pass_through(self):
        value = {
            "mode": "fallback", "segments": [], "raw_sensevoice": [], "metrics": {},
        }
        with patch.dict(
            "os.environ",
            {"SENSEVOICE_MODEL_DIR": "s", "PARAFORMER_MODEL_DIR": "p"},
        ), patch("courselens_worker.asr.transcribe", return_value=value):
            result = process_job(self._job("subtitle"))
        self.assertEqual(result["status"], "completed")

    def test_process_profile_admits_media_kind(self):
        value = {
            "mode": "fallback", "segments": [], "raw_sensevoice": [], "metrics": {},
        }
        with patch.dict(
            "os.environ",
            {
                "COURSELENS_WORKFLOW_PROFILE": "process-v1",
                "SENSEVOICE_MODEL_DIR": "s",
                "PARAFORMER_MODEL_DIR": "p",
            },
        ), patch("courselens_worker.asr.transcribe", return_value=value):
            result = process_job(self._job("subtitle"))
        self.assertEqual(result["status"], "completed")

    def test_summary_kind_stays_legal_on_llm_profile(self):
        def _fake_summary(api_key, *, title, transcript, ppt_pages, prior_checkpoint,
                          checkpoint, usage_sink=None, **kwargs):
            return {"model": "deepseek-flash", "markdown": "笔记", "chapters": []}

        job = self._job("summary")
        job["payload"] = {"title": "t", "transcript": [], "slides": []}
        with patch.dict("os.environ", {"COURSELENS_WORKFLOW_PROFILE": "llm-v1"}), patch(
            "courselens_worker.llm.create_summary", side_effect=_fake_summary
        ):
            result = process_job(job)
        self.assertEqual(result["status"], "completed")


class SignedProgressPublisherClosedSetTests(unittest.TestCase):
    def test_unknown_status_coerces_to_running(self):
        messages = []
        publisher = SignedProgressPublisher(messages.append, heartbeat_seconds=60)
        publisher.update("asr", status="suspicious-value", force=True)
        publisher.close()
        self.assertEqual(messages[-1]["status"], "running", "闭集外状态塌缩为 running")

    def test_all_closed_set_statuses_pass_through(self):
        for status in ("running", "waiting", "failed", "completed"):
            messages = []
            publisher = SignedProgressPublisher(messages.append, heartbeat_seconds=60)
            publisher.update("asr", status=status, force=True)
            publisher.close()
            self.assertEqual(messages[-1]["status"], status)


if __name__ == "__main__":
    unittest.main()
