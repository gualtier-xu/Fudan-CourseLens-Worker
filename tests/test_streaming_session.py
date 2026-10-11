"""D13-PROD 流式会话接线钉：绝不挡课降级/双门收口/snapshot 刷新/幂等."""

from __future__ import annotations

import unittest
from typing import Any

from courselens_worker.streaming_asr import (
    CODE_RECOGNIZER_FAILED,
    StreamingAsrError,
    StreamingTranscriber,
)
from courselens_worker.streaming_session import (
    CHECKPOINT_KEY_INCOMPLETE_CODE,
    CHECKPOINT_KEY_SEGMENTS,
    CHECKPOINT_KEY_STATE,
    CHECKPOINT_KEY_SUMMARY,
    SESSION_STATE_COMPLETED,
    SESSION_STATE_INCOMPLETE,
    SESSION_STATE_RUNNING,
    StreamingLectureSession,
)
from courselens_worker.streaming_summary import RollingSummarizer
from courselens_worker.streaming_trigger import (
    SNAPSHOT_REFRESH_EVENT,
    ClassEndTrigger,
)

SCHEDULED_END = 1_000_000.0


class _ScriptedStream:
    def __init__(self, recognizer):
        self._recognizer = recognizer

    def accept_waveform(self, sample_rate, samples):
        self._recognizer.chunks += 1

    def input_finished(self):
        self._recognizer.finished = True


class _ScriptedRecognizer:
    def __init__(self, *, final_text="终态收尾"):
        self.chunks = 0
        self.decodes = 0
        self.resets = 0
        self.finished = False
        self.final_text = final_text

    def create_stream(self):
        return _ScriptedStream(self)

    def is_ready(self, stream):
        return self.chunks > self.decodes

    def decode_stream(self, stream):
        self.decodes += 1

    def get_result(self, stream):
        if self.finished:
            return self.final_text
        return {1: "", 2: "增量", 3: "增量结果"}.get(self.chunks, "")

    def is_endpoint(self, stream):
        return self.chunks >= 3 and self.resets == 0

    def reset(self, stream):
        self.resets += 1


def _part(mark: str) -> dict[str, Any]:
    return {"markdown": f"# {mark}", "chapters": []}


class _SessionHarness(unittest.TestCase):
    def _session(self, *, recognize=None, merge_results=None) -> StreamingLectureSession:
        events: list[dict[str, Any]] = []
        transcriber = StreamingTranscriber(
            _fake_config(), recognizer=recognize or _ScriptedRecognizer()
        )
        transcriber.create_stream()

        def refine(window: dict) -> dict:
            return _part(f"窗{window['rolling_index']}")

        def merge(payload: dict) -> dict:
            results = merge_results if merge_results is not None else [_part("终稿")]
            return results[0]

        summarizer = RollingSummarizer(
            refine=refine, merge=merge, retry_backoff_seconds=0.0
        )
        trigger = ClassEndTrigger(scheduled_end_epoch=SCHEDULED_END, emit=events.append)
        session = StreamingLectureSession(
            lecture_id="lecture-1",
            transcriber=transcriber,
            summarizer=summarizer,
            trigger=trigger,
            emit=events.append,
        )
        session._test_events = events  # 测试观测面（非产品字段）
        return session

    def _feed(self, session: StreamingLectureSession, count: int = 3) -> None:
        chunk = [0.1] * 16000  # 1s
        for index in range(count):
            session.on_audio(chunk, offset_ms=index * 1000)
        session.summarizer.buffer_segments(
            [{"start_ms": 0, "end_ms": count * 1000, "text": "课上一句"}]
        )
        session.summarizer.jump()


def _fake_config():
    from pathlib import Path

    from courselens_worker.streaming_asr import StreamingAsrConfig

    return StreamingAsrConfig(
        encoder=Path("e"), decoder=Path("d"), tokens=Path("t")
    )


class HappyPathTests(_SessionHarness):
    def test_dual_gate_finalize_produces_transcript_summary_and_refresh(self):
        session = self._session()
        self.assertEqual(session.state, SESSION_STATE_RUNNING)
        self._feed(session)
        # 课表门未到：不收口。
        self.assertFalse(session.on_clock_tick(SCHEDULED_END - 1))
        self.assertEqual(session.state, SESSION_STATE_RUNNING)
        # 翻转门（先 live 后 ended）+ 课表门到点 → 收口。
        session.on_stream_state("live", observed_at_epoch=SCHEDULED_END - 60)
        self.assertFalse(session.on_clock_tick(SCHEDULED_END - 1))
        session.on_stream_state("ended", observed_at_epoch=SCHEDULED_END - 50)
        self.assertTrue(session.on_clock_tick(SCHEDULED_END + 1))
        self.assertEqual(session.state, SESSION_STATE_COMPLETED)
        result = session.result()
        self.assertEqual(result[CHECKPOINT_KEY_STATE], SESSION_STATE_COMPLETED)
        self.assertIsNone(result[CHECKPOINT_KEY_INCOMPLETE_CODE])
        segments = result[CHECKPOINT_KEY_SEGMENTS]
        self.assertTrue(all(item["source"] == "streaming" for item in segments))
        self.assertEqual(segments[0]["text"], "增量结果")
        self.assertEqual(result[CHECKPOINT_KEY_SUMMARY]["markdown"], "# 终稿")
        # snapshot 刷新请求事件（回导非推送）：触发面+收口面各一次。
        refreshes = [
            event
            for event in session._test_events
            if event["event"] == SNAPSHOT_REFRESH_EVENT
        ]
        self.assertEqual(len(refreshes), 2)
        self.assertEqual(
            {event["reason"] for event in refreshes},
            {"class_end", "session_finalized"},
        )

    def test_stream_state_flip_after_schedule_fires(self):
        session = self._session()
        self._feed(session)
        session.on_stream_state("live", observed_at_epoch=SCHEDULED_END - 10)
        self.assertFalse(session.on_clock_tick(SCHEDULED_END + 100))
        self.assertTrue(
            session.on_stream_state("ended", observed_at_epoch=SCHEDULED_END + 700, now_epoch=SCHEDULED_END + 701)
        )
        self.assertEqual(session.state, SESSION_STATE_COMPLETED)

    def test_unknown_stream_state_ignored_never_fires(self):
        session = self._session()
        self._feed(session)
        self.assertFalse(
            session.on_stream_state("recording", now_epoch=SCHEDULED_END + 10)
        )
        self.assertEqual(session.state, SESSION_STATE_RUNNING)

    def test_finalize_idempotent(self):
        session = self._session()
        self._feed(session)
        session.on_stream_state("live")
        session.on_stream_state("ended", observed_at_epoch=1.0)
        first = session.on_clock_tick(SCHEDULED_END + 1)
        self.assertTrue(first)
        state = session.result()[CHECKPOINT_KEY_STATE]
        # 已收口的会话：后续滴答仍如实报告已触发态，状态不再变化。
        self.assertTrue(session.on_clock_tick(SCHEDULED_END + 2))
        self.assertEqual(session.result()[CHECKPOINT_KEY_STATE], state)


class NeverBlockTests(_SessionHarness):
    """绝不挡课：流式腿任何失败闭集降级，绝不向调用方抛。"""

    def test_recognizer_failure_degrades_session_not_raises(self):
        class _Boom(_ScriptedRecognizer):
            def decode_stream(self, stream):
                raise RuntimeError("onnx exploded")

        session = self._session(recognize=_Boom())
        partial = session.on_audio([0.1] * 16000)
        self.assertEqual(partial, "")
        self.assertEqual(session.state, SESSION_STATE_INCOMPLETE)
        self.assertEqual(
            session.result()[CHECKPOINT_KEY_INCOMPLETE_CODE], CODE_RECOGNIZER_FAILED
        )
        # 降级后继续喂音频静默忽略；后续滴答不误报收口（未触发）。
        self.assertEqual(session.on_audio([0.1] * 16000), "")
        self.assertFalse(session.on_clock_tick(SCHEDULED_END + 1))
        self.assertEqual(session.state, SESSION_STATE_INCOMPLETE)

    def test_runtime_unavailable_model_missing_is_closed_code(self):
        """模型缺席（构造腿）闭集码：会话前失败由接线层整体降级。"""
        import os
        import tempfile
        from unittest.mock import patch

        from courselens_worker.streaming_asr import (
            CODE_MODEL_MISSING,
            load_default_config,
        )

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"STREAMING_MODEL_DIR": tmp}):
                with self.assertRaises(StreamingAsrError) as caught:
                    load_default_config()
        self.assertEqual(caught.exception.code, CODE_MODEL_MISSING)

    def test_summary_failure_keeps_transcript_and_closes_cleanly(self):
        session = self._session(merge_results=[{"markdown": ""}])
        self._feed(session)
        session.on_stream_state("live")
        session.on_stream_state("ended", observed_at_epoch=1.0)
        self.assertTrue(session.on_clock_tick(SCHEDULED_END + 1))
        result = session.result()
        # 转写稿保住，摘要腿闭集降级：绝不伪造空笔记。
        self.assertEqual(session.state, SESSION_STATE_INCOMPLETE)
        self.assertEqual(
            result[CHECKPOINT_KEY_INCOMPLETE_CODE], "streaming_summary_merge_failed"
        )
        self.assertTrue(result[CHECKPOINT_KEY_SEGMENTS])
        self.assertIsNone(result[CHECKPOINT_KEY_SUMMARY])

    def test_degrade_entry_is_idempotent_and_keeps_segments(self):
        session = self._session()
        self._feed(session)
        session.degrade("live_playlist_http")
        self.assertEqual(session.state, SESSION_STATE_INCOMPLETE)
        self.assertEqual(
            session.result()[CHECKPOINT_KEY_INCOMPLETE_CODE], "live_playlist_http"
        )
        session.degrade("live_segment_http")
        self.assertEqual(
            session.result()[CHECKPOINT_KEY_INCOMPLETE_CODE], "live_playlist_http"
        )
        self.assertTrue(session.result()[CHECKPOINT_KEY_SEGMENTS])


class NamespaceTests(_SessionHarness):
    def test_checkpoint_keys_all_streaming_prefixed(self):
        session = self._session()
        self._feed(session)
        checkpoint = session.checkpoint()
        for key in checkpoint:
            self.assertTrue(
                key.startswith("streaming_"), f"non-namespaced key: {key}"
            )
        self.assertIn("streaming_segments", checkpoint)
        self.assertIn("streaming_summary_completed_windows", checkpoint)


if __name__ == "__main__":
    unittest.main()
