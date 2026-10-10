"""D13-PROD 滚动摘要状态机钉：create_summary 检查点语义复用/降级/末跳 merge."""

from __future__ import annotations

import unittest
from typing import Any

from courselens_worker.streaming_summary import (
    CHECKPOINT_KEY_ANCHORED_MS,
    CHECKPOINT_KEY_COMPLETED,
    CHECKPOINT_KEY_PARTS,
    CHECKPOINT_KEY_PLAN,
    CHECKPOINT_KEY_SKIPPED,
    CODE_ALL_FAILED,
    CODE_MERGE_FAILED,
    RollingSummarizer,
    StreamingSummaryError,
    window_fingerprint,
)


def _segments(start_s: float, end_s: float) -> list[dict[str, Any]]:
    return [
        {"start_ms": int(start_s * 1000), "end_ms": int(end_s * 1000), "text": f"段{start_s}"}
    ]


def _part(mark: str) -> dict[str, Any]:
    return {"markdown": f"# {mark}", "chapters": []}


class _Harness(unittest.TestCase):
    def _summarizer(self, *, refine_results=None, merge_results=None, prior=None, attempts=2):
        self.refine_calls: list[dict] = []
        self.merge_calls: list[dict] = []
        self.refine_cursor = 0
        self.merge_cursor = 0
        refine_results = refine_results if refine_results is not None else [_part("窗1")]
        merge_results = merge_results if merge_results is not None else [_part("终稿")]
        lines: list[str] = []

        def refine(window: dict) -> dict:
            self.refine_calls.append(window)
            result = refine_results[min(self.refine_cursor, len(refine_results) - 1)]
            self.refine_cursor += 1
            if isinstance(result, Exception):
                raise result
            return result

        def merge(payload: dict) -> dict:
            self.merge_calls.append(payload)
            result = merge_results[min(self.merge_cursor, len(merge_results) - 1)]
            self.merge_cursor += 1
            if isinstance(result, Exception):
                raise result
            return result

        summarizer = RollingSummarizer(
            refine=refine,
            merge=merge,
            prior_checkpoint=prior,
            emit=lines.append,
            attempts=attempts,
            retry_backoff_seconds=0.0,
        )
        return summarizer, lines


class RollingJumpTests(_Harness):
    def test_empty_jump_is_noop(self):
        summarizer, _lines = self._summarizer()
        self.assertIsNone(summarizer.jump())
        self.assertEqual(summarizer.completed_windows, 0)

    def test_jump_appends_part_and_advances_plan(self):
        summarizer, _lines = self._summarizer()
        summarizer.buffer_segments(_segments(0, 600))
        part = summarizer.jump()
        self.assertEqual(part["markdown"], "# 窗1")
        self.assertEqual(summarizer.completed_windows, 1)
        self.assertEqual(len(summarizer.window_plan), 1)
        self.assertEqual(summarizer.anchored_ms, 600_000)
        # 窗载荷携转写段与滚动序号（LLM 数据通道）。
        self.assertEqual(self.refine_calls[0]["rolling_index"], 0)
        self.assertEqual(self.refine_calls[0]["transcript"], _segments(0, 600))

    def test_due_threshold_by_audio_span(self):
        summarizer, _lines = self._summarizer()
        summarizer.window_seconds = 600.0
        summarizer.buffer_segments(_segments(0, 300))
        self.assertFalse(summarizer.due())
        summarizer.buffer_segments(_segments(300, 700))
        self.assertTrue(summarizer.due())

    def test_invalid_part_shape_never_enters_parts(self):
        summarizer, lines = self._summarizer(refine_results=[{"markdown": ""}])
        summarizer.buffer_segments(_segments(0, 60))
        self.assertIsNone(summarizer.jump())
        self.assertEqual(summarizer.completed_windows, 0)
        self.assertEqual(summarizer.skipped_windows, 1)
        self.assertEqual(summarizer.parts, [])
        # 遥测闭集：失败行只出计数与码。
        self.assertTrue(any("streaming-summary-jump-failed" in line for line in lines))

    def test_retry_ladder_then_skip_degrade(self):
        failing = StreamingSummaryError(CODE_MERGE_FAILED, "transient")
        calls: list[int] = []

        def flaky(window: dict) -> dict:
            calls.append(1)
            raise failing

        lines: list[str] = []
        summarizer = RollingSummarizer(
            refine=flaky,
            merge=lambda payload: _part("终稿"),
            emit=lines.append,
            attempts=3,
            retry_backoff_seconds=0.0,
        )
        summarizer.buffer_segments(_segments(0, 60))
        self.assertIsNone(summarizer.jump())
        self.assertEqual(len(calls), 3)
        self.assertEqual(summarizer.skipped_windows, 1)


class FinalizeTests(_Harness):
    def test_finalize_merges_parts_and_marks_namespace(self):
        summarizer, _lines = self._summarizer()
        summarizer.buffer_segments(_segments(0, 600))
        summarizer.jump()
        note = summarizer.finalize(title="离散数学")
        self.assertEqual(note["markdown"], "# 终稿")
        self.assertTrue(note["streaming_generated"])
        self.assertEqual(self.merge_calls[0]["title"], "离散数学")
        self.assertEqual(self.merge_calls[0]["parts"], [_part("窗1")])
        self.assertTrue(self.merge_calls[0]["rolling"])

    def test_finalize_without_parts_fails_closed(self):
        summarizer, _lines = self._summarizer()
        with self.assertRaises(StreamingSummaryError) as caught:
            summarizer.finalize(title="离散数学")
        self.assertEqual(caught.exception.code, CODE_ALL_FAILED)

    def test_finalize_invalid_merge_shape_fails_closed(self):
        summarizer, _lines = self._summarizer(merge_results=[{"markdown": "  "}])
        summarizer.buffer_segments(_segments(0, 60))
        summarizer.jump()
        with self.assertRaises(StreamingSummaryError) as caught:
            summarizer.finalize(title="离散数学")
        self.assertEqual(caught.exception.code, CODE_MERGE_FAILED)

    def test_finalize_carries_context_and_glossary(self):
        summarizer, _lines = self._summarizer()
        summarizer.buffer_segments(_segments(0, 60))
        summarizer.jump()
        summarizer.finalize(
            title="离散数学",
            course_context={"course_name": "离散数学"},
            glossary=("图论", "偏序"),
        )
        self.assertEqual(self.merge_calls[0]["course_context"], {"course_name": "离散数学"})
        self.assertEqual(self.merge_calls[0]["glossary"], ["图论", "偏序"])


class CheckpointResumeTests(_Harness):
    def test_checkpoint_uses_streaming_prefixed_keys(self):
        summarizer, _lines = self._summarizer()
        summarizer.buffer_segments(_segments(0, 600))
        summarizer.jump()
        checkpoint = summarizer.checkpoint()
        self.assertEqual(checkpoint[CHECKPOINT_KEY_COMPLETED], 1)
        self.assertEqual(len(checkpoint[CHECKPOINT_KEY_PLAN]), 1)
        self.assertEqual(len(checkpoint[CHECKPOINT_KEY_PARTS]), 1)
        self.assertEqual(checkpoint[CHECKPOINT_KEY_SKIPPED], 0)
        self.assertEqual(checkpoint[CHECKPOINT_KEY_ANCHORED_MS], 600_000)
        # 命名空间共存：批量链键零出现。
        for banned in ("summary_completed_windows", "summary_window_plan", "summary_parts", "raw_rough"):
            self.assertNotIn(banned, checkpoint)

    def test_resume_restores_completed_state(self):
        summarizer, _lines = self._summarizer()
        summarizer.buffer_segments(_segments(0, 600))
        summarizer.jump()
        checkpoint = summarizer.checkpoint()
        resumed, _ = self._summarizer(prior=checkpoint)
        self.assertEqual(resumed.completed_windows, 1)
        self.assertEqual(resumed.window_plan, summarizer.window_plan)
        self.assertEqual(resumed.parts, summarizer.parts)
        self.assertEqual(resumed.anchored_ms, 600_000)
        # 续跑第二窗：plan 追加不重算。
        resumed.buffer_segments(_segments(600, 1200))
        resumed.jump()
        self.assertEqual(resumed.completed_windows, 2)
        self.assertEqual(len(resumed.window_plan), 2)

    def test_resume_drops_corrupt_parts(self):
        prior = {
            CHECKPOINT_KEY_COMPLETED: 2,
            CHECKPOINT_KEY_PLAN: ["a", "b"],
            CHECKPOINT_KEY_PARTS: [_part("窗1"), {"markdown": ""}],
            CHECKPOINT_KEY_SKIPPED: 0,
            CHECKPOINT_KEY_ANCHORED_MS: 1200_000,
        }
        resumed, _lines = self._summarizer(prior=prior)
        self.assertEqual(resumed.completed_windows, 2)
        self.assertEqual(len(resumed.parts), 1)

    def test_window_fingerprint_deterministic_and_content_sensitive(self):
        first = window_fingerprint(_segments(0, 600))
        self.assertEqual(first, window_fingerprint(_segments(0, 600)))
        self.assertNotEqual(first, window_fingerprint(_segments(0, 601)))
        self.assertNotEqual(first, window_fingerprint(_segments(0, 600) * 2))


if __name__ == "__main__":
    unittest.main()
