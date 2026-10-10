"""AS12 platform-first rough-leg pins (第五十二案): transcript fetch, chain
selection matrix, provenance forms, resume-guard chain identity, and the
kill switch.  Fake recognizers and stubbed platform calls only; no school
contact, no media, no model download.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

from courselens_worker import asr
from courselens_worker.platform_session import (
    TRANSCRIPT_RESPONSE_MAX_BYTES,
    TRANSCRIPT_RECORD_STORM_LIMIT,
    PlatformSession,
    PlatformSessionError,
    _job_needs_rough_transcript,
    materialize_job_sources,
)


def _transcript_row(index: int, span_ms: int = 60_000) -> dict:
    start = index * span_ms
    return {"start_ms": start, "end_ms": start + span_ms, "text": f"平台段{index}"}


class PlatformTranscriptCoverageTests(unittest.TestCase):
    def test_sentence_pauses_merge_so_natural_gaps_never_break_coverage(self):
        # U6 语义：句级 cue 间 1-5s 自然停顿粘合，不再把覆盖压穿阈值
        # （u2c 产品路径 E2E 抓到的洞：裸并集把语音占比当内容缺失）。
        sentences = []
        clock = 0
        for _ in range(20):
            sentences.append({"start_ms": clock, "end_ms": clock + 4_000, "text": "句"})
            clock += 5_000  # 1s 停顿
        coverage = asr.platform_transcript_coverage(sentences, 100_000)
        self.assertGreaterEqual(coverage, 0.8)

    def test_large_holes_still_count_as_missing_content(self):
        with_hole = [{"start_ms": 0, "end_ms": 100_000, "text": "前半"}]
        with_hole.append({"start_ms": 700_000, "end_ms": 800_000, "text": "后半"})
        coverage = asr.platform_transcript_coverage(with_hole, 800_000)
        self.assertLess(coverage, 0.5)

    def test_explicit_zero_gap_keeps_pure_union_semantics(self):
        segments = [
            {"start_ms": 0, "end_ms": 600, "text": "a"},
            {"start_ms": 500, "end_ms": 900, "text": "b"},
            {"start_ms": 1900, "end_ms": 2600, "text": "c"},
        ]
        # merge_gap_ms=0：[0,900]+[1900,2600)→clip 2000 → 1000/2000
        self.assertEqual(
            asr.platform_transcript_coverage(segments, 2000, merge_gap_ms=0), 0.5
        )

    def test_empty_text_rows_and_degenerate_intervals_never_cover(self):
        segments = [
            {"start_ms": 0, "end_ms": 500, "text": "  "},
            {"start_ms": 0, "end_ms": 0, "text": "a"},
            {"start_ms": 800, "end_ms": 300, "text": "b"},
        ]
        self.assertEqual(asr.platform_transcript_coverage(segments, 1000), 0.0)

    def test_zero_duration_and_missing_rows_are_zero(self):
        self.assertEqual(asr.platform_transcript_coverage([], 1000), 0.0)
        self.assertEqual(asr.platform_transcript_coverage([_transcript_row(0)], 0), 0.0)


class RoughTranscriptFetchTests(unittest.TestCase):
    def _session(self) -> PlatformSession:
        return PlatformSession(transport="requests")

    def test_rows_convert_second_stamps_to_milliseconds(self):
        session = self._session()
        payload = {"code": 0, "list": [{"all_content": [
            {"BeginSec": 3, "EndSec": 7, "Text": " 你好 "},
            {"BeginSec": 8, "Text": "无结束秒"},
            {"BeginSec": 9, "EndSec": 5, "Text": "倒挂段"},
            {"Text": "缺起始"},
            {"BeginSec": "bad", "Text": "坏形状"},
            {"BeginSec": 10, "EndSec": 11, "Text": ""},
        ]}]}
        with patch.object(session, "_course_json", return_value=payload) as course_json:
            rows = session.rough_transcript_segments("s1")
        course_json.assert_called_once_with(
            "/courseapi/v3/web-socket/search-trans-result",
            params={"sub_id": "s1", "format": "json"},
            max_bytes=TRANSCRIPT_RESPONSE_MAX_BYTES,
        )
        self.assertEqual(rows, [
            {"start_ms": 3000, "end_ms": 7000, "text": "你好"},
            {"start_ms": 8000, "end_ms": 8000, "text": "无结束秒"},
            {"start_ms": 9000, "end_ms": 9000, "text": "倒挂段"},
            {"start_ms": 0, "end_ms": 0, "text": "缺起始"},
        ])

    def test_nonzero_code_empty_list_and_missing_content_fall_back_to_empty(self):
        session = self._session()
        with patch.object(session, "_course_json", return_value={"code": 401}):
            with self.assertRaises(PlatformSessionError):
                session.rough_transcript_segments("s1")
        with patch.object(session, "_course_json", return_value={"code": 0, "list": []}):
            self.assertEqual(session.rough_transcript_segments("s1"), [])
        with patch.object(session, "_course_json", return_value={"code": 0, "list": [{}]}):
            self.assertEqual(session.rough_transcript_segments("s1"), [])

    def test_record_storm_fails_closed(self):
        session = self._session()
        content = [
            {"BeginSec": index, "EndSec": index + 1, "Text": "t"}
            for index in range(TRANSCRIPT_RECORD_STORM_LIMIT + 1)
        ]
        with patch.object(
            session, "_course_json",
            return_value={"code": 0, "list": [{"all_content": content}]},
        ):
            with self.assertRaises(PlatformSessionError):
                session.rough_transcript_segments("s1")


class MaterializeRoughTranscriptTests(unittest.TestCase):
    def _job(self, kind: str, requested=None) -> dict:
        job = {
            "job_kind": kind,
            "payload": {
                "source_session": {"course_id": "c1", "sub_id": "s1"},
            },
            "secrets": {"source_credentials": {"account": "a", "password": "p"}},
        }
        if requested is not None:
            job["requested_outputs"] = requested
        return job

    def _run(self, job, rows=None, error=None):
        with (
            patch.object(PlatformSession, "login", return_value=None),
            patch.object(PlatformSession, "close", return_value=None),
            patch.object(
                PlatformSession, "rough_transcript_segments",
                side_effect=error,
                return_value=rows if rows is not None else [],
            ) as fetch,
        ):
            result = materialize_job_sources(job)
        return result, fetch

    def test_subtitle_job_fetches_and_carries_rows(self):
        rows = [_transcript_row(0)]
        result, fetch = self._run(self._job("subtitle"), rows=rows)
        fetch.assert_called_once_with("s1")
        self.assertEqual(result["payload"]["platform_transcript"], rows)
        self.assertEqual(
            result["payload"]["platform_transcript_state"], "transcript_fetched"
        )

    def test_learning_pack_fetches_only_when_subtitle_requested(self):
        _result, fetch = self._run(
            self._job("learning_pack", requested=["summary"])
        )
        fetch.assert_not_called()
        result, fetch = self._run(
            self._job("learning_pack", requested=["summary", "subtitle"]),
            rows=[_transcript_row(0)],
        )
        fetch.assert_called_once_with("s1")
        self.assertIn("platform_transcript", result["payload"])

    def test_summary_job_never_fetches(self):
        _result, fetch = self._run(self._job("summary"))
        fetch.assert_not_called()

    def test_fetch_failure_downgrades_to_closed_state_without_raising(self):
        result, _fetch = self._run(
            self._job("subtitle"),
            error=PlatformSessionError("platform_course_request_failed"),
        )
        self.assertEqual(
            result["payload"]["platform_transcript_state"],
            "transcript_fetch_failed",
        )
        self.assertNotIn("platform_transcript", result["payload"])

    def test_empty_transcript_records_empty_state(self):
        result, _fetch = self._run(self._job("subtitle"), rows=[])
        self.assertEqual(
            result["payload"]["platform_transcript_state"], "transcript_empty"
        )

    def test_job_needs_rough_transcript_matrix(self):
        self.assertTrue(_job_needs_rough_transcript({"job_kind": "subtitle"}))
        self.assertTrue(_job_needs_rough_transcript({
            "job_kind": "learning_pack", "requested_outputs": ["subtitle"],
        }))
        self.assertFalse(_job_needs_rough_transcript({
            "job_kind": "learning_pack", "requested_outputs": ["ocr"],
        }))
        self.assertFalse(_job_needs_rough_transcript({"job_kind": "echo"}))
        self.assertFalse(_job_needs_rough_transcript({}))


class PlatformTranscriptMalformedRowTests(unittest.TestCase):
    """夜10-C T4：异常文稿形态的覆盖度/归一化免疫（低覆盖走闭集回落，绝不崩溃）。"""

    def test_coverage_skips_malformed_rows_instead_of_crashing(self):
        rows = [
            None,
            "not-a-row",
            {"start_ms": "bad", "end_ms": 100, "text": "坏形状"},
            {"start_ms": 0, "end_ms": 500, "text": "好段"},
        ]
        self.assertEqual(asr.platform_transcript_coverage(rows, 1000), 0.5)

    def test_coverage_treats_numeric_strings_as_valid(self):
        rows = [{"start_ms": "0", "end_ms": "400", "text": "字符串秒"}]
        self.assertEqual(asr.platform_transcript_coverage(rows, 1000), 0.4)

    def test_normalize_segments_skips_malformed_rows_and_repairs_end(self):
        from courselens_worker.formats import normalize_segments

        rows = [
            None,
            {"start_ms": "bad", "end_ms": 900, "text": "坏起始"},
            {"start_ms": 0, "end_ms": "bad", "text": "坏结束"},
            {"start_ms": 1000, "end_ms": 2000, "text": "  好段  "},
        ]
        cleaned = normalize_segments(rows)
        self.assertEqual([row["text"] for row in cleaned], ["坏结束", "好段"])
        self.assertEqual(cleaned[0]["start_ms"], 0)
        self.assertEqual(cleaned[0]["end_ms"], 1000, "结束时间戳不可解析按缺省修复")
        self.assertEqual(cleaned[0]["start_ms"] <= cleaned[0]["end_ms"], True)

    def test_normalize_segments_sorting_unchanged_for_valid_rows(self):
        from courselens_worker.formats import normalize_segments

        rows = [
            {"start_ms": 2000, "end_ms": 3000, "text": "乙"},
            {"start_ms": 0, "end_ms": 1000, "text": "甲"},
        ]
        self.assertEqual(
            [row["text"] for row in normalize_segments(rows)], ["甲", "乙"]
        )


class _FakeStream:
    def __init__(self) -> None:
        self.waveform = None
        self.result = None

    def accept_waveform(self, sample_rate, samples) -> None:
        self.waveform = None


class PlatformRoughSourceChainTests(unittest.TestCase):
    """platform-first / 回落 / 杀开关 / 续跑守卫 整链矩阵（伪 pool，零媒体）。"""

    def _pool(self, ran_backends: list[str]):
        pool = Mock()
        def _transcribe(_path, backend, *, offset_seconds):
            ran_backends.append((int(offset_seconds), backend))
            return [{
                "start_ms": int(offset_seconds * 1000),
                "end_ms": int(offset_seconds * 1000) + 1000,
                "text": f"{backend}@{int(offset_seconds)}",
            }]
        pool.transcribe_pcm.side_effect = _transcribe
        return pool

    def _run(
        self,
        pool,
        *,
        payload_extra=None,
        prior=None,
        env=None,
        proofread="mock",
        telemetry=None,
    ):
        def prefetch(_url, target, *, duration):
            target.write_bytes(b"")

        def create_pcm(_full, target, *, offset, duration):
            target.write_bytes(b"pcm-bytes")

        payload = {
            "mode": "automatic",
            "media": {
                "url": "https://media.example.com/lecture.mp4",
                "duration_seconds": 1250,
            },
        }
        payload.update(payload_extra or {})
        if prior is not None:
            payload["checkpoint"] = prior
        proofread_fn = (
            Mock(return_value=[{"start_ms": 0, "end_ms": 1000, "text": "校对后"}])
            if proofread == "mock" else proofread
        )
        environment = {"SUBTITLE_BACKENDS": "sensevoice,paraformer"}
        environment.update(env or {})
        emit = telemetry if telemetry is not None else []
        with (
            patch.object(asr, "RecognizerPool", return_value=pool),
            patch.object(asr, "pinned_media_proxy"),
            patch.object(asr, "_prefetch_media_pcm", side_effect=prefetch),
            patch.object(asr, "_slice_pcm_chunk", side_effect=create_pcm),
            patch.object(asr, "_emit_telemetry", side_effect=lambda line: emit.append(line)),
            patch.dict(os.environ, environment),
        ):
            return asr.transcribe(
                {"payload": payload},
                sensevoice_dir=Mock(),
                paraformer_dir=Mock(),
                proofread=proofread_fn,
                progress=Mock(),
                checkpoint=None,
            )

    def _full_rows(self) -> list[dict]:
        return [_transcript_row(index) for index in range(20)]  # 0..1_200_000ms

    def test_platform_first_skips_rough_leg_and_stamps_honest_provenance(self):
        ran: list = []
        telemetry: list[str] = []
        result = self._run(
            self._pool(ran),
            payload_extra={
                "platform_transcript": self._full_rows(),
                "platform_transcript_state": "transcript_fetched",
            },
            telemetry=telemetry,
        )
        self.assertEqual(
            [backend for _offset, backend in ran], ["paraformer", "paraformer", "paraformer"]
        )
        self.assertEqual(result["metrics"]["rough_source"], "platform")
        self.assertNotIn("rough_source_fallback_reason", result["metrics"])
        self.assertEqual(
            result["segments"][0]["provenance"]["model"],
            "paraformer+platform:proofread",
        )
        self.assertEqual(
            result["raw_sensevoice"][0]["provenance"]["model"],
            "platform:transcript",
        )
        self.assertEqual(
            result["raw_paraformer"][0]["provenance"]["model"], "paraformer"
        )
        self.assertTrue(
            any(line.startswith("stage=rough-source source=platform") for line in telemetry)
        )

    def test_platform_alternates_reach_proofread_slot_normalized(self):
        proofread = Mock(return_value=[{"start_ms": 0, "end_ms": 1000, "text": "校对后"}])
        ran: list = []
        self._run(
            self._pool(ran),
            payload_extra={"platform_transcript": self._full_rows()},
            proofread=proofread,
        )
        alternates = proofread.call_args[0][0]
        self.assertEqual(len(alternates), 20)
        self.assertEqual(alternates[0]["text"], "平台段0")
        self.assertEqual(alternates[0]["start_ms"], 0)

    def test_low_coverage_falls_back_to_full_dual_chain(self):
        ran: list = []
        telemetry: list[str] = []
        result = self._run(
            self._pool(ran),
            payload_extra={
                "platform_transcript": [_transcript_row(0)],  # 60s / 1250s
                "platform_transcript_state": "transcript_fetched",
            },
            telemetry=telemetry,
        )
        self.assertEqual(
            sorted(backend for _offset, backend in ran),
            ["paraformer", "paraformer", "paraformer", "sensevoice", "sensevoice", "sensevoice"],
        )
        self.assertEqual(result["metrics"]["rough_source"], "sensevoice")
        self.assertEqual(
            result["metrics"]["rough_source_fallback_reason"], "coverage_low"
        )
        self.assertEqual(
            result["segments"][0]["provenance"]["model"],
            "sensevoice+paraformer:proofread",
        )
        self.assertTrue(
            any("reason=coverage_low" in line for line in telemetry)
        )

    def test_coverage_threshold_boundary_is_inclusive(self):
        # 夜10-C：裁决边界含等号（>=0.8 恰好达标走平台链）；用 0.5/0.5 精确验证
        ran: list = []
        with patch.object(asr, "PLATFORM_TRANSCRIPT_MIN_COVERAGE", 0.5):
            result = self._run(
                self._pool(ran),
                payload_extra={
                    "platform_transcript": [
                        {"start_ms": 0, "end_ms": 625_000, "text": "恰好半讲"},
                    ],
                    "platform_transcript_state": "transcript_fetched",
                },
            )
        self.assertEqual(result["metrics"]["rough_source"], "platform")
        self.assertNotIn("rough_source_fallback_reason", result["metrics"])

    def test_resume_rejects_platform_chain_when_rows_are_lost(self):
        # 夜10-C 反向失配：平台链检查点 + 平台行在续跑时丢失 → 拒绝混链
        prior = {
            "completed_chunks": 1,
            "total_chunks": 3,
            "mode": "automatic",
            "backends": ["sensevoice", "paraformer"],
            "rough_source": "platform",
            "raw_sensevoice": [{"start_ms": 0, "end_ms": 1000, "text": "粗"}],
            "raw_paraformer": [{"start_ms": 0, "end_ms": 1000, "text": "精"}],
        }
        with self.assertRaises(asr.ASRError):
            self._run(
                self._pool([]),
                payload_extra={"platform_transcript_state": "transcript_fetch_failed"},
                prior=prior,
            )

    def test_kill_switch_env_forces_legacy_chain(self):
        ran: list = []
        telemetry: list[str] = []
        result = self._run(
            self._pool(ran),
            payload_extra={
                "platform_transcript": self._full_rows(),
                "platform_transcript_state": "transcript_fetched",
            },
            env={"COURSELENS_ASR_ROUGH_SOURCE": "sensevoice"},
            telemetry=telemetry,
        )
        self.assertEqual(len(ran), 6)
        self.assertEqual(result["metrics"]["rough_source"], "sensevoice")
        self.assertEqual(
            result["metrics"]["rough_source_fallback_reason"], "env_disabled"
        )

    def test_state_fallback_reasons_are_closed_set_members(self):
        for state, reason in (
            ("transcript_empty", "transcript_empty"),
            ("transcript_fetch_failed", "transcript_fetch_failed"),
            (None, "platform_transcript_missing"),
        ):
            ran: list = []
            extra = {}
            if state is not None:
                extra["platform_transcript_state"] = state
            result = self._run(self._pool(ran), payload_extra=extra)
            self.assertEqual(result["metrics"]["rough_source"], "sensevoice")
            self.assertEqual(
                result["metrics"]["rough_source_fallback_reason"], reason, state
            )
            self.assertEqual(len(ran), 6)

    def test_platform_first_runs_single_leg_at_full_threads_even_parallel(self):
        ran: list = []
        result = self._run(
            self._pool(ran),
            payload_extra={"platform_transcript": self._full_rows()},
            env={"COURSELENS_ASR_STRATEGY": "parallel"},
        )
        self.assertEqual(
            [backend for _offset, backend in ran], ["paraformer"] * 3
        )
        self.assertEqual(result["metrics"]["threads_per_model"], 4)

    def test_platform_first_checkpoint_carries_chain_identity(self):
        checkpoints: list[dict] = []
        ran: list = []

        def checkpoint_writer(value: dict) -> None:
            checkpoints.append(value)

        payload = {
            "mode": "automatic",
            "media": {
                "url": "https://media.example.com/lecture.mp4",
                "duration_seconds": 1250,
            },
            "platform_transcript": self._full_rows(),
            "platform_transcript_state": "transcript_fetched",
        }
        pool = self._pool(ran)
        with (
            patch.object(asr, "RecognizerPool", return_value=pool),
            patch.object(asr, "pinned_media_proxy"),
            patch.object(asr, "_prefetch_media_pcm",
                         side_effect=lambda _u, t, *, duration: t.write_bytes(b"")),
            patch.object(asr, "_slice_pcm_chunk",
                         side_effect=lambda _f, t, *, offset, duration: t.write_bytes(b"p")),
            patch.dict(os.environ, {"SUBTITLE_BACKENDS": "sensevoice,paraformer"}),
        ):
            asr.transcribe(
                {"payload": payload},
                sensevoice_dir=Mock(),
                paraformer_dir=Mock(),
                proofread=Mock(return_value=[{"start_ms": 0, "end_ms": 1000, "text": "校"}]),
                progress=Mock(),
                checkpoint=checkpoint_writer,
            )
        self.assertTrue(checkpoints)
        for state in checkpoints:
            self.assertEqual(state["rough_source"], "platform")
            self.assertEqual(state["backends"], ["sensevoice", "paraformer"])
            self.assertEqual(
                {key for key in state if key.startswith("raw_")},
                {"raw_sensevoice", "raw_paraformer"},
            )

    def test_resume_with_matching_platform_chain_continues(self):
        ran: list = []
        prior = {
            "completed_chunks": 1,
            "total_chunks": 3,
            "mode": "automatic",
            "backends": ["sensevoice", "paraformer"],
            "rough_source": "platform",
            "raw_sensevoice": [_transcript_row(0)],
            "raw_paraformer": [{"start_ms": 0, "end_ms": 1000, "text": "精"}],
        }
        result = self._run(
            self._pool(ran),
            payload_extra={"platform_transcript": self._full_rows()},
            prior=prior,
        )
        self.assertEqual(result["metrics"]["rough_source"], "platform")
        # 续跑只补剩余两块
        self.assertEqual(len(ran), 2)

    def test_resume_rejects_rough_source_mismatch(self):
        prior = {
            "completed_chunks": 1,
            "total_chunks": 3,
            "mode": "automatic",
            "backends": ["sensevoice", "paraformer"],
            "rough_source": "sensevoice",
            "raw_sensevoice": [{"start_ms": 0, "end_ms": 1000, "text": "粗"}],
            "raw_paraformer": [{"start_ms": 0, "end_ms": 1000, "text": "精"}],
        }
        with self.assertRaises(asr.ASRError):
            self._run(
                self._pool([]),
                payload_extra={"platform_transcript": self._full_rows()},
                prior=prior,
            )

    def test_resume_rejects_backend_sequence_mismatch(self):
        prior = {
            "completed_chunks": 1,
            "total_chunks": 3,
            "mode": "automatic",
            "backends": ["paraformer", "sensevoice"],
            "rough_source": "sensevoice",
            "raw_sensevoice": [{"start_ms": 0, "end_ms": 1000, "text": "粗"}],
            "raw_paraformer": [{"start_ms": 0, "end_ms": 1000, "text": "精"}],
        }
        with self.assertRaises(asr.ASRError):
            self._run(self._pool([]), prior=prior)

    def test_legacy_checkpoint_completes_on_legacy_chain_with_reason(self):
        ran: list = []
        prior = {
            "completed_chunks": 1,
            "total_chunks": 3,
            "mode": "automatic",
            "raw_sensevoice": [{"start_ms": 0, "end_ms": 1000, "text": "粗"}],
            "raw_paraformer": [{"start_ms": 0, "end_ms": 1000, "text": "精"}],
        }
        result = self._run(
            self._pool(ran),
            payload_extra={"platform_transcript": self._full_rows()},
            prior=prior,
        )
        # 旧检查点：剩余块沿旧双模链，不混交替源来源
        self.assertEqual(
            sorted(backend for _offset, backend in ran),
            ["paraformer", "paraformer", "sensevoice", "sensevoice"],
        )
        self.assertEqual(result["metrics"]["rough_source"], "sensevoice")
        self.assertEqual(
            result["metrics"]["rough_source_fallback_reason"], "legacy_checkpoint"
        )

    def test_no_proofread_marks_rough_source_not_applicable(self):
        ran: list = []
        result = self._run(
            self._pool(ran),
            payload_extra={"platform_transcript": self._full_rows()},
            proofread=None,
        )
        self.assertEqual(result["metrics"]["rough_source"], "not_applicable")
        self.assertNotIn("rough_source_fallback_reason", result["metrics"])


if __name__ == "__main__":
    unittest.main()
