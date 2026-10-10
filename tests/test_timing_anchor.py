"""SUBTITLE-DEEP-1 Phase C: dual-timestamp anchor correction pins.

Deterministic pairing, bounded shift, interpolation smoothing, kill switch,
and integration into the subtitle chain (mocked pool, zero media).
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

from courselens_worker import asr


def seg(start_ms, end_ms, text):
    return {"start_ms": start_ms, "end_ms": end_ms, "text": text}


class AnchorCorrectTimingTests(unittest.TestCase):
    def test_shifts_matched_segment_toward_official_midpoint(self):
        segments = [seg(100_000, 110_000, "识别段")]
        official = [seg(104_000, 114_000, "官方行")]
        stats = asr._anchor_correct_timing(segments, official)
        # 官方中点 109000 vs 识别中点 105000 → +4000
        self.assertEqual(segments[0]["start_ms"], 104_000)
        self.assertEqual(segments[0]["end_ms"], 114_000)
        self.assertEqual(stats["matched"], 1)
        self.assertEqual(stats["shifted"], 1)
        self.assertEqual(stats["max_shift_ms"], 4000)

    def test_small_delta_within_snap_does_not_move(self):
        segments = [seg(100_000, 110_000, "识别段")]
        official = [seg(100_500, 110_500, "官方行")]
        stats = asr._anchor_correct_timing(segments, official)
        self.assertEqual(segments[0]["start_ms"], 100_000)
        # 直配计数含小位移段（它们被记录但不移动，也不参与后续被邻居位移）
        self.assertEqual(stats["matched"], 1)
        self.assertEqual(stats["shifted"], 0)

    def test_shift_is_capped(self):
        segments = [seg(100_000, 110_000, "识别段")]
        # 官方行长且中点远超帽：重叠 5s 满足配对，位移被帽到 ±8s
        official = [seg(105_000, 500_000, "超长官方行")]
        stats = asr._anchor_correct_timing(segments, official)
        duration = 110_000 - 100_000
        self.assertEqual(segments[0]["end_ms"] - segments[0]["start_ms"], duration)
        self.assertEqual(
            segments[0]["end_ms"] - 110_000, asr.TIME_ANCHOR_MAX_SHIFT_MS
        )
        self.assertEqual(stats["max_shift_ms"], asr.TIME_ANCHOR_MAX_SHIFT_MS)

    def test_unmatched_segment_interpolates_between_matched_neighbors(self):
        segments = [
            seg(0, 10_000, "左"),
            seg(20_000, 30_000, "中（无配对）"),
            seg(40_000, 50_000, "右"),
        ]
        official = [
            seg(2_000, 12_000, "左官方"),
            seg(45_000, 55_000, "右官方"),
        ]
        asr._anchor_correct_timing(segments, official)
        self.assertEqual(segments[0]["start_ms"], 2_000)
        self.assertEqual(segments[2]["start_ms"], 45_000)
        # 中段：左右位移 +2000/+5000 → 均值 +3500
        self.assertEqual(segments[1]["start_ms"], 23_500)

    def test_low_overlap_is_not_matched(self):
        segments = [seg(100_000, 110_000, "识别段")]
        official = [seg(109_500, 112_000, "擦边官方行")]
        stats = asr._anchor_correct_timing(segments, official)
        # 重叠 500ms < 800ms 门槛 → 不动
        self.assertEqual(segments[0]["start_ms"], 100_000)
        self.assertEqual(stats["matched"], 0)

    def test_empty_rows_or_segments_noop(self):
        segments = [seg(0, 1000, "x")]
        self.assertEqual(asr._anchor_correct_timing(segments, [])["matched"], 0)
        self.assertEqual(asr._anchor_correct_timing([], [seg(0, 1000, "y")])["matched"], 0)
        self.assertEqual(segments[0]["start_ms"], 0)


class AnchorChainIntegrationTests(unittest.TestCase):
    def _run(self, platform_rows, *, env=None):
        pool = Mock()
        pool.transcribe_pcm.side_effect = lambda _path, backend, *, offset_seconds: [{
            "start_ms": int(offset_seconds * 1000) + 100_000,
            "end_ms": int(offset_seconds * 1000) + 110_000,
            "text": f"{backend}@{int(offset_seconds)}",
        }]

        def create_pcm(_url, target, *, offset, duration):
            target.write_bytes(b"pcm-bytes")

        payload = {
            "mode": "automatic",
            "media": {"url": "https://media.example.com/lecture.mp4", "duration_seconds": 1250},
            "platform_transcript": platform_rows,
        }
        environ = {"SUBTITLE_BACKENDS": "sensevoice,paraformer"}
        environ.update(env or {})
        with (
            patch.object(asr, "RecognizerPool", return_value=pool),
            patch.object(asr, "pinned_media_proxy"),
            patch.object(asr, "_prefetch_media_pcm",
                         side_effect=lambda _u, t, *, duration: t.write_bytes(b"")),
            patch.object(asr, "_slice_pcm_chunk", side_effect=create_pcm),
            patch.dict(os.environ, environ, clear=False),
        ):
            return asr.transcribe(
                {"payload": payload},
                sensevoice_dir=Mock(),
                paraformer_dir=Mock(),
                proofread=Mock(return_value=[{"start_ms": 0, "end_ms": 1000, "text": "校对后"}]),
                progress=Mock(),
            )

    def test_platform_rows_anchor_corrects_refined_timing(self):
        official = [seg(0, 600_000, "官方大段")]
        result = self._run(official)
        # 官方中点 300000；识别段 [100000,110000] 中点 105000 → +8000 帽
        raw = result["raw_paraformer"]
        self.assertEqual(raw[0]["end_ms"] - 110_000, asr.TIME_ANCHOR_MAX_SHIFT_MS)
        self.assertEqual(
            result["metrics"]["timing_anchor"]["shifted"], 3, "三个 chunk 全部校正"
        )

    def test_kill_switch_disables_anchor_correction(self):
        official = [seg(0, 600_000, "官方大段")]
        result = self._run(official, env={"COURSELENS_SUBTITLE_TIME_ANCHOR": "0"})
        self.assertEqual(result["raw_paraformer"][0]["start_ms"], 100_000)
        self.assertNotIn("timing_anchor", result["metrics"])

    def test_no_platform_rows_skips_stage(self):
        result = self._run([])
        self.assertEqual(result["raw_paraformer"][0]["start_ms"], 100_000)
        self.assertNotIn("timing_anchor", result["metrics"])


if __name__ == "__main__":
    unittest.main()
