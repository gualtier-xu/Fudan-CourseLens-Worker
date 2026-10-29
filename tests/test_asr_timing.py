"""Synthetic PCM timing tests: silence-aware regions, tokens, evidence IDs.

Real NumPy arrays and fake recognizers only; no model, media, or network.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from courselens_worker import asr
from courselens_worker.formats import normalize_segments, to_srt, to_vtt
from shared.evidence_contract import NAMESPACE_SEGMENT, compute_id


SAMPLE_RATE = asr.SAMPLE_RATE


def _tone(start: float, end: float, total: float, amplitude: float = 0.4) -> np.ndarray:
    samples = np.zeros(int(total * SAMPLE_RATE), dtype=np.float32)
    samples[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)] = amplitude
    return samples


def _write_pcm(directory: Path, name: str, samples: np.ndarray) -> Path:
    path = Path(directory) / name
    np.asarray(samples, dtype=np.float32).tofile(path)
    return path


class _FakeStream:
    def __init__(self) -> None:
        self.waveform = None
        self.result = None

    def accept_waveform(self, sample_rate, samples) -> None:
        # 与真实 sherpa 绑定一致：拷贝波形，避免保留 memmap 视图锁住文件
        self.waveform = np.array(samples, dtype=np.float32)


class _FakeRecognizer:
    """Amplitude-driven fake: voiced waveforms decode to a fixed text."""

    def __init__(self, text: str = "合成语音", result_builder=None) -> None:
        self.text = text
        self.result_builder = result_builder
        self.streams: list[_FakeStream] = []
        self.batches: list[list[_FakeStream]] = []

    def create_stream(self) -> _FakeStream:
        stream = _FakeStream()
        self.streams.append(stream)
        return stream

    def decode_streams(self, streams) -> None:
        self.batches.append(list(streams))
        for stream in streams:
            if self.result_builder is not None:
                stream.result = self.result_builder(stream.waveform)
                continue
            peak = float(np.max(np.abs(stream.waveform))) if stream.waveform.size else 0.0
            stream.result = SimpleNamespace(text=self.text if peak > 0.05 else "")


class VoicedRegionTests(unittest.TestCase):
    def test_silence_produces_no_regions(self):
        self.assertEqual(asr.detect_voiced_regions(np.zeros(5 * SAMPLE_RATE, dtype=np.float32)), [])

    def test_continuous_speech_degrades_to_one_bounded_region(self):
        regions = asr.detect_voiced_regions(_tone(0.0, 30.0, 30.0))
        self.assertEqual(len(regions), 1)
        start, end = regions[0]
        self.assertEqual(start, 0)
        self.assertGreaterEqual(end, int(29.9 * SAMPLE_RATE))
        self.assertLessEqual(end, 30 * SAMPLE_RATE)

    def test_two_speech_islands_yield_two_ordered_disjoint_regions(self):
        regions = asr.detect_voiced_regions(_tone(1.0, 2.0, 6.0) + _tone(4.0, 5.0, 6.0))
        self.assertEqual(len(regions), 2)
        first_start, first_end = regions[0]
        second_start, second_end = regions[1]
        self.assertLessEqual(first_end, second_start)
        # 锚点落在语音岛附近（含 bounded padding），而不是整个窗口
        self.assertGreaterEqual(first_start, int(0.8 * SAMPLE_RATE))
        self.assertLessEqual(first_start, int(1.0 * SAMPLE_RATE))
        self.assertGreaterEqual(first_end, int(2.0 * SAMPLE_RATE))
        self.assertLessEqual(first_end, int(2.2 * SAMPLE_RATE))
        self.assertGreaterEqual(second_start, int(3.8 * SAMPLE_RATE))
        self.assertLessEqual(second_start, int(4.0 * SAMPLE_RATE))
        self.assertGreaterEqual(second_end, int(5.0 * SAMPLE_RATE))
        self.assertLessEqual(second_end, int(5.2 * SAMPLE_RATE))

    def test_short_gap_below_merge_tolerance_is_merged(self):
        regions = asr.detect_voiced_regions(_tone(1.0, 1.5, 3.0) + _tone(1.7, 2.2, 3.0))
        self.assertEqual(len(regions), 1)
        start, end = regions[0]
        self.assertLessEqual(start, int(0.9 * SAMPLE_RATE))
        self.assertGreaterEqual(end, int(2.3 * SAMPLE_RATE))

    def test_regions_shorter_than_minimum_are_dropped(self):
        samples = _tone(1.0, 1.05, 5.0) + _tone(2.0, 2.5, 5.0) + _tone(4.0, 4.5, 5.0)
        regions = asr.detect_voiced_regions(samples)
        self.assertEqual(len(regions), 2)

    def test_regions_exceeding_maximum_duration_are_split(self):
        regions = asr.detect_voiced_regions(
            _tone(0.0, 3.0, 3.0),
            pad_seconds=0.0,
            max_region_seconds=1.0,
        )
        self.assertEqual(len(regions), 3)
        for start, end in regions:
            self.assertLessEqual(end - start, int(1.0 * SAMPLE_RATE))
        self.assertEqual(regions[0][0], 0)
        for previous, current in zip(regions, regions[1:]):
            self.assertEqual(previous[1], current[0])

    def test_trailing_pad_never_leaves_the_window(self):
        regions = asr.detect_voiced_regions(_tone(0.0, 3.0, 3.0))
        self.assertEqual(len(regions), 1)
        self.assertLessEqual(regions[0][1], 3 * SAMPLE_RATE)


class CalibrationKnobTests(unittest.TestCase):
    def test_invalid_values_fall_back_to_the_safe_default(self):
        for raw in ("banana", "0.1", "50", "nan", "1e400", "", "   "):
            with patch.dict(os.environ, {asr.ASR_ENERGY_RATIO_ENV: raw}):
                self.assertEqual(asr.asr_energy_ratio(), asr.ASR_ENERGY_RATIO_DEFAULT, raw)
        with patch.dict(os.environ, clear=True):
            self.assertEqual(asr.asr_energy_ratio(), asr.ASR_ENERGY_RATIO_DEFAULT)
        with patch.dict(os.environ, {asr.ASR_ENERGY_RATIO_ENV: "4.5"}):
            self.assertEqual(asr.asr_energy_ratio(), 4.5)


class TranscribePcmTimingTests(unittest.TestCase):
    def _transcribe(self, samples: np.ndarray, *, offset_seconds: float = 0.0, recognizer=None):
        recognizer = recognizer or _FakeRecognizer()
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_pcm(Path(temporary), "chunk.f32le", samples)
            pool = asr.RecognizerPool.__new__(asr.RecognizerPool)
            pool.threads = 1
            pool._recognizers = {}
            pool.sensevoice_dir = Path("sensevoice")
            with patch.object(asr.RecognizerPool, "get", return_value=recognizer):
                return pool.transcribe_pcm(path, "sensevoice", offset_seconds=offset_seconds), recognizer

    def test_anchors_follow_speech_regions_not_the_window(self):
        segments, recognizer = self._transcribe(_tone(5.0, 8.0, 30.0))  # speech 5-8s, silence elsewhere
        self.assertEqual(len(recognizer.streams), 1)
        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertGreaterEqual(segment["start_ms"], 4700)
        self.assertLessEqual(segment["start_ms"], 5000)
        self.assertGreaterEqual(segment["end_ms"], 8000)
        self.assertLessEqual(segment["end_ms"], 8300)

    def test_absolute_nonzero_offset_is_preserved(self):
        segments, _ = self._transcribe(_tone(5.0, 8.0, 30.0), offset_seconds=600.0)
        self.assertEqual(len(segments), 1)
        self.assertGreaterEqual(segments[0]["start_ms"], 604700)
        self.assertLessEqual(segments[0]["start_ms"], 605000)

    def test_silence_decodes_nothing(self):
        segments, recognizer = self._transcribe(np.zeros(30 * SAMPLE_RATE, dtype=np.float32))
        self.assertEqual(segments, [])
        self.assertEqual(recognizer.streams, [])

    def test_continuous_speech_stays_one_bounded_segment(self):
        segments, _ = self._transcribe(np.full(30 * SAMPLE_RATE, 0.4, dtype=np.float32))
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["start_ms"], 0)
        self.assertGreaterEqual(segments[0]["end_ms"], 29_900)

    def test_well_shaped_native_tokens_become_absolute_optional_timing(self):
        def builder(_waveform):
            return SimpleNamespace(text="你好", tokens=["你", "好"], timestamps=[0.1, 0.5])

        segments, _ = self._transcribe(
            _tone(0.0, 5.0, 30.0) + _tone(5.0, 8.0, 30.0), recognizer=_FakeRecognizer(result_builder=builder),
        )
        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual(segment["tokens"], [
            ["你", segment["start_ms"] + 100, None],
            ["好", segment["start_ms"] + 500, None],
        ])

    def test_malformed_native_timing_is_omitted_entirely(self):
        cases = (
            SimpleNamespace(text="你好", tokens=["你", "好"], timestamps=[0.5, 0.1]),  # descending
            SimpleNamespace(text="你好", tokens=["你"], timestamps=[0.1, 0.5]),  # length mismatch
            SimpleNamespace(text="你好", tokens=["你", "好"], timestamps=[0.1, "x"]),  # unparsable
            SimpleNamespace(text="你好", tokens=["你", "好"], timestamps=[0.1, 99.0]),  # outside anchors
            SimpleNamespace(text="你好", tokens=None, timestamps=None),  # absent
        )
        for result in cases:
            with self.subTest(result=type(result)):
                segments, _ = self._transcribe(
                    _tone(0.0, 5.0, 30.0) + _tone(5.0, 8.0, 30.0),
                    recognizer=_FakeRecognizer(result_builder=lambda _waveform, result=result: result),
                )
                self.assertEqual(len(segments), 1)
                self.assertNotIn("tokens", segments[0])


class _TaggedRecognizer(_FakeRecognizer):
    """Tags streams in creation order and decodes to a per-stream text."""

    def create_stream(self) -> _FakeStream:
        stream = super().create_stream()
        stream.tag = len(self.streams)
        return stream

    def decode_streams(self, streams) -> None:
        self.batches.append(list(streams))
        for stream in streams:
            stream.result = SimpleNamespace(text=f"段{stream.tag:02d}")


class _LegacyRecognizer:
    """Pre-decode_streams sherpa shape: one stream at a time, no batches."""

    def __init__(self) -> None:
        self.streams: list[SimpleNamespace] = []
        self.decoded: list[SimpleNamespace] = []

    def create_stream(self) -> SimpleNamespace:
        stream = SimpleNamespace(result=None, tag=len(self.streams) + 1, waveform=None)

        def accept_waveform(_sample_rate, samples, _stream=stream):
            _stream.waveform = np.array(samples, dtype=np.float32)

        stream.accept_waveform = accept_waveform
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream) -> None:
        stream.result = SimpleNamespace(text=f"段{stream.tag:02d}")
        self.decoded.append(stream)


class DecodeBatchingTests(unittest.TestCase):
    """decode_streams memory is bounded: activation size scales with the
    total audio seconds of one call, so a 600s chunk of continuous speech
    decoded as one batch exhausted 16GB hosted runners (ASRMEM-1)."""

    def _transcribe(self, samples: np.ndarray, recognizer):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_pcm(Path(temporary), "chunk.f32le", samples)
            pool = asr.RecognizerPool.__new__(asr.RecognizerPool)
            pool.threads = 1
            pool._recognizers = {}
            pool.sensevoice_dir = Path("sensevoice")
            with patch.object(asr.RecognizerPool, "get", return_value=recognizer):
                return pool.transcribe_pcm(path, "sensevoice", offset_seconds=0.0)

    def _island_window(self) -> np.ndarray:
        # 30s window: two ~10s speech islands; every window yields the same
        # two regions, so N windows give 2N streams of known durations.
        return _tone(0.0, 10.0, 30.0) + _tone(15.0, 25.0, 30.0)

    def test_batched_calls_never_exceed_the_seconds_cap(self):
        samples = np.concatenate([self._island_window() for _ in range(3)])  # 90s, 6 streams
        recognizer = _FakeRecognizer()
        segments = self._transcribe(samples, recognizer)
        self.assertEqual(len(recognizer.streams), 6)
        self.assertEqual(len(recognizer.batches), 3)
        decoded = [stream for batch in recognizer.batches for stream in batch]
        self.assertEqual(len(decoded), 6)
        self.assertEqual(len({id(stream) for stream in decoded}), 6)
        for batch in recognizer.batches:
            seconds = sum(len(stream.waveform) for stream in batch) / SAMPLE_RATE
            self.assertGreater(seconds, 0.0)
            self.assertLessEqual(seconds, asr.ASR_DECODE_BATCH_SECONDS)
        self.assertEqual(len(segments), 6)

    def test_continuous_speech_no_longer_decodes_one_whole_chunk_per_call(self):
        # Two full-window regions: the old single decode_streams call held
        # 60s of audio in one activation; the cap must split it in two.
        samples = np.full(60 * SAMPLE_RATE, 0.4, dtype=np.float32)
        recognizer = _FakeRecognizer()
        segments = self._transcribe(samples, recognizer)
        self.assertEqual(len(recognizer.streams), 2)
        self.assertEqual(len(recognizer.batches), 2)
        for batch in recognizer.batches:
            self.assertEqual(len(batch), 1)
            seconds = len(batch[0].waveform) / SAMPLE_RATE
            self.assertLessEqual(seconds, asr.ASR_DECODE_BATCH_SECONDS)
        self.assertEqual(len(segments), 2)

    def test_segment_order_is_stable_across_batches(self):
        samples = np.concatenate([self._island_window() for _ in range(3)])
        recognizer = _TaggedRecognizer()
        segments = self._transcribe(samples, recognizer)
        self.assertEqual(len(recognizer.batches), 3)
        self.assertEqual(
            [segment["text"] for segment in segments],
            [f"段{index:02d}" for index in range(1, 7)],
        )
        starts = [segment["start_ms"] for segment in segments]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(len(set(starts)), len(starts))

    def test_legacy_recognizers_still_decode_stream_by_stream(self):
        samples = np.concatenate([self._island_window() for _ in range(2)])
        recognizer = _LegacyRecognizer()
        segments = self._transcribe(samples, recognizer)
        self.assertEqual(len(recognizer.streams), 4)
        self.assertEqual(len(recognizer.decoded), 4)
        self.assertEqual(
            [stream.tag for stream in recognizer.decoded],
            [1, 2, 3, 4],
        )
        self.assertEqual(
            [segment["text"] for segment in segments],
            [f"段{index:02d}" for index in range(1, 5)],
        )


class NormalizeEvidenceTests(unittest.TestCase):
    def test_valid_overlap_is_preserved_not_clamped(self):
        merged = normalize_segments([
            {"start_ms": 1500, "end_ms": 3000, "text": "乙"},
            {"start_ms": 1000, "end_ms": 2000, "text": "甲"},
        ])
        self.assertEqual(
            [(item["start_ms"], item["end_ms"]) for item in merged],
            [(1000, 2000), (1500, 3000)],
        )

    def test_empty_text_is_dropped_and_nonpositive_duration_repaired(self):
        merged = normalize_segments([
            {"start_ms": 0, "end_ms": 1000, "text": "   "},
            {"start_ms": 100, "end_ms": 100, "text": "甲"},
            {"start_ms": 200, "end_ms": 0, "text": "乙"},
        ])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["end_ms"], 300)
        self.assertEqual(merged[1]["end_ms"], 1200)

    def test_evidence_metadata_survives_normalization(self):
        merged = normalize_segments([{
            "start_ms": 0, "end_ms": 1000, "text": "第一句",
            "segment_id": "seg:0123456789ab",
            "source_hash": "a" * 64,
            "provenance": {"producer": "courselens-worker", "model": "sensevoice"},
            "tokens": [["第", 10, None]],
            "lang": "zh",
            "unsupported": "dropped",
        }])
        self.assertEqual(merged[0]["segment_id"], "seg:0123456789ab")
        self.assertEqual(merged[0]["source_hash"], "a" * 64)
        self.assertEqual(merged[0]["provenance"]["model"], "sensevoice")
        self.assertEqual(merged[0]["tokens"], [["第", 10, None]])
        self.assertEqual(merged[0]["lang"], "zh")
        self.assertNotIn("unsupported", merged[0])

    def test_srt_and_vtt_textual_format_is_unchanged(self):
        segments = [{"start_ms": 1000, "end_ms": 2000, "text": "第一句"}]
        self.assertEqual(
            to_srt(segments),
            "1\n00:00:01,000 --> 00:00:02,000\n第一句\n",
        )
        self.assertEqual(
            to_vtt(segments),
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n第一句\n",
        )


class FingerprintChainTests(unittest.TestCase):
    def test_chain_is_deterministic_and_resume_reproduces_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "a.f32le"
            second = Path(temporary) / "b.f32le"
            other = Path(temporary) / "c.f32le"
            first.write_bytes(b"chunk-one")
            second.write_bytes(b"chunk-two")
            other.write_bytes(b"chunk-alt")
            first_digest = asr._pcm_file_digest(first)
            second_digest = asr._pcm_file_digest(second)
            other_digest = asr._pcm_file_digest(other)
            fresh = asr._advance_pcm_fingerprint(
                asr._advance_pcm_fingerprint(None, first_digest), second_digest
            )
            # 断点续跑：checkpoint 只携带第一个 chunk 之后的 hex 状态
            checkpoint_state = asr._advance_pcm_fingerprint(None, first_digest)
            resumed = asr._advance_pcm_fingerprint(checkpoint_state, second_digest)
            self.assertEqual(fresh, resumed)
            self.assertEqual(
                fresh,
                asr._advance_pcm_fingerprint(
                    asr._advance_pcm_fingerprint(None, first_digest), second_digest
                ),
            )
            self.assertNotEqual(
                fresh,
                asr._advance_pcm_fingerprint(
                    asr._advance_pcm_fingerprint(None, other_digest), second_digest
                ),
            )


class SegmentIdentityTests(unittest.TestCase):
    def test_stamped_ids_match_the_contract_identity_and_are_deterministic(self):
        fingerprint = "ab" * 32
        source_id = asr._source_evidence_id(fingerprint, 12.5)
        self.assertTrue(source_id.startswith("src:"))
        provenance = {
            "producer": asr.PRODUCER_ID,
            "model": "sensevoice",
            "config_hash": asr._timing_config_hash(asr.ASR_ENERGY_RATIO_DEFAULT),
        }
        first = [{"start_ms": 1000, "end_ms": 2000, "text": "第一句"}]
        asr._stamp_segment_identity(first, source_id=source_id, source_hash=fingerprint, provenance=provenance)
        second = [{"start_ms": 1000, "end_ms": 2000, "text": "第一句"}]
        asr._stamp_segment_identity(second, source_id=source_id, source_hash=fingerprint, provenance=provenance)
        self.assertEqual(first[0]["segment_id"], second[0]["segment_id"])
        expected = compute_id(NAMESPACE_SEGMENT, {
            "source_id": source_id,
            "start_ms": 1000,
            "end_ms": 2000,
            "text": "第一句",
            "lang": None,
            "no_speech": False,
            "producer": provenance["producer"],
            "model": provenance["model"],
            "config_hash": provenance["config_hash"],
        })
        self.assertEqual(first[0]["segment_id"], expected)
        self.assertEqual(first[0]["source_hash"], fingerprint)
        self.assertEqual(first[0]["provenance"], provenance)

    def test_changed_text_or_config_changes_identity(self):
        fingerprint = "cd" * 32
        source_id = asr._source_evidence_id(fingerprint, 12.5)
        provenance = {
            "producer": asr.PRODUCER_ID,
            "model": "sensevoice",
            "config_hash": asr._timing_config_hash(3.0),
        }
        first = [{"start_ms": 1000, "end_ms": 2000, "text": "第一句"}]
        asr._stamp_segment_identity(first, source_id=source_id, source_hash=fingerprint, provenance=provenance)
        second = [{"start_ms": 1000, "end_ms": 2000, "text": "第二句"}]
        asr._stamp_segment_identity(second, source_id=source_id, source_hash=fingerprint, provenance=provenance)
        self.assertNotEqual(first[0]["segment_id"], second[0]["segment_id"])
        other_config = dict(provenance, config_hash=asr._timing_config_hash(4.0))
        third = [{"start_ms": 1000, "end_ms": 2000, "text": "第一句"}]
        asr._stamp_segment_identity(third, source_id=source_id, source_hash=fingerprint, provenance=other_config)
        self.assertNotEqual(first[0]["segment_id"], third[0]["segment_id"])

    def test_config_hash_is_bounded_hex_and_knob_sensitive(self):
        base = asr._timing_config_hash(3.0)
        self.assertRegex(base, r"^[0-9a-f]{12,64}$")
        self.assertNotEqual(base, asr._timing_config_hash(4.0))


if __name__ == "__main__":
    unittest.main()
