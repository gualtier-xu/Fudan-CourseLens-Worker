"""D13-IMPL 流式转写 POC 三态钉：模型加载 / 增量中间结果 / 终态结果.

真实模型测试在 sherpa-onnx 运行时与 int8 流式 Paraformer 模型（POC 手工
获取，见 courselens_worker/streaming_asr_poc.py 模块文档）齐备时运行；
二者任一缺席即跳过——CI 与 client venv（conftest 桩）只跑纯逻辑钉，
绝不伪造识别结果（fail-closed 跳过语义）。
"""

from __future__ import annotations

import os
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from courselens_worker import streaming_asr_poc as poc_mod
from courselens_worker.streaming_asr_poc import (
    DEFAULT_MODEL_DIRNAME,
    PocConfig,
    PocError,
    StreamingAsrPoc,
    chunk_pcm,
    default_model_dir,
    resolve_model_paths,
    sherpa_runtime_available,
    synthesize_speech_pcm,
)


def _real_model_available() -> tuple[bool, str]:
    if not sherpa_runtime_available():
        return False, "real sherpa-onnx runtime required (stub or missing)"
    try:
        resolve_model_paths()
    except PocError as exc:
        return False, str(exc)
    return True, "streaming paraformer model + runtime available"


class ConfigResolutionTests(unittest.TestCase):
    def test_default_model_dir_env_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"COURSELENS_STREAMING_POC_MODEL_DIR": tmp}):
                self.assertEqual(default_model_dir(), Path(tmp))

    def test_default_model_dir_falls_back_to_lane_scratch(self):
        with patch.dict(os.environ, {"COURSELENS_STREAMING_POC_MODEL_DIR": ""}):
            directory = default_model_dir()
        self.assertEqual(directory.name, DEFAULT_MODEL_DIRNAME)
        self.assertEqual(directory.parent.name, "models")
        self.assertEqual(directory.parent.parent.name, ".tmp-d13impl")

    def test_resolve_model_paths_fails_closed_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PocError) as caught:
                resolve_model_paths(Path(tmp))
        self.assertIn("model files missing", str(caught.exception))

    def test_config_defaults_pin_streaming_shape(self):
        encoder, decoder, tokens = Path("e"), Path("d"), Path("t")
        config = PocConfig(encoder=encoder, decoder=decoder, tokens=tokens)
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.feature_dim, 80)
        self.assertEqual(config.num_threads, 2)
        self.assertEqual(config.chunk_seconds, 1.0)
        self.assertEqual(config.decoding_method, "greedy_search")
        self.assertTrue(config.enable_endpoint)
        self.assertAlmostEqual(config.rule2_min_trailing_silence, 1.2)


class SyntheticAudioTests(unittest.TestCase):
    def test_deterministic_same_seed_same_samples(self):
        first = synthesize_speech_pcm(3.0, seed=7)
        second = synthesize_speech_pcm(3.0, seed=7)
        self.assertEqual(first, second)

    def test_length_and_amplitude_range(self):
        samples = synthesize_speech_pcm(2.5, seed=3)
        self.assertEqual(len(samples), int(2.5 * 16000))
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in samples))

    def test_burst_gap_energy_contrast(self):
        samples = synthesize_speech_pcm(
            6.0, seed=5, burst_seconds=2.0, gap_seconds=2.0
        )
        burst = samples[: 2 * 16000]
        gap = samples[2 * 16000 : 4 * 16000]

        def rms(chunk):
            return (sum(value * value for value in chunk) / len(chunk)) ** 0.5

        self.assertGreater(rms(burst), 10 * max(rms(gap), 1e-6))

    def test_chunk_pcm_fixed_sizes(self):
        samples = list(range(2500))
        chunks = chunk_pcm(samples, 0.1, sample_rate=16000)
        self.assertEqual([len(chunk) for chunk in chunks], [1600, 900])

    def test_chunk_pcm_rejects_nonpositive(self):
        with self.assertRaises(PocError):
            chunk_pcm([0.0], 0.0)


class _FakeStream:
    def __init__(self, recognizer):
        self._recognizer = recognizer
        self.accepted: list[list[float]] = []

    def accept_waveform(self, sample_rate, samples):
        self.accepted.append(list(samples))
        self._recognizer.accepted_chunks += 1

    def input_finished(self):
        self._recognizer.finished = True


class _FakeRecognizer:
    """脚本化流式识别替身：第 3 块后触发端点，final 给出收尾文本。"""

    def __init__(self):
        self.accepted_chunks = 0
        self.decode_calls = 0
        self.resets = 0
        self.finished = False
        self.stream = _FakeStream(self)

    def create_stream(self):
        return self.stream

    def is_ready(self, stream):
        return self.accepted_chunks > self.decode_calls

    def decode_stream(self, stream):
        self.decode_calls += 1

    def get_result(self, stream):
        if self.finished:
            return "终态收尾文本"
        return {1: "", 2: "增量中间", 3: "增量中间结果"}.get(self.accepted_chunks, "")

    def is_endpoint(self, stream):
        return self.accepted_chunks >= 3 and self.resets == 0

    def reset(self, stream):
        self.resets += 1


class LoopLogicTests(unittest.TestCase):
    """不依赖 sherpa-onnx：喂入→增量→端点→终态循环逻辑钉。"""

    def _feed_chunks(self, count: int, fake=None):
        fake = fake or _FakeRecognizer()
        config = PocConfig(
            encoder=Path("e"), decoder=Path("d"), tokens=Path("t"), chunk_seconds=0.5
        )
        poc = StreamingAsrPoc(config, recognizer=fake)
        stream = poc.create_stream()
        chunk = [0.1] * 8000  # 0.5s @16kHz
        steps = [poc.feed_samples(chunk, sample_rate=16000) for _ in range(count)]
        return poc, fake, stream, steps

    def test_load_state_factory_invoked_with_config(self):
        fake = _FakeRecognizer()
        config = PocConfig(encoder=Path("e"), decoder=Path("d"), tokens=Path("t"))
        calls: list[PocConfig] = []
        StreamingAsrPoc(config, recognizer_factory=lambda cfg: calls.append(cfg) or fake)
        self.assertEqual(calls, [config])

    def test_load_state_records_metrics_without_model(self):
        poc, fake, _stream, _steps = self._feed_chunks(1)
        self.assertIsNotNone(fake)
        self.assertIsInstance(poc.metrics.model_load_seconds, float)
        self.assertGreaterEqual(poc.metrics.model_load_seconds, 0.0)

    def test_incremental_state_partial_before_finish(self):
        poc, _fake, _stream, steps = self._feed_chunks(2)
        # 增量语义：喂入过程中即可取到中间结果（第 2 块起非空），无需 finish，
        # 也未触发端点（端点脚本在第 3 块之后）。
        self.assertEqual(steps[0].partial_text, "")
        self.assertTrue(steps[1].partial_text)
        self.assertGreaterEqual(poc.metrics.decode_calls, 2)
        self.assertAlmostEqual(poc.metrics.audio_seconds_fed, 1.0)
        self.assertIsNotNone(poc.metrics.first_partial_seconds)
        self.assertEqual(poc.metrics.endpoint_count, 0)

    def test_endpoint_state_captures_segment_and_resets(self):
        poc, fake, _stream, steps = self._feed_chunks(3)
        endpoint_steps = [step for step in steps if step.endpoint_triggered]
        self.assertEqual(len(endpoint_steps), 1)
        self.assertEqual(fake.resets, 1)
        self.assertEqual(
            [(seg.text, seg.trigger) for seg in poc.segments],
            [("增量中间结果", "endpoint")],
        )
        self.assertEqual(poc.metrics.endpoint_count, 1)

    def test_final_state_captures_tail_segment(self):
        poc, fake, _stream, _steps = self._feed_chunks(3)
        final = poc.finish()
        self.assertTrue(fake.finished)
        self.assertIsNotNone(final)
        self.assertEqual(final.text, "终态收尾文本")
        self.assertEqual(final.trigger, "final")
        self.assertEqual(
            [(seg.text, seg.trigger) for seg in poc.segments],
            [("增量中间结果", "endpoint"), ("终态收尾文本", "final")],
        )
        self.assertEqual(poc.metrics.final_segment_count, 1)
        self.assertGreaterEqual(poc.metrics.process_cpu_seconds, 0.0)

    def test_metrics_rtf_against_audio_seconds(self):
        poc, _fake, _stream, _steps = self._feed_chunks(3)
        poc.finish()
        self.assertAlmostEqual(poc.metrics.audio_seconds_fed, 1.5)
        self.assertIsNotNone(poc.metrics.rtf)
        self.assertGreater(poc.metrics.rtf, 0.0)
        self.assertEqual(len(poc.metrics.step_wall_seconds), 3)

    def test_feed_before_create_stream_fails_closed(self):
        config = PocConfig(encoder=Path("e"), decoder=Path("d"), tokens=Path("t"))
        poc = StreamingAsrPoc(config, recognizer=_FakeRecognizer())
        with self.assertRaises(PocError):
            poc.feed_samples([0.0] * 1600)


class RealStreamingParaformerTests(unittest.TestCase):
    """真实模型三态钉（模型+运行时缺席=跳过，绝不伪造）。"""

    _shared_recognizer = None

    @classmethod
    def setUpClass(cls):
        available, reason = _real_model_available()
        if not available:
            raise unittest.SkipTest(reason)
        encoder, decoder, tokens = resolve_model_paths()
        config = PocConfig(encoder=encoder, decoder=decoder, tokens=tokens)
        cls._shared_recognizer = poc_mod._build_recognizer(config)

    def _make_poc(self) -> StreamingAsrPoc:
        encoder, decoder, tokens = resolve_model_paths()
        config = PocConfig(encoder=encoder, decoder=decoder, tokens=tokens)
        return StreamingAsrPoc(config, recognizer=self._shared_recognizer)

    def _wav_samples(self) -> tuple[list[float], int]:
        wav_path = resolve_model_paths()[0].parent / "test_wavs" / "0.wav"
        return poc_mod.load_wav_mono(wav_path)

    def test_real_model_load_state(self):
        # 加载态用独立实例真实走一次构造（setUpClass 已证可加载）。
        encoder, decoder, tokens = resolve_model_paths()
        config = PocConfig(encoder=encoder, decoder=decoder, tokens=tokens)
        poc = StreamingAsrPoc(config)
        self.assertIsNotNone(poc.recognizer)
        self.assertGreater(poc.metrics.model_load_seconds, 0.0)
        self.assertGreater(poc.metrics.rss_after_load_bytes or 0, 0)

    def test_real_model_incremental_state(self):
        samples, rate = self._wav_samples()
        poc = self._make_poc()
        poc.create_stream()
        # 只喂前 2 秒真实人声，不 finish——增量中间结果必须先于终态出现。
        steps = poc.feed_chunks(
            chunk_pcm(samples[: rate * 2], 0.5, sample_rate=rate)[:4]
        )
        partial = steps[-1].partial_text
        self.assertTrue(partial and partial.strip())
        self.assertIsNotNone(poc.metrics.first_partial_seconds)
        self.assertGreater(poc.metrics.decode_calls, 0)

    def test_real_model_final_state(self):
        samples, rate = self._wav_samples()
        poc = self._make_poc()
        poc.create_stream()
        poc.feed_chunks(chunk_pcm(samples, 0.5, sample_rate=rate))
        final = poc.finish()
        self.assertIsNotNone(final)
        self.assertTrue(final.text.strip())
        self.assertEqual(final.trigger, "final")
        self.assertGreater(poc.metrics.audio_seconds_fed, 8.0)
        self.assertIsNotNone(poc.metrics.rtf)

    def test_real_model_endpoint_on_trailing_silence(self):
        samples, rate = self._wav_samples()
        poc = self._make_poc()
        poc.create_stream()
        poc.feed_chunks(chunk_pcm(samples, 0.5, sample_rate=rate))
        # 人声后接 3 秒数字静音（> rule2 1.2s）→ 端点必触发且段已捕获。
        silence = [0.0] * (rate * 3)
        steps = poc.feed_chunks(chunk_pcm(silence, 0.5))
        endpoint_steps = [step for step in steps if step.endpoint_triggered]
        self.assertTrue(endpoint_steps)
        self.assertEqual(poc.segments[-1].trigger, "endpoint")
        self.assertTrue(poc.segments[-1].text.strip())


class WavLoaderTests(unittest.TestCase):
    def test_load_wav_mono_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                frames = [1000, -1000, 2000, -2000]
                handle.writeframes(struct.pack("<4h", *frames))
            samples, rate = poc_mod.load_wav_mono(path)
            self.assertEqual(rate, 16000)
            self.assertEqual(len(samples), 2)
            self.assertAlmostEqual(samples[0], 0.0, places=5)
            self.assertAlmostEqual(samples[1], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
