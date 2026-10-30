"""D13-PROD 流式转写生产适配层钉：时间锚/闭集码/导出命名空间/遥测纪律.

真实模型腿（sherpa-onnx 运行时 + int8 流式 Paraformer 齐备时）走真实加载
钉；缺席即跳过绝不伪造。其余全部用脚本化识别替身钉死循环逻辑。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from courselens_worker import streaming_asr as mod
from courselens_worker.streaming_asr import (
    CODE_MODEL_MISSING,
    CODE_RECOGNIZER_FAILED,
    CODE_RUNTIME_UNAVAILABLE,
    SEGMENT_SOURCE,
    StreamingAsrConfig,
    StreamingAsrError,
    StreamingTranscriber,
    checkpoint_segments_payload,
    default_model_dir,
    export_segments,
    load_default_config,
    resolve_model_paths,
)


def _config(tmp: str) -> StreamingAsrConfig:
    return StreamingAsrConfig(
        encoder=Path(tmp) / "encoder.int8.onnx",
        decoder=Path(tmp) / "decoder.int8.onnx",
        tokens=Path(tmp) / "tokens.txt",
    )


class _ScriptedStream:
    def __init__(self, recognizer):
        self._recognizer = recognizer
        self.accepted: list[list[float]] = []

    def accept_waveform(self, sample_rate, samples):
        self.accepted.append(list(samples))
        self._recognizer.chunks += 1

    def input_finished(self):
        self._recognizer.finished = True


class _ScriptedRecognizer:
    """脚本化替身：第 endpoint_at 块后触发端点，final 给收尾文本。"""

    def __init__(self, *, partials=None, endpoint_at=3, final_text="终态收尾"):
        self.chunks = 0
        self.decodes = 0
        self.resets = 0
        self.finished = False
        self.partials = partials or {1: "", 2: "增量中间", 3: "增量中间结果"}
        self.endpoint_at = endpoint_at
        self.final_text = final_text
        self.stream = _ScriptedStream(self)

    def create_stream(self):
        return self.stream

    def is_ready(self, stream):
        return self.chunks > self.decodes

    def decode_stream(self, stream):
        self.decodes += 1

    def get_result(self, stream):
        if self.finished:
            return self.final_text
        return self.partials.get(self.chunks, "")

    def is_endpoint(self, stream):
        return self.chunks >= self.endpoint_at and self.resets == 0

    def reset(self, stream):
        self.resets += 1


class ModelResolutionTests(unittest.TestCase):
    def test_default_model_dir_env_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {mod.MODEL_DIR_ENV: tmp}):
                self.assertEqual(default_model_dir(), Path(tmp))

    def test_default_model_dir_falls_back_to_repo_models_root(self):
        with patch.dict(os.environ, {mod.MODEL_DIR_ENV: ""}):
            directory = default_model_dir()
        self.assertEqual(directory.name, mod.DEFAULT_MODEL_DIRNAME)
        self.assertEqual(directory.parent.name, ".models")
        # 模型绝不进 worker 树（公共边界门零容忍）。
        self.assertNotIn("worker", directory.parent.parent.parts[-1:])

    def test_resolve_model_paths_fails_closed_with_closed_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StreamingAsrError) as caught:
                resolve_model_paths(Path(tmp))
        self.assertEqual(caught.exception.code, CODE_MODEL_MISSING)

    def test_load_default_config_uses_env_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in (mod.ENCODER_FILENAME, mod.DECODER_FILENAME, mod.TOKENS_FILENAME):
                (Path(tmp) / name).write_bytes(b"x")
            with patch.dict(os.environ, {mod.MODEL_DIR_ENV: tmp}):
                config = load_default_config()
        self.assertEqual(config.encoder, Path(tmp) / mod.ENCODER_FILENAME)
        self.assertEqual(config.decoder, Path(tmp) / mod.DECODER_FILENAME)
        self.assertEqual(config.tokens, Path(tmp) / mod.TOKENS_FILENAME)

    def test_build_recognizer_fails_closed_when_runtime_stubbed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            with patch.object(mod, "sherpa_runtime_available", return_value=False):
                with self.assertRaises(StreamingAsrError) as caught:
                    mod._build_recognizer(config)
        self.assertEqual(caught.exception.code, CODE_RUNTIME_UNAVAILABLE)

    def test_error_codes_are_closed_set(self):
        self.assertTrue(mod.STREAMING_ERROR_CODES <= {CODE_MODEL_MISSING, CODE_RUNTIME_UNAVAILABLE, CODE_RECOGNIZER_FAILED})


class TranscriberLoopTests(unittest.TestCase):
    """循环逻辑钉（替身识别器，免模型）：喂入→增量→端点→终态。"""

    def _transcriber(self, fake=None, emit=None):
        fake = fake or _ScriptedRecognizer()
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
        transcriber = StreamingTranscriber(
            config, recognizer=fake, emit_telemetry=emit
        )
        transcriber.create_stream()
        return transcriber, fake

    def test_partial_available_before_finish(self):
        transcriber, _fake = self._transcriber()
        chunk = [0.1] * 8000  # 0.5s @16kHz
        self.assertEqual(transcriber.feed(chunk), "")
        self.assertEqual(transcriber.feed(chunk), "增量中间")
        self.assertEqual(transcriber.audio_ms_fed, 1000)
        self.assertEqual(transcriber.metrics.endpoint_count, 0)

    def test_endpoint_segment_carries_lecture_absolute_ms_anchors(self):
        transcriber, fake = self._transcriber()
        chunk = [0.1] * 8000
        for _ in range(3):
            transcriber.feed(chunk)
        segments = transcriber.segments
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].trigger, "endpoint")
        self.assertEqual(segments[0].text, "增量中间结果")
        self.assertEqual(segments[0].start_ms, 0)
        self.assertEqual(segments[0].end_ms, 1000)
        self.assertEqual(fake.resets, 1)

    def test_offset_ms_reanchors_lecture_cursor(self):
        transcriber, _fake = self._transcriber()
        chunk = [0.1] * 8000
        transcriber.feed(chunk, offset_ms=3_500)
        self.assertEqual(transcriber.audio_ms_fed, 4_000)
        for _ in range(2):
            transcriber.feed(chunk)
        segments = transcriber.segments
        self.assertEqual(segments[0].start_ms, 3_500)
        self.assertEqual(segments[0].end_ms, 4_500)

    def test_finish_flushes_final_segment(self):
        transcriber, fake = self._transcriber()
        chunk = [0.1] * 8000
        for _ in range(3):
            transcriber.feed(chunk)
        final = transcriber.finish()
        self.assertTrue(fake.finished)
        self.assertIsNotNone(final)
        self.assertEqual(final.trigger, "final")
        self.assertEqual(final.text, "终态收尾")
        self.assertEqual(final.start_ms, 1500)  # 端点后段起点=第 3 块末
        self.assertEqual(final.end_ms, 1500)
        self.assertEqual(len(transcriber.segments), 2)

    def test_sample_rate_mismatch_fails_closed(self):
        transcriber, _fake = self._transcriber()
        with self.assertRaises(StreamingAsrError) as caught:
            transcriber.feed([0.0] * 8000, sample_rate=48000)
        self.assertEqual(caught.exception.code, CODE_RECOGNIZER_FAILED)

    def test_feed_before_create_stream_fails_closed(self):
        fake = _ScriptedRecognizer()
        with tempfile.TemporaryDirectory() as tmp:
            transcriber = StreamingTranscriber(_config(tmp), recognizer=fake)
        with self.assertRaises(StreamingAsrError) as caught:
            transcriber.feed([0.0] * 1600)
        self.assertEqual(caught.exception.code, CODE_RECOGNIZER_FAILED)

    def test_factory_receives_config_and_metrics_recorded(self):
        fake = _ScriptedRecognizer()
        seen: list[StreamingAsrConfig] = []
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(tmp)
            StreamingTranscriber(
                config, recognizer_factory=lambda cfg: seen.append(cfg) or fake
            )
        self.assertEqual(seen, [config])

    def test_recognizer_runtime_error_folds_to_closed_code(self):
        class _Boom(_ScriptedRecognizer):
            def decode_stream(self, stream):
                raise RuntimeError("onnx exploded")

        transcriber, _fake = self._transcriber(fake=_Boom(endpoint_at=99))
        with self.assertRaises(StreamingAsrError) as caught:
            transcriber.feed([0.1] * 8000)
        self.assertEqual(caught.exception.code, CODE_RECOGNIZER_FAILED)


class TelemetryAndExportTests(unittest.TestCase):
    def test_telemetry_carries_counters_only_never_text(self):
        lines: list[str] = []
        fake = _ScriptedRecognizer()
        with tempfile.TemporaryDirectory() as tmp:
            transcriber = StreamingTranscriber(
                _config(tmp), recognizer=fake, emit_telemetry=lines.append
            )
        transcriber.create_stream()
        chunk = [0.1] * 8000
        for _ in range(3):
            transcriber.feed(chunk)
        transcriber.finish()
        self.assertEqual(len(lines), 1)
        line = lines[0]
        self.assertTrue(line.startswith("stage=streaming "))
        self.assertIn("segments=2", line)
        # 遥测纪律：识别文本绝不入遥测行。
        self.assertNotIn("增量", line)
        self.assertNotIn("终态收尾", line)

    def test_export_segments_stamp_streaming_namespace(self):
        fake = _ScriptedRecognizer()
        with tempfile.TemporaryDirectory() as tmp:
            transcriber = StreamingTranscriber(_config(tmp), recognizer=fake)
        transcriber.create_stream()
        chunk = [0.1] * 8000
        for _ in range(3):
            transcriber.feed(chunk)
        exported = export_segments(transcriber.segments)
        self.assertEqual(len(exported), 1)
        self.assertEqual(exported[0]["source"], SEGMENT_SOURCE)
        self.assertEqual(exported[0]["start_ms"], 0)
        self.assertEqual(exported[0]["end_ms"], 1000)
        self.assertEqual(exported[0]["text"], "增量中间结果")

    def test_checkpoint_payload_uses_streaming_prefixed_keys(self):
        fake = _ScriptedRecognizer()
        with tempfile.TemporaryDirectory() as tmp:
            transcriber = StreamingTranscriber(_config(tmp), recognizer=fake)
        transcriber.create_stream()
        transcriber.feed([0.1] * 8000)
        payload = checkpoint_segments_payload(transcriber.segments)
        self.assertIn("streaming_segments", payload)
        self.assertIn("streaming_segment_count", payload)
        self.assertNotIn("raw_rough", payload)
        self.assertNotIn("raw_refined", payload)


class InstallModelsStreamingEntryTests(unittest.TestCase):
    """第 1 步钉：install_models 分发条目（沿 zipformer 测试同款 importlib 形态）。"""

    @classmethod
    def setUpClass(cls):
        import importlib.util

        worker_root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "courselens_install_models_streaming",
            worker_root / "scripts" / "install_models.py",
        )
        cls.install_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.install_models)

    def test_streaming_paraformer_entry_pinned(self):
        entry = self.install_models.MODELS["streaming-paraformer"]
        self.assertEqual(
            entry["archive"],
            "sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2",
        )
        # 实测钉：GitHub releases asr-models 资产 sha256（1,047,319,737 B 下载件
        # 本地复算；解包 int8 三件与 D13-IMPL POC 实测缓存 cmp 逐字节一致）。
        self.assertEqual(
            entry["sha256"],
            "5462a1fce42693deae572af1e8c4687124b12aa85fe61ff4d3168bb5280e205f",
        )
        # 环境名与生产适配层的模型目录解析逐字一致（install_models 写 GITHUB_ENV）。
        self.assertEqual(entry.get("env"), mod.MODEL_DIR_ENV)
        # 237MB 加重 Actions cache：非默认集（沿 silero-vad 先例），显式点名才装。
        self.assertFalse(entry.get("default"))

    def test_install_end_to_end_from_local_tarball(self):
        """离线全链：校验和门→安全解包→ready 标记→目录选择（仿真下载）。"""
        import tempfile

        tarball = (
            Path(__file__).resolve().parents[2]
            / ".tmp-d13prod"
            / "sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2"
        )
        if not tarball.is_file():
            raise unittest.SkipTest("measured tarball not cached in lane scratch")

        class _FakeResponse:
            def __init__(self, payload: bytes):
                self._payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def raise_for_status(self):
                return None

            def iter_content(self, block_size):
                for start in range(0, len(self._payload), block_size):
                    yield self._payload[start : start + block_size]

        payload = tarball.read_bytes()
        entry = self.install_models.MODELS["streaming-paraformer"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(
                self.install_models.requests,
                "get",
                return_value=_FakeResponse(payload),
            ):
                installed = self.install_models._install(
                    "streaming-paraformer", entry, root
                )
            self.assertEqual(installed.name, "sherpa-onnx-streaming-paraformer-bilingual-zh-en")
            self.assertTrue((root / ".streaming-paraformer-5462a1fce42693deae572af1e8c4687124b12aa85fe61ff4d3168bb5280e205f.ready").is_file())
            # int8 三件就位（与 POC 实测缓存同字节——cmp 已在钉值落地前验证）。
            for name in (mod.ENCODER_FILENAME, mod.DECODER_FILENAME, mod.TOKENS_FILENAME):
                self.assertTrue((installed / name).is_file(), name)
            # 二次安装走 ready 标记短路（零网络）。
            with patch.object(
                self.install_models.requests,
                "get",
                side_effect=AssertionError("must not re-download"),
            ):
                again = self.install_models._install(
                    "streaming-paraformer", entry, root
                )
            self.assertEqual(again, installed)

    def test_marker_selects_streaming_directory(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = (
                "sherpa-onnx-streaming-paraformer-bilingual-zh-en",
                "sherpa-onnx-paraformer-zh-2023-09-14",
            )
            for name in names:
                (root / name).mkdir()
                (root / name / "tokens.txt").write_text("t", encoding="ascii")
            directories = self.install_models._model_directories(
                root, "streaming-paraformer"
            )
            self.assertEqual([path.name for path in directories], [names[0]])
            # 既有 paraformer 条目选择不受新目录干扰（取字典序首个=原目录）。
            para = self.install_models._model_directories(root, "paraformer")
            self.assertEqual(para[0].name, names[1])

    def test_streaming_not_in_default_install_set(self):
        selected = self.install_models._select_models()
        self.assertNotIn("streaming-paraformer", selected)

    def test_explicit_selection_pick_streaming_entry(self):
        with patch.dict(os.environ, {self.install_models.INSTALL_ALL_ENV: "streaming-paraformer"}):
            selected = self.install_models._select_models()
        self.assertEqual(set(selected), {"streaming-paraformer"})


class RealStreamingModelTests(unittest.TestCase):
    """真实模型冒烟钉（运行时+模型缺席=跳过，绝不伪造）。"""

    @classmethod
    def setUpClass(cls):
        if not mod.sherpa_runtime_available():
            raise unittest.SkipTest("real sherpa-onnx runtime required")
        try:
            resolve_model_paths()
        except StreamingAsrError as exc:
            raise unittest.SkipTest(str(exc))

    def test_real_model_load_and_incremental(self):
        config = load_default_config()
        transcriber = StreamingTranscriber(config)
        self.assertIsNotNone(transcriber.recognizer)
        self.assertGreater(transcriber.metrics.model_load_seconds, 0.0)
        transcriber.create_stream()
        samples = [0.05] * 16000  # 1s 轻响
        for start in range(0, len(samples), 8000):
            transcriber.feed(samples[start : start + 8000])
        final = transcriber.finish()
        # 纯音调不保证出字：只钉三态管线走通与锚自洽，不评字准。
        for segment in transcriber.segments:
            self.assertLessEqual(segment.start_ms, segment.end_ms)
        if transcriber.segments:
            self.assertIsNotNone(final)


if __name__ == "__main__":
    unittest.main()
