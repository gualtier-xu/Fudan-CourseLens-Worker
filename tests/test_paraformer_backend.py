"""M4 Paraformer backend pins (ASRBENCH-1 A5): install entry, pool branch,
SUBTITLE_BACKENDS sequence, env override, and a PCM decode smoke.

Fake recognizers and mocked sherpa_onnx only; no model download, no media.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from courselens_worker import asr

_WORKER_ROOT = Path(__file__).resolve().parents[1]
_INSTALL_MODELS_SPEC = importlib.util.spec_from_file_location(
    "courselens_install_models_para",
    _WORKER_ROOT / "scripts" / "install_models.py",
)
install_models = importlib.util.module_from_spec(_INSTALL_MODELS_SPEC)
_INSTALL_MODELS_SPEC.loader.exec_module(install_models)


class InstallModelsParaformerEntryTests(unittest.TestCase):
    def test_paraformer_entry_is_pinned_to_measured_sha256(self):
        entry = install_models.MODELS["paraformer"]
        self.assertEqual(
            entry["archive"], "sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2"
        )
        # 实测钉（ASRBENCH 下载件 sha256，M4-ENABLE-1 U1）：条目进入安装面。
        self.assertEqual(entry["sha256"], "9c49fd9c6fb63de8e18c1054cf3d100f804741b7e608e187923cd8ff09fa9f03")

    def test_unpinned_entry_skips_download_and_env_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            env_file = Path(temporary) / "models.env"
            with (
                patch.dict(os.environ, {
                    "COURSELENS_MODEL_ROOT": str(Path(temporary) / "models"),
                    "GITHUB_ENV": str(env_file),
                }),
                patch.object(
                    install_models,
                    "MODELS",
                    {"ghost": {"archive": "ghost.tar.bz2", "sha256": ""}},
                ),
                patch.object(
                    install_models.requests,
                    "get",
                    side_effect=AssertionError("unpinned entry must not download"),
                ),
            ):
                install_models.main()
            self.assertEqual(env_file.read_text(encoding="utf-8"), "")

    def test_marker_branch_selects_only_paraformer_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = (
                "sherpa-onnx-paraformer-zh-2023-09-14",
                "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
            )
            for name in names:
                tokens = root / name / "tokens.txt"
                tokens.parent.mkdir(parents=True)
                tokens.write_text("a 0\n", encoding="ascii")
            self.assertEqual(
                [path.name for path in install_models._model_directories(root, "paraformer")],
                ["sherpa-onnx-paraformer-zh-2023-09-14"],
            )
            self.assertEqual(len(install_models._model_directories(root, "sensevoice")), 1)


class RecognizerPoolParaformerTests(unittest.TestCase):
    def _paraformer_directory(self, root: Path) -> Path:
        directory = root / "sherpa-onnx-paraformer-zh-2023-09-14"
        directory.mkdir(parents=True)
        (directory / "model.int8.onnx").write_bytes(b"model")
        (directory / "tokens.txt").write_text("a 0\n", encoding="ascii")
        return directory

    def test_paraformer_branch_builds_via_from_paraformer(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._paraformer_directory(Path(temporary))
            pool = asr.RecognizerPool(Path("sense"), directory, threads=2)
            with patch.object(asr, "sherpa_onnx") as sherpa:
                recognizer = pool.get("paraformer")
                # 构造缓存：同 backend 第二次 get 不再触达构造器
                pool.get("paraformer")
            sherpa.OfflineRecognizer.from_paraformer.assert_called_once_with(
                paraformer=str(directory / "model.int8.onnx"),
                tokens=str(directory / "tokens.txt"),
                num_threads=2,
                debug=False,
                provider="cpu",
            )
            self.assertIs(
                recognizer, sherpa.OfflineRecognizer.from_paraformer.return_value
            )

    def test_paraformer_requires_configured_directory(self):
        pool = asr.RecognizerPool(Path("sense"), None, threads=1)
        with self.assertRaises(asr.ASRError):
            pool.get("paraformer")

    def test_unknown_backend_still_fails_closed(self):
        pool = asr.RecognizerPool(Path("sense"), None, threads=1)
        with self.assertRaises(asr.ASRError):
            pool.get("whisper")


class SubtitleBackendSequenceTests(unittest.TestCase):
    def test_default_sequence_is_m4_paraformer_refinement(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                asr.subtitle_backend_sequence(), ["sensevoice", "paraformer"]
            )

    def test_env_override_selects_paraformer_refinement(self):
        for raw in ("sensevoice,paraformer", " sensevoice , paraformer "):
            with patch.dict(os.environ, {"SUBTITLE_BACKENDS": raw}):
                self.assertEqual(
                    asr.subtitle_backend_sequence(), ["sensevoice", "paraformer"], raw
                )
        with patch.dict(os.environ, {"SUBTITLE_BACKENDS": ""}):
            self.assertEqual(
                asr.subtitle_backend_sequence(), ["sensevoice", "paraformer"]
            )

    def test_invalid_sequences_fail_closed(self):
        for raw in (
            "sensevoice",
            "sensevoice,sensevoice",
            "sensevoice,gpt",
            "sensevoice,paraformer,gpt",
            "   ",
        ):
            with patch.dict(os.environ, {"SUBTITLE_BACKENDS": raw}):
                with self.assertRaises(asr.ASRError):
                    asr.subtitle_backend_sequence()


class _FakeStream:
    def __init__(self) -> None:
        self.waveform = None
        self.result = None

    def accept_waveform(self, sample_rate, samples) -> None:
        # 与真实 sherpa 绑定一致：拷贝波形，避免保留 memmap 视图锁住文件
        self.waveform = np.array(samples, dtype=np.float32)


class _FakeRecognizer:
    def __init__(self) -> None:
        self.streams: list[_FakeStream] = []
        self.batches: list[list[_FakeStream]] = []

    def create_stream(self) -> _FakeStream:
        stream = _FakeStream()
        self.streams.append(stream)
        return stream

    def decode_streams(self, streams) -> None:
        self.batches.append(list(streams))
        for stream in streams:
            peak = float(np.max(np.abs(stream.waveform))) if stream.waveform.size else 0.0
            stream.result = SimpleNamespace(text="合成语音" if peak > 0.05 else "")


class ParaformerDecodeSmokeTests(unittest.TestCase):
    """transcribe_pcm 走真 VAD/分窗/批量解码路径（真 numpy、伪 recognizer）。"""

    def test_paraformer_pcm_decode_produces_anchored_segments(self):
        samples = np.zeros(int(30 * asr.SAMPLE_RATE), dtype=np.float32)
        samples[int(5 * asr.SAMPLE_RATE):int(8 * asr.SAMPLE_RATE)] = 0.4
        recognizer = _FakeRecognizer()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "chunk.f32le"
            np.asarray(samples, dtype=np.float32).tofile(path)
            pool = asr.RecognizerPool(Path("sense"), Path("para"))
            with patch.object(asr.RecognizerPool, "get", return_value=recognizer):
                segments = pool.transcribe_pcm(path, "paraformer", offset_seconds=0.0)
        self.assertEqual(len(segments), 1)
        self.assertGreaterEqual(segments[0]["start_ms"], 4700)
        self.assertLessEqual(segments[0]["end_ms"], 8300)
        self.assertEqual(len(recognizer.batches), 1)
        self.assertEqual(len(recognizer.batches[0]), 1)
        self.assertEqual(len(recognizer.streams), 1)


class ParaformerChainTranscribeTests(unittest.TestCase):
    """SUBTITLE_BACKENDS=sensevoice,paraformer 整链冒烟（伪 pool，零媒体）。"""

    def _pool(self):
        pool = Mock()
        pool.transcribe_pcm.side_effect = lambda _path, backend, *, offset_seconds: [{
            "start_ms": int(offset_seconds * 1000),
            "end_ms": int(offset_seconds * 1000) + 1000,
            "text": f"{backend}@{int(offset_seconds)}",
        }]
        return pool

    def _run(self, pool, *, prior=None, checkpoints=None, proofread="mock"):
        def create_pcm(_url, target, *, offset, duration):
            target.write_bytes(b"pcm-bytes")

        payload = {
            "mode": "automatic",
            "media": {
                "url": "https://media.example.com/lecture.mp4",
                "duration_seconds": 1250,
            },
        }
        if prior is not None:
            payload["checkpoint"] = prior
        proofread_fn = (
            Mock(return_value=[{"start_ms": 0, "end_ms": 1000, "text": "校对后"}])
            if proofread == "mock" else proofread
        )
        with (
            patch.object(asr, "RecognizerPool", return_value=pool),
            patch.object(asr, "pinned_media_proxy"),
            patch.object(asr, "_prefetch_media_pcm",
                         side_effect=lambda _u, t, *, duration: t.write_bytes(b"")),
            patch.object(asr, "_slice_pcm_chunk", side_effect=create_pcm),
            patch.dict(os.environ, {"SUBTITLE_BACKENDS": "sensevoice,paraformer"}),
        ):
            return asr.transcribe(
                {"payload": payload},
                sensevoice_dir=Mock(),
                paraformer_dir=Mock(),
                proofread=proofread_fn,
                progress=Mock(),
                checkpoint=checkpoints,
            )

    def test_paraformer_refinement_outputs_and_provenance(self):
        checkpoints: list[dict] = []
        result = self._run(self._pool(), checkpoints=checkpoints.append)
        self.assertIn("raw_sensevoice", result)
        self.assertIn("raw_paraformer", result)
        self.assertEqual(
            {key for key in result if key.startswith("raw_")},{"raw_sensevoice", "raw_paraformer"},
        )
        self.assertEqual(result["segments"][0]["text"], "校对后")
        self.assertEqual(
            result["segments"][0]["provenance"]["model"],
            "sensevoice+paraformer:proofread",
        )
        self.assertEqual(
            result["raw_sensevoice"][0]["provenance"]["model"], "sensevoice"
        )
        self.assertEqual(
            result["raw_paraformer"][0]["provenance"]["model"], "paraformer"
        )
        for checkpoint in checkpoints:
            self.assertIn("raw_paraformer", checkpoint)

    def test_no_proofread_final_model_is_the_refined_backend(self):
        result = self._run(self._pool(), proofread=None)
        self.assertEqual(result["segments"][0]["provenance"]["model"], "paraformer")

    def test_override_resume_rejects_checkpoint_without_refined_raw(self):
        prior = {
            "completed_chunks": 1,
            "total_chunks": 3,
            "mode": "automatic",
            "raw_sensevoice": [{"start_ms": 0, "end_ms": 1000, "text": "x"}],
        }
        with self.assertRaises(asr.ASRError):
            self._run(self._pool(), prior=prior)


if __name__ == "__main__":
    unittest.main()
