"""SUBTITLE-DEEP-1 Phase B: zipformer hotword backend pins.

Install entry, backend sequence, pool branch (from_transducer with hotwords),
model-missing fallback to the legacy chain, hotword file materialization, and
pre-warm failure semantics. Mocked sherpa_onnx only; no model download.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from courselens_worker import asr

_WORKER_ROOT = Path(__file__).resolve().parents[1]
_INSTALL_MODELS_SPEC = importlib.util.spec_from_file_location(
    "courselens_install_models_zip",
    _WORKER_ROOT / "scripts" / "install_models.py",
)
install_models = importlib.util.module_from_spec(_INSTALL_MODELS_SPEC)
_INSTALL_MODELS_SPEC.loader.exec_module(install_models)


class InstallModelsZipformerEntryTests(unittest.TestCase):
    def test_zipformer_entry_is_pinned_to_measured_sha256(self):
        entry = install_models.MODELS["zipformer"]
        self.assertEqual(
            entry["archive"], "sherpa-onnx-zipformer-multi-zh-hans-2023-9-2.tar.bz2"
        )
        # 实测钉（BENCH-ASR-1 下载件 sha256，SUBTITLE-DEEP-1 本地复算）。
        self.assertEqual(
            entry["sha256"],
            "c4925a6b0f998800d16f80caf90d2decff7b7a8c156d044c6cffdf141c847d94",
        )

    def test_marker_selects_zipformer_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = (
                "sherpa-onnx-zipformer-multi-zh-hans-2023-9-2",
                "sherpa-onnx-paraformer-zh-2023-09-14",
            )
            for name in names:
                tokens = root / name / "tokens.txt"
                tokens.parent.mkdir(parents=True)
                tokens.write_text("a 0\n", encoding="ascii")
            self.assertEqual(
                [path.name for path in install_models._model_directories(root, "zipformer")],
                ["sherpa-onnx-zipformer-multi-zh-hans-2023-9-2"],
            )


class ZipformerBackendSequenceTests(unittest.TestCase):
    def test_default_sequence_unchanged(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(asr.subtitle_backend_sequence(), ["sensevoice", "paraformer"])

    def test_zipformer_sequence_is_accepted(self):
        for raw in ("sensevoice,zipformer", " sensevoice , zipformer "):
            with patch.dict(os.environ, {"SUBTITLE_BACKENDS": raw}):
                self.assertEqual(
                    asr.subtitle_backend_sequence(), ["sensevoice", "zipformer"], raw
                )

    def test_invalid_zipformer_sequences_fail_closed(self):
        for raw in ("zipformer", "zipformer,zipformer", "sensevoice,zipformer,gpt"):
            with patch.dict(os.environ, {"SUBTITLE_BACKENDS": raw}):
                with self.assertRaises(asr.ASRError):
                    asr.subtitle_backend_sequence()


class ZipformerPoolTests(unittest.TestCase):
    def _zipformer_directory(self, root: Path) -> Path:
        directory = root / "sherpa-onnx-zipformer-multi-zh-hans-2023-9-2"
        directory.mkdir(parents=True)
        for role in ("encoder", "decoder", "joiner"):
            (directory / f"{role}-epoch-20-avg-1.int8.onnx").write_bytes(b"m")
            (directory / f"{role}-epoch-20-avg-1.onnx").write_bytes(b"m")
        (directory / "tokens.txt").write_text("a 0\n", encoding="ascii")
        return directory

    def test_transducer_files_prefer_int8(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._zipformer_directory(Path(temporary))
            encoder, decoder, joiner = asr.RecognizerPool._transducer_files(directory)
            self.assertTrue(encoder.name.startswith("encoder-"))
            self.assertIn(".int8.", encoder.name)
            self.assertIn(".int8.", decoder.name)
            self.assertIn(".int8.", joiner.name)

    def test_transducer_files_missing_role_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "encoder-a.int8.onnx").write_bytes(b"m")
            with self.assertRaises(asr.ASRError):
                asr.RecognizerPool._transducer_files(directory)

    def test_zipformer_ready_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._zipformer_directory(Path(temporary))
            pool = asr.RecognizerPool(Path("s"), Path("p"), zipformer_dir=directory)
            self.assertTrue(pool.zipformer_ready())
            (directory / "joiner-epoch-20-avg-1.int8.onnx").unlink()
            (directory / "joiner-epoch-20-avg-1.onnx").unlink()
            self.assertFalse(pool.zipformer_ready())
        missing = asr.RecognizerPool(Path("s"), Path("p"), zipformer_dir=Path("nope"))
        self.assertFalse(missing.zipformer_ready())
        none_pool = asr.RecognizerPool(Path("s"), Path("p"))
        self.assertFalse(none_pool.zipformer_ready())

    def test_zipformer_branch_builds_via_from_transducer_with_hotwords(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._zipformer_directory(Path(temporary))
            pool = asr.RecognizerPool(Path("s"), None, threads=2, zipformer_dir=directory)
            hotwords = Path(temporary) / "hotwords.txt"
            pool.hotwords_file = hotwords
            with patch.object(asr, "sherpa_onnx") as sherpa:
                recognizer = pool.get("zipformer")
                pool.get("zipformer")
            kwargs = sherpa.OfflineRecognizer.from_transducer.call_args.kwargs
            self.assertEqual(kwargs["encoder"], str(directory / "encoder-epoch-20-avg-1.int8.onnx"))
            self.assertEqual(kwargs["decoder"], str(directory / "decoder-epoch-20-avg-1.int8.onnx"))
            self.assertEqual(kwargs["joiner"], str(directory / "joiner-epoch-20-avg-1.int8.onnx"))
            self.assertEqual(kwargs["hotwords_file"], str(hotwords))
            self.assertEqual(kwargs["hotwords_score"], asr.ZIPFORMER_HOTWORDS_SCORE)
            self.assertEqual(kwargs["decoding_method"], asr.ZIPFORMER_HOTWORD_DECODING)
            self.assertIs(recognizer, sherpa.OfflineRecognizer.from_transducer.return_value)

    def test_zipformer_cold_build_uses_default_decoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self._zipformer_directory(Path(temporary))
            pool = asr.RecognizerPool(Path("s"), None, threads=2, zipformer_dir=directory)
            with patch.object(asr, "sherpa_onnx") as sherpa:
                pool.get("zipformer")
            kwargs = sherpa.OfflineRecognizer.from_transducer.call_args.kwargs
            self.assertEqual(kwargs["hotwords_file"], "")
            self.assertEqual(kwargs["decoding_method"], "default")


class ZipformerChainTranscribeTests(unittest.TestCase):
    """SUBTITLE_BACKENDS=sensevoice,zipformer 整链冒烟（伪 pool，零媒体）。"""

    def _pool(self):
        pool = Mock()
        pool.zipformer_ready.return_value = True
        pool.transcribe_pcm.side_effect = lambda _path, backend, *, offset_seconds: [{
            "start_ms": int(offset_seconds * 1000),
            "end_ms": int(offset_seconds * 1000) + 1000,
            "text": f"{backend}@{int(offset_seconds)}",
        }]
        return pool

    def _run(self, pool, *, zipformer_dir="set", hotwords=(), prior=None):
        def create_pcm(_url, target, *, offset, duration):
            target.write_bytes(b"pcm-bytes")

        payload = {
            "mode": "automatic",
            "media": {"url": "https://media.example.com/lecture.mp4", "duration_seconds": 1250},
        }
        if prior is not None:
            payload["checkpoint"] = prior
        captured = {}

        def pool_factory(_sense, _para, *, threads, zipformer_dir=None):
            captured["zipformer_dir"] = zipformer_dir
            return pool

        with (
            patch.object(asr, "RecognizerPool", side_effect=pool_factory),
            patch.object(asr, "pinned_media_proxy"),
            patch.object(asr, "_prefetch_media_pcm",
                         side_effect=lambda _u, t, *, duration: t.write_bytes(b"")),
            patch.object(asr, "_slice_pcm_chunk", side_effect=create_pcm),
            patch.dict(os.environ, {"SUBTITLE_BACKENDS": "sensevoice,zipformer"}),
        ):
            result = asr.transcribe(
                {"payload": payload},
                sensevoice_dir=Mock(),
                paraformer_dir=Mock(),
                zipformer_dir=Mock() if zipformer_dir == "set" else None,
                hotwords=hotwords,
                proofread=Mock(return_value=[{"start_ms": 0, "end_ms": 1000, "text": "校对后"}]),
                progress=Mock(),
            )
        result["_captured"] = captured
        return result

    def test_zipformer_chain_outputs_and_hotword_file(self):
        pool = self._pool()
        # 热词文件断言：预暖时挂到 pool.hotwords_file，内容在临时目录存活期内读出。
        hotword_snapshots = []

        def get_side_effect(backend):
            if pool.hotwords_file is not None:
                hotword_snapshots.append(Path(pool.hotwords_file).read_text(encoding="utf-8"))
            return Mock()

        pool.get.side_effect = get_side_effect
        result = self._run(pool, hotwords=("能带图", "电势"))
        self.assertEqual(result["segments"][0]["text"], "校对后")
        self.assertEqual(
            result["segments"][0]["provenance"]["model"], "sensevoice+zipformer:proofread"
        )
        self.assertIn("raw_zipformer", result)
        self.assertEqual(hotword_snapshots, ["能带图\n电势\n"], "预暖恰一次且内容为术语行")
        self.assertIsNotNone(result["_captured"]["zipformer_dir"])

    def test_missing_model_falls_back_to_paraformer(self):
        pool = self._pool()
        pool.zipformer_ready.return_value = False
        result = self._run(pool, zipformer_dir="set", hotwords=("能带图",))
        self.assertIn("raw_paraformer", result)
        self.assertNotIn("raw_zipformer", result)
        self.assertIsNone(result["_captured"]["zipformer_dir"], "回退后不再持 zipformer 目录")
        self.assertEqual(
            result["segments"][0]["provenance"]["model"], "sensevoice+paraformer:proofread"
        )


if __name__ == "__main__":
    unittest.main()
