"""V4NONTHINK-1 pins: local quality network + wiring.

Covers ct-punc adapter (env modes, content-identity fail-closed gate, model
missing fallback, CTPUNC-DEF-1 fill default + one-probe load-failure cache),
silero-vad dispatch (env default off, fail-closed fallback,
region discipline), install_models multi-file pin math + default install set,
and the runner ct-punc stage wiring.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def _load_install_models():
    source = Path(__file__).resolve().parents[1] / "scripts" / "install_models.py"
    spec = importlib.util.spec_from_file_location("v4nt_install_models", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seg(start_ms, end_ms, text, **extra):
    value = {"start_ms": start_ms, "end_ms": end_ms, "text": text}
    value.update(extra)
    return value


class CtPuncModeTests(unittest.TestCase):
    def test_default_mode_is_fill(self):
        # CTPUNC-DEF-1：缺省 fill（A 序缺口填充），显式 off 才退回
        from courselens_worker.punct import ct_punc_mode

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ct_punc_mode(), "fill")
        with patch.dict(os.environ, {"COURSELENS_SUBTITLE_CTPUNC": "off"}):
            self.assertEqual(ct_punc_mode(), "off")

    def test_mode_closed_set(self):
        from courselens_worker.punct import ct_punc_mode

        with patch.dict(os.environ, {"COURSELENS_SUBTITLE_CTPUNC": "FILL"}):
            self.assertEqual(ct_punc_mode(), "fill")
        with patch.dict(os.environ, {"COURSELENS_SUBTITLE_CTPUNC": "full"}):
            self.assertEqual(ct_punc_mode(), "full")
        with patch.dict(os.environ, {"COURSELENS_SUBTITLE_CTPUNC": "bogus"}):
            # 闭集外随缺省 fill（与 vad_engine「未知回落缺省」同构；
            # fill 有内容等值门 fail-closed 兜底，显式 off 不受影响）
            self.assertEqual(ct_punc_mode(), "fill")

    def test_model_dir_resolves_missing_as_none(self):
        from courselens_worker.punct import ct_punc_model_dir

        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(ct_punc_model_dir())
        with patch.dict(os.environ, {"CTPUNC_MODEL_DIR": "Z:/no/such/dir"}):
            self.assertIsNone(ct_punc_model_dir())


class CtPuncGateTests(unittest.TestCase):
    def _engine(self, output):
        engine = Mock()
        engine.side_effect = lambda text, **kwargs: [(text, output)]
        return engine

    def _patched_env(self):
        return (
            patch.dict(os.environ, {
                "COURSELENS_SUBTITLE_CTPUNC": "fill",
                "CTPUNC_MODEL_DIR": "Z:/models/ct-punc",
            }),
        )

    def test_fill_mode_touches_only_unpunctuated_segments(self):
        from courselens_worker import punct

        segments = [
            seg(0, 1000, "这是没有标点的句子"),
            seg(2000, 3000, "这已经有标点了。"),
        ]
        with (
            patch.dict(os.environ, {
                "COURSELENS_SUBTITLE_CTPUNC": "fill",
                "CTPUNC_MODEL_DIR": "Z:/models/ct-punc",
            }),
            patch.object(punct, "ct_punc_model_dir", return_value=Path("Z:/models/ct-punc/model.onnx")),
            patch.object(punct, "_load_engine", return_value=self._engine("这是，没有标点的句子。")),
        ):
            stats = punct.apply_ct_punc(segments)
        self.assertEqual(stats["applied"], 1)
        self.assertEqual(segments[0]["text"], "这是，没有标点的句子。")
        self.assertEqual(segments[1]["text"], "这已经有标点了。")

    def test_content_identity_gate_keeps_original_on_drift(self):
        # 幻觉/换字：去标点内容不等值 → 保留原文（fail-closed）
        from courselens_worker import punct

        segments = [seg(0, 1000, "完全没有标点的一句话")]
        with (
            patch.dict(os.environ, {
                "COURSELENS_SUBTITLE_CTPUNC": "full",
                "CTPUNC_MODEL_DIR": "Z:/models/ct-punc",
            }),
            patch.object(punct, "ct_punc_model_dir", return_value=Path("Z:/models/ct-punc/model.onnx")),
            patch.object(punct, "_load_engine", return_value=self._engine("完全换掉了内容的话。")),
        ):
            stats = punct.apply_ct_punc(segments)
        self.assertEqual(stats["applied"], 0)
        self.assertEqual(stats["kept"], 1)
        self.assertEqual(segments[0]["text"], "完全没有标点的一句话")

    def test_engine_load_failure_one_probe_quiet_skip(self):
        # CTPUNC-DEF-1 件3：默认开+引擎不可用（缺依赖/坏模型）→ 每讲一次探测
        # → 安静跳过整讲 + 闭集遥测记账，禁止逐段报错/逐段重试。
        from courselens_worker import punct

        segments = [seg(0, 1000, "没有标点"), seg(1000, 2000, "第二句也没标点")]
        telemetry: list[str] = []
        with (
            patch.dict(os.environ, {
                "COURSELENS_SUBTITLE_CTPUNC": "full",
                "CTPUNC_MODEL_DIR": "Z:/models/ct-punc",
            }),
            patch.object(punct, "ct_punc_model_dir", return_value=Path("Z:/models/ct-punc")),
            patch.object(punct, "_load_engine", side_effect=RuntimeError("onnx boom")) as load_mock,
            patch.object(punct, "_failed_dir", ""),
        ):
            stats = punct.apply_ct_punc(segments, telemetry=telemetry)
        self.assertEqual(stats["model_load_failed"], 1)
        self.assertEqual(load_mock.call_count, 1, "两段只允许一次加载探测")
        self.assertEqual(telemetry, ["stage=ct-punc-fallback reason=model_load_failed"])
        self.assertEqual([s["text"] for s in segments], ["没有标点", "第二句也没标点"])

    def test_engine_load_failure_cached_across_lectures(self):
        # CTPUNC-DEF-1 件3：进程内一次探测缓存——第二讲零 import/零构造重试。
        from courselens_worker import punct

        fake_module = Mock()
        fake_module.CT_Transformer.side_effect = RuntimeError("no real onnx runtime")
        segments = [seg(0, 1000, "没有标点"), seg(1000, 2000, "第二句")] * 2
        with (
            patch.dict(os.environ, {
                "COURSELENS_SUBTITLE_CTPUNC": "full",
                "CTPUNC_MODEL_DIR": "Z:/models/ct-punc",
            }),
            patch.object(punct, "ct_punc_model_dir", return_value=Path("Z:/models/ct-punc")),
            patch.dict(sys.modules, {"funasr_onnx": fake_module}),
            patch.object(punct, "_engine", None),
            patch.object(punct, "_engine_dir", ""),
            patch.object(punct, "_failed_dir", ""),
        ):
            first = punct.apply_ct_punc(list(segments))
            second = punct.apply_ct_punc(list(segments))
        self.assertEqual(fake_module.CT_Transformer.call_count, 1, "进程内只探测一次")
        self.assertEqual(first["model_load_failed"], 1)
        self.assertEqual(second["model_load_failed"], 1)

    def test_inference_exception_keeps_original_fail_closed(self):
        # 推理面（引擎已加载后调用抛错）保持逐段 fail-closed 保留原文。
        from courselens_worker import punct

        segments = [seg(0, 1000, "没有标点")]
        engine = Mock(side_effect=RuntimeError("inference boom"))
        with (
            patch.dict(os.environ, {
                "COURSELENS_SUBTITLE_CTPUNC": "full",
                "CTPUNC_MODEL_DIR": "Z:/models/ct-punc",
            }),
            patch.object(punct, "ct_punc_model_dir", return_value=Path("Z:/models/ct-punc")),
            patch.object(punct, "_load_engine", return_value=engine),
        ):
            stats = punct.apply_ct_punc(segments)
        self.assertEqual(stats["kept"], 1)
        self.assertEqual(segments[0]["text"], "没有标点")

    def test_model_missing_falls_back_with_telemetry(self):
        from courselens_worker import punct

        segments = [seg(0, 1000, "没有标点")]
        telemetry: list[str] = []
        with (
            patch.dict(os.environ, {"COURSELENS_SUBTITLE_CTPUNC": "full"}, clear=True),
            patch.object(punct, "ct_punc_model_dir", return_value=None),
        ):
            stats = punct.apply_ct_punc(segments, telemetry=telemetry)
        self.assertEqual(stats["model_missing"], 1)
        self.assertEqual(telemetry, ["stage=ct-punc-fallback reason=model_missing"])
        self.assertEqual(segments[0]["text"], "没有标点")

    def test_off_mode_is_zero_touch(self):
        from courselens_worker import punct

        segments = [seg(0, 1000, "没有标点")]
        with patch.dict(os.environ, {"COURSELENS_SUBTITLE_CTPUNC": "off"}, clear=True):
            stats = punct.apply_ct_punc(segments)
        self.assertEqual(stats["mode_off"], 1)
        self.assertEqual(segments[0]["text"], "没有标点")

    def test_runner_stage_wires_apply_and_off_is_silent(self):
        from courselens_worker.punct import ct_punc_mode
        from courselens_worker.runner import _apply_ct_punc_stage

        value = {"segments": [seg(0, 1000, "没有标点")]}
        with patch.dict(os.environ, {"COURSELENS_SUBTITLE_CTPUNC": "off"}, clear=True):
            self.assertEqual(ct_punc_mode(), "off")
            _apply_ct_punc_stage(value, warnings=[])  # off → 零调用零输出
        with (
            patch.dict(os.environ, {"COURSELENS_SUBTITLE_CTPUNC": "full"}, clear=True),
            patch("courselens_worker.runner.apply_ct_punc") as apply_mock,
        ):
            apply_mock.return_value = {"applied": 2, "kept": 1}
            _apply_ct_punc_stage(value, warnings=[])
        apply_mock.assert_called_once()


class SileroVadTests(unittest.TestCase):
    def test_default_engine_is_energy(self):
        from courselens_worker.asr import VAD_ENGINE_ENERGY, vad_engine

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(vad_engine(), VAD_ENGINE_ENERGY)

    def test_engine_closed_set(self):
        from courselens_worker.asr import VAD_ENGINE_ENERGY, VAD_ENGINE_SILERO, vad_engine

        with patch.dict(os.environ, {"COURSELENS_ASR_VAD_ENGINE": "silero"}):
            self.assertEqual(vad_engine(), VAD_ENGINE_SILERO)
        with patch.dict(os.environ, {"COURSELENS_ASR_VAD_ENGINE": "weird"}):
            self.assertEqual(vad_engine(), VAD_ENGINE_ENERGY)

    def test_silero_model_path_resolves_dir_or_missing(self):
        from courselens_worker.asr import silero_model_path

        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(silero_model_path())
        with patch.dict(os.environ, {"SILERO_MODEL_DIR": "Z:/no/such"}):
            self.assertIsNone(silero_model_path())

    def test_pool_dispatch_energy_when_not_ready(self):
        from courselens_worker.asr import RecognizerPool

        pool = RecognizerPool(Path("Z:/models/sensevoice"))
        with patch(
            "courselens_worker.asr.detect_voiced_regions",
            side_effect=lambda w, *, energy_ratio: [("hit",)],
        ) as module_detect:
            out = pool._voiced_regions(Mock(), energy_ratio=3.0)
        self.assertEqual(out, [("hit",)])
        module_detect.assert_called_once()

    def test_silero_exception_falls_back_fail_closed(self):
        from courselens_worker.asr import RecognizerPool

        pool = RecognizerPool(Path("Z:/models/sensevoice"))
        pool.silero_model_path = Path("Z:/models/silero/silero_vad.onnx")
        lines: list[str] = []

        def boom(_window):
            raise RuntimeError("model exploded")

        with (
            patch("courselens_worker.asr.vad_engine", return_value="silero"),
            patch.object(pool, "_silero_regions", side_effect=boom),
            patch(
                "courselens_worker.asr.detect_voiced_regions",
                side_effect=lambda w, *, energy_ratio: [("energy",)],
            ),
            patch("courselens_worker.asr._emit_telemetry", side_effect=lines.append),
        ):
            out = pool._voiced_regions(Mock(), energy_ratio=3.0)
        self.assertEqual(out, [("energy",)])
        self.assertTrue(any("silero-vad-fallback" in line for line in lines))
        self.assertIsNone(pool._silero_vad, "失败后清缓存，后续窗重新构建")

    def test_silero_regions_collect_and_finalize(self):
        from courselens_worker.asr import RecognizerPool

        class _Segment:
            def __init__(self, start, count):
                self.start = start
                self.samples = [0.0] * count

        class FakeVad:
            def __init__(self, items):
                self._items = list(items)
                self.reset_called = False

            def empty(self):
                return not self._items

            @property
            def front(self):
                return self._items[0]

            def pop(self):
                self._items.pop(0)

            def reset(self):
                self.reset_called = True

            def accept_waveform(self, samples):
                return None

            def flush(self):
                return None

        pool = RecognizerPool(Path("Z:/models/sensevoice"))
        pool.silero_model_path = Path("Z:/models/silero/silero_vad.onnx")
        # 两段须过 min（1600 样本）且间隔 ≥ merge gap（6400 样本）才保持分离
        vad = FakeVad([_Segment(0, 15000), _Segment(21600, 2400)])
        pool._silero_vad = vad
        out = pool._silero_regions([0.0] * 24000)
        self.assertEqual(out[0][0], 0)
        self.assertEqual(len(out), 2)
        self.assertLessEqual(out[-1][1], 24000)
        for (_, end_a), (start_b, _) in zip(out, out[1:]):
            self.assertGreaterEqual(start_b, end_a)
        self.assertTrue(vad.reset_called)

    def test_finalize_regions_matches_energy_discipline(self):
        from courselens_worker.asr import _finalize_regions

        regions = _finalize_regions(
            [[0, 100], [5000, 5200], [5250, 300000]],
            total=320000,
            min_region_samples=160,
            merge_gap_samples=6400,
            max_region_samples=480000,
            pad_samples=2400,
            sample_rate=16000,
        )
        # [0,100] 被 min 过滤；后两段 gap<merge → 合并；pad 受半隙收缩
        self.assertEqual(regions, [(2600, 302400)])


class InstallModelsPinTests(unittest.TestCase):
    def test_pins_and_env_names(self):
        module = _load_install_models()
        self.assertEqual(
            module._files_marker_digest(module.MODELS["ct-punc"]),
            "5dfb9eddce4b90be07ad2442ffd3454b993eb0b703866eb5f7f1e86ad78b17d7",
        )
        self.assertEqual(len(module.MODELS["ct-punc"]["files"]), 4)
        self.assertEqual(
            module.MODELS["silero-vad"]["sha256"],
            "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
        )
        self.assertEqual(module.MODELS["silero-vad"]["env"], "SILERO_MODEL_DIR")
        self.assertEqual(module.MODELS["ct-punc"]["env"], "CTPUNC_MODEL_DIR")

    def test_default_set_membership(self):
        # CTPUNC-DEF-1 件2：ct-punc 进默认安装集；silero-vad 钉在册但不进
        # 默认集（翻默认前置=全讲级验证，仍 COURSELENS_INSTALL_MODELS 可选）。
        module = _load_install_models()
        default_names = {name for name, spec in module.MODELS.items() if spec.get("default")}
        self.assertIn("ct-punc", default_names)
        self.assertNotIn("silero-vad", default_names)
        self.assertEqual(default_names, {"sensevoice", "paraformer", "zipformer", "ct-punc"})
        # silero 钉本体零改动：raw_file/钉哈希/env 完整保留
        self.assertEqual(module.MODELS["silero-vad"]["raw_file"], "silero_vad.onnx")
        self.assertEqual(module.MODELS["silero-vad"]["default"], False)

    def test_select_models_closed_set(self):
        module = _load_install_models()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                set(module._select_models()),
                {"sensevoice", "paraformer", "zipformer", "ct-punc"},
            )
        with patch.dict(os.environ, {module.INSTALL_ALL_ENV: "all"}):
            self.assertEqual(set(module._select_models()), set(module.MODELS))
        with patch.dict(os.environ, {module.INSTALL_ALL_ENV: "silero-vad, ct-punc"}):
            self.assertEqual(set(module._select_models()), {"silero-vad", "ct-punc"})
        with patch.dict(os.environ, {module.INSTALL_ALL_ENV: "no-such-model"}):
            with self.assertRaises(SystemExit):
                module._select_models()

    def test_main_dispatch_and_env_output(self):
        module = _load_install_models()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(module, "_install", side_effect=lambda n, s, r: root / n),
                patch.object(module, "_install_files", side_effect=lambda n, s, r: root / s["dir_name"]),
                patch.object(module, "_install_raw", side_effect=lambda n, s, r: root),
                # CI 里 GITHUB_ENV 已设，main() 会优先写它——显式指到 scratch
                # 文件（test_paraformer_backend 家规），断言改读该文件。
                patch.dict(os.environ, {
                    "COURSELENS_MODEL_ROOT": str(root),
                    "GITHUB_ENV": str(root / "github_env.txt"),
                }),
            ):
                module.main()
            lines = (root / "github_env.txt").read_text(encoding="utf-8").splitlines()
        # 默认集含 ct-punc、不含 silero（CTPUNC-DEF-1 件2）
        self.assertTrue(any(line.startswith("CTPUNC_MODEL_DIR=") for line in lines))
        self.assertNotIn(f"SILERO_MODEL_DIR={root}", lines)
        self.assertTrue(any(line.startswith("SENSEVOICE_MODEL_DIR=") for line in lines))
        self.assertTrue(any(line.startswith("PARAFORMER_MODEL_DIR=") for line in lines))

    def test_main_explicit_opt_in_installs_silero(self):
        module = _load_install_models()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(module, "_install", side_effect=lambda n, s, r: root / n),
                patch.object(module, "_install_files", side_effect=lambda n, s, r: root / s["dir_name"]),
                patch.object(module, "_install_raw", side_effect=lambda n, s, r: root),
                patch.dict(os.environ, {
                    "COURSELENS_MODEL_ROOT": str(root),
                    "GITHUB_ENV": str(root / "github_env.txt"),
                    module.INSTALL_ALL_ENV: "silero-vad",
                }),
            ):
                module.main()
            lines = (root / "github_env.txt").read_text(encoding="utf-8").splitlines()
        self.assertIn(f"SILERO_MODEL_DIR={root}", lines)
        self.assertFalse(any(line.startswith("CTPUNC_MODEL_DIR=") for line in lines))

    def test_install_files_rejects_bad_hash_without_residue(self):
        module = _load_install_models()
        good = "a" * 64
        spec = {
            "files": (("model.onnx", good),),
            "base": "https://example.com/base",
            "dir_name": "m",
            "sha256": "x",
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self):
                return None

            def iter_content(self, size):
                yield b"payload"

        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(module.requests, "get", return_value=FakeResponse()):
                with self.assertRaises(RuntimeError):
                    module._install_files("m", spec, Path(temporary))
            self.assertFalse((Path(temporary) / "m" / "model.onnx.part").exists())

    def test_install_files_reuses_verified_files(self):
        module = _load_install_models()
        import hashlib

        payload = b"payload"
        digest = hashlib.sha256(payload).hexdigest()
        spec = {
            "files": (("model.onnx", digest),),
            "base": "https://example.com/base",
            "dir_name": "m",
            "sha256": "x",
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self):
                return None

            def iter_content(self, size):
                yield payload

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(module.requests, "get", return_value=FakeResponse()) as get_mock:
                module._install_files("m", spec, root)
                self.assertEqual(get_mock.call_count, 1)
                # 复跑：已验证文件跳过下载
                module._install_files("m", spec, root)
                self.assertEqual(get_mock.call_count, 1)
            self.assertTrue((root / "m" / "model.onnx").is_file())
            expected_marker = f".m-{module._files_marker_digest(spec)}.ready"
            self.assertTrue((root / expected_marker).is_file())


class SummaryGlossaryTests(unittest.TestCase):
    """件6：摘要术语注入——window 提示词变体 + merge 数据通道，旧调用逐位不变。"""

    def _chat_recorder(self, responses):
        calls = []

        def _chat(api_key, messages, **kwargs):
            calls.append([dict(item) for item in messages])
            return json.dumps(responses[min(len(calls) - 1, len(responses) - 1)])

        return _chat, calls

    def test_glossary_flows_into_window_prompt_and_merge_input(self):
        from courselens_worker.llm import (
            _SUMMARY_WINDOW_PROMPT_WITH_GLOSSARY,
            REVIEW_VIEWS_ENV,
            create_summary,
        )

        window = {"markdown": "笔记", "chapters": []}
        merge = {"markdown": "# 笔记", "chapters": []}
        _chat, calls = self._chat_recorder([window, merge])
        # 本钉=window 提示词变体与 merge 数据通道（视图派生在 test_review_views.py）。
        with patch("courselens_worker.llm._chat", _chat), \
                patch.dict(os.environ, {REVIEW_VIEWS_ENV: "off"}):
            create_summary(
                "key", title="t", transcript=[{"start_ms": 0, "end_ms": 900, "text": "内容"}],
                ppt_pages=[], glossary=("能带图", "电势"),
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0]["content"], _SUMMARY_WINDOW_PROMPT_WITH_GLOSSARY)
        window_payload = json.loads(calls[0][1]["content"])
        self.assertEqual(window_payload["glossary"], ["能带图", "电势"])
        merge_payload = json.loads(calls[1][1]["content"])
        self.assertEqual(merge_payload["glossary"], ["能带图", "电势"])

    def test_no_glossary_keeps_prompts_byte_identical(self):
        from courselens_worker.llm import (
            _SUMMARY_WINDOW_PROMPT,
            create_summary,
        )

        window = {"markdown": "笔记", "chapters": []}
        merge = {"markdown": "# 笔记", "chapters": []}
        _chat, calls = self._chat_recorder([window, merge])
        with patch("courselens_worker.llm._chat", _chat):
            create_summary(
                "key", title="t", transcript=[{"start_ms": 0, "end_ms": 900, "text": "内容"}],
                ppt_pages=[],
            )
        self.assertEqual(calls[0][0]["content"], _SUMMARY_WINDOW_PROMPT)
        self.assertNotIn("glossary", json.loads(calls[1][1]["content"]))

    def test_merge_prompt_stays_within_300_pin(self):
        from courselens_worker.llm import _SUMMARY_MERGE_PROMPT

        self.assertLessEqual(len(_SUMMARY_MERGE_PROMPT), 300)


if __name__ == "__main__":
    unittest.main()
