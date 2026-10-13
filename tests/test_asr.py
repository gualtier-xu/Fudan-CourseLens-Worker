from __future__ import annotations

import unittest
from pathlib import Path
import sys
import tempfile
from contextlib import contextmanager
from unittest.mock import Mock, patch

from courselens_worker import asr

# 夜10-C T23：可选依赖（sherpa_onnx/numpy）由 conftest 在收集前按需装桩。
# 此前的 with patch.dict 整体还原会在退出时把窗口内导入的 numpy 一并逐出，
# 下一个测试模块再导入即得第二实例（哨兵失配跨文件炸穿）；且 asr/platform_session
# 被逐出后再 import 会二次执行出第二个类对象——梯的 except 咬不住测试抛的
# 异常（P55 自检实测踩中）。现在全进程共享同一模块实例，此类问题整体消除。
PlatformSessionError = asr.PlatformSessionError


class ASRProxyLifecycleTests(unittest.TestCase):
    def test_all_pcm_chunks_share_one_pinned_media_session(self):
        pool = Mock()
        pool.transcribe_pcm.side_effect = lambda _path, _backend, *, offset_seconds: [{
            "start_ms": int(offset_seconds * 1000),
            "end_ms": int(offset_seconds * 1000) + 1000,
            "text": "test",
        }]
        progress = Mock()

        def prefetch(_url, target, *, duration):
            self.assertGreater(duration, 0)
            Path(target).write_bytes(b"")

        def slice_pcm(_full, target, *, offset, duration):
            self.assertGreaterEqual(offset, 0)
            self.assertGreater(duration, 0)
            target.write_bytes(b"pcm")

        with (
            patch.object(asr, "RecognizerPool", return_value=pool),
            patch.object(asr, "pinned_media_proxy") as media_proxy,
            patch.object(asr, "_prefetch_media_pcm", side_effect=prefetch) as prefetch_mock,
            patch.object(asr, "_slice_pcm_chunk", side_effect=slice_pcm) as materialize,
        ):
            proxy = media_proxy.return_value.__enter__.return_value
            proxy.url = "http://127.0.0.1/session"
            result = asr.transcribe(
                {
                    "payload": {
                        "mode": "automatic",
                        "media": {
                            "url": "https://media.example.com/lecture.mp4",
                            "duration_seconds": 1250,
                        },
                    },
                },
                sensevoice_dir=Mock(),
                proofread=None,
                progress=progress,
            )

        media_proxy.assert_called_once()
        # 夜10-C 第七波①：预取恰一次，之后任务中段零校方请求（refresh 恒 0）
        self.assertEqual(prefetch_mock.call_count, 1)
        self.assertEqual(proxy.refresh_source.call_count, 0)
        self.assertEqual(materialize.call_count, 3)
        self.assertEqual(
            [item.kwargs["offset"] for item in materialize.call_args_list],
            [0.0, 600.0, 1200.0],
        )
        self.assertEqual([item.args for item in progress.call_args_list], [
            ("asr", 1, 3),
            ("asr", 2, 3),
            ("asr", 3, 3),
        ])
        self.assertEqual(result["metrics"]["chunks"], 3)


class ASREvidenceIdentityTests(unittest.TestCase):
    """Evidence provenance stamping over the mocked transcribe flow."""

    def _pool(self):
        pool = Mock()
        pool.transcribe_pcm.side_effect = lambda _path, backend, *, offset_seconds: [{
            "start_ms": int(offset_seconds * 1000),
            "end_ms": int(offset_seconds * 1000) + 1000,
            "text": f"{backend}@{int(offset_seconds)}",
        }]
        return pool

    def _run(self, pool, *, mode="automatic", prior=None, capture=None, media_seconds=1250, proofread="mock"):
        payload = {
            "mode": mode,
            "media": {
                "url": "https://media.example.com/lecture.mp4",
                "duration_seconds": media_seconds,
            },
        }
        if prior is not None:
            payload["checkpoint"] = prior
        proofread_fn = (
            Mock(return_value=[{"start_ms": 0, "end_ms": 1000, "text": "校对后"}])
            if proofread == "mock" else proofread
        )
        if proofread is None:
            proofread_fn = None
        with (
            patch.object(asr, "RecognizerPool", return_value=pool),
            patch.object(asr, "pinned_media_proxy"),
            patch.object(asr, "_prefetch_media_pcm",
                         side_effect=lambda _url, target, *, duration: Path(target).write_bytes(b"")),
            patch.object(asr, "_slice_pcm_chunk",
                         side_effect=lambda _full, target, *, offset, duration: target.write_bytes(b"pcm")),
        ):
            return asr.transcribe(
                {"payload": payload},
                sensevoice_dir=Mock(),
                proofread=proofread_fn,
                progress=Mock(),
                checkpoint=capture,
            )

    def test_fresh_run_stamps_contract_provenance(self):
        checkpoints = []
        result = self._run(self._pool(), capture=checkpoints.append, proofread=None)
        segments = result["segments"]
        self.assertEqual(len(segments), 3)
        for segment in segments:
            self.assertRegex(segment["segment_id"], r"^seg:[0-9a-f]{12}$")
            self.assertRegex(segment["source_hash"], r"^[0-9a-f]{64}$")
            self.assertEqual(segment["provenance"]["producer"], asr.PRODUCER_ID)
            self.assertEqual(segment["provenance"]["model"], "paraformer")
            self.assertRegex(segment["provenance"]["config_hash"], r"^[0-9a-f]{12,64}$")
        # 每个 chunk 的检查点都携带可续跑的非秘密指纹状态
        self.assertEqual(len(checkpoints), 3)
        for checkpoint in checkpoints:
            self.assertRegex(checkpoint["pcm_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertNotRegex(str(checkpoints), r"media\.example\.com")

    def test_resumed_run_reproduces_the_fresh_run_identity(self):
        pool = self._pool()
        checkpoints = []
        fresh = self._run(pool, capture=checkpoints.append, proofread=None)
        interrupted = checkpoints[0]
        resumed = self._run(pool, prior=interrupted, proofread=None)
        self.assertEqual(
            [(item["start_ms"], item["end_ms"], item["text"]) for item in fresh["segments"]],
            [(item["start_ms"], item["end_ms"], item["text"]) for item in resumed["segments"]],
        )
        self.assertEqual(
            [item["segment_id"] for item in fresh["segments"]],
            [item["segment_id"] for item in resumed["segments"]],
        )
        self.assertEqual(
            fresh["segments"][0]["source_hash"],
            resumed["segments"][0]["source_hash"],
        )

    def test_checkpoint_without_current_chain_raw_fails_explicitly(self):
        # M4-ENABLE-1 U4：缺当前链精修 raw 的检查点（如退役链产物）续跑必须
        # 显式失败——静默混链会丢前段输出。
        pool = self._pool()
        prior = {
            "completed_chunks": 1,
            "total_chunks": 3,
            "mode": "automatic",
            "raw_sensevoice": [{"start_ms": 0, "end_ms": 1000, "text": "legacy@0"}],
        }
        with self.assertRaises(asr.ASRError):
            self._run(pool, prior=prior, proofread=None)

    def test_legacy_checkpoint_omits_unverifiable_provenance(self):
        pool = self._pool()
        checkpoints = []
        prior = {
            "completed_chunks": 1,
            "total_chunks": 3,
            "mode": "automatic",
            "raw_sensevoice": [{"start_ms": 0, "end_ms": 1000, "text": "legacy@0"}],
            "raw_paraformer": [{"start_ms": 0, "end_ms": 1000, "text": "legacy-fire@0"}],
        }
        result = self._run(pool, prior=prior, capture=checkpoints.append, proofread=None)
        self.assertEqual(len(result["segments"]), 3)
        for segment in result["segments"]:
            self.assertNotIn("segment_id", segment)
            self.assertNotIn("evidence_id", segment)
            self.assertNotIn("source_hash", segment)
            self.assertNotIn("provenance", segment)
        for checkpoint in checkpoints:
            self.assertNotIn("pcm_fingerprint", checkpoint)

    def test_automatic_proofread_stamps_final_and_raw_segments(self):
        pool = self._pool()
        result = self._run(pool, media_seconds=600)
        self.assertEqual(result["segments"][0]["text"], "校对后")
        self.assertEqual(
            result["segments"][0]["provenance"]["model"],
            "sensevoice+paraformer:proofread",
        )
        self.assertEqual(result["raw_sensevoice"][0]["provenance"]["model"], "sensevoice")
        self.assertEqual(result["raw_paraformer"][0]["provenance"]["model"], "paraformer")
        # 校对后的文本是独立证据：final ID 不与 raw ID 共享
        final_id = result["segments"][0]["segment_id"]
        raw_ids = {
            result["raw_sensevoice"][0]["segment_id"],
            result["raw_paraformer"][0]["segment_id"],
        }
        self.assertNotIn(final_id, raw_ids)


class MediaPrefetchPins(unittest.TestCase):
    """夜10-C 第七波①：媒体开局预取——任务中段零校方请求（跨期 runner
    再认证墙根修）。预取失败按既有闭集媒体码如实失败；预取后的分块全部
    本地切片，块界授权刷新不再进入循环（梯函数保留为库语义，单测见下组）。
    """

    def _pool(self):
        pool = Mock()
        pool.transcribe_pcm.side_effect = lambda _path, backend, *, offset_seconds: [{
            "start_ms": int(offset_seconds * 1000),
            "end_ms": int(offset_seconds * 1000) + 1000,
            "text": backend,
        }]
        return pool

    def _run_with(self, prefetch_side_effect):
        pool = self._pool()
        with (
            patch.object(asr, "RecognizerPool", return_value=pool),
            patch.object(asr, "pinned_media_proxy") as media_proxy,
            patch.object(asr, "_prefetch_media_pcm", side_effect=prefetch_side_effect) as prefetch,
            patch.object(asr, "_slice_pcm_chunk",
                         side_effect=lambda _full, target, *, offset, duration: target.write_bytes(b"pcm")),
        ):
            proxy = media_proxy.return_value.__enter__.return_value
            proxy.url = "http://127.0.0.1/session"
            result = asr.transcribe(
                {
                    "payload": {
                        "mode": "automatic",
                        "media": {
                            "url": "https://media.example.com/lecture.mp4",
                            "duration_seconds": 1250,
                        },
                    },
                },
                sensevoice_dir=Mock(),
                proofread=None,
                progress=Mock(),
            )
        return result, proxy, prefetch

    def test_prefetch_runs_once_and_boundaries_never_touch_the_school(self):
        calls = {"count": 0}

        def prefetch(_url, target, *, duration):
            calls["count"] += 1
            Path(target).write_bytes(b"")

        result, proxy, prefetch_mock = self._run_with(prefetch)
        self.assertEqual(result["metrics"]["chunks"], 3)
        self.assertEqual(calls["count"], 1, "预取恰一次")
        self.assertEqual(proxy.refresh_source.call_count, 0,
                         "ASR 主循环零校方请求（墙已根修）")

    def test_prefetch_failure_fails_closed_with_media_code_without_refresh(self):
        pool = self._pool()
        with (
            patch.object(asr, "RecognizerPool", return_value=pool),
            patch.object(asr, "pinned_media_proxy") as media_proxy,
            patch.object(asr, "_prefetch_media_pcm",
                         side_effect=lambda _u, _t, *, duration: (_ for _ in ()).throw(
                             asr.ASRError("authorized media upstream connection failed"))),
        ):
            proxy = media_proxy.return_value.__enter__.return_value
            with self.assertRaises(asr.ASRError) as caught:
                asr.transcribe(
                    {
                        "payload": {
                            "mode": "automatic",
                            "media": {
                                "url": "https://media.example.com/lecture.mp4",
                                "duration_seconds": 1250,
                            },
                        },
                    },
                    sensevoice_dir=Mock(),
                    proofread=None,
                    progress=Mock(),
                )
        self.assertEqual(
            str(caught.exception),
            "authorized media upstream connection failed",
            "预取失败按既有闭集媒体码如实失败",
        )
        self.assertEqual(proxy.refresh_source.call_count, 0,
                         "预取期失败不触发任何块界刷新（一次性授权边界）")

    def test_slice_pcm_chunk_zero_byte_mid_media_fails_closed(self):
        # 预取截断（中段切片为空且请求时长为正）→ 真实根因闭集码
        # media_prefetch_incomplete，绝不静默空段、不再错标 decode 失败
        with tempfile.TemporaryDirectory() as folder:
            full = Path(folder) / "full.f32le"
            full.write_bytes(b"")
            target = Path(folder) / "chunk.f32le"
            with self.assertRaises(asr.ASRError) as caught:
                asr._slice_pcm_chunk(full, target, offset=600.0, duration=600.0)
            self.assertEqual(
                str(caught.exception),
                "authorized media prefetch was incomplete",
            )
            self.assertFalse(target.exists())


class RefreshLadderLibraryPins(unittest.TestCase):
    """P55 有界梯语义保留为库钉（夜10-C 第七波①后主循环不再进入，
    梯本身仍是媒体授权路径的守卫面——直接以 mock proxy 单测梯语义）。"""

    def _ladder(self, proxy, *, chunk=1):
        lines = []
        with patch.object(asr.time, "sleep"):
            asr._refresh_media_authorization(proxy, chunk=chunk, elapsed=lambda: 0)
        return lines

    def test_transient_failure_retries_then_succeeds(self):
        proxy = Mock()
        outcomes = [PlatformSessionError("platform_auth_context_missing"), None]
        def refresh():
            outcome = outcomes.pop(0)
            if outcome is not None:
                raise outcome
        proxy.refresh_source.side_effect = refresh
        asr._refresh_media_authorization(proxy, chunk=1, elapsed=lambda: 0)
        self.assertEqual(proxy.refresh_source.call_count, 2)

    def test_ladder_exhaustion_raises_last_closed_set_code(self):
        proxy = Mock()
        proxy.refresh_source.side_effect = PlatformSessionError("platform_auth_context_missing")
        with self.assertRaises(PlatformSessionError) as caught:
            asr._refresh_media_authorization(proxy, chunk=1, elapsed=lambda: 0)
        self.assertEqual(str(caught.exception), "platform_auth_context_missing")
        self.assertEqual(proxy.refresh_source.call_count, 3, "有界梯恰 3 次")

    def test_deterministic_code_never_enters_the_ladder(self):
        proxy = Mock()
        proxy.refresh_source.side_effect = PlatformSessionError("platform_media_missing")
        with self.assertRaises(PlatformSessionError) as caught:
            asr._refresh_media_authorization(proxy, chunk=1, elapsed=lambda: 0)
        self.assertEqual(str(caught.exception), "platform_media_missing")
        self.assertEqual(proxy.refresh_source.call_count, 1)


class ProofreadDegradationTests(unittest.TestCase):
    """G7（ASRBENCH P1）：AI 校对失败降级交付原始识别，不整单带崩。"""

    def test_proofread_failure_degrades_to_raw_segments_with_warning(self):
        pool = Mock()
        pool.transcribe_pcm.side_effect = lambda _path, backend, *, offset_seconds: [{
            "start_ms": int(offset_seconds * 1000),
            "end_ms": int(offset_seconds * 1000) + 1000,
            "text": backend,
        }]
        # 注意：用 asr.LLMError 保证与生产 except 引用同一类对象
        # （双路径导入下 courselens_worker.llm 可能存在两个模块实例）。
        flaky = Mock(side_effect=asr.LLMError("boom"))

        def create_pcm(_url, target, *, offset, duration):
            target.write_bytes(b"pcm")

        with (
            patch.object(asr, "RecognizerPool", return_value=pool),
            patch.object(asr, "pinned_media_proxy"),
            patch.object(asr, "_prefetch_media_pcm",
                         side_effect=lambda _url, target, *, duration: Path(target).write_bytes(b"")),
            patch.object(asr, "_slice_pcm_chunk",
                         side_effect=lambda _full, target, *, offset, duration: target.write_bytes(b"pcm")),
        ):
            result = asr.transcribe(
                {
                    "payload": {
                        "mode": "automatic",
                        "media": {
                            "url": "https://media.example.com/lecture.mp4",
                            "duration_seconds": 1250,
                        },
                    },
                },
                sensevoice_dir=Mock(),
                proofread=flaky,
                progress=Mock(),
            )
        self.assertEqual(result["warnings"], ["proofread_degraded"])
        # 交付的是原始精修识别结果（三条 chunk 段），不是空手而归
        self.assertEqual(len(result["segments"]), 3)
        self.assertNotIn(":proofread", str(result))

    def test_proofread_success_has_no_warning(self):
        pool = Mock()
        pool.transcribe_pcm.side_effect = lambda _path, backend, *, offset_seconds: [{
            "start_ms": int(offset_seconds * 1000),
            "end_ms": int(offset_seconds * 1000) + 1000,
            "text": backend,
        }]
        good = Mock(return_value=[{
            "start_ms": 0, "end_ms": 1000, "text": "校对后",
        }])

        def create_pcm(_url, target, *, offset, duration):
            target.write_bytes(b"pcm")

        with (
            patch.object(asr, "RecognizerPool", return_value=pool),
            patch.object(asr, "pinned_media_proxy"),
            patch.object(asr, "_prefetch_media_pcm",
                         side_effect=lambda _url, target, *, duration: Path(target).write_bytes(b"")),
            patch.object(asr, "_slice_pcm_chunk",
                         side_effect=lambda _full, target, *, offset, duration: target.write_bytes(b"pcm")),
        ):
            result = asr.transcribe(
                {
                    "payload": {
                        "mode": "automatic",
                        "media": {
                            "url": "https://media.example.com/lecture.mp4",
                            "duration_seconds": 1250,
                        },
                    },
                },
                sensevoice_dir=Mock(),
                proofread=good,
                progress=Mock(),
            )
        self.assertNotIn("warnings", result)


class DurationProbeRetryTests(unittest.TestCase):
    """A5 邻接扫：时长探测与分块同族——秒败先重取会话材料再试一次。"""

    def test_probe_recovers_after_one_refresh_retry(self):
        calls = []

        def fake_run(source, command, *, timeout, capture_stdout):
            calls.append(dict(source))
            if len(calls) == 1:
                return 1, b"", b"err"
            return 0, b"1250.5", b""

        with patch.object(asr, "_run_media_proxy", side_effect=fake_run):
            self.assertEqual(asr._probe_duration({"url": "https://media.example.com/l.mp4"}), 1250.5)
        self.assertEqual(len(calls), 2)

    def test_probe_fails_closed_after_bounded_budget(self):
        with patch.object(asr, "_run_media_proxy", return_value=(1, b"", b"e")):
            with self.assertRaises(asr.ASRError) as caught:
                asr._probe_duration({"url": "https://media.example.com/l.mp4"})
        self.assertEqual(
            str(caught.exception),
            "authorized media duration could not be determined",
        )

    def test_probe_timeout_fails_fast_without_second_attempt(self):
        import subprocess

        def fake_run(source, command, *, timeout, capture_stdout):
            raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=120)

        with patch.object(asr, "_run_media_proxy", side_effect=fake_run):
            with self.assertRaises(asr.ASRError) as caught:
                asr._probe_duration({"url": "https://media.example.com/l.mp4"})
        self.assertEqual(
            str(caught.exception),
            "authorized media duration probe timed out",
        )


if __name__ == "__main__":
    unittest.main()
