"""MEDIA-FIX-PREFETCH-1：媒体 prefetch 完整性门与截断归真钉。

452282 定因（2026-10-02）：上游提前断流（41.3% 处 clean FIN）时代理转发
read(amt) 静默空串收尾、ffmpeg 对提前 EOF 按正常输入结束 exit 0，截断 PCM
静默过关——chunk6 短块静默解码、chunk7 空切片才以错位的 media_decode_failed
迟败。本文件钉四层守卫：

1. ``_prefetch_media_pcm`` 期望字节门（时长×采样率×4，1s 容差）；
2. ``_slice_pcm_chunk`` 短块/空切片 fail-closed（真实根因上行）；
3. 闭集码映射 ``media_prefetch_incomplete`` 与 transcribe 层有界重取梯
   （恰一次重取，重取前 refresh_source 换新签名 URL）；
4. 真 ffmpeg + 真 截断 HTTP 等价复现（ffmpeg 在位时；配方=
   声明全量 Content-Length、发送部分字节后 FIN，452282 沙箱 CONFIRMED 同款）。
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

from courselens_worker import asr
from courselens_worker.runner import safe_worker_error_detail

INCOMPLETE = "authorized media prefetch was incomplete"


def _pcm_bytes(seconds: float) -> int:
    return int(round(seconds * asr.SAMPLE_RATE)) * asr._PCM_SAMPLE_BYTES


class PrefetchCompletenessGateTests(unittest.TestCase):
    """期望字节门：ffmpeg 打桩为「解码出给定秒数的 PCM 并 exit 0」。"""

    def _run_prefetch(self, *, decoded_seconds: float, duration: float) -> None:
        def fake_run(command, timeout=None, capture_stdout=False):
            if decoded_seconds > 0:
                Path(command[-1]).write_bytes(b"\x00" * _pcm_bytes(decoded_seconds))
            return 0, None, b""

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "media-full.f32le"
            with patch.object(asr, "_run_bounded_process", side_effect=fake_run):
                asr._prefetch_media_pcm("http://127.0.0.1/session", target, duration=duration)
            self.assertTrue(target.exists(), "门内通过时不得删除产物")

    def test_full_decode_passes(self):
        self._run_prefetch(decoded_seconds=600.0, duration=600.0)

    def test_sub_second_shortfall_within_tolerance_passes(self):
        # 容器时长元数据与实际解码样本的常规毫秒级偏差（<1s）不误伤
        self._run_prefetch(decoded_seconds=599.5, duration=600.0)

    def test_truncation_beyond_tolerance_raises_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "media-full.f32le"

            def fake_run(command, timeout=None, capture_stdout=False):
                # 7034.99s / 9768s ≈ 真跑 41.3% 覆盖形状
                Path(command[-1]).write_bytes(b"\x00" * _pcm_bytes(4034.99))
                return 0, None, b""

            with patch.object(asr, "_run_bounded_process", side_effect=fake_run):
                with self.assertRaises(asr.ASRError) as caught:
                    asr._prefetch_media_pcm("http://127.0.0.1/session", target, duration=9768.0)
            self.assertEqual(str(caught.exception), INCOMPLETE)
            self.assertFalse(target.exists(), "截断产物必须删除，绝不留给分块消费")

    def test_shortfall_just_beyond_one_second_tolerance_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "media-full.f32le"

            def fake_run(command, timeout=None, capture_stdout=False):
                Path(command[-1]).write_bytes(b"\x00" * _pcm_bytes(597.0))
                return 0, None, b""

            with patch.object(asr, "_run_bounded_process", side_effect=fake_run):
                with self.assertRaises(asr.ASRError) as caught:
                    asr._prefetch_media_pcm("http://127.0.0.1/session", target, duration=600.0)
            self.assertEqual(str(caught.exception), INCOMPLETE)


class SliceShortBlockGateTests(unittest.TestCase):
    """分块切片门：满块精确、短块/空切片按真实根因 fail-closed（真文件 I/O）。"""

    def test_full_chunk_slices_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            full = Path(folder) / "full.f32le"
            full.write_bytes(b"\x00" * _pcm_bytes(600.0))
            target = Path(folder) / "chunk.f32le"
            asr._slice_pcm_chunk(full, target, offset=0.0, duration=600.0)
            self.assertEqual(target.stat().st_size, _pcm_bytes(600.0))

    def test_mid_chunk_offsets_are_byte_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            full = Path(folder) / "full.f32le"
            payload = bytes(range(256)) * ((_pcm_bytes(100.0) // 256) + 1)
            full.write_bytes(payload[: _pcm_bytes(100.0)])
            target = Path(folder) / "chunk.f32le"
            asr._slice_pcm_chunk(full, target, offset=30.0, duration=20.0)
            expected_start = int(30.0 * asr.SAMPLE_RATE) * asr._PCM_SAMPLE_BYTES
            self.assertEqual(
                target.read_bytes(),
                payload[expected_start: expected_start + _pcm_bytes(20.0)],
            )

    def test_short_tail_chunk_within_tolerance_passes(self):
        # 尾块实解码 599.5s/600s：0.5s 缺口在容差带内，正常收尾
        with tempfile.TemporaryDirectory() as folder:
            full = Path(folder) / "full.f32le"
            full.write_bytes(b"\x00" * _pcm_bytes(599.5))
            target = Path(folder) / "chunk.f32le"
            asr._slice_pcm_chunk(full, target, offset=0.0, duration=600.0)
            self.assertEqual(target.stat().st_size, _pcm_bytes(599.5))

    def test_shortfall_chunk6_shape_fails_closed(self):
        # 452282 chunk6 形状：请求满块、PCM 只剩部分——短块不再静默通过
        with tempfile.TemporaryDirectory() as folder:
            full = Path(folder) / "full.f32le"
            full.write_bytes(b"\x00" * _pcm_bytes(434.99))
            target = Path(folder) / "chunk.f32le"
            with self.assertRaises(asr.ASRError) as caught:
                asr._slice_pcm_chunk(full, target, offset=0.0, duration=600.0)
            self.assertEqual(str(caught.exception), INCOMPLETE)
            self.assertFalse(target.exists())

    def test_empty_slice_beyond_eof_reports_prefetch_incomplete(self):
        # 452282 chunk7 形状：切片起点越过 PCM 末尾——错位码归正的正面钉
        with tempfile.TemporaryDirectory() as folder:
            full = Path(folder) / "full.f32le"
            full.write_bytes(b"\x00" * _pcm_bytes(100.0))
            target = Path(folder) / "chunk.f32le"
            with self.assertRaises(asr.ASRError) as caught:
                asr._slice_pcm_chunk(full, target, offset=150.0, duration=50.0)
            self.assertEqual(str(caught.exception), INCOMPLETE)
            self.assertNotEqual(
                str(caught.exception),
                "ffmpeg could not decode the authorized media stream",
                "空切片不得再错标 decode 失败",
            )
            self.assertFalse(target.exists())


class PrefetchErrorCodeTests(unittest.TestCase):
    def test_incomplete_message_maps_to_closed_set_code(self):
        code = safe_worker_error_detail(asr.ASRError(INCOMPLETE))
        self.assertEqual(code, "media_prefetch_incomplete")


class PrefetchRefetchLadderTests(unittest.TestCase):
    """transcribe 层有界重取梯：截断码恰一次重取，重取前换新签名 URL。"""

    def _run_transcribe(self, prefetch_side_effect):
        pool = Mock()
        pool.transcribe_pcm.side_effect = lambda _path, backend, *, offset_seconds: [{
            "start_ms": int(offset_seconds * 1000),
            "end_ms": int(offset_seconds * 1000) + 1000,
            "text": backend,
        }]
        with (
            patch.object(asr, "RecognizerPool", return_value=pool),
            patch.object(asr, "pinned_media_proxy") as media_proxy,
            patch.object(asr, "_prefetch_media_pcm", side_effect=prefetch_side_effect) as prefetch,
            patch.object(asr, "_slice_pcm_chunk",
                         side_effect=lambda _full, target, *, offset, duration: target.write_bytes(b"pcm")),
            patch.object(asr.time, "sleep") as sleep_mock,
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
        return result, proxy, prefetch, sleep_mock

    def test_truncated_prefetch_refetches_once_with_fresh_authorization(self):
        attempts = {"count": 0}

        def prefetch(_url, target, *, duration):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise asr.ASRError(INCOMPLETE)
            Path(target).write_bytes(b"")

        result, proxy, prefetch_mock, sleep_mock = self._run_transcribe(prefetch)
        self.assertEqual(result["metrics"]["chunks"], 3, "重取成功后任务正常完成")
        self.assertEqual(prefetch_mock.call_count, 2, "截断码恰重取一次")
        self.assertEqual(proxy.refresh_source.call_count, 1, "重取前换新签名 URL")
        sleep_mock.assert_called_once_with(asr._MEDIA_RETRY_BACKOFF_SECONDS[0])

    def test_ladder_exhaustion_fails_with_incomplete_code(self):
        def prefetch(_url, _target, *, duration):
            raise asr.ASRError(INCOMPLETE)

        with self.assertRaises(asr.ASRError) as caught:
            self._run_transcribe(prefetch)
        self.assertEqual(str(caught.exception), INCOMPLETE)

    def test_exhaustion_stops_at_one_refetch(self):
        calls = {"count": 0}

        def prefetch(_url, _target, *, duration):
            calls["count"] += 1
            raise asr.ASRError(INCOMPLETE)

        proxy = Mock()
        with (
            patch.object(asr, "RecognizerPool", return_value=Mock()),
            patch.object(asr, "pinned_media_proxy") as media_proxy,
            patch.object(asr, "_prefetch_media_pcm", side_effect=prefetch),
            patch.object(asr.time, "sleep"),
        ):
            media_proxy.return_value.__enter__.return_value = proxy
            proxy.url = "http://127.0.0.1/session"
            with self.assertRaises(asr.ASRError):
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
        self.assertEqual(calls["count"], 2, "初跑+恰一次重取，梯尽即败")
        self.assertEqual(proxy.refresh_source.call_count, 1)


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not available")
class TruncatedStreamEquivalenceTests(unittest.TestCase):
    """452282 等价复现（真 ffmpeg + 真 截断 HTTP）。

    截断配方（DIAG 沙箱 CONFIRMED 同款）：声明全量 Content-Length、发送
    部分字节后 FIN——ffmpeg 对提前 EOF exit 0，完整性门必须以
    media_prefetch_incomplete 拦下，全量流则按期望字节通过。
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="mfix-it-")
        cls.media = Path(cls._tmp.name) / "synthetic.mp4"
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
                "-c:a", "aac", "-movflags", "+faststart", "-y", str(cls.media),
            ],
            check=True, timeout=120, capture_output=True,
        )
        cls.payload = cls.media.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _serve(self, fraction: float) -> str:
        payload = self.payload
        limit = max(1, int(len(payload) * fraction))

        class TruncatedHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "audio/mp4")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload[:limit])
                self.close_connection = True

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), TruncatedHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        # addCleanup LIFO：shutdown → join → server_close，避免关套接字时
        # serve_forever 仍在 select 上唤醒（WinError 10038 噪声）
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}/media.mp4"

    def test_truncated_stream_fails_closed_with_incomplete_code(self):
        url = self._serve(0.35)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "media-full.f32le"
            with self.assertRaises(asr.ASRError) as caught:
                asr._prefetch_media_pcm(url, target, duration=12.0)
            self.assertEqual(str(caught.exception), INCOMPLETE)
            self.assertFalse(target.exists())

    def test_full_stream_passes_with_expected_pcm_coverage(self):
        url = self._serve(1.0)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "media-full.f32le"
            asr._prefetch_media_pcm(url, target, duration=12.0)
            size = target.stat().st_size
            self.assertGreaterEqual(size, _pcm_bytes(12.0))
            self.assertLess(size, _pcm_bytes(14.0), "远超期望时长的产出同样异常")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
