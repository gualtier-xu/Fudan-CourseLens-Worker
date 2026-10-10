"""N12（WORKER-R2 2026-10-07）：媒体预取并行 Range 分段——慢网兜底与零回退钉。

生产实测单流带宽方差 2.1↔8.7MB/s：慢网端预取占媒体墙钟 15-25%，并行分段
可压约一半。本文件钉五面：

1. Range 能力源：探测→分段→拼接→同一 ffmpeg 命令→同一完整性门，慢网端
   收益路径全链可用（拼接逐字节等价，fake ffmpeg 捕获 ``-i`` 输入逐位比对）；
2. 上游不认 Range（带 Range 请求回全量 200）→ 逐字回落单流（``-i`` 收到
   原 URL），零行为回退；
3. 探测连接失败（口未听）→ 回落单流；
4. 段级短读：退避梯内重试成功通过；梯尽按 ``media_prefetch_incomplete``
   如实失败，段目录/拼接容器/截断产物全清理；
5. 真 ffmpeg 端到端（在位时）：真实 mp4 容器经并行分段预取解码过完整性门；
   段截断变体在段级即拦下（解码前），同码如实失败。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from courselens_worker import asr

INCOMPLETE = "authorized media prefetch was incomplete"


def _pcm_bytes(seconds: float) -> int:
    return int(round(seconds * asr.SAMPLE_RATE)) * asr._PCM_SAMPLE_BYTES


class _QuietServer(ThreadingHTTPServer):
    """Silence handler-thread tracebacks from intentionally early-closed probes."""

    def handle_error(self, _request, _client_address):
        return


def _serve(
    payload: bytes,
    *,
    support_range: bool = True,
    truncate_ranges: set[tuple[int, int]] | None = None,
    truncate_once: bool = False,
) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    """Local origin serving exact byte ranges with per-test behavior knobs.

    ``truncate_ranges``：这些 (start, end) 段声明全长 Content-Range 但只发送
    一半字节后 FIN（452282 截断配方段级形）；``truncate_once`` 时同段自第二
    次请求起恢复全长（重试恢复面）。
    """
    total = len(payload)
    truncate_ranges = truncate_ranges or set()
    seen: dict[tuple[int, int], int] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            range_header = str(self.headers.get("Range") or "").strip()
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", range_header)
            if not support_range or not match:
                self.send_response(200)
                self.send_header("Content-Length", str(total))
                self.end_headers()
                self.wfile.write(payload)
                return
            key = (int(match.group(1)), int(match.group(2)))
            start, end = key
            chunk = payload[start: end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            if key in truncate_ranges:
                seen[key] = seen.get(key, 0) + 1
                if not (truncate_once and seen[key] > 1):
                    self.wfile.write(chunk[: len(chunk) // 2])
                    self.close_connection = True
                    return
            self.wfile.write(chunk)

        def log_message(self, *_args):
            pass

    server = _QuietServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # addCleanup LIFO：shutdown → join → server_close（既有文件同款，
    # 避免关套接字时 serve_forever 仍在 select 上唤醒）
    return server, thread, f"http://127.0.0.1:{server.server_port}/media.bin"


def _cleanup_server(test_case, server, thread) -> None:
    test_case.addCleanup(server.server_close)
    test_case.addCleanup(thread.join, 5)
    test_case.addCleanup(server.shutdown)


class ParallelReassemblyTests(unittest.TestCase):
    """Range 能力源：分段→拼接逐字节等价→同一解码命令→同一门。"""

    def test_parallel_reassembly_is_byte_exact_and_passes_gate(self):
        payload = bytes(range(256)) * 4096  # 1 MiB 确定性容器字节
        server, thread, url = _serve(payload)
        _cleanup_server(self, server, thread)
        captured = {}

        def fake_run(command, timeout=None, capture_stdout=False):
            input_ref = command[command.index("-i") + 1]
            captured["input"] = Path(input_ref).read_bytes()
            Path(command[-1]).write_bytes(b"\x00" * _pcm_bytes(600.0))
            return 0, None, b""

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "media-full.f32le"
            with (
                patch.object(asr, "_MEDIA_PREFETCH_MIN_SEGMENT_BYTES", 16 * 1024),
                patch.object(asr, "_run_bounded_process", side_effect=fake_run),
            ):
                asr._prefetch_media_pcm(url, target, duration=600.0)
            self.assertEqual(captured["input"], payload, "拼接容器必须与原容器逐字节等价")
            self.assertTrue(target.exists())
            self.assertFalse(
                (Path(folder) / "prefetch-parts").exists(), "段目录用后必须清理"
            )

    def test_parallel_engages_only_above_threshold(self):
        # 低于 2×段门槛的体量不值得并行：逐字回落单流（24KB < 2×16KB）
        payload = b"x" * (24 * 1024)
        server, thread, url = _serve(payload)
        _cleanup_server(self, server, thread)
        captured = {}

        def fake_run(command, timeout=None, capture_stdout=False):
            captured["input"] = command[command.index("-i") + 1]
            Path(command[-1]).write_bytes(b"\x00" * _pcm_bytes(600.0))
            return 0, None, b""

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "media-full.f32le"
            with (
                patch.object(asr, "_MEDIA_PREFETCH_MIN_SEGMENT_BYTES", 16 * 1024),
                patch.object(asr, "_run_bounded_process", side_effect=fake_run),
            ):
                asr._prefetch_media_pcm(url, target, duration=600.0)
        self.assertEqual(captured["input"], url, "门槛下必须走原单流 URL 路径")


class FallbackPreservationTests(unittest.TestCase):
    """零行为回退钉：无 Range 能力 / 探测失败一律逐字回落原单流路径。"""

    def _run_fallback(self, url: str) -> str:
        captured = {}

        def fake_run(command, timeout=None, capture_stdout=False):
            captured["input"] = command[command.index("-i") + 1]
            Path(command[-1]).write_bytes(b"\x00" * _pcm_bytes(600.0))
            return 0, None, b""

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "media-full.f32le"
            with patch.object(asr, "_run_bounded_process", side_effect=fake_run):
                asr._prefetch_media_pcm(url, target, duration=600.0)
            self.assertTrue(target.exists())
        return captured["input"]

    def test_no_range_support_falls_back_to_single_stream_url(self):
        payload = b"x" * (512 * 1024)
        server, thread, url = _serve(payload, support_range=False)
        _cleanup_server(self, server, thread)
        self.assertEqual(self._run_fallback(url), url)

    def test_probe_connection_failure_falls_back_to_single_stream_url(self):
        # 起一个服务拿端口后立即关闭：端口必然拒绝连接（探测失败面）
        probe_server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        port = probe_server.server_port
        probe_server.server_close()
        self.assertEqual(
            self._run_fallback(f"http://127.0.0.1:{port}/media.bin"),
            f"http://127.0.0.1:{port}/media.bin",
        )


class SegmentLadderTests(unittest.TestCase):
    """段级截断防线：退避梯内恢复；梯尽 media_prefetch_incomplete + 清理。"""

    def test_segment_retry_recovers_after_one_short_read(self):
        payload = bytes(range(256)) * 4096
        segment_size = max(16 * 1024, -(-len(payload) // asr._MEDIA_PREFETCH_SEGMENTS))
        # 截断第二段（index=1）一次，重试恢复全长
        first_truncated = (segment_size, 2 * segment_size - 1)
        server, thread, url = _serve(
            payload, truncate_ranges={first_truncated}, truncate_once=True
        )
        _cleanup_server(self, server, thread)

        def fake_run(command, timeout=None, capture_stdout=False):
            Path(command[-1]).write_bytes(b"\x00" * _pcm_bytes(600.0))
            return 0, None, b""

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "media-full.f32le"
            with (
                patch.object(asr, "_MEDIA_PREFETCH_MIN_SEGMENT_BYTES", 16 * 1024),
                patch.object(asr, "_run_bounded_process", side_effect=fake_run),
                patch.object(asr.time, "sleep"),
            ):
                asr._prefetch_media_pcm(url, target, duration=600.0)
            self.assertTrue(target.exists(), "段级重试恢复后任务必须正常完成")

    def test_segment_short_read_ladder_exhaustion_fails_closed(self):
        payload = bytes(range(256)) * 4096
        segment_size = max(16 * 1024, -(-len(payload) // asr._MEDIA_PREFETCH_SEGMENTS))
        server, thread, url = _serve(
            payload, truncate_ranges={(0, segment_size - 1)}
        )
        _cleanup_server(self, server, thread)

        def fake_run(command, timeout=None, capture_stdout=False):
            raise AssertionError("段梯尽时绝不允许进入解码步")

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "media-full.f32le"
            with (
                patch.object(asr, "_MEDIA_PREFETCH_MIN_SEGMENT_BYTES", 16 * 1024),
                patch.object(asr, "_run_bounded_process", side_effect=fake_run),
                patch.object(asr.time, "sleep"),
            ):
                with self.assertRaises(asr.ASRError) as caught:
                    asr._prefetch_media_pcm(url, target, duration=600.0)
            self.assertEqual(str(caught.exception), INCOMPLETE)
            self.assertFalse(target.exists(), "截断产物必须删除")
            self.assertFalse(
                (Path(folder) / "prefetch-parts").exists(), "段目录必须清理"
            )
            self.assertFalse(
                (Path(folder) / "prefetch-container.bin").exists(),
                "拼接容器必须清理",
            )


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not available")
class ParallelRealFfmpegTests(unittest.TestCase):
    """真 ffmpeg 端到端：真实 mp4 经并行分段预取过完整性门；段截断段级拦下。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="n12-it-")
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

    def test_parallel_prefetch_decodes_full_stream_through_gate(self):
        server, thread, url = _serve(self.payload)
        _cleanup_server(self, server, thread)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "media-full.f32le"
            with patch.object(asr, "_MEDIA_PREFETCH_MIN_SEGMENT_BYTES", 16 * 1024):
                asr._prefetch_media_pcm(url, target, duration=12.0)
            size = target.stat().st_size
            self.assertGreaterEqual(size, _pcm_bytes(12.0))
            self.assertLess(size, _pcm_bytes(14.0), "远超期望时长的产出同样异常")

    def test_truncated_segment_fails_closed_before_decode(self):
        segment_size = max(16 * 1024, -(-len(self.payload) // asr._MEDIA_PREFETCH_SEGMENTS))
        server, thread, url = _serve(
            self.payload, truncate_ranges={(0, segment_size - 1)}
        )
        _cleanup_server(self, server, thread)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "media-full.f32le"
            with (
                patch.object(asr, "_MEDIA_PREFETCH_MIN_SEGMENT_BYTES", 16 * 1024),
                patch.object(asr.time, "sleep"),
            ):
                with self.assertRaises(asr.ASRError) as caught:
                    asr._prefetch_media_pcm(url, target, duration=12.0)
            self.assertEqual(str(caught.exception), INCOMPLETE)
            self.assertFalse(target.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
