from __future__ import annotations

import re
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

from courselens_worker.runner import safe_worker_error_detail
from courselens_worker.source import MediaResponseProfile


class ASRError(RuntimeError):
    pass


class ASRDiagnosticsTests(unittest.TestCase):
    def test_reason_codes_are_closed_set_and_do_not_echo_input(self):
        self.assertEqual(
            safe_worker_error_detail(ASRError("ffmpeg could not decode the authorized media stream")),
            "media_decode_failed",
        )
        secret = "https://example.invalid/video?token=secret"
        reason = safe_worker_error_detail(ASRError(secret))
        self.assertEqual(reason, "asr_error")
        self.assertNotIn("secret", reason)

    def test_ffmpeg_text_is_reduced_to_a_fixed_http_reason(self):
        error = ASRError("authorized media request returned HTTP 403")
        self.assertEqual(safe_worker_error_detail(error), "media_http_403")

    def test_media_response_failures_are_reduced_to_closed_reason_codes(self):
        from courselens_worker import asr  # 夜10-C：conftest 装桩，见 conftest.py
        cases = (
            (MediaResponseProfile("http_2xx", "content_html", "magic_html"), "media_content_html"),
            (MediaResponseProfile("http_2xx", "content_json", "magic_json"), "media_content_json"),
            (MediaResponseProfile("http_2xx", "content_video", "magic_unknown"), "media_magic_rejected"),
            (MediaResponseProfile("http_403", "content_html", "magic_html"), "media_http_403"),
        )
        for profile, expected in cases:
            with self.subTest(expected=expected):
                error = asr._media_response_error(profile)
                self.assertIsNotNone(error)
                self.assertEqual(safe_worker_error_detail(error), expected)

    def test_slice_decode_uses_loopback_range_input_and_fast_seek(self):
        from courselens_worker import asr  # 夜10-C：conftest 装桩，见 conftest.py
        command = asr._ffmpeg_proxy_command(
            Path("slice.f32le"),
            "http://127.0.0.1:4321/random",
            offset=600,
            duration=300,
        )
        input_index = command.index("-i")
        seek_index = command.index("-ss")
        self.assertLess(seek_index, input_index)
        self.assertEqual(command[input_index + 1], "http://127.0.0.1:4321/random")
        self.assertNotIn("pipe:0", command)


class MediaDecodeProbeTelemetryTests(unittest.TestCase):
    """DIAG34970-1 P1（2026-10-20）：解码失败路径的 ffprobe 流清单闭集遥测钉。

    34970 三讲残余子形（无音轨 S1/音轨不可解 S2/深损 S3）共享同一兜底
    media_decode_failed——一条流清单证据即闭合。钉四件事：①本地容器失败
    必发闭集遥测行且拒绝语义零变化；②URL 源绝不发起观测进程（零网络外呼）；
    ③探测自身失败按闭集形状如实上报；④闭集外类型归 unknown、名消毒。
    """

    DECODE_FAILURE = "ffmpeg could not decode the authorized media stream"

    def _run_decode_failure(self, source_ref: str, container_bytes: bytes | None):
        """Drive one decode failure with a stubbed process runner; return
        (raised error, telemetry lines, all commands)."""
        from courselens_worker import asr

        telemetry: list[str] = []
        commands: list[list[str]] = []

        def fake_run(command, timeout=None, capture_stdout=False):
            commands.append(list(command))
            if command[0] == "ffmpeg":
                return 1, None, b"stderr-without-classifier-features"
            # ffprobe：回给定流清单（由容器字节内容决定，这里固定两流）
            return 0, b"1,video,h264\n2,audio,aac\n", b""

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "media.f32le"
            if container_bytes is not None:
                container = Path(folder) / "container.bin"
                container.write_bytes(container_bytes)
                source = str(container)
            else:
                source = source_ref
            with (
                patch.object(asr, "_run_bounded_process", side_effect=fake_run),
                patch.object(asr, "_emit_telemetry", side_effect=telemetry.append),
            ):
                with self.assertRaises(asr.ASRError) as caught:
                    asr._decode_media_to_pcm(source, target, duration=600.0)
        return caught.exception, telemetry, commands

    def test_local_container_failure_emits_closed_set_stream_inventory(self):
        error, telemetry, commands = self._run_decode_failure("", b"container-bytes")
        self.assertEqual(str(error), self.DECODE_FAILURE, "拒绝语义零变化")
        self.assertEqual(safe_worker_error_detail(error), "media_decode_failed")
        self.assertEqual(len(telemetry), 1, "恰一条观测行")
        self.assertEqual(
            telemetry[0],
            "stage=media-decode-probe streams=2 codecs=video:h264,audio:aac",
        )
        probe_commands = [item for item in commands if item[0] == "ffprobe"]
        self.assertEqual(len(probe_commands), 1, "观测进程恰一次且有界")
        probe_command = probe_commands[0]
        self.assertIn("-show_entries", probe_command)
        self.assertIn("stream=index,codec_type,codec_name", probe_command)
        self.assertIn("-of", probe_command)
        self.assertIn("csv=p=0", probe_command)
        self.assertEqual(
            [item for item in probe_command if item.startswith("http")], [],
            "观测命令零 URL（闭集安全面）",
        )
        self.assertTrue(all("secret" not in item and "token" not in item for item in probe_command))

    def test_url_source_never_spawns_probe_process(self):
        error, telemetry, commands = self._run_decode_failure(
            "http://127.0.0.1:1/session-media", None
        )
        self.assertEqual(str(error), self.DECODE_FAILURE)
        self.assertEqual([item for item in commands if item[0] == "ffprobe"], [],
                         "URL 源零观测进程（观测绝不发起网络外呼）")
        self.assertEqual(telemetry, [], "跳过探测时零观测行")

    def test_probe_failure_reports_closed_set_shape(self):
        from courselens_worker import asr

        telemetry: list[str] = []

        def fake_run(command, timeout=None, capture_stdout=False):
            if command[0] == "ffprobe":
                raise OSError("ffprobe missing")
            return 1, None, b""

        with tempfile.TemporaryDirectory() as folder:
            container = Path(folder) / "container.bin"
            container.write_bytes(b"container-bytes")
            with (
                patch.object(asr, "_run_bounded_process", side_effect=fake_run),
                patch.object(asr, "_emit_telemetry", side_effect=telemetry.append),
            ):
                with self.assertRaises(asr.ASRError):
                    asr._decode_media_to_pcm(str(container), Path(folder) / "m.f32le", duration=600.0)
        self.assertEqual(telemetry, ["stage=media-decode-probe probe=failed"])

    def test_zero_stream_inventory_is_the_s1_evidence_shape(self):
        from courselens_worker import asr

        telemetry: list[str] = []

        def fake_run(command, timeout=None, capture_stdout=False):
            if command[0] == "ffprobe":
                return 0, b"", b""
            return 1, None, b""

        with tempfile.TemporaryDirectory() as folder:
            container = Path(folder) / "container.bin"
            container.write_bytes(b"container-bytes")
            with (
                patch.object(asr, "_run_bounded_process", side_effect=fake_run),
                patch.object(asr, "_emit_telemetry", side_effect=telemetry.append),
            ):
                with self.assertRaises(asr.ASRError):
                    asr._decode_media_to_pcm(str(container), Path(folder) / "m.f32le", duration=600.0)
        self.assertEqual(telemetry, ["stage=media-decode-probe streams=0 codecs=none"])

    def test_unknown_stream_type_and_names_stay_in_closed_set(self):
        from courselens_worker import asr

        telemetry: list[str] = []

        def fake_run(command, timeout=None, capture_stdout=False):
            if command[0] == "ffprobe":
                return 0, b"1,weird-type,<odd name>\n", b""
            return 1, None, b""

        with tempfile.TemporaryDirectory() as folder:
            container = Path(folder) / "container.bin"
            container.write_bytes(b"container-bytes")
            with (
                patch.object(asr, "_run_bounded_process", side_effect=fake_run),
                patch.object(asr, "_emit_telemetry", side_effect=telemetry.append),
            ):
                with self.assertRaises(asr.ASRError):
                    asr._decode_media_to_pcm(str(container), Path(folder) / "m.f32le", duration=600.0)
        self.assertEqual(telemetry, ["stage=media-decode-probe streams=1 codecs=unknown:oddname"])


class RunnerGateEmittershipTests(unittest.TestCase):
    """worker 门新码钉（DIAG34970-1 P2）：媒体码发射权边界。

    media_decode_failed 是 worker 签名发射的单讲素材死因（门表映射必须
    在）；media_unprocessable 是客户端侧课程级派生码（worker 门绝不认领
    ——发射权单向，客户端组合证据，worker 只报单讲事实）。
    """

    def test_worker_gate_maps_single_lecture_decode_code(self):
        self.assertEqual(
            safe_worker_error_detail(ASRError("ffmpeg could not decode the authorized media stream")),
            "media_decode_failed",
        )

    def test_course_level_media_code_is_never_worker_emitted(self):
        runner_source = (
            Path(__file__).resolve().parents[1] / "courselens_worker" / "runner.py"
        ).read_text(encoding="utf-8")
        anchor = "_ASR_ERROR_CODES = {"
        block = runner_source[runner_source.index(anchor) + len(anchor):]
        block = block[:block.index("\n}")]
        # 与 tests/test_worker_code_parity 同一抽取口径（NL 键→码值对，取值集）。
        codes = {
            code for _, code in re.findall(r'"([^"]+)":\s*"([a-z0-9_]+)"', block)
        }
        self.assertIn("media_decode_failed", codes, "单讲素材死因必须是 worker 门发射码")
        self.assertNotIn(
            "media_unprocessable", codes,
            "课程级 media_unprocessable 是客户端派生码，worker 门不得认领",
        )


if __name__ == "__main__":
    unittest.main()
