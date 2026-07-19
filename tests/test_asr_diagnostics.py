from __future__ import annotations

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
        with patch.dict(sys.modules, {"numpy": Mock(), "sherpa_onnx": Mock()}):
            from courselens_worker import asr
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

    def test_slice_decode_discards_after_opening_the_single_input_stream(self):
        with patch.dict(sys.modules, {"numpy": Mock(), "sherpa_onnx": Mock()}):
            from courselens_worker import asr
        command = asr._ffmpeg_pipe_command(Path("slice.f32le"), offset=600, duration=300)
        input_index = command.index("-i")
        seek_index = command.index("-ss")
        self.assertLess(input_index, seek_index)
        self.assertEqual(command[input_index + 1], "pipe:0")


if __name__ == "__main__":
    unittest.main()
