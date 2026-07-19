from __future__ import annotations

import unittest

from courselens_worker.runner import safe_worker_error_detail


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


if __name__ == "__main__":
    unittest.main()
