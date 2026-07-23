from __future__ import annotations

import unittest
import sys
from unittest.mock import Mock, patch

with patch.dict(sys.modules, {"numpy": Mock(), "sherpa_onnx": Mock()}):
    from courselens_worker import asr


class ASRProxyLifecycleTests(unittest.TestCase):
    def test_all_pcm_chunks_share_one_pinned_media_session(self):
        pool = Mock()
        pool.transcribe_pcm.side_effect = lambda _path, _backend, *, offset_seconds: [{
            "start_ms": int(offset_seconds * 1000),
            "end_ms": int(offset_seconds * 1000) + 1000,
            "text": "test",
        }]
        progress = Mock()

        def create_pcm(_url, target, *, offset, duration):
            self.assertGreaterEqual(offset, 0)
            self.assertGreater(duration, 0)
            target.write_bytes(b"pcm")

        with (
            patch.object(asr, "RecognizerPool", return_value=pool),
            patch.object(asr, "pinned_media_proxy") as media_proxy,
            patch.object(asr, "_decode_chunk_from_url", side_effect=create_pcm) as decode,
        ):
            media_proxy.return_value.__enter__.return_value.url = "http://127.0.0.1/session"
            result = asr.transcribe(
                {
                    "payload": {
                        "mode": "fast",
                        "media": {
                            "url": "https://media.example.com/lecture.mp4",
                            "duration_seconds": 1250,
                        },
                    },
                },
                sensevoice_dir=Mock(),
                firered_dir=Mock(),
                proofread=None,
                progress=progress,
            )

        media_proxy.assert_called_once()
        self.assertEqual(decode.call_count, 3)
        self.assertEqual(
            [item.args[0] for item in decode.call_args_list],
            ["http://127.0.0.1/session"] * 3,
        )
        self.assertEqual(
            [item.kwargs["offset"] for item in decode.call_args_list],
            [0.0, 600.0, 1200.0],
        )
        self.assertEqual([item.args for item in progress.call_args_list], [
            ("asr", 1, 3),
            ("asr", 2, 3),
            ("asr", 3, 3),
        ])
        self.assertEqual(result["metrics"]["chunks"], 3)


if __name__ == "__main__":
    unittest.main()
