from __future__ import annotations

import unittest

from courselens_worker.runner import SignedProgressPublisher


class SignedProgressPublisherTests(unittest.TestCase):
    def test_progress_is_bounded_and_does_not_invent_percentages(self):
        messages = []
        publisher = SignedProgressPublisher(messages.append, heartbeat_seconds=60)
        publisher.update("asr", status="waiting", force=True)
        publisher.update("asr", 1, 100)
        publisher.update("asr", 2, 100)
        publisher.update("asr", 3, 100)
        publisher.close()
        self.assertEqual(len(messages), 3)
        self.assertIsNone(messages[0]["completed"])
        self.assertIsNone(messages[0]["total"])
        self.assertEqual(messages[-1]["completed"], 3)
        self.assertEqual(messages[-1]["total"], 100)
        self.assertNotIn("percent", messages[-1])

    def test_failure_uses_closed_status_and_code(self):
        messages = []
        publisher = SignedProgressPublisher(messages.append, heartbeat_seconds=60)
        publisher.update("asr", status="failed", error_code="media_http_403", force=True)
        publisher.close()
        self.assertEqual(messages[-1]["status"], "failed")
        self.assertEqual(messages[-1]["error_code"], "media_http_403")


if __name__ == "__main__":
    unittest.main()
