from __future__ import annotations

import unittest
from unittest.mock import patch

from courselens_worker.platform_session import (
    PlatformSessionError,
    _validate_url,
    materialize_job_sources,
)


class _FakeConnector:
    login_values = None

    def login(self, account, password):
        type(self).login_values = (account, password)

    def media_source(self, course_id, sub_id):
        return {"url": "https://example.org/stream.mp4", "headers": {"Cookie": "sealed"}}

    def slide_sources(self, course_id, sub_id):
        return [{"page_num": 1, "created_sec": 2, "source": {"url": "https://example.org/1.png"}}]


class PlatformSessionTests(unittest.TestCase):
    def test_redirect_target_is_closed_to_expected_https_hosts(self):
        expected = "https://icourse.fudan.edu.cn/a"
        self.assertEqual(_validate_url(expected), expected)
        for value in (
            "http://icourse.fudan.edu.cn/a",
            "https://127.0.0.1/a",
            "https://example.org/a",
        ):
            with self.assertRaises(PlatformSessionError):
                _validate_url(value)

    def test_materialization_removes_credentials_and_preserves_slice(self):
        job = {
            "payload": {
                "media": {"start_seconds": 600, "duration_seconds": 300},
                "source_session": {
                    "provider": "runner-session-v1", "course_id": "1", "sub_id": "2",
                    "media": True, "slides": True,
                },
            },
            "secrets": {
                "source_credentials": {"account": "account", "password": "password"},
                "deepseek_api_key": "key",
            },
        }
        with patch("courselens_worker.platform_session.PlatformSession", _FakeConnector):
            result = materialize_job_sources(job)
        self.assertEqual(_FakeConnector.login_values, ("account", "password"))
        self.assertNotIn("source_session", result["payload"])
        self.assertNotIn("source_credentials", result["secrets"])
        self.assertEqual(result["payload"]["media"]["start_seconds"], 600)
        self.assertEqual(len(result["payload"]["slides"]), 1)


if __name__ == "__main__":
    unittest.main()
