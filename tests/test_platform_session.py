from __future__ import annotations

import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from courselens_worker.platform_session import (
    PlatformSession,
    PlatformSessionError,
    _validate_url,
    materialize_job_sources,
)


class _FakeConnector:
    login_values = None

    class _Session:
        def close(self):
            return None

    def __init__(self):
        self.session = self._Session()

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

    def test_media_source_stays_inside_runner_webvpn_session(self):
        connector = object.__new__(PlatformSession)
        connector._course_json = lambda *_args, **_kwargs: {
            "data": {
                "now": 1,
                "video_list": {
                    "main": {"preview_url": "https://media.example.edu/lecture.mp4"}
                },
            }
        }
        connector._sign = lambda value, _now: value + "?clientUUID=test&t=test"
        connector._source_headers = lambda: {"Cookie": "sealed"}
        source = connector.media_source("1", "2")
        parsed = urlsplit(source["url"])
        self.assertEqual(parsed.hostname, "webvpn.fudan.edu.cn")
        self.assertIn("clientUUID=test", parsed.query)

    def test_connection_failure_retries_without_retrying_authentication_errors(self):
        class FlakyConnector(_FakeConnector):
            attempts = 0

            def login(self, account, password):
                type(self).attempts += 1
                if type(self).attempts < 3:
                    raise PlatformSessionError("platform_connection_failed")
                super().login(account, password)

        job = {
            "payload": {
                "media": {},
                "source_session": {
                    "provider": "runner-session-v1", "course_id": "1", "sub_id": "2",
                    "media": True, "slides": False,
                },
            },
            "secrets": {"source_credentials": {"account": "a", "password": "p"}},
        }
        with patch("courselens_worker.platform_session.PlatformSession", FlakyConnector), patch(
            "courselens_worker.platform_session.time.sleep"
        ) as sleep:
            materialize_job_sources(job)
        self.assertEqual(FlakyConnector.attempts, 3)
        self.assertEqual(sleep.call_count, 2)

        class RejectedConnector(_FakeConnector):
            attempts = 0

            def login(self, account, password):
                type(self).attempts += 1
                raise PlatformSessionError("platform_auth_rejected")

        rejected = {
            "payload": {
                "media": {},
                "source_session": {
                    "provider": "runner-session-v1", "course_id": "1", "sub_id": "2",
                    "media": True, "slides": False,
                },
            },
            "secrets": {"source_credentials": {"account": "a", "password": "p"}},
        }
        with patch("courselens_worker.platform_session.PlatformSession", RejectedConnector):
            with self.assertRaises(PlatformSessionError):
                materialize_job_sources(rejected)
        self.assertEqual(RejectedConnector.attempts, 1)


if __name__ == "__main__":
    unittest.main()
