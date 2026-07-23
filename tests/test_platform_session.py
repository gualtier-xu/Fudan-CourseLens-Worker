from __future__ import annotations

import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

import requests

from curl_cffi.requests.exceptions import RequestException as CurlRequestException
from courselens_worker.platform_session import (
    PlatformSession,
    PlatformSessionError,
    _validate_url,
    materialize_job_sources,
)
from courselens_worker.runner import safe_worker_error_detail
from courselens_worker.source import ResolvedSource, SourceSecurityError


class _FakeConnector:
    login_values = None

    class _Session:
        def close(self):
            return None

    def __init__(self):
        self.session = self._Session()

    def close(self):
        self.session.close()

    def login(self, account, password):
        type(self).login_values = (account, password)

    def media_source(self, course_id, sub_id):
        return {"url": "https://example.org/stream.mp4", "headers": {"Cookie": "sealed"}}

    def slide_sources(self, course_id, sub_id):
        return [{"page_num": 1, "created_sec": 2, "source": {"url": "https://example.org/1.png"}}]


class PlatformSessionTests(unittest.TestCase):
    def test_login_prefers_direct_course_session_after_webvpn(self):
        connector = object.__new__(PlatformSession)
        connector._login_webvpn = Mock()
        connector._login_course_direct = Mock()
        connector._login_course = Mock()

        connector.login("account", "password")

        connector._login_webvpn.assert_called_once_with("account", "password")
        connector._login_course_direct.assert_called_once_with("account", "password")
        connector._login_course.assert_not_called()

    def test_login_falls_back_only_for_direct_session_failures(self):
        connector = object.__new__(PlatformSession)
        connector._login_webvpn = Mock()
        connector._login_course_direct = Mock(side_effect=PlatformSessionError(
            "platform_connection_failed",
            connection_stage="course_ticket_follow_direct",
        ))
        connector._login_course = Mock()
        connector._course_direct = True
        connector._course_bearer = "temporary"

        connector.login("account", "password")

        connector._login_course.assert_called_once_with("account", "password")
        self.assertFalse(connector._course_direct)
        self.assertEqual(connector._course_bearer, "")

        connector._login_course_direct.side_effect = PlatformSessionError(
            "platform_auth_failed"
        )
        connector._login_course.reset_mock()
        with self.assertRaisesRegex(PlatformSessionError, "platform_auth_failed"):
            connector.login("account", "password")
        connector._login_course.assert_not_called()

    def test_course_requests_use_the_isolated_direct_session(self):
        connector = object.__new__(PlatformSession)
        connector._course_direct = True
        connector._course_bearer = "bounded-test-token"
        response = Mock(status_code=200)
        response.json.return_value = {"code": 0, "data": {}}
        connector._direct_once = Mock(return_value=response)
        connector._once = Mock()

        result = connector._course_json("/userapi/v1/infosimple", params={})

        self.assertEqual(result["code"], 0)
        connector._once.assert_not_called()
        request = connector._direct_once.call_args
        self.assertEqual(request.args[:2], ("GET", "https://icourse.fudan.edu.cn/userapi/v1/infosimple"))
        self.assertEqual(request.kwargs["headers"], {"Authorization": "Bearer bounded-test-token"})

    def test_authorized_catalog_requires_successful_detail_for_each_course(self):
        connector = object.__new__(PlatformSession)
        connector._userinfo = None
        calls = []

        def course_json(path, *, params):
            calls.append((path, dict(params)))
            if path.endswith("get-course-list"):
                if params["term"] == "24":
                    return {"code": 0, "data": {"total": 2, "list": [
                        {"id": "1", "title": "A", "term_name": "2026", "kkxy_name": "Dept"},
                        {"id": "2", "title": "B", "term_name": "2026", "kkxy_name": "Dept"},
                    ]}}
                return {"code": 0, "data": {"total": 0, "list": []}}
            if params["course_id"] == "2":
                raise PlatformSessionError("platform_course_request_failed")
            return {"code": 0, "data": {"title": "A", "realname": "Teacher", "sub_list": {}}}

        connector._course_json = course_json
        courses = connector.discover_authorized_courses()
        self.assertEqual([item["course_id"] for item in courses], ["1"])
        self.assertEqual(courses[0]["authorization_state"], "verified")
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

    def _media_connector(self):
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
        connector._source_headers = lambda: {
            "Cookie": "sealed", "User-Agent": "CourseLens", "Accept": "*/*"
        }
        return connector

    def test_media_source_prefers_verified_direct_route_without_cookie(self):
        connector = self._media_connector()
        resolved = ResolvedSource(
            "https://media.example.edu/lecture.mp4?clientUUID=test&t=test",
            {"User-Agent": "CourseLens", "Accept": "*/*"},
            "93.184.216.34",
        )
        with patch("courselens_worker.source.resolve_source_address", return_value=resolved) as resolve:
            source = connector.media_source("1", "2")
        self.assertEqual(source["url"], resolved.url)
        self.assertEqual(source["resolved_public_ip"], "93.184.216.34")
        self.assertNotIn("Cookie", source["headers"])
        self.assertNotIn("Cookie", resolve.call_args.args[1])
        self.assertTrue(callable(source["_refresh_source"]))

    def test_media_source_falls_back_to_runner_webvpn_session(self):
        connector = self._media_connector()
        with patch(
            "courselens_worker.source.resolve_source_address",
            side_effect=SourceSecurityError("source request failed: OSError"),
        ):
            source = connector.media_source("1", "2")
        parsed = urlsplit(source["url"])
        self.assertEqual(parsed.hostname, "webvpn.fudan.edu.cn")
        self.assertIn("clientUUID=test", parsed.query)
        self.assertEqual(source["headers"]["Cookie"], "sealed")

    def test_media_source_refreshes_the_signed_url_for_each_proxy_request(self):
        connector = self._media_connector()
        resolved = ResolvedSource(
            "https://media.example.edu/lecture.mp4?clientUUID=first&t=first",
            {"User-Agent": "CourseLens"},
            "93.184.216.34",
        )
        connector._sign = Mock(side_effect=[
            resolved.url,
            "https://media.example.edu/lecture.mp4?clientUUID=second&t=second",
        ])
        with patch(
            "courselens_worker.source.resolve_source_address",
            side_effect=lambda url, headers: ResolvedSource(url, headers, "93.184.216.34"),
        ):
            source = connector.media_source("1", "2")
            refreshed = source["_refresh_source"]()
        self.assertEqual(connector._sign.call_count, 2)
        self.assertIn("second", refreshed["url"])

    def test_media_source_uses_strictly_increasing_signing_seconds(self):
        connector = self._media_connector()
        connector._sign = Mock(side_effect=lambda value, now: f"{value}?t={now}")
        with (
            patch("courselens_worker.platform_session.time.time", side_effect=[100, 100, 100, 101]),
            patch("courselens_worker.platform_session.time.sleep") as sleep,
            patch(
                "courselens_worker.source.resolve_source_address",
                side_effect=lambda url, headers: ResolvedSource(url, headers, "93.184.216.34"),
            ),
        ):
            source = connector.media_source("1", "2")
            source["_refresh_source"]()
        signed_seconds = [item.args[1] for item in connector._sign.call_args_list]
        self.assertEqual(signed_seconds, sorted(set(signed_seconds)))
        sleep.assert_called_once_with(1)

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

    def test_connection_failure_retains_only_a_closed_set_stage(self):
        connector = PlatformSession()
        connector.session.request = Mock(
            side_effect=requests.ConnectionError("secret URL must not escape")
        )
        with self.assertRaises(PlatformSessionError) as captured:
            connector._once(
                "GET",
                "https://webvpn.fudan.edu.cn/",
                connection_stage="webvpn_context",
            )
        self.assertEqual(str(captured.exception), "platform_connection_failed")
        self.assertEqual(captured.exception.connection_stage, "webvpn_context")
        self.assertEqual(
            safe_worker_error_detail(captured.exception),
            "platform_connection_failed_webvpn_context",
        )

        with self.assertRaises(PlatformSessionError) as captured:
            connector._once(
                "GET",
                "https://webvpn.fudan.edu.cn/",
                connection_stage="secret-host-and-path",
            )
        self.assertEqual(captured.exception.connection_stage, "")
        self.assertEqual(
            safe_worker_error_detail(captured.exception),
            "platform_connection_failed",
        )

    def test_curl_transport_failure_uses_the_same_closed_stage(self):
        connector = PlatformSession()
        connector.session.request = Mock(
            side_effect=CurlRequestException("secret URL must not escape")
        )
        with self.assertRaises(PlatformSessionError) as captured:
            connector._once(
                "GET",
                "https://webvpn.fudan.edu.cn/",
                connection_stage="webvpn_ticket_follow",
            )
        self.assertEqual(str(captured.exception), "platform_connection_failed")
        self.assertEqual(
            safe_worker_error_detail(captured.exception),
            "platform_connection_failed_webvpn_ticket_follow",
        )


if __name__ == "__main__":
    unittest.main()
