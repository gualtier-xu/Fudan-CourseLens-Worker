from __future__ import annotations

import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

import requests

from curl_cffi import requests as curl_requests
from curl_cffi.requests.exceptions import RequestException as CurlRequestException
from courselens_worker.platform_session import (
    PlatformSession,
    PlatformSessionError,
    _validate_url,
    cloud_session_from_environment,
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
    def test_source_headers_support_curl_cookie_mapping(self):
        connector = object.__new__(PlatformSession)
        connector.session = curl_requests.Session(impersonate="chrome")
        try:
            connector.session.cookies.set("session", "sealed")
            headers = connector._source_headers()
        finally:
            connector.session.close()

        self.assertEqual(headers["Cookie"], "session=sealed")

    def test_source_headers_reject_control_characters(self):
        connector = object.__new__(PlatformSession)
        connector.session = Mock()
        connector.session.cookies.items.return_value = [("session", "value\r\ninjected")]

        with self.assertRaisesRegex(PlatformSessionError, "platform_session_rejected") as captured:
            connector._source_headers()
        self.assertEqual(captured.exception.connection_stage, "course_request")
        self.assertEqual(
            safe_worker_error_detail(captured.exception),
            "platform_session_rejected_course_request",
        )

    def test_session_rejection_retains_only_a_closed_set_stage(self):
        error = PlatformSessionError(
            "platform_session_rejected", connection_stage="course_verify_direct"
        )
        self.assertEqual(
            safe_worker_error_detail(error),
            "platform_session_rejected_course_verify_direct",
        )
        redacted = PlatformSessionError(
            "platform_session_rejected", connection_stage="secret-host-and-path"
        )
        self.assertEqual(safe_worker_error_detail(redacted), "platform_session_rejected")

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

    def test_webvpn_personal_catalog_requires_bearer_and_rejects_global_fallback(self):
        connector = object.__new__(PlatformSession)
        connector._course_direct = False
        connector.session = Mock()
        connector._extract_course_bearer = Mock(return_value="bounded-test-token")
        response = Mock(status_code=200)
        response.json.return_value = {"code": 0, "list": []}
        connector._once = Mock(return_value=response)

        connector._course_json(
            "/courseapi/v2/course-live/get-my-course-month",
            params={"month": "2026-07"},
            authorization_required=True,
            timeout=(5, 15),
        )

        request = connector._once.call_args
        self.assertIn("get-my-course-month", request.args[1])
        self.assertEqual(request.kwargs["headers"], {"Authorization": "Bearer bounded-test-token"})
        self.assertEqual(request.kwargs["timeout"], (5, 15))

    def test_authorized_catalog_uses_identity_scoped_schedule_and_verifies_details(self):
        connector = object.__new__(PlatformSession)
        connector._userinfo = None
        calls = []

        def course_json(path, *, params, **_kwargs):
            calls.append((path, dict(params)))
            if path.endswith("infosimple"):
                return {"code": 0, "data": {"id": "u", "account": "student", "tenant_id": "222"}}
            if path.endswith("get-my-course-month"):
                return {"code": 0, "list": [{"course": [
                    {"id": "1", "title": "A", "term_name": "2026", "kkxy_name": "Dept"},
                    {"id": "2", "title": "B", "term_name": "2026", "kkxy_name": "Dept"},
                ]}]}
            if params["course_id"] == "2":
                raise PlatformSessionError("platform_course_request_failed")
            return {"code": 0, "data": {"title": "A", "realname": "Teacher", "sub_list": {}}}

        connector._course_json = course_json
        courses = connector.discover_authorized_courses()
        self.assertEqual([item["course_id"] for item in courses], ["1"])
        self.assertEqual(courses[0]["authorization_state"], "verified")
        self.assertTrue(any(path.endswith("get-my-course-month") for path, _ in calls))
        self.assertFalse(any(path.endswith("get-course-list") for path, _ in calls))

    def test_personal_catalog_deadline_fails_closed_before_any_request(self):
        connector = object.__new__(PlatformSession)
        connector._userinfo = {"id": "u", "account": "student", "tenant_id": "222"}
        connector._course_json = Mock()
        with self.assertRaisesRegex(PlatformSessionError, "platform_course_request_failed"):
            connector._user_courses(deadline=0.1)
        connector._course_json.assert_not_called()
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

    def test_materialization_retains_only_refreshable_session_until_cleanup(self):
        closed = []

        class RefreshableConnector(_FakeConnector):
            def close(self):
                closed.append(True)

            def media_source(self, course_id, sub_id):
                return {
                    "url": "https://example.org/stream.mp4",
                    "_refresh_source": lambda: {
                        "url": "https://example.org/stream.mp4"
                    },
                }

        job = {
            "payload": {
                "media": {"duration_seconds": 60},
                "source_session": {
                    "provider": "runner-session-v1", "course_id": "1",
                    "sub_id": "2", "media": True, "slides": False,
                },
            },
            "secrets": {
                "source_credentials": {"account": "account", "password": "password"}
            },
        }
        with patch("courselens_worker.platform_session.PlatformSession", RefreshableConnector):
            result = materialize_job_sources(job)

        self.assertEqual(closed, [])
        closer = result["payload"].pop("_close_source_session")
        self.assertTrue(callable(closer))
        closer()
        self.assertEqual(closed, [True])
        self.assertNotIn("source_credentials", result["secrets"])

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

    def test_media_refresh_fetches_a_new_platform_base_url(self):
        connector = self._media_connector()
        connector._course_json = Mock(side_effect=[
            {"data": {"now": 100, "video_list": {"main": {
                "preview_url": "https://media.example.edu/lecture.mp4?base=first"
            }}}},
            {"data": {"now": 101, "video_list": {"main": {
                "preview_url": "https://media.example.edu/lecture.mp4?base=second"
            }}}},
        ])
        connector._sign = Mock(side_effect=lambda value, now: f"{value}&t={now}")
        with (
            patch("courselens_worker.platform_session.time.time", side_effect=[100, 100, 101, 101]),
            patch(
                "courselens_worker.source.resolve_source_address",
                side_effect=lambda url, headers: ResolvedSource(
                    url, headers, "93.184.216.34"
                ),
            ),
        ):
            source = connector.media_source("1", "2")
            refreshed = source["_refresh_source"]()

        self.assertEqual(connector._course_json.call_count, 2)
        self.assertIn("base=first", connector._sign.call_args_list[0].args[0])
        self.assertIn("base=second", connector._sign.call_args_list[1].args[0])
        self.assertIn("base=second", refreshed["url"])

    def test_media_refresh_keeps_initial_clock_offset_when_platform_now_is_stale(self):
        connector = self._media_connector()
        connector._course_json = Mock(side_effect=[
            {"data": {"now": 100, "video_list": {"main": {
                "preview_url": "https://media.example.edu/lecture.mp4?base=first"
            }}}},
            {"data": {"now": 100, "video_list": {"main": {
                "preview_url": "https://media.example.edu/lecture.mp4?base=second"
            }}}},
        ])
        connector._sign = Mock(side_effect=lambda value, now: f"{value}&t={now}")
        with (
            patch("courselens_worker.platform_session.time.time", side_effect=[100, 100, 165]),
            patch(
                "courselens_worker.source.resolve_source_address",
                side_effect=lambda url, headers: ResolvedSource(
                    url, headers, "93.184.216.34"
                ),
            ),
        ):
            source = connector.media_source("1", "2")
            source["_refresh_source"]()

        signed_seconds = [item.args[1] for item in connector._sign.call_args_list]
        self.assertEqual(signed_seconds, [100, 165])

    def test_media_source_uses_strictly_increasing_signing_seconds(self):
        connector = self._media_connector()
        connector._sign = Mock(side_effect=lambda value, now: f"{value}?t={now}")
        with (
            patch(
                "courselens_worker.platform_session.time.time",
                side_effect=[100, 100, 100, 101],
            ),
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

    def test_cloud_login_rebuilds_transient_rejected_sessions(self):
        class FlakyConnector(_FakeConnector):
            attempts = 0

            def login(self, account, password):
                type(self).attempts += 1
                if type(self).attempts < 3:
                    raise PlatformSessionError("platform_session_rejected")
                super().login(account, password)

        with (
            patch.dict(
                "os.environ",
                {
                    "COURSELENS_CLOUD_STUDENT_ID": "student",
                    "COURSELENS_CLOUD_PASSWORD": "password",
                },
                clear=True,
            ),
            patch("courselens_worker.platform_session.PlatformSession", FlakyConnector),
            patch("courselens_worker.platform_session.time.sleep") as sleep,
        ):
            connector = cloud_session_from_environment()
        connector.close()
        self.assertEqual(FlakyConnector.attempts, 3)
        self.assertEqual(sleep.call_count, 2)

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
