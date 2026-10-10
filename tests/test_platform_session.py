from __future__ import annotations

import json
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import quote, urlsplit
from zoneinfo import ZoneInfo

import requests

from curl_cffi import requests as curl_requests
from curl_cffi.requests.exceptions import RequestException as CurlRequestException
from courselens_worker.platform_session import (
    GATE_TZ,
    PlatformSession,
    PlatformSessionError,
    _parse_end_evidence,
    _validate_url,
    cloud_session_from_environment,
    materialize_job_sources,
)
from courselens_worker.runner import safe_worker_error_detail
from courselens_worker.source import ResolvedSource, SourceSecurityError


class _FakeConnector:
    login_values = None
    transport_values = []

    class _Session:
        def close(self):
            return None

    def __init__(self, *, transport="curl"):
        self.session = self._Session()
        type(self).transport_values.append(transport)

    def close(self):
        self.session.close()

    def login(self, account, password):
        type(self).login_values = (account, password)

    def media_source(self, course_id, sub_id, *, relogin=None):
        return {"url": "https://example.org/stream.mp4", "headers": {"Cookie": "sealed"}}

    def slide_sources(self, course_id, sub_id):
        return [{"page_num": 1, "created_sec": 2, "source": {"url": "https://example.org/1.png"}}]


class PlatformSessionTests(unittest.TestCase):
    def test_requests_transport_disables_environment_proxies(self):
        connector = PlatformSession(transport="requests")
        try:
            self.assertIsInstance(connector.session, requests.Session)
            self.assertIsInstance(connector.course_session, requests.Session)
            self.assertFalse(connector.session.trust_env)
            self.assertFalse(connector.course_session.trust_env)
        finally:
            connector.close()
        with self.assertRaisesRegex(ValueError, "unsupported platform transport"):
            PlatformSession(transport="unknown")

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

    def test_login_prefers_verified_direct_course_without_webvpn(self):
        connector = object.__new__(PlatformSession)
        connector._login_webvpn = Mock()
        connector._login_course_direct = Mock()
        connector._login_course = Mock()

        connector.login("account", "password")

        connector._login_course_direct.assert_called_once_with("account", "password")
        connector._login_webvpn.assert_not_called()
        connector._login_course.assert_not_called()

    def test_login_falls_back_to_webvpn_only_for_direct_session_failures(self):
        connector = object.__new__(PlatformSession)
        connector._login_webvpn = Mock()
        connector._login_course_direct = Mock(side_effect=PlatformSessionError(
            "platform_connection_failed",
            connection_stage="course_ticket_follow_direct",
        ))
        connector._login_course = Mock()
        connector._course_direct = True
        connector._course_bearer = "temporary"
        connector._webvpn_ready = False

        connector.login("account", "password")

        connector._login_webvpn.assert_called_once_with("account", "password")
        connector._login_course.assert_called_once_with("account", "password")
        self.assertFalse(connector._course_direct)
        self.assertTrue(connector._webvpn_ready)
        self.assertEqual(connector._course_bearer, "")

        connector._login_course_direct.side_effect = PlatformSessionError(
            "platform_auth_failed"
        )
        connector._login_course.reset_mock()
        connector._login_webvpn.reset_mock()
        with self.assertRaisesRegex(PlatformSessionError, "platform_auth_failed"):
            connector.login("account", "password")
        connector._login_webvpn.assert_not_called()
        connector._login_course.assert_not_called()

    def test_webvpn_full_login_retries_three_times_with_backoff_and_fresh_sessions(self):
        connector = PlatformSession(transport="requests")
        try:
            webvpn_calls: list[int] = []

            def flaky_webvpn(account, password):
                webvpn_calls.append(id(connector.session))
                if len(webvpn_calls) < 3:
                    raise PlatformSessionError(
                        "platform_connection_failed",
                        connection_stage="webvpn_ticket_follow",
                    )

            connector._login_webvpn = Mock(side_effect=flaky_webvpn)
            connector._login_course = Mock()
            with patch("courselens_worker.platform_session.time.sleep") as sleep:
                connector._login_webvpn_full("account", "password")

            self.assertEqual(len(webvpn_calls), 3)
            self.assertEqual(len(set(webvpn_calls)), 3)
            connector._login_course.assert_called_once_with("account", "password")
            self.assertTrue(connector._webvpn_ready)
            self.assertEqual(
                [call.args for call in sleep.call_args_list], [(2.0,), (4.0,)]
            )
        finally:
            connector.close()

    def test_webvpn_full_login_gives_up_after_three_attempts_without_half_state(self):
        connector = PlatformSession(transport="requests")
        try:
            connector._login_webvpn = Mock(
                side_effect=PlatformSessionError("platform_connection_failed")
            )
            connector._login_course = Mock()
            with patch("courselens_worker.platform_session.time.sleep") as sleep:
                with self.assertRaisesRegex(
                    PlatformSessionError, "platform_connection_failed"
                ):
                    connector._login_webvpn_full("account", "password")

            self.assertEqual(connector._login_webvpn.call_count, 3)
            self.assertEqual(
                [call.args for call in sleep.call_args_list], [(2.0,), (4.0,)]
            )
            self.assertFalse(connector._webvpn_ready)
            self.assertEqual(connector._course_bearer, "")
            self.assertIsNone(connector._userinfo)
        finally:
            connector.close()

    def test_webvpn_full_login_never_retries_non_retryable_codes(self):
        connector = PlatformSession(transport="requests")
        try:
            connector._login_webvpn = Mock(
                side_effect=PlatformSessionError("platform_auth_failed")
            )
            connector._login_course = Mock()
            with patch("courselens_worker.platform_session.time.sleep") as sleep:
                with self.assertRaisesRegex(PlatformSessionError, "platform_auth_failed"):
                    connector._login_webvpn_full("account", "password")

            self.assertEqual(connector._login_webvpn.call_count, 1)
            sleep.assert_not_called()
        finally:
            connector.close()

    def test_cloud_login_allows_webvpn_fallback_only_on_final_attempt(self):
        class DirectOnlyConnector(_FakeConnector):
            attempts = 0
            fallback_values = []
            transport_values = []

            def login(self, account, password, *, allow_webvpn_fallback=True):
                type(self).attempts += 1
                type(self).fallback_values.append(allow_webvpn_fallback)
                if type(self).attempts < 3:
                    raise PlatformSessionError(
                        "platform_session_rejected",
                        connection_stage="course_verify_direct",
                    )
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
            patch("courselens_worker.platform_session.PlatformSession", DirectOnlyConnector),
            patch("courselens_worker.platform_session.time.sleep") as sleep,
        ):
            connector = cloud_session_from_environment()
        connector.close()
        self.assertEqual(DirectOnlyConnector.attempts, 3)
        self.assertEqual(DirectOnlyConnector.fallback_values, [False, False, True])
        self.assertEqual(
            DirectOnlyConnector.transport_values, ["curl", "requests", "requests"]
        )
        self.assertEqual(sleep.call_count, 2)

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

    def test_webvpn_personal_catalog_attaches_bearer_when_carrier_present_and_rejects_global_fallback(self):
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

    def test_webvpn_authorized_request_omits_bearer_when_token_carrier_absent(self):
        # PLATFORM-REWORK 形态钉（2026-10-07 run 37557294406 闭集定谳）：
        # wengine 网关把 origin cookie 服务端映射，webvpn 会话罐内恒无
        # _token serialized 载体；authorization_required 请求此时必须省略
        # Authorization 头照发（走登录时 _verify_course 已上游验证的会话
        # cookie），绝不硬抛 platform_session_rejected 的
        # course_ticket_follow_direct 塌缩码——旧懒提取硬失败曾把门腿
        # （rows=-1 fail-open）与发现腿双双打死，verify 链因从不发
        # authorization_required 请求而恒 PASS 的形态差即源于此。
        connector = object.__new__(PlatformSession)
        connector._course_direct = False
        connector.session = SimpleNamespace(cookies={"wengine_vpn_ticketwebvpn": "sealed"})
        response = Mock(status_code=200)
        response.json.return_value = {"code": 0, "list": []}
        connector._once = Mock(return_value=response)

        result = connector._course_json(
            "/courseapi/v2/course-live/get-my-course-month",
            params={"month": "2026-10"},
            authorization_required=True,
        )

        self.assertEqual(result["code"], 0)
        request = connector._once.call_args
        self.assertIn("get-my-course-month", request.args[1])
        self.assertIsNone(request.kwargs["headers"])

    def test_webvpn_authorized_request_attaches_bearer_from_legacy_urlencoded_jar_carrier(self):
        # 客户端同法（src/api/icourse.py:919-931）的另一半：旧格式载体仍在罐
        # 时照样附带（真实罐扫描 + URL 解码链，非 mock）。
        # legacy_carrier 为合成假值非真实凭据；变量名避开 gitleaks
        # generic-api-key 关键词预滤（token/secret/key 等），保持全文件扫描强度。
        legacy_carrier = "legacy-carrier-bearer-0123456789"
        serialized = quote('{i:0;s:6:"_token";i:1;s:%d:"%s";}' % (len(legacy_carrier), legacy_carrier), safe="")
        connector = object.__new__(PlatformSession)
        connector._course_direct = False
        connector.session = SimpleNamespace(cookies={"legacy_carrier": serialized})
        response = Mock(status_code=200)
        response.json.return_value = {"code": 0, "list": []}
        connector._once = Mock(return_value=response)

        connector._course_json(
            "/courseapi/v2/course-live/get-my-course-month",
            params={"month": "2026-10"},
            authorization_required=True,
        )

        request = connector._once.call_args
        self.assertEqual(request.kwargs["headers"], {"Authorization": f"Bearer {legacy_carrier}"})

    def test_direct_session_keeps_strict_bearer_extraction(self):
        # direct 会话零回退钉：bearer 是 direct 路径唯一凭证，载体缺席仍
        # 硬抛（与 webvpn 路径的省略语义有意不同）。
        connector = object.__new__(PlatformSession)
        connector._course_direct = True
        connector.course_session = SimpleNamespace(cookies={"unrelated_cookie": "value"})
        with self.assertRaisesRegex(PlatformSessionError, "platform_session_rejected"):
            connector._extract_course_bearer()

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

            def media_source(self, course_id, sub_id, *, relogin=None):
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
        connector._course_direct = True
        connector._webvpn_ready = True
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

    def test_direct_media_does_not_offer_unverified_webvpn_fallback(self):
        connector = self._media_connector()
        connector._webvpn_ready = False
        resolved = ResolvedSource(
            "https://media.example.edu/lecture.mp4?clientUUID=test&t=test",
            {"User-Agent": "CourseLens"},
            "93.184.216.34",
        )
        with patch(
            "courselens_worker.source.resolve_source_address", return_value=resolved
        ):
            source = connector.media_source("1", "2")
        self.assertNotIn("_fallback_source", source)

    def test_direct_slide_sources_do_not_require_webvpn_cookie(self):
        connector = object.__new__(PlatformSession)
        connector._course_direct = True
        connector._webvpn_ready = False
        connector._source_headers = lambda: {
            "Cookie": "not-forwarded", "User-Agent": "CourseLens", "Accept": "*/*"
        }
        connector._course_json = Mock(side_effect=[{
            "list": [{
                "created_sec": 3,
                "content": '{"pptimgurl":"https://media.example.edu/slide.jpg"}',
            }]
        }])

        sources = connector.slide_sources("1", "2")

        self.assertEqual(sources[0]["source"]["url"], "https://media.example.edu/slide.jpg")
        self.assertNotIn("Cookie", sources[0]["source"]["headers"])
        alternate = sources[0]["source"]["_alternate_source"]
        self.assertEqual(urlsplit(alternate["url"]).hostname, "webvpn.fudan.edu.cn")
        self.assertEqual(alternate["headers"]["Cookie"], "not-forwarded")

    def test_webvpn_slide_sources_wrap_upstream_with_session_cookie(self):
        connector = object.__new__(PlatformSession)
        connector._course_direct = False
        connector._webvpn_ready = True
        connector._source_headers = lambda: {
            "Cookie": "session-scoped", "User-Agent": "CourseLens", "Accept": "*/*"
        }
        connector._course_json = Mock(side_effect=[{
            "list": [{
                "created_sec": 3,
                "content": '{"pptimgurl":"https://media.example.edu/slide.jpg"}',
            }]
        }])

        sources = connector.slide_sources("1", "2")

        parsed = urlsplit(sources[0]["source"]["url"])
        self.assertEqual(parsed.hostname, "webvpn.fudan.edu.cn")
        self.assertEqual(parsed.scheme, "https")
        self.assertIn("/https/", sources[0]["source"]["url"])
        self.assertEqual(sources[0]["source"]["headers"]["Cookie"], "session-scoped")
        self.assertEqual(sources[0]["page_num"], 1)
        alternate = sources[0]["source"]["_alternate_source"]
        self.assertEqual(alternate["url"], "https://media.example.edu/slide.jpg")
        self.assertNotIn("Cookie", alternate["headers"])

    def test_slide_sources_carry_bounded_deterministic_deck_scope(self):
        connector = object.__new__(PlatformSession)
        connector._course_direct = True
        connector._webvpn_ready = False
        connector._source_headers = lambda: {"User-Agent": "CourseLens", "Accept": "*/*"}
        connector._course_json = Mock(side_effect=[
            {"list": [{
                "created_sec": 3,
                "content": '{"pptimgurl":"https://media.example.edu/slide.jpg"}',
            }]},
            {"list": [{
                "created_sec": 4,
                "content": '{"pptimgurl":"https://media.example.edu/slide2.jpg"}',
            }]},
            {"list": [{
                "created_sec": 5,
                "content": '{"pptimgurl":"https://media.example.edu/slide3.jpg"}',
            }]},
        ])

        first = connector.slide_sources("course-1", "sub-2")
        second = connector.slide_sources("course-1", "sub-2")
        other = connector.slide_sources("course-1", "sub-3")

        deck = first[0]["deck"]
        self.assertRegex(deck["deck_id"], r"^deck-[0-9a-f]{12}$")
        self.assertRegex(deck["source_id"], r"^src:[0-9a-f]{12}$")
        self.assertEqual(deck, second[0]["deck"])
        self.assertNotEqual(deck, other[0]["deck"])
        self.assertEqual([item["page_num"] for item in first], [1])
        rendered = json.dumps(first)
        self.assertNotIn("course-1", rendered)
        self.assertNotIn("sub-2", rendered)

    def test_slide_only_materialization_closes_the_connector_after_enumeration(self):
        closed = []

        class SlidesConnector(_FakeConnector):
            def close(self):
                closed.append(True)

        job = {
            "payload": {
                "source_session": {
                    "provider": "runner-session-v1", "course_id": "1",
                    "sub_id": "2", "media": False, "slides": True,
                },
            },
            "secrets": {
                "source_credentials": {"account": "account", "password": "password"}
            },
        }
        with patch("courselens_worker.platform_session.PlatformSession", SlidesConnector):
            result = materialize_job_sources(job)

        self.assertEqual(len(result["payload"]["slides"]), 1)
        self.assertNotIn("_close_source_session", result["payload"])
        self.assertEqual(closed, [True])

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

    def test_m15b_constants_are_pinned_against_drift(self):
        """夜4 testbench §T5/T8 推荐束以常量+钉落地：att3 外梯、票据读超时 (5,20)。"""
        from courselens_worker import platform_session as module

        self.assertEqual(module._MATERIALIZE_LOGIN_ATTEMPTS, 3)
        self.assertEqual(module._TICKET_READ_TIMEOUT, (5, 20))
        with open(module.__file__, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("timeout=_TICKET_READ_TIMEOUT", source)
        self.assertIn("range(_MATERIALIZE_LOGIN_ATTEMPTS)", source)
        self.assertNotIn("timeout=(5, 12)", source)

    def test_m15b_context_missing_is_now_retried_by_the_login_ladder(self):
        """B1 核心钉：context 族从单请求硬死变三次全链重试（H1 主力失败类）。"""
        class ContextMissingConnector(_FakeConnector):
            attempts = 0

            def login(self, account, password):
                type(self).attempts += 1
                if type(self).attempts < 3:
                    raise PlatformSessionError("platform_auth_context_missing")
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
        with patch("courselens_worker.platform_session.PlatformSession", ContextMissingConnector), patch(
            "courselens_worker.platform_session.time.sleep"
        ) as sleep:
            materialize_job_sources(job)
        self.assertEqual(ContextMissingConnector.attempts, 3)
        self.assertEqual(sleep.call_count, 2)

        class AlwaysContextMissingConnector(_FakeConnector):
            attempts = 0

            def login(self, account, password):
                type(self).attempts += 1
                raise PlatformSessionError("platform_auth_context_missing")

        exhausted_job = {
            "payload": {
                "media": {},
                "source_session": {
                    "provider": "runner-session-v1", "course_id": "1", "sub_id": "2",
                    "media": True, "slides": False,
                },
            },
            "secrets": {"source_credentials": {"account": "a", "password": "p"}},
        }
        with patch("courselens_worker.platform_session.PlatformSession", AlwaysContextMissingConnector), patch(
            "courselens_worker.platform_session.time.sleep"
        ) as sleep:
            with self.assertRaises(PlatformSessionError):
                materialize_job_sources(exhausted_job)
        self.assertEqual(AlwaysContextMissingConnector.attempts, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_p55_midtask_relogin_ladder_retries_transient_context_missing(self):
        """P55 核心钉：中途续登从单发裸死变三梯（真机事故 2026-09-24 117s 处）。"""
        from courselens_worker.platform_session import _bounded_midtask_relogin

        attempts = {"count": 0}
        lines = []

        def relogin():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise PlatformSessionError("platform_auth_context_missing")

        with patch("courselens_worker.platform_session._emit_session_telemetry", side_effect=lines.append), \
             patch("courselens_worker.platform_session.time.sleep") as sleep:
            _bounded_midtask_relogin(relogin)
        self.assertEqual(attempts["count"], 3)
        self.assertEqual([c.args for c in sleep.call_args_list], [(2.0,), (4.0,)])
        self.assertEqual(lines, [
            "stage=relogin attempt=1 outcome=retry reason=platform_auth_context_missing",
            "stage=relogin attempt=2 outcome=retry reason=platform_auth_context_missing",
        ])

    def test_p55_midtask_relogin_ladder_exhausts_to_closed_set_failure(self):
        """P55：梯尽按原闭集码诚实失败，遥测留全程痕（worker_failed 语义不变）。"""
        from courselens_worker.platform_session import _bounded_midtask_relogin

        attempts = {"count": 0}
        lines = []

        def relogin():
            attempts["count"] += 1
            raise PlatformSessionError("platform_auth_context_missing")

        with patch("courselens_worker.platform_session._emit_session_telemetry", side_effect=lines.append), \
             patch("courselens_worker.platform_session.time.sleep"):
            with self.assertRaises(PlatformSessionError) as captured:
                _bounded_midtask_relogin(relogin)
        self.assertEqual(str(captured.exception), "platform_auth_context_missing")
        self.assertEqual(attempts["count"], 3)
        self.assertEqual(
            lines[-1],
            "stage=relogin attempt=3 outcome=failed reason=platform_auth_context_missing",
        )

    def test_p55_midtask_relogin_non_retryable_code_fails_once_without_ladder(self):
        """P55：挑战页等非瞬态码不入梯——一次即败，不放大等待。"""
        from courselens_worker.platform_session import _bounded_midtask_relogin

        attempts = {"count": 0}
        lines = []

        def relogin():
            attempts["count"] += 1
            raise PlatformSessionError("platform_challenge_required")

        with patch("courselens_worker.platform_session._emit_session_telemetry", side_effect=lines.append), \
             patch("courselens_worker.platform_session.time.sleep") as sleep:
            with self.assertRaises(PlatformSessionError):
                _bounded_midtask_relogin(relogin)
        self.assertEqual(attempts["count"], 1)
        self.assertEqual(sleep.call_count, 0)
        self.assertEqual(
            lines,
            ["stage=relogin attempt=1 outcome=failed reason=platform_challenge_required"],
        )

    def test_p55_login_registers_the_laddered_session_relogin(self):
        """P55 接线钉：登录成功登记的本人续登回调自带三梯。"""
        session = object.__new__(PlatformSession)
        session._login_course_direct = Mock()
        session.login("a", "p")
        self.assertIsNotNone(session._relogin)

        logins = {"count": 0}

        def flaky(account, password):
            logins["count"] += 1
            if logins["count"] == 1:
                raise PlatformSessionError("platform_auth_context_missing")

        session.login = Mock(side_effect=flaky)
        lines = []
        with patch("courselens_worker.platform_session._emit_session_telemetry", side_effect=lines.append), \
             patch("courselens_worker.platform_session.time.sleep"):
            session._relogin()
        self.assertEqual(logins["count"], 2, "瞬态一发不终局：梯内重试后成功")
        self.assertEqual(len(lines), 1, "失败尝试留遥测行，成功静默")

    def test_p55_materialize_media_relogin_goes_through_the_same_ladder(self):
        """P55 接线钉：materialize 显式续登回调与 _relogin 登记面同梯。"""
        class LadderRecordingConnector(_FakeConnector):
            login_count = 0
            captured_relogin = None

            def login(self, account, password):
                type(self).login_count += 1
                if type(self).login_count == 2:
                    raise PlatformSessionError("platform_auth_context_missing")

            def media_source(self, course_id, sub_id, *, relogin=None):
                type(self).captured_relogin = relogin
                return {"url": "https://example.org/stream.mp4", "headers": {"Cookie": "sealed"}}

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
        with patch("courselens_worker.platform_session.PlatformSession", LadderRecordingConnector), \
             patch("courselens_worker.platform_session._emit_session_telemetry") as telemetry, \
             patch("courselens_worker.platform_session.time.sleep") as sleep:
            materialize_job_sources(job)
            callback = LadderRecordingConnector.captured_relogin
            self.assertIsNotNone(callback)
            callback()
        self.assertEqual(LadderRecordingConnector.login_count, 3, "续登撞瞬态一发→梯内重试后成功")
        self.assertEqual(telemetry.call_count, 1)
        self.assertEqual(sleep.call_count, 1)

    def test_p55_midtask_relogin_constants_are_pinned(self):
        """P55：中途梯与启动登录梯同族同量，接线面钉源防漂移。"""
        from courselens_worker import platform_session as module

        self.assertEqual(module._MIDTASK_RELOGIN_ATTEMPTS, module._MATERIALIZE_LOGIN_ATTEMPTS)
        self.assertEqual(module._MIDTASK_RELOGIN_BACKOFF_CAP, 8.0)
        with open(module.__file__, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("stage=relogin attempt=", source)
        self.assertEqual(
            source.count("_bounded_midtask_relogin("),
            3,
            "一次 def + 两处接线（_relogin 登记、materialize 显式回调）",
        )

    def _context_page(self, body):
        """A non-redirect 200 response for the webvpn context stage."""
        page = Mock(status_code=200)
        page.headers = {"Location": ""}
        page.content = body.encode("utf-8")
        return page

    def test_b3_challenge_page_gets_its_own_code(self):
        """B3 核心钉：挑战页与「IDP 未发 lck」塌缩解耦为独立码（testbench §T9）。"""
        connector = object.__new__(PlatformSession)
        page = self._context_page(
            "<html><head><meta http-equiv=\"refresh\" content='2'></head>"
            "<body><div id=\"challenge\">risk-check</div>"
            "<script>document.cookie='risk=1';</script></body></html>"
        )
        connector._once = Mock(return_value=page)
        with self.assertRaises(PlatformSessionError) as captured:
            connector._login_webvpn("account", "password")
        self.assertEqual(str(captured.exception), "platform_challenge_required")
        self.assertEqual(captured.exception.connection_stage, "webvpn_context")

    def test_b3_plain_no_lck_page_keeps_the_historical_code(self):
        """普通无 lck 登录页（form/password 词汇、无挑战特征）维持既有码。"""
        connector = object.__new__(PlatformSession)
        connector._once = Mock(return_value=self._context_page(
            "<html><body><form action='/login'><input type='password'>"
            "login page</form></body></html>"
        ))
        with self.assertRaises(PlatformSessionError) as captured:
            connector._login_webvpn("account", "password")
        self.assertEqual(str(captured.exception), "platform_auth_context_missing")

    def test_b3_each_conservative_marker_classifies_as_challenge(self):
        for marker in (
            '<meta http-equiv="refresh"',
            '<div id="challenge">',
            "var value = document.cookie;",
        ):
            connector = object.__new__(PlatformSession)
            connector._once = Mock(return_value=self._context_page(f"<html>{marker}</html>"))
            with self.assertRaises(PlatformSessionError) as captured:
                connector._login_webvpn("account", "password")
            self.assertEqual(str(captured.exception), "platform_challenge_required", marker)

    def test_b3_challenge_code_never_enters_the_retry_closed_sets(self):
        """挑战页需人工确认：独立码不入任何登录重试闭集（不重试可指引）。"""
        from courselens_worker.platform_session import (
            _RETRYABLE_LOGIN_ERRORS,
            _RETRYABLE_SESSION_ERRORS,
        )

        self.assertNotIn("platform_challenge_required", _RETRYABLE_LOGIN_ERRORS)
        self.assertNotIn(
            "platform_challenge_required", PlatformSession._RETRYABLE_WEBVPN_LEG_ERRORS
        )
        self.assertNotIn("platform_challenge_required", _RETRYABLE_SESSION_ERRORS)
        # 而既有 context 族码保持在登录重试闭集内（B1 行为不回退）。
        self.assertIn("platform_auth_context_missing", _RETRYABLE_LOGIN_ERRORS)

    def test_b3_challenge_code_survives_runner_reduction_closed_set(self):
        """码表等集钉：platform 族 14 码逐一过归约且不塌缩；兜底码不收编。"""
        platform_codes = {
            "platform_credentials_missing", "platform_connection_failed",
            "platform_redirect_rejected", "platform_auth_context_missing",
            "platform_auth_method_missing", "platform_key_rejected",
            "platform_auth_failed", "platform_ticket_missing",
            "platform_ticket_rejected", "platform_session_rejected",
            "platform_course_context_missing", "platform_course_request_failed",
            "platform_media_missing", "platform_challenge_required",
        }
        for code in platform_codes:
            self.assertEqual(safe_worker_error_detail(PlatformSessionError(code)), code)
        self.assertEqual(
            safe_worker_error_detail(PlatformSessionError("platform_unknown_code")),
            "platform_session_failed",
        )

    def test_p65_already_authed_variant_returns_success_without_lck(self):
        """P65 (a)：SSO 直通链无 lck、终页=服务门户 → 验证过=登录成功零异常。"""
        connector = object.__new__(PlatformSession)
        portal = Mock(status_code=200)
        portal.headers = {"Location": ""}
        portal.url = "https://webvpn.fudan.edu.cn/webvpn/portal"
        portal.content = b"<html><body>portal-ok</body></html>"
        connector._once = Mock(return_value=portal)
        lines = []
        with patch("courselens_worker.platform_session._emit_session_telemetry", side_effect=lines.append), \
             patch("courselens_worker.platform_session.time.sleep"):
            connector._login_webvpn("account", "password")
        self.assertEqual(lines, ["stage=relogin-variant outcome=already-authed"])
        self.assertEqual(connector._once.call_count, 2, "链一进+验证一探，零多余请求")

    def test_p65_variant_verify_failure_purges_cookies_and_retries_fresh_once(self):
        """P65 (b)：验证不过 → 清态（强制全新登录流）→ 同一 attempt 全新链一次。"""
        connector = object.__new__(PlatformSession)
        connector._userinfo = {"stale": "identity"}
        connector._course_bearer = "stale-bearer"
        connector._webvpn_ready = True
        connector._once = Mock(return_value=self._context_page(
            "<html><body>webvpn session expired</body></html>"
        ))
        connector._verify_webvpn_bounded = Mock(return_value=False)
        connector._rebuild_sessions = Mock()
        real_method = PlatformSession._login_webvpn
        calls = []

        def fake(account, password, *, _fresh_retry=False):
            calls.append(_fresh_retry)
            if _fresh_retry:
                return None  # 全新 cookie jar 下链路拿到 lck：成功
            return real_method(connector, account, password)

        connector._login_webvpn = fake
        lines = []
        with patch("courselens_worker.platform_session._emit_session_telemetry", side_effect=lines.append):
            connector._login_webvpn("account", "password")
        self.assertEqual(calls, [False, True], "干净重试恰一次，且有防重入护栏")
        connector._rebuild_sessions.assert_called_once_with()
        connector._verify_webvpn_bounded.assert_called_once_with()
        self.assertIsNone(connector._userinfo, "陈旧身份缓存随清态一起作废")
        self.assertEqual(connector._course_bearer, "")
        self.assertFalse(connector._webvpn_ready)
        self.assertEqual(lines, ["stage=relogin-variant outcome=fresh-retry"])

    def test_p65_challenge_page_still_wins_over_the_variant(self):
        """P65 (c)：挑战页分类仍是第一分支——会话实际可用也保持挑战码。"""
        connector = object.__new__(PlatformSession)
        page = self._context_page(
            "<html><head><meta http-equiv=\"refresh\" content='2'></head>"
            "<body><div id=\"challenge\">risk-check</div>"
            "<script>document.cookie='risk=1';</script></body></html>"
        )
        connector._once = Mock(return_value=page)
        connector._verify_webvpn_bounded = Mock(return_value=True)
        with self.assertRaises(PlatformSessionError) as captured:
            connector._login_webvpn("account", "password")
        self.assertEqual(str(captured.exception), "platform_challenge_required")
        self.assertEqual(captured.exception.connection_stage, "webvpn_context")
        connector._verify_webvpn_bounded.assert_not_called()

    def test_p65_unknown_page_after_failed_verify_and_fresh_retry_keeps_closed_set(self):
        """P65 (d)：三皆非的未知页 → 验证败+干净重试败 → 既有闭集码诚实失败。"""
        connector = object.__new__(PlatformSession)
        connector._once = Mock(return_value=self._context_page(
            "<html><body>maintenance</body></html>"
        ))
        connector._verify_webvpn_bounded = Mock(return_value=False)
        connector._rebuild_sessions = Mock()
        lines = []
        with patch("courselens_worker.platform_session._emit_session_telemetry", side_effect=lines.append), \
             patch("courselens_worker.platform_session.time.sleep"):
            with self.assertRaises(PlatformSessionError) as captured:
                connector._login_webvpn("account", "password")
        self.assertEqual(str(captured.exception), "platform_auth_context_missing")
        self.assertEqual(lines, ["stage=relogin-variant outcome=fresh-retry"])
        self.assertEqual(connector._rebuild_sessions.call_count, 1)
        self.assertEqual(connector._verify_webvpn_bounded.call_count, 2, "首次+干净重试各验证一次")

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

    def test_media_refresh_relogins_after_session_error_and_retries(self):
        from courselens_worker.platform_session import _bounded_session_refresh

        refreshes = {"count": 0}
        relogins = {"count": 0}

        def refresh():
            refreshes["count"] += 1
            if refreshes["count"] == 1:
                raise PlatformSessionError("platform_course_request_failed")
            return {"url": "https://example.org/fresh.mp4"}

        with patch("courselens_worker.platform_session.time.sleep") as sleep:
            source = _bounded_session_refresh(refresh, lambda: relogins.__setitem__("count", relogins["count"] + 1))
        self.assertEqual(source["url"], "https://example.org/fresh.mp4")
        self.assertEqual(refreshes["count"], 2)
        self.assertEqual(relogins["count"], 1)
        sleep.assert_called_once_with(2.0)

    def test_media_refresh_exhausts_and_raises_after_bounded_relogins(self):
        from courselens_worker.platform_session import _bounded_session_refresh

        calls = {"refresh": 0, "relogin": 0}

        def refresh():
            calls["refresh"] += 1
            raise PlatformSessionError("platform_connection_failed")

        with patch("courselens_worker.platform_session.time.sleep"), \
             self.assertRaises(PlatformSessionError):
            _bounded_session_refresh(refresh, lambda: calls.__setitem__("relogin", calls["relogin"] + 1))
        # 第四十一案：窗口钉在档（8 次尝试/7 次续登），改窗必改此钉并复核覆盖时长。
        self.assertEqual(calls["refresh"], 8)
        self.assertEqual(calls["relogin"], 7)

    def test_media_refresh_window_covers_a_client_relogin_and_stays_bounded(self):
        """第四十一案：窗口必须比「学生重启客户端+重新登录」更长，且仍然有界。

        旧窗 2s+4s＝6s 短于重登录耗时，16/27 块处会话作废即整单失败。
        """
        from courselens_worker.platform_session import (
            _SESSION_REFRESH_BACKOFF_CAP,
            _bounded_session_refresh,
        )

        sleeps = []

        def refresh():
            raise PlatformSessionError("platform_course_request_failed")

        with patch("courselens_worker.platform_session.time.sleep", side_effect=sleeps.append), \
             self.assertRaises(PlatformSessionError):
            _bounded_session_refresh(refresh, lambda: None)
        self.assertGreaterEqual(sum(sleeps), 120.0, "退避窗必须覆盖客户端重登录时长")
        self.assertLessEqual(max(sleeps), _SESSION_REFRESH_BACKOFF_CAP, "单次退避有界")
        self.assertEqual(len(sleeps), 7)

    def test_media_refresh_without_relogin_tries_once(self):
        from courselens_worker.platform_session import _bounded_session_refresh

        calls = {"refresh": 0}

        def refresh():
            calls["refresh"] += 1
            raise PlatformSessionError("platform_course_request_failed")

        with self.assertRaises(PlatformSessionError):
            _bounded_session_refresh(refresh)
        self.assertEqual(calls["refresh"], 1)

    def test_media_refresh_non_session_error_is_neither_retried_nor_relogged(self):
        from courselens_worker.platform_session import _bounded_session_refresh

        calls = {"refresh": 0, "relogin": 0}

        def refresh():
            calls["refresh"] += 1
            raise PlatformSessionError("platform_media_missing")

        with self.assertRaises(PlatformSessionError):
            _bounded_session_refresh(refresh, lambda: calls.__setitem__("relogin", calls["relogin"] + 1))
        self.assertEqual(calls, {"refresh": 1, "relogin": 0})

    @staticmethod
    def _media_session_double(base):
        """会话替身：只保留取源/签名/解析三个接缝，不触网。"""
        session = object.__new__(PlatformSession)
        session._webvpn_ready = False
        session._relogin = None
        session._source_headers = lambda: {}
        session._sign = lambda url, now: url
        session._media_base = base
        return session

    @staticmethod
    def _stub_resolution():
        return patch(
            "courselens_worker.source.resolve_source_address",
            side_effect=lambda url, headers=None, public_ip_hint="": ResolvedSource(
                url=url, headers=dict(headers or {}), ip="203.0.113.7",
            ),
        )

    def test_media_source_defaults_to_the_session_owned_relogin(self):
        """第四十一案主症：每日自动化链不传 relogin，媒体重取也必须能续登。"""
        relogin = lambda: None
        session = self._media_session_double(
            lambda course_id, sub_id: ("https://icourse.fudan.edu.cn/lecture.mp4", 0)
        )
        session._relogin = relogin
        with self._stub_resolution(), patch(
            "courselens_worker.platform_session._bounded_session_refresh"
        ) as bounded:
            source = session.media_source("36941", "l-1")
            source["_refresh_source"]()
        self.assertIs(bounded.call_args.args[1], relogin, "缺省必须用会话本人续登回调")

    def test_session_owned_relogin_reauthenticates_between_refresh_tries(self):
        """会话自持回调端到端：一次会话失效 → 自动续登 → 重取成功。"""
        from courselens_worker.platform_session import _bounded_session_refresh

        logins = {"count": 0}
        session = object.__new__(PlatformSession)
        session._relogin = None
        session._login_course_direct = lambda account, password: logins.__setitem__(
            "count", logins["count"] + 1
        )
        session._login_webvpn_full = Mock()
        session.login("2020001", "synthetic")
        session._login_webvpn_full.assert_not_called()
        self.assertTrue(callable(session._relogin))

        refreshes = {"count": 0}

        def refresh():
            refreshes["count"] += 1
            if refreshes["count"] == 1:
                raise PlatformSessionError("platform_course_request_failed")
            return {"url": "https://example.org/fresh.mp4"}

        with patch("courselens_worker.platform_session.time.sleep"):
            value = _bounded_session_refresh(refresh, session._relogin)
        self.assertEqual(value["url"], "https://example.org/fresh.mp4")
        self.assertEqual(refreshes["count"], 2)
        self.assertEqual(logins["count"], 2, "首次登录 + 一次续登")

    def test_media_source_without_a_logged_in_session_keeps_single_try(self):
        """从未登录过的会话没有可用的续登回调：保持一次即败，不误重试。"""
        bases = {"count": 0}

        def base(course_id, sub_id):
            bases["count"] += 1
            if bases["count"] == 1:
                return ("https://icourse.fudan.edu.cn/lecture.mp4", 0)
            raise PlatformSessionError("platform_course_request_failed")

        session = self._media_session_double(base)
        with self._stub_resolution():
            source = session.media_source("36941", "l-1")
            with self.assertRaises(PlatformSessionError):
                source["_refresh_source"]()
        self.assertEqual(bases["count"], 2, "无回调时不重取")

    def test_close_releases_the_session_owned_relogin(self):
        session = object.__new__(PlatformSession)
        session._relogin = lambda: None
        session._course_bearer = ""
        session._userinfo = None
        session._webvpn_ready = False
        session.course_session = Mock()
        session.session = Mock()
        session.close()
        self.assertIsNone(session._relogin)

    def test_materialize_media_source_receives_relogin_callback(self):
        captured = {}

        class CapturingConnector(_FakeConnector):
            def media_source(self, course_id, sub_id, *, relogin=None):
                captured["relogin_callable"] = callable(relogin)
                return {"url": "https://example.org/stream.mp4", "headers": {"Cookie": "sealed"}}

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
        with patch("courselens_worker.platform_session.PlatformSession", CapturingConnector):
            materialize_job_sources(job)
        self.assertTrue(captured["relogin_callable"])

    def test_persistent_connection_failure_backs_off_through_all_three_attempts(self):
        class DownConnector(_FakeConnector):
            attempts = 0

            def login(self, account, password):
                type(self).attempts += 1
                raise PlatformSessionError("platform_connection_failed")

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
        with patch("courselens_worker.platform_session.PlatformSession", DownConnector), patch(
            "courselens_worker.platform_session.time.sleep"
        ) as sleep:
            with self.assertRaises(PlatformSessionError):
                materialize_job_sources(job)
        self.assertEqual(DownConnector.attempts, 3)
        self.assertEqual([item.args[0] for item in sleep.call_args_list], [2.0, 4.0])

    def test_bounded_reverify_recovers_after_transient_verify_failures(self):
        checks = {"calls": 0}

        def flaky_verify():
            checks["calls"] += 1
            return checks["calls"] >= 3

        with patch("courselens_worker.platform_session.time.sleep") as sleep:
            self.assertTrue(PlatformSession._bounded_reverify(flaky_verify))
        self.assertEqual(checks["calls"], 3)
        self.assertEqual(sleep.call_count, 2)

    def test_bounded_reverify_reports_persistent_failure(self):
        with patch("courselens_worker.platform_session.time.sleep"):
            self.assertFalse(PlatformSession._bounded_reverify(lambda: False, attempts=2))

    def test_cloud_login_rebuilds_transient_rejected_sessions(self):
        class FlakyConnector(_FakeConnector):
            attempts = 0
            transport_values = []

            def login(self, account, password, *, allow_webvpn_fallback=True):
                type(self).attempts += 1
                self.assert_direct_only = not allow_webvpn_fallback
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
        self.assertEqual(FlakyConnector.transport_values, ["curl", "requests", "requests"])
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


class _SlideRow:
    @staticmethod
    def row(row_id, created_sec, image="https://media.example.edu/slide.jpg"):
        return {"id": str(row_id), "created_sec": created_sec,
                "content": json.dumps({"pptimgurl": image})}


class SlidePaginationGuardTests(unittest.TestCase):
    """search-ppt 分页护栏：取代旧 50 页上限的大预算熔断。"""

    def _connector(self):
        connector = object.__new__(PlatformSession)
        connector._course_direct = True
        connector._webvpn_ready = False
        connector._source_headers = lambda: {"User-Agent": "CourseLens", "Accept": "*/*"}
        return connector

    def test_slide_sources_walk_full_pages_until_the_server_short_page(self):
        connector = self._connector()
        connector._course_json = Mock(side_effect=[
            {"list": [_SlideRow.row("1", 1), _SlideRow.row("2", 2)]},
            {"list": [_SlideRow.row("3", 3), _SlideRow.row("4", 4)]},
            {"list": [_SlideRow.row("5", 5)]},
        ])
        with patch("courselens_worker.platform_session.SLIDE_PAGE_SIZE", 2):
            sources = connector.slide_sources("course", "sub")
        self.assertEqual([item["page_num"] for item in sources], [1, 2, 3, 4, 5])
        self.assertEqual(connector._course_json.call_count, 3)
        first_call = connector._course_json.call_args_list[0]
        self.assertEqual(first_call.kwargs["params"]["per_page"], 2)

    def test_slide_sources_skip_rows_that_fail_shape_validation(self):
        connector = self._connector()
        connector._course_json = Mock(return_value={"list": [
            "not-a-dict",
            {"id": "2", "created_sec": 2, "content": "{broken"},
            {"id": "3", "created_sec": "nonsense", "content": json.dumps({"pptimgurl": "https://media.example.edu/3.jpg"})},
            {"id": "4", "created_sec": 4, "content": json.dumps({"pptthumb": "only"})},
            _SlideRow.row("5", 5),
        ]})
        sources = connector.slide_sources("course", "sub")
        self.assertEqual([item["page_num"] for item in sources], [1])
        self.assertEqual(sources[0]["created_sec"], 5)

    def test_slide_sources_fail_closed_on_a_stalled_server_cursor(self):
        connector = self._connector()
        stuck = [_SlideRow.row("1", 1), _SlideRow.row("2", 2)]
        connector._course_json = Mock(side_effect=[
            {"list": list(stuck)},
            {"list": list(stuck)},
        ])
        with patch("courselens_worker.platform_session.SLIDE_PAGE_SIZE", 2):
            with self.assertRaisesRegex(PlatformSessionError, "platform_slide_pagination_stalled"):
                connector.slide_sources("course", "sub")

    def test_slide_sources_fail_closed_on_a_record_storm(self):
        connector = self._connector()
        connector._course_json = Mock(return_value={"list": [
            _SlideRow.row("1", 1), _SlideRow.row("2", 2),
        ]})
        with patch("courselens_worker.platform_session.SLIDE_RECORD_STORM_LIMIT", 1):
            with self.assertRaisesRegex(PlatformSessionError, "platform_slide_record_storm"):
                connector.slide_sources("course", "sub")

    def test_course_json_rejects_oversized_response_bodies(self):
        connector = self._connector()
        connector._course_bearer = "bounded-test-token"
        response = Mock(status_code=200)
        response.content = b"x" * 100
        connector._direct_once = Mock(return_value=response)
        with self.assertRaisesRegex(PlatformSessionError, "platform_slide_response_too_large"):
            connector._course_json(
                "/pptnote/v1/schedule/search-ppt", params={"page": 1}, max_bytes=10,
            )

class ProbeSubLegStageLineTests(unittest.TestCase):
    """DIAG 提案 2 口径（cloudverifyfix 恰域注记后续建议）：探针子腿拆行——
    ui_session（authExecute 成功位）与 webvpn_ticket（票据跟单验证完成位）
    各发一行闭集 stage 行（仅腿名+elapsed 秒数，零 URL 零账号值零票据面），
    三条登录链同口径。"""

    @staticmethod
    def _response(*, status=200, headers=None, url="https://icourse.fudan.edu.cn/user",
                  text="", json_value=None):
        response = Mock()
        response.status_code = status
        response.headers = dict(headers or {})
        response.url = url
        response.text = text
        response.json = Mock(return_value=json_value)
        response.close = Mock()
        return response

    def _webvpn_connector(self):
        connector = PlatformSession(transport="requests")

        def fake_once(method, url, **kwargs):
            stage = (kwargs or {}).get("connection_stage")
            if stage == "webvpn_context":
                return self._response(
                    status=302,
                    headers={"Location": "https://id.fudan.edu.cn/ac/?lck=test-lck"},
                )
            if stage == "webvpn_auth_methods":
                return self._response(json_value={
                    "data": [{"moduleCode": "userAndPwd", "authChainCode": "chain-1"}],
                    "requestType": "chain_type",
                })
            if stage == "webvpn_key":
                return self._response(json_value={"data": "k"})
            if stage == "webvpn_auth_execute":
                return self._response(json_value={"code": "200", "loginToken": "tok"})
            if stage == "webvpn_ticket":
                return self._response(text=(
                    "<script>locationValue=\"https://webvpn.fudan.edu.cn/"
                    "1/portal?ticket=ST-1\"</script>"))
            raise AssertionError(f"unexpected stage {stage}")

        connector._once = Mock(side_effect=fake_once)
        connector._follow = Mock(return_value=(self._response(), []))
        connector._encrypt_password = Mock(return_value="enc")
        connector._verify_webvpn = Mock(return_value=True)
        return connector

    def test_webvpn_chain_emits_both_subleg_lines(self):
        lines: list[str] = []
        connector = self._webvpn_connector()
        with patch("courselens_worker.platform_session._emit_session_telemetry",
                   side_effect=lines.append):
            connector._login_webvpn("account", "password")
        sublegs = [line for line in lines if line.startswith("stage=")]
        self.assertEqual(len(sublegs), 2, lines)
        self.assertTrue(sublegs[0].startswith("stage=ui_session elapsed="),
                        "authExecute 成功位=ui_session 行")
        self.assertTrue(sublegs[1].startswith("stage=webvpn_ticket elapsed="),
                        "票据跟单验证完成位=webvpn_ticket 行")
        self.assertLess(float(sublegs[0].split("elapsed=")[1]), 5.0)
        self.assertLess(float(sublegs[1].split("elapsed=")[1]), 5.0)
        joined = " ".join(sublegs)
        for leaked in ("account", "password", "test-lck", "ST-1", "tok"):
            self.assertNotIn(leaked, joined, "零敏感面")

    def test_direct_chain_emits_both_subleg_lines(self):
        lines: list[str] = []
        connector = PlatformSession(transport="requests")

        def fake_follow(method, url, **kwargs):
            stage = (kwargs or {}).get("connection_stage")
            if stage == "course_context_direct":
                return (self._response(
                    text="window.location=\"https://icourse.fudan.edu.cn/casapi/"
                         "?forward=x&lck=test-lck\"",
                    url="https://icourse.fudan.edu.cn/casapi/",
                ), [])
            if stage == "course_ticket_follow_direct":
                return (self._response(), [])
            raise AssertionError(f"unexpected follow stage {stage}")

        def fake_direct_once(method, url, **kwargs):
            stage = (kwargs or {}).get("connection_stage")
            if stage == "course_auth_methods_direct":
                return self._response(json_value={
                    "data": [{"moduleCode": "userAndPwd", "authChainCode": "chain-1"}],
                    "requestType": "chain_type",
                })
            if stage == "course_key_direct":
                return self._response(json_value={"data": "k"})
            if stage == "course_auth_execute_direct":
                return self._response(json_value={"code": "200", "loginToken": "tok"})
            if stage == "course_ticket_direct":
                return self._response(text=(
                    "<script>locationValue=\"https://icourse.fudan.edu.cn/"
                    "?ticket=ST-1\"</script>"))
            raise AssertionError(f"unexpected stage {stage}")

        connector._follow = Mock(side_effect=fake_follow)
        connector._direct_once = Mock(side_effect=fake_direct_once)
        connector._encrypt_password = Mock(return_value="enc")
        connector._extract_course_bearer = Mock(return_value="bearer")
        connector._verify_course_direct = Mock(return_value=True)
        with patch("courselens_worker.platform_session._emit_session_telemetry",
                   side_effect=lines.append):
            connector._login_course_direct("account", "password")
        sublegs = [line for line in lines if line.startswith("stage=")]
        self.assertEqual(
            [line.split(" elapsed=")[0] for line in sublegs],
            ["stage=ui_session", "stage=webvpn_ticket"], lines)

    def test_failed_auth_execute_emits_no_ui_session_line(self):
        lines: list[str] = []
        connector = self._webvpn_connector()
        base_once = connector._once.side_effect

        def failing_once(method, url, **kwargs):
            if (kwargs or {}).get("connection_stage") == "webvpn_auth_execute":
                return self._response(json_value={"code": "401"})
            return base_once(method, url, **kwargs)

        connector._once = Mock(side_effect=failing_once)
        with patch("courselens_worker.platform_session._emit_session_telemetry",
                   side_effect=lines.append):
            with self.assertRaises(PlatformSessionError):
                connector._login_webvpn("account", "password")
        self.assertEqual(
            [line for line in lines if line.startswith("stage=ui_session")], [],
            "口令腿失败不发完成行（诚实进度面）")


class TodayScheduleGateLegTests(unittest.TestCase):
    """SMART-SCHED 智能门数据腿：闭集解析钉 + today_schedule_rows 形状/边界钉。

    解析语义逐条对齐客户端 icourse._parse_occurrence_span 的 end 源：纪元
    秒/毫秒、裸 HH:MM[:SS]+YYYY-MM-DD 锚、ISO（naive 补 Asia/Shanghai）；
    纯日期/缺失/不可解析一律 None=不确定行，绝不猜测。
    """

    def test_parse_end_evidence_closed_set_matrix(self):
        epoch = int(datetime(2026, 10, 6, 9, 40, tzinfo=GATE_TZ).timestamp())
        cases = [
            ({"end_at": "09:40"}, "2026-10-06", datetime(2026, 10, 6, 9, 40, tzinfo=GATE_TZ)),
            ({"end_at": "9:40"}, "2026-10-06", datetime(2026, 10, 6, 9, 40, tzinfo=GATE_TZ)),
            ({"end_at": "09:40:30"}, "2026-10-06", datetime(2026, 10, 6, 9, 40, 30, tzinfo=GATE_TZ)),
            ({"end_time": "09:40"}, "2026-10-06", datetime(2026, 10, 6, 9, 40, tzinfo=GATE_TZ)),
            ({"ends_at": "09:40"}, "2026-10-06", datetime(2026, 10, 6, 9, 40, tzinfo=GATE_TZ)),
            ({"end_at": str(epoch)}, "", datetime(2026, 10, 6, 9, 40, tzinfo=GATE_TZ)),
            ({"end_at": str(epoch * 1000)}, "", datetime(2026, 10, 6, 9, 40, tzinfo=GATE_TZ)),
            ({"end_at": "2026-10-06T09:40"}, "", datetime(2026, 10, 6, 9, 40, tzinfo=GATE_TZ)),
            ({"end_at": "2026-10-06T09:40:00+08:00"}, "", datetime(2026, 10, 6, 9, 40, tzinfo=ZoneInfo("Asia/Shanghai"))),
        ]
        for row, anchor, expected in cases:
            self.assertEqual(_parse_end_evidence(row, anchor), expected, row)
        # 不确定行：纪元出界/裸时刻无锚/纯日期/缺失/不可解析 → 一律 None。
        for row, anchor in (
            ({"end_at": "123"}, ""),
            ({"end_at": "99999999999999"}, ""),
            ({"end_at": "09:40"}, ""),
            ({"end_at": "09:40"}, "2026-10-6"),
            ({"end_at": "2026-10-06"}, ""),
            ({"end_at": "", "end_time": "", "ends_at": ""}, "2026-10-06"),
            ({"end_at": "soon"}, "2026-10-06"),
            ({"end_at": "25:70"}, "2026-10-06"),
            ({}, "2026-10-06"),
        ):
            self.assertIsNone(_parse_end_evidence(row, anchor), row)
        # 源序：end_at 可解析即优先；end_at 不可解析回落 end_time。
        self.assertEqual(
            _parse_end_evidence({"end_at": "10:00", "end_time": "09:40"}, "2026-10-06"),
            datetime(2026, 10, 6, 10, 0, tzinfo=GATE_TZ),
        )
        self.assertEqual(
            _parse_end_evidence({"end_at": "soon", "end_time": "09:40"}, "2026-10-06"),
            datetime(2026, 10, 6, 9, 40, tzinfo=GATE_TZ),
        )

    def _connector(self, payload):
        connector = object.__new__(PlatformSession)
        connector._course_json = Mock(return_value=payload)
        return connector

    def test_today_schedule_rows_uses_one_authorized_month_get(self):
        from datetime import date

        connector = self._connector({"code": 0, "list": []})
        rows = connector.today_schedule_rows(today=date(2026, 10, 6))
        self.assertEqual(rows, [])
        connector._course_json.assert_called_once()
        args, kwargs = connector._course_json.call_args
        self.assertEqual(args[0], "/courseapi/v2/course-live/get-my-course-month")
        self.assertEqual(kwargs["params"], {"month": "2026-10"})
        self.assertTrue(kwargs["authorization_required"])
        self.assertEqual(kwargs["timeout"], (10, 30))

    def test_today_schedule_rows_projects_closed_set_shape_only(self):
        from datetime import date

        epoch = int(datetime(2026, 10, 6, 11, 35, tzinfo=GATE_TZ).timestamp())
        payload = {"code": 0, "list": [
            {"date": "2026-10-06", "course": [
                {"id": "36941", "sub_id": "s-1", "title": "机密课名", "end_at": "09:40"},
                {"id": "36941", "sub_id": "s-2", "date": "2026-10-06", "end_time": "11:35"},
                {"course_id": "777", "sub_id": "s-3", "ends_at": str(epoch)},
                {"id": "888", "sub_id": "s-4", "end_at": "2026-10-06"},
                {"id": "888", "sub_id": "s-5"},
                {"sub_id": "s-6", "end_at": "09:40"},
            ]},
            {"course": [{"id": "36941", "sub_id": "s-7", "end_at": "09:40"}]},
        ]}
        connector = self._connector(payload)
        rows = connector.today_schedule_rows(today=date(2026, 10, 6))
        self.assertEqual(len(rows), 6)
        for row in rows:
            # 闭集形状：零标题、零 URL、零原始载荷。
            self.assertEqual(set(row.keys()), {"course_id", "sub_id", "date", "end_precise"})
            self.assertTrue(row["course_id"])
        self.assertEqual(rows[0], {
            "course_id": "36941", "sub_id": "s-1", "date": "2026-10-06",
            "end_precise": datetime(2026, 10, 6, 9, 40, tzinfo=GATE_TZ),
        })
        # 行级 date 覆盖父日锚。
        self.assertEqual(rows[1]["end_precise"], datetime(2026, 10, 6, 11, 35, tzinfo=GATE_TZ))
        # 纪元秒行。
        self.assertEqual(rows[2]["end_precise"], datetime(2026, 10, 6, 11, 35, tzinfo=GATE_TZ))
        # 纯日期 → 不确定；缺 end 源 → 不确定；无 course_id 行被丢弃。
        self.assertIsNone(rows[3]["end_precise"])
        self.assertIsNone(rows[4]["end_precise"])
        # 父日无 date 时锚缺省 → 裸时刻不确定。
        self.assertEqual(rows[5]["sub_id"], "s-7")
        self.assertEqual(rows[5]["date"], "")
        self.assertIsNone(rows[5]["end_precise"])

    def test_today_schedule_rows_rejects_bad_payloads_with_closed_code(self):
        from datetime import date

        storm = {"code": 0, "list": [{"date": "2026-10-06", "course": [
            {"id": str(index), "sub_id": f"s-{index}"} for index in range(1001)
        ]}]}
        for payload in (
            {"code": 997, "list": []},
            {"code": 0, "list": "nope"},
            {"code": 0, "list": [f"day-{index}" for index in range(32)]},
            storm,
        ):
            connector = self._connector(payload)
            with self.assertRaises(PlatformSessionError) as captured:
                connector.today_schedule_rows(today=date(2026, 10, 6))
            self.assertEqual(str(captured.exception), "platform_course_request_failed")

    def test_today_schedule_rows_propagates_transport_failures(self):
        connector = object.__new__(PlatformSession)
        connector._course_json = Mock(side_effect=PlatformSessionError("platform_connection_failed"))
        with self.assertRaises(PlatformSessionError):
            connector.today_schedule_rows()


if __name__ == "__main__":
    unittest.main()
