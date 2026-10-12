"""Chain-level login tests over the local simulator (fixtures/sim_server.py).

B3 链路级用例（夜4 login-testbench §落地建议 #3/#6）：
- challenge 挑战页 → 独立码 platform_challenge_required，零重试（挑战页需要
  人工点一次人机确认，程序重试只会连续撞上同一页）；
- 无 lck 普通页 → platform_auth_context_missing 保持在登录重试闭集内，外层
  梯全链重跑（fresh lck/ticket，绝不重放）后可恢复。
零真实端点：仅 127.0.0.1，OS 分配端口。
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from unittest.mock import patch
from urllib.parse import urlparse

from courselens_worker import platform_session as ps
from courselens_worker.platform_session import (
    PlatformSession,
    PlatformSessionError,
    _MATERIALIZE_LOGIN_ATTEMPTS,
)
from courselens_worker.runner import safe_worker_error_detail

try:
    from fixtures.sim_server import STATS, set_mode, start
except ImportError:  # 直接从 worker/tests 目录运行时
    from sim_server import STATS, set_mode, start


class LoginChainSimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = start()

    def setUp(self):
        set_mode("delivered")

    @contextmanager
    def _endpoints(self):
        saved = (
            ps.WEBVPN_BASE, ps.IDP_BASE, ps.ICOURSE_BASE,
            ps._ALLOWED_HOSTS, ps._validate_url, ps._validate_upstream_url,
        )
        ps.WEBVPN_BASE = self.base
        ps.IDP_BASE = self.base
        ps.ICOURSE_BASE = self.base
        ps._ALLOWED_HOSTS = {"127.0.0.1"}

        def _validate_local(value):
            try:
                parsed = urlparse(str(value or ""))
            except ValueError:
                raise ps._fail("platform_redirect_rejected")
            if (
                parsed.scheme != "http"
                or parsed.hostname != "127.0.0.1"
                or parsed.username
                or parsed.password
            ):
                raise ps._fail("platform_redirect_rejected")
            return parsed.geturl()

        ps._validate_url = _validate_local
        ps._validate_upstream_url = _validate_local
        try:
            yield
        finally:
            (
                ps.WEBVPN_BASE, ps.IDP_BASE, ps.ICOURSE_BASE,
                ps._ALLOWED_HOSTS, ps._validate_url, ps._validate_upstream_url,
            ) = saved

    def test_delivered_chain_signs_in_end_to_end(self):
        """§T1 冒烟判据：净链一次通过完整 WebVPN+课程登录。"""
        with self._endpoints():
            connector = PlatformSession(transport="requests")
            connector._login_webvpn_full("account", "password", attempts=1)
        self.assertTrue(connector._webvpn_ready)
        self.assertEqual(STATS["webvpn_entries"], 1)

    def test_challenge_page_raises_independent_code_without_retry(self):
        """挑战页 → 新码 + 可归约 + 不入任何重试闭集 + 服务端仅见一次进入。"""
        set_mode("challenge")
        with self._endpoints():
            connector = PlatformSession(transport="requests")
            with self.assertRaises(PlatformSessionError) as captured:
                connector._login_webvpn_full("account", "password", attempts=1)
        self.assertEqual(str(captured.exception), "platform_challenge_required")
        self.assertEqual(captured.exception.connection_stage, "webvpn_context")
        self.assertEqual(
            safe_worker_error_detail(captured.exception),
            "platform_challenge_required",
        )
        self.assertNotIn("platform_challenge_required", ps._RETRYABLE_LOGIN_ERRORS)
        self.assertNotIn(
            "platform_challenge_required", PlatformSession._RETRYABLE_WEBVPN_LEG_ERRORS
        )
        self.assertEqual(STATS["webvpn_entries"], 1, "挑战页零重试")

    def test_missing_lck_stays_retryable_and_ladder_recovers(self):
        """无 lck 普通页 → 维持既有码入外层闭集；平台恢复发 lck 后全链重跑成功。"""
        set_mode("not_delivered")
        codes = []
        connector = None
        with self._endpoints():
            for attempt in range(_MATERIALIZE_LOGIN_ATTEMPTS):
                connector = PlatformSession(transport="requests")
                try:
                    connector._login_webvpn_full("account", "password", attempts=1)
                    break
                except PlatformSessionError as exc:
                    codes.append(str(exc))
                    if (
                        str(exc) not in ps._RETRYABLE_LOGIN_ERRORS
                        or attempt == _MATERIALIZE_LOGIN_ATTEMPTS - 1
                    ):
                        raise
                    set_mode("delivered")  # 平台恢复发 lck：模拟外层梯的下一次全链重跑
        self.assertEqual(codes, ["platform_auth_context_missing"])
        self.assertTrue(connector._webvpn_ready)
        # set_mode 每次翻转都会清零计数：这里只对「恢复后的成功跑」计数——
        # 恰一次进入即成功；每次重试独立进入已由挑战页用例（零重试）钉住。
        self.assertEqual(STATS["webvpn_entries"], 1)

    def test_p65_sso_direct_chain_signs_in_without_lck(self):
        """P65 (a) 链路级：SSO 直通（全链无 lck、直发票）→ 变体验证过 → 成功。"""
        set_mode("sso_direct")
        lines = []
        with self._endpoints():
            connector = PlatformSession(transport="requests")
            with patch(
                "courselens_worker.platform_session._emit_session_telemetry",
                side_effect=lines.append,
            ), patch("courselens_worker.platform_session.time.sleep"):
                connector._login_webvpn_full("account", "password", attempts=1)
        self.assertTrue(connector._webvpn_ready)
        # DIAG 提案 2 口径：子腿完成行（ui_session/webvpn_ticket）随真链拆行
        # ——SSO 变体过验后课程腿照常走完整链，两条子腿行依次在案。
        self.assertEqual(
            [line.split(" elapsed=")[0] for line in lines],
            [
                "stage=relogin-variant outcome=already-authed",
                "stage=ui_session",
                "stage=webvpn_ticket",
            ],
        )
        self.assertEqual(STATS["webvpn_entries"], 1, "已登录变体一次进入即成功")

    def test_p65_stale_sso_ticket_recovers_via_fresh_retry(self):
        """P65 (b) 链路级：陈旧票据验证不过 → 清态全新 cookie jar 重试一次 → 成功。"""
        set_mode("sso_then_delivered")
        lines = []
        with self._endpoints():
            connector = PlatformSession(transport="requests")
            with patch(
                "courselens_worker.platform_session._emit_session_telemetry",
                side_effect=lines.append,
            ), patch("courselens_worker.platform_session.time.sleep"):
                connector._login_webvpn_full("account", "password", attempts=1)
        self.assertTrue(connector._webvpn_ready)
        # 子腿完成行随真链拆行：清态重试后 webvpn 腿全链 + 课程腿全链，各两条。
        self.assertEqual(
            [line.split(" elapsed=")[0] for line in lines],
            [
                "stage=relogin-variant outcome=fresh-retry",
                "stage=ui_session",
                "stage=webvpn_ticket",
                "stage=ui_session",
                "stage=webvpn_ticket",
            ],
        )
        self.assertEqual(
            STATS["webvpn_entries"], 2, "首进入验证败 + 清态后全新一次"
        )

    def test_challenge_at_course_raises_independent_code_without_retry(self):
        """夜10-C 挑战页变体：webvpn 腿成功后课程入口撞人机确认墙 → 独立码 +
        零重试（梯内 attempts=3 也只进入一次）；与 webvpn 腿 B3 同语义。"""
        set_mode("challenge_at_course")
        with self._endpoints():
            connector = PlatformSession(transport="requests")
            with self.assertRaises(PlatformSessionError) as captured:
                connector._login_webvpn_full("account", "password", attempts=3)
        self.assertEqual(str(captured.exception), "platform_challenge_required")
        self.assertEqual(captured.exception.connection_stage, "course_context")
        self.assertNotIn("platform_challenge_required", ps._RETRYABLE_LOGIN_ERRORS)
        self.assertNotIn(
            "platform_challenge_required", PlatformSession._RETRYABLE_WEBVPN_LEG_ERRORS
        )
        self.assertEqual(STATS["course_entries"], 1, "课程腿挑战页零重试")
        self.assertEqual(STATS["webvpn_entries"], 1, "挑战页不触发任何全链重跑")

    def test_direct_course_challenge_falls_back_to_webvpn_route(self):
        """夜10-C 补令②b 语义落定：直连腿撞课程入口挑战页 → 独立码驱动
        路线回落（换路线，非重试同一堵墙）→ webvpn 全梯接手；webvpn 路线
        的课程入口同撞挑战墙 → 仍按挑战码零重试诚实直败（B3 语义不变）。"""
        set_mode("challenge_at_course")
        with self._endpoints():
            connector = PlatformSession(transport="requests")
            with self.assertRaises(PlatformSessionError) as captured:
                connector.login("account", "password")
        self.assertEqual(str(captured.exception), "platform_challenge_required")
        self.assertEqual(captured.exception.connection_stage, "course_context")
        self.assertEqual(STATS["course_entries"], 2, "直连腿+webvpn 腿各撞一次墙")
        self.assertEqual(STATS["webvpn_entries"], 1, "挑战页驱动了路线回落")
        self.assertNotIn(
            "platform_challenge_required", ps._RETRYABLE_LOGIN_ERRORS,
            "挑战码依旧不入任何重试闭集（同一面墙零重试）",
        )
        self.assertNotIn(
            "platform_challenge_required", PlatformSession._RETRYABLE_WEBVPN_LEG_ERRORS
        )

    def test_midflow_course_expiry_recovers_via_full_relogin_ladder(self):
        """夜10-C 半途过期：课程会话过期落在普通登录页（无 lck）→ 维持既有
        可重试码 → 外层梯全链重跑（webvpn+course 全新票据，绝不重放）后恢复。"""
        set_mode("expire_then_delivered")
        codes = []
        connector = None
        with self._endpoints():
            for attempt in range(_MATERIALIZE_LOGIN_ATTEMPTS):
                connector = PlatformSession(transport="requests")
                try:
                    connector._login_webvpn_full("account", "password", attempts=1)
                    break
                except PlatformSessionError as exc:
                    codes.append(str(exc))
                    if (
                        str(exc) not in ps._RETRYABLE_LOGIN_ERRORS
                        or attempt == _MATERIALIZE_LOGIN_ATTEMPTS - 1
                    ):
                        raise
        self.assertEqual(codes, ["platform_course_context_missing"])
        self.assertTrue(connector._webvpn_ready)
        self.assertEqual(STATS["course_entries"], 2, "首跑过期 + 重跑各进入一次")
        self.assertEqual(STATS["webvpn_entries"], 2, "重跑含全新 webvpn 腿（零重放）")


if __name__ == "__main__":
    unittest.main()
