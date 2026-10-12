"""Audited platform session used only to authorize derived processing.

This is the only public Worker module allowed to authenticate, discover the
account's verified course catalog, or materialize short-lived media sources.
It exposes no original-media download, resume, archive, or persistence API.
"""

from __future__ import annotations

import base64
import hashlib
import html as html_module
import json
import os
import re
import threading
import time
import uuid
from binascii import hexlify
from datetime import date, datetime
from typing import Any, Callable
from urllib.parse import quote, unquote, urljoin, urlparse
from zoneinfo import ZoneInfo

import requests
from Crypto.Cipher import AES, PKCS1_v1_5
from Crypto.PublicKey import RSA
from curl_cffi import requests as curl_requests
from curl_cffi.requests.exceptions import RequestException as CurlRequestException


WEBVPN_BASE = "https://webvpn.fudan.edu.cn"
IDP_BASE = "https://id.fudan.edu.cn"
ICOURSE_BASE = "https://icourse.fudan.edu.cn"
TENANT_CODE = "222"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
_VPN_KEY = b"wrdvpnisthebest!"
_VPN_IV = b"wrdvpnisthebest!"
_ALLOWED_HOSTS = {
    "webvpn.fudan.edu.cn",
    "id.fudan.edu.cn",
    "icourse.fudan.edu.cn",
}
_REDIRECTS = {301, 302, 303, 307, 308}
_MAX_REDIRECTS = 8
# M15b（夜4 login-testbench §T5/T8 推荐束 B1+att3+B4；参数以常量+钉落地防无感漂移）：
# B1=context 族三码入登录重试闭集（此前「晨单发/日多发失败」主力类单请求硬死，零缓冲）；
# att3=materialize 外梯 5→3（退避 2·2^k 与实测 cfg30 同形，零改动，最坏登录代价 213-231s→126.6s）；
# B4=票据跟随读超时 12s→20s（慢 IDP ≥13s 从必死变可活）。
_MATERIALIZE_LOGIN_ATTEMPTS = 3
_TICKET_READ_TIMEOUT = (5, 20)
_RETRYABLE_LOGIN_ERRORS = frozenset({
    "platform_connection_failed",
    "platform_ticket_rejected",
    "platform_session_rejected",
    "platform_auth_context_missing",
    "platform_course_context_missing",
    "platform_ticket_missing",
})
_RETRYABLE_SESSION_ERRORS = frozenset({
    "platform_connection_failed",
    "platform_course_request_failed",
    "platform_session_rejected",
})


# 第四十一案（SESSION-RELOGIN-RETRY-1）：会话作废后的重取窗口必须覆盖
# 「学生重启客户端→重新登录」这段真实时长。旧窗为 3 次尝试、退避
# 2s+4s＝6s，分块字幕跑到中段（实测 16/27 块、elapsed=1571）一旦撞上
# 会话作废就整单失败——重试窗口短于重登录耗时，等于没有重试。
# 现窗＝8 次尝试，退避 2/4/8/16/30/30/30 合计 120s，再叠加每次重登录
# 自身的耗时；单次退避封顶 30s，保证窗口有界、不无限拖住运行时长。
# 非会话类错误（platform_media_missing、platform_challenge_required 等）
# 不在 _RETRYABLE_SESSION_ERRORS 内，仍然一次即败，不因本窗放大等待。
_SESSION_REFRESH_ATTEMPTS = 8
_SESSION_REFRESH_BACKOFF_CAP = 30.0

# P55（第五十五案）：中途续登此前是单发 lambda——真机事故 2026-09-24：字幕任务
# 第 117 秒块边界刷新授权触发续登，WebVPN 腿撞上跨境抖动没拿到 lck，
# platform_auth_context_missing 一发即死。启动材料化的 3 次登录梯只护启动链，
# 不护中途。中途续登与启动登录同族：同一重试闭集、同量次数、指数退避
# （封顶对齐 WebVPN 全梯 8s），每次失败发一行遥测（仅计数与闭集码，零敏感值）；
# 梯尽按原闭集码诚实失败，绝不吞错、不伪造成功。
_MIDTASK_RELOGIN_ATTEMPTS = 3
_MIDTASK_RELOGIN_BACKOFF_CAP = 8.0


def _emit_session_telemetry(line: str) -> None:
    # runner._progress discipline: counters and closed-set codes only —
    # never URLs, account values, headers, or provider text.
    print(line, flush=True)


def _bounded_midtask_relogin(relogin: Callable[[], None]) -> None:
    """One logical mid-task relogin: a bounded ladder over retryable login errors."""
    for attempt in range(_MIDTASK_RELOGIN_ATTEMPTS):
        try:
            relogin()
            return
        except PlatformSessionError as exc:
            if (
                str(exc) not in _RETRYABLE_LOGIN_ERRORS
                or attempt == _MIDTASK_RELOGIN_ATTEMPTS - 1
            ):
                _emit_session_telemetry(
                    f"stage=relogin attempt={attempt + 1} outcome=failed reason={exc}"
                )
                raise
            _emit_session_telemetry(
                f"stage=relogin attempt={attempt + 1} outcome=retry reason={exc}"
            )
            time.sleep(min(2.0 * (2 ** attempt), _MIDTASK_RELOGIN_BACKOFF_CAP))
    raise AssertionError("unreachable")


def _bounded_session_refresh(refresh, relogin=None, *, attempts: int = _SESSION_REFRESH_ATTEMPTS):
    """Retry one media-source refresh, re-authenticating between tries.

    A long decode can outlive the WebVPN/iCourse session, so the runner's
    periodic media re-authorization may fail with a closed-set session error.
    When a relogin callback is supplied, back off, re-authenticate, and retry
    the refresh; single-use URL and signed-timestamp rules are unaffected.
    """
    total = attempts if callable(relogin) else 1
    for attempt in range(total):
        try:
            return refresh()
        except PlatformSessionError as exc:
            if str(exc) not in _RETRYABLE_SESSION_ERRORS or attempt == total - 1:
                raise
            time.sleep(min(2.0 * 2 ** attempt, _SESSION_REFRESH_BACKOFF_CAP))
            relogin()
    raise AssertionError("unreachable")


_SLIDE_SCOPE_DOMAIN = "courselens-slide-deck-scope-v1"

# search-ppt 分页资源护栏（大预算熔断，不是页数上限）：单响应体大小、
# 服务器游标停滞与总事件密度三类闭集守卫取代旧的 50 页有效上限。
SLIDE_PAGE_SIZE = 100
SLIDE_RESPONSE_MAX_BYTES = 8 * 1024 * 1024
SLIDE_RECORD_STORM_LIMIT = 20000

# 平台原生文稿拉取护栏（AS12）：与 search-ppt 同量纲的响应体/记录密度熔断。
# 一节真实课程的文稿（约每 3 秒一条、几千条）远低于密度阈值；超界按闭集码
# 失败并由调用方回落旧链，绝不整段吞下不可信体量。
TRANSCRIPT_RESPONSE_MAX_BYTES = SLIDE_RESPONSE_MAX_BYTES
TRANSCRIPT_RECORD_STORM_LIMIT = SLIDE_RECORD_STORM_LIMIT


def _slide_deck_scope(course_id: str, sub_id: str) -> dict[str, str]:
    """Bounded nonsecret deck identity minted inside this adapter.

    The scope digest covers only the course/lecture pair, so slide entities
    and events stay stable across runs of the same lecture while raw
    account, URL, cookie, and course-title values never enter the job
    payload.  ``source_id`` follows the evidence.v1 source identity rules
    (a ``slide_deck`` source whose ``source_sha256`` is the scope digest).
    """
    from shared.evidence_contract import NAMESPACE_SOURCE, compute_id

    scope_sha256 = hashlib.sha256(
        f"{_SLIDE_SCOPE_DOMAIN}\0{course_id}\0{sub_id}".encode("utf-8")
    ).hexdigest()
    source_id = compute_id(NAMESPACE_SOURCE, {
        "kind": "slide_deck",
        "origin": "live_capture",
        "title": None,
        "duration_ms": None,
        "source_sha256": scope_sha256,
    })
    return {"deck_id": f"deck-{scope_sha256[:12]}", "source_id": source_id}


_CONNECTION_STAGES = frozenset({
    "webvpn_context",
    "webvpn_auth_methods",
    "webvpn_key",
    "webvpn_auth_execute",
    "webvpn_ticket",
    "webvpn_ticket_follow",
    "webvpn_verify",
    "course_context",
    "course_auth_methods",
    "course_key",
    "course_auth_execute",
    "course_ticket",
    "course_ticket_follow",
    "course_verify",
    "course_request",
    "course_context_direct",
    "course_auth_methods_direct",
    "course_key_direct",
    "course_auth_execute_direct",
    "course_ticket_direct",
    "course_ticket_follow_direct",
    "course_verify_direct",
    "course_request_direct",
})


class PlatformSessionError(RuntimeError):
    """Closed-set failure suitable for reduction in public logs."""

    def __init__(self, code: str, *, connection_stage: str = "") -> None:
        super().__init__(code)
        self.connection_stage = (
            str(connection_stage) if connection_stage in _CONNECTION_STAGES else ""
        )


def _fail(code: str, *, connection_stage: str = "") -> PlatformSessionError:
    return PlatformSessionError(code, connection_stage=connection_stage)


def _validate_url(value: str) -> str:
    try:
        parsed = urlparse(str(value or ""))
    except ValueError as exc:
        raise _fail("platform_redirect_rejected") from exc
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.hostname.lower() not in _ALLOWED_HOSTS
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise _fail("platform_redirect_rejected")
    return parsed.geturl()


def _validate_upstream_url(value: str) -> str:
    """Validate a platform-returned HTTPS asset before signing or VPN wrapping."""
    try:
        parsed = urlparse(str(value or ""))
    except ValueError as exc:
        raise _fail("platform_course_request_failed") from exc
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise _fail("platform_course_request_failed")
    return parsed.geturl()


def _vpn_url(value: str) -> str:
    parsed = urlparse(_validate_upstream_url(value))
    cipher = AES.new(_VPN_KEY, AES.MODE_CFB, _VPN_IV, segment_size=128)
    encrypted_host = hexlify(cipher.encrypt(str(parsed.hostname).encode("utf-8"))).decode("ascii")
    path = parsed.path.lstrip("/")
    if parsed.query:
        path += "?" + parsed.query
    output = f"{WEBVPN_BASE}/{parsed.scheme}/{hexlify(_VPN_IV).decode('ascii')}{encrypted_host}"
    return output + (f"/{path}" if path else "")


def _json(response: requests.Response, code: str) -> dict[str, Any]:
    try:
        value = response.json()
    except (ValueError, requests.RequestException) as exc:
        raise _fail(code) from exc
    if not isinstance(value, dict):
        raise _fail(code)
    return value


def _ticket_from_html(value: str, code: str) -> str:
    match = re.search(r'locationValue\s*=\s*"([^"]*ticket=[^"]*)"', value)
    if not match:
        match = re.search(r"(https?://[^\s\"'<>]*ticket=[^\s\"'<>]*)", value)
    if not match:
        raise _fail(code)
    return _validate_url(html_module.unescape(match.group(1)))


def _is_login_target(value: str) -> bool:
    try:
        path = urlparse(str(value or "")).path.rstrip("/").lower()
    except ValueError:
        return True
    return path == "/login" or path.endswith("/login")


# B3（夜4 login-testbench §T9 原型，60 样本 100% 分离）：风控挑战页识别器。
# 保守三特征（meta refresh / challenge 元素 / JS 写 cookie）；规则集对模拟器
# 页面特征拟合，真实 IDP 挑战页特征待 REALTEST 采样。只读有界前缀做分类，
# 正文永不进入日志或错误文本。
_CHALLENGE_SNIFF_BYTES = 65536
_CHALLENGE_PAGE_MARKERS = (
    '<meta http-equiv="refresh',
    "<meta http-equiv=refresh",
    'id="challenge"',
    "id=challenge",
    "document.cookie",
)


def _challenge_page_sniff(response: Any) -> str:
    """Return a bounded, case-folded body prefix for page classification."""
    try:
        content = bytes(response.content[:_CHALLENGE_SNIFF_BYTES])
    except Exception:
        return ""
    return content.decode("utf-8", errors="ignore").casefold()


def _looks_like_challenge_page(body: str) -> bool:
    return any(marker in body for marker in _CHALLENGE_PAGE_MARKERS)


def _looks_like_login_form(body: str, url: str = "") -> bool:
    """P65：登录表单形态判定，全部复用既有判据，不发明新词汇表。

    终页 URL 或页内表单 action 的落点命中 _is_login_target 即视为登录
    表单。SSO 已登录变体直通服务页，链上不会出现登录表单落点。
    """
    if _is_login_target(url):
        return True
    return any(
        _is_login_target(action)
        for action in re.findall(r"action\s*=\s*['\"]([^'\"]+)['\"]", body)
    )


# SMART-SCHED（2026-10-06）：月课表行结束时刻闭集解析。语义逐条对齐客户端
# icourse._parse_occurrence_span 的 end 源（src/api/icourse.py:229-277）：纪元
# 秒/毫秒（纪元界 1e9..4e9）、裸 HH:MM[:SS] 由 YYYY-MM-DD 锚定、ISO 日期时间
# （naive 补 Asia/Shanghai）；纯日期/缺失/不可解析一律不产出时刻=不确定行，
# 绝不猜测。行产物零标题、零 URL、零原始载荷。
GATE_TZ = ZoneInfo("Asia/Shanghai")
_END_SOURCES = ("end_at", "end_time", "ends_at")
_END_EPOCH_MIN = 1_000_000_000
_END_EPOCH_MAX = 4_000_000_000
_END_PURE_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?")
_END_DATE_ANCHOR_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _parse_end_evidence(row: dict[str, Any], day_anchor: str) -> datetime | None:
    """One closed-set end instant from a monthly row's end sources, or None.

    Sources are tried in closed-set order and the first parseable one wins,
    exactly like the client's occurrence-span consumption.  Anything missing,
    out-of-era, unanchored, or unparseable contributes no time evidence
    (None = uncertain row) and never a guess.
    """
    for source in _END_SOURCES:
        text = str(row.get(source) or "").strip()
        if not text:
            continue
        instant: datetime | None = None
        if text.isdigit():
            seconds = int(text)
            if seconds >= 1_000_000_000_000:
                seconds //= 1000
            if _END_EPOCH_MIN <= seconds <= _END_EPOCH_MAX:
                try:
                    instant = datetime.fromtimestamp(seconds, tz=GATE_TZ)
                except (OverflowError, OSError, ValueError):
                    instant = None
        else:
            match = _END_PURE_TIME_RE.fullmatch(text)
            if match is not None:
                anchor = str(day_anchor or "").strip()
                if _END_DATE_ANCHOR_RE.fullmatch(anchor):
                    try:
                        instant = datetime(
                            int(anchor[0:4]), int(anchor[5:7]), int(anchor[8:10]),
                            int(match.group(1)), int(match.group(2)),
                            int(match.group(3) or 0),
                            tzinfo=GATE_TZ,
                        )
                    except ValueError:
                        instant = None
            else:
                try:
                    parsed = datetime.fromisoformat(text)
                except ValueError:
                    parsed = None
                if parsed is not None and len(text) != 10:
                    # A bare date is not a precise end instant.
                    instant = (
                        parsed if parsed.tzinfo is not None
                        else parsed.replace(tzinfo=GATE_TZ)
                    )
        if instant is not None:
            return instant
    return None


class PlatformSession:
    """Narrow session capable only of authorizing one requested lecture."""

    CATALOG_TIMEOUT = (5, 15)
    CATALOG_DEADLINE_SECONDS = 90
    CATALOG_MAX_COURSES = 100
    CATALOG_MONTHS_BACK = 12
    CATALOG_MONTHS_AHEAD = 1
    CATALOG_MAX_ROWS_PER_MONTH = 1000
    # 本人续登回调：登录成功时登记，close() 释放。缺省 None 表示这个会话
    # 从未登录过，媒体重取退回「一次即败」语义（无凭据不重试）。
    _relogin: Callable[[], None] | None = None

    def __init__(self, *, transport: str = "curl") -> None:
        if transport not in {"curl", "requests"}:
            raise ValueError("unsupported platform transport")
        self._transport = transport
        self._userinfo: dict[str, Any] | None = None
        self._course_direct = False
        self._course_bearer = ""
        self._webvpn_ready = False
        self._build_sessions()

    def _build_sessions(self) -> None:
        """Fresh cookie jars; a half-written jar from a failed attempt never survives a retry."""
        if self._transport == "curl":
            self.session = curl_requests.Session(impersonate="chrome")
            self.course_session = curl_requests.Session(impersonate="chrome")
        else:
            self.session = requests.Session()
            self.course_session = requests.Session()
            self.session.trust_env = False
            self.course_session.trust_env = False
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.course_session.headers.update({"User-Agent": USER_AGENT})

    def _rebuild_sessions(self) -> None:
        try:
            self.course_session.close()
        finally:
            self.session.close()
        self._build_sessions()
    @staticmethod
    def _request_once(
        session: Any,
        method: str,
        url: str,
        *,
        connection_stage: str = "",
        **kwargs,
    ) -> requests.Response:
        target = _validate_url(url)
        kwargs["allow_redirects"] = False
        kwargs.setdefault("timeout", (10, 60))
        try:
            return session.request(method, target, **kwargs)
        except (requests.RequestException, CurlRequestException) as exc:
            raise _fail(
                "platform_connection_failed",
                connection_stage=connection_stage,
            ) from exc

    def _once(self, method: str, url: str, **kwargs) -> requests.Response:
        return self._request_once(self.session, method, url, **kwargs)

    def _direct_once(self, method: str, url: str, **kwargs) -> requests.Response:
        return self._request_once(self.course_session, method, url, **kwargs)

    def _follow(
        self,
        method: str,
        url: str,
        *,
        connection_stage: str = "",
        direct: bool = False,
        **kwargs,
    ) -> tuple[requests.Response, list[str]]:
        current = _validate_url(url)
        current_method = method.upper()
        trace = [current]
        for _ in range(_MAX_REDIRECTS + 1):
            request_once = self._direct_once if direct else self._once
            response = request_once(
                current_method,
                current,
                connection_stage=connection_stage,
                **kwargs,
            )
            status = int(response.status_code)
            if status not in _REDIRECTS:
                return response, trace
            location = response.headers.get("Location", "")
            response.close()
            if not location:
                raise _fail("platform_redirect_rejected")
            current = _validate_url(urljoin(current, location))
            trace.append(current)
            if status in {301, 302, 303} and current_method != "GET":
                current_method = "GET"
                kwargs.pop("data", None)
                kwargs.pop("json", None)
        raise _fail("platform_redirect_rejected")

    # Retryable WebVPN-leg failures.  Each retry reruns the complete flow —
    # fresh context, fresh lck, fresh loginToken, and a fresh single-use
    # ticket — because a consumed or dropped ticket can never be replayed.
    _RETRYABLE_WEBVPN_LEG_ERRORS = frozenset({
        "platform_connection_failed",
        "platform_ticket_rejected",
        "platform_session_rejected",
    })

    def _login_webvpn_full(self, account: str, password: str, *, attempts: int = 3) -> None:
        """Full WebVPN fallback with per-attempt fresh tickets and verification.

        The 2026-09-13 public-runner failure died at ``webvpn_ticket_follow``
        when a cross-border route dropped the streamed ticket response.  One
        transient drop must not fail the whole login: each bounded attempt
        reruns the full WebVPN login plus the course login behind it, ending
        in post-login session verification.  Tickets are minted per attempt
        and never replayed; every retry also starts from a freshly rebuilt
        session pair with exponential backoff, so a half-written cookie jar
        or a flaky route cannot poison the remaining attempts.
        """
        total = max(1, attempts)
        for attempt in range(total):
            try:
                self._login_webvpn(account, password)
                self._webvpn_ready = True
                self._login_course(account, password)
                return
            except PlatformSessionError as exc:
                self._webvpn_ready = False
                self._course_bearer = ""
                self._userinfo = None
                if (
                    str(exc) not in self._RETRYABLE_WEBVPN_LEG_ERRORS
                    or attempt == total - 1
                ):
                    raise
                self._rebuild_sessions()
                time.sleep(min(2.0 * (2 ** attempt), 8.0))
        raise AssertionError("unreachable")

    def login(
        self, account: str, password: str, *, allow_webvpn_fallback: bool = True
    ) -> None:
        if not account or not password:
            raise _fail("platform_credentials_missing")
        try:
            try:
                self._login_course_direct(account, password)
            except PlatformSessionError as exc:
                if str(exc) not in {
                    "platform_connection_failed",
                    "platform_course_context_missing",
                    "platform_ticket_missing",
                    "platform_ticket_rejected",
                    "platform_session_rejected",
                    # 夜10-C：挑战页属路线级风控而非账号级墙——直连腿撞墙后
                    # 换 webvpn 路线是学生可感知的可达性收益；回落是换路线，
                    # 不是对同一挑战墙重试（webvpn 腿自己的挑战墙仍零重试，
                    # B3 语义不变）。
                    "platform_challenge_required",
                }:
                    raise
                if not allow_webvpn_fallback:
                    raise
                self._course_direct = False
                self._course_bearer = ""
                self._login_webvpn_full(account, password)
        except PlatformSessionError:
            raise
        except Exception as exc:
            raise _fail("platform_auth_failed") from exc
        # 第四十一案：登录成功即登记本人续登回调。分块字幕跑到中段时，
        # 学生重启客户端并重新登录会让本会话被服务端作废，而调用方
        # （每日自动化链）并不持有口令可自行重登录；媒体重取撞上会话失效
        # 时按 _SESSION_REFRESH_* 窗口自动续登重试。口令只活在闭包里，
        # 不写日志、不进产物，close() 时随会话一起释放。
        # P55：登记面自带三梯——中途续登与启动登录同族，撞上瞬态失效
        # （跨境抖动、单会话作废）先按梯重试，梯尽才按闭集码诚实失败。
        self._relogin = lambda: _bounded_midtask_relogin(
            lambda: self.login(account, password)
        )

    @staticmethod
    def _encrypt_password(password: str, public_key: str) -> str:
        try:
            pem = "-----BEGIN PUBLIC KEY-----\n" + public_key + "\n-----END PUBLIC KEY-----"
            encrypted = PKCS1_v1_5.new(RSA.import_key(pem)).encrypt(password.encode("utf-8"))
            return base64.b64encode(encrypted).decode("ascii")
        except (ValueError, IndexError, TypeError) as exc:
            raise _fail("platform_key_rejected") from exc

    @staticmethod
    def _auth_method(data: dict[str, Any]) -> tuple[str, str]:
        for method in data.get("data") or []:
            if isinstance(method, dict) and method.get("moduleCode") == "userAndPwd":
                code = str(method.get("authChainCode") or "")
                if code:
                    return code, str(data.get("requestType") or "chain_type")
        raise _fail("platform_auth_method_missing")

    def _login_webvpn(
        self, account: str, password: str, *, _fresh_retry: bool = False
    ) -> None:
        started = time.monotonic()
        service = f"{WEBVPN_BASE}/login?cas_login=true"
        current = f"{IDP_BASE}/idp/authCenter/authenticate?service={quote(service, safe='')}"
        lck = ""
        final_body = ""
        final_url = ""
        for _ in range(_MAX_REDIRECTS + 1):
            response = self._once(
                "GET", current, connection_stage="webvpn_context"
            )
            location = response.headers.get("Location", "")
            status = response.status_code
            match = re.search(r"[?&]lck=([^&]+)", location)
            if match:
                response.close()
                lck = match.group(1)
                break
            if status not in _REDIRECTS or not location:
                # 终止页（通常 200 正文页）：读有界前缀供挑战页分类。
                final_body = _challenge_page_sniff(response)
                final_url = str(response.url or "")
                response.close()
                break
            response.close()
            current = _validate_url(urljoin(current, location))
        if not lck:
            # B3：挑战页与「IDP 未发 lck」塌缩解耦。挑战页需要人工完成一次
            # 人机确认，独立码且不入任何重试闭集——程序重试只会连续撞上
            # 同一挑战页；普通无 lck 页维持既有码（外层登录梯会重试）。
            if _looks_like_challenge_page(final_body):
                raise _fail("platform_challenge_required", connection_stage="webvpn_context")
            # P65（第六十五案）：IDP「已登录」变体。runner 启动时已登录过一次，
            # IDP 的 SSO 会话（TGT cookie，随 self.session 存活）中途仍有效；
            # 会话失效触发重登时 IDP 跳过登录表单直接发票，整条重定向链不出现
            # lck（lck 只在出新登录表单时发放）——2026-09-24 真机 chunk 3
            # 九连败的确定性根源。终页非挑战页且非登录表单时，先验证会话
            # 实际可用性（票据超时路径「宁可验证不可重放」同款判据）：可用
            # 即按 webvpn 腿成功语义返回，外层照常走课程腿并登记续登回调。
            if not _looks_like_login_form(final_body, final_url):
                if self._verify_webvpn_bounded():
                    _emit_session_telemetry("stage=relogin-variant outcome=already-authed")
                    return
                if not _fresh_retry:
                    _emit_session_telemetry("stage=relogin-variant outcome=fresh-retry")
                    # 干净重试：陈旧 SSO/TGT cookie 会把下一条链再次引离新
                    # 登录表单。重建全新 cookie jar（=强制全新登录流）后，
                    # 同一次 attempt 内按全新链路再试一次；挑战页分类仍在前，
                    # 梯尽仍按既有闭集码诚实失败。
                    self._webvpn_ready = False
                    self._course_bearer = ""
                    self._userinfo = None
                    self._rebuild_sessions()
                    self._login_webvpn(account, password, _fresh_retry=True)
                    return
            raise _fail("platform_auth_context_missing")

        method_data = _json(self._once(
            "POST", f"{IDP_BASE}/idp/authn/queryAuthMethods",
            json={"lck": lck, "entityId": WEBVPN_BASE},
            headers={"Content-Type": "application/json", "Referer": f"{IDP_BASE}/ac/", "Origin": IDP_BASE},
            connection_stage="webvpn_auth_methods",
        ), "platform_auth_method_missing")
        chain, request_type = self._auth_method(method_data)
        key_data = _json(self._once(
            "GET", f"{IDP_BASE}/idp/authn/getJsPublicKey",
            headers={"Referer": f"{IDP_BASE}/ac/"},
            connection_stage="webvpn_key",
        ), "platform_key_rejected")
        encrypted = self._encrypt_password(password, str(key_data.get("data") or ""))
        auth_data = _json(self._once(
            "POST", f"{IDP_BASE}/idp/authn/authExecute",
            json={
                "authModuleCode": "userAndPwd", "authChainCode": chain,
                "entityId": WEBVPN_BASE, "requestType": request_type, "lck": lck,
                "authPara": {"loginName": account, "password": encrypted, "verifyCode": ""},
            },
            headers={"Content-Type": "application/json", "Referer": f"{IDP_BASE}/ac/", "Origin": IDP_BASE},
            connection_stage="webvpn_auth_execute",
        ), "platform_auth_failed")
        if str(auth_data.get("code")) != "200" or not auth_data.get("loginToken"):
            raise _fail("platform_auth_failed")
        # DIAG 提案 2 口径（cloudverifyfix 恰域注记的后续一行级建议）：探针四
        # 腿中 ui_session/webvpn_ticket 两子腿的完成事件在登录链内部——
        # authExecute 成功位与票据跟单验证完成位各发一行闭集 stage 行（仅腿
        # 名+秒数，零敏感面），不持 GitHub 鉴权即可从 run 行直读腿位；三条登
        # 链（course-direct/webvpn/behind-webvpn）同口径拆行。
        _emit_session_telemetry(
            f"stage=ui_session elapsed={time.monotonic() - started:.1f}"
        )
        ticket_response = self._once(
            "POST", f"{IDP_BASE}/idp/authCenter/authnEngine",
            data={"loginToken": str(auth_data["loginToken"])},
            headers={"Referer": f"{IDP_BASE}/ac/", "Origin": IDP_BASE},
            connection_stage="webvpn_ticket",
        )
        ticket = _ticket_from_html(ticket_response.text[:256 * 1024], "platform_ticket_missing")
        ticket_response.close()
        try:
            response, _ = self._follow(
                "GET",
                ticket,
                stream=True,
                timeout=_TICKET_READ_TIMEOUT,
                connection_stage="webvpn_ticket_follow",
            )
        except PlatformSessionError as exc:
            # The portal can set its cookie before a streamed ticket response
            # times out. Never replay the single-use ticket; verify instead.
            # Cross-border runner routes drop connections intermittently, so
            # re-verify a few times before declaring the session rejected.
            if str(exc) != "platform_connection_failed" or not self._verify_webvpn_bounded():
                raise
        else:
            try:
                if not 200 <= response.status_code < 300 or _is_login_target(response.url):
                    raise _fail("platform_ticket_rejected")
            finally:
                response.close()
        if not self._verify_webvpn():
            raise _fail("platform_session_rejected", connection_stage="webvpn_verify")
        _emit_session_telemetry(
            f"stage=webvpn_ticket elapsed={time.monotonic() - started:.1f}"
        )

    @staticmethod
    def _bounded_reverify(verify: Callable[[], bool], attempts: int = 3, delay: float = 2.0) -> bool:
        for attempt in range(attempts):
            if attempt:
                time.sleep(delay)
            try:
                if verify():
                    return True
            except Exception:
                pass
        return False

    def _verify_webvpn_bounded(self) -> bool:
        return self._bounded_reverify(self._verify_webvpn)

    def _verify_webvpn(self) -> bool:
        response = None
        try:
            response = self._once(
                "GET",
                WEBVPN_BASE + "/",
                stream=True,
                timeout=(3, 8),
                connection_stage="webvpn_verify",
            )
            location = response.headers.get("Location", "")
            return (
                response.status_code == 200
                and not _is_login_target(response.url)
                and not _is_login_target(urljoin(response.url, location))
            )
        except PlatformSessionError:
            return False
        finally:
            if response is not None:
                response.close()

    @staticmethod
    def _cookie_value(session: Any, name: str) -> str:
        try:
            items = session.cookies.items()
        except (AttributeError, TypeError):
            return ""
        for key, value in items:
            if str(key) == name:
                return str(value or "")
        return ""

    def _extract_course_bearer(self, session: Any | None = None) -> str:
        pattern = re.compile(r'\{i:\d+;s:\d+:"_token";i:\d+;s:\d+:"(.+?)";\}')
        target = session or self.course_session
        try:
            values = [str(value or "") for _name, value in target.cookies.items()]
        except (AttributeError, TypeError):
            values = []
        for value in values:
            encoded = value
            for _ in range(3):
                decoded = unquote(encoded)
                match = pattern.search(decoded)
                if match is not None:
                    token = match.group(1)
                    if 16 <= len(token) <= 4096 and "\r" not in token and "\n" not in token:
                        return token
                if decoded == encoded:
                    break
                encoded = decoded
        raise _fail(
            "platform_session_rejected",
            connection_stage="course_ticket_follow_direct",
        )

    def _optional_course_bearer(self, session: Any) -> str:
        """Client-aligned optional bearer scan for the WebVPN path.

        The wengine gateway maps origin cookies server-side, so the client-side
        WebVPN jar carries no serialized ``_token`` value in practice (client
        bounded probe 2026-09-07: catalog index/detail succeed with no
        Authorization header and zero _token cookies in the jar).  Attach the
        bearer when a legacy carrier happens to exist; omit otherwise and let
        the gateway session cookie — already verified by the login's own
        ``_verify_course`` — carry identity.  Only the direct session keeps the
        strict extraction (the bearer is its sole credential there).
        """
        try:
            return self._extract_course_bearer(session)
        except PlatformSessionError:
            return ""

    def _login_course_direct(self, account: str, password: str) -> None:
        started = time.monotonic()
        cas = (
            f"{ICOURSE_BASE}/casapi/index.php?r=auth/login&school_login=1"
            f"&tenant_code={TENANT_CODE}&forward={quote(ICOURSE_BASE + '/', safe='')}"
        )
        response, trace = self._follow(
            "GET", cas, connection_stage="course_context_direct", direct=True
        )
        lck = ""
        body_prefix = ""
        try:
            for candidate in trace + [response.url]:
                match = re.search(r'lck=([^&#"\s]+)', str(candidate or ""))
                if match:
                    lck = match.group(1)
                    break
            if not lck:
                body_prefix = _challenge_page_sniff(response)
                match = re.search(r'lck=([^&#"\s]+)', response.text[:5000])
                if match:
                    lck = match.group(1)
        finally:
            response.close()
        if not lck:
            # 夜10-C 直连课程腿挑战页解耦（上侧 wrapped 腿同款）。
            if _looks_like_challenge_page(body_prefix):
                raise _fail("platform_challenge_required", connection_stage="course_context_direct")
            raise _fail("platform_course_context_missing")

        method_data = _json(self._direct_once(
            "POST", f"{IDP_BASE}/idp/authn/queryAuthMethods",
            json={"lck": lck, "entityId": ICOURSE_BASE},
            headers={"Content-Type": "application/json", "Referer": f"{IDP_BASE}/ac/", "Origin": IDP_BASE},
            connection_stage="course_auth_methods_direct",
        ), "platform_auth_method_missing")
        chain, request_type = self._auth_method(method_data)
        key_data = _json(self._direct_once(
            "GET", f"{IDP_BASE}/idp/authn/getJsPublicKey",
            headers={"Referer": f"{IDP_BASE}/ac/"},
            connection_stage="course_key_direct",
        ), "platform_key_rejected")
        encrypted = self._encrypt_password(password, str(key_data.get("data") or ""))
        auth_data = _json(self._direct_once(
            "POST", f"{IDP_BASE}/idp/authn/authExecute",
            json={
                "authModuleCode": "userAndPwd", "authChainCode": chain,
                "entityId": ICOURSE_BASE, "requestType": request_type, "lck": lck,
                "authPara": {"loginName": account, "password": encrypted, "verifyCode": ""},
            },
            headers={"Content-Type": "application/json", "Referer": f"{IDP_BASE}/ac/", "Origin": IDP_BASE},
            connection_stage="course_auth_execute_direct",
        ), "platform_auth_failed")
        if str(auth_data.get("code")) != "200" or not auth_data.get("loginToken"):
            raise _fail("platform_auth_failed")
        _emit_session_telemetry(
            f"stage=ui_session elapsed={time.monotonic() - started:.1f}"
        )
        ticket_response = self._direct_once(
            "POST", f"{IDP_BASE}/idp/authCenter/authnEngine",
            data={"loginToken": str(auth_data["loginToken"])},
            headers={"Referer": f"{IDP_BASE}/ac/", "Origin": IDP_BASE},
            connection_stage="course_ticket_direct",
        )
        ticket = _ticket_from_html(ticket_response.text[:256 * 1024], "platform_ticket_missing")
        ticket_response.close()
        if urlparse(ticket).hostname != urlparse(ICOURSE_BASE).hostname:
            raise _fail("platform_ticket_rejected")
        try:
            response, _ = self._follow(
                "GET", ticket, stream=True, timeout=_TICKET_READ_TIMEOUT,
                connection_stage="course_ticket_follow_direct", direct=True,
            )
        except PlatformSessionError as exc:
            if str(exc) != "platform_connection_failed":
                raise
        else:
            try:
                if not 200 <= response.status_code < 300 or _is_login_target(response.url):
                    raise _fail("platform_ticket_rejected")
            finally:
                response.close()
        self._course_bearer = self._extract_course_bearer()
        if not self._verify_course_direct():
            self._course_bearer = ""
            raise _fail(
                "platform_session_rejected",
                connection_stage="course_verify_direct",
            )
        self._course_direct = True
        _emit_session_telemetry(
            f"stage=webvpn_ticket elapsed={time.monotonic() - started:.1f}"
        )

    def _verify_course_direct(self) -> bool:
        if not self._course_bearer:
            return False
        response = None
        try:
            response = self._direct_once(
                "GET", f"{ICOURSE_BASE}/userapi/v1/infosimple",
                headers={"Authorization": f"Bearer {self._course_bearer}"},
                timeout=(3, 8), connection_stage="course_verify_direct",
            )
            data = _json(response, "platform_session_rejected")
            return response.status_code == 200 and data.get("code") in (0, 200)
        except PlatformSessionError:
            return False
        finally:
            if response is not None:
                response.close()

    def _login_course(self, account: str, password: str) -> None:
        started = time.monotonic()
        cas = (
            f"{ICOURSE_BASE}/casapi/index.php?r=auth/login&school_login=1"
            f"&tenant_code={TENANT_CODE}&forward={quote(ICOURSE_BASE + '/', safe='')}"
        )
        response, trace = self._follow(
            "GET", _vpn_url(cas), connection_stage="course_context"
        )
        lck = ""
        body_prefix = ""
        try:
            for candidate in trace + [response.url]:
                match = re.search(r'lck=([^&#"]+)', candidate)
                if match:
                    lck = match.group(1)
                    break
            if not lck:
                body_prefix = _challenge_page_sniff(response)
                match = re.search(r'lck=([^&#"]+)', response.text[:5000])
                if match:
                    lck = match.group(1)
        finally:
            response.close()
        if not lck:
            # 夜10-C 课程腿挑战页变体（webvpn 腿 B3 同族解耦）：挑战页需要
            # 人工完成一次人机确认，独立码且不入任何重试闭集——课程腿此前
            # 塌缩成 platform_course_context_missing（登录重试闭集内），梯内
            # 重试只会连续撞上同一挑战页。普通无 lck 页维持既有码（外层梯
            # 全链重跑后可恢复）。
            if _looks_like_challenge_page(body_prefix):
                raise _fail("platform_challenge_required", connection_stage="course_context")
            raise _fail("platform_course_context_missing")

        idp_vpn = _vpn_url(IDP_BASE)
        method_data = _json(self._once(
            "POST", _vpn_url(f"{IDP_BASE}/idp/authn/queryAuthMethods"),
            json={"lck": lck, "entityId": ICOURSE_BASE},
            headers={"Content-Type": "application/json", "Referer": f"{idp_vpn}/ac/", "Origin": WEBVPN_BASE},
            connection_stage="course_auth_methods",
        ), "platform_auth_method_missing")
        chain, request_type = self._auth_method(method_data)
        key_data = _json(self._once(
            "GET", _vpn_url(f"{IDP_BASE}/idp/authn/getJsPublicKey"),
            headers={"Referer": f"{idp_vpn}/ac/"},
            connection_stage="course_key",
        ), "platform_key_rejected")
        encrypted = self._encrypt_password(password, str(key_data.get("data") or ""))
        auth_data = _json(self._once(
            "POST", _vpn_url(f"{IDP_BASE}/idp/authn/authExecute"),
            json={
                "authModuleCode": "userAndPwd", "authChainCode": chain,
                "entityId": ICOURSE_BASE, "requestType": request_type, "lck": lck,
                "authPara": {"loginName": account, "password": encrypted, "verifyCode": ""},
            },
            headers={"Content-Type": "application/json", "Referer": f"{idp_vpn}/ac/", "Origin": WEBVPN_BASE},
            connection_stage="course_auth_execute",
        ), "platform_auth_failed")
        if str(auth_data.get("code")) != "200" or not auth_data.get("loginToken"):
            raise _fail("platform_auth_failed")
        _emit_session_telemetry(
            f"stage=ui_session elapsed={time.monotonic() - started:.1f}"
        )
        ticket_response = self._once(
            "POST", _vpn_url(f"{IDP_BASE}/idp/authCenter/authnEngine"),
            data={"loginToken": str(auth_data["loginToken"])},
            headers={"Referer": f"{idp_vpn}/ac/", "Origin": WEBVPN_BASE},
            connection_stage="course_ticket",
        )
        ticket = _ticket_from_html(ticket_response.text[:256 * 1024], "platform_ticket_missing")
        ticket_response.close()
        if urlparse(ticket).hostname != urlparse(WEBVPN_BASE).hostname:
            ticket = _vpn_url(ticket)
        try:
            response, _ = self._follow(
                "GET",
                ticket,
                stream=True,
                timeout=_TICKET_READ_TIMEOUT,
                connection_stage="course_ticket_follow",
            )
        except PlatformSessionError as exc:
            if str(exc) != "platform_connection_failed" or not self._bounded_reverify(self._verify_course):
                raise
        else:
            try:
                if not 200 <= response.status_code < 300 or _is_login_target(response.url):
                    raise _fail("platform_ticket_rejected")
            finally:
                response.close()
        if not self._verify_course():
            raise _fail("platform_session_rejected", connection_stage="course_verify")
        _emit_session_telemetry(
            f"stage=webvpn_ticket elapsed={time.monotonic() - started:.1f}"
        )

    def _verify_course(self) -> bool:
        response = None
        try:
            response = self._once(
                "GET",
                _vpn_url(f"{ICOURSE_BASE}/userapi/v1/infosimple"),
                timeout=(3, 8),
                connection_stage="course_verify",
            )
            data = _json(response, "platform_session_rejected")
            return response.status_code == 200 and data.get("code") in (0, 200)
        except PlatformSessionError:
            return False
        finally:
            if response is not None:
                response.close()

    def _course_json(
        self,
        path: str,
        *,
        params: dict[str, Any],
        authorization_required: bool = False,
        timeout: tuple[float, float] | None = None,
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        request_timeout = timeout or (10, 60)
        if self._course_direct:
            response = self._direct_once(
                "GET", ICOURSE_BASE + path, params=params,
                headers={"Authorization": f"Bearer {self._course_bearer}"},
                timeout=request_timeout,
                connection_stage="course_request_direct",
            )
        else:
            headers = None
            if authorization_required:
                # 与客户端 catalog 同法（src/api/icourse.py:919-931，2026-09-07
                # 有界探针实证 webvpn 会话 cookie 足以访问目录且罐内零 _token
                # 载体）：bearer 载体在罐时照样附带，缺席时省略 Authorization
                # 头走已验证的 webvpn 会话 cookie。runner 的 schedule 触发形态
                # 下罐内恒无载体，旧「懒提取硬失败」曾把门腿与发现腿双双打成
                # platform_session_rejected 的 course_ticket_follow_direct 塌缩
                # 码（2026-10-07 run 37557294406 闭集定谳，verify 链从不发
                # authorization_required 请求故恒 PASS 的形态差）。direct 会话
                # 仍强制 bearer（该路径唯一凭证）。
                bearer = self._optional_course_bearer(self.session)
                if bearer:
                    headers = {"Authorization": f"Bearer {bearer}"}
            response = self._once(
                "GET", _vpn_url(ICOURSE_BASE + path), params=params,
                headers=headers,
                timeout=request_timeout,
                connection_stage="course_request",
            )
        try:
            if response.status_code != 200:
                raise _fail("platform_course_request_failed")
            if max_bytes is not None and len(response.content) > max_bytes:
                raise _fail("platform_slide_response_too_large")
            return _json(response, "platform_course_request_failed")
        finally:
            response.close()

    def _userinfo_value(
        self, *, timeout: tuple[float, float] | None = None
    ) -> dict[str, Any]:
        if self._userinfo is None:
            data = self._course_json(
                "/userapi/v1/infosimple", params={}, timeout=timeout
            )
            if data.get("code") not in (0, 200):
                raise _fail("platform_course_request_failed")
            self._userinfo = dict(data.get("params") or data.get("data") or {})
        return self._userinfo

    @staticmethod
    def _lecture_rows(course_data: dict[str, Any]) -> list[dict[str, Any]]:
        lectures: list[dict[str, Any]] = []
        for year, months in dict(course_data.get("sub_list") or {}).items():
            for month, days in dict(months or {}).items():
                for day, items in dict(days or {}).items():
                    for item in list(items or []):
                        if not isinstance(item, dict) or not item.get("id"):
                            continue
                        date = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
                        lectures.append({
                            "sub_id": str(item["id"]),
                            "sub_title": str(item.get("sub_title") or ""),
                            "lecturer_name": str(item.get("lecturer_name") or ""),
                            "date": date,
                            "has_playback": str(item.get("playback_status") or "") == "1",
                            "duration_seconds": max(0, int(item.get("duration") or item.get("duration_sec") or 0)),
                        })
        return lectures

    def course_detail(
        self,
        course_id: str,
        *,
        timeout: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        data = self._course_json(
            "/courseapi/v3/multi-search/get-course-detail",
            params={"course_id": str(course_id)},
            timeout=timeout,
        )
        if data.get("code") not in (0, 200):
            raise _fail("platform_course_request_failed")
        raw = dict(data.get("data") or {})
        return {
            "course_id": str(course_id),
            "title": str(raw.get("title") or ""),
            "teacher": str(raw.get("realname") or raw.get("teacher") or ""),
            "lectures": self._lecture_rows(raw),
        }

    def _user_courses(
        self, *, today: date | None = None, deadline: float | None = None
    ) -> list[dict[str, Any]]:
        end = deadline or (time.monotonic() + self.CATALOG_DEADLINE_SECONDS)
        remaining = end - time.monotonic()
        if remaining <= 0:
            raise _fail("platform_course_request_failed")
        user = self._userinfo_value(
            timeout=(self.CATALOG_TIMEOUT[0], min(self.CATALOG_TIMEOUT[1], remaining))
        )
        if not user.get("id") or not str(user.get("account") or "").strip() or not (
            user.get("tenant_id") or TENANT_CODE
        ):
            raise _fail("platform_course_request_failed")
        anchor = today or date.today()
        month_index = anchor.year * 12 + anchor.month - 1
        months = [
            f"{value // 12:04d}-{value % 12 + 1:02d}"
            for value in range(
                month_index - self.CATALOG_MONTHS_BACK,
                month_index + self.CATALOG_MONTHS_AHEAD + 1,
            )
        ]
        courses: list[dict[str, Any]] = []
        by_course: dict[str, dict[str, Any]] = {}
        for month in months:
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise _fail("platform_course_request_failed")
            data = self._course_json(
                "/courseapi/v2/course-live/get-my-course-month",
                params={"month": month},
                authorization_required=True,
                timeout=(self.CATALOG_TIMEOUT[0], min(self.CATALOG_TIMEOUT[1], remaining)),
            )
            if data.get("code") not in (0, 200):
                raise _fail("platform_course_request_failed")
            days = data.get("list") or []
            if not isinstance(days, list) or len(days) > 31:
                raise _fail("platform_course_request_failed")
            rows = [
                row
                for day_value in days
                if isinstance(day_value, dict)
                for row in day_value.get("course") or []
                if isinstance(row, dict)
            ]
            if len(rows) > self.CATALOG_MAX_ROWS_PER_MONTH:
                raise _fail("platform_course_request_failed")
            for raw in rows:
                course_id = str(raw.get("id") or raw.get("course_id") or "").strip()
                if not course_id:
                    continue
                if course_id not in by_course:
                    if len(courses) >= self.CATALOG_MAX_COURSES:
                        raise _fail("platform_course_request_failed")
                    candidate = {
                        "course_id": course_id,
                        "title": str(raw.get("title") or ""),
                        "teacher": str(raw.get("realname") or raw.get("lecturer_name") or ""),
                        "department": str(
                            raw.get("kkxy_name") or raw.get("school_name")
                            or raw.get("dept_name") or raw.get("kkxy") or ""
                        ),
                        "term": str(raw.get("term_name") or raw.get("term") or ""),
                        "_lecture_durations": {},
                    }
                    by_course[course_id] = candidate
                    courses.append(candidate)
                sub_id = str(raw.get("sub_id") or "").strip()
                try:
                    duration = max(0.0, float(raw.get("sub_duration") or 0.0))
                except (TypeError, ValueError):
                    duration = 0.0
                if sub_id and duration > 0:
                    by_course[course_id]["_lecture_durations"][sub_id] = duration
        return courses

    def today_schedule_rows(self, *, today: date | None = None) -> list[dict[str, Any]]:
        """Today's monthly-schedule rows as closed-set gate evidence.

        SMART-SCHED 智能门数据腿：恰 1 个授权 GET（今日月），行产物仅
        ``{course_id, sub_id, date, end_precise}``——零标题、零 URL、零原始
        载荷。``end_precise`` 为 Asia/Shanghai 感知时刻或 None（不确定行，
        由门层 fail-open）。请求失败按既有闭集码
        ``platform_course_request_failed`` 上抛，由调用方转 fail-open。
        """
        anchor = today or datetime.now(GATE_TZ).date()
        data = self._course_json(
            "/courseapi/v2/course-live/get-my-course-month",
            params={"month": f"{anchor.year:04d}-{anchor.month:02d}"},
            authorization_required=True,
            timeout=(10, 30),
        )
        if data.get("code") not in (0, 200):
            raise _fail("platform_course_request_failed")
        days = data.get("list") or []
        if not isinstance(days, list) or len(days) > 31:
            raise _fail("platform_course_request_failed")
        rows: list[dict[str, Any]] = []
        for day_value in days:
            if not isinstance(day_value, dict):
                continue
            day_date = str(day_value.get("date") or "").strip()
            for row in day_value.get("course") or []:
                if not isinstance(row, dict):
                    continue
                course_id = str(row.get("id") or row.get("course_id") or "").strip()
                if not course_id:
                    continue
                # Date anchor mirrors the client's occurrence-anchor order
                # (row date first, then the parent day); no title-derived
                # fallback — an unresolvable anchor stays an uncertain row.
                row_date = str(row.get("date") or "").strip()
                anchor_text = (
                    row_date if _END_DATE_ANCHOR_RE.fullmatch(row_date) else day_date
                )
                if not _END_DATE_ANCHOR_RE.fullmatch(anchor_text):
                    anchor_text = ""
                rows.append({
                    "course_id": course_id,
                    "sub_id": str(row.get("sub_id") or "").strip(),
                    "date": anchor_text,
                    "end_precise": _parse_end_evidence(row, anchor_text),
                })
                if len(rows) > self.CATALOG_MAX_ROWS_PER_MONTH:
                    raise _fail("platform_course_request_failed")
        return rows

    def discover_authorized_courses(self) -> list[dict[str, Any]]:
        """Verify courses from the identity-scoped personal schedule only."""
        deadline = time.monotonic() + self.CATALOG_DEADLINE_SECONDS
        candidates = self._user_courses(deadline=deadline)
        output: list[dict[str, Any]] = []
        failures = 0
        for candidate in candidates:
            course_id = str(candidate.get("course_id") or "")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _fail("platform_course_request_failed")
            try:
                detail = self.course_detail(
                    course_id,
                    timeout=(
                        self.CATALOG_TIMEOUT[0],
                        min(self.CATALOG_TIMEOUT[1], remaining),
                    ),
                )
            except PlatformSessionError:
                failures += 1
                continue
            durations = dict(candidate.get("_lecture_durations") or {})
            detail["lectures"] = [
                {
                    **lecture,
                    **(
                        {"duration_seconds": durations[str(lecture.get("sub_id"))]}
                        if str(lecture.get("sub_id")) in durations else {}
                    ),
                }
                for lecture in detail.get("lectures") or []
            ]
            detail.update({
                "term": str(candidate.get("term") or ""),
                "department": str(candidate.get("department") or ""),
                "authorization_state": "verified",
            })
            output.append(detail)
        if candidates and not output and failures:
            raise _fail("platform_course_request_failed")
        return output

    def _sign(self, media_url: str, now: int | None) -> str:
        info = self._userinfo_value()
        timestamp = int(now or time.time())
        media_url = _validate_upstream_url(media_url)
        path = urlparse(media_url).path
        raw = f"{path}{info.get('id', '')}{info.get('tenant_id', '')}{str(info.get('phone', ''))[::-1]}{timestamp}"
        token = f"{info.get('id', '')}-{timestamp}-{hashlib.md5(raw.encode()).hexdigest()}"
        separator = "&" if "?" in media_url else "?"
        return f"{media_url}{separator}clientUUID={uuid.uuid4()}&t={token}"

    def _media_base(self, course_id: str, sub_id: str) -> tuple[str, int]:
        data = self._course_json(
            "/courseapi/v3/portal-home-setting/get-sub-info",
            params={"course_id": course_id, "sub_id": sub_id},
        )
        info = dict(data.get("data") or {})
        base = ""
        for item in (info.get("video_list") or {}).values():
            candidate = str(item.get("preview_url") or "") if isinstance(item, dict) else ""
            if candidate and urlparse(candidate).path.lower().endswith(".mp4"):
                base = candidate
                break
        if not base:
            for key, candidate in (info.get("playurl") or {}).items():
                if key != "now" and isinstance(candidate, str) and urlparse(candidate).path.lower().endswith(".mp4"):
                    base = candidate
                    break
        if not base:
            base = str(((info.get("content") or {}).get("playback") or {}).get("url") or "")
        if not base:
            detail = self._course_json(
                "/courseapi/v3/multi-search/get-sub-detail",
                params={"course_id": course_id, "sub_id": sub_id},
            )
            base = str((((detail.get("data") or {}).get("content") or {}).get("playback") or {}).get("url") or "")
        if not base or not urlparse(base).path.lower().endswith(".mp4"):
            raise _fail("platform_media_missing")
        return base, int(info.get("now") or 0)

    def media_source(self, course_id: str, sub_id: str, *, relogin=None) -> dict[str, Any]:
        # A signed CDN URL may authorize only one media request. The desktop
        # client already obtains a new URL per browser Range; expose the same
        # behavior to the runner's in-memory proxy without retaining account
        # credentials. The platform's base preview URL is also short-lived, so
        # every refresh must obtain a new base before applying the CDN signature.
        from .source import SourceSecurityError, resolve_source_address

        # 第四十一案：调用方没显式给续登回调时，用会话登录时登记的本人回调。
        # 每日自动化链不持有口令，但它的分块解码照样会在中段撞上会话作废；
        # 从未登录过的会话（_relogin 为空）保持「一次即败」，不误重试。
        if not callable(relogin):
            relogin = self._relogin

        direct_headers = {
            name: value
            for name, value in self._source_headers().items()
            if name.casefold() not in {"cookie", "origin", "referer"}
        }
        sign_lock = threading.Lock()
        clock_offset: int | None = None
        last_signed_at = 0

        def issue_signed_url() -> str:
            nonlocal clock_offset, last_signed_at
            # The CDN rejects a repeated byte range when two otherwise fresh
            # URLs carry the same second-granularity signing timestamp. FFmpeg
            # legitimately repeats MP4 index ranges, so serialize issuance and
            # wait for the next real server-aligned second instead of inventing
            # a future timestamp or retrying a rejected URL.
            with sign_lock:
                base, server_now = self._media_base(course_id, sub_id)
                # Some platform responses are cached together with their
                # `now` field. Recalibrating from that stale value would move
                # later CDN signatures backwards, even though the refreshed
                # base capability is valid. Calibrate once per task session
                # and advance from the local clock afterwards.
                if clock_offset is None:
                    clock_offset = server_now - int(time.time()) if server_now else 0
                signed_at = int(time.time()) + clock_offset
                while signed_at <= last_signed_at:
                    time.sleep(max(0.01, min(1.0, last_signed_at + 1 - signed_at)))
                    signed_at = int(time.time()) + clock_offset
                last_signed_at = signed_at
                return self._sign(base, last_signed_at)

        def refresh_source() -> dict[str, Any]:
            signed = issue_signed_url()
            resolved = resolve_source_address(signed, direct_headers)
            return {
                "url": resolved.url,
                "headers": resolved.headers,
                "resolved_public_ip": resolved.ip,
            }

        def fallback_source() -> dict[str, Any]:
            if not self._webvpn_ready:
                raise _fail(
                    "platform_connection_failed",
                    connection_stage="course_request",
                )
            return {
                "url": _vpn_url(issue_signed_url()),
                "headers": self._source_headers(),
            }

        try:
            source = refresh_source()
        except SourceSecurityError:
            if not self._webvpn_ready:
                raise
            # WebVPN 回退源同样要能扛过会话作废：同一有界续登窗，别让回退
            # 分支成为重试缺口（第四十一案）。
            return {
                **fallback_source(),
                "_refresh_source": lambda: _bounded_session_refresh(fallback_source, relogin),
            }
        output = {
            **source,
            "_refresh_source": lambda: _bounded_session_refresh(refresh_source, relogin),
        }
        if self._webvpn_ready:
            output["_fallback_source"] = fallback_source
        return output

    def _slide_source(self, image: str) -> dict[str, Any]:
        """One slide source plus the opposite transport for one bounded retry.

        Direct and WebVPN-wrapped slide hosts fail on different networks (the
        2026-09-05 real-sample diagnosis), so every candidate carries the
        other route in ``_alternate_source``. Validation and the header rules
        mirror the media source's fallback semantics exactly.
        """
        headers = self._source_headers()
        direct_headers = {
            name: value
            for name, value in headers.items()
            if name.casefold() not in {"cookie", "origin", "referer"}
        }
        if self._course_direct:
            return {
                "url": _validate_upstream_url(image),
                "headers": direct_headers,
                "_alternate_source": {"url": _vpn_url(image), "headers": headers},
            }
        return {
            "url": _vpn_url(image),
            "headers": headers,
            "_alternate_source": {"url": _validate_upstream_url(image), "headers": direct_headers},
        }

    def slide_sources(self, course_id: str, sub_id: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        scope = _slide_deck_scope(course_id, sub_id)
        seen_signatures: set[tuple[str, ...]] = set()
        page = 1
        while True:
            data = self._course_json(
                "/pptnote/v1/schedule/search-ppt",
                params={"course_id": course_id, "sub_id": sub_id,
                        "page": page, "per_page": SLIDE_PAGE_SIZE},
                max_bytes=SLIDE_RESPONSE_MAX_BYTES,
            )
            rows = data.get("list")
            if not isinstance(rows, list):
                raise _fail("platform_slide_payload_invalid")
            if not rows:
                break
            # A repeated page row-id signature means the server cursor is
            # stuck: fail closed instead of looping or silently truncating.
            signature = tuple(
                str(row.get("id") if isinstance(row, dict) else "")
                for row in rows
            )
            if signature in seen_signatures:
                raise _fail("platform_slide_pagination_stalled")
            seen_signatures.add(signature)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    content = json.loads(str(row.get("content") or "{}"))
                except (TypeError, ValueError):
                    continue
                if not isinstance(content, dict):
                    continue
                image = str(content.get("pptimgurl") or "")
                if not image:
                    continue
                try:
                    created_sec = int(row.get("created_sec") or 0)
                except (TypeError, ValueError):
                    continue
                # The source row's opaque record id survives into the
                # transient slide source so the courseware plan can reference
                # captures without any URL, cookie, or title ever leaving
                # this adapter.  Anything overlong or URL-shaped is dropped.
                record_id = str(row.get("id") if row.get("id") is not None else "").strip()
                if (
                    not record_id
                    or len(record_id) > 64
                    or "://" in record_id
                    or "@" in record_id
                    or any(char.isspace() for char in record_id)
                ):
                    record_id = ""
                items.append({
                    "page_num": len(items) + 1,
                    "created_sec": created_sec,
                    "record_id": record_id,
                    "source": self._slide_source(image),
                    "deck": scope,
                })
                if len(items) >= SLIDE_RECORD_STORM_LIMIT:
                    # Gross event density, not a page-count ceiling: a
                    # legitimate lecture stays far below this; exceeding it
                    # fails the job honestly instead of guessing away a tail.
                    raise _fail("platform_slide_record_storm")
            if len(rows) < SLIDE_PAGE_SIZE:
                break
            page += 1
        return items

    def rough_transcript_segments(self, sub_id: str) -> list[dict[str, Any]]:
        """Platform-native transcript rows as rough-leg alternate candidates.

        AS12: one GET per lecture against the same authorized icourse host as
        every other course request; rows are ``{start_ms, end_ms, text}`` with
        second-based platform stamps converted to milliseconds exactly like
        the client's transcript seam (icourse.py ``search-trans-result``).
        Returns ``[]`` when the platform holds no transcript.  Fails with
        closed-set PlatformSessionError codes only — callers treat every
        failure as a fallback signal and never fail the job on it.
        """
        data = self._course_json(
            "/courseapi/v3/web-socket/search-trans-result",
            params={"sub_id": str(sub_id), "format": "json"},
            max_bytes=TRANSCRIPT_RESPONSE_MAX_BYTES,
        )
        if data.get("code") not in (0, 200):
            raise _fail("platform_course_request_failed")
        rows = data.get("list")
        segments: list[dict[str, Any]] = []
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            return segments
        content = rows[0].get("all_content")
        if not isinstance(content, list):
            return segments
        for row in content:
            if not isinstance(row, dict):
                continue
            try:
                start_sec = float(row.get("BeginSec") or 0)
                end_sec = float(row.get("EndSec", row.get("BeginSec") or 0))
            except (TypeError, ValueError):
                continue
            text = str(row.get("Text") or "").strip()
            if not text:
                continue
            start_ms = int(start_sec * 1000)
            segments.append({
                "start_ms": start_ms,
                "end_ms": max(start_ms, int(end_sec * 1000)),
                "text": text,
            })
            if len(segments) >= TRANSCRIPT_RECORD_STORM_LIMIT:
                raise _fail("platform_course_request_failed")
        return segments

    def _source_headers(self) -> dict[str, str]:
        try:
            cookie_items = list(self.session.cookies.items())
        except (AttributeError, TypeError) as exc:
            raise _fail(
                "platform_session_rejected", connection_stage="course_request"
            ) from exc
        normalized: list[tuple[str, str]] = []
        for raw_name, raw_value in cookie_items:
            name, value = str(raw_name or ""), str(raw_value or "")
            if not name or any(char in name or char in value for char in "\r\n"):
                raise _fail(
                    "platform_session_rejected", connection_stage="course_request"
                )
            normalized.append((name, value))
        cookies = "; ".join(f"{name}={value}" for name, value in normalized)
        return {
            "Cookie": cookies,
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Accept-Encoding": "identity;q=1, *;q=0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

    def close(self) -> None:
        self._course_bearer = ""
        self._userinfo = None
        self._webvpn_ready = False
        # 关会话即释放续登回调（连同闭包里的口令），不给关掉的会话留后门。
        self._relogin = None
        try:
            self.course_session.close()
        finally:
            self.session.close()


def _job_needs_rough_transcript(job: dict[str, Any]) -> bool:
    """True for jobs whose subtitle stage consumes rough-leg alternates (AS12)."""
    kind = str(job.get("job_kind") or "")
    if kind == "subtitle":
        return True
    if kind == "learning_pack":
        return "subtitle" in set(job.get("requested_outputs") or [])
    return False


def materialize_job_sources(job: dict[str, Any]) -> dict[str, Any]:
    """Replace a sealed one-job request with runner-local HTTPS sources."""
    payload = dict(job.get("payload") or {})
    request = dict(payload.pop("source_session", {}) or {})
    if not request:
        return job
    secrets = dict(job.get("secrets") or {})
    credentials = dict(secrets.pop("source_credentials", {}) or {})
    account = str(credentials.pop("account", "") or "")
    password = str(credentials.pop("password", "") or "")
    credentials.clear()
    connector: PlatformSession | None = None
    retained_connector = False
    retain_for_media_refresh = False
    try:
        # Cross-border runner routes to the platform drop connections in
        # bursts; a login attempt takes seconds, so back off between tries
        # instead of burning all attempts inside one outage window.
        for attempt in range(_MATERIALIZE_LOGIN_ATTEMPTS):
            connector = PlatformSession()
            try:
                connector.login(account, password)
                break
            except PlatformSessionError as exc:
                connector.close()
                connector = None
                if str(exc) not in _RETRYABLE_LOGIN_ERRORS or attempt == _MATERIALIZE_LOGIN_ATTEMPTS - 1:
                    raise
                time.sleep(2.0 * (2 ** attempt))
        if connector is None:
            raise _fail("platform_connection_failed")
        course_id = str(request.get("course_id") or "")
        sub_id = str(request.get("sub_id") or "")
        if request.get("media"):
            media = dict(payload.get("media") or {})
            # P55：显式续登回调与 _relogin 登记面同梯，等价调用面不留单发缺口。
            media.update(connector.media_source(
                course_id, sub_id,
                relogin=(
                    lambda a=account, p=password, target=connector:
                    _bounded_midtask_relogin(lambda: target.login(a, p))
                ) if account else None,
            ))
            payload["media"] = media
            retain_for_media_refresh = callable(media.get("_refresh_source"))
        if request.get("slides"):
            payload["slides"] = connector.slide_sources(course_id, sub_id)
        if _job_needs_rough_transcript(job):
            # AS12 平台代粗腿：文稿获取失败/空文稿只降级回落，绝不失败任务。
            # 闭集状态随 payload 留在 runner 内存（sealed job 已拆封，绝不回写），
            # 供 ASR 链诚实记账 rough_source 回落原因。
            try:
                rows = connector.rough_transcript_segments(sub_id)
            except PlatformSessionError:
                payload["platform_transcript_state"] = "transcript_fetch_failed"
            else:
                if rows:
                    payload["platform_transcript"] = rows
                    payload["platform_transcript_state"] = "transcript_fetched"
                else:
                    payload["platform_transcript_state"] = "transcript_empty"
        if retain_for_media_refresh:
            payload["_close_source_session"] = connector.close
            retained_connector = True
    finally:
        if connector is not None and not retained_connector:
            connector.close()
        account = ""
        password = ""
    job["payload"] = payload
    job["secrets"] = secrets
    return job


def cloud_session_from_environment() -> PlatformSession:
    """Authenticate from Environment Secrets without exposing values to callers."""
    account = os.environ.pop("COURSELENS_CLOUD_STUDENT_ID", "")
    password = os.environ.pop("COURSELENS_CLOUD_PASSWORD", "")
    if not account or not password:
        account = ""
        password = ""
        raise _fail("platform_credentials_missing")
    try:
        attempts = (
            ("curl", False),
            ("requests", False),
            ("requests", True),
        )
        for attempt, (transport, allow_webvpn_fallback) in enumerate(attempts):
            connector = PlatformSession(transport=transport)
            try:
                connector.login(
                    account,
                    password,
                    allow_webvpn_fallback=allow_webvpn_fallback,
                )
                return connector
            except PlatformSessionError as exc:
                connector.close()
                if str(exc) not in _RETRYABLE_LOGIN_ERRORS or attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
    finally:
        account = ""
        password = ""
    raise _fail("platform_connection_failed")
