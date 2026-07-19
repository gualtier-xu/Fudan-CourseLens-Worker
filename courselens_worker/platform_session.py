"""One-job platform session used only to authorize derived processing.

The connector deliberately exposes no course catalog, original-media download,
resume, archive, or persistence API. Credentials arrive inside the existing
sealed job, are used on the runner's own egress address, and are removed from
the in-memory job before model work starts.
"""

from __future__ import annotations

import base64
import hashlib
import html as html_module
import json
import re
import time
import uuid
from binascii import hexlify
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import requests
from Crypto.Cipher import AES, PKCS1_v1_5
from Crypto.PublicKey import RSA


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


class PlatformSessionError(RuntimeError):
    """Closed-set failure suitable for reduction in public logs."""


def _fail(code: str) -> PlatformSessionError:
    return PlatformSessionError(code)


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


class PlatformSession:
    """Narrow session capable only of authorizing one requested lecture."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._userinfo: dict[str, Any] | None = None

    def _once(self, method: str, url: str, **kwargs) -> requests.Response:
        target = _validate_url(url)
        kwargs["allow_redirects"] = False
        kwargs.setdefault("timeout", (10, 60))
        try:
            return self.session.request(method, target, **kwargs)
        except requests.RequestException as exc:
            raise _fail("platform_connection_failed") from exc

    def _follow(self, method: str, url: str, **kwargs) -> tuple[requests.Response, list[str]]:
        current = _validate_url(url)
        current_method = method.upper()
        trace = [current]
        for _ in range(_MAX_REDIRECTS + 1):
            response = self._once(current_method, current, **kwargs)
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

    def login(self, account: str, password: str) -> None:
        if not account or not password:
            raise _fail("platform_credentials_missing")
        try:
            self._login_webvpn(account, password)
            self._login_course(account, password)
        except PlatformSessionError:
            raise
        except Exception as exc:
            raise _fail("platform_auth_failed") from exc

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

    def _login_webvpn(self, account: str, password: str) -> None:
        service = f"{WEBVPN_BASE}/login?cas_login=true"
        current = f"{IDP_BASE}/idp/authCenter/authenticate?service={quote(service, safe='')}"
        lck = ""
        for _ in range(_MAX_REDIRECTS + 1):
            response = self._once("GET", current)
            location = response.headers.get("Location", "")
            status = response.status_code
            response.close()
            match = re.search(r"[?&]lck=([^&]+)", location)
            if match:
                lck = match.group(1)
                break
            if status not in _REDIRECTS or not location:
                break
            current = _validate_url(urljoin(current, location))
        if not lck:
            raise _fail("platform_auth_context_missing")

        method_data = _json(self._once(
            "POST", f"{IDP_BASE}/idp/authn/queryAuthMethods",
            json={"lck": lck, "entityId": WEBVPN_BASE},
            headers={"Content-Type": "application/json", "Referer": f"{IDP_BASE}/ac/", "Origin": IDP_BASE},
        ), "platform_auth_method_missing")
        chain, request_type = self._auth_method(method_data)
        key_data = _json(self._once(
            "GET", f"{IDP_BASE}/idp/authn/getJsPublicKey",
            headers={"Referer": f"{IDP_BASE}/ac/"},
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
        ), "platform_auth_failed")
        if str(auth_data.get("code")) != "200" or not auth_data.get("loginToken"):
            raise _fail("platform_auth_failed")
        ticket_response = self._once(
            "POST", f"{IDP_BASE}/idp/authCenter/authnEngine",
            data={"loginToken": str(auth_data["loginToken"])},
            headers={"Referer": f"{IDP_BASE}/ac/", "Origin": IDP_BASE},
        )
        ticket = _ticket_from_html(ticket_response.text[:256 * 1024], "platform_ticket_missing")
        ticket_response.close()
        try:
            response, _ = self._follow("GET", ticket, stream=True, timeout=(5, 12))
        except PlatformSessionError as exc:
            # The portal can set its cookie before a streamed ticket response
            # times out. Never replay the single-use ticket; verify instead.
            if str(exc) != "platform_connection_failed" or not self._verify_webvpn():
                raise
        else:
            try:
                if not 200 <= response.status_code < 300 or _is_login_target(response.url):
                    raise _fail("platform_ticket_rejected")
            finally:
                response.close()
        if not self._verify_webvpn():
            raise _fail("platform_session_rejected")

    def _verify_webvpn(self) -> bool:
        response = None
        try:
            response = self._once("GET", WEBVPN_BASE + "/", stream=True, timeout=(3, 8))
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

    def _login_course(self, account: str, password: str) -> None:
        cas = (
            f"{ICOURSE_BASE}/casapi/index.php?r=auth/login&school_login=1"
            f"&tenant_code={TENANT_CODE}&forward={quote(ICOURSE_BASE + '/', safe='')}"
        )
        response, trace = self._follow("GET", _vpn_url(cas))
        lck = ""
        try:
            for candidate in trace + [response.url]:
                match = re.search(r'lck=([^&#"]+)', candidate)
                if match:
                    lck = match.group(1)
                    break
            if not lck:
                match = re.search(r'lck=([^&#"]+)', response.text[:5000])
                if match:
                    lck = match.group(1)
        finally:
            response.close()
        if not lck:
            raise _fail("platform_course_context_missing")

        idp_vpn = _vpn_url(IDP_BASE)
        method_data = _json(self._once(
            "POST", _vpn_url(f"{IDP_BASE}/idp/authn/queryAuthMethods"),
            json={"lck": lck, "entityId": ICOURSE_BASE},
            headers={"Content-Type": "application/json", "Referer": f"{idp_vpn}/ac/", "Origin": WEBVPN_BASE},
        ), "platform_auth_method_missing")
        chain, request_type = self._auth_method(method_data)
        key_data = _json(self._once(
            "GET", _vpn_url(f"{IDP_BASE}/idp/authn/getJsPublicKey"),
            headers={"Referer": f"{idp_vpn}/ac/"},
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
        ), "platform_auth_failed")
        if str(auth_data.get("code")) != "200" or not auth_data.get("loginToken"):
            raise _fail("platform_auth_failed")
        ticket_response = self._once(
            "POST", _vpn_url(f"{IDP_BASE}/idp/authCenter/authnEngine"),
            data={"loginToken": str(auth_data["loginToken"])},
            headers={"Referer": f"{idp_vpn}/ac/", "Origin": WEBVPN_BASE},
        )
        ticket = _ticket_from_html(ticket_response.text[:256 * 1024], "platform_ticket_missing")
        ticket_response.close()
        if urlparse(ticket).hostname != urlparse(WEBVPN_BASE).hostname:
            ticket = _vpn_url(ticket)
        try:
            response, _ = self._follow("GET", ticket, stream=True, timeout=(5, 12))
        except PlatformSessionError as exc:
            if str(exc) != "platform_connection_failed" or not self._verify_course():
                raise
        else:
            try:
                if not 200 <= response.status_code < 300 or _is_login_target(response.url):
                    raise _fail("platform_ticket_rejected")
            finally:
                response.close()
        if not self._verify_course():
            raise _fail("platform_session_rejected")

    def _verify_course(self) -> bool:
        response = None
        try:
            response = self._once("GET", _vpn_url(f"{ICOURSE_BASE}/userapi/v1/infosimple"), timeout=(3, 8))
            data = _json(response, "platform_session_rejected")
            return response.status_code == 200 and data.get("code") in (0, 200)
        except PlatformSessionError:
            return False
        finally:
            if response is not None:
                response.close()

    def _course_json(self, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
        response = self._once("GET", _vpn_url(ICOURSE_BASE + path), params=params)
        try:
            if response.status_code != 200:
                raise _fail("platform_course_request_failed")
            return _json(response, "platform_course_request_failed")
        finally:
            response.close()

    def _userinfo_value(self) -> dict[str, Any]:
        if self._userinfo is None:
            data = self._course_json("/userapi/v1/infosimple", params={})
            if data.get("code") not in (0, 200):
                raise _fail("platform_course_request_failed")
            self._userinfo = dict(data.get("params") or data.get("data") or {})
        return self._userinfo

    def _sign(self, media_url: str, now: int | None) -> str:
        info = self._userinfo_value()
        timestamp = int(now or time.time())
        media_url = _validate_upstream_url(media_url)
        path = urlparse(media_url).path
        raw = f"{path}{info.get('id', '')}{info.get('tenant_id', '')}{str(info.get('phone', ''))[::-1]}{timestamp}"
        token = f"{info.get('id', '')}-{timestamp}-{hashlib.md5(raw.encode()).hexdigest()}"
        separator = "&" if "?" in media_url else "?"
        return f"{media_url}{separator}clientUUID={uuid.uuid4()}&t={token}"

    def media_source(self, course_id: str, sub_id: str) -> dict[str, Any]:
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
        signed = self._sign(base, int(info.get("now") or 0) or None)
        # Keep authorization and the media stream on the same runner-side
        # WebVPN session.  The upstream CDN name is not guaranteed to resolve
        # from GitHub-hosted runners, and a direct request would also bypass
        # the session whose egress address was used to mint the signature.
        return {"url": _vpn_url(signed), "headers": self._source_headers()}

    def slide_sources(self, course_id: str, sub_id: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page = 1
        while page <= 50:
            data = self._course_json(
                "/pptnote/v1/schedule/search-ppt",
                params={"course_id": course_id, "sub_id": sub_id, "page": page, "per_page": 100},
            )
            rows = list(data.get("list") or [])
            if not rows:
                break
            for row in rows:
                try:
                    content = json.loads(str(row.get("content") or "{}"))
                except (TypeError, ValueError):
                    continue
                image = str(content.get("pptimgurl") or "")
                if not image:
                    continue
                items.append({
                    "page_num": len(items) + 1,
                    "created_sec": int(row.get("created_sec") or 0),
                    "source": {"url": _vpn_url(image), "headers": self._source_headers()},
                })
            if len(rows) < 100:
                break
            page += 1
        return items

    def _source_headers(self) -> dict[str, str]:
        cookies = "; ".join(f"{cookie.name}={cookie.value}" for cookie in self.session.cookies)
        return {
            "Cookie": cookies,
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Accept-Encoding": "identity;q=1, *;q=0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }


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
    try:
        connector = PlatformSession()
        connector.login(account, password)
        course_id = str(request.get("course_id") or "")
        sub_id = str(request.get("sub_id") or "")
        if request.get("media"):
            media = dict(payload.get("media") or {})
            media.update(connector.media_source(course_id, sub_id))
            payload["media"] = media
        if request.get("slides"):
            payload["slides"] = connector.slide_sources(course_id, sub_id)
    finally:
        account = ""
        password = ""
    job["payload"] = payload
    job["secrets"] = secrets
    return job
