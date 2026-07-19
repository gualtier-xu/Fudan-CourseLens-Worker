"""SSRF-resistant access to a user-authorized generic HTTPS source."""

from __future__ import annotations

import ipaddress
import http.client
import http.server
import re
import secrets
import ssl
import socket
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import urljoin, urlsplit


ALLOWED_HEADERS = {
    "accept", "accept-encoding", "accept-language", "cache-control", "cookie",
    "origin", "pragma", "referer", "user-agent",
}
MAX_REDIRECTS = 5
_SINGLE_RANGE = re.compile(r"^bytes=(?:\d+-\d*|-\d+)$", re.IGNORECASE)


class SourceSecurityError(ValueError):
    pass


def safe_source_error_code(error: BaseException) -> str:
    """Return a fixed, non-sensitive diagnostic code for public CI logs.

    Source failures can happen before any model work starts.  The authorized
    URL and headers must never enter a public log, so callers only receive one
    of these closed-set reason codes.
    """
    message = str(error)
    exact = {
        "source header contains a line break": "header_line_break",
        "source must be an authenticated-free HTTPS URL": "invalid_https_url",
        "source uses a non-HTTPS port": "invalid_https_port",
        "source host could not be resolved": "dns_resolution_failed",
        "source host has no addresses": "dns_no_addresses",
        "source resolved to a non-public address": "non_public_address",
        "source supplied an invalid public address": "invalid_public_address_hint",
        "source redirect has no location": "redirect_without_location",
        "source exceeded the redirect limit": "redirect_limit",
        "source image exceeds the size limit": "image_size_limit",
    }
    if message in exact:
        return exact[message]
    if message.startswith("source request failed:"):
        return "connection_failed"
    if message.startswith("source authorization probe returned HTTP "):
        return "authorization_http_error"
    if message.startswith("source image returned HTTP "):
        return "image_http_error"
    return "source_security_error"


@dataclass(frozen=True)
class ResolvedSource:
    url: str
    headers: dict[str, str]
    ip: str


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS with certificate/SNI for the hostname and TCP fixed to one IP."""

    def __init__(self, host: str, ip: str, *, timeout: int):
        super().__init__(host, 443, timeout=timeout, context=ssl.create_default_context())
        self._pinned_ip = ip

    def connect(self) -> None:
        raw = socket.create_connection((self._pinned_ip, 443), self.timeout, self.source_address)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)


def safe_headers(raw: dict[str, Any] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in dict(raw or {}).items():
        normalized = str(name).strip().lower()
        if normalized not in ALLOWED_HEADERS:
            continue
        text = str(value)
        if "\r" in text or "\n" in text:
            raise SourceSecurityError("source header contains a line break")
        result["-".join(part.capitalize() for part in normalized.split("-"))] = text
    return result


def _validate_and_resolve(url: str, public_ip_hint: str = "") -> tuple[str, str]:
    value = str(url or "").strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise SourceSecurityError("source must be an authenticated-free HTTPS URL")
    if parsed.port not in (None, 443):
        raise SourceSecurityError("source uses a non-HTTPS port")
    if public_ip_hint:
        try:
            hinted_ip = ipaddress.ip_address(str(public_ip_hint).strip())
        except ValueError as exc:
            raise SourceSecurityError("source supplied an invalid public address") from exc
        if not hinted_ip.is_global:
            raise SourceSecurityError("source supplied an invalid public address")
        return value, str(hinted_ip)
    try:
        addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SourceSecurityError("source host could not be resolved") from exc
    if not addresses:
        raise SourceSecurityError("source host has no addresses")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise SourceSecurityError("source resolved to a non-public address")
    preferred = next((item[4][0] for item in addresses if item[0] == socket.AF_INET), addresses[0][4][0])
    return value, str(preferred)


def validate_https_url(url: str) -> str:
    return _validate_and_resolve(url)[0]


def _request_once(
    url: str,
    headers: dict[str, str],
    ip: str,
    *,
    timeout: int,
    probe: bool,
):
    parsed = urlsplit(url)
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"
    request_headers = dict(headers)
    request_headers["Host"] = str(parsed.hostname)
    if probe:
        request_headers["Range"] = "bytes=0-0"
    connection = _PinnedHTTPSConnection(str(parsed.hostname), ip, timeout=timeout)
    try:
        connection.request("GET", path, headers=request_headers)
        return connection, connection.getresponse()
    except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
        connection.close()
        raise SourceSecurityError(f"source request failed: {type(exc).__name__}") from exc


def resolve_source(
    url: str,
    headers: dict[str, str],
    *,
    timeout: int = 20,
    public_ip_hint: str = "",
) -> ResolvedSource:
    current, ip = _validate_and_resolve(url, public_ip_hint)
    current_headers = dict(headers)
    previous_host = str(urlsplit(current).hostname or "").lower()
    for _ in range(MAX_REDIRECTS + 1):
        connection, response = _request_once(current, current_headers, ip, timeout=timeout, probe=True)
        try:
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location", "")
                if not location:
                    raise SourceSecurityError("source redirect has no location")
                next_url, next_ip = _validate_and_resolve(urljoin(current, location))
                next_host = str(urlsplit(next_url).hostname or "").lower()
                if next_host != previous_host:
                    current_headers = {
                        name: value for name, value in current_headers.items()
                        if name.lower() not in {"cookie", "origin", "referer"}
                    }
                current, ip, previous_host = next_url, next_ip, next_host
                continue
            if response.status >= 400:
                raise SourceSecurityError(f"source authorization probe returned HTTP {response.status}")
            return ResolvedSource(current, current_headers, ip)
        finally:
            response.close()
            connection.close()
    raise SourceSecurityError("source exceeded the redirect limit")


def resolve_redirects(url: str, headers: dict[str, str], *, timeout: int = 20) -> str:
    return resolve_source(url, headers, timeout=timeout).url


def ffmpeg_headers(headers: dict[str, str]) -> str:
    return "".join(f"{name}: {value}\r\n" for name, value in headers.items())


def pinned_curl_command(source: dict[str, Any]) -> list[str]:
    headers = safe_headers(source.get("headers"))
    resolved = resolve_source(
        str(source.get("url") or ""),
        headers,
        public_ip_hint=str(source.get("resolved_public_ip") or ""),
    )
    parsed = urlsplit(resolved.url)
    address = f"[{resolved.ip}]" if ":" in resolved.ip else resolved.ip
    command = [
        "curl", "--fail", "--silent", "--show-error", "--no-progress-meter",
        "--proto", "=https", "--connect-timeout", "30", "--max-time", "21600",
        "--resolve", f"{parsed.hostname}:443:{address}",
    ]
    for name, value in resolved.headers.items():
        command += ["--header", f"{name}: {value}"]
    command += [resolved.url]
    return command


class _PinnedRangeProxy(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, resolved: ResolvedSource, path: str):
        self.resolved = resolved
        self.media_path = path
        self.failure_code = ""
        super().__init__(("127.0.0.1", 0), _PinnedRangeHandler)


@dataclass(frozen=True)
class PinnedMediaProxy:
    url: str
    _server: _PinnedRangeProxy

    @property
    def failure_code(self) -> str:
        return str(self._server.failure_code or "")


class _PinnedRangeHandler(http.server.BaseHTTPRequestHandler):
    server: _PinnedRangeProxy

    def log_message(self, _format: str, *args: Any) -> None:
        return

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._forward(head_only=True)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._forward(head_only=False)

    def _forward(self, *, head_only: bool) -> None:
        if urlsplit(self.path).path != self.server.media_path:
            self.send_error(404)
            return
        range_header = str(self.headers.get("Range") or "").strip()
        if range_header and not _SINGLE_RANGE.fullmatch(range_header):
            self.server.failure_code = "invalid_range"
            self.send_error(416)
            return
        resolved = self.server.resolved
        headers = dict(resolved.headers)
        if range_header:
            headers["Range"] = range_header
        connection = response = None
        try:
            connection, response = _request_once(
                resolved.url,
                headers,
                resolved.ip,
                timeout=60,
                probe=False,
            )
            status = int(response.status)
            if status not in {200, 206, 416}:
                if status in {301, 302, 303, 307, 308}:
                    self.server.failure_code = "upstream_redirect"
                elif status in {401, 403, 404, 429}:
                    self.server.failure_code = f"upstream_http_{status}"
                elif 500 <= status <= 599:
                    self.server.failure_code = "upstream_http_5xx"
                else:
                    self.server.failure_code = "upstream_http_other"
                self.send_error(status if 400 <= status <= 599 else 502)
                return
            self.send_response(status)
            for name in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges", "ETag"):
                value = response.getheader(name, "")
                if value:
                    self.send_header(name, value)
            self.send_header("Cache-Control", "private, no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if head_only or status == 416:
                return
            while True:
                block = response.read(64 * 1024)
                if not block:
                    break
                self.wfile.write(block)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            self.server.failure_code = "upstream_connection_failed"
            try:
                self.send_error(502)
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            if response is not None:
                response.close()
            if connection is not None:
                connection.close()


@contextmanager
def pinned_media_proxy(source: dict[str, Any]) -> Iterator[PinnedMediaProxy]:
    """Expose one validated source as a loopback-only, transient Range URL.

    FFmpeg needs random-access Range requests for many MP4 files.  The proxy
    keeps the authorized upstream and its headers out of the process command,
    pins every connection to the validated public IP, and never writes media
    bytes to disk.
    """
    headers = safe_headers(source.get("headers"))
    source_url = str(source.get("url") or "")
    public_ip_hint = str(source.get("resolved_public_ip") or "")
    if public_ip_hint:
        # A short-lived source may permit only one authorization request.
        # The IP hint is still independently checked as globally routable and
        # the actual request still performs hostname TLS/SNI validation, so an
        # extra network probe is unnecessary and can consume that request.
        validated_url, validated_ip = _validate_and_resolve(source_url, public_ip_hint)
        resolved = ResolvedSource(validated_url, headers, validated_ip)
    else:
        resolved = resolve_source(source_url, headers)
    path = f"/{secrets.token_urlsafe(24)}"
    server = _PinnedRangeProxy(resolved, path)
    thread = threading.Thread(target=server.serve_forever, name="courselens-range-proxy", daemon=True)
    thread.start()
    try:
        yield PinnedMediaProxy(
            f"http://127.0.0.1:{server.server_port}{path}",
            server,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def fetch_bytes(source: dict[str, Any], *, max_bytes: int = 25 * 1024 * 1024) -> bytes:
    headers = safe_headers(source.get("headers"))
    resolved = resolve_source(
        str(source.get("url") or ""),
        headers,
        public_ip_hint=str(source.get("resolved_public_ip") or ""),
    )
    connection, response = _request_once(
        resolved.url, resolved.headers, resolved.ip, timeout=30, probe=False
    )
    try:
        if response.status != 200:
            raise SourceSecurityError(f"source image returned HTTP {response.status}")
        output = bytearray()
        while True:
            block = response.read(64 * 1024)
            if not block:
                break
            output.extend(block)
            if len(output) > max_bytes:
                raise SourceSecurityError("source image exceeds the size limit")
        return bytes(output)
    finally:
        response.close()
        connection.close()
