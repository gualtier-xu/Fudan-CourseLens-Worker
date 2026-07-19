"""SSRF-resistant access to a user-authorized generic HTTPS source."""

from __future__ import annotations

import ipaddress
import http.client
import ssl
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlsplit


ALLOWED_HEADERS = {
    "accept", "accept-encoding", "accept-language", "cache-control", "cookie",
    "origin", "pragma", "referer", "user-agent",
}
MAX_REDIRECTS = 5


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
