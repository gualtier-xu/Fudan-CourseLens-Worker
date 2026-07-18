"""SSRF-resistant access to a user-authorized generic HTTPS source."""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests


ALLOWED_HEADERS = {
    "accept", "accept-encoding", "accept-language", "cache-control", "cookie",
    "origin", "pragma", "referer", "user-agent",
}
MAX_REDIRECTS = 5


class SourceSecurityError(ValueError):
    pass


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


def validate_https_url(url: str) -> str:
    value = str(url or "").strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise SourceSecurityError("source must be an authenticated-free HTTPS URL")
    if parsed.port not in (None, 443):
        raise SourceSecurityError("source uses a non-HTTPS port")
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
    return value


def resolve_redirects(url: str, headers: dict[str, str], *, timeout: int = 20) -> str:
    current = validate_https_url(url)
    session = requests.Session()
    for _ in range(MAX_REDIRECTS + 1):
        try:
            response = session.get(
                current,
                headers=headers,
                allow_redirects=False,
                stream=True,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise SourceSecurityError(f"source authorization probe failed: {type(exc).__name__}") from exc
        try:
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location", "")
                if not location:
                    raise SourceSecurityError("source redirect has no location")
                current = validate_https_url(urljoin(current, location))
                continue
            if response.status_code >= 400:
                raise SourceSecurityError(f"source authorization probe returned HTTP {response.status_code}")
            return current
        finally:
            response.close()
    raise SourceSecurityError("source exceeded the redirect limit")


def ffmpeg_headers(headers: dict[str, str]) -> str:
    return "".join(f"{name}: {value}\r\n" for name, value in headers.items())


def fetch_bytes(source: dict[str, Any], *, max_bytes: int = 25 * 1024 * 1024) -> bytes:
    headers = safe_headers(source.get("headers"))
    url = resolve_redirects(str(source.get("url") or ""), headers)
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
    except requests.RequestException as exc:
        raise SourceSecurityError(f"source image request failed: {type(exc).__name__}") from exc
    try:
        if response.status_code != 200:
            raise SourceSecurityError(f"source image returned HTTP {response.status_code}")
        output = bytearray()
        for block in response.iter_content(64 * 1024):
            output.extend(block)
            if len(output) > max_bytes:
                raise SourceSecurityError("source image exceeds the size limit")
        return bytes(output)
    finally:
        response.close()
