from __future__ import annotations

import socket
import unittest
from unittest.mock import Mock, call, patch

from courselens_worker.source import (
    SourceSecurityError,
    _PinnedHTTPSConnection,
    resolve_source,
    safe_headers,
    safe_source_error_code,
    validate_https_url,
)


class FakeResponse:
    def __init__(self, status, location=""):
        self.status = status
        self.location = location

    def getheader(self, name, default=""):
        return self.location if name.lower() == "location" else default

    def close(self):
        return None


class SourceSecurityTests(unittest.TestCase):
    def test_header_allowlist_and_crlf(self):
        self.assertEqual(
            safe_headers({"Cookie": "a=b", "Host": "bad", "Range": "bytes=0-"}),
            {"Cookie": "a=b"},
        )
        with self.assertRaises(SourceSecurityError):
            safe_headers({"Cookie": "a=b\r\nX-Evil: 1"})

    @patch("courselens_worker.source.socket.getaddrinfo")
    def test_private_address_is_rejected(self, resolve):
        resolve.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]
        with self.assertRaises(SourceSecurityError):
            validate_https_url("https://example.invalid/media")

    @patch("courselens_worker.source.socket.getaddrinfo")
    def test_public_https_is_accepted(self, resolve):
        resolve.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        self.assertEqual(validate_https_url("https://example.com/media"), "https://example.com/media")

    @patch("courselens_worker.source._request_once")
    @patch("courselens_worker.source.socket.getaddrinfo")
    def test_public_ip_hint_bypasses_unavailable_dns_but_keeps_tls_hostname(self, resolve, request):
        request.return_value = (Mock(), FakeResponse(200))
        result = resolve_source(
            "https://media.example.com/video",
            {},
            public_ip_hint="93.184.216.34",
        )
        self.assertEqual(result.ip, "93.184.216.34")
        resolve.assert_not_called()
        self.assertEqual(request.call_args.args[0], "https://media.example.com/video")
        self.assertEqual(request.call_args.args[2], "93.184.216.34")

    @patch("courselens_worker.source.socket.getaddrinfo")
    def test_private_or_malformed_ip_hint_is_rejected_before_connect(self, resolve):
        for hint in ("127.0.0.1", "10.0.0.1", "not-an-ip"):
            with self.subTest(hint=hint), self.assertRaises(SourceSecurityError):
                resolve_source("https://media.example.com/video", {}, public_ip_hint=hint)
        resolve.assert_not_called()

    def test_public_log_reason_is_closed_set_and_never_echoes_input(self):
        self.assertEqual(
            safe_source_error_code(SourceSecurityError("source resolved to a non-public address")),
            "non_public_address",
        )
        secret_url = "https://example.invalid/media?token=secret"
        reason = safe_source_error_code(SourceSecurityError(secret_url))
        self.assertEqual(reason, "source_security_error")
        self.assertNotIn("secret", reason)

    @patch("courselens_worker.source._request_once")
    @patch("courselens_worker.source.socket.getaddrinfo")
    def test_cross_host_redirect_drops_authorization_headers(self, resolve, request):
        resolve.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
        ]
        request.side_effect = [
            (Mock(), FakeResponse(302, "https://cdn.example.net/media")),
            (Mock(), FakeResponse(200)),
        ]
        result = resolve_source(
            "https://example.com/media",
            {"Cookie": "secret", "Origin": "https://example.com", "User-Agent": "CourseLens"},
        )
        self.assertEqual(result.url, "https://cdn.example.net/media")
        self.assertEqual(result.headers, {"User-Agent": "CourseLens"})
        self.assertEqual(request.call_args_list[1].args[2], "93.184.216.34")

    @patch("courselens_worker.source.ssl.SSLContext.wrap_socket")
    @patch("courselens_worker.source.socket.create_connection")
    def test_tls_connection_uses_validated_ip_and_original_sni(self, connect, wrap):
        raw = Mock()
        connect.return_value = raw
        wrap.return_value = Mock()
        connection = _PinnedHTTPSConnection("media.example.com", "93.184.216.34", timeout=20)
        connection.connect()
        connect.assert_called_once_with(("93.184.216.34", 443), 20, None)
        wrap.assert_called_once_with(raw, server_hostname="media.example.com")


if __name__ == "__main__":
    unittest.main()
