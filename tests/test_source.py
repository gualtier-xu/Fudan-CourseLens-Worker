from __future__ import annotations

import socket
import unittest
import urllib.request
from unittest.mock import Mock, call, patch

from courselens_worker.source import (
    SourceSecurityError,
    _PinnedHTTPSConnection,
    classify_media_response,
    open_pinned_stream,
    pinned_connect_proxy,
    pinned_media_proxy,
    resolve_source,
    resolve_source_address,
    safe_headers,
    safe_source_error_code,
    validate_https_url,
)


class FakeResponse:
    def __init__(self, status, location="", *, body=b"", headers=None):
        self.status = status
        self.location = location
        self.body = body
        self.offset = 0
        self.headers = dict(headers or {})

    def getheader(self, name, default=""):
        if name.lower() == "location":
            return self.location
        return self.headers.get(name, self.headers.get(name.lower(), default))

    def read(self, size=-1):
        if size is None or size < 0:
            size = len(self.body) - self.offset
        block = self.body[self.offset:self.offset + size]
        self.offset += len(block)
        return block

    def close(self):
        return None


class SourceSecurityTests(unittest.TestCase):
    def test_media_response_profile_is_closed_and_detects_expected_types(self):
        mp4 = classify_media_response(200, "video/mp4", b"\x00\x00\x00\x20ftypisom")
        self.assertEqual((mp4.http, mp4.content, mp4.magic), (
            "http_2xx", "content_video", "magic_iso_bmff"
        ))
        html = classify_media_response(200, "text/html; charset=utf-8", b" <!doctype html>")
        self.assertEqual((html.content, html.magic), ("content_html", "magic_html"))
        json_value = classify_media_response(403, "application/json", b'{"error":true}')
        self.assertEqual((json_value.http, json_value.content, json_value.magic), (
            "http_403", "content_json", "magic_json"
        ))

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
    def test_address_resolution_does_not_consume_the_source(self, resolve, request):
        resolve.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        value = resolve_source_address("https://example.com/media", {"User-Agent": "CourseLens"})
        self.assertEqual(value.ip, "93.184.216.34")
        request.assert_not_called()

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

    @patch("courselens_worker.source._request_once")
    @patch("courselens_worker.source.socket.getaddrinfo")
    def test_stream_with_public_hint_uses_one_non_probe_request(self, resolve, request):
        connection = Mock()
        request.return_value = (
            connection,
            FakeResponse(200, body=b"\x00\x00\x00\x20ftypisom", headers={"Content-Type": "video/mp4"}),
        )
        source = {
            "url": "https://media.example.com/video",
            "headers": {"Cookie": "secret"},
            "resolved_public_ip": "93.184.216.34",
        }
        with open_pinned_stream(source) as stream:
            prefix = stream.response.read(64)
            profile = classify_media_response(
                stream.response.status,
                stream.response.getheader("Content-Type", ""),
                prefix,
            )
        self.assertEqual(profile.magic, "magic_iso_bmff")
        resolve.assert_not_called()
        request.assert_called_once_with(
            "https://media.example.com/video",
            {"Cookie": "secret"},
            "93.184.216.34",
            timeout=60,
            probe=False,
        )
        connection.close.assert_called_once()

    @patch("courselens_worker.source._request_once")
    @patch("courselens_worker.source.socket.getaddrinfo")
    def test_stream_redirect_revalidates_and_drops_cross_host_credentials(self, resolve, request):
        resolve.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.35", 443)),
        ]
        request.side_effect = [
            (Mock(), FakeResponse(302, "https://cdn.example.net/media")),
            (Mock(), FakeResponse(200, body=b"\x00\x00\x00\x20ftypisom")),
        ]
        source = {
            "url": "https://media.example.com/video",
            "headers": {"Cookie": "secret", "Origin": "https://media.example.com", "User-Agent": "CourseLens"},
            "resolved_public_ip": "93.184.216.34",
        }
        with open_pinned_stream(source) as stream:
            self.assertEqual(stream.response.status, 200)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[1].args[1], {"User-Agent": "CourseLens"})
        self.assertEqual(request.call_args_list[1].args[2], "93.184.216.35")
        self.assertFalse(request.call_args_list[0].kwargs["probe"])
        self.assertFalse(request.call_args_list[1].kwargs["probe"])

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
    def test_loopback_proxy_forwards_one_bounded_range_without_disk(self, request):
        request.side_effect = lambda *_args, **_kwargs: (Mock(), FakeResponse(
            206,
            body=b"test",
            headers={
                "Content-Type": "video/mp4",
                "Content-Length": "4",
                "Content-Range": "bytes 10-13/100",
                "Accept-Ranges": "bytes",
            },
        ))
        source = {
            "url": "https://media.example.com/video",
            "headers": {"User-Agent": "CourseLens"},
            "resolved_public_ip": "93.184.216.34",
        }
        with pinned_media_proxy(source) as proxy:
            self.assertTrue(proxy.url.startswith("http://127.0.0.1:"))
            probe = urllib.request.Request(proxy.url, headers={"Range": "bytes=10-13"})
            with urllib.request.urlopen(probe, timeout=5) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), b"test")
            self.assertEqual(proxy.failure_code, "")
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[2], "93.184.216.34")
        self.assertEqual(request.call_args.args[1]["Range"], "bytes=10-13")

    @patch("courselens_worker.source._request_once")
    def test_loopback_proxy_records_only_a_fixed_upstream_failure_code(self, request):
        request.return_value = (Mock(), FakeResponse(403))
        source = {
            "url": "https://media.example.com/video?secret=value",
            "resolved_public_ip": "93.184.216.34",
        }
        with pinned_media_proxy(source) as proxy:
            probe = urllib.request.Request(proxy.url, headers={"Range": "bytes=0-0"})
            with self.assertRaises(Exception):
                urllib.request.urlopen(probe, timeout=5)
            self.assertEqual(proxy.failure_code, "upstream_http_403")
            self.assertNotIn("secret", proxy.failure_code)

    @patch("courselens_worker.source._request_once")
    def test_loopback_proxy_refreshes_a_single_use_source_for_each_range(self, request):
        request.side_effect = lambda *_args, **_kwargs: (Mock(), FakeResponse(
            206,
            body=b"test",
            headers={
                "Content-Type": "video/mp4",
                "Content-Length": "4",
                "Content-Range": "bytes 0-3/8",
            },
        ))
        calls = []

        def refresh():
            calls.append(len(calls) + 1)
            return {
                "url": f"https://media.example.com/video?request={calls[-1]}",
                "resolved_public_ip": "93.184.216.34",
            }

        source = {**refresh(), "_refresh_source": refresh}
        with pinned_media_proxy(source) as proxy:
            for _ in range(2):
                probe = urllib.request.Request(proxy.url, headers={"Range": "bytes=0-3"})
                with urllib.request.urlopen(probe, timeout=5) as response:
                    self.assertEqual(response.read(), b"test")
        self.assertEqual(len(calls), 3)
        self.assertNotEqual(request.call_args_list[0].args[0], request.call_args_list[1].args[0])

    @patch("courselens_worker.source._connect_pinned_upstream")
    def test_connect_proxy_pins_target_and_tunnels_without_inspecting_bytes(self, connect):
        proxy_upstream, test_upstream = socket.socketpair()
        connect.return_value = proxy_upstream
        source = {
            "url": "https://media.example.com/video?secret=value",
            "headers": {"Cookie": "secret"},
            "resolved_public_ip": "93.184.216.34",
        }
        try:
            with pinned_connect_proxy(source) as proxy:
                with socket.create_connection(("127.0.0.1", int(proxy.proxy_url.rsplit(":", 1)[1]))) as client:
                    client.sendall(
                        b"CONNECT media.example.com:443 HTTP/1.1\r\nHost: media.example.com:443\r\n\r\n"
                    )
                    self.assertTrue(client.recv(256).startswith(b"HTTP/1.1 200"))
                    client.sendall(b"opaque-tls-bytes")
                    self.assertEqual(test_upstream.recv(256), b"opaque-tls-bytes")
                    test_upstream.sendall(b"opaque-response")
                    self.assertEqual(client.recv(256), b"opaque-response")
                self.assertEqual(proxy.failure_code, "")
        finally:
            test_upstream.close()
        connect.assert_called_once_with("93.184.216.34")

    @patch("courselens_worker.source._connect_pinned_upstream")
    def test_connect_proxy_rejects_any_other_hostname(self, connect):
        source = {
            "url": "https://media.example.com/video",
            "resolved_public_ip": "93.184.216.34",
        }
        with pinned_connect_proxy(source) as proxy:
            with socket.create_connection(("127.0.0.1", int(proxy.proxy_url.rsplit(":", 1)[1]))) as client:
                client.sendall(b"CONNECT other.example.com:443 HTTP/1.1\r\n\r\n")
                self.assertTrue(client.recv(256).startswith(b"HTTP/1.1 403"))
            self.assertEqual(proxy.failure_code, "target_mismatch")
        connect.assert_not_called()

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
