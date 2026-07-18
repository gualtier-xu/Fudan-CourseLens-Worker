from __future__ import annotations

import socket
import unittest
from unittest.mock import patch

from courselens_worker.source import SourceSecurityError, safe_headers, validate_https_url


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


if __name__ == "__main__":
    unittest.main()
