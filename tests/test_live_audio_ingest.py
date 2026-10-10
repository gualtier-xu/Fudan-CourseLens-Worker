"""D13-PROD 直播音频摄取器钉：请求前 host 校验/content-type 闭集/内存环.

离线可跑：URL 校验用字面量 IP 与注入 resolver，取流/解码全部替身；
真 ffmpeg 解码腿在 ffmpeg 在场时冒烟（缺席跳过）。
"""

from __future__ import annotations

import io
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from courselens_worker import source as source_mod
from courselens_worker.live_audio_ingest import (
    CODE_CONNECTION_FAILED,
    CODE_DECODER_FAILED,
    CODE_DNS_NO_ADDRESSES,
    CODE_DNS_RESOLUTION_FAILED,
    CODE_INVALID_HTTPS_PORT,
    CODE_INVALID_HTTPS_URL,
    CODE_NON_PUBLIC_ADDRESS,
    CODE_PLAYLIST_CONTENT_TYPE,
    CODE_PLAYLIST_HTTP,
    CODE_PLAYLIST_INVALID,
    CODE_SEGMENT_CONTENT_TYPE,
    CODE_SEGMENT_HTTP,
    FetchedBody,
    LiveAudioIngestor,
    LiveIngestError,
    StreamingRingBuffer,
    decode_segment_to_pcm,
    parse_media_playlist,
    validate_stream_url,
)

PUBLIC_URL = "https://live-media.example.edu/live/audio.m3u8"


def _resolver_single(address: str):
    def resolver(hostname, port, type=None):  # noqa: A002 - socket.getaddrinfo 形状
        return [(2, 1, 6, "", (address, port))]

    return resolver


class ValidateStreamUrlTests(unittest.TestCase):
    def test_https_and_userinfo_and_port_gates(self):
        for bad in (
            "http://live-media.example.edu/live/audio.m3u8",
            "ftp://live-media.example.edu/live/audio.m3u8",
            "https://user:pw@live-media.example.edu/live/audio.m3u8",
            "https://live-media.example.edu:8443/live/audio.m3u8",
            "",
            "not a url",
            "//live-media.example.edu/live/audio.m3u8",
        ):
            with self.assertRaises(LiveIngestError) as caught:
                validate_stream_url(bad)
            self.assertIn(
                caught.exception.code,
                {CODE_INVALID_HTTPS_URL, CODE_INVALID_HTTPS_PORT},
                bad,
            )
        # 非 443 显式端口单独钉码。
        with self.assertRaises(LiveIngestError) as caught:
            validate_stream_url("https://live-media.example.edu:8080/a.m3u8")
        self.assertEqual(caught.exception.code, CODE_INVALID_HTTPS_PORT)

    def test_non_public_addresses_rejected_before_any_request(self):
        for host in (
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.9",
            "192.168.1.1",
            "169.254.3.4",
            "100.64.0.1",
            "[::1]",
            "[fe80::1]",
            "0.0.0.0",
        ):
            with self.assertRaises(LiveIngestError) as caught:
                validate_stream_url(f"https://{host}/live/audio.m3u8")
            self.assertEqual(caught.exception.code, CODE_NON_PUBLIC_ADDRESS, host)

    def test_public_literal_accepted_with_pinned_ip(self):
        validated, ip = validate_stream_url("https://8.8.8.8/live/audio.m3u8")
        self.assertEqual(validated, "https://8.8.8.8/live/audio.m3u8")
        self.assertEqual(ip, "8.8.8.8")

    def test_injected_resolver_public_host(self):
        validated, ip = validate_stream_url(
            PUBLIC_URL, resolver=_resolver_single("93.184.216.34")
        )
        self.assertEqual(validated, PUBLIC_URL)
        self.assertEqual(ip, "93.184.216.34")

    def test_injected_resolver_mixed_private_rejects(self):
        def resolver(hostname, port, type=None):  # noqa: A002
            return [
                (2, 1, 6, "", ("93.184.216.34", port)),
                (2, 1, 6, "", ("192.168.1.1", port)),
            ]

        with self.assertRaises(LiveIngestError) as caught:
            validate_stream_url(PUBLIC_URL, resolver=resolver)
        self.assertEqual(caught.exception.code, CODE_NON_PUBLIC_ADDRESS)

    def test_resolver_failure_codes(self):
        def gaierror_resolver(hostname, port, type=None):  # noqa: A002
            import socket

            raise socket.gaierror("no dns")

        with self.assertRaises(LiveIngestError) as caught:
            validate_stream_url(PUBLIC_URL, resolver=gaierror_resolver)
        self.assertEqual(caught.exception.code, CODE_DNS_RESOLUTION_FAILED)

        def empty_resolver(hostname, port, type=None):  # noqa: A002
            return []

        with self.assertRaises(LiveIngestError) as caught:
            validate_stream_url(PUBLIC_URL, resolver=empty_resolver)
        self.assertEqual(caught.exception.code, CODE_DNS_NO_ADDRESSES)

    def test_error_code_parity_with_source_precedent(self):
        """host 校验腿与 source.py 既有闭集码逐例对齐（清单先例）。"""
        matrix = [
            "http://live-media.example.edu/a.m3u8",
            "https://user:pw@live-media.example.edu/a.m3u8",
            "https://live-media.example.edu:8443/a.m3u8",
            "https://127.0.0.1/a.m3u8",
            "https://192.168.1.1/a.m3u8",
        ]
        for url in matrix:
            source_code = "ok"
            try:
                source_mod.validate_https_url(url)
            except source_mod.SourceSecurityError as exc:
                source_code = source_mod.safe_source_error_code(exc)
            lane_code = "ok"
            try:
                validate_stream_url(url)
            except LiveIngestError as exc:
                lane_code = exc.code
            self.assertEqual(lane_code, source_code, url)


class ParseMediaPlaylistTests(unittest.TestCase):
    def test_extm3u_required(self):
        with self.assertRaises(LiveIngestError) as caught:
            parse_media_playlist("seg0.ts\nseg1.ts\n", base_url=PUBLIC_URL)
        self.assertEqual(caught.exception.code, CODE_PLAYLIST_INVALID)

    def test_extinf_pairs_and_endlist(self):
        text = (
            "#EXTM3U\n"
            "#EXT-X-TARGETDURATION:4\n"
            "#EXTINF:4.0,\n"
            "seg0.ts\n"
            "#EXTINF:4.0,\n"
            "seg1.ts\n"
            "#EXT-X-ENDLIST\n"
        )
        plan = parse_media_playlist(text, base_url=PUBLIC_URL)
        self.assertEqual(
            plan.segment_urls,
            (
                "https://live-media.example.edu/live/seg0.ts",
                "https://live-media.example.edu/live/seg1.ts",
            ),
        )
        self.assertTrue(plan.ended)

    def test_live_playlist_without_endlist_and_absolute_uris(self):
        text = "#EXTM3U\n#EXTINF:2.0,\n/abs/seg9.ts\n"
        plan = parse_media_playlist(text, base_url=PUBLIC_URL)
        self.assertEqual(plan.segment_urls, ("https://live-media.example.edu/abs/seg9.ts",))
        self.assertFalse(plan.ended)

    def test_variant_playlist_lines_ignored(self):
        text = "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=128000\nlo.m3u8\n"
        plan = parse_media_playlist(text, base_url=PUBLIC_URL)
        self.assertEqual(plan.segment_urls, ())


class RingBufferTests(unittest.TestCase):
    def test_fifo_extend_and_drain(self):
        ring = StreamingRingBuffer(10.0, sample_rate=16000)
        ring.extend([1.0] * 16000)
        ring.extend([2.0] * 8000)
        self.assertEqual(ring.buffered_seconds, 1.5)
        drained = ring.drain(0.5)
        self.assertEqual(len(drained), 8000)
        self.assertEqual(drained[0], 1.0)
        rest = ring.drain()
        self.assertEqual(rest[0], 1.0)
        self.assertEqual(rest[-1], 2.0)
        self.assertEqual(ring.buffered_seconds, 0.0)

    def test_overflow_drops_oldest_and_accounts_seconds(self):
        ring = StreamingRingBuffer(1.0, sample_rate=16000)
        dropped = ring.extend([0.5] * 32000)
        self.assertAlmostEqual(dropped, 1.0)
        self.assertAlmostEqual(ring.dropped_seconds, 1.0)
        self.assertAlmostEqual(ring.buffered_seconds, 1.0)
        self.assertEqual(ring.drain()[0], 0.5)  # 保留的是最新 1s

    def test_rejects_nonpositive_cap(self):
        with self.assertRaises(ValueError):
            StreamingRingBuffer(0.0)


def _playlist_fetch_factory(segments: list[str], *, ended=False, segment_body=b"ts"):
    manifest = "#EXTM3U\n"
    for uri in segments:
        manifest += f"#EXTINF:4.0,\n{uri}\n"
    if ended:
        manifest += "#EXT-X-ENDLIST\n"
    calls: list[str] = []

    def fetch(url: str) -> FetchedBody:
        calls.append(url)
        if url == PUBLIC_URL:
            return FetchedBody(200, "application/vnd.apple.mpegurl", manifest.encode())
        return FetchedBody(200, "video/mp2t", segment_body)

    return fetch, calls


def _make_ingestor(fetch, decoder=None) -> LiveAudioIngestor:
    return LiveAudioIngestor(
        PUBLIC_URL,
        resolver=_resolver_single("93.184.216.34"),
        fetch=fetch,
        decoder=decoder or (lambda data: [0.25] * 16000),
        ring_seconds=600.0,
    )


def _honest_resolver(hostname, port, type=None):  # noqa: A002 - getaddrinfo 形状
    """字面量 IP 返回自身（离线校验真实可达），域名返回固定公网地址。"""
    import ipaddress

    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return [(2, 1, 6, "", ("93.184.216.34", port))]
    return [(2, 1, 6, "", (hostname, port))]


class IngestorPollTests(unittest.TestCase):
    def _ingestor(self, fetch, decoder=None, url=PUBLIC_URL) -> LiveAudioIngestor:
        return LiveAudioIngestor(
            url,
            resolver=_honest_resolver,
            fetch=fetch,
            decoder=decoder or (lambda data: [0.25] * 16000),
            ring_seconds=600.0,
        )

    def test_constructor_validates_before_any_request(self):
        fetch_calls: list[str] = []

        def fetch(url):
            fetch_calls.append(url)
            return FetchedBody(200, "application/vnd.apple.mpegurl", b"")

        with self.assertRaises(LiveIngestError) as caught:
            self._ingestor(fetch, url="https://127.0.0.1/a.m3u8")
        self.assertEqual(caught.exception.code, CODE_NON_PUBLIC_ADDRESS)
        self.assertEqual(fetch_calls, [])

    def test_poll_fetches_new_segments_and_buffers_pcm(self):
        fetch, calls = _playlist_fetch_factory(["seg0.ts", "seg1.ts"])
        ingestor = self._ingestor(fetch)
        result = ingestor.poll()
        self.assertEqual(result.segments_fetched, 2)
        self.assertAlmostEqual(result.pcm_seconds, 2.0)
        self.assertAlmostEqual(ingestor.buffered_seconds, 2.0)
        self.assertFalse(result.ended)
        self.assertEqual(len(calls), 3)  # 清单 1 + 分片 2

    def test_second_poll_skips_seen_segments(self):
        fetch, calls = _playlist_fetch_factory(["seg0.ts", "seg1.ts", "seg2.ts"])
        ingestor = self._ingestor(fetch)
        first = ingestor.poll()
        self.assertEqual(first.segments_fetched, 3)
        seen_after_first = len(calls)
        result = ingestor.poll()
        self.assertEqual(result.segments_fetched, 0)
        # 第二轮只有清单请求（新分片为零），每次请求前都重新校验。
        self.assertEqual(len(calls), seen_after_first + 1)
        self.assertEqual(calls[-1], PUBLIC_URL)

    def test_ended_playlist_marks_done_and_short_circuits(self):
        fetch, calls = _playlist_fetch_factory(["seg0.ts"], ended=True)
        ingestor = self._ingestor(fetch)
        result = ingestor.poll()
        self.assertTrue(result.ended)
        self.assertTrue(ingestor.ended)
        calls.clear()
        again = ingestor.poll()
        self.assertTrue(again.ended)
        self.assertEqual(calls, [])

    def test_playlist_http_and_content_type_gates(self):
        def http_fail(url):
            return FetchedBody(403, "application/vnd.apple.mpegurl", b"")

        with self.assertRaises(LiveIngestError) as caught:
            self._ingestor(http_fail).poll()
        self.assertEqual(caught.exception.code, CODE_PLAYLIST_HTTP)

        def type_fail(url):
            return FetchedBody(200, "text/html", b"<html></html>")

        with self.assertRaises(LiveIngestError) as caught:
            self._ingestor(type_fail).poll()
        self.assertEqual(caught.exception.code, CODE_PLAYLIST_CONTENT_TYPE)

    def test_segment_http_and_content_type_gates(self):
        def segment_fail(url):
            if url == PUBLIC_URL:
                return FetchedBody(200, "application/vnd.apple.mpegurl", b"#EXTM3U\n#EXTINF:4.0,\ns0.ts\n")
            return FetchedBody(404, "video/mp2t", b"")

        with self.assertRaises(LiveIngestError) as caught:
            self._ingestor(segment_fail).poll()
        self.assertEqual(caught.exception.code, CODE_SEGMENT_HTTP)

        def type_fail(url):
            if url == PUBLIC_URL:
                return FetchedBody(200, "application/vnd.apple.mpegurl", b"#EXTM3U\n#EXTINF:4.0,\ns0.ts\n")
            return FetchedBody(200, "application/json", b"{}")

        with self.assertRaises(LiveIngestError) as caught:
            self._ingestor(type_fail).poll()
        self.assertEqual(caught.exception.code, CODE_SEGMENT_CONTENT_TYPE)

    def test_decoder_failure_is_fail_closed_not_fake_silence(self):
        def failing_decoder(data):
            raise LiveIngestError(CODE_DECODER_FAILED, "boom")

        fetch, _calls = _playlist_fetch_factory(["seg0.ts"])
        ingestor = self._ingestor(fetch, decoder=failing_decoder)
        with self.assertRaises(LiveIngestError) as caught:
            ingestor.poll()
        self.assertEqual(caught.exception.code, CODE_DECODER_FAILED)
        self.assertEqual(ingestor.segments_fetched, 0)
        self.assertAlmostEqual(ingestor.buffered_seconds, 0.0)

    def test_default_fetcher_validates_url_before_connecting(self):
        from courselens_worker.live_audio_ingest import _pinned_fetch

        with self.assertRaises(LiveIngestError) as caught:
            _pinned_fetch("https://127.0.0.1/a.m3u8")
        self.assertEqual(caught.exception.code, CODE_NON_PUBLIC_ADDRESS)


class DecodeSegmentTests(unittest.TestCase):
    def test_real_wav_through_ffmpeg_decode(self):
        import shutil

        if shutil.which("ffmpeg") is None:
            raise unittest.SkipTest("ffmpeg required")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(8000)
                frames = [1000, -1000, 2000, -2000] * 1000
                handle.writeframes(struct.pack(f"<{len(frames)}h", *frames))
            payload = path.read_bytes()
            samples = decode_segment_to_pcm(payload, sample_rate=16000)
            # 8000Hz×0.25s（2000 帧）双声道 → 16k 单声道 0.25s。
            self.assertEqual(len(samples), 4000)
            self.assertTrue(all(-1.0 <= value <= 1.0 for value in samples))

    def test_garbage_payload_fails_closed(self):
        import shutil

        if shutil.which("ffmpeg") is None:
            raise unittest.SkipTest("ffmpeg required")
        with self.assertRaises(LiveIngestError) as caught:
            decode_segment_to_pcm(b"this is not media")
        self.assertEqual(caught.exception.code, CODE_DECODER_FAILED)


class TelemetryDisciplineTests(unittest.TestCase):
    def test_ingest_telemetry_carries_counters_only(self):
        lines: list[str] = []
        fetch, _calls = _playlist_fetch_factory(["seg0.ts"])
        ingestor = LiveAudioIngestor(
            PUBLIC_URL,
            resolver=_resolver_single("93.184.216.34"),
            fetch=fetch,
            decoder=(lambda data: [0.25] * 16000),
            ring_seconds=600.0,
            emit=lines.append,
        )
        ingestor.poll()
        self.assertEqual(len(lines), 1)
        line = lines[0]
        self.assertTrue(line.startswith("stage=streaming-ingest "))
        self.assertIn("segments=1", line)
        self.assertNotIn(PUBLIC_URL, line)
        self.assertNotIn("seg0", line)


if __name__ == "__main__":
    unittest.main()
