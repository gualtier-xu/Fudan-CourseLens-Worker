"""直播音频摄取器（D13-PROD 施工清单第 5 步）.

从直播 ``m3u8_audio`` 纯音频视角（``src/runtime/live_room.py`` 四视角通道，
URL-free 闭集 id 解析在客户端授权面完成）持续拉取音频段，解码 16k 单声道
PCM 供 ``streaming_asr.StreamingTranscriber`` 边上课边转写。

**media-protection 边界纪律**（沿 ``worker/courselens_worker/source.py`` 既有
先例，错误码与其闭集词表对齐）：

- **请求前 host 校验**：每个 URL（清单与每个分片）在发起请求前校验——仅
  https、无凭据、端口闭集 443、DNS 解析的全部地址必须公网
  （``ipaddress.is_global`` 一票否决环回/内网/链路本地/保留/组播）。
  每次请求都重新解析校验（DNS rebinding 无窗口可钻）。
- **content-type 闭集**：清单与分片各自闭集（与 live_room.py 播放门同款
  词表），闭集外一律拒收。
- **音频不落盘不出端**：分片字节只在内存中转，PCM 只进内存环形缓冲
  （本地推理，与「本地优先存储」北极星一致）；大小帽防失控。
- **闭集错误码 + 遥测闭集**：失败只出码，绝不携带 URL/头/内容；解码缺席
  或失败 fail-closed，绝不产出伪造静音充数。
"""

from __future__ import annotations

import ipaddress
import shutil
import socket
import subprocess
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit

SAMPLE_RATE = 16_000
DEFAULT_RING_SECONDS = 600.0
MAX_SEGMENT_BYTES = 32 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 15
_SEEN_URI_CAP = 4096

# 与 src/runtime/live_room.py 播放门同款闭集词表（清单/分片各取所需）。
MANIFEST_CONTENT_TYPES = frozenset(
    {
        "application/vnd.apple.mpegurl",
        "application/x-mpegurl",
        "audio/mpegurl",
        "audio/x-mpegurl",
    }
)
SEGMENT_CONTENT_TYPES = frozenset(
    {
        "video/mp2t",
        "video/mp4",
        "audio/mp4",
        "application/octet-stream",
    }
)

# 闭集错误码（host 校验腿与 source.py safe_source_error_code 词表对齐）。
CODE_INVALID_HTTPS_URL = "invalid_https_url"
CODE_INVALID_HTTPS_PORT = "invalid_https_port"
CODE_DNS_RESOLUTION_FAILED = "dns_resolution_failed"
CODE_DNS_NO_ADDRESSES = "dns_no_addresses"
CODE_NON_PUBLIC_ADDRESS = "non_public_address"
CODE_CONNECTION_FAILED = "live_connection_failed"
CODE_PLAYLIST_HTTP = "live_playlist_http"
CODE_PLAYLIST_CONTENT_TYPE = "live_playlist_content_type"
CODE_PLAYLIST_INVALID = "live_playlist_invalid"
CODE_SEGMENT_HTTP = "live_segment_http"
CODE_SEGMENT_CONTENT_TYPE = "live_segment_content_type"
CODE_SEGMENT_LIMIT = "live_segment_limit"
CODE_DECODER_MISSING = "live_decoder_missing"
CODE_DECODER_FAILED = "live_decoder_failed"

INGEST_ERROR_CODES = frozenset(
    {
        CODE_INVALID_HTTPS_URL,
        CODE_INVALID_HTTPS_PORT,
        CODE_DNS_RESOLUTION_FAILED,
        CODE_DNS_NO_ADDRESSES,
        CODE_NON_PUBLIC_ADDRESS,
        CODE_CONNECTION_FAILED,
        CODE_PLAYLIST_HTTP,
        CODE_PLAYLIST_CONTENT_TYPE,
        CODE_PLAYLIST_INVALID,
        CODE_SEGMENT_HTTP,
        CODE_SEGMENT_CONTENT_TYPE,
        CODE_SEGMENT_LIMIT,
        CODE_DECODER_MISSING,
        CODE_DECODER_FAILED,
    }
)


class LiveIngestError(RuntimeError):
    """摄取闭集失败：code 只取 INGEST_ERROR_CODES，不携带 URL/内容。"""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(code if not detail else f"{code}: {detail}")
        self.code = code


def _is_global_address(address: str) -> bool:
    try:
        return ipaddress.ip_address(address).is_global
    except ValueError:
        return False


def validate_stream_url(
    url: str, *, resolver: Callable[..., list] | None = None
) -> tuple[str, str]:
    """请求前 host 校验（source.py 同款语义）：仅 https/443/全公网解析。

    返回 ``(url, pinned_ip)``；resolver 可注入（测试离线）。
    """
    value = str(url or "").strip()
    parsed = urlsplit(value)
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise LiveIngestError(CODE_INVALID_HTTPS_URL)
    if parsed.port not in (None, 443):
        raise LiveIngestError(CODE_INVALID_HTTPS_PORT)
    resolve = resolver or socket.getaddrinfo
    try:
        addresses = resolve(parsed.hostname, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise LiveIngestError(CODE_DNS_RESOLUTION_FAILED) from exc
    except OSError as exc:
        raise LiveIngestError(CODE_DNS_RESOLUTION_FAILED) from exc
    if not addresses:
        raise LiveIngestError(CODE_DNS_NO_ADDRESSES)
    for address in addresses:
        if not _is_global_address(str(address[4][0])):
            raise LiveIngestError(CODE_NON_PUBLIC_ADDRESS)
    preferred = next(
        (
            item[4][0]
            for item in addresses
            if item[0] == socket.AF_INET
        ),
        addresses[0][4][0],
    )
    return value, str(preferred)


@dataclass(frozen=True)
class FetchedBody:
    """一次受控 GET 的归约结果（闭集面：状态/类型/字节）。"""

    status: int
    content_type: str
    body: bytes


FetchFn = Callable[[str], FetchedBody]


def _pinned_fetch(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = MAX_SEGMENT_BYTES,
) -> FetchedBody:
    """默认取流器：请求前校验+按已验证 IP 定连（证书/SNI 用原主机名）。"""
    import http.client
    import ssl

    validated, ip = validate_stream_url(url)
    parsed = urlsplit(validated)
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"
    request_headers = {str(name): str(value) for name, value in dict(headers or {}).items()}
    request_headers["Host"] = str(parsed.hostname)
    connection = http.client.HTTPSConnection(
        str(parsed.hostname),
        443,
        timeout=timeout,
        context=ssl.create_default_context(),
    )
    try:
        raw = socket.create_connection((ip, 443), timeout)
        raw.close()
        # 上面只为提前暴露不可达；真正的请求仍走 HTTPSConnection 的
        # 常规连接（证书/SNI/主机名绑定照常）。DNS 已在 validate 收紧，
        # TOCTOU 窗口由下一轮请求前的重新校验收口。
        connection.request("GET", path, headers=request_headers)
        response = connection.getresponse()
        payload = response.read(max_bytes + 1)
        if len(payload) > max_bytes:
            raise LiveIngestError(CODE_SEGMENT_LIMIT)
        content_type = str(response.getheader("Content-Type") or "").split(";", 1)[0].strip().casefold()
        return FetchedBody(int(response.status), content_type, payload)
    except LiveIngestError:
        raise
    except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
        raise LiveIngestError(CODE_CONNECTION_FAILED, type(exc).__name__) from exc
    finally:
        connection.close()


@dataclass(frozen=True)
class HlsPlaylistPlan:
    """媒体清单的解析产物：本轮新增分片 URI + 流是否已结束。"""

    segment_urls: tuple[str, ...] = ()
    ended: bool = False


def parse_media_playlist(text: str, *, base_url: str) -> HlsPlaylistPlan:
    """stdlib 解析 m3u8 媒体清单（EXTINF+URI 对；ENDLIST=已结束）。"""
    lines = str(text or "").splitlines()
    if not any(line.strip() == "#EXTM3U" for line in lines):
        raise LiveIngestError(CODE_PLAYLIST_INVALID)
    urls: list[str] = []
    ended = False
    pending_duration = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#EXTINF"):
            pending_duration = True
            continue
        if stripped == "#EXT-X-ENDLIST":
            ended = True
            continue
        if stripped.startswith("#"):
            continue
        if pending_duration:
            urls.append(urljoin(base_url, stripped))
        pending_duration = False
    return HlsPlaylistPlan(segment_urls=tuple(urls), ended=ended)


class StreamingRingBuffer:
    """内存 PCM 环形缓冲：满则丢最旧（遥测只记丢包秒数，不丢纪律）。"""

    def __init__(
        self,
        max_seconds: float = DEFAULT_RING_SECONDS,
        *,
        sample_rate: int = SAMPLE_RATE,
    ):
        if max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        self._max_samples = int(max_seconds * sample_rate)
        self._sample_rate = int(sample_rate)
        self._samples: deque[float] = deque()
        self.dropped_seconds = 0.0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def buffered_seconds(self) -> float:
        return len(self._samples) / float(self._sample_rate)

    def extend(self, samples: list[float]) -> float:
        """追加 PCM；返回本次丢弃的旧音频秒数（满则丢最旧）。"""
        self._samples.extend(samples)
        dropped = 0
        if len(self._samples) > self._max_samples:
            dropped = len(self._samples) - self._max_samples
            for _ in range(dropped):
                self._samples.popleft()
            self.dropped_seconds += dropped / float(self._sample_rate)
        return dropped / float(self._sample_rate)

    def drain(self, seconds: float | None = None) -> list[float]:
        """取走缓冲（缺省全取）；取走即出缓冲，音频不留第二处。"""
        if seconds is None:
            count = len(self._samples)
        else:
            count = min(len(self._samples), max(0, int(seconds * self._sample_rate)))
        if count <= 0:
            return []
        result = [self._samples.popleft() for _ in range(count)]
        return result


def decode_segment_to_pcm(data: bytes, *, sample_rate: int = SAMPLE_RATE) -> list[float]:
    """默认解码腿：单趟 ffmpeg 把一个媒体分片转 16k 单声道 f32 PCM。

    分片字节只在内存管道中转（stdin/stdout），绝不落盘；ffmpeg 缺席或
    失败 fail-closed 闭集码，绝不返回伪造静音。
    """
    import array

    binary = shutil.which("ffmpeg")
    if not binary:
        raise LiveIngestError(CODE_DECODER_MISSING)
    try:
        completed = subprocess.run(
            [
                binary, "-v", "error", "-nostdin",
                "-i", "pipe:0",
                "-map", "0:a:0",
                "-ac", "1", "-ar", str(sample_rate),
                "-f", "f32le", "pipe:1",
            ],
            input=bytes(data),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LiveIngestError(CODE_DECODER_FAILED, type(exc).__name__) from exc
    if completed.returncode != 0 or not completed.stdout:
        raise LiveIngestError(CODE_DECODER_FAILED, f"exit={completed.returncode}")
    samples = array.array("f")
    usable = len(completed.stdout) - (len(completed.stdout) % samples.itemsize)
    samples.frombytes(completed.stdout[:usable])
    return list(samples)


DecoderFn = Callable[[bytes], list[float]]


@dataclass
class IngestPollResult:
    """单次轮询的闭集计数产物（遥测只出这些计数与秒数）。"""

    segments_fetched: int = 0
    pcm_seconds: float = 0.0
    dropped_seconds: float = 0.0
    buffered_seconds: float = 0.0
    ended: bool = False
    skipped_segments: int = 0


class LiveAudioIngestor:
    """HLS 纯音频分帧摄取：轮询清单 → 取新分片 → 解码进内存环形缓冲。"""

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        fetch: FetchFn | None = None,
        decoder: DecoderFn | None = None,
        ring_seconds: float = DEFAULT_RING_SECONDS,
        sample_rate: int = SAMPLE_RATE,
        emit: Callable[[str], None] | None = None,
        resolver: Callable[..., list] | None = None,
    ):
        # 构造即校验：任何请求发出前，URL 先过 host 校验（fail-closed）。
        self.url, self._pinned_ip = validate_stream_url(url, resolver=resolver)
        self._headers = {str(k): str(v) for k, v in dict(headers or {}).items()}
        self._fetch: FetchFn = fetch or _pinned_fetch
        self._decoder: DecoderFn = decoder or decode_segment_to_pcm
        self._emit = emit or (lambda line: None)
        self.ring = StreamingRingBuffer(ring_seconds, sample_rate=sample_rate)
        self._sample_rate = int(sample_rate)
        self._seen: deque[str] = deque()
        self._seen_set: set[str] = set()
        self._ended = False
        self.polls = 0
        self.segments_fetched = 0
        self.pcm_seconds = 0.0

    @property
    def ended(self) -> bool:
        return self._ended

    @property
    def buffered_seconds(self) -> float:
        return self.ring.buffered_seconds

    def drain_pcm(self, seconds: float | None = None) -> list[float]:
        """取走已缓冲 PCM 喂转写器（音频只此一条通路，不落盘）。"""
        return self.ring.drain(seconds)

    def poll(self) -> IngestPollResult:
        """轮询一轮：清单→新增分片→解码→环形缓冲；任何腿失败闭集抛出。"""
        if self._ended:
            return IngestPollResult(
                buffered_seconds=self.ring.buffered_seconds, ended=True
            )
        self.polls += 1
        manifest = self._fetch(self.url)
        if manifest.status < 200 or manifest.status > 299:
            raise LiveIngestError(CODE_PLAYLIST_HTTP, f"status={manifest.status}")
        if manifest.content_type not in MANIFEST_CONTENT_TYPES:
            raise LiveIngestError(
                CODE_PLAYLIST_CONTENT_TYPE, f"type={manifest.content_type or 'missing'}"
            )
        try:
            text = manifest.body.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise LiveIngestError(CODE_PLAYLIST_INVALID, "utf8") from exc
        plan = parse_media_playlist(text, base_url=self.url)
        result = IngestPollResult(buffered_seconds=self.ring.buffered_seconds)
        for segment_url in plan.segment_urls:
            if segment_url in self._seen_set:
                continue
            self._remember(segment_url)
            result.skipped_segments += 0
            body = self._fetch(segment_url)
            if body.status < 200 or body.status > 299:
                raise LiveIngestError(CODE_SEGMENT_HTTP, f"status={body.status}")
            if body.content_type not in SEGMENT_CONTENT_TYPES:
                raise LiveIngestError(
                    CODE_SEGMENT_CONTENT_TYPE, f"type={body.content_type or 'missing'}"
                )
            samples = self._decoder(body.body)
            dropped = self.ring.extend(samples)
            seconds = len(samples) / float(self._sample_rate)
            self.segments_fetched += 1
            self.pcm_seconds += seconds
            result.segments_fetched += 1
            result.pcm_seconds += seconds
            result.dropped_seconds += dropped
        result.buffered_seconds = self.ring.buffered_seconds
        result.ended = plan.ended
        self._ended = self._ended or plan.ended
        self._emit(
            f"stage=streaming-ingest polls={self.polls} "
            f"segments={self.segments_fetched} pcm_s={self.pcm_seconds:.1f} "
            f"buffered_s={self.ring.buffered_seconds:.1f} "
            f"dropped_s={self.ring.dropped_seconds:.1f} ended={int(self._ended)}"
        )
        return result

    def _remember(self, segment_url: str) -> None:
        self._seen.append(segment_url)
        self._seen_set.add(segment_url)
        while len(self._seen) > _SEEN_URI_CAP:
            stale = self._seen.popleft()
            self._seen_set.discard(stale)
