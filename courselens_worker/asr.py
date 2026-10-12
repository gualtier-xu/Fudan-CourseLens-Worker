"""CPU-only ASR over bounded transient PCM chunks.

Subtitle anchors come from frame-energy voiced regions inside each decoding
window instead of the full window, so silence never produces a cue and
continuous speech degrades to one bounded region.  Recognized segments carry
optional evidence.v1 provenance stamped only when the decoded-PCM fingerprint
chain is verifiable for the whole run.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests
import sherpa_onnx

from .llm import LLMError

from shared.evidence_contract import (
    NAMESPACE_SEGMENT,
    NAMESPACE_SOURCE,
    canonical_json,
    compute_id,
)

from .formats import normalize_segments
from .platform_session import (
    PlatformSessionError,
    _RETRYABLE_LOGIN_ERRORS,
    _RETRYABLE_SESSION_ERRORS,
)
from .source import (
    MediaResponseProfile,
    pinned_media_proxy,
)

SAMPLE_RATE = 16_000
PCM_CHUNK_SECONDS = 10 * 60
ASR_WINDOW_SECONDS = 30
# decode_streams keeps every stream of one call inside a single activation
# whose memory scales with the batch's total audio seconds, so a 600s chunk
# of continuous speech decoded as one batch exhausted 16GB hosted runners.
# Capping each call at the per-stream window ceiling bounds that activation
# to the envelope proven safe on 4-core runners.
ASR_DECODE_BATCH_SECONDS = 30

# 精修管线 backend 序列策略（ASRBENCH-1 A5 立项，M4-ENABLE-1 U3 翻默认）：
# 「粗识别, 精识别」两元序列默认即 M4（sensevoice 主识别 + paraformer 精修），
# 环境变量可覆写；旧链如需回退走 git revert，不留运行时后门。
# SUBTITLE-DEEP-1 Phase B：``zipformer``（zipformer-transducer + 术语热词文件，
# BENCH-ASR-1 实证热词 -35% 术语错）加入可选精修腿；默认序列不变，热词路径
# 仅由 SUBTITLE_BACKENDS 显式启用。
SUPPORTED_ASR_BACKENDS = ("sensevoice", "paraformer", "zipformer")
DEFAULT_SUBTITLE_BACKENDS = "sensevoice,paraformer"
# BENCH-ASR-1 热词腿配方钉：hotwords_score=2.0 + modified_beam_search（ziptrans
# 冷→热唯一变量对照 0.207→0.135）。热词文件缺失时按冷腿默认解码。
ZIPFORMER_HOTWORDS_SCORE = 2.0
ZIPFORMER_HOTWORD_DECODING = "modified_beam_search"
# 热词文件条目上限：课程词表（OCR 高频候选）可能含上下文短语，按频序截断。
ASR_HOTWORD_LIMIT = 100

# AS12（第五十二案）：平台原生文稿可代 SenseVoice 粗识别腿——只当校对交替
# 源，绝不直接成为字幕输出（用户拍板 2026-09-23：平台文稿差，质量由精识别
# Paraformer + DeepSeek 校对链把守）。platform-first 命中文稿且时间覆盖达阈
# 即整讲跳过粗腿；COURSELENS_ASR_ROUGH_SOURCE=sensevoice 为杀开关强制旧双
# 模链。获取失败/空文稿/覆盖不足一律整讲回落旧链：失败=降级，绝不失败任务。
ASR_ROUGH_SOURCE_ENV = "COURSELENS_ASR_ROUGH_SOURCE"
ASR_ROUGH_SOURCE_PLATFORM = "platform"
ASR_ROUGH_SOURCE_SENSEVOICE = "sensevoice"
# 文稿时间覆盖门：平台段并集至少盖住待转写时长的这一比例才跳过粗腿。钉死
# 常量、无 env 后门；实测校准随 U2 护栏数字记录。
PLATFORM_TRANSCRIPT_MIN_COVERAGE = 0.8
# U6（夜批 8 实测修正）：平台 cue 是句级的，句间自然停顿（1-5s）会把裸并集
# 覆盖压到阈值之下——u2c 产品路径 E2E 抓到该语义洞。合并容忍=小于该 gap 的
# 停顿粘合后再算覆盖：正常停顿不再误伤，真正的长段缺失（>30s）仍如实算洞。
PLATFORM_TRANSCRIPT_MERGE_GAP_MS = 30_000
# 闭集回落原因（只进 metrics 与 stage=rough-source 遥测行，绝不外扩）。
ROUGH_SOURCE_FALLBACK_REASONS = frozenset({
    "env_disabled", "transcript_fetch_failed", "transcript_empty",
    "platform_transcript_missing", "coverage_low", "legacy_checkpoint",
})

# Frame-energy voice activity: deterministic, NumPy-only, no model.  The one
# documented calibration knob is COURSELENS_ASR_ENERGY_RATIO (voiced threshold
# as a ratio over the window noise floor); everything else is a fixed default
# chosen conservatively so noisy recordings expand toward the old full-window
# behavior instead of losing speech.
VAD_FRAME_SECONDS = 0.025
VAD_HOP_SECONDS = 0.010
VAD_NOISE_PERCENTILE = 10.0
VAD_SILENCE_FLOOR_RMS = 1e-4
VAD_PEAK_GUARD_RATIO = 0.1
VAD_MERGE_GAP_SECONDS = 0.4
VAD_PAD_SECONDS = 0.15
VAD_MIN_REGION_SECONDS = 0.1
VAD_MAX_REGION_SECONDS = float(ASR_WINDOW_SECONDS)
ASR_ENERGY_RATIO_ENV = "COURSELENS_ASR_ENERGY_RATIO"
ASR_ENERGY_RATIO_DEFAULT = 3.0
ASR_ENERGY_RATIO_MIN = 1.5
ASR_ENERGY_RATIO_MAX = 10.0

# V4NONTHINK-1 件7：silero-vad 段界（默认关）。攻「跨段词中断裂」根因家族
# （能量 VAD 在弱音节处断界，如「考试方|面」）。env=COURSELENS_ASR_VAD_ENGINE:
#   energy（缺省）=现役帧能量 VAD，行为逐位不变；silero=SileroVadModelConfig，
#   模型经 SILERO_MODEL_DIR 注入（install_models 钉）。模型缺席/引擎名未知一律
#   回落能量 VAD（失败=降级，绝不失败任务），闭集遥测记账。
VAD_ENGINE_ENV = "COURSELENS_ASR_VAD_ENGINE"
VAD_ENGINE_ENERGY = "energy"
VAD_ENGINE_SILERO = "silero"
SILERO_MODEL_DIR_ENV = "SILERO_MODEL_DIR"
SILERO_VAD_THRESHOLD = 0.5
SILERO_VAD_MIN_SPEECH_SECONDS = VAD_MIN_REGION_SECONDS
SILERO_VAD_MIN_SILENCE_SECONDS = VAD_MERGE_GAP_SECONDS
SILERO_VAD_WINDOW_SAMPLES = 512  # 官方 release silero_vad.onnx（v5，643KB）实窗


def vad_engine() -> str:
    raw = os.environ.get(VAD_ENGINE_ENV, "").strip().lower()
    return raw if raw in {VAD_ENGINE_ENERGY, VAD_ENGINE_SILERO} else VAD_ENGINE_ENERGY


def silero_model_path() -> Path | None:
    value = os.environ.get(SILERO_MODEL_DIR_ENV, "").strip()
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_dir():
        candidate = candidate / "silero_vad.onnx"
    return candidate if candidate.is_file() else None

# Evidence provenance stamped on complete-run output.  The fingerprint hashes
# only the decoded PCM representation; URLs, secrets, and course identifiers
# never enter it, and no original-media hash is claimed.
PRODUCER_ID = "courselens-worker"
PCM_FINGERPRINT_DOMAIN = b"courselens-pcm-fingerprint-v1"
_PCM_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


def asr_energy_ratio() -> float:
    """Resolve the single calibration knob; anything invalid falls back."""
    raw = os.environ.get(ASR_ENERGY_RATIO_ENV)
    if raw is None or str(raw).strip() == "":
        return ASR_ENERGY_RATIO_DEFAULT
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return ASR_ENERGY_RATIO_DEFAULT
    if not math.isfinite(value) or not (
        ASR_ENERGY_RATIO_MIN <= value <= ASR_ENERGY_RATIO_MAX
    ):
        return ASR_ENERGY_RATIO_DEFAULT
    return value


def detect_voiced_regions(
    samples: "np.ndarray",
    *,
    sample_rate: int = SAMPLE_RATE,
    energy_ratio: float | None = None,
    merge_gap_seconds: float = VAD_MERGE_GAP_SECONDS,
    pad_seconds: float = VAD_PAD_SECONDS,
    min_region_seconds: float = VAD_MIN_REGION_SECONDS,
    max_region_seconds: float = VAD_MAX_REGION_SECONDS,
) -> list[tuple[int, int]]:
    """Bounded voiced (start_sample, end_sample) regions inside one window.

    Frames are 25 ms with a 10 ms hop; a frame is voiced when its RMS exceeds
    max(noise_floor * energy_ratio, silence_floor) with the threshold capped at
    peak * guard so continuous speech can never push the threshold above
    itself.  Runs shorter than ``min_region_seconds`` are dropped, gaps below
    ``merge_gap_seconds`` are merged, regions longer than
    ``max_region_seconds`` are split, and padding is bounded by the window and
    by half of each neighboring gap so regions stay ordered and disjoint.
    """
    if energy_ratio is None:
        energy_ratio = asr_energy_ratio()
    total = int(len(samples))
    frame = max(1, int(VAD_FRAME_SECONDS * sample_rate))
    hop = max(1, int(VAD_HOP_SECONDS * sample_rate))
    if total < frame:
        return []
    squared = np.square(samples, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    frame_count = (total - frame) // hop + 1
    frame_starts = np.arange(frame_count, dtype=np.int64) * hop
    energies = np.sqrt((cumulative[frame_starts + frame] - cumulative[frame_starts]) / frame)
    noise_floor = float(np.percentile(energies, VAD_NOISE_PERCENTILE))
    peak = float(np.max(energies))
    threshold = max(noise_floor * energy_ratio, VAD_SILENCE_FLOOR_RMS)
    threshold = min(threshold, max(peak * VAD_PEAK_GUARD_RATIO, VAD_SILENCE_FLOOR_RMS))
    voiced = energies >= threshold
    regions: list[list[int]] = []
    index = 0
    while index < frame_count:
        if not voiced[index]:
            index += 1
            continue
        stop = index
        while stop + 1 < frame_count and voiced[stop + 1]:
            stop += 1
        regions.append([int(frame_starts[index]), min(int(frame_starts[stop]) + frame, total)])
        index = stop + 1
    min_samples = max(1, int(min_region_seconds * sample_rate))
    regions = [region for region in regions if region[1] - region[0] >= min_samples]
    merged: list[list[int]] = []
    gap_samples = int(merge_gap_seconds * sample_rate)
    for region in regions:
        if merged and region[0] - merged[-1][1] < gap_samples:
            merged[-1][1] = max(merged[-1][1], region[1])
        else:
            merged.append(region)
    max_samples = max(1, int(max_region_seconds * sample_rate))
    bounded: list[list[int]] = []
    for start, end in merged:
        cursor = start
        while end - cursor > max_samples:
            bounded.append([cursor, cursor + max_samples])
            cursor += max_samples
        bounded.append([cursor, end])
    pad_samples = int(pad_seconds * sample_rate)
    padded: list[tuple[int, int]] = []
    for position, (start, end) in enumerate(bounded):
        previous_end = padded[-1][1] if padded else 0
        next_start = bounded[position + 1][0] if position + 1 < len(bounded) else total
        region_start = max(previous_end, start - min(pad_samples, (start - previous_end) // 2))
        region_end = min(next_start, end + min(pad_samples, (next_start - end) // 2))
        padded.append((region_start, max(region_start, region_end)))
    return padded


def _finalize_regions(
    regions: list[list[int]],
    total: int,
    *,
    min_region_samples: int,
    merge_gap_samples: int,
    max_region_samples: int,
    pad_samples: int,
    sample_rate: int,
) -> list[tuple[int, int]]:
    """Shared bounded-region discipline (min filter → merge → max split → pad).

    与 detect_voiced_regions 的后段纪律逐位同构（min/merge/max/pad 帽与邻界
    对半收缩），供 silero 路径复用；能量路径保留自身内联实现零回归。
    """
    bounded = [region for region in regions if region[1] - region[0] >= min_region_samples]
    merged: list[list[int]] = []
    for region in bounded:
        if merged and region[0] - merged[-1][1] < merge_gap_samples:
            merged[-1][1] = max(merged[-1][1], region[1])
        else:
            merged.append(region)
    split: list[list[int]] = []
    for start, end in merged:
        cursor = start
        while end - cursor > max_region_samples:
            split.append([cursor, cursor + max_region_samples])
            cursor += max_region_samples
        split.append([cursor, end])
    padded: list[tuple[int, int]] = []
    for position, (start, end) in enumerate(split):
        previous_end = padded[-1][1] if padded else 0
        next_start = split[position + 1][0] if position + 1 < len(split) else total
        region_start = max(previous_end, start - min(pad_samples, (start - previous_end) // 2))
        region_end = min(next_start, end + min(pad_samples, (next_start - end) // 2))
        padded.append((region_start, max(region_start, region_end)))
    return padded


def _native_token_timing(result: Any, start_ms: int, end_ms: int) -> list[list[Any]] | None:
    """Absolute [text, start_ms, None] triples when native timing is well-shaped.

    Any doubt omits the whole set: tokens and timestamps must exist with equal
    non-zero length, each stamp must parse, stay inside the segment anchors,
    and be non-descending.  Timing is never clamped or otherwise fabricated.
    """
    tokens = getattr(result, "tokens", None)
    timestamps = getattr(result, "timestamps", None)
    if not isinstance(tokens, (list, tuple)) or not isinstance(timestamps, (list, tuple)):
        return None
    if not tokens or len(tokens) != len(timestamps):
        return None
    output: list[list[Any]] = []
    previous = start_ms - 1
    for token, stamp in zip(tokens, timestamps):
        try:
            moment = start_ms + int(round(float(stamp) * 1000.0))
        except (TypeError, ValueError, OverflowError):
            return None
        text = str(token or "").strip()
        if not text or moment < previous or moment < start_ms or moment > end_ms:
            return None
        output.append([text, moment, None])
        previous = moment
    return output


class ASRError(RuntimeError):
    pass


def _decode_failure(stderr: str) -> ASRError:
    """Classify bounded FFmpeg diagnostics without exposing their text."""
    value = str(stderr or "").casefold()
    for status, message in (
        ("401", "authorized media request returned HTTP 401"),
        ("403", "authorized media request returned HTTP 403"),
        ("404", "authorized media request returned HTTP 404"),
        ("429", "authorized media request returned HTTP 429"),
    ):
        if (
            f"server returned {status}" in value
            or f"http error {status}" in value
            or f"returned error: {status}" in value
        ):
            return ASRError(message)
    if "server returned 5" in value or "http error 5" in value:
        return ASRError("authorized media request returned HTTP 5xx")
    if "moov atom" in value:
        return ASRError("authorized media is missing a readable MP4 index")
    if "invalid data" in value:
        return ASRError("authorized media format was rejected by ffmpeg")
    return ASRError("ffmpeg could not decode the authorized media stream")


def _media_response_error(profile: MediaResponseProfile) -> ASRError | None:
    http_messages = {
        "http_401": "authorized media request returned HTTP 401",
        "http_403": "authorized media request returned HTTP 403",
        "http_404": "authorized media request returned HTTP 404",
        "http_429": "authorized media request returned HTTP 429",
        "http_4xx": "authorized media request returned HTTP 4xx",
        "http_5xx": "authorized media request returned HTTP 5xx",
        "http_3xx": "authorized media request returned an unsupported redirect",
        "http_other": "authorized media request returned an unsupported status",
    }
    if profile.http in http_messages:
        return ASRError(http_messages[profile.http])
    if profile.content == "content_html" or profile.magic == "magic_html":
        return ASRError("authorized media response contained HTML")
    if profile.content == "content_json" or profile.magic == "magic_json":
        return ASRError("authorized media response contained JSON")
    if profile.magic != "magic_iso_bmff":
        return ASRError("authorized media signature was rejected")
    return None


def _drain_bounded(pipe, output: bytearray, *, limit: int = 16 * 1024) -> None:
    while True:
        block = pipe.read(4096)
        if not block:
            return
        remaining = limit - len(output)
        if remaining > 0:
            output.extend(block[:remaining])


def _run_bounded_process(
    command: list[str],
    *,
    timeout: int,
    capture_stdout: bool,
) -> tuple[int, bytes, bytes]:
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    stdout = bytearray()
    stderr = bytearray()
    readers: list[threading.Thread] = []
    if process.stdout is not None:
        readers.append(threading.Thread(
            target=_drain_bounded,
            args=(process.stdout, stdout),
            name="courselens-media-stdout",
            daemon=True,
        ))
    if process.stderr is not None:
        readers.append(threading.Thread(
            target=_drain_bounded,
            args=(process.stderr, stderr),
            name="courselens-media-stderr",
            daemon=True,
        ))
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + max(1, int(timeout))
    try:
        remaining = max(1, int(deadline - time.monotonic()))
        process.wait(timeout=remaining)
    except Exception:
        process.kill()
        process.wait()
        raise
    finally:
        for reader in readers:
            reader.join(timeout=5)
    return int(process.returncode or 0), bytes(stdout), bytes(stderr)


def _run_media_proxy(
    source: dict[str, Any],
    command: Callable[[str], list[str]],
    *,
    timeout: int,
    capture_stdout: bool,
) -> tuple[int, bytes, bytes]:
    with pinned_media_proxy(source) as proxy:
        return _run_bounded_process(
            command(proxy.url),
            timeout=timeout,
            capture_stdout=capture_stdout,
        )


def _probe_duration(source: dict[str, Any]) -> float:
    # 第二十一案同族（A5 邻接扫）：时长探测发生在分块代理建立之前，学校单
    # 会话作废/签名过期在这里同样秒败。有界=至多两次探测，每次起手都会重取
    # 会话材料（pinned_media_proxy 检测到 _refresh_source 即先刷新授权）；
    # 仍败按闭集码如实失败。超时不重试——那不是秒败族，重试只放大等待。
    for _attempt in (0, 1):
        try:
            returncode, stdout, _ = _run_media_proxy(
                source,
                lambda media_url: [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=nw=1:nk=1", "-i", media_url,
                ],
                timeout=120,
                capture_stdout=True,
            )
        except (subprocess.TimeoutExpired, OSError):
            raise ASRError("authorized media duration probe timed out")
        try:
            duration = float(stdout.decode("ascii", errors="ignore").strip())
        except (TypeError, ValueError):
            duration = 0.0
        if returncode == 0 and duration > 0:
            return duration
    raise ASRError("authorized media duration could not be determined")


def subtitle_backend_sequence() -> list[str]:
    raw = os.environ.get("SUBTITLE_BACKENDS") or DEFAULT_SUBTITLE_BACKENDS
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if (
        len(names) != 2
        or len(set(names)) != 2
        or any(name not in SUPPORTED_ASR_BACKENDS for name in names)
    ):
        raise ASRError("subtitle backend sequence is invalid")
    return names


class RecognizerPool:
    def __init__(
        self,
        sensevoice_dir: Path,
        paraformer_dir: Path | None = None,
        *,
        threads: int = 4,
        zipformer_dir: Path | None = None,
    ):
        self.sensevoice_dir = sensevoice_dir
        self.paraformer_dir = paraformer_dir
        self.zipformer_dir = zipformer_dir
        self.threads = max(1, min(4, int(threads)))
        self._recognizers: dict[str, Any] = {}
        # 术语热词文件由 transcribe 在任务临时目录内落盘后挂上（懒加载语义：
        # 只在 zipformer 腿首次构建 recognizer 时被读取）。
        self.hotwords_file: Path | None = None

        self.silero_model_path: Path | None = None
        self._silero_vad: Any = None

    def silero_ready(self) -> bool:
        return vad_engine() == VAD_ENGINE_SILERO and self.silero_model_path is not None

    def _voiced_regions(
        self,
        window: "np.ndarray",
        *,
        energy_ratio: float,
    ) -> list[tuple[int, int]]:
        """Dispatch per configured engine; silero falls back to energy fail-closed."""
        if not self.silero_ready():
            return detect_voiced_regions(window, energy_ratio=energy_ratio)
        try:
            return self._silero_regions(window)
        except Exception as exc:  # noqa: BLE001 - 模型/推理任何异常都回落能量 VAD
            _emit_telemetry(f"stage=silero-vad-fallback reason={type(exc).__name__}")
            self._silero_vad = None
            return detect_voiced_regions(window, energy_ratio=energy_ratio)

    def _silero_regions(self, window: "np.ndarray") -> list[tuple[int, int]]:
        if self._silero_vad is None:
            config = sherpa_onnx.VadModelConfig()
            config.silero_vad.model = str(self.silero_model_path)
            config.silero_vad.threshold = SILERO_VAD_THRESHOLD
            config.silero_vad.min_speech_duration = SILERO_VAD_MIN_SPEECH_SECONDS
            config.silero_vad.min_silence_duration = SILERO_VAD_MIN_SILENCE_SECONDS
            config.silero_vad.window_size = SILERO_VAD_WINDOW_SAMPLES
            config.sample_rate = SAMPLE_RATE
            self._silero_vad = sherpa_onnx.VoiceActivityDetector(
                config, buffer_size_in_seconds=120,
            )
        vad = self._silero_vad
        samples = np.asarray(window, dtype=np.float32)
        step = SILERO_VAD_WINDOW_SAMPLES
        for start in range(0, len(samples), step):
            vad.accept_waveform(samples[start:start + step])
        vad.flush()
        regions: list[list[int]] = []
        while not vad.empty():
            segment = vad.front
            start = int(segment.start)
            end = start + len(segment.samples)
            if end > start:
                regions.append([start, min(end, len(samples))])
            vad.pop()
        vad.reset()
        return _finalize_regions(
            regions,
            len(samples),
            min_region_samples=max(1, int(VAD_MIN_REGION_SECONDS * SAMPLE_RATE)),
            merge_gap_samples=int(VAD_MERGE_GAP_SECONDS * SAMPLE_RATE),
            max_region_samples=max(1, int(VAD_MAX_REGION_SECONDS * SAMPLE_RATE)),
            pad_samples=int(VAD_PAD_SECONDS * SAMPLE_RATE),
            sample_rate=SAMPLE_RATE,
        )

    @staticmethod
    def _model(directory: Path) -> Path:
        for name in ("model.int8.onnx", "model.onnx"):
            path = directory / name
            if path.is_file():
                return path
        raise ASRError("configured ASR model directory is incomplete")

    @staticmethod
    def _transducer_files(directory: Path) -> tuple[Path, Path, Path]:
        """Resolve (encoder, decoder, joiner), preferring int8 variants."""
        chosen: list[Path] = []
        for role in ("encoder", "decoder", "joiner"):
            candidates = sorted(directory.glob(f"{role}-*.onnx"))
            int8 = [path for path in candidates if ".int8." in path.name]
            pool = int8 or candidates
            if not pool:
                raise ASRError("configured ASR model directory is incomplete")
            chosen.append(pool[0])
        return chosen[0], chosen[1], chosen[2]

    def zipformer_ready(self) -> bool:
        if self.zipformer_dir is None or not self.zipformer_dir.is_dir():
            return False
        try:
            self._transducer_files(self.zipformer_dir)
        except ASRError:
            return False
        return (self.zipformer_dir / "tokens.txt").is_file()

    def get(self, backend: str):
        if backend in self._recognizers:
            return self._recognizers[backend]
        directories = {
            "sensevoice": self.sensevoice_dir,
            "paraformer": self.paraformer_dir,
            "zipformer": self.zipformer_dir,
        }
        directory = directories.get(backend)
        if directory is None:
            if backend not in SUPPORTED_ASR_BACKENDS:
                raise ASRError("unsupported ASR backend")
            raise ASRError(f"{backend} model directory is not configured")
        if backend == "zipformer":
            encoder, decoder, joiner = self._transducer_files(directory)
            tokens = directory / "tokens.txt"
        else:
            model = self._model(directory)
            tokens = directory / "tokens.txt"
        if not tokens.is_file():
            raise ASRError("configured ASR token file is missing")
        if backend == "sensevoice":
            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model), tokens=str(tokens), num_threads=self.threads,
                use_itn=True, debug=False, provider="cpu",
            )
        elif backend == "paraformer":
            # 参数名对 sherpa-onnx 1.13.4 官方绑定实核（from_paraformer）。
            recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=str(model), tokens=str(tokens), num_threads=self.threads,
                debug=False, provider="cpu",
            )
        elif backend == "zipformer":
            # SUBTITLE-DEEP-1 Phase B：热词经 from_transducer 公开工厂参数注入
            # （公开工厂直接支持热词参数；BENCH 证实的 pybind 直构造配方只属于
            # paraformer 静默忽略热词的死路，不再需要）。
            hotwords_file = self.hotwords_file
            recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(encoder), decoder=str(decoder), joiner=str(joiner),
                tokens=str(tokens), num_threads=self.threads,
                debug=False, provider="cpu",
                hotwords_file=str(hotwords_file) if hotwords_file else "",
                hotwords_score=ZIPFORMER_HOTWORDS_SCORE if hotwords_file else 1.0,
                decoding_method=ZIPFORMER_HOTWORD_DECODING if hotwords_file else "default",
            )
        else:
            raise ASRError("unsupported ASR backend")
        self._recognizers[backend] = recognizer
        return recognizer

    def transcribe_pcm(self, path: Path, backend: str, *, offset_seconds: float) -> list[dict[str, Any]]:
        recognizer = self.get(backend)
        samples = np.memmap(path, dtype=np.float32, mode="r")
        window_samples = SAMPLE_RATE * ASR_WINDOW_SECONDS
        energy_ratio = asr_energy_ratio()
        base = int(offset_seconds * 1000)
        work: list[tuple[Any, int, int]] = []
        for start in range(0, len(samples), window_samples):
            end = min(len(samples), start + window_samples)
            if end - start < SAMPLE_RATE // 2:
                continue
            window = samples[start:end]
            for region_start, region_end in self._voiced_regions(
                window, energy_ratio=energy_ratio,
            ):
                stream = recognizer.create_stream()
                stream.accept_waveform(
                    SAMPLE_RATE, np.asarray(window[region_start:region_end])
                )
                work.append((
                    stream,
                    base + int((start + region_start) / SAMPLE_RATE * 1000),
                    base + int((start + region_end) / SAMPLE_RATE * 1000),
                ))
        if not work:
            del samples
            return []
        if hasattr(recognizer, "decode_streams"):
            # Batches keep work order and each stream is decoded exactly once,
            # so batching never changes segment order or anchors.
            batch: list[Any] = []
            batch_seconds = 0.0
            for stream, start_ms, end_ms in work:
                stream_seconds = max(0, end_ms - start_ms) / 1000.0
                if batch and batch_seconds + stream_seconds > ASR_DECODE_BATCH_SECONDS:
                    recognizer.decode_streams(batch)
                    batch = []
                    batch_seconds = 0.0
                batch.append(stream)
                batch_seconds += stream_seconds
            if batch:
                recognizer.decode_streams(batch)
        else:
            for stream, _, _ in work:
                recognizer.decode_stream(stream)
        segments: list[dict[str, Any]] = []
        for stream, start_ms, end_ms in work:
            result = stream.result
            text = " ".join(str(result.text or "").replace("<sil>", "").split()).strip()
            if not text:
                continue
            segment: dict[str, Any] = {"start_ms": start_ms, "end_ms": end_ms, "text": text}
            tokens = _native_token_timing(result, start_ms, end_ms)
            if tokens is not None:
                segment["tokens"] = tokens
            segments.append(segment)
        del samples
        return normalize_segments(segments)


def _ffmpeg_proxy_command(
    target: Path,
    media_url: str,
    *,
    offset: float,
    duration: float,
) -> list[str]:
    return [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{offset:.3f}", "-i", media_url, "-t", f"{duration:.3f}",
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-y", str(target),
    ]


# 第二十案(2026-09-21 真机定案)：chunk6 取流 0 字节秒败（media_format_rejected
# 族），与学校单会话作废 runner 媒体会话/签名过期强相关。该族不再立即整单
# 失败：每次失败先有界重取会话材料（刷新授权，平台侧自带重登退避），再重试
# 当前块；次数与退避有界，穷尽后按最后一个闭集码如实失败。4xx/404/429 与
# moov 缺索引属确定性拒绝，不进重试族。
_MEDIA_RETRY_BACKOFF_SECONDS = (2.0, 5.0)
_MEDIA_RETRY_MESSAGES = frozenset({
    "authorized media format was rejected by ffmpeg",
    "ffmpeg could not decode the authorized media stream",
    "authorized media upstream connection failed",
    "authorized media request returned HTTP 5xx",
})

# MEDIA-FIX-PREFETCH-1（2026-10-02 452282 定因）：上游提前断流时，代理转发
# 循环对 read(amt) 的静默空串收尾无从分辨、ffmpeg 对传输层提前 EOF 按正常
# 输入结束 exit 0（stderr 的 demuxing I/O error 在成功路径不可见），截断 PCM
# 由此静默过关，迟至空切片才以错位的 decode 码上行。完整性门=期望 PCM 字节
# （时长×采样率×4，1s 容差吸收容器时长元数据与实际解码样本的常规毫秒级偏差）
# fail-closed；分块短块/空切片同码归真，不再错标 decode 失败。
_MEDIA_PREFETCH_INCOMPLETE_MESSAGE = "authorized media prefetch was incomplete"

# P55（第五十五案）：块边界授权刷新此前裸奔——真机事故 2026-09-24：粗腿与前两
# 块正常，chunk 2 边界 refresh 抛 platform_auth_context_missing 整单即死，且死在
# decode-start 遥测之前（死窗无痕）。边界刷新与块内媒体重取同族：同量退避梯
# （2s/5s，共 3 次尝试）+ 逐次遥测；重试闭集=登录梯∪会话梯的瞬态码并集，
# 确定性拒绝（platform_media_missing、platform_challenge_required 等）不进梯、
# 一次即败不放大等待。梯尽按最后闭集码如实失败（worker_failed 语义不变）。
_SOURCE_REFRESH_ATTEMPTS = len(_MEDIA_RETRY_BACKOFF_SECONDS) + 1
_SOURCE_REFRESH_RETRY_CODES = frozenset(_RETRYABLE_LOGIN_ERRORS | _RETRYABLE_SESSION_ERRORS)


def _refresh_media_authorization(
    proxy: Any,
    *,
    chunk: int,
    elapsed: Callable[[], int],
    label: str | None = None,
) -> None:
    """Bounded refresh ladder for one chunk-boundary authorization refresh.

    ``label`` rebrands the telemetry face for non-chunk callers (the prefetch
    refetch path) without changing the chunk-boundary semantics.
    """
    for attempt in range(_SOURCE_REFRESH_ATTEMPTS):
        try:
            proxy.refresh_source()
            return
        except PlatformSessionError as exc:
            if str(exc) not in _SOURCE_REFRESH_RETRY_CODES:
                raise
            failed = attempt == _SOURCE_REFRESH_ATTEMPTS - 1
            face = label if label is not None else f"chunk={chunk}"
            _emit_telemetry(
                f"stage=source-refresh-{'failed' if failed else 'retry'} "
                f"{face} attempt={attempt + 1} reason={exc} elapsed={elapsed()}"
            )
            if failed:
                raise
            time.sleep(_MEDIA_RETRY_BACKOFF_SECONDS[attempt])
    raise AssertionError("unreachable")


class _RangeUnsupported(Exception):
    """Upstream answered a ranged GET with 200: Range is not honored."""


# N12（2026-10-07，WORKER-R2）：媒体预取并行 Range 分段——慢网兜底。
# 生产实测单流带宽方差 2.1↔8.7MB/s（286s ↔ 44s/380MB）：慢网端预取占媒体
# 墙钟 15-25%。上游认 Range 时按 Content-Range 总量均分 ≤4 段并行拉取、
# 拼接出逐字节等价容器，再走同一条 ffmpeg 命令与同一期望字节完整性门
# （MEDIA-FIX-PREFETCH-1 门零改动）；探测不支持 Range / 低于并行门槛时
# 逐字回落原单流路径，零行为回退。段级唯一截断防线=收到字节数恰等于
# 请求长度（代理转发循环对上游提前断流静默收尾，与 452282 同族）；
# 段级瞬态失败按既有媒体退避梯有界重试（每次重试=代理内新一次授权刷新
# 机会），梯尽按 media_prefetch_incomplete 如实失败，由 transcribe 层
# 既有重取梯（refresh+恰一次重跑）接手。
_MEDIA_PREFETCH_SEGMENTS = 4
_MEDIA_PREFETCH_MIN_SEGMENT_BYTES = 8 * 1024 * 1024
_MEDIA_PREFETCH_SEGMENT_ATTEMPTS = len(_MEDIA_RETRY_BACKOFF_SECONDS) + 1
_CONTENT_RANGE_TOTAL = re.compile(r"^bytes\s+\d+-\d+/(\d+)$", re.IGNORECASE)


def _probe_range_total(media_url: str) -> int:
    """Return total container bytes when the source honors Range; else 0.

    探测失败（连接异常/非 206/Content-Range 缺失或无总量）一律返回 0，
    由调用方逐字回落单流路径——失败分类与闭集码语义全部留给既有路径。
    """
    try:
        with requests.get(
            media_url,
            headers={"Range": "bytes=0-0"},
            timeout=(15, 30),
            stream=True,
        ) as response:
            if int(response.status_code) != 206:
                return 0
            match = _CONTENT_RANGE_TOTAL.match(
                str(response.headers.get("Content-Range") or "").strip()
            )
    except requests.RequestException:
        return 0
    return int(match.group(1)) if match else 0


def _fetch_media_segment(
    media_url: str,
    index: int,
    start: int,
    end: int,
    destination: Path,
    *,
    deadline: float,
) -> None:
    """Fetch one container byte range through the loopback proxy, byte-exact."""
    expected = end - start + 1
    reason = "unknown"
    for attempt in range(_MEDIA_PREFETCH_SEGMENT_ATTEMPTS):
        if attempt:
            _emit_telemetry(
                "stage=media-prefetch-segment-retry "
                f"segment={index} attempt={attempt + 1} reason={reason}"
            )
            time.sleep(
                _MEDIA_RETRY_BACKOFF_SECONDS[
                    min(attempt - 1, len(_MEDIA_RETRY_BACKOFF_SECONDS) - 1)
                ]
            )
        try:
            received = 0
            with requests.get(
                media_url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(30, 120),
                stream=True,
            ) as response:
                status = int(response.status_code)
                if status == 200:
                    # 上游对带 Range 的请求回全量 200：无 Range 能力，整单回落。
                    raise _RangeUnsupported()
                if status != 206:
                    reason = f"http_{status}"
                    continue
                with destination.open("wb") as sink:
                    for block in response.iter_content(chunk_size=1 << 20):
                        if block:
                            sink.write(block)
                            received += len(block)
                            if time.monotonic() > deadline:
                                raise ASRError("authorized media decode timed out")
        except _RangeUnsupported:
            raise
        except requests.RequestException:
            reason = "connection"
            continue
        if received != expected:
            reason = "short_read"
            continue
        return
    raise ASRError(_MEDIA_PREFETCH_INCOMPLETE_MESSAGE)


def _prefetch_media_pcm_parallel(
    media_url: str,
    target: Path,
    *,
    duration: float,
) -> bool:
    """Parallel-range prefetch; return False to fall back to the single stream.

    段文件与拼接容器只落在任务自有临时目录（target.parent），成功/失败/回落
    全路径清理；解码与完整性门复用 `_decode_media_to_pcm`，与单流逐位同语义。
    """
    total = _probe_range_total(media_url)
    if total < 2 * _MEDIA_PREFETCH_MIN_SEGMENT_BYTES:
        return False
    segment_size = max(
        _MEDIA_PREFETCH_MIN_SEGMENT_BYTES,
        math.ceil(total / _MEDIA_PREFETCH_SEGMENTS),
    )
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(total - 1, start + segment_size - 1)
        bounds.append((start, end))
        start = end + 1
    deadline = time.monotonic() + max(900, min(7200, int(duration) * 2))
    _emit_telemetry(f"stage=media-prefetch-parallel segments={len(bounds)} bytes={total}")
    parts_dir = target.parent / "prefetch-parts"
    container = target.parent / "prefetch-container.bin"
    try:
        parts_dir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(
            max_workers=len(bounds), thread_name_prefix="prefetch"
        ) as executor:
            futures = [
                executor.submit(
                    _fetch_media_segment,
                    media_url,
                    index,
                    segment_start,
                    segment_end,
                    parts_dir / f"seg-{index:04d}.bin",
                    deadline=deadline,
                )
                for index, (segment_start, segment_end) in enumerate(bounds)
            ]
            for future in futures:
                future.result()
        with container.open("wb") as sink:
            for index in range(len(bounds)):
                with (parts_dir / f"seg-{index:04d}.bin").open("rb") as source:
                    shutil.copyfileobj(source, sink, length=1 << 20)
        _decode_media_to_pcm(str(container), target, duration=duration)
        return True
    except _RangeUnsupported:
        return False
    except ASRError:
        target.unlink(missing_ok=True)
        raise
    except OSError:
        # 本地编排 I/O 意外（目录/拼接/落盘）同样按预取未完成如实失败，
        # 绝不让非闭集异常逃逸（N18 同族纪律）。
        target.unlink(missing_ok=True)
        raise ASRError(_MEDIA_PREFETCH_INCOMPLETE_MESSAGE)
    finally:
        shutil.rmtree(parts_dir, ignore_errors=True)
        try:
            container.unlink(missing_ok=True)
        except OSError:
            pass


def _decode_media_to_pcm(source_ref: str, target: Path, *, duration: float) -> None:
    """One bounded ffmpeg pass from a URL or local container to gated PCM.

    单流与并行分段两条预取路径共用的解码+完整性门；闭集码与 MEDIA-FIX-
    PREFETCH-1 期望字节门在此唯一维护。
    """
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", source_ref,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-y", str(target),
    ]
    try:
        returncode, _, ffmpeg_stderr = _run_bounded_process(
            command,
            timeout=max(900, min(7200, int(duration) * 2)),
            capture_stdout=False,
        )
    except subprocess.TimeoutExpired:
        target.unlink(missing_ok=True)
        raise ASRError("authorized media decode timed out")
    except OSError:
        target.unlink(missing_ok=True)
        raise ASRError("authorized media upstream connection failed")
    if returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        raise _decode_failure(ffmpeg_stderr.decode("utf-8", errors="replace"))
    # 完整性门（MEDIA-FIX-PREFETCH-1）：ffmpeg 对提前 EOF exit 0，期望字节
    # 门是截断的唯一守卫——低于期望量减 1s 容差即按新闭集码如实失败，绝不
    # 放行截断文件去烧后续分块 ASR。
    expected_bytes = int(round(duration * SAMPLE_RATE)) * _PCM_SAMPLE_BYTES
    minimum_bytes = expected_bytes - SAMPLE_RATE * _PCM_SAMPLE_BYTES
    if target.stat().st_size < minimum_bytes:
        target.unlink(missing_ok=True)
        raise ASRError(_MEDIA_PREFETCH_INCOMPLETE_MESSAGE)


def _prefetch_media_pcm(
    media_url: str,
    target: Path,
    *,
    duration: float,
) -> None:
    """夜10-C 第七波①：媒体开局预取——新鲜授权窗口内单趟拉全量 PCM。

    一条 ffmpeg 流式命令从 loopback 授权代理读完整媒体并解码为与分块文件
    同格式的 PCM（f32le/单声道/SAMPLE_RATE）；此后 ASR 全程离线本地切片，
    任务中段零校方请求（跨期 runner 再认证墙根修）。预取失败按既有闭集
    媒体码如实失败；代理内的单次 401/403 刷新（如有）发生在预取早期，
    满足「续登仅限一次且尽量早」的授权边界。

    N12（2026-10-07）：上游认 Range 且体量达并行门槛时，先并行分段拉容器
    字节再本地解码（同一条 ffmpeg 命令、同一完整性门，慢网端墙钟 ~-50%）；
    探测不支持 Range 或体量不足时逐字回落原单流路径，零行为回退。
    """
    if _prefetch_media_pcm_parallel(media_url, target, duration=duration):
        return
    _decode_media_to_pcm(media_url, target, duration=duration)


_PCM_SAMPLE_BYTES = 4  # f32le


def _slice_pcm_chunk(
    full_pcm: Path,
    target: Path,
    *,
    offset: float,
    duration: float,
) -> None:
    """夜10-C 第七波①：从预取全量 PCM 按字节切片一个分块（纯本地 I/O）。

    字节地址 = 秒 × SAMPLE_RATE × 4（f32le）；越界尾部按实际剩余字节截断
    （与解码越界的截断行为一致）。MEDIA-FIX-PREFETCH-1：满块/尾块请求长度
    本身精确，短于请求量减 1s 容差只可能来自预取截断——短块与空切片
    （452282 chunk7 形状）都按真实根因 media_prefetch_incomplete 上行，
    绝不静默空段、不再错标 decode 失败。
    """
    start = int(max(0.0, offset) * SAMPLE_RATE) * _PCM_SAMPLE_BYTES
    length = int(max(0.0, duration) * SAMPLE_RATE) * _PCM_SAMPLE_BYTES
    written = 0
    try:
        with full_pcm.open("rb") as source, target.open("wb") as destination:
            source.seek(start)
            while written < length:
                block = source.read(min(1 << 20, length - written))
                if not block:
                    break
                destination.write(block)
                written += len(block)
    except OSError:
        target.unlink(missing_ok=True)
        raise ASRError("authorized media upstream connection failed")
    if duration > 0 and (
        written == 0 or written < length - SAMPLE_RATE * _PCM_SAMPLE_BYTES
    ):
        target.unlink(missing_ok=True)
        raise ASRError(_MEDIA_PREFETCH_INCOMPLETE_MESSAGE)


# ---- 夜10-C 第九波任务1：邻接重复折叠（口齿不清的 ASR 连续重复字词） ----
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+")
# 合法叠词白名单：单字 run==2 时的常见实词叠形，保留不折叠。
_LEGIT_REDUP = frozenset("""慢慢 刚刚 天天 人人 常常 往往 渐渐 仅仅 统统 恰恰 微微 轻轻 深深 久久 缓缓 悄悄 匆匆 淡淡 默默 徐徐 频频 高高 远远 好好 多多 早早 爸爸 妈妈 哥哥 姐姐 弟弟 妹妹 爷爷 奶奶 叔叔 星星""".split())
_ADJACENT_FOLD_WINDOW_MS = 5000


_ADJACENT_FOLD_PUNCT_RUN = re.compile(r"([，。？！、；：,.!?;:])[，。？！、；：,.!?;:]+")
_ADJACENT_FOLD_LEADING_PUNCT = re.compile(r"^[，。？！、；：,.!?;:]+")


def collapse_repeated_tokens(text: str) -> tuple[str, int]:
    """折叠一段转写文本内的 ASR 连续重复字词（确定性、零增字）。

    R3=单字残叠 X+XY（X 为 XY 首字）→XY；R1=立即同字串 run≥2→1（合法叠词
    run==2 保留）；R2=块级重复——连续两个及以上的等长块（2/3/4/6/8 字，
    词级 ABAB 在字面即 4 字块 ABCDABCD）→保留首块。迭代到不动点。
    """
    matches = list(_TOKEN_RE.finditer(text))
    if len(matches) < 2:
        return text, 0
    values = [m.group(0) for m in matches]
    drop = [False] * len(values)
    folds = 0
    changed = True
    while changed:
        changed = False
        live = [i for i, d in enumerate(drop) if not d]
        vals = [values[i] for i in live]
        # R3：单字残叠
        for k in range(len(vals) - 1):
            a, b = vals[k], vals[k + 1]
            if len(a) == 1 and len(b) >= 2 and b.startswith(a):
                drop[live[k]] = True
                folds += 1
                changed = True
                break
        if changed:
            continue
        # R1：立即同字串
        for k in range(len(vals)):
            run = 1
            while k + run < len(vals) and vals[k + run] == vals[k]:
                run += 1
            if run >= 2:
                tok = vals[k]
                if not (len(tok) == 1 and run == 2 and (tok + tok) in _LEGIT_REDUP):
                    for j in range(k + 1, k + run):
                        drop[live[j]] = True
                    folds += 1
                    changed = True
                    break
        if changed:
            continue
        # R2：块级重复（块长 2/3/4/6/8 字；词级口吃在字面即 4 字块 ABCDABCD）
        for length in (2, 3, 4, 6, 8):
            limit = len(vals) - length
            for k in range(limit + 1):
                block = vals[k:k + length]
                if len(set(block)) == 1:
                    continue  # 单字重复串归 R1 管
                j = k + length
                reps = 1
                while j + length <= len(vals) and vals[j:j + length] == block:
                    reps += 1
                    j += length
                if reps < 2:
                    continue
                # 尾部残叠仅当其为块前缀时收敛（我我也我也→我也）；
                # 非前缀的后续正文（这个这个这样）原样保留。
                # 尾部残叠仅当其为块前缀时收敛（我我也我也→我也）；
                # 非前缀的后续正文（这个这个这样）原样保留。
                partial = vals[j:len(vals)]
                if partial and len(partial) < length and partial == block[:len(partial)]:
                    j += len(partial)
                for t in range(k + length, j):
                    drop[live[t]] = True
                folds += reps - 1
                changed = True
                break
            if changed:
                break
    pieces = []
    last = 0
    for m, d in zip(matches, drop):
        if d:
            pieces.append(text[last:m.start()])
            last = m.end()
    pieces.append(text[last:])
    joined = "".join(pieces)
    # SUBTITLE-DEEP-1 Phase C：折叠删掉叠用词后，其各自尾标点会在接缝处连用
    # （所以，所以，→折叠→，，，）——挤压同族标点串并剥句首标点。无标点输入
    # 逐位不变。
    joined = _ADJACENT_FOLD_PUNCT_RUN.sub(r"\1", joined)
    joined = _ADJACENT_FOLD_LEADING_PUNCT.sub("", joined)
    return joined, folds

def fold_transcript_repetitions(segments: list[dict[str, Any]]) -> dict[str, int]:
    """就地对转写段列表做重复折叠（段内 token 折叠+相邻同文段合并）。

    相邻同文合并窗口=5s 且要求文本完全一致；合并保留首段并延展末时。
    返回遥测计数（仅计数，零内容）。
    """
    folded_tokens = 0
    merged_adjacent = 0
    ordered = sorted(
        segments,
        key=lambda s: (int(s.get("start_ms") or 0), int(s.get("end_ms") or 0)),
    )
    kept: list[dict[str, Any]] = []
    for seg in ordered:
        text, folds = collapse_repeated_tokens(str(seg.get("text") or ""))
        if folds:
            seg["text"] = text
        folded_tokens += folds
        if kept:
            prev = kept[-1]
            gap = int(seg.get("start_ms") or 0) - int(prev.get("end_ms") or 0)
            if (
                str(prev.get("text") or "").strip()
                and prev.get("text") == seg.get("text")
                and 0 <= gap <= _ADJACENT_FOLD_WINDOW_MS
            ):
                prev["end_ms"] = max(
                    int(prev.get("end_ms") or 0), int(seg.get("end_ms") or 0)
                )
                merged_adjacent += 1
                continue
        kept.append(seg)
    # 就地收敛：保留段写回调用方列表
    segments[:] = kept
    return {
        "folded_tokens": folded_tokens,
        "merged_adjacent": merged_adjacent,
        "segments": len(kept),
    }


# ---- SUBTITLE-DEEP-1 Phase C：双时间戳锚校正 --------------------------------
# 平台官方文稿（payload.platform_transcript，即 AS12 的交替源行）的 cue 时刻来自
# 校方播放器，可作 Paraformer 识别锚漂移的校正基准（BENCH timing 实测中位偏差
# 12-18s）。校正确定性：官方 cue 与识别段窗口重叠最优者配对，段中点向官方中点
# 平移（时长保持），位移有帽；未命中段在相邻已校正段之间线性插值。
TIME_ANCHOR_ENV = "COURSELENS_SUBTITLE_TIME_ANCHOR"
TIME_ANCHOR_MATCH_GAP_MS = 3000
TIME_ANCHOR_MATCH_MIN_OVERLAP_MS = 800
TIME_ANCHOR_MAX_SHIFT_MS = 8000
TIME_ANCHOR_SNAP_MS = 800


def _anchor_correct_timing(
    segments: list[dict[str, Any]],
    official_rows: list[dict[str, Any]],
) -> dict[str, int]:
    """Anchor-correct refined segment timing toward official cue midpoints, in place.

    配对：官方行与识别段在 ±TIME_ANCHOR_MATCH_GAP_MS 窗口内重叠最大者；重叠
    <TIME_ANCHOR_MATCH_MIN_OVERLAP_MS 不配对。位移=官方中点−识别中点，帽
    ±TIME_ANCHOR_MAX_SHIFT_MS，|位移|≤TIME_ANCHOR_SNAP_MS 视为零漂移不动。
    未配对段取左右已配对邻位的位移线性插值（按中点距离加权），同样受帽。
    返回遥测计数（仅计数，零内容）。
    """
    rows: list[tuple[int, int]] = []
    for row in official_rows or []:
        if not isinstance(row, dict):
            continue
        try:
            start = max(0, int(row.get("start_ms") or 0))
            end = max(0, int(row.get("end_ms") or 0))
        except (TypeError, ValueError):
            continue
        if end > start:
            rows.append((start, end))
    rows.sort()
    if not rows or not segments:
        return {"official_rows": len(rows), "matched": 0, "shifted": 0, "max_shift_ms": 0}

    ordered = sorted(
        range(len(segments)),
        key=lambda index: (int(segments[index].get("start_ms") or 0), index),
    )
    direct: dict[int, int] = {}
    for index in ordered:
        segment = segments[index]
        start = int(segment.get("start_ms") or 0)
        end = int(segment.get("end_ms") or start)
        best_overlap = 0
        best_mid = 0
        for row_start, row_end in rows:
            if row_end < start - TIME_ANCHOR_MATCH_GAP_MS:
                continue
            if row_start > end + TIME_ANCHOR_MATCH_GAP_MS:
                break
            overlap = min(end, row_end) - max(start, row_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_mid = (row_start + row_end) // 2
        if best_overlap < TIME_ANCHOR_MATCH_MIN_OVERLAP_MS:
            continue
        mid = (start + end) // 2
        delta = best_mid - mid
        if abs(delta) > TIME_ANCHOR_MAX_SHIFT_MS:
            delta = TIME_ANCHOR_MAX_SHIFT_MS if delta > 0 else -TIME_ANCHOR_MAX_SHIFT_MS
        direct[index] = delta

    matched = len(direct)
    max_shift = max((abs(value) for value in direct.values()), default=0)
    # 位移应用：直配段用自身位移（|位移|≤snap 视为零漂移不动）；仅无官方
    # 重叠的段取左右直配邻位插值（已对齐的直配段绝不被邻居位移）。
    shifted = 0
    for position, index in enumerate(ordered):
        if index in direct:
            delta = direct[index]
        else:
            left = next(
                (direct[prior] for prior in reversed(ordered[:position]) if prior in direct),
                None,
            )
            right = next(
                (direct[later] for later in ordered[position + 1:] if later in direct),
                None,
            )
            if left is None and right is None:
                continue
            if left is None:
                delta = right
            elif right is None:
                delta = left
            else:
                delta = (left + right) // 2
            if abs(delta) > TIME_ANCHOR_MAX_SHIFT_MS:
                delta = TIME_ANCHOR_MAX_SHIFT_MS if delta > 0 else -TIME_ANCHOR_MAX_SHIFT_MS
        if abs(delta) <= TIME_ANCHOR_SNAP_MS:
            continue
        segment = segments[index]
        segment["start_ms"] = max(0, int(segment.get("start_ms") or 0) + delta)
        segment["end_ms"] = max(
            int(segment["start_ms"]), int(segment.get("end_ms") or 0) + delta
        )
        shifted += 1
    return {
        "official_rows": len(rows), "matched": matched,
        "shifted": shifted, "max_shift_ms": max_shift,
    }


def _decode_chunk_from_url(
    media_url: str,
    target: Path,
    *,
    offset: float,
    duration: float,
) -> None:
    try:
        returncode, _, ffmpeg_stderr = _run_bounded_process(
            _ffmpeg_proxy_command(
                target, media_url, offset=offset, duration=duration,
            ),
            timeout=900,
            capture_stdout=False,
        )
    except subprocess.TimeoutExpired:
        target.unlink(missing_ok=True)
        raise ASRError("authorized media decode timed out")
    except OSError:
        target.unlink(missing_ok=True)
        raise ASRError("authorized media upstream connection failed")
    if returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        raise _decode_failure(ffmpeg_stderr.decode("utf-8", errors="replace"))


def _pcm_file_digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.digest()


def _advance_pcm_fingerprint(state: str | None, chunk_digest: bytes) -> str:
    """Chain one chunk digest into the serializable run fingerprint state.

    The chain state is a plain hex string, so a checkpoint taken after any
    chunk lets a resumed run reproduce the exact final fingerprint — and
    therefore the same segment IDs — as an uninterrupted run.
    """
    chained = hashlib.sha256()
    chained.update(bytes.fromhex(state) if state else PCM_FINGERPRINT_DOMAIN)
    chained.update(chunk_digest)
    return chained.hexdigest()


def _timing_config_hash(energy_ratio: float) -> str:
    config = {
        "algorithm": "frame-energy-v1",
        "energy_ratio": energy_ratio,
        "frame_seconds": VAD_FRAME_SECONDS,
        "hop_seconds": VAD_HOP_SECONDS,
        "noise_percentile": VAD_NOISE_PERCENTILE,
        "silence_floor_rms": VAD_SILENCE_FLOOR_RMS,
        "peak_guard_ratio": VAD_PEAK_GUARD_RATIO,
        "merge_gap_seconds": VAD_MERGE_GAP_SECONDS,
        "pad_seconds": VAD_PAD_SECONDS,
        "min_region_seconds": VAD_MIN_REGION_SECONDS,
        "max_region_seconds": VAD_MAX_REGION_SECONDS,
        "window_seconds": ASR_WINDOW_SECONDS,
    }
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:16]


def _source_evidence_id(fingerprint: str, duration: float) -> str:
    return compute_id(NAMESPACE_SOURCE, {
        "kind": "recording",
        "origin": "external_import",
        "title": None,
        "duration_ms": int(duration * 1000),
        "source_sha256": fingerprint,
    })


def platform_transcript_coverage(
    segments: list[dict[str, Any]],
    duration_ms: int,
    *,
    merge_gap_ms: int = PLATFORM_TRANSCRIPT_MERGE_GAP_MS,
) -> float:
    """Content coverage of transcript intervals clipped to [0, duration_ms].

    Intervals separated by less than ``merge_gap_ms`` are merged first, so
    natural inter-sentence pauses in platform cues do not read as missing
    content; only gaps at least that wide count as holes (U6).
    """
    if duration_ms <= 0 or not segments:
        return 0.0
    intervals: list[tuple[int, int]] = []
    for item in segments:
        # 夜10-C 边界加固：非 dict/时间戳不可解析的畸形行跳过（覆盖度只在
        # 合法行上计算），链裁决永不因异常文稿形态崩溃——低覆盖走闭集回落。
        if not isinstance(item, dict) or not str(item.get("text") or "").strip():
            continue
        try:
            start = max(0, int(item.get("start_ms") or 0))
            end = max(0, int(item.get("end_ms") or 0))
        except (TypeError, ValueError):
            continue
        intervals.append((start, end))
    intervals.sort()
    merged: list[list[int]] = []
    for start, end in intervals:
        if end <= start:
            continue
        if merged and start - merged[-1][1] < merge_gap_ms:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    covered = sum(max(0, min(end, duration_ms) - start) for start, end in merged)
    return covered / duration_ms


def _stamp_segment_identity(
    segments: list[dict[str, Any]],
    *,
    source_id: str,
    source_hash: str,
    provenance: dict[str, Any],
) -> None:
    """Stamp contract-shaped segment IDs plus bounded provenance in place."""
    for segment in segments:
        segment["source_hash"] = source_hash
        segment["segment_id"] = compute_id(NAMESPACE_SEGMENT, {
            "source_id": source_id,
            "start_ms": int(segment["start_ms"]),
            "end_ms": int(segment["end_ms"]),
            "text": str(segment["text"]),
            "lang": segment.get("lang"),
            "no_speech": False,
            "producer": provenance.get("producer"),
            "model": provenance.get("model"),
            "config_hash": provenance.get("config_hash"),
        })
        segment["provenance"] = dict(provenance)


TELEMETRY_TICK_SECONDS = 30.0


def _emit_telemetry(line: str) -> None:
    # runner._progress discipline: counters, seconds, and fixed stage
    # identifiers only — never URLs, paths, titles, or provider text.
    print(line, flush=True)


def _mem_available_kb_from(text: str) -> int:
    """MemAvailable KiB from /proc/meminfo text; -1 when absent or unshaped."""
    for line in str(text).splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    return int(fields[1])
                except ValueError:
                    return -1
            return -1
    return -1


def _mem_available_kb(path: Path = Path("/proc/meminfo")) -> int:
    try:
        return _mem_available_kb_from(path.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return -1


class _ChunkTicker:
    """Death-window telemetry for the subtitle chunk loop.

    A daemon thread printing one bounded line per interval — elapsed seconds,
    current chunk index, transient PCM file size, and MemAvailable — plus a
    termination line on stop.  It only reads the loop's small state dict and
    the filesystem; it never touches decode, ASR, fingerprint, or checkpoint
    state, so it cannot change run semantics.
    """

    def __init__(
        self,
        state: dict[str, Any],
        *,
        interval: float = TELEMETRY_TICK_SECONDS,
        emit: Callable[[str], None] = _emit_telemetry,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._state = state
        self._interval = max(0.05, float(interval))
        self._emit = emit
        self._clock = clock
        self._stop = threading.Event()
        self._started = False
        self.started_at = self._clock()
        self._thread = threading.Thread(
            target=self._loop, name="asr-chunk-ticker", daemon=True,
        )

    def start(self) -> None:
        self._started = True
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._started:
            self._thread.join(timeout=timeout)
        self._emit(
            f"stage=asr-tick-end elapsed={self._elapsed_seconds()} "
            f"chunks={max(0, int(self._state.get('done') or 0))}"
        )

    def _elapsed_seconds(self) -> int:
        return max(0, int(round(self._clock() - self.started_at)))

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            pcm = self._state.get("pcm")
            size = 0
            if pcm is not None:
                try:
                    size = int(pcm.stat().st_size)
                except OSError:
                    size = 0
            self._emit(
                f"stage=asr-tick elapsed={self._elapsed_seconds()} "
                f"chunk={max(0, int(self._state.get('chunk') or 0))} "
                f"pcm_bytes={size} mem_avail_kb={_mem_available_kb()}"
            )


def transcribe(
    job: dict[str, Any],
    *,
    sensevoice_dir: Path,
    paraformer_dir: Path | None = None,
    zipformer_dir: Path | None = None,
    hotwords: tuple[str, ...] = (),
    proofread: Callable[..., list[dict[str, Any]]] | None,
    progress: Callable[[str, int, int], None],
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    payload = dict(job.get("payload") or {})
    source = dict(payload.get("media") or {})
    mode = str(payload.get("mode") or "automatic")
    if mode != "automatic":
        raise ASRError("unsupported subtitle mode")
    # 自动策略：配置了校对提供方（DeepSeek Key）时走双模型+AI 校对链，
    # 否则走非 AI 回退（仅精识别模型）。分支键是行为，不是历史模式标签。
    proofread_enabled = proofread is not None
    start_seconds = float(source.get("start_seconds") or 0)
    duration = float(source.get("duration_seconds") or 0)
    if duration <= 0:
        duration = _probe_duration(source) - start_seconds
    if start_seconds < 0:
        raise ASRError("media start is invalid")
    if duration <= 0 or duration > 12 * 60 * 60:
        raise ASRError("media duration is missing or outside the supported range")
    # AS12 rough 源裁决：默认 platform-first，命中即整讲跳过粗腿；任何未命中
    # 都整讲回落现有双模链（失败=降级，绝不失败任务），原因走闭集记账。
    rough_source_request = str(
        os.environ.get(ASR_ROUGH_SOURCE_ENV) or ASR_ROUGH_SOURCE_PLATFORM
    ).strip().lower()
    if rough_source_request not in {ASR_ROUGH_SOURCE_PLATFORM, ASR_ROUGH_SOURCE_SENSEVOICE}:
        rough_source_request = ASR_ROUGH_SOURCE_PLATFORM
    platform_rows = payload.get("platform_transcript")
    platform_rows = platform_rows if isinstance(platform_rows, list) else []
    platform_state = str(payload.get("platform_transcript_state") or "")
    rough_fallback_reason = ""
    use_platform_alternates = False
    platform_coverage = 0.0
    if not proofread_enabled:
        rough_source = "not_applicable"
    else:
        if rough_source_request == ASR_ROUGH_SOURCE_SENSEVOICE:
            rough_fallback_reason = "env_disabled"
        elif not platform_rows:
            rough_fallback_reason = (
                platform_state
                if platform_state in {"transcript_fetch_failed", "transcript_empty"}
                else "platform_transcript_missing"
            )
        else:
            platform_coverage = platform_transcript_coverage(
                platform_rows, int(duration * 1000)
            )
            if platform_coverage >= PLATFORM_TRANSCRIPT_MIN_COVERAGE:
                use_platform_alternates = True
            else:
                rough_fallback_reason = "coverage_low"
        rough_source = (
            ASR_ROUGH_SOURCE_PLATFORM if use_platform_alternates
            else ASR_ROUGH_SOURCE_SENSEVOICE
        )
        _emit_telemetry(
            f"stage=rough-source source={rough_source} "
            f"reason={rough_fallback_reason or 'none'} "
            f"segments={len(platform_rows) if use_platform_alternates else 0} "
            f"coverage={round(platform_coverage, 3)}"
        )
    strategy = str(os.environ.get("COURSELENS_ASR_STRATEGY") or "sequential").strip().lower()
    if strategy not in {"sequential", "parallel"}:
        strategy = "sequential"
    # 平台链下每块只剩精识别单模型，独占全部线程；parallel 策略只属回落链。
    recognizer_threads = (
        2 if proofread_enabled and strategy == "parallel" and not use_platform_alternates
        else 4
    )
    backends = subtitle_backend_sequence()
    rough, refined = backends
    # SUBTITLE-DEEP-1 Phase B：zipformer 模型缺席时整讲回退原链（失败=降级，
    # 绝不失败任务）。回退只发生在任何块解码与检查点写入之前，检查点链身份
    # （backends/raw_* 键）始终与实际生效序列一致；已装但损坏的模型文件在
    # 预暖处按既有闭集码如实失败，不做静默混链。
    zipformer_requested = zipformer_dir if "zipformer" in (rough, refined) else None
    if zipformer_requested is not None:
        probe = RecognizerPool(
            sensevoice_dir, paraformer_dir, threads=recognizer_threads,
            zipformer_dir=zipformer_requested,
        )
        if not probe.zipformer_ready():
            replacement = "paraformer" if "paraformer" not in (rough, refined) else "sensevoice"
            _emit_telemetry(
                f"stage=zipformer-fallback "
                f"backend={'refined' if refined == 'zipformer' else 'rough'} "
                f"replacement={replacement}"
            )
            if refined == "zipformer":
                refined = replacement
            if rough == "zipformer":
                rough = "sensevoice" if refined != "sensevoice" else "paraformer"
            zipformer_requested = None
            backends = [rough, refined]
    pool = RecognizerPool(
        sensevoice_dir, paraformer_dir, threads=recognizer_threads,
        zipformer_dir=zipformer_requested,
    )
    # V4NONTHINK-1 件7：silero-vad 模型注入（env 请求时解析；缺席回落能量 VAD）。
    if vad_engine() == VAD_ENGINE_SILERO:
        pool.silero_model_path = silero_model_path()
        _emit_telemetry(
            f"stage=vad-engine engine=silero model="
            f"{'ready' if pool.silero_model_path else 'missing'}"
        )
    prior = dict(payload.get("checkpoint") or {})
    if prior and str(prior.get("mode") or "") != mode:
        raise ASRError("checkpoint subtitle mode does not match the job")
    total_chunks = max(1, int((duration + PCM_CHUNK_SECONDS - 1) // PCM_CHUNK_SECONDS))
    completed_chunks = max(0, min(total_chunks, int(prior.get("completed_chunks") or 0)))
    # 续跑的检查点必须已携带同序列车型的 raw 段，否则精识别会从
    # completed_chunks 起步而丢失前段输出——缺键宁可显式失败。
    if completed_chunks > 0 and any(f"raw_{name}" not in prior for name in backends):
        raise ASRError("checkpoint raw segments do not match the subtitle backends")
    # AS12 续跑守卫：rough 源与列车序同属链身份。旧版检查点没有这些键——
    # 剩余块沿旧链跑完（不混合交替源来源），显式记账 legacy_checkpoint；
    # 带键但与当前链不匹配则显式失败，宁可重跑也不静默混链。
    legacy_checkpoint = completed_chunks > 0 and "rough_source" not in prior
    if legacy_checkpoint and use_platform_alternates:
        use_platform_alternates = False
        rough_source = ASR_ROUGH_SOURCE_SENSEVOICE
        rough_fallback_reason = "legacy_checkpoint"
    if completed_chunks > 0 and not legacy_checkpoint:
        if str(prior.get("rough_source") or "") != rough_source:
            raise ASRError("checkpoint rough source does not match the subtitle chain")
        prior_backends = prior.get("backends")
        if prior_backends is not None and list(prior_backends) != list(backends):
            raise ASRError("checkpoint raw segments do not match the subtitle backends")
    rough_segments: list[dict[str, Any]] = (
        normalize_segments(list(platform_rows))
        if use_platform_alternates
        else list(prior.get(f"raw_{rough}") or [])
    )
    refined_segments: list[dict[str, Any]] = list(prior.get(f"raw_{refined}") or [])
    if proofread_enabled and strategy == "parallel" and not use_platform_alternates:
        pool.get(rough)
        pool.get(refined)
    # Provenance is stamped only when the fingerprint chain covers every chunk
    # of the run.  A legacy checkpoint without chain state makes that
    # impossible, so the output omits provenance entirely and the client
    # compatibility seam mints its honest fallback identity instead.
    fingerprint_state: str | None = None
    verifiable_fingerprint = True
    if completed_chunks > 0:
        prior_state = prior.get("pcm_fingerprint")
        if isinstance(prior_state, str) and _PCM_FINGERPRINT_RE.match(prior_state):
            fingerprint_state = prior_state
        else:
            verifiable_fingerprint = False
    started = time.monotonic()
    # Keep one authorized CDN playback session for the complete task.  The
    # runner still launches one bounded FFmpeg process and retains only one
    # transient PCM file per chunk. Rotate the proxy's hidden signed URL before
    # each later chunk so an expired URL is never the first request of a new
    # decode session; a real 401/403 inside a chunk still gets only one bounded
    # refresh retry in the proxy.
    with tempfile.TemporaryDirectory(prefix="courselens-pcm-") as temporary, pinned_media_proxy(source) as proxy:
        root = Path(temporary)
        # Death-window telemetry: fixed phase lines plus a 30s ticker line so
        # one real run pinpoints a runner death to the phase and the second.
        # Observation only — decode, fingerprint, and checkpoint math are
        # untouched, and the lines carry counters and seconds exclusively.
        t0 = time.monotonic()

        def _elapsed_ticks() -> int:
            return max(0, int(round(time.monotonic() - t0)))

        telemetry_state: dict[str, Any] = {
            "chunk": completed_chunks, "pcm": None, "done": completed_chunks,
        }
        ticker = _ChunkTicker(telemetry_state)
        try:
            _emit_telemetry(f"stage=proxy-resolved elapsed={_elapsed_ticks()}")
            # 夜10-C 第七波①：媒体开局预取——新鲜授权窗口内单趟拉全量 PCM，
            # 此后分块全部本地切片：任务中段零校方请求，块界授权刷新梯与
            # 媒体重试梯退出主循环（闭集码与有界梯语义在预取路径保留）。
            full_pcm = root / "media-full.f32le"
            prefetch_started = time.monotonic()
            _emit_telemetry(f"stage=media-prefetch-start elapsed={_elapsed_ticks()}")
            # MEDIA-FIX-PREFETCH-1：完整性门判为截断时有界重取一次——退避后
            # 先 refresh_source 换新签名 URL（同一性守卫沿用代理面），再走同
            # 一条预取命令；梯尽按 media_prefetch_incomplete 如实失败。
            try:
                _prefetch_media_pcm(proxy.url, full_pcm, duration=duration)
            except ASRError as exc:
                if str(exc) != _MEDIA_PREFETCH_INCOMPLETE_MESSAGE:
                    raise
                _emit_telemetry(
                    "stage=media-prefetch-retry attempt=1 "
                    f"reason=media_prefetch_incomplete elapsed={_elapsed_ticks()}"
                )
                time.sleep(_MEDIA_RETRY_BACKOFF_SECONDS[0])
                _refresh_media_authorization(
                    proxy,
                    chunk=completed_chunks,
                    elapsed=_elapsed_ticks(),
                    label="face=prefetch",
                )
                _prefetch_media_pcm(proxy.url, full_pcm, duration=duration)
            _emit_telemetry(
                f"stage=media-prefetch-done bytes={full_pcm.stat().st_size} "
                f"expected={int(round(duration * SAMPLE_RATE)) * _PCM_SAMPLE_BYTES} "
                f"seconds={round(time.monotonic() - prefetch_started, 3)} "
                f"elapsed={_elapsed_ticks()}"
            )
            if pool.zipformer_dir is not None:
                # 热词文件落盘在任务自有临时目录内（零新持久化路径），懒加载
                # 语义：只被 zipformer 腿首次构建时读取。预暖在任何块解码前
                # 完成——加载失败按闭集码如实失败，绝不半链混跑。
                if hotwords:
                    pool.hotwords_file = root / "hotwords.txt"
                    pool.hotwords_file.write_text(
                        "\n".join(str(term).strip() for term in hotwords[:ASR_HOTWORD_LIMIT] if str(term).strip()) + "\n",
                        encoding="utf-8",
                    )
                pool.get("zipformer")
                _emit_telemetry(
                    f"stage=zipformer-ready hotwords={min(len(hotwords), ASR_HOTWORD_LIMIT) if hotwords else 0}"
                )
            ticker.start()
            for index in range(completed_chunks, total_chunks):
                telemetry_state["chunk"] = index
                relative_offset = index * PCM_CHUNK_SECONDS
                absolute_offset = start_seconds + relative_offset
                chunk_duration = min(PCM_CHUNK_SECONDS, duration - relative_offset)
                pcm = root / f"chunk-{index:04d}.f32le"
                telemetry_state["pcm"] = pcm
                _emit_telemetry(f"stage=decode-start chunk={index} elapsed={_elapsed_ticks()}")
                decode_started = time.monotonic()
                _slice_pcm_chunk(
                    full_pcm,
                    pcm,
                    offset=absolute_offset,
                    duration=chunk_duration,
                )
                _emit_telemetry(
                    f"stage=decode-done chunk={index} bytes={pcm.stat().st_size} "
                    f"seconds={round(time.monotonic() - decode_started, 3)} "
                    f"elapsed={_elapsed_ticks()}"
                )
                if verifiable_fingerprint:
                    fingerprint_state = _advance_pcm_fingerprint(
                        fingerprint_state, _pcm_file_digest(pcm)
                    )
                _emit_telemetry(f"stage=asr-start chunk={index} elapsed={_elapsed_ticks()}")
                if use_platform_alternates:
                    # AS12：粗腿已由讲级平台文稿顶替，本块仍照常跑精识别。
                    refined_segments.extend(
                        pool.transcribe_pcm(pcm, refined, offset_seconds=absolute_offset)
                    )
                elif proofread_enabled and strategy == "parallel":
                    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr") as executor:
                        rough_future = executor.submit(
                            pool.transcribe_pcm, pcm, rough, offset_seconds=absolute_offset
                        )
                        refined_future = executor.submit(
                            pool.transcribe_pcm, pcm, refined, offset_seconds=absolute_offset
                        )
                        rough_segments.extend(rough_future.result())
                        refined_segments.extend(refined_future.result())
                else:
                    if proofread_enabled:
                        rough_segments.extend(
                            pool.transcribe_pcm(pcm, rough, offset_seconds=absolute_offset)
                        )
                    refined_segments.extend(
                        pool.transcribe_pcm(pcm, refined, offset_seconds=absolute_offset)
                    )
                _emit_telemetry(f"stage=asr-end chunk={index} elapsed={_elapsed_ticks()}")
                pcm.unlink(missing_ok=True)
                telemetry_state["done"] = index + 1
                progress("asr", index + 1, total_chunks)
                if checkpoint is not None:
                    state: dict[str, Any] = {
                        "completed_chunks": index + 1,
                        "total_chunks": total_chunks,
                        "mode": mode,
                        "backends": list(backends),
                        "rough_source": rough_source,
                        f"raw_{rough}": normalize_segments(rough_segments),
                        f"raw_{refined}": normalize_segments(refined_segments),
                    }
                    if fingerprint_state is not None:
                        state["pcm_fingerprint"] = fingerprint_state
                    checkpoint(state)
        finally:
            ticker.stop()
    # 夜10-C 第九波任务1：邻接重复折叠——口齿不清的 ASR 连续重复字词在
    # 校对/落库前折叠（校对与摘要拿到干净文本；遥测仅计数）。
    refined_segments_fold = fold_transcript_repetitions(refined_segments)
    fold_transcript_repetitions(rough_segments)
    _emit_telemetry(
        f"stage=token-fold tokens={refined_segments_fold['folded_tokens']} "
        f"adjacent={refined_segments_fold['merged_adjacent']} "
        f"segments={refined_segments_fold['segments']}"
    )
    # SUBTITLE-DEEP-1 Phase C：双时间戳锚校正——平台官方 cue 时刻为锚校识别
    # 段漂移（确定性配对+有帽平移+插值平滑）。平台行缺席/杀开关时整段跳过；
    # 遥测仅计数。校正在检查点循环之后确定性执行，续跑同果。
    timing_anchor_stats: dict[str, int] = {}
    if platform_rows and os.environ.get(TIME_ANCHOR_ENV, "").strip() != "0":
        timing_anchor_stats = _anchor_correct_timing(refined_segments, platform_rows)
        _emit_telemetry(
            f"stage=time-anchor official_rows={timing_anchor_stats.get('official_rows', 0)} "
            f"matched={timing_anchor_stats.get('matched', 0)} "
            f"shifted={timing_anchor_stats.get('shifted', 0)} "
            f"max_shift_ms={timing_anchor_stats.get('max_shift_ms', 0)}"
        )
    proofread_degraded = False
    if not proofread_enabled:
        final = refined_segments
    else:
        def proofread_checkpoint(value: dict[str, Any]) -> None:
            if checkpoint is not None:
                state: dict[str, Any] = {
                    "stage": "proofread",
                    "completed_chunks": total_chunks,
                    "total_chunks": total_chunks,
                    "mode": mode,
                    "backends": list(backends),
                    "rough_source": rough_source,
                    f"raw_{rough}": normalize_segments(rough_segments),
                    f"raw_{refined}": normalize_segments(refined_segments),
                    **value,
                }
                if fingerprint_state is not None:
                    state["pcm_fingerprint"] = fingerprint_state
                checkpoint(state)

        try:
            final = proofread(
                rough_segments,
                refined_segments,
                prior,
                proofread_checkpoint,
            )
        except LLMError:
            # ASRBENCH P1（G7）：AI 校对失败不再把已完成的识别整单带崩——
            # 降级交付未经校订的原始结果，闭集警告随产物上屏（诚实标注）。
            proofread_degraded = True
            final = refined_segments
    final_segments = normalize_segments(final)
    raw_rough = normalize_segments(rough_segments)
    raw_refined = normalize_segments(refined_segments)
    if verifiable_fingerprint and fingerprint_state is not None:
        config_hash = _timing_config_hash(asr_energy_ratio())
        source_id = _source_evidence_id(fingerprint_state, duration)
        _stamp_segment_identity(
            raw_rough,
            source_id=source_id,
            source_hash=fingerprint_state,
            provenance={
                "producer": PRODUCER_ID,
                # AS12：平台链下 raw_{rough} 槽位承载平台文稿交替候选，
                # 诚实标注来源，绝不冒认 sensevoice 识别输出。
                "model": "platform:transcript" if use_platform_alternates else rough,
                "config_hash": config_hash,
            },
        )
        _stamp_segment_identity(
            raw_refined,
            source_id=source_id,
            source_hash=fingerprint_state,
            provenance={
                "producer": PRODUCER_ID,
                "model": refined,
                "config_hash": config_hash,
            },
        )
        final_model = (
            refined if (not proofread_enabled or proofread_degraded)
            else (
                f"{refined}+platform:proofread"
                if use_platform_alternates
                else f"{rough}+{refined}:proofread"
            )
        )
        _stamp_segment_identity(
            final_segments,
            source_id=source_id,
            source_hash=fingerprint_state,
            provenance={
                "producer": PRODUCER_ID,
                "model": final_model,
                "config_hash": config_hash,
            },
        )
    return {
        "mode": mode,
        "segments": final_segments,
        f"raw_{rough}": raw_rough,
        f"raw_{refined}": raw_refined,
        # V4NONTHINK-1 件4：分歧跨度裁决的交替源（与词级校对同一 rough 槽位）。
        # 只进 worker 内部术语层（suspects wire），不随 outputs 上行（键名不以
        # raw_ 开头，runner 转发面零变化）。
        "proofread_alternates": raw_rough,
        **({"warnings": ["proofread_degraded"]} if proofread_degraded else {}),
        "metrics": {
            "duration_seconds": duration,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "chunks": total_chunks,
            "threads_per_model": recognizer_threads,
            "strategy": strategy if proofread_enabled else "single-model",
            "start_seconds": start_seconds,
            "rough_source": rough_source,
            **(
                {"rough_source_fallback_reason": rough_fallback_reason}
                if rough_fallback_reason else {}
            ),
            **({"timing_anchor": timing_anchor_stats} if timing_anchor_stats else {}),
        },
    }
