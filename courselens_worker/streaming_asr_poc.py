"""Streaming ASR proof-of-concept（D13-IMPL · P2 路线可行性验证）.

**本模块是 POC，不接生产管线。** 生产入口（runner/asr/cloud_automation）
不导入本模块；唯一消费方 = ``worker/tests/test_streaming_asr_poc.py`` 与
手工基准（``python -m courselens_worker.streaming_asr_poc``）。

验证的问题（D13-RESEARCH §4-P2 提出待验证）：

1. 既有依赖（sherpa-onnx==1.13.4，worker/requirements.txt 钉）内的
   ``OnlineRecognizer`` + 流式 Paraformer 双语模型（zh-en）能否实现
   「音频分块喂入 → 增量中间结果 → 终态结果」的 API 形态；
2. 模型加载 / 增量延迟 / 实时率（RTF）/ 内存 / CPU 的量级；
3. 「上课中边听边转写、下课 transcript 已就绪」的可行性证据。

模型获取（POC 手工步骤，未进 install_models 分发清单；缓存放车道暂存
``.tmp-d13impl/``——worker 树内禁放任何媒体文件，公共边界门会拦 .wav）::

    base=https://hf-mirror.com/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en/resolve/main
    dir=.tmp-d13impl/models/streaming-paraformer-bilingual-zh-en
    curl -L -o $dir/encoder.int8.onnx  $base/encoder.int8.onnx   # 165,462,184 B
    curl -L -o $dir/decoder.int8.onnx  $base/decoder.int8.onnx   #  71,664,561 B
    curl -L -o $dir/tokens.txt         $base/tokens.txt          #     75,756 B
    mkdir -p $dir/test_wavs
    curl -L -o $dir/test_wavs/0.wav    $base/test_wavs/0.wav     #    321,744 B

生产落地时应把同一模型条目按 install_models.py 的 sha256 钉纪律接入
（GitHub releases 源 + 校验和 + .ready 标记），见结果文件施工清单。

设计约束：

- **fail-closed**：运行时缺位（sherpa-onnx 被 conftest 桩住 / 模型文件缺席）
  时显式报错或跳过，绝不产出伪造识别结果。
- **纯 stdlib 依赖面**：合成音频与指标收集只用 stdlib（随机数/时间/ctypes），
  模块在 client venv（无 numpy/sherpa-onnx）内可导入、纯逻辑测试可跑。
- **识别器可注入**：``StreamingAsrPoc(recognizer=...)`` 接受测试替身，
  流式循环逻辑（喂入→增量→端点→终态）不依赖真实模型即可钉死。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import random
import struct
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SAMPLE_RATE = 16_000
DEFAULT_CHUNK_SECONDS = 1.0
DEFAULT_NUM_THREADS = 2
# sherpa-onnx 端点检测三规则（官方默认值；与流式 Paraformer 配套）。
RULE1_MIN_TRAILING_SILENCE = 2.4
RULE2_MIN_TRAILING_SILENCE = 1.2
RULE3_MIN_UTTERANCE_LENGTH = 20.0

POC_MODEL_DIR_ENV = "COURSELENS_STREAMING_POC_MODEL_DIR"
DEFAULT_MODEL_DIRNAME = "streaming-paraformer-bilingual-zh-en"
ENCODER_FILENAME = "encoder.int8.onnx"
DECODER_FILENAME = "decoder.int8.onnx"
TOKENS_FILENAME = "tokens.txt"

# 合成「讲课」节奏：发声段/静音段交错，静音长于 rule2 以触发端点。
_BURST_SECONDS = 4.0
_GAP_SECONDS = 1.4


class PocError(RuntimeError):
    """POC 任何失败都 fail-closed，不回落到伪结果。"""


def default_model_dir() -> Path:
    """POC 模型目录解析：env 覆盖 > 车道暂存 ``.tmp-d13impl/models/``。

    模型缓存不进 worker 树（公共边界门对媒体文件零容忍），也不进
    install_models 默认根——POC 阶段手工管理，生产化时按 install_models
    的 sha256 钉纪律接入分发清单。
    """
    import os

    raw = os.environ.get(POC_MODEL_DIR_ENV, "").strip()
    if raw:
        return Path(raw)
    return (
        Path(__file__).resolve().parents[2]
        / ".tmp-d13impl"
        / "models"
        / DEFAULT_MODEL_DIRNAME
    )


def resolve_model_paths(model_dir: Path | None = None) -> tuple[Path, Path, Path]:
    directory = Path(model_dir) if model_dir else default_model_dir()
    encoder = directory / ENCODER_FILENAME
    decoder = directory / DECODER_FILENAME
    tokens = directory / TOKENS_FILENAME
    missing = [str(path) for path in (encoder, decoder, tokens) if not path.is_file()]
    if missing:
        raise PocError(
            "streaming paraformer model files missing (POC fetch steps are in the "
            "module docstring): " + ", ".join(missing)
        )
    return encoder, decoder, tokens


def sherpa_runtime_available() -> bool:
    """真实 sherpa-onnx 运行时探测（conftest 的 Mock 桩不算运行时）。"""
    try:
        import sherpa_onnx  # noqa: PLC0415 - 惰性导入，保持零依赖导入面
    except Exception:  # noqa: BLE001 - 探测口：任何导入失败=不可用
        return False
    return type(sherpa_onnx).__module__ != "unittest.mock"


def _build_recognizer(config: "PocConfig") -> Any:
    """构造真实 OnlineRecognizer（只在真实运行时路径被调用）。"""
    if not sherpa_runtime_available():
        raise PocError(
            "sherpa-onnx runtime unavailable (stubbed or missing); real-model POC "
            "requires worker/.venv-worker with sherpa-onnx==1.13.4"
        )
    import sherpa_onnx  # noqa: PLC0415

    return sherpa_onnx.OnlineRecognizer.from_paraformer(
        encoder=str(config.encoder),
        decoder=str(config.decoder),
        tokens=str(config.tokens),
        num_threads=config.num_threads,
        sample_rate=config.sample_rate,
        feature_dim=config.feature_dim,
        enable_endpoint_detection=config.enable_endpoint,
        rule1_min_trailing_silence=config.rule1_min_trailing_silence,
        rule2_min_trailing_silence=config.rule2_min_trailing_silence,
        rule3_min_utterance_length=config.rule3_min_utterance_length,
        decoding_method=config.decoding_method,
    )


@dataclass
class PocConfig:
    """流式识别配置（默认值即生产候选形态的量级）。"""

    encoder: Path
    decoder: Path
    tokens: Path
    num_threads: int = DEFAULT_NUM_THREADS
    sample_rate: int = SAMPLE_RATE
    feature_dim: int = 80
    enable_endpoint: bool = True
    rule1_min_trailing_silence: float = RULE1_MIN_TRAILING_SILENCE
    rule2_min_trailing_silence: float = RULE2_MIN_TRAILING_SILENCE
    rule3_min_utterance_length: float = RULE3_MIN_UTTERANCE_LENGTH
    decoding_method: str = "greedy_search"
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS


@dataclass
class PocSegment:
    """一个已终态化的转写段（端点触发或输入结束触发）。"""

    text: str
    start_audio_seconds: float
    end_audio_seconds: float
    trigger: str  # "endpoint" | "final"


@dataclass
class PocStep:
    """单次喂块后的增量快照。"""

    fed_audio_seconds: float
    partial_text: str
    decode_calls: int
    endpoint_triggered: bool
    step_wall_seconds: float


@dataclass
class PocMetrics:
    """量化指标收集（对照批量基线：全链 RTF 0.55-0.60，progress.py 锚）。"""

    model_load_seconds: float = 0.0
    audio_seconds_fed: float = 0.0
    feed_wall_seconds: float = 0.0
    decode_calls: int = 0
    step_wall_seconds: list[float] = field(default_factory=list)
    first_partial_seconds: float | None = None
    endpoint_count: int = 0
    final_segment_count: int = 0
    rss_before_load_bytes: int | None = None
    rss_after_load_bytes: int | None = None
    rss_peak_bytes: int | None = None
    process_cpu_seconds: float = 0.0

    @property
    def rtf(self) -> float | None:
        """实时率 = 处理墙钟 / 音频时长（<1 表示快于实时）。"""
        if self.audio_seconds_fed <= 0:
            return None
        return self.feed_wall_seconds / self.audio_seconds_fed

    def latency_percentile(self, q: float) -> float | None:
        if not self.step_wall_seconds:
            return None
        ordered = sorted(self.step_wall_seconds)
        index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[index]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_load_seconds": round(self.model_load_seconds, 3),
            "audio_seconds_fed": round(self.audio_seconds_fed, 3),
            "feed_wall_seconds": round(self.feed_wall_seconds, 3),
            "rtf": round(self.rtf, 4) if self.rtf is not None else None,
            "decode_calls": self.decode_calls,
            "step_latency_p50_seconds": round(self.latency_percentile(0.5) or 0.0, 4),
            "step_latency_p95_seconds": round(self.latency_percentile(0.95) or 0.0, 4),
            "step_latency_max_seconds": round(self.latency_percentile(1.0) or 0.0, 4),
            "first_partial_seconds": (
                round(self.first_partial_seconds, 3)
                if self.first_partial_seconds is not None
                else None
            ),
            "endpoint_count": self.endpoint_count,
            "final_segment_count": self.final_segment_count,
            "rss_before_load_bytes": self.rss_before_load_bytes,
            "rss_after_load_bytes": self.rss_after_load_bytes,
            "rss_peak_bytes": self.rss_peak_bytes,
            "process_cpu_seconds": round(self.process_cpu_seconds, 3),
        }


def _process_memory_bytes() -> tuple[int | None, int | None]:
    """(当前工作集, 峰值工作集) 字节；平台不可用时 (None, None)。

    Windows 走 psapi GetProcessMemoryInfo（64 位下须显式 restype/argtypes）；
    POSIX 走 resource.ru_maxrss（峰值可测、当前值不可测则当前=峰值）。
    无第三方依赖。
    """
    try:
        if sys.platform == "win32":

            class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_uint32),
                    ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = (
                ctypes.c_void_p,
                ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
                ctypes.c_uint32,
            )
            handle = kernel32.GetCurrentProcess()
            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
            if not psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                return (None, None)
            return (int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize))
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak_bytes = peak * 1024 if sys.platform != "darwin" else peak
        return (peak_bytes, peak_bytes)
    except Exception:  # noqa: BLE001 - 指标收集永不影响 POC 本体
        return (None, None)


class StreamingAsrPoc:
    """流式识别 POC：分块喂入 → 增量中间结果 → 终态结果。

    识别器默认真实构造（``_build_recognizer``）；测试可注入替身
    （``recognizer_factory`` / ``recognizer``），循环逻辑对两者一致。
    """

    def __init__(
        self,
        config: PocConfig,
        *,
        recognizer: Any = None,
        recognizer_factory: Callable[[PocConfig], Any] | None = None,
    ):
        self.config = config
        self.segments: list[PocSegment] = []
        self.metrics = PocMetrics()
        factory = recognizer_factory or _build_recognizer
        self._rss_before, _ = _process_memory_bytes()
        self.metrics.rss_before_load_bytes = self._rss_before
        started = time.perf_counter()
        self._recognizer = recognizer if recognizer is not None else factory(config)
        self.metrics.model_load_seconds = time.perf_counter() - started
        self._rss_after, self._rss_peak = _process_memory_bytes()
        self.metrics.rss_after_load_bytes = self._rss_after
        self.metrics.rss_peak_bytes = self._rss_peak
        self._stream: Any = None
        self._segment_start_seconds = 0.0
        self._audio_cursor_seconds = 0.0
        self._feed_started: float | None = None
        self._cpu_started: float | None = None

    # -- 流生命周期 -------------------------------------------------------

    @property
    def recognizer(self) -> Any:
        """底层识别器（真实 OnlineRecognizer 或注入替身）。"""
        return self._recognizer

    def create_stream(self) -> Any:
        self._stream = self._recognizer.create_stream()
        return self._stream

    def reset_run_state(self) -> None:
        """清零运行累计（多轮复用同一已加载识别器时用，如基准的冒烟段）。"""
        self.segments.clear()
        self.metrics.audio_seconds_fed = 0.0
        self.metrics.feed_wall_seconds = 0.0
        self.metrics.decode_calls = 0
        self.metrics.step_wall_seconds.clear()
        self.metrics.first_partial_seconds = None
        self.metrics.endpoint_count = 0
        self.metrics.final_segment_count = 0
        self._feed_started = None
        self._cpu_started = None
        self._audio_cursor_seconds = 0.0
        self._segment_start_seconds = 0.0

    def feed_samples(
        self,
        samples: Sequence[float],
        *,
        sample_rate: int | None = None,
    ) -> PocStep:
        """喂入一个音频块并推进增量解码（一个喂块=一次完整增量步）。"""
        if self._stream is None:
            raise PocError("stream not created; call create_stream() first")
        rate = sample_rate or self.config.sample_rate
        audio_seconds = len(samples) / float(rate)
        started = time.perf_counter()
        if self._feed_started is None:
            self._feed_started = started
            self._cpu_started = time.process_time()
        self._stream.accept_waveform(rate, list(samples))
        decode_calls = 0
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
            decode_calls += 1
        partial = str(self._recognizer.get_result(self._stream) or "")
        if partial and self.metrics.first_partial_seconds is None:
            self.metrics.first_partial_seconds = time.perf_counter() - self._feed_started
        endpoint = False
        if self.config.enable_endpoint and self._recognizer.is_endpoint(self._stream):
            if partial:
                self.segments.append(
                    PocSegment(
                        text=partial,
                        start_audio_seconds=self._segment_start_seconds,
                        end_audio_seconds=self._audio_cursor_seconds,
                        trigger="endpoint",
                    )
                )
                self.metrics.endpoint_count += 1
            self._segment_start_seconds = self._audio_cursor_seconds
            self._recognizer.reset(self._stream)
            endpoint = True
        step_wall = time.perf_counter() - started
        self.metrics.audio_seconds_fed += audio_seconds
        self.metrics.decode_calls += decode_calls
        self.metrics.step_wall_seconds.append(step_wall)
        self.metrics.feed_wall_seconds = time.perf_counter() - self._feed_started
        self._audio_cursor_seconds += audio_seconds
        return PocStep(
            fed_audio_seconds=audio_seconds,
            partial_text=partial,
            decode_calls=decode_calls,
            endpoint_triggered=endpoint,
            step_wall_seconds=step_wall,
        )

    def feed_chunks(
        self,
        chunks: Iterable[Sequence[float]],
        *,
        sample_rate: int | None = None,
    ) -> list[PocStep]:
        return [self.feed_samples(chunk, sample_rate=sample_rate) for chunk in chunks]

    def finish(self) -> PocSegment | None:
        """输入结束：冲刷缓冲，产出终态段（trigger="final"）。"""
        if self._stream is None:
            raise PocError("stream not created; call create_stream() first")
        self._stream.input_finished()
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
            self.metrics.decode_calls += 1
        text = str(self._recognizer.get_result(self._stream) or "")
        segment: PocSegment | None = None
        if text:
            segment = PocSegment(
                text=text,
                start_audio_seconds=self._segment_start_seconds,
                end_audio_seconds=self._audio_cursor_seconds,
                trigger="final",
            )
            self.segments.append(segment)
            self.metrics.final_segment_count += 1
        if self._cpu_started is not None:
            self.metrics.process_cpu_seconds = time.process_time() - self._cpu_started
        return segment


# -- 合成音频（纯 stdlib，确定性） ----------------------------------------


def synthesize_speech_pcm(
    total_seconds: float,
    *,
    seed: int = 20261007,
    sample_rate: int = SAMPLE_RATE,
    burst_seconds: float = _BURST_SECONDS,
    gap_seconds: float = _GAP_SECONDS,
) -> list[float]:
    """确定性合成「讲课节奏」音频：发声段（谐波+颤音）与静音段交错。

    发声段频谱随时间变化以产生持续可解码的音素流；静音段长于 rule2
    （1.2s）以触发端点。输出已归一到 [-1, 1]。
    """
    if total_seconds <= 0:
        raise PocError("total_seconds must be positive")
    rng = random.Random(seed)
    samples: list[float] = []
    position = 0.0
    voiced = True
    while position < total_seconds:
        span = min(burst_seconds if voiced else gap_seconds, total_seconds - position)
        count = int(span * sample_rate)
        base = 140.0 + 60.0 * rng.random()
        for index in range(count):
            t = (position + index / sample_rate)
            if voiced:
                vibrato = 1.0 + 0.02 * math.sin(2 * math.pi * 4.5 * t)
                envelope = 0.5 * (1 + math.sin(math.pi * index / max(1, count)))
                value = (
                    math.sin(2 * math.pi * base * vibrato * t)
                    + 0.5 * math.sin(2 * math.pi * base * 2.0 * t)
                    + 0.25 * math.sin(2 * math.pi * base * 3.3 * t)
                )
                samples.append(0.25 * envelope * value * (0.8 + 0.4 * rng.random()))
            else:
                samples.append(0.001 * (rng.random() - 0.5))
        position += span
        voiced = not voiced
    return samples


def chunk_pcm(
    samples: Sequence[float],
    chunk_seconds: float,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> list[list[float]]:
    """按固定时长切块（末块不足一块保持原样）。"""
    if chunk_seconds <= 0:
        raise PocError("chunk_seconds must be positive")
    size = max(1, int(chunk_seconds * sample_rate))
    return [
        list(samples[start : start + size]) for start in range(0, len(samples), size)
    ]


def load_wav_mono(path: Path) -> tuple[list[float], int]:
    """stdlib 读取 wav（单声道化、[-1,1] 归一），供真实语音冒烟。"""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if width == 2:
        count = len(raw) // 2
        values = list(struct.unpack(f"<{count}h", raw[: count * 2]))
        scale = 32768.0
    elif width == 4:
        count = len(raw) // 4
        values = list(struct.unpack(f"<{count}i", raw[: count * 4]))
        scale = 2147483648.0
    else:
        raise PocError(f"unsupported wav sample width: {width}")
    if channels > 1:
        values = [
            sum(values[start : start + channels]) / channels
            for start in range(0, len(values), channels)
        ]
    return [value / scale for value in values], rate


# -- 手工基准入口 ---------------------------------------------------------


def run_benchmark(
    *,
    model_dir: Path | None = None,
    seconds: float = 60.0,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    num_threads: int = DEFAULT_NUM_THREADS,
    wav_path: Path | None = None,
) -> dict[str, Any]:
    """跑一次量化 POC 并返回 JSON 可序列化指标（数字进结果文件）。"""
    encoder, decoder, tokens = resolve_model_paths(model_dir)
    config = PocConfig(
        encoder=encoder,
        decoder=decoder,
        tokens=tokens,
        num_threads=num_threads,
        chunk_seconds=chunk_seconds,
    )
    poc = StreamingAsrPoc(config)
    result: dict[str, Any] = {"model_dir": str(encoder.parent), "threads": num_threads}
    result["model_load_seconds"] = round(poc.metrics.model_load_seconds, 3)
    result["rss_after_load_bytes"] = poc.metrics.rss_after_load_bytes

    # 真实语音冒烟（test_wavs/0.wav ≈10s 人声）：证识别非空，不评字准。
    if wav_path is not None and Path(wav_path).is_file():
        samples, rate = load_wav_mono(Path(wav_path))
        poc.create_stream()
        for chunk in chunk_pcm(samples, chunk_seconds, sample_rate=rate):
            poc.feed_samples(chunk, sample_rate=rate)
        final = poc.finish()
        result["wav_smoke"] = {
            "path": str(wav_path),
            "audio_seconds": round(len(samples) / rate, 2),
            "final_text_chars": len(final.text) if final else 0,
            "segments": len(poc.segments),
        }
        poc.reset_run_state()

    # 合成讲课节奏流式跑：RTF/增量延迟/端点行为量化。
    samples = synthesize_speech_pcm(seconds)
    chunks = chunk_pcm(samples, chunk_seconds)
    poc.create_stream()
    steps = poc.feed_chunks(chunks)
    final = poc.finish()
    endpointed = [seg for seg in poc.segments if seg.trigger == "endpoint"]
    partials_seen = sum(1 for step in steps if step.partial_text)
    result["synthetic_stream"] = {
        "requested_seconds": seconds,
        "chunk_seconds": chunk_seconds,
        "metrics": poc.metrics.as_dict(),
        "partial_chunks": partials_seen,
        "endpoint_segments": len(endpointed),
        "final_text_chars": len(final.text) if final else 0,
        "first_segment_text_chars": len(endpointed[0].text) if endpointed else 0,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--threads", type=int, default=DEFAULT_NUM_THREADS)
    parser.add_argument(
        "--wav",
        type=Path,
        default=None,
        help="optional real-speech wav smoke before the synthetic run",
    )
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()
    result = run_benchmark(
        model_dir=args.model_dir,
        seconds=args.seconds,
        chunk_seconds=args.chunk_seconds,
        num_threads=args.threads,
        wav_path=args.wav,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
