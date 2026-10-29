"""Streaming ASR production adapter（D13-PROD 施工清单第 2 步）.

D13-IMPL POC（``streaming_asr_poc.py``，不接生产管线）验证过的
sherpa-onnx ``OnlineRecognizer`` + 流式 Paraformer 双语模型（zh-en）在这里
转成生产形态。本模块**不 import POC 模块**（POC 保持不入生产 import 图）；
POC 验证的三态 API（分块喂入 → 增量中间结果 → 终态结果）按同一语义重实现。

生产语义（D13-RESEARCH §4-P2 设计，POC 已证量级 RTF 0.04-0.17）：

- **段级时间锚**：段带讲次内绝对毫秒时刻（``start_ms``/``end_ms``），
  上课中边听边转写，下课时刻 transcript 已就绪——课后只剩摘要一跳。
- **fail-closed 回落**：运行时缺位/模型缺席/识别失败都以闭集码显式失败
  （``StreamingAsrError.code``），由会话层（streaming_session）降级为
  「流式腿不完整」标记——录播发布后现链照跑产精修稿，**绝不挡课、绝不
  产出伪造识别结果**。
- **识别器可注入**：``StreamingTranscriber(recognizer=...)`` 接受测试替身，
  循环逻辑（喂入→增量→端点→终态）不依赖真实模型即可钉死。
- **遥测闭集**：计数器、秒数与闭集 stage 标识；绝不携带识别文本、路径、
  URL 或账号值（与 runner._progress / asr / llm 遥测纪律同款）。
- **模型分发**：模型文件绝不进 worker 树（公共边界门零容忍）；目录解析
  env ``STREAMING_MODEL_DIR``（install_models 写出）> 仓根 ``.models/``
  条目。sha256 钉在 install_models.py（第 1 步）。
- **evidence 命名空间**：``export_segments`` 产出的段字典带
  ``source="streaming"`` 标记，检查点键用 ``streaming_segments``——与批量链
  的 ``raw_{rough}``/``raw_{refined}`` 槽位零碰撞（清单第 6 步共存语义）。
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

SAMPLE_RATE = 16_000
DEFAULT_NUM_THREADS = 2
# sherpa-onnx 端点检测三规则（官方默认值；与流式 Paraformer 配套）。
RULE1_MIN_TRAILING_SILENCE = 2.4
RULE2_MIN_TRAILING_SILENCE = 1.2
RULE3_MIN_UTTERANCE_LENGTH = 20.0

MODEL_DIR_ENV = "STREAMING_MODEL_DIR"
DEFAULT_MODEL_DIRNAME = "streaming-paraformer-bilingual-zh-en"
ENCODER_FILENAME = "encoder.int8.onnx"
DECODER_FILENAME = "decoder.int8.onnx"
TOKENS_FILENAME = "tokens.txt"

SEGMENT_SOURCE = "streaming"
SEGMENT_TRIGGER_ENDPOINT = "endpoint"
SEGMENT_TRIGGER_FINAL = "final"

# 闭集失败码（会话层按码降级，绝不向学生面透出原始异常文本）。
CODE_MODEL_MISSING = "streaming_model_missing"
CODE_RUNTIME_UNAVAILABLE = "streaming_runtime_unavailable"
CODE_RECOGNIZER_FAILED = "streaming_recognizer_failed"


class StreamingAsrError(RuntimeError):
    """流式腿闭集失败：code 只取 STREAMING_ERROR_CODES，不携带敏感值。"""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(code if not detail else f"{code}: {detail}")
        self.code = code


STREAMING_ERROR_CODES = frozenset(
    {CODE_MODEL_MISSING, CODE_RUNTIME_UNAVAILABLE, CODE_RECOGNIZER_FAILED}
)


def default_model_dir() -> Path:
    """模型目录解析：env 覆盖 > 仓根 ``.models/<DEFAULT_MODEL_DIRNAME>``。

    模型缓存不进 worker 树（公共边界门对媒体文件零容忍）；生产分发走
    install_models 的 sha256 钉纪律。
    """
    raw = os.environ.get(MODEL_DIR_ENV, "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[2] / ".models" / DEFAULT_MODEL_DIRNAME


def resolve_model_paths(model_dir: Path | None = None) -> tuple[Path, Path, Path]:
    directory = Path(model_dir) if model_dir else default_model_dir()
    encoder = directory / ENCODER_FILENAME
    decoder = directory / DECODER_FILENAME
    tokens = directory / TOKENS_FILENAME
    missing = [str(path) for path in (encoder, decoder, tokens) if not path.is_file()]
    if missing:
        raise StreamingAsrError(
            CODE_MODEL_MISSING, "missing " + ", ".join(Path(item).name for item in missing)
        )
    return encoder, decoder, tokens


def sherpa_runtime_available() -> bool:
    """真实 sherpa-onnx 运行时探测（conftest 的 Mock 桩不算运行时）。"""
    try:
        import sherpa_onnx  # noqa: PLC0415 - 惰性导入，保持零依赖导入面
    except Exception:  # noqa: BLE001 - 探测口：任何导入失败=不可用
        return False
    return type(sherpa_onnx).__module__ != "unittest.mock"


def _build_recognizer(config: "StreamingAsrConfig") -> Any:
    """构造真实 OnlineRecognizer（只在真实运行时路径被调用）。"""
    if not sherpa_runtime_available():
        raise StreamingAsrError(
            CODE_RUNTIME_UNAVAILABLE,
            "sherpa-onnx runtime stubbed or missing; real streaming requires the "
            "pinned runtime with the streaming-paraformer model installed",
        )
    import sherpa_onnx  # noqa: PLC0415

    try:
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
    except StreamingAsrError:
        raise
    except Exception as exc:  # noqa: BLE001 - 运行时构造失败收拢为闭集码
        raise StreamingAsrError(CODE_RECOGNIZER_FAILED, type(exc).__name__) from exc


@dataclass
class StreamingAsrConfig:
    """流式识别配置（默认值=POC 实测量级的最稳形态：2 线程 1s 块）。"""

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


def load_default_config(model_dir: Path | None = None) -> StreamingAsrConfig:
    encoder, decoder, tokens = resolve_model_paths(model_dir)
    return StreamingAsrConfig(encoder=encoder, decoder=decoder, tokens=tokens)


@dataclass(frozen=True)
class StreamingSegment:
    """一个已终态化的转写段（端点触发或输入结束触发）。

    ``start_ms``/``end_ms`` 是讲次内绝对毫秒时刻（时间锚），由喂入侧的
    讲次偏移推进，与媒体时间轴对齐——课后回导/与录播稿对位都吃这个锚。
    """

    text: str
    start_ms: int
    end_ms: int
    trigger: str  # "endpoint" | "final"

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "source": SEGMENT_SOURCE,
        }


def export_segments(segments: Sequence[StreamingSegment]) -> list[dict[str, Any]]:
    """导出为检查点/回导友好的段字典列表（evidence 命名空间标记在案）。"""
    return [segment.as_dict() for segment in segments]


@dataclass
class StreamingMetrics:
    """量化指标（遥测只出计数与秒数，不出内容）。"""

    model_load_seconds: float = 0.0
    audio_seconds_fed: float = 0.0
    decode_calls: int = 0
    endpoint_count: int = 0
    segment_count: int = 0
    first_partial_seconds: float | None = None
    feed_wall_seconds: float = 0.0

    @property
    def rtf(self) -> float | None:
        if self.audio_seconds_fed <= 0:
            return None
        return self.feed_wall_seconds / self.audio_seconds_fed

    def telemetry_line(self) -> str:
        rtf = f"{self.rtf:.3f}" if self.rtf is not None else "na"
        first = (
            f"{self.first_partial_seconds:.3f}"
            if self.first_partial_seconds is not None
            else "na"
        )
        return (
            f"stage=streaming fed_s={self.audio_seconds_fed:.1f} "
            f"decode={self.decode_calls} endpoints={self.endpoint_count} "
            f"segments={self.segment_count} rtf={rtf} first_partial_s={first}"
        )


class StreamingTranscriber:
    """流式识别生产适配层：分块喂入 → 增量中间结果 → 终态结果。

    与 POC 同一循环语义；生产差异=讲次内绝对毫秒时间锚、闭集失败码、
    遥测发射点、段导出格式。识别器默认真实构造（``load_default_config``
    + ``_build_recognizer``），测试注入 ``recognizer``/``recognizer_factory``。
    """

    def __init__(
        self,
        config: StreamingAsrConfig,
        *,
        recognizer: Any = None,
        recognizer_factory: Callable[[StreamingAsrConfig], Any] | None = None,
        emit_telemetry: Callable[[str], None] | None = None,
    ):
        self.config = config
        self.metrics = StreamingMetrics()
        self._emit = emit_telemetry or (lambda line: None)
        factory = recognizer_factory or _build_recognizer
        started = time.perf_counter()
        try:
            self._recognizer = (
                recognizer if recognizer is not None else factory(config)
            )
        except StreamingAsrError:
            raise
        except Exception as exc:  # noqa: BLE001 - 收拢为闭集码
            raise StreamingAsrError(CODE_RECOGNIZER_FAILED, type(exc).__name__) from exc
        self.metrics.model_load_seconds = time.perf_counter() - started
        self._stream: Any = None
        self._segments: list[StreamingSegment] = []
        self._segment_start_ms = 0
        self._cursor_ms = 0
        self._feed_started: float | None = None

    # -- 流生命周期 -------------------------------------------------------

    @property
    def recognizer(self) -> Any:
        return self._recognizer

    @property
    def segments(self) -> tuple[StreamingSegment, ...]:
        return tuple(self._segments)

    @property
    def audio_ms_fed(self) -> int:
        return self._cursor_ms

    def create_stream(self) -> Any:
        self._stream = self._recognizer.create_stream()
        return self._stream

    def _require_stream(self) -> None:
        if self._stream is None:
            raise StreamingAsrError(CODE_RECOGNIZER_FAILED, "stream not created")

    def feed(
        self,
        samples: Sequence[float],
        *,
        sample_rate: int | None = None,
        offset_ms: int | None = None,
    ) -> str:
        """喂入一个音频块，推进增量解码，返回当前增量中间文本。

        ``offset_ms``：本块在讲次内的绝对起点（重连/追赶时显式对锚）；
        缺省沿内部游标连续推进。
        """
        self._require_stream()
        if offset_ms is not None:
            # 显式重锚（重连/追赶）：游标与当前段起点一起对到讲次时刻。
            self._cursor_ms = max(0, int(offset_ms))
            self._segment_start_ms = self._cursor_ms
        rate = sample_rate or self.config.sample_rate
        if rate != self.config.sample_rate:
            raise StreamingAsrError(
                CODE_RECOGNIZER_FAILED, f"sample rate {rate} != configured"
            )
        audio_seconds = len(samples) / float(rate)
        started = time.perf_counter()
        if self._feed_started is None:
            self._feed_started = started
        try:
            self._stream.accept_waveform(rate, list(samples))
            decode_calls = 0
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
                decode_calls += 1
            partial = str(self._recognizer.get_result(self._stream) or "")
        except StreamingAsrError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StreamingAsrError(CODE_RECOGNIZER_FAILED, type(exc).__name__) from exc
        if partial.strip() and self.metrics.first_partial_seconds is None:
            self.metrics.first_partial_seconds = (
                time.perf_counter() - self._feed_started
            )
        if self.config.enable_endpoint and self._recognizer.is_endpoint(self._stream):
            if partial.strip():
                self._capture_segment(
                    partial, self._cursor_ms, SEGMENT_TRIGGER_ENDPOINT
                )
                self.metrics.endpoint_count += 1
            self._segment_start_ms = self._cursor_ms + int(audio_seconds * 1000)
            self._recognizer.reset(self._stream)
        self.metrics.audio_seconds_fed += audio_seconds
        self.metrics.decode_calls += decode_calls
        self.metrics.feed_wall_seconds += time.perf_counter() - started
        self._cursor_ms += int(round(audio_seconds * 1000))
        return partial

    def finish(self) -> StreamingSegment | None:
        """输入结束：冲刷缓冲，产出终态段（trigger="final"）。"""
        self._require_stream()
        try:
            self._stream.input_finished()
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
                self.metrics.decode_calls += 1
            text = str(self._recognizer.get_result(self._stream) or "")
        except StreamingAsrError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StreamingAsrError(CODE_RECOGNIZER_FAILED, type(exc).__name__) from exc
        segment: StreamingSegment | None = None
        if text.strip():
            segment = self._capture_segment(text, self._cursor_ms, SEGMENT_TRIGGER_FINAL)
        self._emit(self.metrics.telemetry_line())
        return segment

    def _capture_segment(
        self, text: str, end_ms: int, trigger: str
    ) -> StreamingSegment:
        segment = StreamingSegment(
            text=text.strip(),
            start_ms=self._segment_start_ms,
            end_ms=max(self._segment_start_ms, int(end_ms)),
            trigger=trigger,
        )
        self._segments.append(segment)
        self.metrics.segment_count = len(self._segments)
        return segment


def checkpoint_segments_payload(segments: Sequence[StreamingSegment]) -> dict[str, Any]:
    """流式腿检查点载荷：``streaming_`` 前缀键，与批量链 raw_* 零碰撞。"""
    return {
        "streaming_segments": export_segments(segments),
        "streaming_segment_count": len(segments),
    }
