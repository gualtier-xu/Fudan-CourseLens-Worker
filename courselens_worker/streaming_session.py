"""流式转写会话组装（D13-PROD 施工清单第 3 步接线面）.

把摄取器（live_audio_ingest）→ 转写器（streaming_asr）→ 滚动摘要
（streaming_summary）→ 下课双门触发器（streaming_trigger）串成一条
「上课中边听边转写、下课摘要一跳即出」的会话流水线。

**绝不挡课**（P2 设计核心纪律）：流式腿任何失败都以闭集码降级为
``state="incomplete"`` 并如实记账——录播发布后现链照跑产精修稿，
本会话绝不向调用方抛异常、绝不产出伪造转写/摘要。

**回导非推送**：下课触发时同时发出 snapshot 刷新请求事件
（``streaming_trigger.SNAPSHOT_REFRESH_EVENT``），客户端拉取面消费。

全部产物/检查点键用 ``streaming_`` 命名空间，与批量链 raw_*/summary_*
零碰撞（evidence 命名空间共存，清单第 6 步）。
"""

from __future__ import annotations

from typing import Any, Callable

from .live_audio_ingest import INGEST_ERROR_CODES
from .streaming_asr import (
    CODE_RECOGNIZER_FAILED,
    STREAMING_ERROR_CODES,
    StreamingAsrError,
    StreamingTranscriber,
    export_segments,
)
from .streaming_summary import (
    CODE_ALL_FAILED,
    CODE_MERGE_FAILED,
    RollingSummarizer,
    StreamingSummaryError,
    valid_summary_part,
)
from .streaming_trigger import (
    CLASS_END_EVENT,
    SNAPSHOT_REFRESH_EVENT,
    ClassEndTrigger,
    TriggerStateError,
)

SESSION_STATE_RUNNING = "running"
SESSION_STATE_COMPLETED = "completed"
SESSION_STATE_INCOMPLETE = "incomplete"

CHECKPOINT_KEY_STATE = "streaming_session_state"
CHECKPOINT_KEY_INCOMPLETE_CODE = "streaming_session_incomplete_code"
CHECKPOINT_KEY_SEGMENTS = "streaming_segments"
CHECKPOINT_KEY_SUMMARY = "streaming_summary_note"

SESSION_ERROR_CODES = frozenset(
    set(STREAMING_ERROR_CODES)
    | set(INGEST_ERROR_CODES)
    | {CODE_ALL_FAILED, CODE_MERGE_FAILED}
)


class StreamingLectureSession:
    """一节课的流式转写会话：喂音频、观测直播状态、下课收口。"""

    def __init__(
        self,
        *,
        lecture_id: str,
        transcriber: StreamingTranscriber,
        summarizer: RollingSummarizer,
        trigger: ClassEndTrigger,
        emit: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.lecture_id = str(lecture_id)
        self.transcriber = transcriber
        self.summarizer = summarizer
        self.trigger = trigger
        self._emit = emit or (lambda event: None)
        self.state = SESSION_STATE_RUNNING
        self.incomplete_code: str | None = None
        self.summary_note: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []

    # -- 上课中 -----------------------------------------------------------

    def on_audio(
        self,
        samples: list[float],
        *,
        sample_rate: int | None = None,
        offset_ms: int | None = None,
    ) -> str:
        """喂入一块直播音频；流式腿失败在此降级，绝不向调用方抛。"""
        if self.state != SESSION_STATE_RUNNING:
            return ""
        try:
            return self.transcriber.feed(
                samples, sample_rate=sample_rate, offset_ms=offset_ms
            )
        except StreamingAsrError as exc:
            self._degrade(exc.code)
            return ""

    def on_stream_state(self, state: str, *, observed_at_epoch: float | None = None, now_epoch: float | None = None) -> bool:
        """观测直播流状态；双门齐备即下课收口（返回是否已触发）。"""
        if self.state != SESSION_STATE_RUNNING:
            return self.trigger.fired
        try:
            self.trigger.observe_stream_state(state, observed_at_epoch=observed_at_epoch)
        except TriggerStateError:
            # 闭集外状态：调用方 bug，按 unknown 语义忽略（不误触发）。
            return False
        if now_epoch is not None and self.trigger.maybe_fire(now_epoch) is not None:
            self.finalize()
            return True
        return False

    def on_clock_tick(self, now_epoch: float) -> bool:
        """课表门时钟滴答：到点且流已翻转则下课收口。"""
        if self.state != SESSION_STATE_RUNNING:
            return self.trigger.fired
        if self.trigger.maybe_fire(now_epoch) is not None:
            self.finalize()
            return True
        return False

    # -- 下课收口 ---------------------------------------------------------

    def finalize(self) -> dict[str, Any]:
        """终态化转写+末跳摘要+snapshot 刷新请求；幂等。

        摘要腿失败不毁掉已就绪的转写稿：state 仍收口，闭集码如实记账
        （转写稿先行回导，摘要留待现链），绝不伪造空笔记。
        """
        if self.state != SESSION_STATE_RUNNING:
            return self.result()
        try:
            self.transcriber.finish()
        except StreamingAsrError as exc:
            self._degrade(exc.code)
            return self.result()
        segments = export_segments(self.transcriber.segments)
        try:
            note = self.summarizer.finalize(title=self.lecture_id)
        except StreamingSummaryError as exc:
            self.incomplete_code = exc.code
            self.summary_note = None
        else:
            if valid_summary_part(note):
                self.summary_note = note
            else:
                self.incomplete_code = CODE_MERGE_FAILED
                self.summary_note = None
        self.state = (
            SESSION_STATE_COMPLETED
            if self.summary_note is not None and self.incomplete_code is None
            else SESSION_STATE_INCOMPLETE
        )
        self._record(
            {
                "event": "streaming_session_finalized",
                "state": self.state,
                "segments": len(segments),
                "incomplete_code": self.incomplete_code,
            }
        )
        # 回导非推送：收口即请求一次客户端 snapshot 刷新（幂等重发无妨）。
        self._record(
            {
                "event": SNAPSHOT_REFRESH_EVENT,
                "reason": "session_finalized",
            }
        )
        return self.result()

    def degrade(self, code: str) -> dict[str, Any]:
        """显式降级入口（如摄取腿的闭集码转发）；幂等收口不删已得段。"""
        if self.state == SESSION_STATE_RUNNING:
            self._degrade(code)
        return self.result()

    def _degrade(self, code: str) -> None:
        self.incomplete_code = code if code in SESSION_ERROR_CODES else CODE_RECOGNIZER_FAILED
        self.state = SESSION_STATE_INCOMPLETE
        try:
            self.transcriber.finish()
        except StreamingAsrError:
            pass
        self._record(
            {
                "event": "streaming_session_degraded",
                "code": self.incomplete_code,
                "segments": len(self.transcriber.segments),
            }
        )

    def _record(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        self._emit(event)

    # -- 产物 -------------------------------------------------------------

    def result(self) -> dict[str, Any]:
        """会话产物（闭集面：状态/码/段/笔记/检查点键全 ``streaming_`` 命名）。"""
        return {
            "lecture_id": self.lecture_id,
            CHECKPOINT_KEY_STATE: self.state,
            CHECKPOINT_KEY_INCOMPLETE_CODE: self.incomplete_code,
            CHECKPOINT_KEY_SEGMENTS: export_segments(self.transcriber.segments),
            CHECKPOINT_KEY_SUMMARY: self.summary_note,
            "checkpoint": self.checkpoint(),
        }

    def checkpoint(self) -> dict[str, Any]:
        """会话级检查点：转写段+滚动摘要检查点（均可断点续跑/恢复）。"""
        payload = self.summarizer.checkpoint()
        payload[CHECKPOINT_KEY_SEGMENTS] = export_segments(self.transcriber.segments)
        payload[CHECKPOINT_KEY_STATE] = self.state
        payload[CHECKPOINT_KEY_INCOMPLETE_CODE] = self.incomplete_code
        return payload
