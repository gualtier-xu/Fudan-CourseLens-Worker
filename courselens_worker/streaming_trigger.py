"""下课双门触发器 + snapshot 刷新请求（D13-PROD 施工清单第 3 步）.

D13-RESEARCH §4-P2 定稿的下课检测语义：**课表结束时刻 + 直播流 state 翻转
双门防误触发**。单门不触发——

- 只有课表到点、直播流未翻转（教师拖堂/流卡住）→ 等翻转；
- 只有直播流翻转（教师提前收流）→ 等课表时刻；
- 直播流从未 live 过（纯录播课，无此通道）→ 永不触发，流式腿不参与该课。

触发时同时**拉起一次 snapshot 刷新请求**：结果回导是客户端拉取非推送
（``src/runtime/automation.py`` 的 snapshot 导入面），事件只携带闭集标识
与计数/时刻，不含任何课程内容。幂等：一次会话只触发一次。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

CLASS_END_EVENT = "streaming_class_end"
SNAPSHOT_REFRESH_EVENT = "streaming_snapshot_refresh"

STREAM_STATE_CLOSED = frozenset({"pending", "live", "ended", "unknown"})


class TriggerStateError(ValueError):
    """闭集外状态一律拒绝（调用方 bug 显式暴露，不静默吞）。"""


@dataclass
class ClassEndTrigger:
    """双门触发器：课表门（墙钟到点）∧ 流翻转门（live→ended 观测到）。

    ``emit`` 为遥测/事件出口（生产接线传 runner 的遥测面；测试收事件）。
    """

    scheduled_end_epoch: float
    emit: Callable[[dict[str, Any]], None] | None = None
    _seen_live: bool = field(default=False, init=False)
    _stream_ended: bool = field(default=False, init=False)
    _stream_flip_epoch: float | None = field(default=None, init=False)
    _fired_at_epoch: float | None = field(default=None, init=False)

    # -- 门观测 -----------------------------------------------------------

    def observe_stream_state(
        self, state: str, *, observed_at_epoch: float | None = None
    ) -> None:
        """观测一次直播流状态（闭集：pending/live/ended/unknown）。

        翻转语义：先观测到 ``live``、其后观测到 ``ended`` 才算翻转门满足；
        从未 live 的流（纯录播）永远不满足翻转门。
        """
        if state not in STREAM_STATE_CLOSED:
            raise TriggerStateError(f"stream state outside closed set: {state!r}")
        if state == "live":
            self._seen_live = True
        elif state == "ended" and self._seen_live and not self._stream_ended:
            self._stream_ended = True
            self._stream_flip_epoch = observed_at_epoch

    @property
    def stream_flip_observed(self) -> bool:
        return self._stream_ended

    @property
    def seen_live(self) -> bool:
        return self._seen_live

    def schedule_gate_reached(self, now_epoch: float) -> bool:
        return now_epoch >= self.scheduled_end_epoch

    # -- 触发 -------------------------------------------------------------

    @property
    def fired(self) -> bool:
        return self._fired_at_epoch is not None

    def maybe_fire(self, now_epoch: float) -> dict[str, Any] | None:
        """双门齐备即触发一次；幂等，重复调用返回 ``None``。

        事件载荷只含闭集标识与时刻——无课程内容、无 URL、无账号值。
        """
        if self._fired_at_epoch is not None:
            return None
        if not (self.schedule_gate_reached(now_epoch) and self._stream_ended):
            return None
        self._fired_at_epoch = float(now_epoch)
        event: dict[str, Any] = {
            "event": CLASS_END_EVENT,
            "fired_at_epoch": self._fired_at_epoch,
            "scheduled_end_epoch": float(self.scheduled_end_epoch),
            "stream_flip_epoch": self._stream_flip_epoch,
        }
        if self.emit is not None:
            self.emit(event)
            # 回导非推送：触发同时请求一次客户端 snapshot 刷新（拉取面语义）。
            self.emit(
                {
                    "event": SNAPSHOT_REFRESH_EVENT,
                    "reason": "class_end",
                    "fired_at_epoch": self._fired_at_epoch,
                }
            )
        return event
