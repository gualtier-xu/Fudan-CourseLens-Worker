"""D13-PROD 下课双门触发器钉：双门语义/幂等/闭集事件/snapshot 刷新请求."""

from __future__ import annotations

import unittest

from courselens_worker.streaming_trigger import (
    CLASS_END_EVENT,
    SNAPSHOT_REFRESH_EVENT,
    ClassEndTrigger,
    TriggerStateError,
)

SCHEDULED_END = 1_000_000.0


class DualGateTests(unittest.TestCase):
    def _trigger(self, emit=None) -> ClassEndTrigger:
        return ClassEndTrigger(scheduled_end_epoch=SCHEDULED_END, emit=emit)

    def test_schedule_gate_alone_does_not_fire(self):
        trigger = self._trigger()
        trigger.observe_stream_state("pending", observed_at_epoch=SCHEDULED_END - 10)
        self.assertIsNone(trigger.maybe_fire(SCHEDULED_END + 60))

    def test_stream_flip_alone_does_not_fire(self):
        trigger = self._trigger()
        trigger.observe_stream_state("live", observed_at_epoch=SCHEDULED_END - 100)
        trigger.observe_stream_state("ended", observed_at_epoch=SCHEDULED_END - 50)
        self.assertTrue(trigger.stream_flip_observed)
        self.assertIsNone(trigger.maybe_fire(SCHEDULED_END - 1))

    def test_both_gates_fire_once(self):
        events: list[dict] = []
        trigger = self._trigger(emit=events.append)
        trigger.observe_stream_state("live", observed_at_epoch=SCHEDULED_END - 100)
        trigger.observe_stream_state("ended", observed_at_epoch=SCHEDULED_END - 50)
        event = trigger.maybe_fire(SCHEDULED_END + 5)
        self.assertIsNotNone(event)
        self.assertEqual(event["event"], CLASS_END_EVENT)
        self.assertEqual(event["scheduled_end_epoch"], SCHEDULED_END)
        self.assertEqual(event["stream_flip_epoch"], SCHEDULED_END - 50)
        # 幂等：二次调用不再触发。
        self.assertIsNone(trigger.maybe_fire(SCHEDULED_END + 10))
        # 双事件：下课事件 + snapshot 刷新请求（回导非推送）。
        self.assertEqual(
            [item["event"] for item in events],
            [CLASS_END_EVENT, SNAPSHOT_REFRESH_EVENT],
        )
        self.assertEqual(events[1]["reason"], "class_end")

    def test_early_stream_end_fires_at_scheduled_time(self):
        """教师提前收流：翻转门先满足，课表门到点即触发（不提前）。"""
        trigger = self._trigger()
        trigger.observe_stream_state("live", observed_at_epoch=SCHEDULED_END - 600)
        trigger.observe_stream_state("ended", observed_at_epoch=SCHEDULED_END - 300)
        self.assertIsNone(trigger.maybe_fire(SCHEDULED_END - 1))
        self.assertIsNotNone(trigger.maybe_fire(SCHEDULED_END))

    def test_late_stream_end_fires_at_flip(self):
        """教师拖堂：课表门先满足，翻转观测到才触发。"""
        trigger = self._trigger()
        trigger.observe_stream_state("live", observed_at_epoch=SCHEDULED_END - 10)
        self.assertIsNone(trigger.maybe_fire(SCHEDULED_END + 600))
        trigger.observe_stream_state("ended", observed_at_epoch=SCHEDULED_END + 700)
        self.assertIsNotNone(trigger.maybe_fire(SCHEDULED_END + 701))

    def test_never_live_stream_never_fires(self):
        """纯录播课（流从未 live）：ended 也永不构成翻转门。"""
        trigger = self._trigger()
        trigger.observe_stream_state("ended", observed_at_epoch=SCHEDULED_END - 50)
        self.assertFalse(trigger.stream_flip_observed)
        self.assertIsNone(trigger.maybe_fire(SCHEDULED_END + 60))

    def test_closed_set_rejects_unknown_state(self):
        trigger = self._trigger()
        with self.assertRaises(TriggerStateError):
            trigger.observe_stream_state("recording")
        with self.assertRaises(TriggerStateError):
            trigger.observe_stream_state("")

    def test_event_payload_carries_no_content(self):
        """闭集载荷：只有事件名/时刻，无课程内容面。"""
        trigger = self._trigger()
        trigger.observe_stream_state("live")
        trigger.observe_stream_state("ended", observed_at_epoch=SCHEDULED_END - 50)
        event = trigger.maybe_fire(SCHEDULED_END + 2.0)
        self.assertEqual(
            set(event),
            {"event", "fired_at_epoch", "scheduled_end_epoch", "stream_flip_epoch"},
        )


if __name__ == "__main__":
    unittest.main()
