"""ARCH-P1-2-R4：retry 单源驱动钉测（worker/tests/test_retry_ladder.py）。

两层钉：
①驱动级——``run_ladder`` 的尝试序、退避形态（常数直传/可调用按失败尝试号）、
梯尽上抛末次失败载体（对象身份）、非 LadderFailure 异常零睡眠穿透、
``before_retry`` 时序、``attempts >= 1`` 前置。
②站点级——llm.py 转换站经真实调用面过驱动：term 窗（site 2）瞬态一次重试
+1.0 退避+闭集遥测行、梯尽上抛末次传输 LLMError（睡眠恰一次）。其余站点
（proofread/summary 窗、summary 合并、answer、judge、review-views）的重试与
耗尽语义由既有钉面覆盖（test_subtitle_correction / test_summary_chain_hardening
/ test_answer_chain_hardening / test_quality_judge / test_review_views）。
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from courselens_worker.llm import LLMError, term_proofread_segments
from courselens_worker.retry import (
    LadderExhausted,
    LadderFailure,
    run_ladder,
)


def seg(start_ms, end_ms, text):
    return {"start_ms": start_ms, "end_ms": end_ms, "text": text}


class RunLadderDriverTests(unittest.TestCase):
    def test_first_attempt_success_zero_sleep(self):
        slept = []
        seen = []
        result = run_ladder(
            lambda attempt: seen.append(attempt) or "ok",
            3,
            seconds_between=1.0,
            sleep=slept.append,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(seen, [0])
        self.assertEqual(slept, [])

    def test_retry_sequence_and_numeric_backoff_forwarded_as_is(self):
        attempts_seen = []
        slept = []

        def attempt_fn(attempt):
            attempts_seen.append(attempt)
            if attempt < 2:
                raise LadderFailure(LLMError(f"transient-{attempt}"))
            return "ok"

        result = run_ladder(
            attempt_fn, 3, seconds_between=1.5, sleep=slept.append,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(attempts_seen, [0, 1, 2])
        # 每次可重试失败恰一次退避，常数逐字节直传。
        self.assertEqual(slept, [1.5, 1.5])

    def test_callable_backoff_receives_failed_attempt_number(self):
        slept = []
        calls = []

        def backoff(attempt):
            calls.append(attempt)
            return attempt + 0.25

        def attempt_fn(attempt):
            if attempt == 0:
                raise LadderFailure(LLMError("transient"))
            return "ok"

        result = run_ladder(
            attempt_fn, 2, seconds_between=backoff, sleep=slept.append,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(calls, [0])
        self.assertEqual(slept, [0.25])

    def test_exhaustion_raises_last_error_identity(self):
        first, second = LLMError("first"), LLMError("second")

        def attempt_fn(attempt):
            raise LadderFailure(second if attempt else first)

        with self.assertRaises(LLMError) as caught:
            run_ladder(attempt_fn, 2, seconds_between=1.0, sleep=lambda _: None)
        # 梯尽上抛的是最后一次失败载体本身（对象身份，非副本）。
        self.assertIs(caught.exception, second)

    def test_ladder_exhausted_payload_raised_at_exhaustion(self):
        sentinel = LadderExhausted()

        def attempt_fn(attempt):
            if attempt + 1 < 2:
                raise LadderFailure(sentinel)
            raise LadderExhausted()

        with self.assertRaises(LadderExhausted):
            run_ladder(attempt_fn, 2, seconds_between=1.0, sleep=lambda _: None)

    def test_non_ladder_exception_passes_through_without_sleep(self):
        slept = []
        boom = ValueError("deterministic")

        def attempt_fn(attempt):
            raise boom

        with self.assertRaises(ValueError) as caught:
            run_ladder(attempt_fn, 3, seconds_between=1.0, sleep=slept.append)
        self.assertIs(caught.exception, boom)
        self.assertEqual(slept, [])

    def test_before_retry_called_before_sleep_with_next_attempt(self):
        order = []

        def before_retry(next_attempt, error):
            order.append(("before_retry", next_attempt, error))

        def attempt_fn(attempt):
            if attempt == 0:
                failure = LLMError("transient")
                order.append(("fail", failure))
                raise LadderFailure(failure)
            return "ok"

        result = run_ladder(
            attempt_fn, 2, seconds_between=1.0,
            sleep=lambda delay: order.append(("sleep", delay)),
            before_retry=before_retry,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(
            [entry[0] for entry in order],
            ["fail", "before_retry", "sleep"],
        )
        self.assertEqual(order[1][1], 1)
        self.assertIs(order[1][2], order[0][1])

    def test_attempts_below_one_asserts(self):
        with self.assertRaises(AssertionError):
            run_ladder(lambda _: "ok", 0, seconds_between=1.0, sleep=lambda _: None)


class TermWindowLadderSiteTests(unittest.TestCase):
    """site 2 in vivo：term 窗过真实 run_ladder 的重试/退避/遥测/梯尽语义。"""

    def test_transient_transport_error_retries_once_with_closed_set_telemetry(self):
        valid = json.dumps([{"id": "t0", "old": "能耐图", "new": "能带图"}])
        slept = []
        telemetry = []

        with patch(
            "courselens_worker.llm._chat",
            side_effect=[LLMError("AI request failed: ConnectionError"), valid],
        ), \
                patch("courselens_worker.llm.time.sleep", side_effect=slept.append), \
                patch("courselens_worker.llm._emit_telemetry",
                      side_effect=telemetry.append):
            result = term_proofread_segments(
                "secret", [seg(0, 1000, "就能得到我的这样一个能耐图")], terms=(),
            )
        self.assertEqual(result[0]["text"], "就能得到我的这样一个能带图")
        # 首败即愈：恰 1 次退避、恰 1 条闭集遥测行（计数，零字幕内容）。
        self.assertEqual(slept, [1.0])
        self.assertEqual(
            [line for line in telemetry
             if line.startswith("stage=term-proofread-window-retry ")],
            ["stage=term-proofread-window-retry attempt=1"],
        )
        # 汇总行照常携带窗口级重试计数（闭集计数，零字幕内容）。
        self.assertTrue(
            any("window_retries=1" in line for line in telemetry),
            repr(telemetry),
        )

    def test_exhaustion_raises_last_transport_error_with_single_backoff(self):
        first, second = LLMError("AI request failed: one"), LLMError(
            "AI request failed: two",
        )
        slept = []

        def fake_chat(api_key, messages, **kwargs):
            raise LLMError("unreachable")

        with patch("courselens_worker.llm._chat", side_effect=fake_chat), \
                patch("courselens_worker.llm.time.sleep", side_effect=slept.append):
            with self.assertRaises(LLMError) as caught:
                term_proofread_segments(
                    "secret", [seg(0, 1000, "就能得到我的这样一个能耐图")],
                    terms=(),
                )
        # attempts=3 全败：睡眠恰 2 次（末次失败后不再睡），上抛末次载体。
        self.assertEqual(slept, [1.0, 1.0])
        self.assertIn("unreachable", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
