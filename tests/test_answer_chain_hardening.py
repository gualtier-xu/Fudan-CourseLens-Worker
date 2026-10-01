"""RR-P5HARD-1 answer_question 本体加固钉：思考档/抢救/重试/空串/fail-closed。

SUMMARY-FIX-1 家族同式（test_summary_chain_hardening 镜像）：缺省关思考、
env 闭集（用例内 patch.dict 显式置值，与 CI/GITHUB_ENV 宿主环境隔离）、
思考档帽放大、chatty/截断两级抢救、空 answer 串按失败、瞬态重试恰一次、
429/授权类绝不入重试集（6fbd077 家规）、穷尽 fail-closed、零提示词膨胀
（sha256 冻结钉）。
"""

from __future__ import annotations

import hashlib
import json
import os
import unittest
from unittest.mock import patch

from courselens_worker.llm import (
    LLMError,
    QUESTION_THINKING,
    QUESTION_THINKING_ENV,
    _QUESTION_MAX_TOKENS,
    _QUESTION_THINKING_MAX_TOKENS,
    _resolve_question_thinking,
    _salvage_answer_object,
    _valid_answer_object,
    answer_question,
)

_EVIDENCE = [{"citation_id": "r1", "text": "证据"}]
_GOOD = json.dumps(
    {"answer": "回答", "grounded": True, "citations": ["r1"]}, ensure_ascii=False
)


class ThinkingTierTests(unittest.TestCase):
    """件1：缺省 disabled + env 闭集（与 SUMMARY_THINKING 同式）。"""

    def test_module_default_is_disabled(self):
        self.assertEqual(QUESTION_THINKING, {"type": "disabled"})

    def test_resolver_closed_set(self):
        cases = {
            "": {"type": "disabled"},
            "default": None,
            "provider-default": None,
            "disabled": {"type": "disabled"},
            "low": {"type": "enabled", "reasoning_effort": "low"},
            "high": {"type": "enabled", "reasoning_effort": "high"},
            "max": {"type": "enabled", "reasoning_effort": "max"},
            "bogus": {"type": "disabled"},
        }
        for raw, expected in cases.items():
            with patch.dict(os.environ, {QUESTION_THINKING_ENV: raw}, clear=False):
                self.assertEqual(_resolve_question_thinking(), expected, raw)

    def test_default_call_carries_disabled_thinking_and_8192_cap(self):
        captured: dict = {}

        def fake_chat(api_key, messages, **kwargs):
            captured["thinking"] = kwargs.get("thinking")
            captured["max_tokens"] = kwargs.get("max_tokens")
            return _GOOD

        with patch.dict(os.environ, {QUESTION_THINKING_ENV: ""}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=fake_chat):
                answer_question("k", query="q", evidence=list(_EVIDENCE))
        self.assertEqual(captured["thinking"], {"type": "disabled"})
        self.assertEqual(captured["max_tokens"], _QUESTION_MAX_TOKENS)
        self.assertEqual(captured["max_tokens"], 8192)

    def test_thinking_env_high_raises_cap_and_effort(self):
        captured: dict = {}

        def fake_chat(api_key, messages, **kwargs):
            captured["thinking"] = kwargs.get("thinking")
            captured["max_tokens"] = kwargs.get("max_tokens")
            return _GOOD

        with patch.dict(os.environ, {QUESTION_THINKING_ENV: "high"}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=fake_chat):
                answer_question("k", query="q", evidence=list(_EVIDENCE))
        self.assertEqual(
            captured["thinking"], {"type": "enabled", "reasoning_effort": "high"}
        )
        self.assertEqual(captured["max_tokens"], _QUESTION_THINKING_MAX_TOKENS)
        self.assertEqual(captured["max_tokens"], 16384, "推理计入 max_tokens，帽须放大")

    def test_provider_default_tier_also_gets_thinking_cap(self):
        captured: dict = {}

        def fake_chat(api_key, messages, **kwargs):
            captured["max_tokens"] = kwargs.get("max_tokens")
            return _GOOD

        with patch.dict(os.environ, {QUESTION_THINKING_ENV: "default"}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=fake_chat):
                answer_question("k", query="q", evidence=list(_EVIDENCE))
        self.assertEqual(captured["max_tokens"], _QUESTION_THINKING_MAX_TOKENS)


class SalvageTests(unittest.TestCase):
    """件2：对象抢救两级 + 空串按失败。"""

    def test_brace_slice_recovers_chatty_object(self):
        value = _salvage_answer_object(
            '前言 {"answer": "回答", "grounded": true, "citations": ["r1"]} 后记'
        )
        self.assertEqual(value["answer"], "回答")

    def test_answer_key_recovers_truncated_object(self):
        raw = '好的，这是回答：{"answer": "被截断的回答", "grounded": true, "ci'
        value = _salvage_answer_object(raw)
        self.assertEqual(value["answer"], "被截断的回答")
        # 宁缺勿假：截断丢失的 citations 绝不凭空补，grounding 门自然拒绝。
        self.assertEqual(value["citations"], [])
        self.assertFalse(value["grounded"])

    def test_fence_wrapped_object_is_recovered(self):
        raw = '```json\n{"answer": "围栏回答", "grounded": true, "citations": ["r1"]}\n```'
        value = _salvage_answer_object(raw)
        self.assertEqual(value["answer"], "围栏回答")

    def test_garbage_returns_none(self):
        self.assertIsNone(_salvage_answer_object(""))
        self.assertIsNone(_salvage_answer_object("完全不是 JSON"))
        self.assertIsNone(_salvage_answer_object('{"answer": "截断即非'))

    def test_valid_gate_rejects_empty_answer_and_non_objects(self):
        self.assertFalse(_valid_answer_object(None))
        self.assertFalse(_valid_answer_object(["不是对象"]))
        self.assertFalse(_valid_answer_object({"answer": "  ", "citations": []}))
        self.assertFalse(_valid_answer_object({"answer": "有正文"}), "citations 缺席非法")
        self.assertTrue(
            _valid_answer_object({"answer": "有正文", "grounded": True, "citations": []})
        )


class RetryAndFailClosedTests(unittest.TestCase):
    """件3/件4：瞬态重试恰一次；429/授权类绝不入重试集；穷尽 fail-closed。"""

    def test_transient_bad_shape_retries_once_then_recovers(self):
        with patch.dict(os.environ, {QUESTION_THINKING_ENV: ""}, clear=False):
            with patch(
                "courselens_worker.llm._chat", side_effect=["垃圾响应", _GOOD]
            ) as chat:
                with patch("courselens_worker.llm.time.sleep"):
                    value = answer_question("k", query="q", evidence=list(_EVIDENCE))
        self.assertEqual(chat.call_count, 2, "重试恰一次")
        self.assertTrue(value["grounded"])
        self.assertEqual(value["answer"], "回答")

    def test_empty_answer_string_is_transient_failure(self):
        """C2 实锤同形：合法 JSON 但 answer 空串必须走抢救/重试，绝不直穿。"""
        empty = json.dumps({"answer": "", "grounded": True, "citations": ["r1"]})
        with patch.dict(os.environ, {QUESTION_THINKING_ENV: ""}, clear=False):
            with patch(
                "courselens_worker.llm._chat", side_effect=[empty, _GOOD]
            ) as chat:
                with patch("courselens_worker.llm.time.sleep"):
                    value = answer_question("k", query="q", evidence=list(_EVIDENCE))
        self.assertEqual(chat.call_count, 2)
        self.assertTrue(value["grounded"])

    def test_exhaustion_fails_closed_after_exactly_one_retry(self):
        with patch.dict(os.environ, {QUESTION_THINKING_ENV: ""}, clear=False):
            with patch(
                "courselens_worker.llm._chat", side_effect=["", ""]
            ) as chat:
                with patch("courselens_worker.llm.time.sleep"):
                    with self.assertRaises(LLMError):
                        answer_question("k", query="q", evidence=list(_EVIDENCE))
        self.assertEqual(chat.call_count, 2, "绝不连环重试")

    def test_429_never_enters_retry_set(self):
        with patch.dict(os.environ, {QUESTION_THINKING_ENV: ""}, clear=False):
            with patch(
                "courselens_worker.llm._chat",
                side_effect=[LLMError("AI request returned HTTP 429")],
            ) as chat:
                with self.assertRaises(LLMError):
                    answer_question("k", query="q", evidence=list(_EVIDENCE))
        self.assertEqual(chat.call_count, 1, "限流 hammer 被家规禁止")

    def test_auth_statuses_never_enter_retry_set(self):
        for status in (401, 403):
            with patch.dict(os.environ, {QUESTION_THINKING_ENV: ""}, clear=False):
                with patch(
                    "courselens_worker.llm._chat",
                    side_effect=[LLMError(f"AI request returned HTTP {status}")],
                ) as chat:
                    with self.assertRaises(LLMError):
                        answer_question("k", query="q", evidence=list(_EVIDENCE))
            self.assertEqual(chat.call_count, 1, f"HTTP {status} 重试无意义")

    def test_transport_transient_retried_once(self):
        with patch.dict(os.environ, {QUESTION_THINKING_ENV: ""}, clear=False):
            with patch(
                "courselens_worker.llm._chat",
                side_effect=[
                    LLMError("AI request failed: ConnectionError"),
                    _GOOD,
                ],
            ) as chat:
                with patch("courselens_worker.llm.time.sleep"):
                    value = answer_question("k", query="q", evidence=list(_EVIDENCE))
        self.assertEqual(chat.call_count, 2)
        self.assertTrue(value["grounded"])

    def test_truncated_response_rescues_to_honest_decline(self):
        """截断抢救成功（answer 正文可回收）→ 宁缺勿假按未落地拒绝，不重试不抛。"""
        raw = '{"answer": "被截断的回答", "grounded": true, "ci'
        with patch.dict(os.environ, {QUESTION_THINKING_ENV: ""}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=[raw]) as chat:
                value = answer_question("k", query="q", evidence=list(_EVIDENCE))
        self.assertEqual(chat.call_count, 1, "抢救成功即收口，不烧第二次调用")
        self.assertEqual(
            value,
            {"answer": "资料不足，无法根据当前课程资料回答。", "citations": [], "grounded": False},
        )

    def test_model_decline_contract_unchanged(self):
        declined = json.dumps(
            {"answer": "资料不足，无法根据当前课程资料回答。", "grounded": False, "citations": []},
            ensure_ascii=False,
        )
        with patch.dict(os.environ, {QUESTION_THINKING_ENV: ""}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=[declined]):
                value = answer_question("k", query="q", evidence=list(_EVIDENCE))
        self.assertEqual(
            value,
            {"answer": "资料不足，无法根据当前课程资料回答。", "citations": [], "grounded": False},
        )

    def test_retry_telemetry_is_closed_set(self):
        lines: list[str] = []
        with patch.dict(os.environ, {QUESTION_THINKING_ENV: ""}, clear=False):
            with patch(
                "courselens_worker.llm._chat", side_effect=["垃圾响应", _GOOD]
            ):
                with patch("courselens_worker.llm.time.sleep"):
                    with patch(
                        "courselens_worker.llm._emit_telemetry", side_effect=lines.append
                    ):
                        answer_question("k", query="q", evidence=list(_EVIDENCE))
        self.assertEqual(lines, ["stage=answer-retry attempt=1"], "闭集计数，零内容")


class PromptFreezeTests(unittest.TestCase):
    """件5：零提示词膨胀——system 串 sha256 冻结钉（改提示词必须有意识地动本钉）。"""

    def test_system_prompt_frozen(self):
        with patch.dict(os.environ, {QUESTION_THINKING_ENV: ""}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=[_GOOD]) as chat:
                answer_question("k", query="q", evidence=list(_EVIDENCE))
        system = chat.call_args.args[1][0]["content"]
        self.assertEqual(len(system), 242)
        self.assertEqual(
            hashlib.sha256(system.encode("utf-8")).hexdigest(),
            "b3d4d269f6bfc19e9c0e7ce6fac0d1d16e9d543ff3b507c72f71cce3dcea28af",
        )


class CourseTermsChannelTests(unittest.TestCase):
    """RR-P6MEM-1：课程记忆术语表数据通道（加性，缺席=旧行为逐字）。

    system 提示词零膨胀（PromptFreezeTests 的冻结钉口径在带 terms 调用下
    同样必须原样通过）；terms 只随 user 消息 JSON 出现；无 terms 时 user
    载荷与历史两键形状逐字相同。
    """

    def test_legacy_payload_has_exactly_two_keys(self):
        with patch.dict(os.environ, {QUESTION_THINKING_ENV: ""}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=[_GOOD]) as chat:
                answer_question("k", query="q", evidence=list(_EVIDENCE))
        user = chat.call_args.args[1][1]["content"]
        self.assertEqual(
            json.loads(user), {"query": "q", "evidence": [{"citation_id": "r1", "text": "证据"}]}
        )

    def test_empty_terms_also_keep_legacy_shape(self):
        with patch.dict(os.environ, {QUESTION_THINKING_ENV: ""}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=[_GOOD]) as chat:
                answer_question("k", query="q", evidence=list(_EVIDENCE), course_terms=())
        self.assertNotIn("course_terms", json.loads(chat.call_args.args[1][1]["content"]))

    def test_course_terms_ride_user_payload_only(self):
        with patch.dict(os.environ, {QUESTION_THINKING_ENV: ""}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=[_GOOD]) as chat:
                answer_question(
                    "k", query="q", evidence=list(_EVIDENCE),
                    course_terms=("费米能级", " 量子力学 ", ""),
                )
        messages = chat.call_args.args[1]
        # system 零膨胀：与无 terms 调用逐字相同（冻结钉同值）
        self.assertEqual(len(messages[0]["content"]), 242)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["course_terms"], ["费米能级", "量子力学"])
        self.assertEqual(payload["query"], "q")
        self.assertEqual(payload["evidence"], [{"citation_id": "r1", "text": "证据"}])


if __name__ == "__main__":
    unittest.main()
