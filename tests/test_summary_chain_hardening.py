"""SUMMARY-FIX-1 总结链加固钉：思考档/max_tokens 帽/抢救重试/降级计数/usage。

FINALWRAP-C2 根因（2026-09-30 真实讲次两连败）：窗口/合并调用裸奔——无
thinking=提供商默认 enabled/high、窗口帽默认 8192、合并帽 12000，思考推理
计入 max_tokens 顶穿帽 → 合法 JSON 但 markdown 空串 → completed+空笔记。
本文件逐钉：缺省关思考、env 闭集、调用参数透传、窗口重试+降级跳过、合并
空 markdown 重试、截断抢救、fail-closed、usage 深账。
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from courselens_worker.llm import (
    _SUMMARY_MERGE_MAX_TOKENS,
    _SUMMARY_WINDOW_MAX_TOKENS,
    _SUMMARY_WINDOW_PROMPT,
    _SUMMARY_MERGE_PROMPT,
    REVIEW_VIEWS_ENV,
    SUMMARY_THINKING,
    SUMMARY_THINKING_ENV,
    _resolve_summary_thinking,
    _salvage_summary_object,
    _valid_summary_part,
    create_summary,
)


def _transcript(count: int) -> list[dict]:
    return [
        {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"text-{index}"}
        for index in range(count)
    ]


def _part(markdown: str) -> dict:
    return {"markdown": markdown, "chapters": []}


class ThinkingTierTests(unittest.TestCase):
    """SUMMARY-FIX-1 件1：缺省 disabled + env 闭集（与 term 族同式）。"""

    def test_module_default_is_disabled(self):
        self.assertEqual(SUMMARY_THINKING, {"type": "disabled"})

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
            with patch.dict(os.environ, {SUMMARY_THINKING_ENV: raw}, clear=False):
                self.assertEqual(_resolve_summary_thinking(), expected, raw)

    def test_calls_carry_thinking_and_caps(self):
        captured: list[dict] = []

        def fake_chat(api_key, messages, **kwargs):
            system = messages[0]["content"]
            captured.append({
                "merge": system == _SUMMARY_MERGE_PROMPT,
                "thinking": kwargs.get("thinking"),
                "max_tokens": kwargs.get("max_tokens"),
            })
            if system == _SUMMARY_MERGE_PROMPT:
                return json.dumps({"markdown": "combined", "chapters": []})
            return json.dumps(_part("part"))

        with patch.dict(os.environ, {SUMMARY_THINKING_ENV: "", REVIEW_VIEWS_ENV: "off"}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=fake_chat):
                create_summary("k", title="t", transcript=_transcript(130), ppt_pages=[])
        self.assertEqual(len(captured), 3)
        for call in captured:
            self.assertEqual(call["thinking"], {"type": "disabled"})
        for window_call in (c for c in captured if not c["merge"]):
            self.assertEqual(window_call["max_tokens"], _SUMMARY_WINDOW_MAX_TOKENS)
            self.assertEqual(window_call["max_tokens"], 8192)
        merge_calls = [c for c in captured if c["merge"]]
        self.assertEqual(len(merge_calls), 1)
        self.assertEqual(merge_calls[0]["max_tokens"], _SUMMARY_MERGE_MAX_TOKENS)
        self.assertEqual(_SUMMARY_MERGE_MAX_TOKENS, 32_768)

    def test_thinking_tier_raises_window_cap(self):
        captured: list[int] = []

        def fake_chat(api_key, messages, **kwargs):
            if messages[0]["content"] != _SUMMARY_MERGE_PROMPT:
                captured.append(int(kwargs.get("max_tokens") or 0))
                return json.dumps(_part("part"))
            return json.dumps({"markdown": "combined", "chapters": []})

        with patch.dict(os.environ, {SUMMARY_THINKING_ENV: "high", REVIEW_VIEWS_ENV: "off"}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=fake_chat):
                create_summary("k", title="t", transcript=_transcript(10), ppt_pages=[])
        self.assertTrue(all(cap == 16384 for cap in captured), captured)


class SalvageAndShapeTests(unittest.TestCase):
    """SUMMARY-FIX-1 件3：对象抢救两级 + 空 markdown 按失败处理。"""

    def test_brace_slice_recovers_chatty_object(self):
        value = _salvage_summary_object('前言 {"markdown": "# 笔记", "chapters": []} 后记')
        self.assertEqual(value["markdown"], "# 笔记")

    def test_markdown_key_recovers_truncated_object(self):
        raw = '好的，这是笔记：{"markdown": "# 截断笔记\\n正文…", "chapters": [{"title"'
        value = _salvage_summary_object(raw)
        self.assertEqual(value["markdown"], "# 截断笔记\n正文…")
        self.assertEqual(value["chapters"], [])

    def test_fence_wrapped_object_is_recovered(self):
        raw = "```json\n{\"markdown\": \"围栏笔记\", \"chapters\": []}\n```"
        value = _salvage_summary_object(raw)
        self.assertEqual(value["markdown"], "围栏笔记")

    def test_salvage_fails_closed_without_object(self):
        self.assertIsNone(_salvage_summary_object("oops not json"))
        self.assertIsNone(_salvage_summary_object('{"chapters": ["cut'))
        # 完整对象（含空 markdown）由 brace-slice 正常返回，空串判决权在
        # _valid_summary_part（见 test_empty_markdown_is_invalid_part）。

    def test_empty_markdown_is_invalid_part(self):
        # C2 实锤形态：合法 JSON + markdown 空串必须按失败处理。
        self.assertFalse(_valid_summary_part({"markdown": "  ", "chapters": []}))
        self.assertFalse(_valid_summary_part({"markdown": "", "chapters": []}))
        self.assertTrue(_valid_summary_part({"markdown": "m", "chapters": []}))
        self.assertFalse(_valid_summary_part({"chapters": []}))
        self.assertFalse(_valid_summary_part(None))


class WindowRetryTests(unittest.TestCase):
    """SUMMARY-FIX-1 件3：窗口重试 1 次 → 痊愈采用 / 仍败降级跳过并计数。"""

    @staticmethod
    def _responder(window_a_responses, window_b_response, merge_response, calls):
        def fake_chat(api_key, messages, **kwargs):
            system = messages[0]["content"]
            user = messages[1]["content"]
            if system == _SUMMARY_MERGE_PROMPT:
                calls.append("merge")
                return merge_response
            calls.append("a" if "text-0\"" in user[:400] else "b")
            key = calls[-1]
            if key == "a" and window_a_responses:
                return window_a_responses.pop(0)
            if key == "a":
                return json.dumps(_part("part-a-final"))
            return window_b_response

        return fake_chat

    def test_bad_shape_then_good_is_used_without_skip(self):
        calls: list[str] = []
        bad = json.dumps({"markdown": "part-a", "chapters": "not-a-list"})
        fake = self._responder(
            [bad], json.dumps(_part("part-b")),
            json.dumps({"markdown": "combined", "chapters": []}), calls,
        )
        with patch.dict(os.environ, {SUMMARY_THINKING_ENV: ""}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=fake):
                result = create_summary("k", title="t", transcript=_transcript(130), ppt_pages=[])
        self.assertEqual(calls.count("a"), 2, calls)
        self.assertEqual(result["deep_usage"]["window_retries"], 1)
        self.assertEqual(result["deep_usage"]["window_skipped"], 0)
        self.assertEqual(result["markdown"], "combined")

    def test_exhausted_window_retries_degrade_to_skip(self):
        calls: list[str] = []
        bad = json.dumps({"markdown": "", "chapters": []})  # C2 空串形态
        fake = self._responder(
            [bad, bad], json.dumps(_part("part-b")),
            json.dumps({"markdown": "combined", "chapters": []}), calls,
        )
        lines: list[str] = []
        with (
            patch.dict(os.environ, {SUMMARY_THINKING_ENV: "", REVIEW_VIEWS_ENV: "off"}, clear=False),
            patch("courselens_worker.llm._chat", side_effect=fake),
            patch("courselens_worker.llm._emit_telemetry", side_effect=lines.append),
        ):
            result = create_summary("k", title="t", transcript=_transcript(130), ppt_pages=[])
        self.assertEqual(calls.count("a"), 2, calls)
        self.assertEqual(calls.count("b"), 1, calls)
        self.assertEqual(result["deep_usage"]["window_skipped"], 1)
        self.assertEqual(result["deep_usage"]["window_retries"], 1)
        self.assertEqual(result["markdown"], "combined")
        self.assertTrue(any(line.startswith("stage=summary-window-skip") for line in lines))
        final = [line for line in lines if line.startswith("stage=summary ")][0]
        self.assertIn("window_skipped=1", final)

    def test_all_windows_skipped_fails_closed(self):
        def fake_chat(api_key, messages, **kwargs):
            if messages[0]["content"] == _SUMMARY_MERGE_PROMPT:
                return json.dumps({"markdown": "combined", "chapters": []})
            return json.dumps({"markdown": "", "chapters": []})

        with patch.dict(os.environ, {SUMMARY_THINKING_ENV: ""}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=fake_chat) as chat:
                with self.assertRaises(Exception) as raised:
                    create_summary("k", title="t", transcript=_transcript(130), ppt_pages=[])
        self.assertIn("windows all failed", str(raised.exception))
        # 两窗 × 两试=4 次调用，合并调用绝不发生（fail-closed 优于 completed 空）
        self.assertEqual(chat.call_count, 4)


class MergeRetryTests(unittest.TestCase):
    """SUMMARY-FIX-1 件3：合并空 markdown/坏 JSON → 抢救 → 重试 1 次 → fail-closed。"""

    @staticmethod
    def _run(merge_responses):
        calls = {"n": 0}

        def fake_chat(api_key, messages, **kwargs):
            if messages[0]["content"] != _SUMMARY_MERGE_PROMPT:
                return json.dumps(_part(f"part-{calls['n']}"))
            response = merge_responses[min(calls["n"], len(merge_responses) - 1)]
            calls["n"] += 1
            return response

        with patch.dict(os.environ, {SUMMARY_THINKING_ENV: "", REVIEW_VIEWS_ENV: "off"}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=fake_chat) as chat:
                result = create_summary("k", title="t", transcript=_transcript(10), ppt_pages=[])
        return result, chat

    def test_empty_markdown_retries_once_then_succeeds(self):
        result, chat = self._run([
            json.dumps({"markdown": "", "chapters": []}),
            json.dumps({"markdown": "combined", "chapters": []}),
        ])
        self.assertEqual(chat.call_count, 3)  # 1 窗口 + 2 合并
        self.assertEqual(result["markdown"], "combined")
        self.assertEqual(result["deep_usage"]["merge_retries"], 1)

    def test_truncated_merge_is_salvaged_without_retry(self):
        result, chat = self._run(['{"markdown": "# 合并笔记", "chapters": [{"title"'])
        self.assertEqual(chat.call_count, 2)  # 抢救层兜住，不再重试
        self.assertEqual(result["markdown"], "# 合并笔记")
        self.assertEqual(result["deep_usage"]["merge_retries"], 0)

    def test_exhausted_merge_retries_fail_closed(self):
        with patch.dict(os.environ, {SUMMARY_THINKING_ENV: ""}, clear=False):
            with patch(
                "courselens_worker.llm._chat",
                side_effect=[json.dumps(_part("p")), "", ""],
            ) as chat:
                with self.assertRaises(Exception) as raised:
                    create_summary("k", title="t", transcript=_transcript(10), ppt_pages=[])
        self.assertIn("merge response has an invalid shape", str(raised.exception))
        self.assertEqual(chat.call_count, 3)


class UsageSinkTests(unittest.TestCase):
    """SUMMARY-FIX-1 件4：总结链 usage 深账（completion/reasoning 拆分+重试计数）。"""

    def test_usage_sink_records_every_paid_call(self):
        def fake_chat(api_key, messages, **kwargs):
            from courselens_worker import llm as llm_mod

            llm_mod._CALL_LOG.append({
                "prompt_tokens": 100, "completion_tokens": 50,
                "reasoning_tokens": 30, "prompt_cache_hit_tokens": 64,
                "latency_ms": 900, "thinking": None,
            })
            if messages[0]["content"] == _SUMMARY_MERGE_PROMPT:
                return json.dumps({"markdown": "combined", "chapters": []})
            return json.dumps(_part("part"))

        sink: list[dict] = []
        with patch.dict(os.environ, {SUMMARY_THINKING_ENV: "", REVIEW_VIEWS_ENV: "off"}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=fake_chat):
                result = create_summary(
                    "k", title="t", transcript=_transcript(10), ppt_pages=[],
                    usage_sink=sink,
                )
        self.assertEqual(len(sink), 2)
        deep = result["deep_usage"]
        self.assertEqual(deep["calls"], 2)
        self.assertEqual(deep["prompt_tokens"], 200)
        self.assertEqual(deep["completion_tokens"], 100)
        self.assertEqual(deep["reasoning_tokens"], 60)
        self.assertEqual(deep["prompt_cache_hit_tokens"], 128)
        self.assertEqual(deep["latency_seconds"], 1.8)
        self.assertEqual(deep["thinking"], "disabled")
        self.assertEqual(deep["window_skipped"], 0)

    def test_final_telemetry_carries_hardening_counters(self):
        lines: list[str] = []

        def fake_chat(api_key, messages, **kwargs):
            from courselens_worker import llm as llm_mod

            llm_mod._CALL_LOG.append({
                "prompt_tokens": 10, "completion_tokens": 5,
                "reasoning_tokens": 0, "prompt_cache_hit_tokens": 0,
                "latency_ms": 100, "thinking": None,
            })
            if messages[0]["content"] == _SUMMARY_MERGE_PROMPT:
                return json.dumps({"markdown": "m", "chapters": []})
            return json.dumps(_part("part"))

        with (
            patch.dict(os.environ, {SUMMARY_THINKING_ENV: "", REVIEW_VIEWS_ENV: "off"}, clear=False),
            patch("courselens_worker.llm._chat", side_effect=fake_chat),
            patch("courselens_worker.llm._emit_telemetry", side_effect=lines.append),
        ):
            create_summary("k", title="t", transcript=_transcript(10), ppt_pages=[])
        final = [line for line in lines if line.startswith("stage=summary ")]
        self.assertEqual(len(final), 1, lines)
        for token in ("thinking=disabled", "window_retries=0", "window_skipped=0",
                      "merge_retries=0", "llm_calls=2", "completion_tokens=10"):
            self.assertIn(token, final[0])


if __name__ == "__main__":
    unittest.main()
