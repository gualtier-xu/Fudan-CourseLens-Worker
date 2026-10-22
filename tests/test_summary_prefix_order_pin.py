# -*- coding: utf-8 -*-
"""L2 稳定前缀顺序钉（DEEPCOST-FIX-1）。

context caching 只按前缀全匹配命中（官方 KV-cache 指南；命中价=未命中
1/50），summary 窗载荷若把逐窗易变的 transcript 排在逐窗恒同的 glossary
之前，每窗都在 50 倍价差下全款重付稳定前缀。本钉以行为级守卫（替身
``_chat`` 捕获真实 messages）锁死两件事：

1. summary 窗用户载荷键序：``glossary``（稳定）在 ``transcript``（易变）
   之前；
2. merge 载荷键序：稳定标量（course_context/glossary）在 ``parts``（易变
   大头）之前，parts 恒为末键。

只锁「稳定前缀在易变段之前」的顺序语义，不锁其余键的相互顺序（允许未来
无语义重排）；载荷键集合/取值语义由既有形状门测试守护。
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from test_review_views import (
    _MERGE,
    _WINDOW,
    _TRANSCRIPT,
    _VIEWS,
    _enabled_env,
    _stage_responder,
)

from courselens_worker.llm import create_summary


def _capture_calls():
    captured: list[list[dict[str, str]]] = []

    def responder(api_key, messages, **kwargs):
        from courselens_worker import llm as llm_module

        captured.append(messages)
        with llm_module._USAGE_LOCK:
            llm_module._CALL_LOG.append({
                "prompt_tokens": 10, "completion_tokens": 5,
                "reasoning_tokens": 0, "prompt_cache_hit_tokens": 0,
                "latency_ms": 10, "thinking": None,
            })
        system = messages[0]["content"]
        user = json.loads(messages[1]["content"])
        if "parts" in user:
            return json.dumps(_MERGE, ensure_ascii=False)
        if "study_guide" in system or "views" in system:
            return json.dumps(_VIEWS, ensure_ascii=False)
        return json.dumps(_WINDOW, ensure_ascii=False)

    return captured, responder


class SummaryPrefixOrderPinTests(unittest.TestCase):
    def test_window_glossary_precedes_transcript(self) -> None:
        captured, responder = _capture_calls()
        with patch("courselens_worker.llm._chat", side_effect=responder), \
                patch("courselens_worker.llm._emit_telemetry", side_effect=lambda line: None), \
                _enabled_env():
            create_summary(
                "k", title="t",
                transcript=_TRANSCRIPT,
                ppt_pages=[],
                glossary=("肌动蛋白", "线粒体"),
            )
        window_payloads = [
            json.loads(messages[1]["content"])
            for messages in captured
            if "transcript" in json.loads(messages[1]["content"])
        ]
        self.assertGreaterEqual(len(window_payloads), 1, "summary window calls must be captured")
        for payload in window_payloads:
            self.assertIn("glossary", payload, "glossary-bearing window must carry glossary")
            keys = list(payload.keys())
            self.assertLess(
                keys.index("glossary"), keys.index("transcript"),
                f"window payload key order {keys}: glossary must precede transcript"
                "（L2 钉：稳定前缀前置吃 context caching，命中价=1/50）",
            )

    def test_merge_volatile_parts_is_last_key(self) -> None:
        captured, responder = _capture_calls()
        with patch("courselens_worker.llm._chat", side_effect=responder), \
                patch("courselens_worker.llm._emit_telemetry", side_effect=lambda line: None), \
                _enabled_env():
            create_summary(
                "k", title="t",
                transcript=_TRANSCRIPT,
                ppt_pages=[],
                glossary=("肌动蛋白",),
                course_context={"course_title": "生物实验"},
            )
        merge_payloads = [
            json.loads(messages[1]["content"])
            for messages in captured
            if "parts" in json.loads(messages[1]["content"])
        ]
        self.assertGreaterEqual(len(merge_payloads), 1, "merge call must be captured")
        for payload in merge_payloads:
            keys = list(payload.keys())
            self.assertEqual(
                keys[-1], "parts",
                f"merge payload key order {keys}: volatile parts must be the last key"
                "（L2 钉：稳定标量前置，跨讲前缀可命中缓存）",
            )
            self.assertLess(
                keys.index("course_context"), keys.index("parts"),
                "stable course_context must precede parts",
            )
            self.assertLess(
                keys.index("glossary"), keys.index("parts"),
                "stable glossary must precede parts",
            )


if __name__ == "__main__":
    unittest.main()
