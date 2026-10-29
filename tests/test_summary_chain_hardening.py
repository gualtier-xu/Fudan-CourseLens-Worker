"""SUMMARY-FIX-1 总结链加固钉：思考档/max_tokens 帽/抢救重试/降级计数/usage。

FINALWRAP-C2 根因（2026-09-30 真实讲次两连败）：窗口/合并调用裸奔——无
thinking=提供商默认 enabled/high、窗口帽默认 8192、合并帽 12000，思考推理
计入 max_tokens 顶穿帽 → 合法 JSON 但 markdown 空串 → completed+空笔记。
本文件逐钉：缺省关思考、env 闭集、调用参数透传、窗口重试+降级跳过、合并
空 markdown 重试、截断抢救、fail-closed、usage 深账。
"""

from __future__ import annotations

import contextlib
import json
import os
import unittest
from unittest.mock import patch

from courselens_worker import llm
from courselens_worker.llm import (
    _SUMMARY_MERGE_MAX_TOKENS,
    _SUMMARY_WINDOW_MAX_TOKENS,
    _SUMMARY_WINDOW_PROMPT,
    _SUMMARY_MERGE_PROMPT_TEACHING,
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
                "merge": system == _SUMMARY_MERGE_PROMPT_TEACHING,
                "thinking": kwargs.get("thinking"),
                "max_tokens": kwargs.get("max_tokens"),
            })
            if system == _SUMMARY_MERGE_PROMPT_TEACHING:
                # RXFIX-2：合并替身带要点面（过质量门=零质量重试，本钉锁调用参数）。
                return json.dumps({"markdown": "combined", "chapters": [], "key_takeaways": ["要点"]})
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
            if messages[0]["content"] != _SUMMARY_MERGE_PROMPT_TEACHING:
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
            if system == _SUMMARY_MERGE_PROMPT_TEACHING:
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
            json.dumps({"markdown": "combined", "chapters": [], "key_takeaways": ["要点"]}), calls,
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
            if messages[0]["content"] == _SUMMARY_MERGE_PROMPT_TEACHING:
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
            if messages[0]["content"] != _SUMMARY_MERGE_PROMPT_TEACHING:
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
            json.dumps({"markdown": "combined", "chapters": [], "key_takeaways": ["要点"]}),
        ])
        self.assertEqual(chat.call_count, 3)  # 1 窗口 + 2 合并
        self.assertEqual(result["markdown"], "combined")
        self.assertEqual(result["deep_usage"]["merge_retries"], 1)
        self.assertEqual(result["deep_usage"]["takeaways_missing"], 0)

    def test_truncated_merge_is_salvaged_without_retry(self):
        # RXFIX-2 后语义：抢救层兜住截断（不烧坏形状重试），但抢救产物无
        # key_takeaways → 触发一次要点面质量重试 → 同响应重掷 → 降级出货。
        result, chat = self._run(['{"markdown": "# 合并笔记", "chapters": [{"title"'])
        self.assertEqual(chat.call_count, 3)  # 1 窗口 + 2 合并（第二次=质量重试）
        self.assertEqual(result["markdown"], "# 合并笔记")
        self.assertEqual(result["deep_usage"]["merge_retries"], 1)
        self.assertEqual(result["deep_usage"]["takeaways_missing"], 1)

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


class TakeawayQualityGateTests(unittest.TestCase):
    """RXFIX-2（2026-10-29）：合并「形状合法但要点面全空」→ 单点质量重试 →
    降级出货闭集计数。实测依据：REPROC-BANQI-2 要点面 8 掷 5 空（62.5%），
    旧逻辑静默出货=缺陷；同构建重掷有恢复概率（665055 二掷 0→5）。
    重试请求在 user 侧追加补产指令（system 串逐位不动=替身路由/300 钉不变）。"""

    @staticmethod
    def _run(merge_responses):
        merge_payloads: list[dict] = []

        def fake_chat(api_key, messages, **kwargs):
            if messages[0]["content"] != _SUMMARY_MERGE_PROMPT_TEACHING:
                return json.dumps(_part(f"part-{len(merge_payloads)}"))
            merge_payloads.append({
                "system": messages[0]["content"],
                "user": messages[1]["content"],
            })
            response = merge_responses[min(len(merge_payloads) - 1, len(merge_responses) - 1)]
            return response

        with patch.dict(os.environ, {SUMMARY_THINKING_ENV: "", REVIEW_VIEWS_ENV: "off"}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=fake_chat) as chat:
                result = create_summary("k", title="t", transcript=_transcript(10), ppt_pages=[])
        return result, chat, merge_payloads

    def test_missing_takeaways_retried_once_with_nudge_then_recovered(self):
        bad = json.dumps({"markdown": "combined", "chapters": []})
        good = json.dumps({
            "markdown": "combined", "chapters": [], "key_takeaways": ["要点一", "要点二"],
        })
        result, chat, payloads = self._run([bad, good])
        self.assertEqual(chat.call_count, 3)  # 1 窗口 + 2 合并
        self.assertEqual(result["key_takeaways"], ["要点一", "要点二"])
        self.assertEqual(result["deep_usage"]["merge_retries"], 1)
        self.assertEqual(result["deep_usage"]["takeaways_missing"], 0)
        # 重试请求 system 逐位不动、user 侧携带补产指令。
        self.assertEqual(payloads[1]["system"], payloads[0]["system"])
        self.assertTrue(payloads[1]["user"].startswith(payloads[0]["user"]))
        self.assertIn("key_takeaways", payloads[1]["user"][len(payloads[0]["user"]):])
        self.assertIn("doc.summary_notes", payloads[1]["user"])

    def test_doc_notes_alone_passes_the_gate(self):
        merge = json.dumps({
            "markdown": "combined", "chapters": [],
            "doc": {
                "overview": "总述。",
                "summary_notes": ["小结一", "小结二"],
                "sections": [{
                    "title": "开篇", "start_ms": 0, "source": "semantic",
                    "points": [{"text": "要点", "anchor_ms": 0}], "formulas": [],
                }],
            },
        })
        result, chat, payloads = self._run([merge])
        self.assertEqual(chat.call_count, 2)  # 门过=零质量重试
        self.assertEqual(result["deep_usage"]["merge_retries"], 0)
        self.assertEqual(result["doc"]["summary_notes"], ["小结一", "小结二"])

    def test_both_attempts_missing_ships_degraded_with_counter(self):
        bad = json.dumps({"markdown": "combined", "chapters": []})
        result, chat, payloads = self._run([bad])
        self.assertEqual(chat.call_count, 3)
        self.assertEqual(result["markdown"], "combined")  # 降级出货不整单失败
        self.assertEqual(result["key_takeaways"], [])
        self.assertEqual(result["deep_usage"]["takeaways_missing"], 1)
        # 两次合并请求：第二次 user 带补产指令（单点重试证据）。
        self.assertIn("doc.summary_notes", payloads[1]["user"])
        self.assertNotIn("（上一响应缺少", payloads[0]["user"])

    def test_summary_takeaways_missing_helper_is_shape_robust(self):
        from courselens_worker.llm import summary_takeaways_missing

        self.assertTrue(summary_takeaways_missing(None))
        self.assertTrue(summary_takeaways_missing({}))
        self.assertTrue(summary_takeaways_missing({"key_takeaways": [], "doc": {}}))
        self.assertFalse(summary_takeaways_missing({"key_takeaways": ["要点"]}))
        self.assertFalse(summary_takeaways_missing({"doc": {"summary_notes": ["小结"]}}))
        # 空白条目=缺（诚实空态）。
        self.assertTrue(summary_takeaways_missing({"key_takeaways": ["", "  "]}))


class TakeawayCarryoverTests(unittest.TestCase):
    """V010-D1-FIX-1（P-9 重试保真，2026-10-29）：体量重试把要点面打空时，
    末试确定性回填前次过门候选的要点面。

    实景依据=669766 掷2（run 6157cd65 遥测）：attempt0 要点过门但 md 7966 破
    地板 14692 → 仅 volume-retry；attempt1 md 38291 过门但要点 0（无
    quality-retry 遥测=要点面是重试回归不是首试失败）→ 旧逻辑降级出货
    takeaways_missing=1，学生拿到要点全空产物。r1 对照（同讲掷1）要点 6/5。
    钉四条：r2 实景回填 / notes 面回填 / r1 对照零动作 / 双空诚实降级。"""

    @staticmethod
    def _transcript(minutes: int) -> list[dict]:
        return [
            {"start_ms": index * 60_000, "end_ms": (index + 1) * 60_000,
             "text": f"第{index}分钟的内容。"}
            for index in range(minutes)
        ]

    @staticmethod
    def _thick(chars: int) -> str:
        line = "本讲机制推导要点覆盖全部章节时间轴，内容充实无留白。"
        return "".join(line[index % len(line)] for index in range(chars))

    @classmethod
    def _run(cls, merge_responses):
        payloads: list[dict] = []

        def fake_chat(api_key, messages, **kwargs):
            if messages[0]["content"] != _SUMMARY_MERGE_PROMPT_TEACHING:
                return json.dumps(_part("窗口笔记"))
            payloads.append({"system": messages[0]["content"], "user": messages[1]["content"]})
            return merge_responses[min(len(payloads) - 1, len(merge_responses) - 1)]

        with patch.dict(os.environ, {SUMMARY_THINKING_ENV: "", REVIEW_VIEWS_ENV: "off"}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=fake_chat) as chat:
                result = create_summary(
                    "k", title="t", transcript=cls._transcript(80), ppt_pages=[],
                )
        return result, chat, payloads

    def test_volume_retry_regression_restores_prior_takeaways(self):
        # r2 实景镜像：attempt0 要点过门+体量破门 → attempt1 体量过门+要点空。
        thin = json.dumps({
            "markdown": self._thick(800), "chapters": [], "key_takeaways": ["要点一", "要点二"],
        })
        thick_bare = json.dumps({"markdown": self._thick(6000), "chapters": []})
        result, chat, payloads = self._run([thin, thick_bare])
        self.assertEqual(chat.call_count, 3)  # 1 窗口 + 2 合并（第二次=体量重试）
        self.assertEqual(payloads[1]["system"], payloads[0]["system"])
        # 回填：要点面=前次过门事实，takeaways_missing 翻 0、salvaged 闭集留痕。
        self.assertEqual(result["key_takeaways"], ["要点一", "要点二"])
        self.assertEqual(result["deep_usage"]["takeaways_salvaged"], 1)
        self.assertEqual(result["deep_usage"]["takeaways_missing"], 0)
        self.assertEqual(result["deep_usage"]["md_volume_retries"], 1)
        self.assertEqual(result["deep_usage"]["md_volume_degraded"], 0)
        self.assertEqual(result["markdown"], self._thick(6000))

    def test_notes_face_regression_restores_prior_summary_notes(self):
        # notes 面镜像：attempt0 唯一非空面=doc.summary_notes；attempt1 保留
        # 合法 doc 但 notes 空 → 回填进 doc，coerce 透传后仍非空。
        valid_doc = {
            "overview": "总述。",
            "sections": [{
                "title": "开篇", "start_ms": 0, "source": "semantic",
                "points": [{"text": "要点", "anchor_ms": 0}], "formulas": [],
            }],
        }
        thin = json.dumps({
            "markdown": self._thick(800), "chapters": [],
            "doc": {**valid_doc, "summary_notes": ["小结一", "小结二"]},
        })
        thick_bare = json.dumps({
            "markdown": self._thick(6000), "chapters": [], "doc": dict(valid_doc),
        })
        result, chat, payloads = self._run([thin, thick_bare])
        self.assertEqual(chat.call_count, 3)
        self.assertEqual(result["doc"]["summary_notes"], ["小结一", "小结二"])
        self.assertEqual(result["deep_usage"]["takeaways_salvaged"], 1)
        self.assertEqual(result["deep_usage"]["takeaways_missing"], 0)

    def test_takeaways_ok_throw_keeps_zero_salvage(self):
        # r1 对照：单掷要点+体量双过门=零重试零回填（回填绝不无中生有）。
        good = json.dumps({
            "markdown": self._thick(6000), "chapters": [], "key_takeaways": ["要点一"],
        })
        result, chat, payloads = self._run([good])
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(result["key_takeaways"], ["要点一"])
        self.assertEqual(result["deep_usage"]["takeaways_salvaged"], 0)
        self.assertEqual(result["deep_usage"]["takeaways_missing"], 0)
        self.assertEqual(result["deep_usage"]["merge_retries"], 0)

    def test_both_empty_still_ships_honest_degraded(self):
        # 诚实边界：全程无过门面 → 无回填源 → 降级计数如实为 1、salvaged=0。
        bad = json.dumps({"markdown": self._thick(6000), "chapters": []})
        result, chat, payloads = self._run([bad])
        self.assertEqual(chat.call_count, 3)
        self.assertEqual(result["key_takeaways"], [])
        self.assertEqual(result["deep_usage"]["takeaways_missing"], 1)
        self.assertEqual(result["deep_usage"]["takeaways_salvaged"], 0)


class ChapterAnchorRescueTests(unittest.TestCase):
    """QSCOREFIX-1④（D-QS-9/12）：章节零锚本地拯救 + 重复标题去重。

    评分卡实测 65/65 全零锚——根因=首段 0ms 使 0 ∈ start_ms 白名单，LLM
    全零/缺省发射整批过门；重复章节 ≥3 对。钉四条：①后续章零锚走 Dice 派生
    （同 takeaway 锚源与阈值，fail-closed：派生不出留 0 绝不编锚）；②白名单
    外且派生失败维持既有丢项；③归一化标题同键保首现丢后现；④遥测计数。
    首章 0 ∈ 白名单时保守保留（0 可能是合法讲首锚，不可凭空改写）。"""

    @staticmethod
    def _run(merge_payload, transcript=None, lines=None):
        def fake_chat(api_key, messages, **kwargs):
            if messages[0]["content"] == _SUMMARY_MERGE_PROMPT_TEACHING:
                return json.dumps(merge_payload)
            return json.dumps(_part("part"))

        with patch.dict(os.environ, {SUMMARY_THINKING_ENV: "", REVIEW_VIEWS_ENV: "off"}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=fake_chat):
                with contextlib.nullcontext() if lines is None else patch(
                    "courselens_worker.llm._emit_telemetry", side_effect=lines.append
                ):
                    return create_summary(
                        "k", title="t",
                        transcript=_transcript(10) if transcript is None else transcript,
                        ppt_pages=[],
                    )

    def test_later_zero_anchors_are_rescued_by_text_similarity(self):
        lines: list[str] = []
        merge = {
            "markdown": "combined",
            "chapters": [
                {"title": "开场章", "summary": "起点", "start_ms": 0},
                {"title": "第三章", "summary": "text-3 讲透了一个概念", "start_ms": 0},
                {"title": "第五章", "summary": "text-5 的延伸", "start_ms": 0},
            ],
        }
        result = self._run(merge, lines=lines)
        self.assertEqual(
            [item["start_ms"] for item in result["chapters"]],
            [0, 3000, 5000],
            "首章 0 保守保留；后续章零锚由转写原文相似度派生真锚",
        )
        final = [line for line in lines if line.startswith("stage=summary ")][0]
        self.assertIn("chapters=3", final)
        self.assertIn("chapter_anchors_derived=2", final)
        self.assertIn("chapter_dup_dropped=0", final)

    def test_out_of_whitelist_without_derivation_is_dropped(self):
        transcript = [{"start_ms": 5_000, "end_ms": 6_000, "text": "text-5"}]
        merge = {
            "markdown": "combined",
            "chapters": [
                {"title": "有救章", "summary": "text-5 扩展", "start_ms": 999},
                {"title": "无救章", "summary": "风马牛不相及的叙述", "start_ms": 999},
            ],
        }
        result = self._run(merge, transcript=transcript)
        self.assertEqual(
            [item["title"] for item in result["chapters"]],
            ["有救章"],
            "白名单外派生成功→真锚；派生失败→维持 fail-closed 丢项",
        )
        self.assertEqual(result["chapters"][0]["start_ms"], 5_000)

    def test_duplicate_normalized_titles_keep_first(self):
        merge = {
            "markdown": "combined",
            "chapters": [
                {"title": "微积分的历史", "summary": "text-1", "start_ms": 1_000},
                {"title": "微积分 的 历史！", "summary": "text-2", "start_ms": 2_000},
                {"title": "习题讲解", "summary": "text-3", "start_ms": 3_000},
                {"title": "习题讲解。", "summary": "text-4", "start_ms": 4_000},
            ],
        }
        result = self._run(merge)
        self.assertEqual(
            [item["title"] for item in result["chapters"]],
            ["微积分的历史", "习题讲解"],
            "归一化（空白/标点剥离）同键保留首现、丢后现",
        )


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
            if messages[0]["content"] == _SUMMARY_MERGE_PROMPT_TEACHING:
                # RXFIX-2：带要点面（过质量门，usage 账本钉不加质量重试调用）。
                return json.dumps({"markdown": "combined", "chapters": [], "key_takeaways": ["要点"]})
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
            if messages[0]["content"] == _SUMMARY_MERGE_PROMPT_TEACHING:
                # RXFIX-2：带要点面（过质量门，遥测钉维持 merge_retries=0/llm_calls=2）。
                return json.dumps({"markdown": "m", "chapters": [], "key_takeaways": ["要点"]})
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


class DocSectionCountDisciplineTests(unittest.TestCase):
    """V010-QUAL-FIX-1（SHUJI-QUAL 移交①+U4 定谳）：doc 侧章节数量纪律根修钉。

    659766/681269 实录：doc.sections 51/52 章超帽产出——chapters 板级已过
    enforce_chapter_count（CONTENT-IMPL-1 件3，20 区间帽）而 doc 路径
    （coerce_llm_doc 候选/本地兜底组装）漏施加同款纪律=代码不对称。本钉以
    合成超长 doc 候选（51 段，锚全在白名单）重放真实生成链，断言落键
    doc.sections 收敛 ≤20（帽=区间帽：时长//3 分钟地板与 20 取小、≥3 地板）
    且合并计数随遥测归档——生成时收敛，非事后丢弃。"""

    @staticmethod
    def _long_transcript() -> list[dict]:
        # 80 段 × 1 分钟 = 80 分钟讲次 → 区间帽=min(4_800_000//180_000, 20)=20。
        return [
            {"start_ms": index * 60_000, "end_ms": (index + 1) * 60_000,
             "text": f"第{index}分钟的内容。"}
            for index in range(80)
        ]

    @staticmethod
    def _overcap_merge_payload(count: int = 51) -> dict:
        # 659766 形态重放：doc 候选 51 段细粒度碎章（LLM 镜像窗级碎章 1:1 的
        # 失控形态）；chapters 空表（板级纪律已有独立钉，本钉只盯 doc 侧）。
        sections = [
            {
                "start_ms": index * 60_000,
                "title": f"碎章{index}",
                "source": "semantic",
                "points": [{"text": f"要点{index}", "anchor_ms": index * 60_000}],
                "formulas": [],
            }
            for index in range(count)
        ]
        return {
            "markdown": "合并笔记",
            "chapters": [],
            "doc": {"overview": "总述。", "sections": sections,
                    "terms": [], "summary_notes": []},
        }

    def test_overcap_doc_sections_converge_to_cap_with_telemetry(self):
        lines: list[str] = []

        def fake_chat(api_key, messages, **kwargs):
            if messages[0]["content"] == _SUMMARY_MERGE_PROMPT_TEACHING:
                return json.dumps(self._overcap_merge_payload())
            return json.dumps(_part("窗口笔记"))

        with (
            patch.dict(os.environ, {SUMMARY_THINKING_ENV: "", REVIEW_VIEWS_ENV: "off"}, clear=False),
            patch("courselens_worker.llm._chat", side_effect=fake_chat),
            patch("courselens_worker.llm._emit_telemetry", side_effect=lines.append),
        ):
            result = create_summary(
                "k", title="t", transcript=self._long_transcript(), ppt_pages=[],
            )
        sections = result["doc"]["sections"]
        self.assertLessEqual(len(sections), 20)
        self.assertGreaterEqual(len(sections), 3)
        starts = [item["start_ms"] for item in sections]
        self.assertEqual(starts, sorted(starts))  # 按时间有序
        # 内容零丢：51 条要点全数保留（合并拼接不删字，锚白名单值不丢形）。
        self.assertEqual(sum(len(item["points"]) for item in sections), 51)
        final = [line for line in lines if line.startswith("stage=summary ")]
        self.assertEqual(len(final), 1, lines)
        self.assertIn("doc_sections_cap_merged=", final[0])

    def test_undercap_doc_sections_passthrough_unchanged(self):
        # 同族常态（12-15 章）零行为漂移：≤帽 doc 原样透传（键集零变化）。
        lines: list[str] = []

        def fake_chat(api_key, messages, **kwargs):
            if messages[0]["content"] == _SUMMARY_MERGE_PROMPT_TEACHING:
                return json.dumps(self._overcap_merge_payload(count=12))
            return json.dumps(_part("窗口笔记"))

        with (
            patch.dict(os.environ, {SUMMARY_THINKING_ENV: "", REVIEW_VIEWS_ENV: "off"}, clear=False),
            patch("courselens_worker.llm._chat", side_effect=fake_chat),
            patch("courselens_worker.llm._emit_telemetry", side_effect=lines.append),
        ):
            result = create_summary(
                "k", title="t", transcript=self._long_transcript(), ppt_pages=[],
            )
        self.assertEqual(len(result["doc"]["sections"]), 12)
        final = [line for line in lines if line.startswith("stage=summary ")]
        self.assertIn("doc_sections_cap_merged=0", final[0])


class EnforceDocSectionCapUnitTests(unittest.TestCase):
    """V010-QUAL-FIX-1 单元钉：doc 侧章节数量纪律（镜像 enforce_chapter_count
    语义；单源常量自 summary_contract 引入）。"""

    @staticmethod
    def _sections(count: int) -> list[dict]:
        return [
            {
                "start_ms": index * 60_000,
                "title": f"碎章{index}",
                "source": "semantic",
                "points": [{"text": f"要点{index}", "anchor_ms": index * 60_000}],
                "formulas": [],
            }
            for index in range(count)
        ]

    def test_over_cap_merges_nearest_neighbor_with_zero_content_loss(self):
        # 681269 形态（52 章）/80 分钟讲次 → 帽=20；52 条要点全数保留。
        doc = {"overview": "总述。", "sections": self._sections(52)}
        merged, count = llm._enforce_doc_section_cap(doc, 80 * 60_000)
        self.assertEqual(len(merged["sections"]), 20)
        self.assertEqual(count, 32)
        self.assertEqual(
            sum(len(item["points"]) for item in merged["sections"]), 52)
        starts = [item["start_ms"] for item in merged["sections"]]
        self.assertEqual(starts, sorted(starts))  # 按时间有序
        self.assertEqual(merged["sections"][0]["title"], "碎章0")  # 合并对取先者
        # 输入不回写（纯函数）。
        self.assertEqual(len(doc["sections"]), 52)

    def test_under_cap_passthrough_is_identity(self):
        doc = {"overview": "总述。", "sections": self._sections(12)}
        result, count = llm._enforce_doc_section_cap(doc, 80 * 60_000)
        self.assertIs(result, doc)  # 逐位原样透传（旧路径零漂移）
        self.assertEqual(count, 0)

    def test_none_and_malformed_sections_fail_closed(self):
        self.assertEqual(llm._enforce_doc_section_cap(None, 80 * 60_000), (None, 0))
        doc = {"overview": "总述。", "sections": ["坏形状", {"start_ms": 0}]}
        result, count = llm._enforce_doc_section_cap(doc, 80 * 60_000)
        self.assertEqual(result["sections"], [{"start_ms": 0}])  # 非法项剔除
        self.assertEqual(count, 0)

    def test_merged_points_and_formulas_are_concatenated(self):
        # 21 段碎章（最小间隙对=首对）/59 分钟讲次 → 帽=19；points/formulas
        # 逐条拼接零删改（合并不删字，锚白名单值不丢形）。
        over = [
            {
                "start_ms": index * 1_000,
                "title": f"章{index}",
                "source": "semantic",
                "points": [{"text": f"点{index}", "anchor_ms": index * 1_000}],
                "formulas": [{"latex": f"L{index}", "anchor_ms": index * 1_000}],
            }
            for index in range(21)
        ]
        merged, count = llm._enforce_doc_section_cap(
            {"overview": "总述。", "sections": over}, 59 * 60_000)
        self.assertEqual(len(merged["sections"]), 19)
        self.assertEqual(count, 2)
        self.assertEqual(sum(len(i["points"]) for i in merged["sections"]), 21)
        self.assertEqual(sum(len(i["formulas"]) for i in merged["sections"]), 21)
        first = merged["sections"][0]
        self.assertEqual([p["text"] for p in first["points"]], ["点0", "点1"])
        self.assertEqual([f["latex"] for f in first["formulas"]], ["L0", "L1"])

    def test_example_singleton_inherits_first_then_second(self):
        sections = [
            {"start_ms": 0, "title": "甲", "source": "semantic",
             "points": [{"text": "甲点", "anchor_ms": 0}], "formulas": [],
             "example": {"text": "先例"}},
            {"start_ms": 1_000, "title": "乙", "source": "semantic",
             "points": [{"text": "乙点", "anchor_ms": 1_000}], "formulas": [],
             "example": {"text": "后例"}},
        ]
        # 触发合并需超帽：复制 21 段把首对挤到最近邻——最小间隙对=首对
        # （0 与 1000），其余 19 段间隙 60_000。
        padded = sections + [
            {"start_ms": (index + 2) * 60_000, "title": f"章{index}",
             "source": "semantic",
             "points": [{"text": f"点{index}",
                         "anchor_ms": (index + 2) * 60_000}],
             "formulas": []}
            for index in range(19)
        ]
        merged, _ = llm._enforce_doc_section_cap(
            {"overview": "总述。", "sections": padded}, 80 * 60_000)
        self.assertEqual(len(merged["sections"]), 20)
        first = merged["sections"][0]
        self.assertEqual(first["title"], "甲")
        self.assertEqual(first["points"][0]["text"], "甲点")
        self.assertEqual(first["points"][1]["text"], "乙点")  # 拼接零丢
        self.assertEqual(first["example"]["text"], "先例")  # 单例取先者
        no_example = dict(padded[0])
        no_example.pop("example")
        padded2 = [no_example] + padded[1:]
        merged2, _ = llm._enforce_doc_section_cap(
            {"overview": "总述。", "sections": padded2}, 80 * 60_000)
        self.assertEqual(merged2["sections"][0]["example"]["text"], "后例")

    def test_discipline_output_is_idempotent(self):
        doc = {"overview": "总述。", "sections": self._sections(51)}
        once, _ = llm._enforce_doc_section_cap(doc, 45 * 60_000)
        twice, count = llm._enforce_doc_section_cap(once, 45 * 60_000)
        self.assertEqual(count, 0)
        self.assertEqual(once, twice)

    def test_unsorted_undercap_input_is_time_ordered(self):
        sections = list(reversed(self._sections(5)))
        doc = {"overview": "总述。", "sections": sections}
        result, count = llm._enforce_doc_section_cap(doc, 80 * 60_000)
        self.assertEqual(count, 0)
        starts = [item["start_ms"] for item in result["sections"]]
        self.assertEqual(starts, sorted(starts))


if __name__ == "__main__":
    unittest.main()
