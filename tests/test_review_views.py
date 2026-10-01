"""P2-CONTRACT-1 PKG-A：多视图复习包派生调用（worker 侧）钉面。

覆盖：提示词 ≤300 帽+底线词、开关闭集（缺省开/off 零调用零开销）、四档 schema
冻结形状、锚白名单（study_guide/faq 非法置 null、timeline 白名单外整条丢弃）、
引用闭集（无 evidence_index 强制空）、exam_alerts 只改写不新造、两败/传输失败
fail-open 且 summary 照常、检查点成功增量写与续跑复用、坏 JSON 逐档抢救、
数量帽/空串丢条/空档省键。遥测与账本纪律：计数与闭集词，零内容。
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from courselens_worker.llm import (
    _REVIEW_VIEWS_PROMPT,
    LLMError,
    REVIEW_VIEWS_ENV,
    _derive_review_views,
    _review_views_enabled,
    _salvage_views_object,
    _validate_review_views,
    create_summary,
)

_WINDOW = {"markdown": "窗口笔记", "chapters": []}
_MERGE = {
    "markdown": "# 笔记",
    "chapters": [],
    "key_takeaways": ["要点一"],
    "assessment_events": [
        {"category": "exam", "title": "期中考试", "due_hint": "下周三", "quote": "窗口笔记"},
    ],
}
_VIEWS = {
    "study_guide": {
        "items": [
            {"question": "链式聚合的特征是什么？", "hint": "从逐步聚合入手",
             "anchor_ms": 0, "citation_ids": []},
        ],
    },
    "faq": {
        "items": [
            {"question": "凝胶化何时出现？", "answer": "临界转化率附近。",
             "anchor_ms": 30000, "citation_ids": []},
        ],
    },
    "timeline": {
        "events": [
            {"start_ms": 30000, "title": "凝胶化", "detail": "临界转化率附近"},
            {"start_ms": 0, "title": "开场", "detail": ""},
        ],
    },
    "briefing": {
        "speed_read": "本讲讲链式聚合与凝胶化。",
        "must_know": ["链式聚合逐步进行"],
        "exam_alerts": [
            {"category": "exam", "title": "期中考试", "due_hint": "下周三"},
        ],
    },
}
_TRANSCRIPT = [
    {"start_ms": 0, "end_ms": 30_000, "text": "链式聚合反应逐步进行。"},
    {"start_ms": 30_000, "end_ms": 60_000, "text": "凝胶化出现在临界转化率。"},
]


def _stage_responder(window_payload, merge_payload, views_payloads):
    """按 system 提示词分流的替身：views 响应重放末项供重试；每调用落一条
    usage 流水（与 test_summary_chain_hardening.UsageSinkTests 同法），供
    deep_usage 账面断言。"""
    calls = {"window": 0, "merge": 0, "views": 0}

    def _chat(api_key, messages, **kwargs):
        from courselens_worker import llm as llm_module

        with llm_module._USAGE_LOCK:
            llm_module._CALL_LOG.append({
                "prompt_tokens": 10, "completion_tokens": 5,
                "reasoning_tokens": 0, "prompt_cache_hit_tokens": 0,
                "latency_ms": 10, "thinking": None,
            })
        system = messages[0]["content"]
        if system == _REVIEW_VIEWS_PROMPT:
            index = min(calls["views"], len(views_payloads) - 1)
            calls["views"] += 1
            return views_payloads[index]
        # 合并提示词携带 assessment_events 枚举；窗口提示词不含。
        if "assessment_events" in system:
            calls["merge"] += 1
            return json.dumps(merge_payload, ensure_ascii=False)
        calls["window"] += 1
        return json.dumps(window_payload, ensure_ascii=False)

    return _chat, calls


def _enabled_env():
    return patch.dict(os.environ, {REVIEW_VIEWS_ENV: ""}, clear=False)


class PromptPinTests(unittest.TestCase):
    def test_prompt_is_bounded_and_carries_the_bottom_lines(self):
        """独立提示词 ≤300 防膨胀帽；四条底线与四档键必须写进提示词。"""
        self.assertLessEqual(len(_REVIEW_VIEWS_PROMPT), 300)
        for needle in (
            "只改写输入", "不新增", "材料没有的不编", "anchor_pool",
            "evidence_index", "study_guide", "faq", "timeline", "briefing",
            "assessment_events",
        ):
            self.assertIn(needle, _REVIEW_VIEWS_PROMPT)

    def test_switch_closed_set(self):
        """闭集 {缺省, "off"}：缺省开；仅 off（大小写/空白容忍）关；其余值=开。"""
        cases = {"": True, "off": False, " OFF ": False, "Off": False,
                 "on": True, "0": True, "bogus": True}
        for raw, expected in cases.items():
            with patch.dict(os.environ, {REVIEW_VIEWS_ENV: raw}, clear=False):
                self.assertEqual(_review_views_enabled(), expected, repr(raw))


class ValidationRuleTests(unittest.TestCase):
    """_validate_review_views 确定性校验面（合同 §① 校验纪律逐条）。"""

    def test_anchor_whitelist_nulls_guide_items_and_drops_timeline_entries(self):
        candidate = {
            "study_guide": {"items": [
                {"question": "Q1", "hint": "", "anchor_ms": 99999, "citation_ids": []},
                {"question": "Q2", "hint": "", "anchor_ms": 30000, "citation_ids": []},
                {"question": "Q3", "hint": "", "anchor_ms": None, "citation_ids": []},
            ]},
            "faq": {"items": [
                {"question": "F", "answer": "A", "anchor_ms": "12:30", "citation_ids": []},
            ]},
            "timeline": {"events": [
                {"start_ms": 0, "title": "合法", "detail": ""},
                {"start_ms": 88888, "title": "错锚整条丢弃", "detail": ""},
                {"start_ms": None, "title": "缺锚整条丢弃", "detail": ""},
            ]},
        }
        views, anchor_rejected, citation_rejected = _validate_review_views(
            candidate, anchor_pool={0, 30_000}, allowed_citations=set(),
            assessment_events=[])
        self.assertEqual(citation_rejected, 0)
        guide = views["study_guide"]["items"]
        self.assertEqual([item["anchor_ms"] for item in guide], [None, 30_000, None])
        faq = views["faq"]["items"]
        self.assertEqual(faq[0]["anchor_ms"], None)
        events = views["timeline"]["events"]
        self.assertEqual([event["title"] for event in events], ["合法"])
        # 99999 + "12:30" + 88888 + None——缺锚/非数值锚同属白名单外，逐条计数。
        self.assertEqual(anchor_rejected, 4)

    def test_citation_closure_keeps_entry_and_drops_unknown_ids(self):
        candidate = {
            "faq": {"items": [
                {"question": "F", "answer": "A", "anchor_ms": None,
                 "citation_ids": ["ckc:ok", "ckc:ghost", ""]},
            ]},
        }
        views, _, citation_rejected = _validate_review_views(
            candidate, anchor_pool=set(), allowed_citations={"ckc:ok"},
            assessment_events=[])
        self.assertEqual(views["faq"]["items"][0]["citation_ids"], ["ckc:ok"])
        self.assertEqual(citation_rejected, 1)

    def test_no_evidence_index_forces_empty_citations(self):
        candidate = {
            "study_guide": {"items": [
                {"question": "Q", "hint": "", "anchor_ms": None,
                 "citation_ids": ["ckc:ghost"]},
            ]},
        }
        views, _, citation_rejected = _validate_review_views(
            candidate, anchor_pool=set(), allowed_citations=set(),
            assessment_events=[])
        self.assertEqual(views["study_guide"]["items"][0]["citation_ids"], [])
        self.assertEqual(citation_rejected, 1)

    def test_exam_alerts_only_rewrite_existing_events(self):
        events = [
            {"category": "exam", "title": "期中考试", "due_hint": "下周三", "quote": "下周三进行期中考试"},
            {"category": "quiz", "title": "随堂小测", "due_hint": "周五", "quote": "周五有小测"},
        ]
        candidate = {"briefing": {"speed_read": "速览。", "must_know": [], "exam_alerts": [
            {"category": "exam", "title": "期中考试提醒", "due_hint": "下周三前"},
            {"category": "exam", "title": "完全无关的新造提醒", "due_hint": ""},
            {"category": "party", "title": "越界类别", "due_hint": ""},
            {"category": "quiz", "title": "小测", "due_hint": "周五"},
        ]}}
        views, _, _ = _validate_review_views(
            candidate, anchor_pool=set(), allowed_citations=set(),
            assessment_events=events)
        alerts = views["briefing"]["exam_alerts"]
        self.assertEqual(
            [(item["category"], item["title"]) for item in alerts],
            [("exam", "期中考试提醒"), ("quiz", "小测")],
        )

    def test_caps_blank_drops_and_unknown_fields_ignored(self):
        candidate = {
            "study_guide": {"items": [
                {"question": "问" * 90, "hint": "示" * 70, "anchor_ms": None,
                 "citation_ids": [], "bonus": "未知字段"},
                {"question": "  ", "hint": "", "anchor_ms": None, "citation_ids": []},
                "not-a-dict",
            ] + [{"question": f"填充{i}", "hint": "", "anchor_ms": None, "citation_ids": []}
                 for i in range(12)]},
        }
        views, _, _ = _validate_review_views(
            candidate, anchor_pool=set(), allowed_citations=set(), assessment_events=[])
        items = views["study_guide"]["items"]
        # 帽 [:12] 作用于原始列表：前 12 条里空串/非 dict 两条被丢 → 落地 10 条。
        self.assertEqual(len(items), 10)
        self.assertEqual(items[0]["question"], "问" * 80)
        self.assertEqual(items[0]["hint"], "示" * 60)
        self.assertNotIn("bonus", items[0])
        self.assertEqual(items[1]["question"], "填充0")

    def test_item_cap_truncates_beyond_twelve(self):
        candidate = {"study_guide": {"items": [
            {"question": f"问{i}", "hint": "", "anchor_ms": None, "citation_ids": []}
            for i in range(15)]}}
        views, _, _ = _validate_review_views(
            candidate, anchor_pool=set(), allowed_citations=set(), assessment_events=[])
        self.assertEqual(len(views["study_guide"]["items"]), 12)

    def test_all_tiers_invalid_returns_none(self):
        candidate = {
            "study_guide": {"items": [{"question": "", "hint": "", "anchor_ms": None, "citation_ids": []}]},
            "briefing": {"speed_read": "  ", "must_know": [], "exam_alerts": []},
        }
        views, _, _ = _validate_review_views(
            candidate, anchor_pool=set(), allowed_citations=set(), assessment_events=[])
        self.assertIsNone(views)


class DeriveCallTests(unittest.TestCase):
    def test_bad_json_then_good_views_retries_once(self):
        usage: list[dict] = []
        views_input = {"title": "t", "note": {"markdown": "m", "chapters": [],
                                              "key_takeaways": [], "assessment_events": [],
                                              "knowledge_points": []},
                       "anchor_pool": [0]}
        responder = _stage_responder(_WINDOW, _MERGE, ["不是 JSON", json.dumps(_VIEWS)])
        with patch("courselens_worker.llm._chat", side_effect=responder[0]), \
                _enabled_env():
            views, stats = _derive_review_views(
                "k", views_input=views_input, tier={"type": "disabled"},
                usage_records=usage)
        self.assertIsNotNone(views)
        self.assertEqual(stats["retries"], 1)
        self.assertEqual(stats["failed"], 0)
        self.assertEqual(len(usage), 2, "失败尝试也落账")

    def test_salvage_recovers_completed_tiers_from_truncated_reply(self):
        usage: list[dict] = []
        truncated = json.dumps(_VIEWS, ensure_ascii=False)
        cut = truncated.rfind('"briefing"')
        truncated = truncated[:cut] + '{"speed_read": "被截断'
        views_input = {"title": "t", "note": {"markdown": "m", "chapters": [],
                                              "key_takeaways": [], "assessment_events": [],
                                              "knowledge_points": []},
                       "anchor_pool": [0, 30_000]}
        responder = _stage_responder(_WINDOW, _MERGE, [truncated])
        with patch("courselens_worker.llm._chat", side_effect=responder[0]), \
                _enabled_env():
            views, stats = _derive_review_views(
                "k", views_input=views_input, tier={"type": "disabled"},
                usage_records=usage)
        self.assertEqual(sorted(views), ["faq", "study_guide", "timeline"])
        self.assertEqual(stats["retries"], 0, "抢救出的档不再重试")

    def test_salvage_scanner_skips_chatty_prefix_and_respects_strings(self):
        raw = '好的，视图如下 {坏} {"study_guide": {"items": [{"question": "花括号{内}字符串", "hint": "", "anchor_ms": null, "citation_ids": []}]}}'
        recovered = _salvage_views_object(raw)
        self.assertEqual(
            recovered["study_guide"]["items"][0]["question"], "花括号{内}字符串",
        )
        self.assertIsNone(_salvage_views_object("完全没有对象"))


class CreateSummaryIntegrationTests(unittest.TestCase):
    def _run(self, views_payloads, *, env=None, prior_checkpoint=None,
             checkpoint=None, transcript=None):
        responder = _stage_responder(_WINDOW, _MERGE, views_payloads)
        lines: list[str] = []
        context = env or _enabled_env()
        with patch("courselens_worker.llm._chat", side_effect=responder[0]), \
                patch("courselens_worker.llm._emit_telemetry", side_effect=lines.append), \
                context:
            value = create_summary(
                "k", title="t",
                transcript=_TRANSCRIPT if transcript is None else transcript,
                ppt_pages=[],
                prior_checkpoint=prior_checkpoint,
                checkpoint=checkpoint,
            )
        return value, responder[1], lines

    def test_four_tier_schema_frozen_shape_and_telemetry(self):
        checkpoints: list[dict] = []
        value, calls, lines = self._run(
            [json.dumps(_VIEWS, ensure_ascii=False)], checkpoint=checkpoints.append)
        views = value["review_views"]
        self.assertEqual(sorted(views), ["briefing", "faq", "study_guide", "timeline"])
        self.assertEqual(sorted(views["study_guide"]["items"][0]), [
            "anchor_ms", "citation_ids", "hint", "question"])
        self.assertEqual(sorted(views["faq"]["items"][0]), [
            "anchor_ms", "answer", "citation_ids", "question"])
        self.assertEqual(sorted(views["timeline"]["events"][0]), [
            "detail", "start_ms", "title"])
        self.assertEqual(
            [event["start_ms"] for event in views["timeline"]["events"]], [0, 30_000],
            "timeline 按 start_ms 升序")
        self.assertEqual(sorted(views["briefing"]), ["exam_alerts", "must_know", "speed_read"])
        final = [line for line in lines if line.startswith("stage=review-views ")]
        self.assertEqual(len(final), 1, lines)
        self.assertIn("study_guide=1", final[0])
        self.assertIn("faq=1", final[0])
        self.assertIn("timeline=2", final[0])
        self.assertIn("briefing=1", final[0])
        self.assertIn("anchor_rejected=0", final[0])
        self.assertIn("citation_rejected=0", final[0])
        self.assertEqual(value["deep_usage"]["views_retries"], 0)
        self.assertEqual(value["deep_usage"]["calls"], 3, "窗口+合并+派生各一次")
        self.assertEqual(calls["views"], 1)
        self.assertEqual(len(checkpoints), 2, "窗口检查点+派生成功检查点")
        self.assertEqual(checkpoints[-1]["stage"], "summary")
        self.assertEqual(checkpoints[-1]["summary_completed_windows"], 1)
        self.assertEqual(checkpoints[-1]["review_views"], views)
        for leaked in ("链式聚合", "期中考试", "速览内容", "# 笔记"):
            self.assertNotIn(leaked, final[0], "遥测零内容")

    def test_views_input_carries_anchor_pool_and_omits_evidence_index(self):
        seen: dict = {}

        def _chat(api_key, messages, **kwargs):
            system = messages[0]["content"]
            if system == _REVIEW_VIEWS_PROMPT:
                seen["payload"] = json.loads(messages[1]["content"])
                return json.dumps(_VIEWS, ensure_ascii=False)
            if "assessment_events" in system:
                return json.dumps(_MERGE, ensure_ascii=False)
            return json.dumps(_WINDOW, ensure_ascii=False)

        with patch("courselens_worker.llm._chat", side_effect=_chat), _enabled_env():
            create_summary("k", title="第3章", transcript=_TRANSCRIPT, ppt_pages=[])
        payload = seen["payload"]
        self.assertEqual(payload["title"], "第3章")
        self.assertEqual(payload["anchor_pool"], [0, 30_000])
        self.assertEqual(payload["note"]["markdown"], "# 笔记")
        self.assertEqual(payload["note"]["key_takeaways"], ["要点一"])
        self.assertNotIn("evidence_index", payload, "无 packet 时整个键省略")
        self.assertNotIn("glossary", payload)

    def test_env_off_makes_zero_calls_and_no_key(self):
        value, calls, lines = self._run(
            [json.dumps(_VIEWS, ensure_ascii=False)],
            env=patch.dict(os.environ, {REVIEW_VIEWS_ENV: "off"}, clear=False))
        self.assertNotIn("review_views", value)
        self.assertEqual(calls["views"], 0, "off=零调用")
        self.assertEqual(value["deep_usage"]["views_retries"], 0)
        self.assertFalse([line for line in lines if "review-views" in line])

    def test_both_failures_fail_open_and_summary_survives(self):
        value, calls, lines = self._run(["垃圾一", "垃圾二"])
        self.assertNotIn("review_views", value)
        self.assertEqual(value["markdown"], "# 笔记", "summary 照常落地")
        self.assertEqual(calls["views"], 2, "attempts=2")
        self.assertEqual(value["deep_usage"]["views_retries"], 1)
        failed = [line for line in lines if line.startswith("stage=review-views-failed")]
        self.assertEqual(len(failed), 1, lines)
        self.assertIn("retries=1", failed[0])
        self.assertEqual(value["deep_usage"]["calls"], 4, "窗口+合并+两败派生")

    def test_transport_failure_fails_open(self):
        def _chat(api_key, messages, **kwargs):
            if messages[0]["content"] == _REVIEW_VIEWS_PROMPT:
                raise LLMError("AI request failed: ConnectionError")
            if "assessment_events" in messages[0]["content"]:
                return json.dumps(_MERGE, ensure_ascii=False)
            return json.dumps(_WINDOW, ensure_ascii=False)

        lines: list[str] = []
        with patch("courselens_worker.llm._chat", side_effect=_chat), \
                patch("courselens_worker.llm._emit_telemetry", side_effect=lines.append), \
                _enabled_env():
            value = create_summary("k", title="t", transcript=_TRANSCRIPT, ppt_pages=[])
        self.assertNotIn("review_views", value)
        self.assertEqual(value["markdown"], "# 笔记")
        self.assertTrue(
            [line for line in lines if line.startswith("stage=review-views-failed")])

    def test_checkpoint_reuse_skips_derivation_call(self):
        reused_views = {"timeline": {"events": [{"start_ms": 0, "title": "旧档", "detail": ""}]}}
        prior = {
            "summary_completed_windows": 1,
            "summary_window_plan": ["transcript"],
            "summary_parts": [{"markdown": "旧窗", "chapters": []}],
            "review_views": reused_views,
        }
        checkpoints: list[dict] = []
        value, calls, lines = self._run(
            [json.dumps(_VIEWS, ensure_ascii=False)],
            prior_checkpoint=prior, checkpoint=checkpoints.append)
        self.assertEqual(calls["views"], 0, "检查点带 review_views=直接复用不重调")
        self.assertEqual(value["review_views"], reused_views)
        self.assertEqual(checkpoints, [], "复用不重写检查点")
        self.assertFalse([line for line in lines if "review-views" in line])
        self.assertEqual(value["deep_usage"]["calls"], 1, "仅合并一调")

    def test_partial_valid_tiers_land_and_empty_tiers_are_omitted(self):
        partial = {
            "study_guide": {"items": [{"question": "", "hint": "", "anchor_ms": None, "citation_ids": []}]},
            "faq": {"items": [{"question": "合法问", "answer": "合法答", "anchor_ms": None, "citation_ids": []}]},
        }
        value, _, _ = self._run([json.dumps(partial, ensure_ascii=False)])
        views = value["review_views"]
        self.assertEqual(sorted(views), ["faq"], "空档/畸形档省键")


if __name__ == "__main__":
    unittest.main()
