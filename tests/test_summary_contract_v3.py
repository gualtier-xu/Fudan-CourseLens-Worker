"""V010-LEARN-IMPL-P0-1 期 0 合同冻结钉（LEARN-D2 §4.6 数据契约包）。

三契约的序列化/解析/向后兼容/加法演进钉：
- timestamp_summary v3：schema_ver=3 加法标记 + doc 字段结构校验/坏形状降级；
- chapters_v2 板块：对齐闭集（首块 0:00/≥3 块/≥180s 地板/回翻不成块）+
  semantic_fallback 本地切片 + 不合闭集整键省略；
- events confidence/source：加法标注（v2 四字段零变化）+ 闭集。
全部零网络（mock _chat / 纯函数），合成转写不含任何真实课程信息。
"""

from __future__ import annotations

import json
from unittest.mock import patch

from courselens_worker.llm import create_summary
from courselens_worker.summary_contract import (
    DOC_AGGREGATIONS,
    DOC_SECTION_SOURCES,
    EVENT_CONFIDENCES,
    EVENT_SOURCES,
    SCHEMA_VER,
    annotate_assessment_events,
    build_chapters_v2,
    build_summary_doc,
    validate_chapters_v2,
    validate_summary_doc,
)


def _doc_sections(count: int = 3, *, start=0, step=300_000) -> list[dict]:
    return [
        {
            "start_ms": start + index * step,
            "title": f"板块{index}",
            "source": "semantic",
            "points": [{"text": f"要点{index}", "anchor_ms": start + index * step}],
            "formulas": [],
        }
        for index in range(count)
    ]


def _good_doc() -> dict:
    return {
        "overview": ["本讲 overview 第一段", "第二段"],
        "sections": _doc_sections(),
        "terms": [],
        "summary_notes": ["小结一", "小结二", "小结三"],
    }


def _block_items(count: int = 3, step: int = 300_000) -> list[dict]:
    return [
        {
            "title": f"板块{index}",
            "start_ms": index * step,
            "summary": f"摘要{index}",
            "source": "semantic",
        }
        for index in range(count)
    ]


class TestSummaryDocContract:
    def test_good_doc_passes_and_serializes_roundtrip(self):
        doc = validate_summary_doc(_good_doc())
        assert doc is not None
        assert doc["overview"] == ["本讲 overview 第一段", "第二段"]
        assert len(doc["sections"]) == 3
        # 序列化/解析钉：JSON 往返后同一闭集门判定不变。
        revived = json.loads(json.dumps(doc, ensure_ascii=False))
        assert validate_summary_doc(revived) == doc

    def test_overview_string_form_accepted(self):
        doc = validate_summary_doc({**_good_doc(), "overview": "单字符串 overview"})
        assert doc is not None and doc["overview"] == "单字符串 overview"

    def test_bad_shapes_degrade_to_none(self):
        # sections 缺失/空=坏形状降级（绝不渲染半残文档，LEARN-D2 §5 风险 2）。
        assert validate_summary_doc({"overview": "x", "sections": []}) is None
        assert validate_summary_doc({"overview": "x"}) is None
        assert validate_summary_doc({"sections": _doc_sections()}) is None
        assert validate_summary_doc(None) is None
        assert validate_summary_doc("doc") is None
        # source 闭集外=拒。
        bad_source = _good_doc()
        bad_source["sections"][0]["source"] = "youtube"
        assert validate_summary_doc(bad_source) is None
        # section 无 points=拒（title-only 板块不成文档）。
        no_points = _good_doc()
        no_points["sections"][0]["points"] = []
        assert validate_summary_doc(no_points) is None
        # formula 缺 latex=拒。
        bad_formula = _good_doc()
        bad_formula["sections"][0]["formulas"] = [{"caption": "无正文"}]
        assert validate_summary_doc(bad_formula) is None
        # 闭集常量冻结钉。
        assert DOC_SECTION_SOURCES == ("slides", "upload", "semantic")
        assert EVENT_CONFIDENCES == ("high", "low")
        assert EVENT_SOURCES == ("transcription", "manual")
        assert DOC_AGGREGATIONS == ("slide_alignment", "semantic_fallback")
        assert SCHEMA_VER == 3


class TestChaptersV2ClosedSet:
    def test_conforming_blocks_pass(self):
        value = validate_chapters_v2(
            {"aggregation": "semantic_fallback", "items": _block_items()}
        )
        assert value is not None and len(value["items"]) == 3
        assert value["aggregation"] == "semantic_fallback"

    def test_closed_set_gates(self):
        # 首块非 0:00=拒。
        shifted = _block_items()
        shifted[0]["start_ms"] = 1000
        assert validate_chapters_v2({
            "aggregation": "semantic_fallback", "items": shifted,
        }) is None
        # <3 块=拒。
        assert validate_chapters_v2({
            "aggregation": "semantic_fallback", "items": _block_items(2),
        }) is None
        # 回翻（start_ms 回退）=拒。
        items = _block_items()
        items[2]["start_ms"] = items[1]["start_ms"] - 1
        assert validate_chapters_v2({
            "aggregation": "semantic_fallback", "items": items,
        }) is None
        # 地板 <180s=拒。
        assert validate_chapters_v2({
            "aggregation": "semantic_fallback",
            "items": _block_items(step=100_000),
        }) is None
        # aggregation 闭集外=拒。
        assert validate_chapters_v2({
            "aggregation": "vibes", "items": _block_items(),
        }) is None

    def test_semantic_fallback_slices_transcript_to_target_blocks(self):
        # 45 分钟讲次（每段 60s），5 分钟目标 → 9 块（尾块 3 分钟 ≥ 地板）。
        transcript = [
            {"start_ms": index * 60_000, "end_ms": (index + 1) * 60_000,
             "text": f"第{index}分钟转写句。"}
            for index in range(45)
        ]
        value = build_chapters_v2([], transcript)
        assert value is not None
        assert value["aggregation"] == "semantic_fallback"
        items = value["items"]
        assert items[0]["start_ms"] == 0
        assert len(items) == 9
        spans = [
            items[index + 1]["start_ms"] - items[index]["start_ms"]
            for index in range(len(items) - 1)
        ]
        assert all(span >= 180_000 for span in spans)
        assert all(item["source"] == "semantic" for item in items)
        # 转写短语兜底命名：无章节时块名来自转写头。
        assert items[0]["title"]

    def test_chapter_titles_land_in_blocks(self):
        transcript = [
            {"start_ms": index * 60_000, "end_ms": (index + 1) * 60_000,
             "text": f"第{index}分钟转写句。"}
            for index in range(45)
        ]
        chapters = [
            {"title": "开场引入", "start_ms": 0, "summary": "开场摘要"},
            {"title": "核心机制", "start_ms": 900_000, "summary": "机制摘要"},
        ]
        value = build_chapters_v2(chapters, transcript)
        assert value is not None
        titles = [item["title"] for item in value["items"]]
        assert titles[0] == "开场引入"  # 0-5 分钟块吃开场章名
        assert "核心机制" in titles      # 15 分钟章名落入对应块

    def test_short_lecture_degrades_to_none(self):
        # 6 分钟讲次：尾块并入后 <3 块 → 整键省略（前端走旧 chapters）。
        transcript = [
            {"start_ms": index * 60_000, "end_ms": (index + 1) * 60_000,
             "text": f"短讲{index}。"}
            for index in range(6)
        ]
        assert build_chapters_v2([], transcript) is None
        # 零转写（纯课件讲次）=诚实省键，不虚构板块。
        assert build_chapters_v2(_block_items(), []) is None
        assert build_chapters_v2([], []) is None

    def test_doc_builds_from_chapters_v2_and_degrades_without_it(self):
        chapters_v2 = validate_chapters_v2(
            {"aggregation": "semantic_fallback", "items": _block_items()}
        )
        doc = build_summary_doc(
            "# 笔记\n\n第一段概述。\n\n第二段概述。", chapters_v2,
            ["结一", "结二", "结三", "结四", "结五", "结六"],
        )
        assert doc is not None
        # 3-5 条小结合同：takeaways(6) → 头部 5 条。
        assert doc["summary_notes"] == ["结一", "结二", "结三", "结四", "结五"]
        assert len(doc["sections"]) == 3
        assert doc["sections"][0]["points"][0]["anchor_ms"] == 0
        # overview=笔记开头段落（诚原子集）。
        assert doc["overview"][0] == "笔记"
        # chapters_v2 缺席=整链降级（绝不半残文档）。
        assert build_summary_doc("# 笔记", None, ["结一"]) is None
        # 空 markdown（无 overview 可提）=降级。
        assert build_summary_doc("", chapters_v2, ["结一"]) is None


class TestEventsAdditive:
    def test_annotate_is_purely_additive(self):
        events = [{
            "category": "quiz", "title": "第3次小测",
            "due_hint": "下周三", "quote": "原文引句",
        }]
        annotated = annotate_assessment_events(events)
        # v2 四字段零变化。
        assert annotated[0]["category"] == "quiz"
        assert annotated[0]["title"] == "第3次小测"
        assert annotated[0]["due_hint"] == "下周三"
        assert annotated[0]["quote"] == "原文引句"
        # 加法两键落在闭集内。
        assert annotated[0]["confidence"] == "low"
        assert annotated[0]["source"] == "transcription"
        # 输入不被回写（纯函数）。
        assert "confidence" not in events[0]
        # 非 dict 项透传（上游拒绝计数语义不变）。
        assert annotate_assessment_events(["junk"])[0] == "junk"

    def test_radar_rule_semantics_high_vs_merge_chain_low(self):
        # 语义钉：规则三证齐=high（雷达侧），合并链候选=low（本侧缺省）。
        rule_draft = annotate_assessment_events(
            [{"category": "exam", "title": "期中", "due_hint": "", "quote": "q",
              "confidence": "high"}]
        )
        assert rule_draft[0]["confidence"] == "high"
        merge_chain = annotate_assessment_events(
            [{"category": "exam", "title": "期中", "due_hint": "", "quote": "q"}]
        )
        assert merge_chain[0]["confidence"] == "low"
        # 闭集外值=缺省降级，不放行任意词表。
        weird = annotate_assessment_events(
            [{"category": "exam", "confidence": "certain", "source": "dream"}]
        )
        assert weird[0]["confidence"] == "low"
        assert weird[0]["source"] == "transcription"


class TestCreateSummaryV3Integration:
    def test_v3_marker_and_additive_keys_reach_the_wire(self):
        """合成讲次全链一次（零真实外呼）：45 分钟合成转写 → v3 标记恒在场，
        doc/chapters_v2 过门落键，v2 既有键形状零变化，事件加法标注随链透传。"""
        transcript = [
            {"start_ms": index * 60_000, "end_ms": (index + 1) * 60_000,
             "text": f"合成转写第{index}分钟。"}
            for index in range(45)
        ]
        window = {
            "markdown": "窗口笔记",
            "chapters": [{"title": "开场", "start_ms": 0, "summary": "s"}],
        }
        merge_payload = {
            "markdown": "# 合成笔记\n\n第一段概述。",
            "chapters": [
                {"title": "开场", "start_ms": 0, "summary": "开场摘要"},
                {"title": "机制", "start_ms": 900_000, "summary": "机制摘要"},
            ],
            "key_takeaways": ["结论一", "结论二", "结论三"],
            "assessment_events": [
                # quote 闭集门=parts 拼接原文子串（窗载荷 markdown）。
                {"category": "quiz", "title": "小测", "due_hint": "下周",
                 "quote": "窗口笔记"},
            ],
        }

        # 窗口调用回窗载荷，合并调用回合并载荷（按 system 串判别，同
        # test_summary_events 走查纪律）。
        def _routed_chat(api_key, messages, **kwargs):
            is_merge = "assessment_events" in messages[0]["content"]
            return json.dumps(merge_payload if is_merge else window)

        with patch("courselens_worker.llm._chat", _routed_chat):
            value = create_summary("key", title="合成讲", transcript=transcript, ppt_pages=[])
        # v3 加法标记恒在场。
        assert value["schema_ver"] == 3
        # 板块/doc 过门落键。
        assert value.get("chapters_v2") is not None
        assert value["chapters_v2"]["aggregation"] == "semantic_fallback"
        assert value["chapters_v2"]["items"][0]["start_ms"] == 0
        assert value.get("doc") is not None
        assert len(value["doc"]["sections"]) == len(value["chapters_v2"]["items"])
        # v2 既有键形状零变化（旧消费者「未知字段忽略」零感知）。
        assert value["markdown"] == "# 合成笔记\n\n第一段概述。"
        assert value["chapters"][0] == {"title": "开场", "start_ms": 0, "summary": "开场摘要"}
        assert value["key_takeaways"] == ["结论一", "结论二", "结论三"]
        assert value["assessment_events"][0]["category"] == "quiz"
        assert value["assessment_events"][0]["confidence"] == "low"
        assert value["assessment_events"][0]["source"] == "transcription"

    def test_v3_marker_survives_degradation_without_doc_keys(self):
        """钉测载荷（零转写）走降级路径：schema_ver=3 仍在，doc/chapters_v2
        整键省略（旧 v2 消费者面对的载荷形状与升级前逐位同形+加法标记）。"""
        window = {
            "markdown": "窗口笔记",
            "chapters": [{"title": "开场", "start_ms": 0, "summary": "s"}],
        }
        merge_payload = {
            "markdown": "# 合成笔记",
            "chapters": [],
            "key_takeaways": ["结论一", "结论二", "结论三"],
            "assessment_events": [],
        }

        def _routed_chat(api_key, messages, **kwargs):
            is_merge = "assessment_events" in messages[0]["content"]
            return json.dumps(merge_payload if is_merge else window)

        with patch("courselens_worker.llm._chat", _routed_chat):
            value = create_summary("key", title="t", transcript=[], ppt_pages=[])
        assert value["schema_ver"] == 3
        assert "doc" not in value
        assert "chapters_v2" not in value
