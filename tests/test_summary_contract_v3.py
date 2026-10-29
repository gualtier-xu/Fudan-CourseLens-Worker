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
    coerce_llm_doc,
    enforce_chapter_count,
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
        # formula 缺 latex/坏式=丢条不拒段（V010-WORKER-QUAL-1 W1：一条残渣
        # 不再核爆整个 LLM doc；段级形状门仍拒段级坏形）。
        bad_formula = _good_doc()
        bad_formula["sections"][0]["formulas"] = [{"caption": "无正文"}]
        doc = validate_summary_doc(bad_formula)
        assert doc is not None
        assert doc["sections"][0]["formulas"] == []
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


class TestChaptersBoundaryGreedy:
    """V010-CONTENT-IMPL-1 件4（CONTENT-R1 C1）：±2 分钟浮动窗内文本重合度
    最低点闭合——零新 LLM 调用的边界感知贪心。"""

    @staticmethod
    def _transcript(minutes: int, topic_words: callable) -> list[dict]:
        return [
            {"start_ms": index * 60_000, "end_ms": (index + 1) * 60_000,
             "text": topic_words(index) + "。"}
            for index in range(minutes)
        ]

    def test_topic_shift_boundary_lands_at_lowest_similarity_point(self):
        # 0-5 分钟词汇族 A、6 分钟起词汇族 B：目标 5 分钟的 ±2min 浮动窗内，
        # 6 分钟切点的「前后文本重合度」最低 → 边界落在 6:00 而非 5:00。
        vocabulary = (
            lambda index: "力加速度牛顿质量惯性摩擦" * 8
            if index < 6 else "电压电流电阻欧姆电路功率" * 8
        )
        value = build_chapters_v2([], self._transcript(20, vocabulary))
        assert value is not None
        starts = [item["start_ms"] for item in value["items"]]
        assert starts[1] == 360_000  # 话题断点=6 分钟（浮动窗内重合度最低点）
        # 地板/单调闭集照守。
        spans = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
        assert all(span >= 180_000 for span in spans)
        assert all(later > earlier for earlier, later in zip(starts, starts[1:]))

    def test_uniform_text_ties_resolve_to_target_mark(self):
        # 均质文本（无重合度低谷）=并列取离目标最近 → 恒在 5 分钟整（期 0
        # 旧行为成为并列特例，确定性回归钉）。
        transcript = self._transcript(
            45, lambda index: f"第{index}分钟转写句。"
        )
        value = build_chapters_v2([], transcript)
        assert value is not None
        assert [item["start_ms"] for item in value["items"]] == [
            index * 300_000 for index in range(9)
        ]

    def test_sparse_transcript_falls_back_to_timed_greedy(self):
        # 稀疏转写（10 分钟大间隙）：浮动窗内无候选段起点 → 回退旧定时贪心
        # （首个 ≥ 目标的段起点），块边界=真实段起点（不虚构窗内时刻）。
        transcript = [
            {"start_ms": index * 600_000, "end_ms": (index + 1) * 600_000,
             "text": f"稀疏段{index}的内容。"}
            for index in range(5)
        ]
        value = build_chapters_v2([], transcript)
        assert value is not None
        assert [item["start_ms"] for item in value["items"]] == [
            index * 600_000 for index in range(5)
        ]

    def test_boundary_greedy_is_deterministic(self):
        transcript = self._transcript(
            30, lambda index: f"确定性内容{index % 7}交替出现。"
        )
        first = build_chapters_v2([], transcript)
        second = build_chapters_v2([], transcript)
        assert first == second  # 纯函数同输入同输出（零随机性）


class TestChapterCountDiscipline:
    """V010-CONTENT-IMPL-1 件3（CONTENT-R1 B2）：章节数量纪律（区间帽）。"""

    def test_over_cap_merges_to_duration_proportional_cap(self):
        # 59 章（实测失控形态）/45 分钟讲次 → 帽=2700s//180s=15。
        chapters = [
            {"title": f"碎章{index}", "start_ms": index * 45_000,
             "summary": f"摘要{index}"}
            for index in range(59)
        ]
        merged = enforce_chapter_count(chapters, 45 * 60_000)
        assert len(merged) == 15
        starts = [item["start_ms"] for item in merged]
        assert starts == sorted(starts)  # 按时间有序
        assert len(set(starts)) == len(starts)  # 合并对消零间隙重复
        # 合并不删字：首章标题保留（合并对取先者）。
        assert merged[0]["title"] == "碎章0"

    def test_under_cap_passthrough_is_sorted_and_value_identical(self):
        chapters = [
            {"title": "乙", "start_ms": 600_000, "summary": "后"},
            {"title": "甲", "start_ms": 0, "summary": "前"},
        ]
        result = enforce_chapter_count(chapters, 45 * 60_000)
        assert [item["title"] for item in result] == ["甲", "乙"]
        assert result[0] == {"title": "甲", "start_ms": 0, "summary": "前"}
        # 输入不回写（纯函数）。
        assert chapters[0]["title"] == "乙"

    def test_short_lecture_floor_cap(self):
        # 短讲次（2 分钟）：帽=max(≥3 地板, min(120s//180s=0, 20))=3。
        chapters = [
            {"title": f"章{index}", "start_ms": index * 20_000,
             "summary": "s"} for index in range(5)
        ]
        assert len(enforce_chapter_count(chapters, 2 * 60_000)) == 3

    def test_discipline_output_is_idempotent(self):
        chapters = [
            {"title": f"碎章{index}", "start_ms": index * 30_000,
             "summary": f"摘要{index}"}
            for index in range(40)
        ]
        once = enforce_chapter_count(chapters, 45 * 60_000)
        twice = enforce_chapter_count(once, 45 * 60_000)
        assert once == twice


class TestCoerceLlmDoc:
    """V010-CONTENT-IMPL-1 件3（期 1 LLM 结构化增量）：doc 候选走同一道门。"""

    _ANCHORS = {0, 300_000, 600_000}

    @staticmethod
    def _doc() -> dict:
        return {
            "overview": ["总述一", "总述二"],
            "sections": [
                {
                    "start_ms": 0,
                    "title": "板块",
                    "source": "semantic",
                    "points": [{"text": "要点", "anchor_ms": 0}],
                    "formulas": [{"latex": "E=mc^2", "anchor_ms": 300_000,
                                  "caption": "质能关系"}],
                },
                {
                    "start_ms": 600_000,
                    "title": "板块二",
                    "source": "youtube",  # 非法自报 → 收敛为 semantic
                    "points": [{"text": "要点二", "anchor_ms": 600_000}],
                    "formulas": [],
                },
            ],
            "summary_notes": ["小结"],
        }

    def test_valid_candidate_passes_same_gate_with_source_coerced(self):
        doc = coerce_llm_doc(self._doc(), self._ANCHORS)
        assert doc is not None
        assert doc == validate_summary_doc(doc)  # 落键形状恒过门
        assert all(item["source"] == "semantic" for item in doc["sections"])
        assert doc["sections"][0]["formulas"][0]["latex"] == "E=mc^2"

    def test_off_whitelist_anchors_prune_points_and_sections(self):
        candidate = self._doc()
        candidate["sections"][0]["points"][0]["anchor_ms"] = 999_999
        doc = coerce_llm_doc(candidate, self._ANCHORS)
        # 第一段点全被剪→段丢弃；只剩第二段（合法锚）——逐级丢弃不整档报废。
        assert doc is not None
        assert [item["title"] for item in doc["sections"]] == ["板块二"]

    def test_all_pruned_or_malformed_returns_none(self):
        assert coerce_llm_doc({"sections": []}, self._ANCHORS) is None
        assert coerce_llm_doc({"sections": "junk"}, self._ANCHORS) is None
        assert coerce_llm_doc("doc", self._ANCHORS) is None
        assert coerce_llm_doc(None, self._ANCHORS) is None
        # 全部段锚白名单外=全剪→None（调用方走本地组装兜底）。
        candidate = self._doc()
        for section in candidate["sections"]:
            section["start_ms"] = 42_000
        assert coerce_llm_doc(candidate, self._ANCHORS) is None

    def test_missing_overview_fails_gate(self):
        # overview 缺席=校验器拒（门不接受半残文档）→ None → 本地兜底。
        candidate = self._doc()
        del candidate["overview"]
        assert coerce_llm_doc(candidate, self._ANCHORS) is None


class TestCoerceLlmDocAnchorRescue:
    """BANQI-FIX1（685785 三连败定谳，2026-10-23）：doc 候选锚拯救与 chapters
    同源（QSCOREFIX-1④ 同族）——白名单外先派生真锚、派生失败维持 fail-closed
    丢段/丢点；pool 缺省 None=逐位旧行为（零回归面）。"""

    _ANCHORS = {0, 300_000, 600_000}
    # 锚池=合成转写（start_ms, text）：段文本含板块/要点的内容词，Dice 可达阈。
    _POOL = [
        (0, "课程安排与考核说明开课介绍"),
        (300_000, "扩散电容与势垒电容的物理机制推导"),
        (600_000, "特征频率与最高振荡频率的关系结论"),
    ]

    @staticmethod
    def _doc() -> dict:
        return {
            "overview": "总述",
            "sections": [
                {
                    "start_ms": 305_841,  # 白名单外（离格时间戳=685785 实证形态）
                    "title": "扩散电容与势垒电容",  # 与池 300_000 段高重合
                    "points": [
                        {"text": "扩散电容与势垒电容的物理机制推导", "anchor_ms": 999_999},
                    ],
                },
            ],
            "summary_notes": ["小结"],
        }

    def test_off_whitelist_section_start_is_rescued_from_title(self):
        stats: dict[str, int] = {}
        doc = coerce_llm_doc(
            self._doc(), self._ANCHORS, anchor_text_pool=self._POOL,
            drop_stats=stats,
        )
        assert doc is not None
        section = doc["sections"][0]
        assert section["start_ms"] == 300_000  # 派生真锚（白名单值，非编造）
        assert section["points"][0]["anchor_ms"] == 300_000
        assert stats.get("sections_anchor_rescued") == 1
        assert stats.get("points_anchor_rescued") == 1
        assert stats.get("accepted_sections") == 1
        # 落键形状恒过同一道门。
        assert doc == validate_summary_doc(doc)

    def test_off_whitelist_point_anchor_rescued_from_point_text(self):
        candidate = self._doc()
        candidate["sections"][0]["start_ms"] = 300_000  # 段锚合法
        stats: dict[str, int] = {}
        doc = coerce_llm_doc(
            candidate, self._ANCHORS, anchor_text_pool=self._POOL,
            drop_stats=stats,
        )
        assert doc is not None
        assert doc["sections"][0]["points"][0]["anchor_ms"] == 300_000
        assert stats.get("points_anchor_rescued") == 1
        assert "sections_dropped_start" not in stats

    def test_non_rescuable_anchors_still_fail_closed(self):
        # 拯救不达阈值=维持既有丢段纪律（绝不编锚）；pool 在场也不放行。
        candidate = self._doc()
        candidate["sections"][0]["title"] = "完全无关的玄学话题甲乙丙丁"
        candidate["sections"][0]["points"][0]["text"] = "风马牛不相及的另一段内容"
        stats: dict[str, int] = {}
        doc = coerce_llm_doc(
            candidate, self._ANCHORS, anchor_text_pool=self._POOL,
            drop_stats=stats,
        )
        assert doc is None
        assert stats.get("sections_dropped_start") == 1

    def test_no_pool_keeps_legacy_behavior_bit_for_bit(self):
        # 不传池=旧行为：白名单外点锚直接剪、离格段锚丢段。
        candidate = self._doc()
        candidate["sections"][0]["start_ms"] = 300_000
        doc = coerce_llm_doc(candidate, self._ANCHORS)
        assert doc is None  # 点锚 999_999 剪空→段丢→None（与改动前逐位一致）
        stats: dict[str, int] = {}
        legacy = coerce_llm_doc(self._doc(), self._ANCHORS, drop_stats=stats)
        assert legacy is None
        assert stats.get("sections_dropped_start") == 1
        assert "sections_anchor_rescued" not in stats

    def test_drop_stats_attributes_validate_rejection(self):
        # 逐级清洗后仍被整门拒=overview 坏形：拒因归档到 validate_rejected。
        candidate = self._doc()
        candidate["sections"][0]["start_ms"] = 300_000
        candidate["sections"][0]["points"][0]["anchor_ms"] = 300_000
        del candidate["overview"]
        stats: dict[str, int] = {}
        doc = coerce_llm_doc(
            candidate, self._ANCHORS, anchor_text_pool=self._POOL,
            drop_stats=stats,
        )
        assert doc is None
        assert stats.get("validate_rejected") == 1


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

    def test_llm_structured_doc_wins_and_chapters_capped_in_chain(self):
        """V010-CONTENT-IMPL-1 件3 全链：合并调用带合法 doc 候选=LLM 结构化
        doc 落键（本地组装让位）；超帽数量章节经纪律合并后落 chapters。"""
        transcript = [
            {"start_ms": index * 60_000, "end_ms": (index + 1) * 60_000,
             "text": f"合成转写第{index}分钟。"}
            for index in range(45)
        ]
        window = {
            "markdown": "窗口笔记",
            "chapters": [{"title": "开场", "start_ms": 0, "summary": "s"}],
        }
        llm_doc = {
            "overview": ["LLM 总述段"],
            "sections": [{
                "start_ms": 0,
                "title": "LLM 板块",
                "source": "semantic",
                "points": [{"text": "锚定要点", "anchor_ms": 0}],
                "formulas": [{"latex": "V=IR", "anchor_ms": 60_000,
                              "caption": "欧姆定律"}],
            }],
            "summary_notes": ["小结一", "小结二", "小结三"],
        }
        merge_payload = {
            "markdown": "# 合成笔记\n\n第一段概述。",
            # 45 分钟讲次帽=15：25 章失控发射→纪律合并。
            "chapters": [
                {"title": f"碎章{index}", "start_ms": index * 60_000,
                 "summary": f"摘要{index}"}
                for index in range(25)
            ],
            "key_takeaways": ["结论一", "结论二", "结论三"],
            "assessment_events": [],
            "doc": llm_doc,
        }

        def _routed_chat(api_key, messages, **kwargs):
            is_merge = "assessment_events" in messages[0]["content"]
            return json.dumps(merge_payload if is_merge else window)

        with patch("courselens_worker.llm._chat", _routed_chat):
            value = create_summary(
                "key", title="合成讲", transcript=transcript, ppt_pages=[],
            )
        # 章节数量纪律：25 → 15（时长比例帽），有序。
        assert len(value["chapters"]) == 15
        starts = [item["start_ms"] for item in value["chapters"]]
        assert starts == sorted(starts)
        # LLM 结构化 doc 落键（overview=LLM 总述，非本地 markdown 段）。
        assert value["doc"]["overview"] == ["LLM 总述段"]
        assert value["doc"]["sections"][0]["title"] == "LLM 板块"
        assert value["doc"]["sections"][0]["formulas"][0]["latex"] == "V=IR"
        # 落键形状恒过同一道门。
        assert validate_summary_doc(value["doc"]) == value["doc"]

    def test_malformed_llm_doc_falls_back_to_local_build(self):
        """坏形状 doc=逐级丢弃→全不可用→本地组装兜底（fail-closed 禁半残）。"""
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
            "chapters": [{"title": "开场", "start_ms": 0, "summary": "开场摘要"}],
            "key_takeaways": ["结论一", "结论二", "结论三"],
            "assessment_events": [],
            "doc": {"sections": "junk"},  # 坏形状候选
        }

        def _routed_chat(api_key, messages, **kwargs):
            is_merge = "assessment_events" in messages[0]["content"]
            return json.dumps(merge_payload if is_merge else window)

        with patch("courselens_worker.llm._chat", _routed_chat):
            value = create_summary(
                "key", title="合成讲", transcript=transcript, ppt_pages=[],
            )
        # doc=本地组装兜底（overview=笔记开头段落），仍过门。
        assert value["doc"]["overview"][0] == "合成笔记"
        assert validate_summary_doc(value["doc"]) == value["doc"]
