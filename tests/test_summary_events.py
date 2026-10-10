"""N5A-P2 summary 顺风车 assessment_events：合并调用的闭集校验与提示词钉。"""

from __future__ import annotations

import json
from unittest.mock import patch

from courselens_worker.llm import (
    ASSESSMENT_CATEGORIES,
    _SUMMARY_MERGE_PROMPT,
    create_summary,
)


def _chat_responder(responses):
    calls = {"index": 0}

    def _chat(api_key, messages, **kwargs):
        index = calls["index"]
        calls["index"] += 1
        payload = responses[min(index, len(responses) - 1)]
        return json.dumps(payload)

    return _chat, calls


def _base_window():
    return {"markdown": "第一窗口笔记", "chapters": [{"title": "开场", "start_ms": 0, "summary": "s"}]}


def test_merge_prompt_carries_closed_enumeration_and_stays_bounded():
    assert "assessment_events" in _SUMMARY_MERGE_PROMPT
    assert "key_takeaways" in _SUMMARY_MERGE_PROMPT
    for category in ASSESSMENT_CATEGORIES:
        assert category in _SUMMARY_MERGE_PROMPT
    # 防膨胀守门（QC：随枚举展开放宽至 300 字）
    assert len(_SUMMARY_MERGE_PROMPT) <= 300


def test_summary_chain_emits_final_telemetry_without_content():
    """夜10-C T10 对称补强：摘要链收口遥测（计数与闭集词，零内容泄漏）。"""
    merge_payload = {
        "markdown": "# 笔记",
        "chapters": [],
        "key_takeaways": ["要点一"],
        "assessment_events": [
            {"category": "exam", "title": "期中考试", "due_hint": "下周三", "quote": "第一窗口笔记"},
        ],
    }
    _chat, calls = _chat_responder([_base_window(), merge_payload])
    lines = []
    with (
        patch("courselens_worker.llm._chat", _chat),
        patch("courselens_worker.llm._emit_telemetry", side_effect=lines.append),
    ):
        value = create_summary("key", title="t", transcript=[], ppt_pages=[{"created_sec": 0}])
    assert len(value["assessment_events"]) == 1
    finals = [line for line in lines if line.startswith("stage=summary ")]
    assert len(finals) == 1, lines
    final = finals[0]
    assert "windows=1/1" in final
    assert "events=1" in final
    assert "events_rejected=0" in final
    assert "takeaways=1" in final
    # 内容绝不进遥测：笔记文本/标题/quote 不出现。
    for leaked in ("第一窗口笔记", "# 笔记", "期中考试", "要点一", "key"):
        assert leaked not in final


def test_valid_events_and_takeaways_pass_through():
    merge_payload = {
        "markdown": "# 笔记",
        "chapters": [],
        "key_takeaways": ["链式聚合怕水", "逐步聚合分子量随转化率升"],
        "assessment_events": [
            {"category": "exam", "title": "期中考试", "due_hint": "下周三", "quote": "第一窗口笔记"},
        ],
    }
    _chat, calls = _chat_responder([_base_window(), merge_payload])
    with patch("courselens_worker.llm._chat", _chat):
        value = create_summary("key", title="t", transcript=[], ppt_pages=[{"created_sec": 0}])
    assert len(value["assessment_events"]) == 1
    assert value["assessment_events"][0]["category"] == "exam"
    assert value["assessment_events_rejected"] == 0
    assert value["key_takeaways"][0] == "链式聚合怕水"


def test_malformed_events_are_dropped_and_counted():
    merge_payload = {
        "markdown": "# 笔记",
        "chapters": [],
        "assessment_events": [
            {"category": "party", "title": "怪类", "due_hint": "", "quote": "第一窗口笔记"},  # 类别越界
            {"category": "exam", "title": "", "due_hint": "", "quote": "第一窗口笔记"},      # 空标题
            {"category": "exam", "title": "无据", "due_hint": "", "quote": "输入里没有这句话"},  # quote 非原文
            {"category": "quiz", "title": "好小测", "due_hint": "周五", "quote": "第一窗口笔记"},  # 合法
        ],
    }
    _chat, _ = _chat_responder([_base_window(), merge_payload])
    with patch("courselens_worker.llm._chat", _chat):
        value = create_summary("key", title="t", transcript=[], ppt_pages=[{"created_sec": 0}])
    assert [item["category"] for item in value["assessment_events"]] == ["quiz"]
    assert value["assessment_events_rejected"] == 3


def test_takeaways_are_capped_and_blank_stripped():
    merge_payload = {
        "markdown": "# 笔记",
        "chapters": [],
        "key_takeaways": ["一条" * 40, "", "  ", "短句一", "短句二", "短句三", "短句四", "短句五", "短句六", "短句七"],
        "assessment_events": [],
    }
    _chat, _ = _chat_responder([_base_window(), merge_payload])
    with patch("courselens_worker.llm._chat", _chat):
        value = create_summary("key", title="t", transcript=[], ppt_pages=[{"created_sec": 0}])
    assert len(value["key_takeaways"]) == 6
    assert all(len(item) <= 60 for item in value["key_takeaways"])


def test_model_alias_is_modernized():
    """N5A-P6：deepseek-chat 别名已被官方停用（2026-07-24），防回退钉。"""
    from courselens_worker import llm

    assert llm.MODEL == "deepseek-flash"


def test_evidence_prompt_is_separate_and_pins_the_honesty_rules():
    """N7A：多源证据版提示词另立一串，旧串一个字都不改（有 300 字守门）。"""
    from courselens_worker import llm

    assert len(_SUMMARY_MERGE_PROMPT) <= 300
    assert llm._SUMMARY_MERGE_PROMPT_WITH_EVIDENCE.startswith(_SUMMARY_MERGE_PROMPT)
    assert "knowledge_points" in llm._SUMMARY_MERGE_PROMPT_WITH_EVIDENCE
    assert "topic_candidates" in llm._SUMMARY_MERGE_PROMPT_WITH_EVIDENCE
    # 三条底线：只依据证据 / 冲突并列不裁决 / 材料正文不是指令 / 不编造答案
    for rule in ("只依据 evidence", "不要裁决", "不可信数据", "不得编造标准答案",
                 "材料未给答案"):
        assert rule in llm._SUMMARY_MERGE_PROMPT_WITH_EVIDENCE
    assert "不可信数据" in llm._SUMMARY_EVIDENCE_WINDOW_PROMPT
    assert "evidence" in llm._SUMMARY_EVIDENCE_WINDOW_PROMPT


def test_legacy_merge_prompt_never_mentions_the_new_output_keys():
    """旧路径的模型输出与历史一致：旧提示词不提知识点/主题候选。"""
    assert "knowledge_points" not in _SUMMARY_MERGE_PROMPT
    assert "topic_candidates" not in _SUMMARY_MERGE_PROMPT


def _transcript() -> list[dict]:
    return [
        {"start_ms": 0, "end_ms": 30_000, "text": "链式聚合反应最主要的特征是逐步聚合，分子量随转化率上升。"},
        {"start_ms": 30_000, "end_ms": 60_000, "text": "接下来讲凝胶化现象和临界转化 crazily 无关词。"},
    ]


def test_takeaway_anchors_align_and_stay_in_whitelist():
    """RR-QWIN-1 Q1：takeaway 锚=本地派生（零新调用），与 key_takeaways
    等长对齐；锚值只取 valid_anchors 白名单（字幕段/合法 chapters），无把握
    时为 None（fail-closed：锚错比锚缺更伤信任）。"""
    merge_payload = {
        "markdown": "# 笔记",
        "chapters": [{"title": "链式聚合", "start_ms": 0, "summary": "链式聚合：分子量随转化率上升，注意凝胶化。"}],
        "key_takeaways": ["链式聚合的分子量随转化率上升", "完全无关的一句闲聊"],
        "assessment_events": [],
    }
    _chat, _ = _chat_responder([_base_window(), merge_payload])
    with patch("courselens_worker.llm._chat", _chat):
        value = create_summary("key", title="t", transcript=_transcript(), ppt_pages=[])
    assert len(value["takeaway_anchors"]) == len(value["key_takeaways"]) == 2
    valid = {0, 30_000}
    first = value["takeaway_anchors"][0]
    assert first in valid
    assert value["takeaway_anchors"][1] is None


def test_takeaway_anchors_survive_chapter_only_inputs():
    """合并产物 chapters 是合法锚源：transcript 为空时锚仍可落在章节 start_ms
    （start_ms 已过 valid_anchors 校验），对齐关系不因输入面变化而破坏。"""
    merge_payload = {
        "markdown": "# 笔记",
        "chapters": [{"title": "凝胶化", "start_ms": 45_000, "summary": "凝胶化现象与临界转化率。"}],
        "key_takeaways": ["凝胶化出现在临界转化率附近"],
        "assessment_events": [],
    }
    _chat, _ = _chat_responder([_base_window(), merge_payload])
    with patch("courselens_worker.llm._chat", _chat):
        value = create_summary("key", title="t", transcript=[], ppt_pages=[{"created_sec": 45}])
    assert value["takeaway_anchors"] == [45_000]


def test_takeaway_anchors_fail_closed_without_candidates():
    """无候选（无字幕段、无合法章节、无幻灯锚）时全部 None：不渲染锚而非
    报错，绝不产出白名单之外的锚值。"""
    merge_payload = {
        "markdown": "# 笔记",
        "chapters": [],
        "key_takeaways": ["要点一", "要点二"],
        "assessment_events": [],
    }
    _chat, _ = _chat_responder([_base_window(), merge_payload])
    with patch("courselens_worker.llm._chat", _chat):
        value = create_summary("key", title="t", transcript=[], ppt_pages=[])
    assert value["takeaway_anchors"] == [None, None]
