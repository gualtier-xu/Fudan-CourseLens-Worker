"""V010-SLIDEALIGN-1 三面钉：课件对齐第二源（抽取/对齐/fail-closed 回退）。

背景=转写源结构性缺公式（BANQI-QUAL U1：老师不念公式，零公式讲次幻灯数
学行 57-61）。本文件钉死 slide_alignment 三面契约与 llm 消费面的加性接线：
slide_pack=None 时逐位旧行为（对照组语义），在场时窗口去重 deck+公式行、
合并输入讲级公式单、提示词后缀变体下线。
"""

from __future__ import annotations

import json
from unittest.mock import patch

from courselens_worker.llm import (
    _MERGE_SLIDE_RULES,
    _SUMMARY_MERGE_PROMPT,
    _SUMMARY_MERGE_PROMPT_TEACHING,
    _SUMMARY_MERGE_PROMPT_TEACHING_SLIDE,
    _SUMMARY_WINDOW_PROMPT,
    _WINDOW_SLIDE_SUFFIX,
    _WINDOW_TEACHING_SUFFIX,
    create_summary,
)
from courselens_worker.slide_alignment import (
    FORMULA_LINE_CAP,
    MAX_DECK_PAGES,
    MAX_FORMULAS,
    PAGE_TEXT_CAP,
    assign_pack_windows,
    build_slide_pack,
    dedupe_pages,
    extract_formula_lines,
)


def _transcript(count: int) -> list[dict]:
    return [
        {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"text-{index}"}
        for index in range(count)
    ]


def _part(markdown: str) -> dict:
    return {"markdown": markdown, "chapters": []}


# ---- 面1：抽取（去重/公式行/打包与帽） ------------------------------------

class TestDedupePages:
    def test_exact_sha_duplicates_collapse_with_repeat_count(self):
        pages = [
            {"page_num": 1, "created_sec": 10, "text": "I_C = I_CBO", "source_sha256": "a" * 64},
            {"page_num": 2, "created_sec": 40, "text": "I_C = I_CBO", "source_sha256": "a" * 64},
            {"page_num": 3, "created_sec": 70, "text": "I_C = I_CBO", "source_sha256": "a" * 64},
        ]
        deck = dedupe_pages(pages)
        assert len(deck) == 1
        assert deck[0]["created_sec"] == 10
        assert deck[0]["repeat_count"] == 2

    def test_fallback_keys_dhash_then_text(self):
        pages = [
            {"page_num": 1, "created_sec": 1, "text": "同文重复", "dhash": "f" * 16},
            {"page_num": 2, "created_sec": 2, "text": "同文重复", "dhash": "f" * 16},
            {"page_num": 3, "created_sec": 3, "text": "同 文 重 复"},
            {"page_num": 4, "created_sec": 4, "text": "同文重复"},
        ]
        deck = dedupe_pages(pages)
        # dhash 对折叠 1+2；3/4 无身份键走归一化文本键折叠 3+4。跨键空间
        # （dhash 组 vs 文本组）不互并=保守语义：宁漏去重不误并内容。
        assert len(deck) == 2
        assert deck[0]["repeat_count"] == 1
        assert deck[1]["repeat_count"] == 1

    def test_identity_less_pages_are_kept_never_dropped(self):
        # 无任何身份键的页按独立页保留（去重绝不丢内容页）。
        deck = dedupe_pages([{"page_num": 1, "created_sec": 0, "text": ""}])
        assert len(deck) == 1

    def test_non_dict_rows_and_caps(self):
        pages = [{"page_num": 1, "created_sec": 5, "text": "长" * (PAGE_TEXT_CAP + 50), "source_sha256": "b" * 64}, None]
        deck = dedupe_pages(pages)
        assert len(deck) == 1, "畸形行按垃圾丢弃"
        assert len(deck[0]["text"]) == PAGE_TEXT_CAP


class TestExtractFormulaLines:
    def test_math_symbols_and_equations_detected(self):
        text = "集电结电流\nI_C = I_{NC} + I_{CBO}\nV_CE ≈ 0.2 V\n这是普通文字行"
        lines = extract_formula_lines(text)
        assert any("I_C" in line for line in lines)
        assert any("V_CE" in line for line in lines)
        assert not any("普通文字" in line for line in lines)

    def test_latex_command_lines_detected(self):
        assert extract_formula_lines("\\frac{dV}{dx} 微分")[0].startswith("\\frac")

    def test_long_lines_capped_out(self):
        assert extract_formula_lines("a = b " * 80) == []

    def test_empty_text(self):
        assert extract_formula_lines("") == []
        assert extract_formula_lines(None) == []


class TestBuildSlidePack:
    def test_pack_shape_and_formula_sheet_dedupe(self):
        pages = [
            {"page_num": 1, "created_sec": 10, "text": "I_C = I_CBO\nV_CE ≈ 0.2", "source_sha256": "a" * 64},
            {"page_num": 2, "created_sec": 30, "text": "I_C = I_CBO（重复公式）", "source_sha256": "b" * 64},
        ]
        pack = build_slide_pack(pages)
        assert pack["deck_pages"] == 2
        assert pack["raw_frames"] == 2
        assert pack["formulas"].count("I_C = I_CBO") == 1
        assert pack["deck"][0]["formulas"]

    def test_empty_or_junk_only_returns_none(self):
        assert build_slide_pack([]) is None
        assert build_slide_pack(None) is None
        assert build_slide_pack([{"page_num": 1, "created_sec": 0, "text": ""}]) is None

    def test_size_caps(self):
        pages = [
            {"page_num": i + 1, "created_sec": i * 10, "text": f"a = {i}", "source_sha256": f"{i:064x}"}
            for i in range(MAX_DECK_PAGES + 30)
        ]
        pack = build_slide_pack(pages, max_formulas=MAX_FORMULAS)
        assert pack["deck_pages"] == MAX_DECK_PAGES
        assert len(pack["formulas"]) == MAX_FORMULAS


# ---- 面2：对齐（窗口分配/近邻兜底/fail-closed） ---------------------------

class TestAssignPackWindows:
    def _pack(self):
        return build_slide_pack([
            {"page_num": 1, "created_sec": 1, "text": "p1 a=1", "source_sha256": "1" * 64},
            {"page_num": 2, "created_sec": 2, "text": "p2 b=2", "source_sha256": "2" * 64},
            {"page_num": 3, "created_sec": 5, "text": "p3 c=3", "source_sha256": "3" * 64},
        ])

    def test_containment_assignment(self):
        slices = assign_pack_windows(self._pack(), [(0, 1500), (1500, 3000), (4000, 6000)])
        assert [s["deck"][0]["page_num"] for s in slices] == [1, 2, 3]

    def test_out_of_range_page_nearest_window_and_never_lost(self):
        # 0 秒页与讲末尾页都归最近窗；内容零丢失（raw 帧按窗过滤会丢边界页，
        # 课件源刻意不同）。
        slices = assign_pack_windows(self._pack(), [(1500, 3000), (3000, 4000)])
        assigned = {page["page_num"] for s in slices if s for page in s["deck"]}
        assert assigned == {1, 2, 3}
        # 1s 页归窗0（距 1500 近于 4000）；5s 页归窗1（距 4000 近于 1500）。
        assert slices[0]["deck"][0]["page_num"] == 1
        assert slices[1]["deck"][0]["page_num"] == 3

    def test_empty_window_gets_none_not_fake_data(self):
        slices = assign_pack_windows(self._pack(), [(500, 1500), (1600, 2400)])
        assert slices[0]["deck"][0]["page_num"] == 1
        assert slices[1]["deck"][0]["page_num"] == 2

    def test_fail_closed_none_pack_and_empty_deck(self):
        assert assign_pack_windows(None, [(0, 1000)]) == [None]
        assert assign_pack_windows({"deck": []}, [(0, 1000)]) == [None]
        assert assign_pack_windows(self._pack(), []) == []

    def test_window_formulas_deduped_in_order(self):
        slices = assign_pack_windows(self._pack(), [(0, 6000)])
        assert slices[0]["formulas"] == ["p1 a=1", "p2 b=2", "p3 c=3"]


# ---- 面3：fail-closed 回退与消费面接线（wire 级） -------------------------

def _chat_responder(responses):
    index = {"n": 0}

    def _chat(api_key, messages, **kwargs):
        position = index["n"]
        index["n"] += 1
        return json.dumps(responses[min(position, len(responses) - 1)])

    return _chat


def test_slide_pack_none_keeps_legacy_wire_byte_identical():
    """slide_pack=None：窗口载荷无 slide 键、提示词无 slide 后缀、合并输入无
    slide_formulas——对照组语义（试点前后可比性的根基）。"""
    merge_payload = {"markdown": "# 笔记", "chapters": []}
    captured = []

    def _recording_chat(api_key, messages, **kwargs):
        captured.append(messages)
        return json.dumps(merge_payload if len(messages[0]["content"]) > 500 else _part("part"))

    with patch("courselens_worker.llm._chat", _recording_chat):
        create_summary("k", title="t", transcript=_transcript(4), ppt_pages=[
            {"page_num": 1, "created_sec": 0, "text": "slide a=1"},
        ])
    window_messages = [m for m in captured if '"transcript"' in m[1]["content"]]
    assert window_messages, "应有窗口调用"
    for messages in window_messages:
        payload = json.loads(messages[1]["content"])
        assert "slide_formulas" not in payload
        assert payload["ppt_pages"] == [{"page_num": 1, "created_sec": 0, "text": "slide a=1"}]
        assert _WINDOW_SLIDE_SUFFIX not in messages[0]["content"]
    merge = [m for m in captured if m[0]["content"] == _SUMMARY_MERGE_PROMPT_TEACHING]
    assert merge, "合并调用应走旧 TEACHING 串"
    assert "slide_formulas" not in json.loads(merge[0][1]["content"])


def test_slide_pack_present_injects_deck_formulas_and_suffixes():
    """slide_pack 在场：窗口 ppt_pages 换去重对齐 deck、附 slide_formulas、
    窗口/合并提示词携 slide 后缀、合并输入携讲级公式单（帽内）。"""
    pack = build_slide_pack([
        {"page_num": 1, "created_sec": 0, "text": "I_C = I_CBO", "source_sha256": "a" * 64},
        {"page_num": 1, "created_sec": 900, "text": "I_C = I_CBO", "source_sha256": "a" * 64},
    ])
    assert pack is not None and pack["deck_pages"] == 1
    merge_payload = {"markdown": "# 笔记", "chapters": [], "key_takeaways": ["一", "二", "三"]}
    captured = []

    def _recording_chat(api_key, messages, **kwargs):
        captured.append(messages)
        system = messages[0]["content"]
        if "key_takeaways" in system:
            return json.dumps(merge_payload)
        return json.dumps(_part("part"))

    with patch("courselens_worker.llm._chat", _recording_chat):
        value = create_summary(
            "k", title="t", transcript=_transcript(4), ppt_pages=[
                {"page_num": 1, "created_sec": 0, "text": "I_C = I_CBO", "source_sha256": "a" * 64},
                {"page_num": 1, "created_sec": 900, "text": "I_C = I_CBO", "source_sha256": "a" * 64},
            ],
            slide_pack=pack,
        )
    window_messages = [m for m in captured if '"transcript"' in m[1]["content"]]
    assert window_messages
    payload = json.loads(window_messages[0][1]["content"])
    assert len(payload["ppt_pages"]) == 1, "重复帧应被去重 deck 替代"
    assert payload["ppt_pages"][0]["created_sec"] == 0
    assert payload["slide_formulas"] == ["I_C = I_CBO"]
    assert _WINDOW_SLIDE_SUFFIX in window_messages[0][0]["content"]
    assert _WINDOW_TEACHING_SUFFIX in window_messages[0][0]["content"]
    merge_messages = [m for m in captured if "key_takeaways" in m[0]["content"]]
    assert merge_messages
    assert _MERGE_SLIDE_RULES in merge_messages[0][0]["content"]
    assert json.loads(merge_messages[0][1]["content"])["slide_formulas"] == ["I_C = I_CBO"]
    # 输出合同零变化（锚白名单仍从全量 ppt_pages 计算，重复帧时间戳仍合法）。
    assert value["markdown"] == "# 笔记"


def test_slide_pack_with_empty_deck_falls_back_everywhere():
    """退化包（deck 空）：逐位旧行为——None 等价（fail-closed 不造空源）。"""
    captured = []

    def _recording_chat(api_key, messages, **kwargs):
        captured.append(messages)
        system = messages[0]["content"]
        if "key_takeaways" in system:
            return json.dumps({"markdown": "# 笔记", "chapters": [], "key_takeaways": ["一", "二", "三"]})
        return json.dumps(_part("part"))

    with patch("courselens_worker.llm._chat", _recording_chat):
        create_summary("k", title="t", transcript=_transcript(2), ppt_pages=[
            {"page_num": 1, "created_sec": 0, "text": "slide x=1"},
        ], slide_pack={"deck": [], "formulas": []})
    window_messages = [m for m in captured if '"transcript"' in m[1]["content"]]
    payload = json.loads(window_messages[0][1]["content"])
    assert payload["ppt_pages"] == [{"page_num": 1, "created_sec": 0, "text": "slide x=1"}]
    assert "slide_formulas" not in payload
    assert _WINDOW_SLIDE_SUFFIX not in window_messages[0][0]["content"]


def test_prompt_variant_family_pins():
    """既有防膨胀钉家族零触：基础串/后缀变体的 startswith/endswith 关系。"""
    assert len(_SUMMARY_MERGE_PROMPT) <= 300
    assert _SUMMARY_MERGE_PROMPT_TEACHING_SLIDE.startswith(_SUMMARY_MERGE_PROMPT_TEACHING)
    assert _SUMMARY_MERGE_PROMPT_TEACHING_SLIDE.endswith(_MERGE_SLIDE_RULES)
    assert _WINDOW_SLIDE_SUFFIX
    assert FORMULA_LINE_CAP == 160


# ---- 面4：用户导入材料第二源（V010-SHUJI-SLIDES-1 S2） ---------------------

class TestDocumentPagesToPptPages:
    def _doc_page(self, num, text, start_ms=0, sha=None, visual=None):
        row = {"page_num": num, "text": text}
        if start_ms is not None:
            row["start_ms"] = start_ms
        if sha is not None:
            row["text_hash"] = sha
        if visual is not None:
            row["visual_hash"] = visual
        return row

    def test_aligned_pages_map_created_sec_and_identity(self):
        from courselens_worker.slide_alignment import document_pages_to_ppt_pages

        pages = document_pages_to_ppt_pages([
            self._doc_page(1, "阈值电压\nV_TH = V_FB + 2φ_F", start_ms=229_432, sha="a" * 64),
            self._doc_page(2, "电流方程 I_D = ½μC(W/L)(V_GS−V_TH)²", start_ms=450_610, sha="b" * 64),
        ])
        assert pages[0]["created_sec"] == 229
        assert pages[1]["created_sec"] == 450
        assert pages[0]["source_sha256"] == "a" * 64
        assert pages[0]["text"].startswith("阈值电压")

    def test_missing_or_malformed_start_ms_falls_to_zero(self):
        from courselens_worker.slide_alignment import (
            assign_pack_windows,
            build_slide_pack,
            document_pages_to_ppt_pages,
        )

        pages = document_pages_to_ppt_pages([
            self._doc_page(1, "无对齐页", start_ms=None),
            {"page_num": 2, "text": "畸形时间", "start_ms": "not-a-number"},
            {"page_num": 3, "text": "负时间", "start_ms": -50},
        ])
        assert [page["created_sec"] for page in pages] == [0, 0, 0]
        # 0 秒页按 assign_pack_windows 近邻兜底归窗，内容不丢。
        slices = assign_pack_windows(build_slide_pack(pages), [(100_000, 200_000)])
        assert {page["page_num"] for page in slices[0]["deck"]} == {1, 2, 3}

    def test_non_dict_and_empty_text_rows_dropped(self):
        from courselens_worker.slide_alignment import document_pages_to_ppt_pages

        pages = document_pages_to_ppt_pages([
            None,
            "junk",
            self._doc_page(1, ""),
            self._doc_page(2, "   "),
            self._doc_page(3, "有效页"),
        ])
        assert len(pages) == 1
        assert pages[0]["page_num"] == 3

    def test_visual_hash_fallback_and_str_page_num(self):
        from courselens_worker.slide_alignment import document_pages_to_ppt_pages

        pages = document_pages_to_ppt_pages([
            {"page_num": "7", "text": "仅有视觉哈希", "visual_hash": "c" * 32},
        ])
        assert pages[0]["page_num"] == 7
        assert pages[0]["dhash"] == "c" * 32
        assert pages[0]["source_sha256"] == ""

    def test_empty_input_fail_closed(self):
        from courselens_worker.slide_alignment import document_pages_to_ppt_pages

        assert document_pages_to_ppt_pages(None) == []
        assert document_pages_to_ppt_pages([]) == []
        assert document_pages_to_ppt_pages([None, {"page_num": 1, "text": ""}]) == []


class TestBuildDocumentSlidePack:
    def _documents(self):
        return [
            {"page_num": 1, "text": "I_C = I_CBO", "text_hash": "a" * 64, "start_ms": 10_000},
            {"page_num": 2, "text": "I_C = I_CBO（重复）", "text_hash": "b" * 64, "start_ms": 30_000},
            {"page_num": 3, "text": "", "text_hash": "c" * 64, "start_ms": 40_000},
        ]

    def test_pack_from_document_pages(self):
        from courselens_worker.slide_alignment import build_document_slide_pack

        pack = build_document_slide_pack(self._documents())
        assert pack is not None
        assert pack["deck_pages"] == 2
        # raw_frames=变换后行数：空文本页在 document_pages_to_ppt_pages 的
        # fail-closed 层已被丢弃，不进入 build_slide_pack 输入。
        assert pack["raw_frames"] == 2
        assert pack["formulas"].count("I_C = I_CBO") == 1
        assert pack["deck"][0]["formulas"] == ["I_C = I_CBO"]

    def test_same_text_identity_dedupes_across_pages(self):
        from courselens_worker.slide_alignment import build_document_slide_pack

        pack = build_document_slide_pack([
            {"page_num": 1, "text": "同文页", "text_hash": "d" * 64, "start_ms": 1_000},
            {"page_num": 2, "text": "同文页", "text_hash": "d" * 64, "start_ms": 2_000},
        ])
        assert pack["deck_pages"] == 1
        assert pack["deck"][0]["repeat_count"] == 1

    def test_fail_closed_none(self):
        from courselens_worker.slide_alignment import build_document_slide_pack

        assert build_document_slide_pack(None) is None
        assert build_document_slide_pack([]) is None
        assert build_document_slide_pack([{"page_num": 1, "text": ""}]) is None

    def test_end_to_end_windows_from_aligned_documents(self):
        from courselens_worker.slide_alignment import (
            assign_pack_windows,
            build_document_slide_pack,
        )

        pack = build_document_slide_pack(self._documents())
        slices = assign_pack_windows(pack, [(0, 20_000), (20_000, 60_000)])
        assert slices[0]["deck"][0]["page_num"] == 1
        assert slices[1]["deck"][0]["page_num"] == 2
        assert slices[0]["formulas"] == ["I_C = I_CBO"]


if __name__ == "__main__":
    # pytest 仅直跑入口需要（镜像 CI=unittest 发现面无 pytest，顶层导入会
    # 炸整模块收集：r14 环 CI 实证，r15 环随形修——测试体纯 assert 零依赖）。
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
