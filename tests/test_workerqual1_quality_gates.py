"""V010-WORKER-QUAL-1 钉测（2026-10-26，用户令「总结文字质量」七项根修）。

CONTENT-QUAL-1 全库取证七项（W1 latex 残渣/W2 caption 占位/W3 术语讲内混用/
W4 要点页码噪声/W5 元话语泄漏/W6 doc terms 全空/W7 相邻近重复章节）的
worker 侧根修钉面，全部离线零活体：

①守卫函数钉——sane_latex/sane_formula_caption/is_noise_point_text/
  strip_meta_discourse/merge_near_duplicate_titles/unify_text_terms 六守卫
  闭集（判据=验收副本实测样本原形）；
②提示词钉——判定内部化（W5 反转钉）/terms 产出指令（W6）/caption 指导+
  占位禁令（W2）/术语统一指令（W3）/要点噪声约束（W4）；基串 300 防膨胀钉
  与变体组合链原样；
③链路钉——create_summary 桩回放（零 LLM）：缺陷形合并产物被逐项拦截，
  合法内容与 terms 透传零误伤；slide 公式单输入过滤（W1 源头）。
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from courselens_worker import llm
from courselens_worker.glossary import unify_text_terms
from courselens_worker.llm import (
    _MERGE_SLIDE_RULES,
    _STEM_TEACHING_RULES,
    _SUMMARY_DOC_RULES,
    _SUMMARY_MERGE_PROMPT,
    _SUMMARY_MERGE_PROMPT_TEACHING,
    create_summary,
)
from courselens_worker.slide_alignment import extract_formula_lines
from courselens_worker.summary_contract import (
    coerce_llm_doc,
    is_noise_point_text,
    merge_near_duplicate_titles,
    sane_formula_caption,
    sane_latex,
    strip_introspective_parens,
    strip_meta_discourse,
    validate_summary_doc,
)

ENV_PINS = {llm.SUMMARY_THINKING_ENV: "", llm.REVIEW_VIEWS_ENV: "off"}


class SaneLatexGatePins(unittest.TestCase):
    """W1 公式残渣门：680954 实测坏式形态逐类拦截，合法式零误伤。"""

    def test_measured_residue_shapes_rejected(self):
        # 原文照录自 680954 artifact（CONTENT-QUAL-1 取证）。
        for bad in (
            "nco =",                      # 悬垂尾等号
            "dI, =[Je(y)-J(y)]Le\\cdot dy=",  # 悬垂尾等号（长式）
            "IB==70\\mu A",               # 双等号
            "\\beta不宜太高",              # 中文入式
            "理想BVcbo=VBR(纯pn结)",       # 中文入式
            "Icbo =(1 +\\beta)Icbo",      # 自指（左式标识符在右式原样再现）
            "Iceo =\\alpha Iceo + I cbo",  # 自指
        ):
            self.assertFalse(sane_latex(bad), bad)

    def test_legitimate_formulas_pass(self):
        for good in (
            "f_T = 1 / (2\\pi \\tau_F)",
            "I_C = I_B + I_CEO",          # I_CEO 含 I_C 前缀但标识符不等=非自指
            "E=mc^2",
            "V=IR",
            "v = v_0 + at",               # 单字符左式不参与自指判定
            "\\frac{dV}{dx}",
        ):
            self.assertTrue(sane_latex(good), good)

    def test_empty_latex_rejected(self):
        self.assertFalse(sane_latex(""))
        self.assertFalse(sane_latex(None))

    def test_shared_gate_drops_entries_without_killing_section(self):
        # 共享门（validate_summary_doc）双路径守卫：坏式丢条、段级零核爆。
        doc = validate_summary_doc({
            "overview": "总述",
            "sections": [{
                "start_ms": 0,
                "title": "板块",
                "source": "semantic",
                "points": [{"text": "要点", "anchor_ms": 0}],
                "formulas": [
                    {"latex": "nco =", "caption": "课件公式行"},
                    {"latex": "f_T = 1/(2\\pi\\tau_F)", "caption": "特征频率"},
                ],
            }],
            "terms": [],
            "summary_notes": ["小结"],
        })
        self.assertIsNotNone(doc)
        formulas = doc["sections"][0]["formulas"]
        self.assertEqual([f["latex"] for f in formulas], ["f_T = 1/(2\\pi\\tau_F)"])


class CaptionPlaceholderPins(unittest.TestCase):
    """W2 caption 占位过滤：680954 实测 39/39 全占位形态。"""

    def test_measured_placeholder_rejected(self):
        self.assertFalse(sane_formula_caption("课件公式行"))
        self.assertFalse(sane_formula_caption(" 课件公式行 "))
        self.assertFalse(sane_formula_caption("课件公式行 3"))
        self.assertFalse(sane_formula_caption("公式"))
        self.assertFalse(sane_formula_caption(""))

    def test_meaningful_captions_pass(self):
        for good in ("特征频率", "发射效率", "基区渡越时间", "理想BVcbo=VBR(纯pn结)"):
            self.assertTrue(sane_formula_caption(good), good)

    def test_coerce_drops_placeholder_keeps_latex_with_counter(self):
        stats: dict[str, int] = {}
        doc = coerce_llm_doc({
            "overview": "总述",
            "sections": [{
                "title": "板块", "start_ms": 0, "source": "semantic",
                "points": [{"text": "要点", "anchor_ms": 0}],
                "formulas": [
                    {"latex": "nco =", "anchor_ms": 0, "caption": "课件公式行"},
                    {"latex": "V=IR", "anchor_ms": 0, "caption": "课件公式行"},
                    {"latex": "P=UI", "anchor_ms": 0, "caption": "功率"},
                ],
            }],
        }, {0}, anchor_text_pool=None, drop_stats=stats)
        self.assertIsNotNone(doc)
        formulas = doc["sections"][0]["formulas"]
        # nco = 被残渣门丢条；占位 caption 丢键保 latex。
        self.assertEqual(
            formulas,
            [{"latex": "V=IR", "anchor_ms": 0}, {"latex": "P=UI", "anchor_ms": 0,
                                                 "caption": "功率"}],
        )
        self.assertEqual(stats.get("formulas_dropped_unsane"), 1)
        self.assertEqual(stats.get("captions_dropped_placeholder"), 1)


class NoisePointPins(unittest.TestCase):
    """W4 要点页码噪声：CONTENT-QUAL-1 实样原形。"""

    def test_measured_noise_shapes_detected(self):
        for noise in (
            "第 5 页标题：2.3.5 反向恢复过程",
            "p688–p689 为第二章目录",
            "第 12 页",
            "第 3 页目录：",
        ):
            self.assertTrue(is_noise_point_text(noise), noise)

    def test_conclusion_points_kept(self):
        for keep in (
            "特征频率由基区渡越时间限制。",
            "发射效率决定电流增益的上限。",
            "第 5 页讲到的反向恢复过程制约开关速度。",  # 页码只是从句→不误删
        ):
            self.assertFalse(is_noise_point_text(keep), keep)

    def test_coerce_drops_noise_points_with_counter(self):
        stats: dict[str, int] = {}
        doc = coerce_llm_doc({
            "overview": "总述",
            "sections": [{
                "title": "板块", "start_ms": 0, "source": "semantic",
                "points": [
                    {"text": "特征频率由基区渡越时间限制。", "anchor_ms": 0},
                    {"text": "第 5 页标题：2.3.5 反向恢复过程", "anchor_ms": 0},
                    {"text": "p688–p689 为第二章目录", "anchor_ms": 0},
                ],
                "formulas": [],
            }],
        }, {0}, anchor_text_pool=None, drop_stats=stats)
        self.assertIsNotNone(doc)
        points = doc["sections"][0]["points"]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["text"], "特征频率由基区渡越时间限制。")
        self.assertEqual(stats.get("points_dropped_noise"), 2)


class MetaDiscoursePins(unittest.TestCase):
    """W5 元话语剥除：27738 实测「判定：…；证据：…」概览开头直出形态。"""

    def test_heading_parenthetical_stripped(self):
        md = "# 第3讲 PN 结与能带（判定：推导型理工；证据：讲次标题与推导）\n\n## 机制\n正文。"
        cleaned = strip_meta_discourse(md)
        self.assertNotIn("判定：", cleaned)
        self.assertNotIn("证据：", cleaned)
        self.assertIn("# 第3讲 PN 结与能带", cleaned)
        self.assertIn("## 机制", cleaned)

    def test_standalone_meta_line_dropped(self):
        md = "判定：推导型理工内容；证据：分波法推导。\n\n本讲内容从驻波出发。"
        cleaned = strip_meta_discourse(md)
        self.assertNotIn("判定：", cleaned)
        self.assertIn("本讲内容从驻波出发。", cleaned)

    def test_normal_content_untouched(self):
        md = "## 适用条件与反例\n重掺杂简并半导体不再适用；判定标准是简并条件。"
        self.assertEqual(strip_meta_discourse(md), md)

    def test_body_after_first_line_untouched(self):
        # 只剥首行：正文正常讨论不误伤（闭集限定）。
        md = "# 笔记\n正文提到判定：需要小心；证据：见下。"
        self.assertEqual(strip_meta_discourse(md), md)

    def test_coerce_strips_overview_meta(self):
        # 混形 overview：元话语前缀剥除、同句正文保留。
        doc = coerce_llm_doc({
            "overview": "判定：推导型理工内容；证据：分波法推导。本讲推导高频响应。",
            "sections": [{
                "title": "板块", "start_ms": 0, "source": "semantic",
                "points": [{"text": "要点", "anchor_ms": 0}],
                "formulas": [],
            }],
        }, {0})
        self.assertIsNotNone(doc)
        overview = str(doc["overview"])
        self.assertNotIn("判定：", overview)
        self.assertIn("本讲推导高频响应。", overview)

    # ---- V010-REPROC-QUALFOLLOWUP-1 扩边钉（2026-10-26 六讲重处理实录归纳）----
    # 六讲存量产物全部带「判定：…」头部残留=strip 闭集缺形四/五+仅首行扫描；
    # 以下钉用脱敏合成形逐一逼真实测形态，并钉行界负例（正文讨论零误伤）。

    def test_keyword_paren_form_stripped(self):
        # 形四a：判定：短词（证据：…）。整行独立段（653998 实录形）。
        md = "判定：推导型（证据：公式推导与器件结构）。\n\n# 课程笔记标题\n\n正文。"
        cleaned = strip_meta_discourse(md)
        self.assertNotIn("判定：", cleaned)
        self.assertNotIn("证据：", cleaned)
        self.assertIn("# 课程笔记标题", cleaned)
        self.assertIn("正文。", cleaned)

    def test_keyword_paren_without_evidence_marker_stripped(self):
        # 形四b：括注内无「证据」标记（669766 实录形）。
        md = "判定：推导型（电流推导与器件结构）。\n\n# 课程笔记标题"
        cleaned = strip_meta_discourse(md)
        self.assertNotIn("判定：", cleaned)
        self.assertIn("# 课程笔记标题", cleaned)

    def test_keyword_paren_semicolon_evidence_inside_paren_stripped(self):
        # 形四c：括注内分号配对（665055 实录形）。
        md = "判定：推导型（开关过程推导；证据：课件公式与推导链）。\n\n# 课程笔记标题"
        cleaned = strip_meta_discourse(md)
        self.assertNotIn("判定：", cleaned)
        self.assertIn("# 课程笔记标题", cleaned)

    def test_keyword_paren_quote_prefixed_line_stripped(self):
        # 形四+行界扩边：元话语落课程头之后的引用行（659504 实录形，剥除窗
        # 头 6 行内锚定形）。
        md = (
            "# 课程笔记标题\n\n"
            "> 判定：本讲为推导型理工内容（依据标题与公式推导链）\n\n"
            "正文第一段。"
        )
        cleaned = strip_meta_discourse(md)
        self.assertNotIn("判定：", cleaned)
        self.assertIn("# 课程笔记标题", cleaned)
        self.assertIn("正文第一段。", cleaned)

    def test_judgment_sentence_plus_evidence_sentence_stripped(self):
        # 形五：判定句+证据句连写（685785/680954 存量实录形）。
        md = "判定：推导型（高频小信号推导）。证据：讲次标题与课件公式行。\n\n# 课程笔记标题"
        cleaned = strip_meta_discourse(md)
        self.assertNotIn("判定：", cleaned)
        self.assertNotIn("证据：", cleaned)
        self.assertIn("# 课程笔记标题", cleaned)

    def test_meta_prefix_with_trailing_body_keeps_body(self):
        # 前缀形后接正文：只剥元话语段留正文（形四/五前缀变体）。
        md = "判定：推导型（结构梳理）。 本讲梳理器件结构。"
        self.assertEqual(strip_meta_discourse(md), "本讲梳理器件结构。")
        md2 = "判定：推导型（电流推导）。证据：课件公式。 本讲从频率特性出发。"
        self.assertEqual(strip_meta_discourse(md2), "本讲从频率特性出发。")

    def test_all_observed_forms_leave_zero_residue(self):
        # 防复发总钉：六讲实录形态合铸一篇，剥后头部判定/证据零残留。
        md = "\n".join([
            "判定：推导型（证据：公式推导与器件结构）。",
            "",
            "# 课程笔记标题",
            "",
            "> 判定：本讲为推导型理工内容（依据标题与推导链）",
            "判定：推导型（高频小信号推导）。证据：讲次标题与课件公式行。",
            "正文第一段，正常讨论载流子浓度。判定标准另节展开。",
        ])
        cleaned = strip_meta_discourse(md)
        self.assertNotIn("判定：", cleaned)
        self.assertNotIn("证据：", cleaned)
        self.assertIn("# 课程笔记标题", cleaned)
        self.assertIn("正文第一段，正常讨论载流子浓度。判定标准另节展开。", cleaned)

    def test_glued_body_after_paren_untouched(self):
        # 黏连无句号界定：正文讨论「判定」零误伤（保守闭集纪律）。
        md = "判定：三段论（逻辑学推理）是本讲方法论要点。"
        self.assertEqual(strip_meta_discourse(md), md)

    def test_body_line_beyond_first_line_untouched_by_loose_form(self):
        # 行界纪律：行 2+ 只接锚定形——正文句中提及（非行首）零误伤。
        md = "# 课程笔记标题\n重掺杂半导体的判定：需要小心；证据：见下节。"
        self.assertEqual(strip_meta_discourse(md), md)

    def test_window_bounded_at_head_lines(self):
        # 剥除窗=头 6 行：窗外（第 9 行）判定头不动（窗口闭集边界钉）。
        md = "\n".join([
            "# 课程笔记标题", "", "引言。", "", "第一节。", "", "更多。", "",
            "判定：推导型（公式推导）。",
        ])
        self.assertEqual(strip_meta_discourse(md), md)


class F9Fix1LeakStripPins(unittest.TestCase):
    """F9-FIX-1 包①（判定头/内省泄漏上屏）钉面（2026-10-29）。

    判据=F9-1-R2 评判道 44 讲全量重扫实录（7/44 讲判定头泄漏，两分支皆漏；
    内省括注 1 讲）：全部实录形经 strip_meta_discourse 五形态闭集剥净（本道
    定谳=存量泄漏为 WORKER-QUAL-1 17419761 之前的旧代产物，管线根因已修）。
    本类把实测形态转产品侧钉测（防复发），并落新增内省括注剥除规则与其
    阴性样本边界（编辑性括注零误伤）。"""

    NL = "\n"

    def _md(self, *segments: str) -> str:
        return self.NL.join(segments)

    # ---- 判定头实录形（脱敏改写，形态逐位对齐评判道 scan-full.json 证据）----

    def test_derived_branch_paren_evidence_head_stripped(self):
        # 452291/658063 实录形：推导型（括注）。证据：…。（两段连写首行）
        md = (
            "判定：推导型（量子力学微扰论、斯塔克效应、精细结构）。"
            "证据：标题与内容含微扰、氢原子能级、选择定则等公式推导。"
        ) + "\n\n# 课程笔记标题\n\n正文第一段。"
        cleaned = strip_meta_discourse(md)
        self.assertNotIn("判定：", cleaned)
        self.assertNotIn("证据：", cleaned)
        self.assertIn("# 课程笔记标题", cleaned)
        self.assertIn("正文第一段。", cleaned)

    def test_non_derived_branch_paren_head_stripped(self):
        # 669510/663596 实录形：非推导型（括注）。无证据段变体。
        md = (
            "判定：非推导型（课程安排+概念梳理，无公式推导链证据）。"
        ) + "\n\n## 课程安排与作业说明\n\n- 本讲共四次作业。"
        cleaned = strip_meta_discourse(md)
        self.assertNotIn("判定：", cleaned)
        self.assertIn("## 课程安排与作业说明", cleaned)
        self.assertIn("- 本讲共四次作业。", cleaned)

    def test_semicolon_paired_head_stripped(self):
        # 452296/679491 实录形：判定：推导型理工内容；证据：…。（分号配对形）
        md = (
            "判定：推导型理工内容；证据：分波法、玻恩近似、格林函数与氢原子能级推导。"
        ) + "\n\n# 课程笔记标题"
        cleaned = strip_meta_discourse(md)
        self.assertNotIn("判定：", cleaned)
        self.assertNotIn("证据：", cleaned)
        self.assertIn("# 课程笔记标题", cleaned)

    def test_all_f9_observed_heads_zero_residue(self):
        # 防复发总钉：F9 实测四形合铸一篇，剥后头部判定/证据零残留。
        md = self._md(
            "判定：推导型（量子力学微扰论）。证据：标题与内容含公式推导。",
            "判定：非推导型（课程安排+概念梳理）。",
            "判定：推导型理工内容；证据：分波法与玻恩近似推导。",
            "",
            "# 课程笔记标题",
            "",
            "正文第一段。",
        )
        cleaned = strip_meta_discourse(md)
        self.assertNotIn("判定：", cleaned)
        self.assertNotIn("证据：", cleaned)
        self.assertIn("# 课程笔记标题", cleaned)
        self.assertIn("正文第一段。", cleaned)

    # ---- F9-FIX-1 新增：内省括注剥除（452284 实录形）----

    def test_introspective_paren_stripped(self):
        md = (
            "## 课程回顾\n\n"
            "- 上讲引入了一维无限深势阱（此处有口语化表达，未形成完整结论）。\n\n"
            "正文继续。"
        )
        cleaned = strip_introspective_parens(md)
        self.assertNotIn("未形成完整结论", cleaned)
        self.assertIn("- 上讲引入了一维无限深势阱。", cleaned)
        self.assertIn("正文继续。", cleaned)

    def test_introspective_paren_keyword_closed_set(self):
        # 闭集四关键词逐一并钉（口语/未形成/未展开/听不清）。
        for keyword in ("口语化表达，", "未形成完整结论", "未展开", "听不清"):
            md = "要点（此处有" + keyword + "）。"
            cleaned = strip_introspective_parens(md)
            self.assertNotIn("此处有", cleaned, keyword)
            self.assertTrue(cleaned.startswith("要点"), keyword)
            self.assertTrue(cleaned.endswith("。"), keyword)

    def test_editorial_clarification_parens_untouched(self):
        # 阴性样本：编辑性澄清括注（F9 归打磨=内容有益）零误伤。
        md = "波形在上升沿翻转（此处指上升沿），下一点讨论采样。"
        self.assertEqual(strip_introspective_parens(md), md)
        md2 = "本节讨论判定问题（此处有详细推导）。"
        self.assertEqual(strip_introspective_parens(md2), md2)

    def test_finalize_pipeline_composition_strips_both_families(self):
        # 定稿链组合钉：判定头（strip_meta_discourse）+内省括注同串不漏。
        md = (
            "判定：推导型（公式推导）。\n\n"
            "# 课程笔记标题\n\n"
            "要点句（此处有口语化表达，未形成完整结论）后续正文。"
        )
        cleaned = strip_introspective_parens(strip_meta_discourse(md))
        self.assertNotIn("判定：", cleaned)
        self.assertNotIn("未形成完整结论", cleaned)
        self.assertIn("# 课程笔记标题", cleaned)
        self.assertEqual(cleaned.splitlines()[-1], "要点句后续正文。")

    def test_doc_overview_introspective_stripped(self):
        # doc overview 继承同款剥除（coerce_llm_doc 路径同权）。
        doc = coerce_llm_doc({
            "overview": "总述（此处有口语化表达，未形成完整结论）。本讲梳理能带。",
            "sections": [{
                "title": "板块", "start_ms": 0, "source": "semantic",
                "points": [{"text": "要点", "anchor_ms": 0}],
                "formulas": [],
            }],
        }, {0})
        self.assertIsNotNone(doc)
        overview = str(doc["overview"])
        self.assertNotIn("未形成完整结论", overview)
        self.assertIn("本讲梳理能带。", overview)


class TermsContractPins(unittest.TestCase):
    """W6 doc terms：提示词产出指令在场 + 校验器透传。"""

    def test_doc_rules_carry_terms_instruction(self):
        self.assertIn("terms", _SUMMARY_DOC_RULES)
        self.assertIn("name", _SUMMARY_DOC_RULES)
        self.assertIn("symbol", _SUMMARY_DOC_RULES)
        self.assertIn("note", _SUMMARY_DOC_RULES)

    def test_coerce_passes_terms_through(self):
        doc = coerce_llm_doc({
            "overview": "总述",
            "sections": [{
                "title": "板块", "start_ms": 0, "source": "semantic",
                "points": [{"text": "要点", "anchor_ms": 0}],
                "formulas": [],
            }],
            "terms": [
                {"name": "特征频率", "symbol": "f_T", "note": "增益降为 1 的频率"},
                {"name": "厄利效应", "symbol": "", "note": "基区宽度调制"},
            ],
        }, {0})
        self.assertIsNotNone(doc)
        self.assertEqual(len(doc["terms"]), 2)
        self.assertEqual(doc["terms"][0]["name"], "特征频率")
        self.assertEqual(doc["terms"][0]["symbol"], "f_T")

    def test_terms_absent_defaults_to_honest_empty(self):
        doc = coerce_llm_doc({
            "overview": "总述",
            "sections": [{
                "title": "板块", "start_ms": 0, "source": "semantic",
                "points": [{"text": "要点", "anchor_ms": 0}],
                "formulas": [],
            }],
        }, {0})
        self.assertIsNotNone(doc)
        self.assertEqual(doc["terms"], [])


class NearDuplicateChapterPins(unittest.TestCase):
    """W7 相邻近重复章节：39178 实测连排形态。"""

    def test_measured_pair_merged(self):
        merged, count = merge_near_duplicate_titles([
            {"title": "线性电源的结构与原理", "start_ms": 437_000, "summary": "结构逐级展开。"},
            {"title": "线性电源：结构与原理", "start_ms": 471_000, "summary": "原理补充。"},
            {"title": "特征频率与截止频率", "start_ms": 3_600_000, "summary": "fT 推导。"},
        ])
        self.assertEqual(count, 1)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["title"], "线性电源的结构与原理")
        self.assertEqual(merged[0]["start_ms"], 437_000)
        self.assertIn("结构逐级展开", merged[0]["summary"])
        self.assertIn("原理补充", merged[0]["summary"])
        self.assertEqual(merged[1]["title"], "特征频率与截止频率")

    def test_distinct_titles_untouched(self):
        merged, count = merge_near_duplicate_titles([
            {"title": "PN 结的形成", "start_ms": 0, "summary": "a"},
            {"title": "PN 结的击穿", "start_ms": 300_000, "summary": "b"},
            {"title": "MOS 阈值电压", "start_ms": 600_000, "summary": "c"},
        ])
        self.assertEqual(count, 0)
        self.assertEqual(len(merged), 3)

    def test_doc_sections_near_dup_merged_in_coerce(self):
        stats: dict[str, int] = {}
        doc = coerce_llm_doc({
            "overview": "总述",
            "sections": [
                {"title": "线性电源的结构与原理", "start_ms": 437_000, "source": "semantic",
                 "points": [{"text": "要点一", "anchor_ms": 437_000}], "formulas": []},
                {"title": "线性电源：结构与原理", "start_ms": 471_000, "source": "semantic",
                 "points": [{"text": "要点二", "anchor_ms": 471_000}], "formulas": []},
                {"title": "特征频率", "start_ms": 3_600_000, "source": "semantic",
                 "points": [{"text": "要点三", "anchor_ms": 3_600_000}], "formulas": []},
            ],
        }, {437_000, 471_000, 3_600_000}, anchor_text_pool=None, drop_stats=stats)
        self.assertIsNotNone(doc)
        self.assertEqual(len(doc["sections"]), 2)
        # 列表字段拼接零删改：后段要点并入先段。
        texts = [p["text"] for p in doc["sections"][0]["points"]]
        self.assertEqual(texts, ["要点一", "要点二"])
        self.assertEqual(stats.get("sections_near_dup_merged"), 1)


class TermUnificationPins(unittest.TestCase):
    """W3 同文种变体统一器：trusted-pairing 纪律（唯一映射+出现≥2）。"""

    GLOSSARY = ("厄利效应", "基区渡越时间")

    def test_variant_unified_when_recurrent(self):
        text = "厄利效应使输出电导增大；厄早效应亦称基区宽度调制，共基组态时厄早效应同样存在。"
        unified = unify_text_terms(text, self.GLOSSARY)
        self.assertNotIn("厄早效应", unified)
        self.assertEqual(unified.count("厄利效应"), 3)

    def test_single_occurrence_left_alone(self):
        # 出现仅 1 次=证据不足，宁漏勿错删。
        text = "厄早效应亦称基区宽度调制。"
        self.assertEqual(unify_text_terms(text, self.GLOSSARY), text)

    def test_empty_glossary_unchanged(self):
        text = "厄早效应出现两次厄早效应。"
        self.assertEqual(unify_text_terms(text, ()), text)

    def test_latin_and_latex_untouched(self):
        text = "Kirk 效应见 \\(f_T\\) 推导；厄早效应两见：厄早效应、厄早效应。"
        unified = unify_text_terms(text, self.GLOSSARY)
        self.assertIn("Kirk 效应", unified)
        self.assertIn("\\(f_T\\)", unified)
        self.assertNotIn("厄早效应", unified)


class SlideFormulaInputPins(unittest.TestCase):
    """W1 输入侧守卫：显然坏式行不进模型输入。"""

    def test_dangling_and_double_equals_filtered(self):
        lines = extract_formula_lines("有效公式 I_C=I_B+I_C\nnco =\nIB==70\\mu A\nV_CE ≈ 0.2 V")
        self.assertEqual(lines, ["有效公式 I_C=I_B+I_C", "V_CE ≈ 0.2 V"])

    def test_recall_net_width_unchanged(self):
        # 召回网其余面零收紧：数学符号行/LaTeX 行照常进单。
        lines = extract_formula_lines("∫f(x)dx\n\\frac{dV}{dx} 微分\n普通文字行")
        self.assertEqual(len(lines), 2)


class PromptDirectivePins(unittest.TestCase):
    """提示词钉：W5 反转/W6 terms/W2 caption/W3 统一/W4 噪声；基串与链原样。"""

    def test_base_prompt_untouched(self):
        self.assertLessEqual(len(_SUMMARY_MERGE_PROMPT), 300)
        self.assertTrue(_SUMMARY_MERGE_PROMPT_TEACHING.startswith(_SUMMARY_MERGE_PROMPT))

    def test_meta_discourse_banned_in_prompt(self):
        self.assertIn("判定在内部完成", _STEM_TEACHING_RULES)
        self.assertIn("元话语", _STEM_TEACHING_RULES)
        self.assertNotIn("笔记首行", _STEM_TEACHING_RULES)

    def test_caption_guidance_and_placeholder_ban(self):
        self.assertIn("含义短名", _SUMMARY_DOC_RULES)
        self.assertIn("省略caption", _SUMMARY_DOC_RULES.replace("（", "").replace("）", ""))
        self.assertIn("含义短名", _MERGE_SLIDE_RULES)
        self.assertIn("课件公式行", _MERGE_SLIDE_RULES)
        self.assertIn("禁回填", _MERGE_SLIDE_RULES)

    def test_points_noise_constraint(self):
        self.assertIn("禁纯页码/目录/", _SUMMARY_DOC_RULES)

    def test_term_unification_directive(self):
        self.assertIn("同一种写法", _STEM_TEACHING_RULES)


class ChainReplayPins(unittest.TestCase):
    """链路钉（零 LLM 桩回放）：缺陷形合并产物被逐项拦截，合法内容零误伤。"""

    def _transcript(self, count: int) -> list[dict]:
        return [
            {"start_ms": i * 1000, "end_ms": (i + 1) * 1000, "text": f"文本{i}"}
            for i in range(count)
        ]

    def _run(self, merge_payload: dict) -> dict:
        window_part = {"markdown": "## 机制\n窗口内容。", "chapters": []}

        def fake_chat(api_key, messages, **kwargs):
            if messages[0]["content"].startswith(_SUMMARY_MERGE_PROMPT):
                return json.dumps(merge_payload, ensure_ascii=False)
            return json.dumps(window_part, ensure_ascii=False)

        with patch.dict(os.environ, ENV_PINS, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=fake_chat):
                return create_summary(
                    "k", title="t", transcript=self._transcript(10), ppt_pages=[],
                    glossary=("厄利效应",),
                )

    def test_defect_shaped_merge_output_is_gated(self):
        payload = {
            "markdown": (
                "# 讲次（判定：推导型理工；证据：标题与推导）\n\n"
                "厄早效应两见：厄早效应、厄早效应。"
            ),
            "chapters": [
                {"title": "线性电源的结构与原理", "start_ms": 0, "summary": "s1"},
                {"title": "线性电源：结构与原理", "start_ms": 4000, "summary": "s2"},
                {"title": "特征频率", "start_ms": 8000, "summary": "s3"},
            ],
            "key_takeaways": ["要点"],
            "doc": {
                "overview": "总述。",
                "sections": [{
                    "title": "特征频率", "start_ms": 8000, "source": "semantic",
                    "points": [
                        {"text": "特征频率由基区渡越时间限制。", "anchor_ms": 8000},
                        {"text": "第 5 页标题：2.3.5 反向恢复", "anchor_ms": 8000},
                    ],
                    "formulas": [
                        {"latex": "f_T=1/(2\\pi\\tau_F)", "anchor_ms": 8000,
                         "caption": "特征频率"},
                        {"latex": "nco =", "anchor_ms": 8000, "caption": "课件公式行"},
                    ],
                }],
                "summary_notes": ["小结"],
                "terms": [{"name": "特征频率", "symbol": "f_T", "note": "含义"}],
            },
        }
        result = self._run(payload)
        md = result["markdown"]
        self.assertNotIn("判定：", md)
        self.assertNotIn("证据：", md)
        self.assertNotIn("厄早效应", md)
        titles = [c["title"] for c in result["chapters"]]
        self.assertEqual(
            titles, ["线性电源的结构与原理", "特征频率"], titles,
        )
        doc = result["doc"]
        self.assertEqual(len(doc["terms"]), 1)
        formulas = doc["sections"][0]["formulas"]
        self.assertEqual(
            formulas,
            [{"latex": "f_T=1/(2\\pi\\tau_F)", "anchor_ms": 8000,
              "caption": "特征频率"}],
        )
        points = doc["sections"][0]["points"]
        self.assertEqual(
            [p["text"] for p in points], ["特征频率由基区渡越时间限制。"],
        )

    def test_clean_merge_output_passes_bit_for_bit(self):
        payload = {
            "markdown": "# 讲次\n特征频率由基区渡越时间限制。",
            "chapters": [
                {"title": "特征频率", "start_ms": 0, "summary": "s"},
                {"title": "截止频率", "start_ms": 4000, "summary": "s"},
                {"title": "总结", "start_ms": 8000, "summary": "s"},
            ],
            "key_takeaways": ["要点"],
            "doc": {
                "overview": "总述。",
                "sections": [{
                    "title": "特征频率", "start_ms": 8000, "source": "semantic",
                    "points": [{"text": "特征频率由基区渡越时间限制。", "anchor_ms": 8000}],
                    "formulas": [
                        {"latex": "f_T=1/(2\\pi\\tau_F)", "anchor_ms": 8000,
                         "caption": "特征频率"},
                    ],
                }],
                "summary_notes": ["小结"],
                "terms": [{"name": "特征频率", "symbol": "f_T", "note": "含义"}],
            },
        }
        result = self._run(payload)
        self.assertEqual(result["markdown"], payload["markdown"])
        self.assertEqual(len(result["chapters"]), 3)
        self.assertEqual(len(result["doc"]["terms"]), 1)
        self.assertEqual(len(result["doc"]["sections"][0]["formulas"]), 1)
        self.assertEqual(len(result["doc"]["sections"][0]["points"]), 1)


class Rxescape1FragmentFunnelPins(unittest.TestCase):
    """RXESCAPE-1（2026-10-30）：deep_usage.md_term_fragment_hits 漏斗计数钉。

    生成侧漂移挂账（SEMGEN 漏斗随形）：md 正文出现「电平性区」族碎片伪影
    （685785 实证伤型=「电中性区」被近形课件词片段改写）即计数非零=车道复
    查信号；正当使用「电平」（数字课逻辑电平）零误伤计数恒零。判据权威面=
    probe gate_F 词对族全扫，此处只锁生成侧计数器本身。"""

    def _run_md(self, markdown: str) -> dict:
        window_part = {"markdown": "## 机制\n窗口内容。", "chapters": []}

        def fake_chat(api_key, messages, **kwargs):
            if messages[0]["content"].startswith(_SUMMARY_MERGE_PROMPT):
                return json.dumps({
                    "markdown": markdown,
                    "chapters": [],
                    "key_takeaways": ["要点"],
                }, ensure_ascii=False)
            return json.dumps(window_part, ensure_ascii=False)

        with patch.dict(os.environ, ENV_PINS, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=fake_chat):
                return create_summary(
                    "k", title="t",
                    transcript=[
                        {"start_ms": i * 1000, "end_ms": (i + 1) * 1000, "text": f"文本{i}"}
                        for i in range(10)
                    ],
                    ppt_pages=[],
                )

    def test_fragment_artifact_repaired_and_counted(self):
        # TERMGEN1-1（2026-10-31）：修复门把「电平性区→电中性区」（always 对，
        # 伪词形）收口后，碎片漏斗转残余观测（修复后残留恒零），修复量改挂
        # md_term_repair_hits=生成侧漂移挂账主计数器。
        result = self._run_md(
            "耗尽层外电平性区以扩散衰减。又一处电平性区。"
        )
        self.assertEqual(result["markdown"].count("电平性区"), 0)
        self.assertIn("电中性区", result["markdown"])
        self.assertEqual(result["deep_usage"]["md_term_fragment_hits"], 0)
        self.assertEqual(result["deep_usage"]["md_term_repair_hits"], 2)

    def test_legit_ping_yong_zero_hits(self):
        # 「电平」本身是合法术语（逻辑电平）——零误伤是两个计数器的共同红线。
        result = self._run_md("逻辑电平为高。输入电平翻转。")
        self.assertEqual(result["deep_usage"]["md_term_fragment_hits"], 0)
        self.assertEqual(result["deep_usage"]["md_term_repair_hits"], 0)


if __name__ == "__main__":
    unittest.main()
