"""V010-TUNING-RXFIX-1 钉测：总结生成质量硬化（TB-RX-A1/A3 + TB-P1-1 生成侧根修）。

三层钉面（全部离线零活体）：
①A1 体量预算门——门形态（时长比例地板×绝对地板取大）、激活下限（30 分钟）、
  二态识别（骨架/通用）、骨架密度下限（A3 门级联动）、梯内单点重试+补量
  指令+降级出货+闭集计数（RXFIX-2 同款模式）；
②A3 prompt 级密度指令——骨架分支密度句在场，基础串 299/300 防膨胀钉不动；
③TB-P1-1 分隔符垃圾生成侧根修——『。；』『；；；』『：；』三形洗除、单分隔
  符与 LaTeX 跨度零触碰、合并候选三文本面（chapters/takeaways/doc）统一过洗
  （markdown 走定稿链），下游闪卡/搜索派生自净。
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from courselens_worker import llm
from courselens_worker.llm import (
    _SKELETON_SECTION_KEYWORDS,
    _STEM_TEACHING_RULES,
    _SUMMARY_MERGE_PROMPT,
    _SUMMARY_MERGE_PROMPT_TEACHING,
    _SUMMARY_VOLUME_GATE_MIN_MS,
    _SUMMARY_VOLUME_RETRY_NUDGE_FMT,
    create_summary,
    summary_markdown_format_state,
    summary_md_volume_degraded,
    summary_md_volume_floor,
    summary_md_volume_gate,
    summary_takeaways_missing,
)
from courselens_worker.summary_contract import strip_separator_junk


def _transcript(minutes: float) -> list[dict]:
    # 1 段=1 分钟（钉测收敛窗口数：80 分钟讲次=80 段=单窗，不烧 40 窗替身调用）。
    count = max(1, int(minutes))
    return [
        {
            "start_ms": index * 60_000,
            "end_ms": (index + 1) * 60_000,
            "text": f"第{index}分钟的内容。",
        }
        for index in range(count)
    ]


def _part(markdown: str) -> dict:
    return {"markdown": markdown, "chapters": []}


def _thick_markdown(chars: int) -> str:
    line = "本讲机制推导要点覆盖全部章节时间轴，内容充实无留白。"
    return "".join(line[index % len(line)] for index in range(chars))


def _skeleton_markdown(section_bodies: list[str]) -> str:
    sections = ("问题与位置", "图像直觉", "机制", "递进推导", "适用条件与反例", "实例")
    parts = ["# 第3讲 推导型理工讲次"]
    for index, body in enumerate(section_bodies):
        title = sections[index % len(sections)]
        parts.append(f"## {title}\n{body}")
    return "\n\n".join(parts)


class VolumeFloorTests(unittest.TestCase):
    """A1 门形态：max(3000, 分钟×55)；激活下限 30 分钟。"""

    def test_floor_formula_absolute_and_duration(self):
        self.assertEqual(summary_md_volume_floor(0), 3000)
        # 45 分钟：45×55=2475 → 绝对地板 3000 兜底。
        self.assertEqual(summary_md_volume_floor(45 * 60_000), 3000)
        # 90 分钟：90×55=4950 → 比例地板生效。
        self.assertEqual(summary_md_volume_floor(90 * 60_000), 4950)
        # 665055 实测讲次（≈218 分钟）：11990 —— 4899 短掷破门、23632 长掷过门。
        floor = summary_md_volume_floor(218 * 60_000)
        self.assertEqual(floor, 11990)
        self.assertTrue(4899 < floor <= 23632)

    def test_gate_activation_minimum(self):
        # 29 分钟：素材本少重掷无益=门不激活（薄文零告警）。
        below = summary_md_volume_gate("太短", 29 * 60_000)
        self.assertFalse(below["failed"])
        self.assertFalse(below["below_floor"])
        # 30 分钟整：门激活，同文破门。
        at = summary_md_volume_gate("太短", _SUMMARY_VOLUME_GATE_MIN_MS)
        self.assertTrue(at["below_floor"])
        self.assertTrue(at["failed"])
        self.assertEqual(_SUMMARY_VOLUME_GATE_MIN_MS, 30 * 60_000)

    def test_gate_boundary_0_8_reference_shape(self):
        # 665055 定谳形态重放：短掷破门、长掷过门（阈值边界对照）。
        duration = 218 * 60_000
        self.assertTrue(summary_md_volume_gate(_thick_markdown(4899), duration)["failed"])
        self.assertFalse(summary_md_volume_gate(_thick_markdown(11990), duration)["failed"])
        self.assertFalse(summary_md_volume_gate(_thick_markdown(23632), duration)["failed"])


class FormatStateTests(unittest.TestCase):
    """二态识别：骨架（教学六段式）| 通用（逐章枚举）；TB 实测全距留边。"""

    def test_skeleton_detected_with_canonical_sections(self):
        markdown = _skeleton_markdown(["正文一", "正文二", "正文三", "正文四", "正文五", "正文六"])
        self.assertEqual(summary_markdown_format_state(markdown), "skeleton")

    def test_generic_many_h2_is_generic(self):
        # 通用态 h2 27-62（TB 实测）：20 章枚举体即使带个别关键词词也不判骨架。
        markdown = "\n".join(
            f"## 章节标题{index}" + ("\n机制展开。" if index == 0 else "") for index in range(28)
        )
        self.assertEqual(summary_markdown_format_state(markdown), "generic")
        # 无 h2 的散文体=通用。
        self.assertEqual(summary_markdown_format_state("纯散文一段。"), "generic")

    def test_keyword_hits_threshold(self):
        # h2 计数在骨架带内但骨架段关键词 <2 命中=通用（防「机制」类通用章题误判）。
        markdown = "\n\n".join(f"## 工作机制{index}\n正文" for index in range(6))
        self.assertEqual(summary_markdown_format_state(markdown), "generic")
        self.assertEqual(len(_SKELETON_SECTION_KEYWORDS), 5)


class SkeletonDensityTests(unittest.TestCase):
    """A3 门级联动：骨架态过半空壳段=门破；单薄段不烧重试（轻快律）。"""

    def test_majority_stub_sections_fail(self):
        # 十倍缩水骨架模式重放：6 段全为 1-2 句空壳。
        markdown = _skeleton_markdown(["太短。", "也短。", "还短。", "同短。", "仍短。", "末短。"])
        gate = summary_md_volume_gate(_thick_markdown(20000), 0)
        self.assertFalse(gate["failed"])  # 长文非骨架路径不受影响
        state = summary_markdown_format_state(markdown)
        self.assertEqual(state, "skeleton")
        violations, checked = llm._skeleton_density_violations(markdown)
        self.assertEqual(checked, 6)
        self.assertEqual(violations, 6)

    def test_rich_sections_or_bullets_pass(self):
        rich = _skeleton_markdown([
            _thick_markdown(320),
            _thick_markdown(320),
            "- 要点一\n- 要点二\n- 要点三",
            _thick_markdown(320),
            _thick_markdown(320),
            _thick_markdown(320),
        ])
        violations, checked = llm._skeleton_density_violations(rich)
        self.assertEqual(checked, 6)
        self.assertEqual(violations, 0)

    def test_density_gate_only_in_skeleton_state(self):
        # 通用态不跑密度（各章长度参差是通用格式常态）——同形态在骨架态必破，
        # 通用态门不破。
        generic = "\n\n".join(f"## 通用章{index}\n短。" for index in range(28))
        self.assertEqual(summary_markdown_format_state(generic), "generic")
        self.assertFalse(summary_md_volume_gate(generic, 0)["failed"])


class SummaryChainVolumeGateTests(unittest.TestCase):
    """A1 梯内集成：薄探单点重试+补量指令→降级出货+闭集计数（RXFIX-2 同款）。"""

    @staticmethod
    def _run(merge_responses, minutes: float):
        payloads: list[dict] = []

        def fake_chat(api_key, messages, **kwargs):
            if messages[0]["content"] != _SUMMARY_MERGE_PROMPT_TEACHING:
                return json.dumps(_part("窗口笔记"))
            payloads.append({
                "system": messages[0]["content"],
                "user": messages[1]["content"],
            })
            return merge_responses[min(len(payloads) - 1, len(merge_responses) - 1)]

        with patch.dict(os.environ, {llm.SUMMARY_THINKING_ENV: "", llm.REVIEW_VIEWS_ENV: "off"}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=fake_chat) as chat:
                result = create_summary(
                    "k", title="t", transcript=_transcript(minutes), ppt_pages=[],
                )
        return result, chat, payloads

    def test_thin_throw_retried_once_then_degraded_with_counter(self):
        thin = json.dumps({
            "markdown": _thick_markdown(800), "chapters": [], "key_takeaways": ["要点"],
        })
        result, chat, payloads = self._run([thin], minutes=80)
        self.assertEqual(chat.call_count, 3)  # 1 窗口 + 2 合并（第二次=体量重试）
        self.assertEqual(result["markdown"], _thick_markdown(800))  # 降级出货不整单失败
        self.assertEqual(result["deep_usage"]["md_volume_retries"], 1)
        self.assertEqual(result["deep_usage"]["md_volume_degraded"], 1)
        self.assertEqual(result["deep_usage"]["takeaways_missing"], 0)
        # 重试请求：system 逐位不动、user 侧带补量指令（含地板数字）。
        self.assertEqual(payloads[1]["system"], payloads[0]["system"])
        extra = payloads[1]["user"][len(payloads[0]["user"]):]
        self.assertIn("体量预算下限", extra)
        self.assertIn("4400", extra)  # 80 分钟 × 55 = 4400
        # 降级警告闭集判定（runner 两站同源）。
        self.assertTrue(summary_md_volume_degraded(result))
        self.assertTrue(summary_md_volume_degraded({"deep_usage": {"md_volume_degraded": 1}}))
        self.assertFalse(summary_md_volume_degraded(result | {"deep_usage": {}}))
        self.assertFalse(summary_md_volume_degraded(None))

    def test_thin_then_thick_recovers_without_degrade(self):
        thin = json.dumps({
            "markdown": _thick_markdown(800), "chapters": [], "key_takeaways": ["要点"],
        })
        thick = json.dumps({
            "markdown": _thick_markdown(6000), "chapters": [], "key_takeaways": ["要点"],
        })
        result, chat, payloads = self._run([thin, thick], minutes=80)
        self.assertEqual(chat.call_count, 3)
        self.assertEqual(result["markdown"], _thick_markdown(6000))
        self.assertEqual(result["deep_usage"]["md_volume_retries"], 1)
        self.assertEqual(result["deep_usage"]["md_volume_degraded"], 0)
        self.assertFalse(summary_md_volume_degraded(result))

    def test_short_recording_gate_inactive_zero_drift(self):
        # <30 分钟：门不激活——薄文零重试（短录音素材本少，重掷无益；钉测
        # 替身微转写天然不适用=RXFIX-2 既有钉面零漂移的同一语义边界）。
        thin = json.dumps({
            "markdown": "combined", "chapters": [], "key_takeaways": ["要点"],
        })
        result, chat, payloads = self._run([thin], minutes=2)
        self.assertEqual(chat.call_count, 2)  # 1 窗口 + 1 合并（零体量重试）
        self.assertEqual(result["deep_usage"]["md_volume_retries"], 0)
        self.assertEqual(result["deep_usage"]["md_volume_degraded"], 0)
        self.assertNotIn("体量预算", payloads[0]["user"])

    def test_skeleton_stump_triggers_density_retry(self):
        # 骨架态过半空壳（TB「十倍缩水骨架模式」）：门级 A3 联动破门。
        stump = json.dumps({
            "markdown": _skeleton_markdown(["太短。"] * 6),
            "chapters": [], "key_takeaways": ["要点"],
        })
        result, chat, payloads = self._run([stump], minutes=0.5)
        self.assertEqual(chat.call_count, 3)  # 密度门不设时长下限（形态纪律）
        self.assertEqual(result["deep_usage"]["md_volume_retries"], 1)
        extra = payloads[1]["user"][len(payloads[0]["user"]):]
        self.assertIn("体量预算下限", extra)
        # 末试仍塌=降级计数（单点梯不重复消耗）。
        self.assertEqual(result["deep_usage"]["md_volume_degraded"], 1)

    def test_volume_and_takeaway_nudges_compose(self):
        # 双缺并串：首掷要点面缺（体量未评），补产指令先行——单点梯语义保持。
        bad = json.dumps({"markdown": _thick_markdown(800), "chapters": []})
        good = json.dumps({
            "markdown": _thick_markdown(6000), "chapters": [], "key_takeaways": ["要点"],
        })
        result, chat, payloads = self._run([bad, good], minutes=80)
        self.assertEqual(chat.call_count, 3)
        self.assertEqual(result["deep_usage"]["md_volume_degraded"], 0)
        self.assertEqual(result["deep_usage"]["takeaways_missing"], 0)
        extra = payloads[1]["user"][len(payloads[0]["user"]):]
        self.assertIn("key_takeaways", extra)

    def test_takeaway_gate_priority_preserved(self):
        # RXFIX-2 既有钉面回归：要点面双空=质量重试+降级（体量门不抢先）。
        bad = json.dumps({"markdown": "combined", "chapters": []})
        result, chat, payloads = self._run([bad], minutes=2)
        self.assertEqual(chat.call_count, 3)
        self.assertEqual(result["deep_usage"]["takeaways_missing"], 1)
        self.assertEqual(result["deep_usage"]["md_volume_degraded"], 0)
        self.assertTrue(summary_takeaways_missing(result))


class PromptDensityInstructionTests(unittest.TestCase):
    """A3 prompt 级：骨架分支密度句在场；基础串防膨胀钉不动。"""

    def test_density_directive_present_in_teaching_rules(self):
        self.assertIn("每个一级段至少 3 条要点或 300 字", _STEM_TEACHING_RULES)
        self.assertIn("覆盖全部章节时间轴", _STEM_TEACHING_RULES)
        self.assertIn("禁整段留白", _STEM_TEACHING_RULES)
        # 指令只住骨架分支语境（教学规则串），基础串一字不改。
        self.assertLessEqual(len(_SUMMARY_MERGE_PROMPT), 300)
        self.assertTrue(_SUMMARY_MERGE_PROMPT_TEACHING.startswith(_SUMMARY_MERGE_PROMPT))

    def test_retry_nudge_format_closed_shape(self):
        text = _SUMMARY_VOLUME_RETRY_NUDGE_FMT.format(min_chars=4400)
        self.assertIn("4400", text)
        self.assertIn("覆盖全部章节时间轴", text)
        self.assertIn("不得整段留白", text)


class SeparatorJunkTests(unittest.TestCase):
    """TB-P1-1 生成侧根修：三形洗除+合法形态零触碰+LaTeX 豁免。"""

    def test_dominant_period_semicolon_form(self):
        # 实采主形态（589 处）：句号后游离分号 → 剥分号留句号。
        self.assertEqual(
            strip_separator_junk("数集无特别多繁琐公式。；教师提供邮箱联系方式"),
            "数集无特别多繁琐公式。教师提供邮箱联系方式",
        )

    def test_semicolon_run_collapses(self):
        # 实采次形态（30 处）：『；；；』跑段归并；行尾空项剥分隔符留冒号。
        self.assertEqual(strip_separator_junk("主要问题（延时随扇入剧增）：；；；"), "主要问题（延时随扇入剧增）：")
        self.assertEqual(strip_separator_junk("面积比较：；；"), "面积比较：")
        self.assertEqual(strip_separator_junk("a；；；b"), "a；b")
        self.assertEqual(strip_separator_junk("a。。b"), "a。b")

    def test_legit_single_separators_untouched(self):
        text = "要点一；要点二；要点三。结尾句。"
        self.assertEqual(strip_separator_junk(text), text)
        # 行尾单分号（正常列表终止）保留。
        self.assertEqual(strip_separator_junk("要点一；要点二；"), "要点一；要点二；")

    def test_latex_and_inline_code_spans_exempt(self):
        text = r"电流满足 \(I_E=I_C+I_B;\; \alpha\) 关系，`a;b;c` 保持。"
        self.assertEqual(strip_separator_junk(text), text)

    def test_idempotent(self):
        once = strip_separator_junk("公式。；要点：；；要点二")
        self.assertEqual(strip_separator_junk(once), once)


class SummaryChainSeparatorTests(unittest.TestCase):
    """合并链三文本面统一过洗：markdown/chapters/takeaways/doc 下游全净。"""

    def test_generated_surfaces_clean(self):
        dirty_markdown = (
            "# 合并笔记\n\n静态逻辑的主要问题（延时随扇入剧增）：；；；\n\n"
            r"公式 \(I_E;\,I_C\) 保持原样。；教师联系方式见课程页。"
        )
        merge = json.dumps({
            "markdown": dirty_markdown,
            "chapters": [
                {"title": "章节一。；", "start_ms": 0, "summary": "摘要。；补充"},
            ],
            "key_takeaways": ["结论一。；"],
            "doc": {
                "overview": "总述。；续",
                "sections": [{
                    "start_ms": 0, "title": "开篇。；", "source": "semantic",
                    "points": [{"text": "要点。；", "anchor_ms": 0}],
                    "formulas": [],
                }],
                "terms": [{"symbol": "I_E", "name": "发射极电流。；", "note": "注。；"}],
                "summary_notes": ["小结。；"],
            },
        })

        def fake_chat(api_key, messages, **kwargs):
            if messages[0]["content"] != _SUMMARY_MERGE_PROMPT_TEACHING:
                return json.dumps(_part("窗口笔记"))
            return merge

        with patch.dict(os.environ, {llm.SUMMARY_THINKING_ENV: "", llm.REVIEW_VIEWS_ENV: "off"}, clear=False):
            with patch("courselens_worker.llm._chat", side_effect=fake_chat):
                result = create_summary(
                    "k", title="t", transcript=_transcript(0.05), ppt_pages=[],
                )
        # markdown 面：『：；；；』行尾剥除+『。；』归并；LaTeX 跨度豁免。
        self.assertIn("（延时随扇入剧增）：\n", result["markdown"])
        self.assertIn("原样。教师联系方式", result["markdown"])
        self.assertNotIn("。；", result["markdown"])
        self.assertNotIn("；；", result["markdown"])
        self.assertIn(r"\(I_E;\,I_C\)", result["markdown"])
        # chapters/takeaways 面（闪卡派生源）。
        self.assertEqual(result["chapters"][0]["title"], "章节一。")
        self.assertEqual(result["chapters"][0]["summary"], "摘要。补充")
        self.assertEqual(result["key_takeaways"], ["结论一。"])
        # doc 面（结构化读面与复习视图派生源）。
        self.assertEqual(result["doc"]["overview"], "总述。续")
        self.assertEqual(result["doc"]["sections"][0]["title"], "开篇。")
        self.assertEqual(result["doc"]["sections"][0]["points"][0]["text"], "要点。")
        self.assertEqual(result["doc"]["terms"][0]["name"], "发射极电流。")
        self.assertEqual(result["doc"]["summary_notes"], ["小结。"])


if __name__ == "__main__":
    unittest.main()
