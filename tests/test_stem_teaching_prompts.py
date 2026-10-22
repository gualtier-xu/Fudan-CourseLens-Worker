"""V010-STEM-P0-1 钉测：总结教学骨架（课型自门控）+出题能力配比+judge 护栏。

三层钉面（全部离线零活体）：
①提示词结构钉——教学规则/门控指令/降级路径/五类图像解读在场，且旧串一字
  不改（299/300 防膨胀钉与底线词家规原样，规则只存在于「+ 后缀」变体）；
②格式合同钉——STEM 六段/无图四段/非 STEM 通用三种 merge 产物样本都能被
  既有校验面（_valid_summary_part/_salvage_summary_object）吃下（客户端
  import 面按具名字段提取，markdown 为自由文本，加性兼容）；
③行为路由钉——调用点组合关系（窗口/合并/出题/judge 四族后缀组合恒等式）。
"""

import json
import unittest

from courselens_worker.llm import (
    _EVIDENCE_RULES,
    _JUDGE_ADJ_SUBTITLE_PROMPT,
    _JUDGE_ADJ_SUMMARY_PROMPT,
    _JUDGE_COACH_SUFFIX,
    _JUDGE_FLASH_SUBTITLE_PROMPT,
    _JUDGE_FLASH_SUMMARY_PROMPT,
    _REVIEW_VIEWS_PROMPT,
    _REVIEW_VIEWS_PROMPT_TEACHING,
    _STEM_TEACHING_RULES,
    _SUMMARY_MERGE_PROMPT,
    _SUMMARY_MERGE_PROMPT_EVIDENCE_TEACHING,
    _SUMMARY_MERGE_PROMPT_TEACHING,
    _SUMMARY_WINDOW_PROMPT,
    _WINDOW_TEACHING_SUFFIX,
    _salvage_summary_object,
    _valid_summary_part,
)


class MergeTeachingPromptPins(unittest.TestCase):
    """P0①：合并提示词教学变体=旧串+课型自门控+六段骨架+无图降级。"""

    def test_variants_compose_from_untouched_base(self):
        self.assertTrue(_SUMMARY_MERGE_PROMPT_TEACHING.startswith(_SUMMARY_MERGE_PROMPT))
        self.assertTrue(_SUMMARY_MERGE_PROMPT_EVIDENCE_TEACHING.startswith(
            _SUMMARY_MERGE_PROMPT_TEACHING))
        self.assertTrue(_SUMMARY_MERGE_PROMPT_EVIDENCE_TEACHING.endswith(_EVIDENCE_RULES))
        # 旧串一字不改：防膨胀钉与既有键面原样（test_summary_events 同源钉）。
        self.assertLessEqual(len(_SUMMARY_MERGE_PROMPT), 300)
        self.assertIn("key_takeaways", _SUMMARY_MERGE_PROMPT)
        self.assertIn("assessment_events", _SUMMARY_MERGE_PROMPT)

    def test_gate_directives_present_with_evidence_citation(self):
        # 课型自门控：判定推导型理工须引用材料证据，判定写进笔记首行。
        for needle in (
            "先判定本讲是否推导型理工内容",
            "证据",
            "笔记首行",
            "≤40字",
        ):
            self.assertIn(needle, _STEM_TEACHING_RULES)

    def test_non_stem_defaults_to_generic_format(self):
        # 默认路径（回归钉的提示词面）：非推导型/证据不足=既有通用格式零变化。
        self.assertIn("非推导型或证据不足", _STEM_TEACHING_RULES)
        self.assertIn("按既有通用格式输出", _STEM_TEACHING_RULES)
        self.assertIn("不加骨架", _STEM_TEACHING_RULES)

    def test_six_segment_skeleton_present(self):
        for segment in ("问题与位置", "图像直觉", "机制", "递进推导", "适用条件与反例", "实例"):
            self.assertIn(segment, _STEM_TEACHING_RULES)
        self.assertIn("含假设与边界", _STEM_TEACHING_RULES)

    def test_no_image_degrades_to_four_segments(self):
        self.assertIn("无图像材料时四段", _STEM_TEACHING_RULES)
        self.assertIn("问题与位置/机制与递进推导/适用条件与反例/实例", _STEM_TEACHING_RULES)

    def test_five_image_types_with_reading_rules_and_closed_material(self):
        for image_type in ("能带图", "电路图", "波形图", "结构图", "流程图"):
            self.assertIn(image_type, _STEM_TEACHING_RULES)
        # 按材料实际出现启用对应解读；未出现不写；素材边界=只重排已有素材。
        self.assertIn("未出现的类型不写", _STEM_TEACHING_RULES)
        self.assertIn("只重排已有素材", _STEM_TEACHING_RULES)
        self.assertIn("不增材料外事实", _STEM_TEACHING_RULES)


class WindowTeachingSuffixPins(unittest.TestCase):
    """P0①：窗口保真后缀=机制/推导链与幻灯图像事实保留（基础串不改）。"""

    def test_suffix_content(self):
        for needle in ("机制与推导链", "每步为何成立", "能带", "电路", "波形",
                       "结构", "流程", "只记输入内事实"):
            self.assertIn(needle, _WINDOW_TEACHING_SUFFIX)
        self.assertTrue(_SUMMARY_WINDOW_PROMPT.startswith("你是严谨的课程学习助理"))
        self.assertNotIn("教学骨架", _WINDOW_TEACHING_SUFFIX,
                         "骨架组织归合并步，窗口只做素材保真")


class ReviewViewsTeachingPromptPins(unittest.TestCase):
    """P0②：出题能力配比+先回忆后识忆+confidence 自评+易混警示（文本约定）。"""

    def test_variant_composes_from_untouched_base(self):
        self.assertTrue(_REVIEW_VIEWS_PROMPT_TEACHING.startswith(_REVIEW_VIEWS_PROMPT))
        self.assertLessEqual(len(_REVIEW_VIEWS_PROMPT), 300)
        for needle in ("只改写输入", "不新增", "材料没有的不编", "anchor_pool",
                       "evidence_index", "study_guide", "faq", "timeline",
                       "briefing", "assessment_events"):
            self.assertIn(needle, _REVIEW_VIEWS_PROMPT)

    def test_capability_matrix_with_course_type_gating(self):
        self.assertIn("理解:应用:迁移≈5:3:2", _REVIEW_VIEWS_PROMPT_TEACHING)
        self.assertIn("非推导型课以理解为主", _REVIEW_VIEWS_PROMPT_TEACHING)
        self.assertIn("〔理解/应用/迁移〕", _REVIEW_VIEWS_PROMPT_TEACHING)

    def test_recall_first_with_confidence_self_rating(self):
        self.assertIn("先凭记忆答，自评：确/大概/猜，再看提示",
                      _REVIEW_VIEWS_PROMPT_TEACHING)
        self.assertIn("题面含尾缀≤80字", _REVIEW_VIEWS_PROMPT_TEACHING,
                      "题面帽与校验器 80 字截断同源")

    def test_hint_is_directional_with_real_confusion_warnings(self):
        self.assertIn("方向提示", _REVIEW_VIEWS_PROMPT_TEACHING)
        self.assertIn("易混点警示", _REVIEW_VIEWS_PROMPT_TEACHING)
        self.assertIn("不给答案", _REVIEW_VIEWS_PROMPT_TEACHING)


class JudgeCoachSuffixPins(unittest.TestCase):
    """P0⑤：AI-as-coach 护栏=对照原文片段再评+事实/主观分离（后缀变体）。"""

    def test_all_four_prompts_compose_with_coach_suffix(self):
        for base in (_JUDGE_FLASH_SUBTITLE_PROMPT, _JUDGE_ADJ_SUBTITLE_PROMPT,
                     _JUDGE_FLASH_SUMMARY_PROMPT, _JUDGE_ADJ_SUMMARY_PROMPT):
            composed = base + _JUDGE_COACH_SUFFIX
            self.assertTrue(composed.startswith(base))
            self.assertIn("对照原文对应片段再判", _JUDGE_COACH_SUFFIX)
            self.assertIn("事实性错误才 warn", _JUDGE_COACH_SUFFIX)
            self.assertIn("可辩护取舍或证据不足只标 info", _JUDGE_COACH_SUFFIX)

    def test_base_prompts_and_pins_untouched(self):
        for prompt in (_JUDGE_FLASH_SUBTITLE_PROMPT, _JUDGE_ADJ_SUBTITLE_PROMPT,
                       _JUDGE_FLASH_SUMMARY_PROMPT, _JUDGE_ADJ_SUMMARY_PROMPT):
            self.assertLessEqual(len(prompt), 300)
            self.assertIn("零自由文本", prompt)
            self.assertIn("禁止编造输入外的问题", prompt)
        self.assertIn("存疑即标", _JUDGE_FLASH_SUBTITLE_PROMPT)
        self.assertIn("确认或否决", _JUDGE_ADJ_SUMMARY_PROMPT)


class FormatContractPins(unittest.TestCase):
    """格式合同钉：三种形态样本被既有校验面原样吃下（加性兼容，零形状变化）。"""

    def _stem_six_segment_markdown(self) -> str:
        return (
            "# 第3讲 PN 结与能带（判定：推导型理工；证据：讲次标题与「能带/内建电场」推导）\n\n"
            "## 问题与位置\n本讲回答「为什么掺杂决定 PN 结内建电场」，承接第2讲能带理论。\n\n"
            "## 图像直觉\n能带图：纵轴能量/横轴位置，费米能级在禁带内对齐——"
            "载流子从高浓度侧扩散，留下不可移动的掺杂离子形成空间电荷区。\n\n"
            "## 机制\n扩散与漂移达到平衡：多子扩散耗尽区，内建电场阻止继续扩散。\n\n"
            "## 递进推导\n（假设：突变结、完全耗尽、一维）由电荷中性 N_A x_n = N_D x_p，"
            "再积分泊松方程得 V_bi = (kT/q)ln(N_A N_D/n_i²)。量纲检查：对数无量纲×kT/q=伏。\n\n"
            "## 适用条件与反例\n上式在热平衡、无外加偏压、非简并掺杂下成立；"
            "重掺杂简并半导体（如隧道结）不再适用。\n\n"
            "## 实例\n硅 PN 结室温 V_bi≈0.7V：代入 N_A=N_D=1e16 cm^-3 可复现。"
        )

    def _stem_four_segment_markdown(self) -> str:
        return (
            "# 第4讲 MOS 阈值电压（判定：推导型理工；证据：讲次标题与阈值公式推导）\n\n"
            "## 问题与位置\n本讲回答「栅压多大时形成反型层」。\n\n"
            "## 机制与递进推导\n（假设：理想 MOS、无界面态）平带→耗尽→反型三段："
            "V_T = V_FB + 2φ_F + γ√(2φ_F)，每项来源逐层展开。\n\n"
            "## 适用条件与反例\n理想 MOS 假设；有界面态/氧化层电荷时需加平带电压修正。\n\n"
            "## 实例\n以 N_A=1e17、t_ox=5nm 代入得 V_T≈0.8V，与讲义例题一致。"
        )

    def _generic_markdown(self) -> str:
        return "# 第5讲 近代文学思潮\n老师按时间线梳理了思潮演变与代表人物观点。"

    def test_all_three_shapes_pass_worker_validation(self):
        for markdown in (self._stem_six_segment_markdown(),
                         self._stem_four_segment_markdown(),
                         self._generic_markdown()):
            sample = {
                "markdown": markdown,
                "chapters": [{"title": "章", "start_ms": 0, "summary": "摘"}],
                "key_takeaways": ["要点"],
            }
            self.assertTrue(_valid_summary_part(sample))
            # 客户端 import 走具名字段提取：加性键被忽略，markdown 自由文本。
            self.assertEqual(_salvage_summary_object(json.dumps(sample)), sample)

    def test_generic_sample_is_byte_stable_legacy_shape(self):
        # 非 STEM 默认路径的产物形状与既有通用格式同键同型（回归钉样本面）。
        sample = {
            "markdown": self._generic_markdown(),
            "chapters": [{"title": "章", "start_ms": 0, "summary": "摘"}],
        }
        self.assertTrue(_valid_summary_part(sample))
        self.assertEqual(
            sorted(sample.keys()), ["chapters", "markdown"],
        )


if __name__ == "__main__":
    unittest.main()
