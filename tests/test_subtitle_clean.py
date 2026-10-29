"""V010-SUBTITLE-D1 字幕清洗腿钉测（Tier1 闭集边界/Tier2 删除门/终界接线/忠实层）。

规格唯一出处=archive V010-SUBTITLE-UX-R1 §2；字段契约=shared/
subtitle_display_contract.py（540ca12f）。所有样本通识虚构，零真实课程/
讲次/账号信息（调优数据脱敏律）。
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from courselens_worker.llm import proofread_segments
from courselens_worker.subtitle_clean import (
    MID_SENTENCE_FILLERS,
    PARTICLE_FILLERS,
    apply_display_cleaning,
    tier1_clean_text,
    validate_tier2_dels,
)
from shared.subtitle_display_contract import (
    CLEAN_TIER1_MAX_DEL_RATIO,
    CLEAN_TIER2_MAX_DEL_RATIO,
    SubtitleDisplayContractError,
    clean_del_ratio_of,
    clean_text_is_deletion_only,
    display_text,
    validate_display_cue_v2,
)


def seg(start_ms, end_ms, text, **extra):
    value = {"start_ms": start_ms, "end_ms": end_ms, "text": text}
    value.update(extra)
    return value


def run_proofread(sensevoice, primary, responses, *, prior=None, clean_plan_sink=None):
    """Mock _chat 跑 proofread_segments；返回 (result, payloads, plan)。"""
    payloads = []

    def fake_chat(api_key, messages, **kwargs):
        payloads.append(json.loads(messages[1]["content"]))
        return responses

    plan: list[dict] = []
    with patch("courselens_worker.llm._chat", side_effect=fake_chat):
        result = proofread_segments(
            "secret",
            sensevoice,
            primary,
            prior_checkpoint=prior,
            clean_plan_sink=clean_plan_sink,
        )
    return result, payloads, plan


class Tier1BoundaryTests(unittest.TestCase):
    """Tier1 闭集边界守卫钉（规格 §2 Tier1 守卫 + fixture 金样本对齐）。"""

    def test_tag_filler_deleted_at_sentence_end(self):
        self.assertEqual(tier1_clean_text("今天天气很好对吧"), "今天天气很好")

    def test_tag_filler_deleted_after_juncture(self):
        self.assertEqual(
            tier1_clean_text("这个公式就是说我们在下一章的学习里面会用到 对吧"),
            "这个公式我们在下一章的学习里面会用到",
        )

    def test_tag_filler_kept_before_content(self):
        # 句首问询垫词（「对吧我们开始」）单边后界守卫：后接内容不删。
        self.assertIsNone(tier1_clean_text("对吧我们开始今天的课"))

    def test_mid_filler_deleted_mid_sentence(self):
        # fixture 金样本对齐：就是说 句中形态直接删（形态本身无歧义）。
        self.assertIsNone(tier1_clean_text("这个公式很重要"))
        self.assertEqual(tier1_clean_text("线性代数就是说研究矩阵的数学"), "线性代数研究矩阵的数学")

    def test_mid_filler_full_table_unconditional(self):
        for form in MID_SENTENCE_FILLERS:
            text = f"这一章{form}我们讲矩阵的乘法与转置的定义"
            cleaned = tier1_clean_text(text)
            self.assertIsNotNone(cleaned, form)
            self.assertNotIn(form, cleaned)

    def test_bare_demonstrative_kept_before_content(self):
        # 「这个公式」的指示用法绝不删（宁脏勿错守卫）。
        self.assertIsNone(tier1_clean_text("这个公式很重要"))

    def test_bare_demonstrative_kept_sentence_final(self):
        # 句尾指代宾语（「我们讲这个」）内容性保留（双边整词跨度守卫）。
        self.assertIsNone(tier1_clean_text("我们讲这个"))

    def test_bare_demonstrative_deleted_as_whole_span(self):
        self.assertEqual(tier1_clean_text("我们讲，然后，收尾"), None)  # 帽拒（3/9>30%）
        self.assertEqual(tier1_clean_text("好的，那个，我们看下一题"), "好的，我们看下一题")

    def test_particle_deleted_between_junctures(self):
        self.assertEqual(tier1_clean_text("嗯，我们讲下一个例子"), "我们讲下一个例子")
        for form in PARTICLE_FILLERS:
            text = f"同学们{form}，我们开始上课"
            cleaned = tier1_clean_text(text)
            self.assertIsNotNone(cleaned, form)
            self.assertNotIn(form, cleaned)

    def test_particle_kept_before_content(self):
        # 语气词后接内容不删（「哇这个好」的「哇」保留）。
        self.assertIsNone(tier1_clean_text("哇这个结果很漂亮"))

    def test_negation_prefix_guard(self):
        # 「不对吧」≠「对吧」垫词形态：紧邻前否定字不删。
        self.assertIsNone(tier1_clean_text("不对吧，他是错的"))

    def test_negation_suffix_guard(self):
        # 「就是说不清楚」的「就是说」处在否定边界上，删除会翻转语义。
        self.assertIsNone(tier1_clean_text("就是说不清楚这个概念"))
        self.assertIsNone(tier1_clean_text("啊不是这样的"))


class MidSentenceExpansionTests(unittest.TestCase):
    """V010-SUBTITLE-CLEAN-R2 U3 句中语气词/话语垫词扩充钉（D4 升级项）。

    每个扩充形态均带否定陷阱与实义词形排除回归（「不对吧」类实证拒绝形态
    的同源面）；守卫三重门（否定边界/保护形/后界排除）全部生效。
    """

    def test_mid_sentence_particle_deleted(self):
        # D4 样本A 行3 实证残留（「判别法呢呃两个」）：呢/呃 句中形态删，
        # 对吧 随贪心入帽；就是说 再入即超帽（7/19=37%）→ 帽内诚实保留
        #（TAG 先于 MID，D4 优先级建议）。
        self.assertEqual(
            tier1_clean_text("比较判别法呢就是说呃两个级数比一比对吧"),
            "比较判别法就是说两个级数比一比",
        )
        self.assertEqual(
            tier1_clean_text("收敛的级数呢它有这个好的性质"),
            "收敛的级数它有这个好的性质",
        )

    def test_head_particle_deleted_aligns_as13_strip(self):
        # 呃/嗯 any 守卫与 AS13 句首剥离拍板闭集同字：句首形态一并删。
        self.assertEqual(
            tier1_clean_text("呃比值判别法也就是达朗贝尔判别法懂吧"),
            "比值判别法也就是达朗贝尔判别法",
        )

    def test_particle_negation_traps(self):
        # 否定陷阱回归（扩充形态逐词）：否定边界上一律不删。
        self.assertIsNone(tier1_clean_text("嗯不是这样的"))
        self.assertIsNone(tier1_clean_text("呃不"))  # 呃后紧邻否定字
        self.assertIsNone(tier1_clean_text("其实不然，他没来"))
        self.assertIsNone(tier1_clean_text("不那么做更稳妥"))
        self.assertIsNone(tier1_clean_text("这个问题呢不存在反例"))  # 呢后否定

    def test_discourse_fillers_deleted(self):
        # 具名话语垫词扩充（就是说吧/其实/那么）句中直接删。
        self.assertEqual(
            tier1_clean_text("这一章就是说吧我们讲矩阵的乘法"),
            "这一章我们讲矩阵的乘法",
        )
        self.assertEqual(tier1_clean_text("其实这道题有两种解法"), "这道题有两种解法")
        self.assertEqual(tier1_clean_text("那么我们来看下一题"), "我们来看下一题")

    def test_mid_form_next_exclusion_traps(self):
        # 实义词形后界排除闭集：那么多/呢子/呢绒/呢喃 类形态保留原文。
        self.assertIsNone(tier1_clean_text("那么多作业要写"))
        self.assertIsNone(tier1_clean_text("这件呢子大衣很贵"))
        self.assertIsNone(tier1_clean_text("呢喃细语很轻"))
        # Tier2 同源排除门：实义形态提案拒（fail-closed 回退 Tier1）。
        cleaned, reason = validate_tier2_dels("那么多作业要写", ["那么"])
        self.assertIsNone(cleaned)
        self.assertEqual(reason, "closed_set")
        cleaned, reason = validate_tier2_dels("那么我们来看下一章的内容", ["那么"])
        self.assertIsNone(reason)
        self.assertEqual(cleaned, "我们来看下一章的内容")

    def test_whole_filler_cue_refused(self):
        # 整句皆垫词：删后 content-core 为空 → 保留原文（cue 级删除属整形层）。
        self.assertIsNone(tier1_clean_text("嗯，对吧。"))

    def test_leading_punct_folded(self):
        # 删后不产生句首标点：首部句读折叠（仍为删除式子序列）。
        self.assertEqual(tier1_clean_text("好，对吧，我们接着讲"), "好，我们接着讲")

    def test_seam_double_punct_folded(self):
        self.assertEqual(tier1_clean_text("定义讲完了对吧，接下来讲定理"), "定义讲完了，接下来讲定理")

    def test_ratio_cap_greedy_partial_clean(self):
        # V010-SUBTITLE-CLEAN-R2 U1：超帽句不再整句放弃——帽内贪心保留
        # （「嗯啊对吧然后呢」：嗯(any)入帽 1/8；然后呢单项 3/8 超帽弃）。
        text = "嗯啊对吧然后呢"
        cleaned = tier1_clean_text(text)
        self.assertEqual(cleaned, "啊对吧然后呢")
        self.assertLessEqual(
            clean_del_ratio_of(text, cleaned), CLEAN_TIER1_MAX_DEL_RATIO
        )

    def test_ratio_cap_no_candidate_fits_keeps_original(self):
        # U1 边界：唯一可删候选自身超帽 → 保留原文（帽语义不变）。
        self.assertIsNone(tier1_clean_text("哎就是说"))

    def test_dense_filler_sentence_greedy_subset(self):
        # U1 主靶（D4 样本A 行8）：14 字句 4 垫词全删=57% 超帽——贪心按
        # 优先级类保留呢/啊（2/14），然后呢/就是说 各自超帽弃。
        text = "然后呢大于一呢就是说它发散啊"
        cleaned = tier1_clean_text(text)
        self.assertEqual(cleaned, "然后呢大于一就是说它发散")
        self.assertLessEqual(
            clean_del_ratio_of(text, cleaned), CLEAN_TIER1_MAX_DEL_RATIO
        )
        self.assertTrue(clean_text_is_deletion_only(cleaned, text))

    def test_greedy_priority_particle_before_mid(self):
        # U1 优先级：纯语气词（呢/啊）先于话语垫词（就是说）入帽，且
        # 「就是说呢」删就是说不再孤儿化句首呢。
        text = "就是说呢先看这个正项级数啊"
        cleaned = tier1_clean_text(text)
        self.assertEqual(cleaned, "就是说先看这个正项级数")
        self.assertNotIn("呢", cleaned)

    def test_greedy_full_fit_identical_to_all_spans(self):
        # U1 等价性：全候选入帽时与原 all-or-nothing 结果全等（零回归面）。
        self.assertEqual(
            tier1_clean_text("呃今天呢我们来讲这个级数收敛的判别法对吧"),
            "今天我们来讲这个级数收敛的判别法",
        )

    def test_protected_partial_overlap_drops_candidate(self):
        # 「三那个十」删「那个」会组合出新保护形「三十」→ 整段拒绝。
        self.assertIsNone(tier1_clean_text("三那个十"))

    def test_protected_forms_survive(self):
        # 数字/公式/否定保护形零触碰：清洗结果逐字保留。
        for text in ("判别式大于零时方程有两个不相等的实数根。", "x=3.14约等于圆周率", "负数不能开偶次方根"):
            cleaned = tier1_clean_text(text)
            if cleaned is not None:
                self.assertIn("3.14" if "3.14" in text else "不能", cleaned)

    def test_tier1_never_mutates_input(self):
        text = "这个公式就是说我们会用到 对吧"
        tier1_clean_text(text)
        self.assertEqual(text, "这个公式就是说我们会用到 对吧")

    def test_no_deletion_means_no_cleaning(self):
        self.assertIsNone(tier1_clean_text("判别式大于零时方程有两个不相等的实数根。"))


class Tier2DeletionGateTests(unittest.TestCase):
    """Tier2 deletion-only 验证门钉（闭集/保护形/帽/形状/content-core）。"""

    def test_fixture_gold_sample_semantics(self):
        text = "然后呢我们来看下一个例子 呃 这个例子在我们的课程里面很有代表性对不对"
        cleaned, reason = validate_tier2_dels(text, ["然后呢", "对不对"])
        self.assertIsNone(reason)
        self.assertEqual(cleaned, "我们来看下一个例子 呃 这个例子在我们的课程里面很有代表性")
        self.assertTrue(clean_text_is_deletion_only(cleaned, text))

    def test_tag_with_negation_waived(self):
        # 具名词表整删豁免：「对不对」的「不」随整词放行（16 字段 3/16≤20%）。
        cleaned, reason = validate_tier2_dels("我们讲到这里面的第三个性质对不对", ["对不对"])
        self.assertIsNone(reason)
        self.assertEqual(cleaned, "我们讲到这里面的第三个性质")

    def test_repeat_stutter_dedup(self):
        cleaned, reason = validate_tier2_dels("我们我们来看这道题的解法过程", ["我们"])
        self.assertIsNone(reason)
        self.assertEqual(cleaned, "我们来看这道题的解法过程")

    def test_non_adjacent_repeat_rejected(self):
        # 「我们，我们讲」是修辞不是口吃：不相邻重复不删。
        cleaned, reason = validate_tier2_dels("我们，我们讲下一部分内容啊", ["我们"])
        self.assertEqual(reason, "closed_set")
        self.assertIsNone(cleaned)

    def test_closed_set_miss_rejected(self):
        cleaned, reason = validate_tier2_dels("方程有两个不相等的实数根", ["方程"])
        self.assertEqual(reason, "closed_set")
        self.assertIsNone(cleaned)

    def test_del_absent_rejected(self):
        cleaned, reason = validate_tier2_dels("我们讲矩阵的乘法法则", ["然后呢"])
        self.assertEqual(reason, "closed_set")
        self.assertIsNone(cleaned)

    def test_ratio_cap_hard_gate(self):
        # 删词比 ≤20% 硬门：超帽整段拒绝（回退 Tier1 由终界处理）。
        text = "嗯，那个，然后呢我们讲两数和的平方公式对吧"
        cleaned, reason = validate_tier2_dels(text, ["嗯", "那个", "然后呢", "对吧"])
        self.assertIsNone(cleaned)
        self.assertIn(reason, {"ratio", "contract", "protected"})

    def test_protected_fragment_rejected_at_closed_set(self):
        # 数字串不在闭集：第一道保护线=闭集门直接拒（护数字不靠重叠判定）。
        cleaned, reason = validate_tier2_dels("圆周率约等于3.14请记住它", ["3.14"])
        self.assertEqual(reason, "closed_set")
        self.assertIsNone(cleaned)

    def test_protected_partial_overlap_rejected(self):
        # 闭集垫词与保护形部分重叠：「他是不是不能来」删「是不是」会撕开
        # 「不能」→ 整段拒（fail-closed）。
        cleaned, reason = validate_tier2_dels("他是不是不能来上课呢", ["是不是"])
        self.assertEqual(reason, "protected")
        self.assertIsNone(cleaned)

    def test_ops_cap_shape_gate(self):
        text = "嗯啊那个然后就是说对吧好"
        cleaned, reason = validate_tier2_dels(text, ["嗯", "啊", "那个", "然后呢", "就是说"])
        self.assertEqual(reason, "shape")
        self.assertIsNone(cleaned)

    def test_empty_and_type_gates(self):
        self.assertEqual(validate_tier2_dels("", ["对吧"])[1], "empty")
        self.assertEqual(validate_tier2_dels("我们讲矩阵的乘法运算", [123])[1], "shape")
        self.assertEqual(validate_tier2_dels("我们讲矩阵的乘法运算", [])[1], "shape")

    def test_content_core_gate(self):
        # 全垫词 cue 删空 → 拒（保留原文）。
        cleaned, reason = validate_tier2_dels("嗯，对吧", ["嗯", "对吧"])
        self.assertIsNone(cleaned)
        self.assertEqual(reason, "content_core")


class FinalBoundaryTests(unittest.TestCase):
    """终界清洗钉：锚匹配/Tier 回退/契约 schema/忠实层/VTT 恒 verbatim。"""

    def test_tier2_plan_wins_when_revalidated(self):
        segments = [seg(0, 2000, "然后呢我们讲矩阵的乘法运算的分配律与结合律的例题对吧")]
        plan = [{"start_ms": 0, "end_ms": 2000, "dels": ["然后呢", "对吧"]}]
        stats = apply_display_cleaning(segments, plan)
        cue = segments[0]
        self.assertEqual(stats["tier2_applied"], 1)
        self.assertEqual(cue["clean_source"], "tier2_llm")
        self.assertEqual(cue["clean_text"], "我们讲矩阵的乘法运算的分配律与结合律的例题")
        validate_display_cue_v2(cue)  # 共享契约 schema 校验绿

    def test_plan_anchor_mismatch_falls_back_to_tier1(self):
        segments = [seg(0, 2000, "矩阵的乘法就是说行乘列的运算规则")]
        plan = [{"start_ms": 999, "end_ms": 999, "dels": ["就是说"]}]
        stats = apply_display_cleaning(segments, plan)
        cue = segments[0]
        self.assertEqual(stats["tier1_applied"], 1)
        self.assertEqual(stats["tier2_plan_unmatched"], 1)
        self.assertEqual(cue["clean_source"], "tier1_rule")
        self.assertEqual(cue["clean_text"], "矩阵的乘法行乘列的运算规则")

    def test_stale_plan_fragment_falls_back_to_tier1(self):
        # term 深校对改写后 del 不再逐字出现 → 该提案拒 → Tier1 兜底。
        segments = [seg(0, 2000, "然后呢我们讲矩阵的乘法的运算规则")]
        plan = [{"start_ms": 0, "end_ms": 2000, "dels": ["然后呢两数和"]}]
        stats = apply_display_cleaning(segments, plan)
        cue = segments[0]
        self.assertEqual(stats["tier2_fallback_tier1"], 1)
        self.assertEqual(cue["clean_source"], "tier1_rule")

    def test_no_deletion_emits_no_keys(self):
        segments = [seg(0, 2000, "判别式大于零时方程有两个不相等的实数根。")]
        stats = apply_display_cleaning(segments, [])
        self.assertEqual(stats["no_cleaning"], 1)
        self.assertNotIn("clean_text", segments[0])
        self.assertNotIn("clean_source", segments[0])
        self.assertNotIn("clean_del_ratio", segments[0])

    def test_text_never_written_by_cleaning(self):
        # 忠实层构造性保证：清洗只加三键，text 逐字节恒等。
        segments = [seg(0, 2000, "然后呢我们讲矩阵的乘法的运算规则")]
        before = segments[0]["text"]
        apply_display_cleaning(segments, [{"start_ms": 0, "end_ms": 2000, "dels": ["然后呢"]}])
        self.assertEqual(segments[0]["text"], before)
        self.assertTrue(clean_text_is_deletion_only(segments[0]["clean_text"], before))

    def test_vtt_export_verbatim_frozen(self):
        from courselens_worker.formats import to_vtt

        segments = [seg(0, 2000, "然后呢我们讲矩阵的乘法的运算规则")]
        vtt_before = to_vtt([dict(item) for item in segments])
        apply_display_cleaning(segments, [{"start_ms": 0, "end_ms": 2000, "dels": ["然后呢"]}])
        self.assertEqual(to_vtt(segments), vtt_before)  # VTT 面恒 verbatim
        self.assertNotIn("clean", vtt_before)

    def test_ratio_honesty_via_contract(self):
        from shared.subtitle_display_contract import clean_del_ratio_of

        segments = [seg(0, 2000, "然后呢我们讲矩阵的乘法的运算规则")]
        apply_display_cleaning(segments, [{"start_ms": 0, "end_ms": 2000, "dels": ["然后呢"]}])
        cue = segments[0]
        self.assertAlmostEqual(
            cue["clean_del_ratio"], clean_del_ratio_of(cue["text"], cue["clean_text"])
        )

    def test_bad_plan_shape_ignored(self):
        segments = [seg(0, 2000, "然后呢我们讲矩阵的乘法运算规则对吧")]
        stats = apply_display_cleaning(segments, [
            "garbage",
            {"start_ms": "x", "end_ms": 2000, "dels": ["然后呢"]},
            {"start_ms": 0, "end_ms": 2000, "dels": []},
        ])
        self.assertEqual(stats["tier1_applied"], 1)  # 坏计划项全弃，Tier1 兜底


class ProofreadPiggybackTests(unittest.TestCase):
    """校对窗 piggyback 钉：del op 顺带产出/计划沉淀/检查点续跑种子。"""

    def test_del_ops_captured_into_plan_without_text_change(self):
        text = "然后呢我们讲矩阵的乘法运算的分配律与结合律的例题对吧"
        sensevoice = [seg(0, 1000, text)]
        primary = [seg(0, 1000, text)]
        response = json.dumps(
            [
                {"id": "p0", "del": "然后呢"},
                {"id": "p0", "del": "对吧"},
            ],
            ensure_ascii=False,
        )
        sink: list[dict] = []
        result, payloads, _plan = run_proofread(
            sensevoice, primary, response, clean_plan_sink=sink
        )
        self.assertEqual(result[0]["text"], text)  # text 恒不变
        self.assertEqual(sink, [{"start_ms": 0, "end_ms": 1000, "dels": ["然后呢", "对吧"]}])
        # 提示词合同钉：del op 形成文（边界规则入钉）。
        self.assertNotIn("del", json.dumps(payloads[0], ensure_ascii=False))

    def test_prompt_carries_deletion_contract(self):
        from courselens_worker.llm import _PROOFREAD_INSTRUCTIONS

        self.assertIn('"del"', _PROOFREAD_INSTRUCTIONS)
        self.assertIn("只删不改", _PROOFREAD_INSTRUCTIONS)
        self.assertIn("不得作为 del", _PROOFREAD_INSTRUCTIONS)

    def test_rejected_del_op_counts_not_breaks(self):
        sensevoice = [seg(0, 1000, "方程有两个不相等的实数根")]
        primary = [seg(0, 1000, "方程有两个不相等的实数根")]
        response = json.dumps(
            [{"id": "p0", "del": "实数根"}], ensure_ascii=False  # 闭集外内容词
        )
        sink: list[dict] = []
        result, _payloads, _plan = run_proofread(sensevoice, primary, response, clean_plan_sink=sink)
        self.assertEqual(result[0]["text"], "方程有两个不相等的实数根")
        self.assertEqual(sink, [])  # 拒绝=零计划，零报错

    def test_checkpoint_carries_clean_ops_plan(self):
        sensevoice = [seg(0, 1000, "然后呢我们讲矩阵的乘法的运算规则")]
        primary = [seg(0, 1000, "然后呢我们讲矩阵的乘法的运算规则")]
        response = json.dumps([{"id": "p0", "del": "然后呢"}], ensure_ascii=False)
        written: list[dict] = []
        payloads = []

        def fake_chat(api_key, messages, **kwargs):
            payloads.append(messages[1]["content"])
            return response

        with patch("courselens_worker.llm._chat", side_effect=fake_chat):
            proofread_segments(
                "secret",
                sensevoice,
                primary,
                checkpoint=written.append,
                clean_plan_sink=(plan_sink := []),
            )
        self.assertTrue(written)
        self.assertEqual(written[-1]["clean_ops_plan"], plan_sink)
        self.assertEqual(plan_sink[0]["dels"], ["然后呢"])

    def test_resume_seeds_plan_from_prior_checkpoint(self):
        # 续跑：prior 计划种子进 sink，已完成窗零重复计费、计划零丢失。
        sensevoice: list[dict] = []
        primary = [seg(0, 1000, "我们讲矩阵的乘法")]
        prior = {
            "proofread_pairing": "temporal-overlap",
            "proofread_completed_windows": 1,
            "proofread_total_windows": 1,
            "proofread_segments": [seg(0, 1000, "我们讲矩阵的乘法")],
            "clean_ops_plan": [{"start_ms": 0, "end_ms": 1000, "dels": ["嗯"]}],
        }
        sink: list[dict] = []
        with patch("courselens_worker.llm._chat", side_effect=AssertionError("no calls")):
            result = proofread_segments(
                "secret", sensevoice, primary, prior_checkpoint=prior, clean_plan_sink=sink
            )
        self.assertEqual(sink, prior["clean_ops_plan"])
        self.assertEqual(result[0]["text"], "我们讲矩阵的乘法")

    def test_legacy_checkpoint_discards_plan(self):
        # 非信任配对整单重启：计划种子一并作废（绝不混用跨链状态）。
        primary = [seg(0, 1000, "我们讲矩阵的乘法")]
        prior = {
            "proofread_completed_windows": 1,
            "proofread_segments": [seg(0, 1000, "我们讲矩阵的乘法")],
            "clean_ops_plan": [{"start_ms": 0, "end_ms": 1000, "dels": ["嗯"]}],
        }
        sink: list[dict] = []
        response = json.dumps([], ensure_ascii=False)
        with patch("courselens_worker.llm._chat", side_effect=lambda *a, **k: response):
            proofread_segments("secret", [], primary, prior_checkpoint=prior, clean_plan_sink=sink)
        self.assertEqual(sink, [])


class As13AlignmentTests(unittest.TestCase):
    """V010-SUBTITLE-CLEAN-R2 U2 钉：AS13 句首剥离×clean 三键失配缝消除。

    展示链（src/runtime/subtitle_reader.py）服务端剥 cue text 句首 呃/嗯
    后，前端镜像按「clean_text ⊑ 送达 text」复核——worker 落键前把 clean
    对齐到剥首视图（消费侧适配；契约本体零改动）。

    正则同源漂移钉（对 src 正则的逐字比对）平移至主仓
    ``tests/test_subtitle_cue_shape.py::As13MirrorSyncTests``（R6 发布道
    平移：镜像载荷禁私模块依赖，export fail-closed 拦截实证；两侧皆可见
    的跨仓复核归主仓测试面）。
    """

    def test_alignment_mutates_clean_when_worker_kept_head_particle(self):
        # 对偶形态：否定边界守卫保留的句首嗯（「嗯不是这样的…」——嗯 紧邻
        # 不），AS13 剥首后由对齐剥离（与展示面一致；否定语义由剥首拍板
        # 闭集自身承载）。合并删词比（worker 3/17 + 嗯 1 字）帽内通过。
        segments = [seg(0, 2000, "嗯不是这样的同学呢我们下次再讲对吧")]
        stats = apply_display_cleaning(segments, [])
        cue = segments[0]
        self.assertEqual(stats["as13_aligned"], 1)
        self.assertEqual(cue["clean_text"], "不是这样的同学我们下次再讲")
        self.assertLessEqual(
            cue["clean_del_ratio"], CLEAN_TIER1_MAX_DEL_RATIO
        )
        validate_display_cue_v2(cue)

    def test_alignment_fail_closed_on_budget(self):
        # 对齐后删词比超帽 → 整键放弃（宁缺勿错，绝不落半对齐三键）。
        segments = [seg(0, 2000, "呃，讲这个呢")]
        stats = apply_display_cleaning(segments, [])
        self.assertEqual(stats["no_cleaning"], 1)
        for key in ("clean_text", "clean_source", "clean_del_ratio"):
            self.assertNotIn(key, segments[0])

    def test_alignment_noop_without_head_form(self):
        # text 无剥首突变 → 三键原样（对齐零触碰）。
        segments = [seg(0, 2000, "然后呢我们讲矩阵的乘法的运算规则")]
        stats = apply_display_cleaning(segments, [])
        self.assertEqual(stats["as13_aligned"], 0)
        self.assertEqual(segments[0]["clean_text"], "我们讲矩阵的乘法的运算规则")


class RunnerStageTests(unittest.TestCase):
    """runner 终界接线钉：_apply_display_cleaning_stage 就地落键+遥测计数。"""

    def test_stage_attaches_trio_and_counts(self):
        from courselens_worker.runner import _apply_display_cleaning_stage

        value = {"segments": [
            seg(0, 2000, "然后呢我们讲矩阵的乘法的运算规则"),
            seg(2000, 4000, "判别式大于零时方程有两个不相等的实数根。"),
        ]}
        plan = [{"start_ms": 0, "end_ms": 2000, "dels": ["然后呢"]}]
        _apply_display_cleaning_stage(value, plan)
        first = value["segments"][0]
        self.assertEqual(first["clean_source"], "tier2_llm")
        validate_display_cue_v2(first)
        self.assertNotIn("clean_text", value["segments"][1])  # 无删除不落键

    def test_stage_without_key_runs_tier1_only(self):
        # 未配置 DeepSeek 的用户=Tier1 only（现状链零回归）。
        from courselens_worker.runner import _apply_display_cleaning_stage

        value = {"segments": [seg(0, 2000, "然后呢我们讲矩阵的乘法的运算规则")]}
        _apply_display_cleaning_stage(value, [])
        self.assertEqual(value["segments"][0]["clean_source"], "tier1_rule")


class ContractIntegrationTests(unittest.TestCase):
    """shared 契约消费钉：schema/预算帽/降级取值三面。"""

    def test_trio_passes_shared_schema(self):
        segments = [seg(0, 2000, "然后呢我们讲矩阵的乘法对吧")]
        apply_display_cleaning(segments, [{"start_ms": 0, "end_ms": 2000, "dels": ["然后呢"]}])
        validate_display_cue_v2(segments[0])  # 不抛=绿

    def test_budget_caps_match_contract_constants(self):
        self.assertEqual(CLEAN_TIER1_MAX_DEL_RATIO, 0.3)
        self.assertEqual(CLEAN_TIER2_MAX_DEL_RATIO, 0.2)

    def test_display_text_prefers_clean_falls_back_verbatim(self):
        cue = seg(0, 2000, "然后呢我们讲矩阵", clean_text="我们讲矩阵",
                  clean_source="tier2_llm", clean_del_ratio=2.0 / 8.0)
        self.assertEqual(display_text(cue), "我们讲矩阵")
        broken = seg(0, 2000, "然后呢我们讲矩阵", clean_text="完全不同的文本",
                     clean_source="tier2_llm", clean_del_ratio=0.5)
        self.assertEqual(display_text(broken), "然后呢我们讲矩阵")  # 坏形状降级 verbatim

    def test_sanitize_strips_invalid_trio(self):
        from shared.subtitle_display_contract import sanitize_display_fields

        cue = seg(0, 2000, "然后呢我们讲矩阵", clean_text="被改写的文本",
                  clean_source="tier2_llm", clean_del_ratio=0.1)
        clean = sanitize_display_fields(cue)
        self.assertNotIn("clean_text", clean)
        self.assertEqual(clean["text"], "然后呢我们讲矩阵")


class ByteIdentityAndDistributionTests(unittest.TestCase):
    """V010-SUBTITLE-CLEAN-R2 钉：全管线 byte-identity 恒等+删词比分布不变量。"""

    # 合成垫词密集语料（V010-SUBTITLE-D4 样本A 同族通识虚构，零真实语料）。
    _SYNTHETIC_CORPUS = (
        "呃今天呢我们来讲这个级数收敛的判别法对吧",
        "就是说呢先看这个正项级数啊",
        "比较判别法呢就是说呃两个级数比一比对吧",
        "然后呢收敛的级数呢它有这个好的性质",
        "呃比值判别法也就是达朗贝尔判别法懂吧",
        "就是后一项比前一项然后呢取这个极限",
        "呃极限小于一呢它就收敛对不对",
        "然后呢大于一呢就是说它发散啊",
        "这个呢根值判别法就是说柯西判别法",
        "呃就是n次开方然后呢取极限啊",
        "然后呢积分判别法呢就是说看这个反常积分",
        "呃收敛还是发散这个呢跟p有关对吧",
        "好今天就讲到这里然后呢下节课讲幂级数",
    )

    def test_full_pipeline_byte_identity_except_trio(self):
        # 忠实层构造性保证（byte-identity 恒等钉）：清洗只新增三键，剥键
        # 后快照与拍前全等（text/时间轴/全部既有字段逐字节恒等）。
        segments = [
            seg(0, 2000, "呃然后呢我们讲矩阵的乘法对吧"),
            seg(2000, 4000, "判别式大于零时方程有两个不相等的实数根。"),
        ]
        snapshot = json.dumps(segments, ensure_ascii=False)
        apply_display_cleaning(segments, [])
        stripped = [
            {k: v for k, v in item.items()
             if k not in ("clean_text", "clean_source", "clean_del_ratio")}
            for item in segments
        ]
        self.assertEqual(json.dumps(stripped, ensure_ascii=False), snapshot)

    def test_del_ratio_distribution_within_cap(self):
        # 删词比分布不变量：合成语料任意清洗结果删词比 ≤ Tier1 帽（超帽
        # =不落键）；U1 贪心后清洗覆盖面只增不减（至少 10/13 行受益）。
        from shared.subtitle_display_contract import clean_del_ratio_of

        ratios = []
        for text in self._SYNTHETIC_CORPUS:
            segments = [seg(0, 2000, text)]
            apply_display_cleaning(segments, [])
            cue = segments[0]
            if "clean_text" in cue:
                ratio = clean_del_ratio_of(cue["text"], cue["clean_text"])
                self.assertLessEqual(ratio, CLEAN_TIER1_MAX_DEL_RATIO)
                ratios.append(ratio)
        self.assertGreaterEqual(len(ratios), 10)


if __name__ == "__main__":
    unittest.main()
