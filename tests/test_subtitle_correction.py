"""Bounded evidence-aware subtitle correction: temporal pairing and fail-closed application."""

import json
import unittest
from unittest.mock import patch

from courselens_worker.formats import to_srt, to_vtt
from courselens_worker.llm import (
    LLMError,
    _CORRECTION_STATUSES,
    _pair_alternates,
    proofread_segments,
)


def seg(start_ms, end_ms, text, **extra):
    value = {"start_ms": start_ms, "end_ms": end_ms, "text": text}
    value.update(extra)
    return value


def run_proofread(sensevoice, primary, responses, *, prior=None):
    """Run proofread_segments with a mocked _chat; returns (result, user_payloads, chat).

    `responses` is either the fixed string returned by every _chat call, or a
    list of per-call return strings.
    """
    payloads = []

    def fake_chat(api_key, messages, **kwargs):
        payloads.append(json.loads(messages[1]["content"]))
        if isinstance(responses, list):
            return responses[min(len(payloads) - 1, len(responses) - 1)]
        return responses

    with patch("courselens_worker.llm._chat", side_effect=fake_chat) as chat:
        result = proofread_segments("secret", sensevoice, primary, prior_checkpoint=prior)
    return result, payloads, chat


class PairAlternatesTests(unittest.TestCase):
    def test_overlap_preferred_over_nearest(self):
        primaries = [seg(0, 1000, "主")]
        alternates = [seg(1500, 2500, "近"), seg(100, 800, "重叠")]
        self.assertEqual(_pair_alternates(primaries, alternates), [alternates[1]])

    def test_larger_overlap_wins(self):
        primaries = [seg(0, 1000, "主")]
        alternates = [seg(0, 400, "短"), seg(400, 1000, "长")]
        self.assertEqual(_pair_alternates(primaries, alternates), [alternates[1]])

    def test_ties_prefer_earlier_alternate_and_are_deterministic(self):
        primaries = [seg(0, 1000, "主")]
        alternates = [seg(-500, 1500, "先"), seg(-500, 1500, "同")]
        first = _pair_alternates(primaries, alternates)
        self.assertEqual(first, [alternates[0]])
        self.assertEqual(_pair_alternates(primaries, alternates), first)

    def test_nearest_within_bound_but_not_beyond(self):
        primaries = [seg(5000, 6000, "主")]
        near = [seg(3000, 4000, "近")]
        far = [seg(1000, 2000, "远")]
        self.assertEqual(_pair_alternates(primaries, near), [near[0]])
        self.assertEqual(_pair_alternates(primaries, far), [None])

    def test_one_alternate_may_support_several_primaries(self):
        primaries = [seg(0, 1000, "一"), seg(1200, 2000, "二")]
        alternates = [seg(0, 2200, "长参考")]
        self.assertEqual(
            _pair_alternates(primaries, alternates),
            [alternates[0], alternates[0]],
        )

    def test_pairing_ignores_input_order(self):
        primaries = [seg(0, 1000, "主")]
        alternates = [seg(5000, 6000, "远"), seg(0, 1000, "合")]
        self.assertEqual(_pair_alternates(primaries, alternates), [alternates[1]])

    def test_no_alternates_leaves_every_primary_unmatched(self):
        primaries = [seg(0, 1000, "主")]
        self.assertEqual(_pair_alternates(primaries, []), [None])


class CorrectionTests(unittest.TestCase):
    def test_unequal_counts_pair_by_time_without_index_drift(self):
        primary = [seg(0, 1000, "甲说第一句"), seg(2000, 3000, "乙说第二句")]
        sensevoice = [
            seg(100, 900, "甲第一句"),
            seg(1300, 1800, "插入"),
            seg(2100, 2900, "乙第二句"),
        ]
        result, payloads, _ = run_proofread(sensevoice, primary, "[]")
        self.assertEqual([item["alt"] for item in payloads[0]], ["甲第一句", "乙第二句"])
        self.assertEqual(len(result), 2)
        self.assertEqual([item["text"] for item in result], ["甲说第一句", "乙说第二句"])
        self.assertEqual([item["start_ms"] for item in result], [0, 2000])

    def test_wire_payload_is_bounded_pairs_with_ids_and_anchors(self):
        primary = [seg(0, 1000, "第一句")]
        sensevoice = [seg(0, 1000, "参考")]
        _, payloads, _ = run_proofread(sensevoice, primary, "[]")
        self.assertEqual(
            payloads[0],
            [{"id": "p0", "start_ms": 0, "end_ms": 1000, "text": "第一句", "alt": "参考"}],
        )

    def test_unmatched_primary_falls_back_to_raw_text(self):
        primary = [seg(0, 1000, "只有主候选")]
        sensevoice = [seg(50000, 51000, "远处片段")]
        result, _, _ = run_proofread(sensevoice, primary, "[]")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "只有主候选")
        self.assertEqual(result[0]["correction"], "unpaired")
        self.assertEqual((result[0]["start_ms"], result[0]["end_ms"]), (0, 1000))

    def test_accepted_replacement_preserves_anchors_and_evidence(self):
        primary = [seg(
            1000, 2000, "温度是25摄氐度",
            tokens=[{"text": "温", "start_ms": 1000, "end_ms": 1050}],
            segment_id="seg-1",
            source_hash="a" * 64,
            provenance={"producer": "p"},
            lang="zh",
        )]
        sensevoice = [seg(1000, 2000, "温度是25摄氏度")]
        response = json.dumps([{"id": "p0", "old": "摄氐", "new": "摄氏"}])
        result, _, _ = run_proofread(sensevoice, primary, response)
        self.assertEqual(result[0]["text"], "温度是25摄氏度")
        self.assertEqual(result[0]["correction"], "applied")
        self.assertIn(result[0]["correction"], _CORRECTION_STATUSES)
        self.assertEqual((result[0]["start_ms"], result[0]["end_ms"]), (1000, 2000))
        self.assertNotIn("tokens", result[0])
        self.assertNotIn("segment_id", result[0])
        self.assertEqual(result[0]["source_hash"], "a" * 64)
        self.assertEqual(result[0]["provenance"], {"producer": "p"})
        self.assertEqual(result[0]["lang"], "zh")

    def test_uncorrected_pair_keeps_tokens_and_identity(self):
        primary = [seg(0, 1000, "原文", tokens=[{"t": 1}], segment_id="seg-1")]
        sensevoice = [seg(0, 1000, "参考")]
        result, _, _ = run_proofread(sensevoice, primary, "[]")
        self.assertEqual(result[0]["tokens"], [{"t": 1}])
        self.assertEqual(result[0]["segment_id"], "seg-1")
        self.assertNotIn("correction", result[0])

    def test_unknown_target_ids_are_ignored(self):
        primary = [seg(0, 1000, "原文片段")]
        sensevoice = [seg(0, 1000, "参考")]
        response = json.dumps([
            {"id": "p99", "old": "原文", "new": "改写"},
            {"id": "p20", "old": "原文", "new": "改写"},
        ])
        result, _, _ = run_proofread(sensevoice, primary, response)
        self.assertEqual(result[0]["text"], "原文片段")
        self.assertNotIn("correction", result[0])

    def test_duplicate_proposals_for_the_same_span_are_ignored(self):
        primary = [seg(0, 1000, "目标词在这里")]
        sensevoice = [seg(0, 1000, "参考")]
        response = json.dumps([
            {"id": "p0", "old": "目标词", "new": "目标句"},
            {"id": "p0", "old": "目标词", "new": "别的改法"},
            {"id": "p0", "old": "目标句", "new": "第三种"},
        ])
        result, _, _ = run_proofread(sensevoice, primary, response)
        self.assertEqual(result[0]["text"], "第三种在这里")
        self.assertEqual(result[0]["correction"], "applied")

    def test_failed_first_op_does_not_block_a_distinct_later_op(self):
        primary = [seg(0, 1000, "目标词在这里")]
        sensevoice = [seg(0, 1000, "参考")]
        response = json.dumps([
            {"id": "p0", "old": "不存在", "new": "x"},
            {"id": "p0", "old": "目标词", "new": "目标句"},
        ])
        result, _, _ = run_proofread(sensevoice, primary, response)
        self.assertEqual(result[0]["text"], "目标句在这里")
        self.assertEqual(result[0]["correction"], "applied")

    def test_ambiguous_old_text_is_rejected(self):
        primary = [seg(0, 1000, "AAA BBB AAA")]
        sensevoice = [seg(0, 1000, "参考")]
        response = json.dumps([{"id": "p0", "old": "AAA", "new": "C"}])
        result, _, _ = run_proofread(sensevoice, primary, response)
        self.assertEqual(result[0]["text"], "AAA BBB AAA")
        self.assertEqual(result[0]["correction"], "rejected-ambiguous")

    def test_empty_result_is_rejected(self):
        primary = [seg(0, 1000, "对")]
        sensevoice = [seg(0, 1000, "参考")]
        response = json.dumps([{"id": "p0", "old": "对", "new": ""}])
        result, _, _ = run_proofread(sensevoice, primary, response)
        self.assertEqual(result[0]["text"], "对")
        self.assertEqual(result[0]["correction"], "rejected-budget")

    def test_over_budget_expansion_is_rejected(self):
        primary = [seg(0, 1000, "一二三四")]
        sensevoice = [seg(0, 1000, "参考")]
        response = json.dumps([{"id": "p0", "old": "一", "new": "一二三四五六七八九十"}])
        result, _, _ = run_proofread(sensevoice, primary, response)
        self.assertEqual(result[0]["text"], "一二三四")
        self.assertEqual(result[0]["correction"], "rejected-budget")

    def test_per_op_growth_cap_is_rejected(self):
        primary = [seg(0, 1000, "abc")]
        sensevoice = [seg(0, 1000, "参考")]
        response = json.dumps([{"id": "p0", "old": "a", "new": "x" * 18}])
        result, _, _ = run_proofread(sensevoice, primary, response)
        self.assertEqual(result[0]["text"], "abc")
        self.assertEqual(result[0]["correction"], "rejected-budget")

    def test_protected_number_change_without_support_is_rejected(self):
        primary = [seg(0, 1000, "用了25分钟")]
        sensevoice = [seg(0, 1000, "用了25分钟")]
        response = json.dumps([{"id": "p0", "old": "25", "new": "26"}])
        result, _, _ = run_proofread(sensevoice, primary, response)
        self.assertEqual(result[0]["text"], "用了25分钟")
        self.assertEqual(result[0]["correction"], "rejected-protected")

    def test_alternate_support_allows_protected_number_change(self):
        primary = [seg(0, 1000, "用了25分钟")]
        sensevoice = [seg(0, 1000, "用了26分钟")]
        response = json.dumps([{"id": "p0", "old": "25", "new": "26"}])
        result, _, _ = run_proofread(sensevoice, primary, response)
        self.assertEqual(result[0]["text"], "用了26分钟")
        self.assertEqual(result[0]["correction"], "applied")

    def test_corroborated_negation_deletion_is_rejected(self):
        primary = [seg(0, 1000, "这个方法不能重复使用")]
        sensevoice = [seg(0, 1000, "这个方法不能重复使用")]
        response = json.dumps([{"id": "p0", "old": "不", "new": ""}])
        result, _, _ = run_proofread(sensevoice, primary, response)
        self.assertEqual(result[0]["text"], "这个方法不能重复使用")
        self.assertEqual(result[0]["correction"], "rejected-protected")

    def test_alternate_supported_negation_fix_is_applied(self):
        primary = [seg(0, 1000, "这个方法能重复使用")]
        sensevoice = [seg(0, 1000, "这个方法不能重复使用")]
        response = json.dumps([{"id": "p0", "old": "能", "new": "不能"}])
        result, _, _ = run_proofread(sensevoice, primary, response)
        self.assertEqual(result[0]["text"], "这个方法不能重复使用")
        self.assertEqual(result[0]["correction"], "applied")

    def test_formula_change_without_support_is_rejected(self):
        primary = [seg(0, 1000, "质能方程E=mc2很著名")]
        sensevoice = [seg(0, 1000, "质能方程E=mc2很著名")]
        response = json.dumps([{"id": "p0", "old": "E=mc2", "new": "E=MC2"}])
        result, _, _ = run_proofread(sensevoice, primary, response)
        self.assertEqual(result[0]["text"], "质能方程E=mc2很著名")
        self.assertEqual(result[0]["correction"], "rejected-protected")

    def test_unit_change_without_support_is_rejected(self):
        primary = [seg(0, 1000, "长度约3km")]
        sensevoice = [seg(0, 1000, "长度约3km")]
        response = json.dumps([{"id": "p0", "old": "3km", "new": "3米"}])
        result, _, _ = run_proofread(sensevoice, primary, response)
        self.assertEqual(result[0]["text"], "长度约3km")
        self.assertEqual(result[0]["correction"], "rejected-protected")

    def test_malformed_json_raises_llm_error(self):
        primary = [seg(0, 1000, "原文")]
        sensevoice = [seg(0, 1000, "参考")]
        with self.assertRaises(LLMError):
            run_proofread(sensevoice, primary, "不是JSON")

    def test_non_array_response_raises_llm_error(self):
        primary = [seg(0, 1000, "原文")]
        sensevoice = [seg(0, 1000, "参考")]
        with self.assertRaises(LLMError):
            run_proofread(sensevoice, primary, json.dumps({"index": 0, "text": "整段改写"}))

    def test_malformed_ops_fail_closed_per_pair(self):
        primary = [seg(0, 1000, "原文")]
        sensevoice = [seg(0, 1000, "参考")]
        response = json.dumps(["junk", 42, {"id": "p0"}, {"id": "p0", "old": "原", "new": "改"}])
        result, _, _ = run_proofread(sensevoice, primary, response)
        self.assertEqual(result[0]["text"], "改文")
        self.assertEqual(result[0]["correction"], "applied")

    def test_malformed_op_without_valid_followup_keeps_raw_text(self):
        primary = [seg(0, 1000, "原文")]
        sensevoice = [seg(0, 1000, "参考")]
        response = json.dumps([{"id": "p0", "old": "", "new": "改"}])
        result, _, _ = run_proofread(sensevoice, primary, response)
        self.assertEqual(result[0]["text"], "原文")
        self.assertEqual(result[0]["correction"], "rejected-shape")

    def test_corrected_segments_render_to_srt_and_vtt(self):
        primary = [seg(1000, 2000, "温度是25摄氐度")]
        sensevoice = [seg(1000, 2000, "温度是25摄氏度")]
        response = json.dumps([{"id": "p0", "old": "摄氐", "new": "摄氏"}])
        result, _, _ = run_proofread(sensevoice, primary, response)
        srt = to_srt(result)
        self.assertIn("00:00:01,000 --> 00:00:02,000", srt)
        self.assertIn("温度是25摄氏度", srt)
        self.assertIn("WEBVTT", to_vtt(result))


class ProofreadTelemetryTests(unittest.TestCase):
    """夜10-C T10：校对链收口遥测钉——计数与闭集词，零内容泄漏。"""

    def test_final_stage_line_carries_counters_without_content(self):
        lines = []
        primary = [seg(0, 1000, "温度是25摄氐度"), seg(3000, 4000, "第二句原文")]
        sensevoice = [seg(0, 1000, "温度是25摄氏度")]
        response = json.dumps([{"id": "p0", "old": "摄氐", "new": "摄氏"}])
        with patch("courselens_worker.llm._emit_telemetry", side_effect=lines.append):
            result, _, _ = run_proofread(sensevoice, primary, response)
        finals = [line for line in lines if line.startswith("stage=proofread ")]
        self.assertEqual(len(finals), 1, "恰一行收口遥测")
        final = finals[0]
        self.assertIn("windows=1/1", final)
        self.assertIn("window_retries=0", final)
        self.assertIn("applied=1", final)
        self.assertIn("none=1", final, "未提案段归入 none 桶")
        for segment in result:
            self.assertNotIn(str(segment["text"]), final, "字幕内容绝不进遥测行")
            evidence = str(segment.get("evidence_id") or "")
            if evidence:
                self.assertNotIn(evidence, final)
        self.assertNotIn("secret", final)

    def test_window_retry_emits_counter_line(self):
        lines = []
        calls = {"count": 0}
        primary = [seg(0, 1000, "原文")]
        sensevoice = [seg(0, 1000, "参考")]

        def flaky_chat(api_key, messages, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise LLMError("transient provider error")
            return json.dumps([])

        with (
            patch("courselens_worker.llm._chat", side_effect=flaky_chat),
            patch("courselens_worker.llm._emit_telemetry", side_effect=lines.append),
            patch("courselens_worker.llm.time.sleep"),
        ):
            result = proofread_segments("secret", sensevoice, primary)
        self.assertNotIn("correction", result[0])
        retries = [line for line in lines if line.startswith("stage=proofread-window-retry ")]
        self.assertEqual(retries, ["stage=proofread-window-retry attempt=1"])
        self.assertTrue(any("window_retries=1" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
