"""P11-CONTRACT-1 PKG-A + THINK-LADDER-2 两段式：LLM-as-judge 离线质检钉面。

覆盖：四条提示词 ≤300 帽+底线词+闭集枚举（flash 筛/思考裁各两 mode）、
thinking env 闭集覆写与输出帽配对（筛段恒 disabled 8192；裁段随 vars 档
16384）、两段级联（零 findings 跳裁/裁段 confirm-deny/裁段失败降级保留筛
段）、_salvage_judge_findings 两级抢救、字幕/总结 mode 闭集校验逐项（越界
target/dimension/code/severity/position 丢弃+计数、附加字段剥除=零自由文本、
数量帽 warn 优先截断）、单 mode null、双 mode 无效 fail-closed、usage_sink
流水（失败尝试不丢账+stage=flash|adj 段标签）、429/授权类不入重试集、传输
失败原样上抛（无本地降级）、遥测仅计数零内容、runner 分支（outputs/
metrics/warnings/诚实失败/裁段降级警告）、协议层 quality_judge 加性注册
roundtrip（job intake 与 result 两端闭集）。
"""

from __future__ import annotations

import json
import os
import time
import unittest
from unittest.mock import patch

from courselens_worker.llm import (
    JUDGE_THINKING,
    JUDGE_THINKING_ENV,
    LLMError,
    _JUDGE_ADJ_SUBTITLE_PROMPT,
    _JUDGE_ADJ_SUMMARY_PROMPT,
    _JUDGE_BOUNDARY_MAX_PAIRS,
    _JUDGE_BOUNDARY_PROMPT,
    _JUDGE_FINDINGS_CAP,
    _JUDGE_FLASH_SUBTITLE_PROMPT,
    _JUDGE_FLASH_SUMMARY_PROMPT,
    _resolve_judge_thinking,
    _salvage_judge_findings,
    _validate_boundary_pairs,
    _validate_judge_subtitle,
    _validate_judge_summary,
    judge_lecture_quality,
    judge_term_boundaries,
)
from courselens_worker.protocol import (
    JOB_SCHEMA,
    PROTOCOL_VERSION,
    RESULT_SCHEMA,
    generate_box_keypair,
    generate_signing_keypair,
    open_job,
    open_result,
    seal_job,
    seal_result,
)
from courselens_worker.runner import process_job

_SEGMENTS = [
    {"index": 0, "start_ms": 0, "end_ms": 30_000, "text": "链式聚合反应逐步进行。"},
    {"index": 1, "start_ms": 30_000, "end_ms": 60_000, "text": "凝胶化出现在临界转化率。"},
]
_SUBTITLE_SAMPLE = {"total_segments": 2, "segments": _SEGMENTS}
_SUMMARY_PACK = {
    "markdown": "本讲讲链式聚合与凝胶化。",
    "chapters": [
        {"title": "开场", "summary": "链式聚合", "start_ms": 0},
        {"title": "凝胶化", "summary": "临界转化率", "start_ms": 30_000},
    ],
    "key_takeaways": ["链式聚合逐步进行", "凝胶化有临界转化率"],
    "transcript": _SEGMENTS,
}
_SUBTITLE_FINDINGS = {
    "findings": [
        {"target": "segment", "position": 1, "dimension": "term_fidelity",
         "code": "homophone_suspect", "severity": "warn"},
    ],
}
_SUMMARY_FINDINGS = {
    "findings": [
        {"target": "chapter", "position": 1, "dimension": "alignment",
         "code": "chapter_mislabel", "severity": "warn"},
        {"target": "body", "position": 0, "dimension": "factuality",
         "code": "no_source_support", "severity": "info"},
    ],
}


def _judge_responder(subtitle_payloads, summary_payloads):
    """按 system 提示词分流的替身（两段式四提示词，mode 内筛/裁共用队列）：
    各自重放末项供重试；每调用落一条 usage 流水（与 test_review_views 同
    法），供 usage_sink 账面断言。"""
    calls = {"subtitle": 0, "summary": 0}

    def _chat(api_key, messages, **kwargs):
        from courselens_worker import llm as llm_module

        with llm_module._USAGE_LOCK:
            llm_module._CALL_LOG.append({
                "prompt_tokens": 10, "completion_tokens": 5,
                "reasoning_tokens": 0, "prompt_cache_hit_tokens": 0,
                "latency_ms": 10, "thinking": None,
            })
        system = messages[0]["content"]
        if system in (_JUDGE_FLASH_SUBTITLE_PROMPT, _JUDGE_ADJ_SUBTITLE_PROMPT):
            payloads = subtitle_payloads or ["垃圾一"]
            index = min(calls["subtitle"], len(payloads) - 1)
            calls["subtitle"] += 1
            return payloads[index]
        assert system in (_JUDGE_FLASH_SUMMARY_PROMPT, _JUDGE_ADJ_SUMMARY_PROMPT), system[:40]
        payloads = summary_payloads or ["垃圾一"]
        index = min(calls["summary"], len(payloads) - 1)
        calls["summary"] += 1
        return payloads[index]

    return _chat, calls


class PromptPinTests(unittest.TestCase):
    def test_prompts_bounded_with_bottom_lines_and_closed_enums(self):
        for prompt, needles in (
            (_JUDGE_FLASH_SUBTITLE_PROMPT, (
                "存疑即标", "宁多勿漏", "没有问题输出空 findings", "禁止编造输入外的问题",
                "只依据输入", "零自由文本", "segment", "term_fidelity", "readability",
                "glossary_violation", "homophone_suspect", "term_inconsistent",
                "broken_flow", "garbled", "warn", "info",
            )),
            (_JUDGE_ADJ_SUBTITLE_PROMPT, (
                "确认或否决", "补漏", "没有问题输出空 findings", "禁止编造输入外的问题",
                "零自由文本", "segment", "term_fidelity", "readability",
                "glossary_violation", "homophone_suspect", "term_inconsistent",
                "broken_flow", "garbled", "warn", "info",
            )),
            (_JUDGE_FLASH_SUMMARY_PROMPT, (
                "存疑即标", "宁多勿漏", "没有问题输出空 findings", "禁止编造输入外的问题",
                "只依据输入", "零自由文本", "chapter", "takeaway", "body", "factuality",
                "alignment", "readability", "no_source_support",
                "contradicts_source", "chapter_mislabel", "duplicate_content",
                "empty_section", "warn", "info",
            )),
            (_JUDGE_ADJ_SUMMARY_PROMPT, (
                "确认或否决", "补漏", "没有问题输出空 findings", "禁止编造输入外的问题",
                "零自由文本", "chapter", "takeaway", "body", "factuality",
                "alignment", "readability", "no_source_support",
                "contradicts_source", "chapter_mislabel", "duplicate_content",
                "empty_section", "warn", "info",
            )),
        ):
            self.assertLessEqual(len(prompt), 300)
            for needle in needles:
                self.assertIn(needle, prompt)


class ThinkingTests(unittest.TestCase):
    def test_closed_env_override(self):
        cases = {
            "": JUDGE_THINKING,
            "disabled": {"type": "disabled"},
            " default ": None,
            "provider-default": None,
            "low": {"type": "enabled", "reasoning_effort": "low"},
            "HIGH": {"type": "enabled", "reasoning_effort": "high"},
            "max": {"type": "enabled", "reasoning_effort": "max"},
            "bogus": JUDGE_THINKING,
        }
        for raw, expected in cases.items():
            with patch.dict(os.environ, {JUDGE_THINKING_ENV: raw}, clear=False):
                self.assertEqual(_resolve_judge_thinking(), expected, repr(raw))

    def test_flash_stage_always_disabled_with_base_cap(self):
        seen: list[dict] = []

        def _chat(api_key, messages, *, max_tokens=8192, thinking=None, timeout=180.0):
            seen.append({"max_tokens": max_tokens, "thinking": thinking})
            if len(seen) == 1:
                return json.dumps(_SUBTITLE_FINDINGS)
            return json.dumps({"findings": []})

        with patch("courselens_worker.llm._chat", side_effect=_chat):
            judge_lecture_quality("k", subtitle_sample=_SUBTITLE_SAMPLE)
        self.assertEqual(
            seen[0], {"max_tokens": 8192, "thinking": {"type": "disabled"}},
            "筛段恒关思考、恒基础帽（档位旋钮只作用于裁段）",
        )

    def test_thinking_tier_scales_adjudication_cap_only(self):
        seen: list[dict] = []

        def _chat(api_key, messages, *, max_tokens=8192, thinking=None, timeout=180.0):
            seen.append({"max_tokens": max_tokens, "thinking": thinking, "timeout": timeout})
            if len(seen) == 1:
                return json.dumps(_SUBTITLE_FINDINGS)
            return json.dumps({"findings": []})

        with patch("courselens_worker.llm._chat", side_effect=_chat), \
                patch.dict(os.environ, {JUDGE_THINKING_ENV: "high"}, clear=False):
            judge_lecture_quality("k", subtitle_sample=_SUBTITLE_SAMPLE)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[1]["max_tokens"], 16384)
        self.assertEqual(seen[1]["thinking"], {"type": "enabled", "reasoning_effort": "high"})
        self.assertEqual(seen[1]["timeout"], 240.0, "思考档裁段同享 N16 长输出预算")


class SalvageTests(unittest.TestCase):
    def test_whole_object_recovered_from_chatty_prefix(self):
        raw = (
            "好的，结果如下 "
            '{"findings": [{"target": "segment", "position": 0, '
            '"dimension": "readability", "code": "garbled", "severity": "info"}]}'
            " 以上。"
        )
        recovered = _salvage_judge_findings(raw)
        self.assertEqual(len(recovered["findings"]), 1)

    def test_truncated_reply_recovers_completed_flat_findings(self):
        two = {"findings": [
            {"target": "segment", "position": 0, "dimension": "readability",
             "code": "garbled", "severity": "info"},
            {"target": "segment", "position": 1, "dimension": "term_fidelity",
             "code": "homophone_suspect", "severity": "warn"},
        ]}
        text = json.dumps(two)
        cut = text.rfind("{")
        recovered = _salvage_judge_findings(text[:cut] + '{"target": "seg')
        self.assertEqual(len(recovered["findings"]), 1)
        self.assertEqual(recovered["findings"][0]["position"], 0)

    def test_no_objects_returns_none(self):
        self.assertIsNone(_salvage_judge_findings("完全没有对象"))


class SubtitleValidationTests(unittest.TestCase):
    def test_valid_entry_passes_with_frozen_keys_only(self):
        candidate = {"findings": [{
            "target": "segment", "position": 1, "dimension": "term_fidelity",
            "code": "homophone_suspect", "severity": "warn",
            "quote": "原文引用必须剥除", "note": "自由文本必须剥除",
        }]}
        mode, dropped, truncated = _validate_judge_subtitle(candidate, sample_size=2)
        self.assertEqual(mode["sample_size"], 2)
        self.assertEqual(mode["targets_total"], 2)
        self.assertEqual(sorted(mode["findings"][0]), [
            "code", "dimension", "position", "severity", "target"])
        self.assertEqual((dropped, truncated), (0, 0))

    def test_out_of_set_entries_dropped_and_counted(self):
        candidate = {"findings": [
            {"target": "chapter", "position": 0, "dimension": "term_fidelity",
             "code": "garbled", "severity": "warn"},
            {"target": "segment", "position": 0, "dimension": "factuality",
             "code": "garbled", "severity": "warn"},
            {"target": "segment", "position": 0, "dimension": "readability",
             "code": "no_source_support", "severity": "warn"},
            {"target": "segment", "position": 0, "dimension": "readability",
             "code": "garbled", "severity": "error"},
            {"target": "segment", "position": 5, "dimension": "readability",
             "code": "garbled", "severity": "warn"},
            {"target": "segment", "position": -1, "dimension": "readability",
             "code": "garbled", "severity": "warn"},
            {"target": "segment", "position": True, "dimension": "readability",
             "code": "garbled", "severity": "warn"},
            {"target": "segment", "position": "1", "dimension": "readability",
             "code": "garbled", "severity": "warn"},
            {"target": "segment", "position": 1, "dimension": "readability",
             "code": "garbled", "severity": "info"},
        ]}
        mode, dropped, truncated = _validate_judge_subtitle(candidate, sample_size=2)
        self.assertEqual(dropped, 8)
        self.assertEqual([item["position"] for item in mode["findings"]], [1])
        self.assertEqual((truncated, mode["targets_total"]), (0, 2))

    def test_non_list_or_non_dict_findings_null_the_mode(self):
        for bad in ({"findings": "oops"}, {"findings": [1, 2]}, {"findings": None}, {}, "junk"):
            mode, dropped, truncated = _validate_judge_subtitle(bad, sample_size=2)
            self.assertIsNone(mode)
            self.assertEqual((dropped, truncated), (0, 0))

    def test_cap_truncates_warn_first(self):
        findings = [
            {"target": "segment", "position": i % 30, "dimension": "readability",
             "code": "broken_flow", "severity": "info"}
            for i in range(20)
        ] + [
            {"target": "segment", "position": i % 30, "dimension": "term_fidelity",
             "code": "term_inconsistent", "severity": "warn"}
            for i in range(10)
        ]
        mode, dropped, truncated = _validate_judge_subtitle(
            {"findings": findings}, sample_size=40)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(mode["findings"]), _JUDGE_FINDINGS_CAP)
        self.assertEqual(truncated, 30 - _JUDGE_FINDINGS_CAP)
        self.assertEqual(
            [item["severity"] for item in mode["findings"]],
            ["warn"] * 10 + ["info"] * (_JUDGE_FINDINGS_CAP - 10),
        )


class SummaryValidationTests(unittest.TestCase):
    def test_targets_total_and_position_bounds(self):
        candidate = {"findings": [
            {"target": "chapter", "position": 0, "dimension": "alignment",
             "code": "chapter_mislabel", "severity": "warn"},
            {"target": "chapter", "position": 2, "dimension": "alignment",
             "code": "chapter_mislabel", "severity": "warn"},
            {"target": "takeaway", "position": 1, "dimension": "factuality",
             "code": "contradicts_source", "severity": "info"},
            {"target": "takeaway", "position": 2, "dimension": "factuality",
             "code": "contradicts_source", "severity": "info"},
            {"target": "body", "position": 0, "dimension": "factuality",
             "code": "no_source_support", "severity": "warn"},
            {"target": "body", "position": 1, "dimension": "factuality",
             "code": "no_source_support", "severity": "warn"},
            {"target": "segment", "position": 0, "dimension": "factuality",
             "code": "garbled", "severity": "warn"},
        ]}
        mode, dropped, _ = _validate_judge_summary(
            candidate, chapters_total=2, takeaways_total=2)
        self.assertEqual(mode["targets_total"], 5, "2 章 + 2 条 + 1 正文")
        self.assertEqual(dropped, 4)
        self.assertEqual(
            [(item["target"], item["position"]) for item in mode["findings"]],
            [("chapter", 0), ("takeaway", 1), ("body", 0)],
        )

    def test_empty_pack_bounds(self):
        mode, dropped, _ = _validate_judge_summary(
            {"findings": [{"target": "takeaway", "position": 0,
                           "dimension": "factuality", "code": "contradicts_source",
                           "severity": "warn"}]},
            chapters_total=0, takeaways_total=0)
        self.assertEqual(mode["targets_total"], 1)
        self.assertEqual(dropped, 1)
        self.assertEqual(mode["findings"], [])


class JudgeCallTests(unittest.TestCase):
    def _run(self, subtitle_payloads=(), summary_payloads=(), *,
             subtitle_sample=None, summary_pack=None, glossary=()):
        if subtitle_sample is None:
            subtitle_sample = _SUBTITLE_SAMPLE
        _chat, calls = _judge_responder(list(subtitle_payloads), list(summary_payloads))
        usage: list[dict] = []
        lines: list[str] = []
        with patch("courselens_worker.llm._chat", side_effect=_chat), \
                patch("courselens_worker.llm._emit_telemetry", side_effect=lines.append), \
                patch("time.sleep"):
            report = judge_lecture_quality(
                "k", subtitle_sample=subtitle_sample, summary_pack=summary_pack,
                glossary=glossary, usage_sink=usage)
        return report, calls, usage, lines

    def test_both_modes_happy_path_shape_telemetry_and_ledger(self):
        report, calls, usage, lines = self._run(
            [json.dumps(_SUBTITLE_FINDINGS)], [json.dumps(_SUMMARY_FINDINGS)],
            summary_pack=_SUMMARY_PACK, glossary=("链式聚合",))
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["subtitle"]["sample_size"], 2)
        self.assertEqual(report["subtitle"]["targets_total"], 2)
        self.assertEqual(len(report["subtitle"]["findings"]), 1)
        self.assertEqual(report["summary"]["targets_total"], 5)
        self.assertEqual(len(report["summary"]["findings"]), 2)
        self.assertEqual(
            calls, {"subtitle": 2, "summary": 2},
            "两段式：每 mode 筛+裁各一跳（findings 非空才裁）")
        self.assertEqual(len(usage), 4, "每跳一条 usage 流水")
        self.assertEqual(
            [record.get("stage") for record in usage],
            ["flash", "adj", "flash", "adj"], "级联段标签随流水")
        self.assertEqual(
            report["stages"],
            {
                "subtitle": {"flash_findings": 1, "adjudicated": True},
                "summary": {"flash_findings": 2, "adjudicated": True},
            },
            "stages 加性键逐 mode 记筛出数与是否裁")
        final = [line for line in lines if line.startswith("stage=quality-judge ")]
        self.assertEqual(len(final), 1, lines)
        self.assertIn("subtitle_findings=1", final[0])
        self.assertIn("summary_findings=2", final[0])
        self.assertIn("subtitle_flash_findings=1", final[0])
        self.assertIn("subtitle_adjudicated=1", final[0])
        self.assertIn("summary_flash_findings=2", final[0])
        self.assertIn("summary_adjudicated=1", final[0])
        self.assertIn("findings_truncated=0", final[0])
        self.assertIn("findings_invalid_dropped=0", final[0])
        for leaked in ("链式聚合", "凝胶化", "本讲讲", "k"):
            self.assertNotIn(leaked, final[0], "遥测零内容零账号值")

    def test_glossary_travels_data_channel_and_empty_omits_key(self):
        capture: dict = {}

        def _chat(api_key, messages, *, max_tokens=8192, thinking=None, timeout=180.0):
            capture.setdefault("inputs", []).append(json.loads(messages[1]["content"]))
            return json.dumps({"findings": []})

        with patch("courselens_worker.llm._chat", side_effect=_chat):
            judge_lecture_quality(
                "k", subtitle_sample=_SUBTITLE_SAMPLE, glossary=("链式聚合", "凝胶化"))
        with patch("courselens_worker.llm._chat", side_effect=_chat):
            judge_lecture_quality("k", subtitle_sample=_SUBTITLE_SAMPLE)
        first, second = capture["inputs"]
        self.assertEqual(first["glossary"], ["链式聚合", "凝胶化"])
        self.assertNotIn("glossary", second, "空词表省键（:6189 家规）")
        self.assertEqual(len(first["segments"]), 2)

    def test_clean_mode_makes_single_flash_call(self):
        report, calls, _, _ = self._run([json.dumps({"findings": []})])
        self.assertIsNotNone(report["subtitle"])
        self.assertIsNone(report["summary"])
        self.assertEqual(calls, {"subtitle": 1, "summary": 0}, "筛段零 findings→裁段零调用")
        self.assertEqual(
            report["stages"]["subtitle"], {"flash_findings": 0, "adjudicated": False})

    def test_adjudication_input_carries_flash_findings(self):
        capture: dict = {}

        def _chat(api_key, messages, *, max_tokens=8192, thinking=None, timeout=180.0):
            system = messages[0]["content"]
            if system == _JUDGE_FLASH_SUBTITLE_PROMPT:
                return json.dumps(_SUBTITLE_FINDINGS)
            assert system == _JUDGE_ADJ_SUBTITLE_PROMPT, system[:40]
            capture["adj_input"] = json.loads(messages[1]["content"])
            return json.dumps({"findings": []})

        with patch("courselens_worker.llm._chat", side_effect=_chat):
            report = judge_lecture_quality("k", subtitle_sample=_SUBTITLE_SAMPLE)
        self.assertEqual(
            capture["adj_input"]["flash_findings"], _SUBTITLE_FINDINGS["findings"],
            "裁段输入附筛段 findings 供逐条复核")
        self.assertEqual(capture["adj_input"]["segments"], _SEGMENTS)
        self.assertEqual(report["subtitle"]["findings"], [], "裁段 deny 修剪误报")
        self.assertEqual(
            report["stages"]["subtitle"], {"flash_findings": 1, "adjudicated": True})

    def test_adjudication_degrades_to_flash_on_shape_failure(self):
        report, calls, usage, lines = self._run(
            [json.dumps(_SUBTITLE_FINDINGS), "垃圾一"])
        self.assertEqual(len(report["subtitle"]["findings"]), 1, "裁段两败降级保留筛段")
        self.assertEqual(
            report["stages"]["subtitle"], {"flash_findings": 1, "adjudicated": False})
        self.assertEqual(calls, {"subtitle": 3, "summary": 0}, "筛 1+裁 2（attempts=2）")
        self.assertEqual(len(usage), 3, "失败尝试也落账")
        degraded = [line for line in lines if line.startswith("stage=quality-judge-adj-degraded")]
        self.assertEqual(len(degraded), 1, lines)

    def test_one_mode_invalid_nulls_it_and_other_survives(self):
        report, calls, usage, lines = self._run(
            [json.dumps(_SUBTITLE_FINDINGS), "垃圾一"], summary_pack=_SUMMARY_PACK,
            subtitle_sample=_SUBTITLE_SAMPLE, summary_payloads=["垃圾一"])
        self.assertIsNotNone(report["subtitle"])
        self.assertIsNone(report["summary"], "形状两败置 null，不拖垮另一 mode")
        self.assertEqual(calls["summary"], 2, "attempts=2")
        self.assertEqual(len(usage), 5, "字幕筛 1+裁 2+总结筛 2 全落账（失败尝试不丢）")
        retries = [line for line in lines if line.startswith("stage=quality-judge-retry")]
        self.assertEqual(len(retries), 2, lines)
        final = [line for line in lines if line.startswith("stage=quality-judge ")]
        self.assertIn("summary_findings=0", final[0])

    def test_both_modes_invalid_fails_closed(self):
        with self.assertRaises(LLMError):
            self._run(["垃圾一"], ["垃圾二"], summary_pack=_SUMMARY_PACK)


class TransportFailureTests(unittest.TestCase):
    def test_transport_failure_raises_after_retry_no_llm_pending(self):
        attempts = {"n": 0}

        def _chat(api_key, messages, **kwargs):
            attempts["n"] += 1
            raise LLMError("AI request failed: ConnectionError")

        usage: list[dict] = []
        with patch("courselens_worker.llm._chat", side_effect=_chat), \
                patch("time.sleep"), \
                self.assertRaises(LLMError):
            judge_lecture_quality(
                "k", subtitle_sample=_SUBTITLE_SAMPLE, usage_sink=usage)
        self.assertEqual(attempts["n"], 2, "attempts=2 后诚实失败")
        self.assertEqual(usage, [], "传输失败无 usage 流水")

    def test_429_never_retries(self):
        attempts = {"n": 0}

        def _chat(api_key, messages, **kwargs):
            attempts["n"] += 1
            raise LLMError("AI request returned HTTP 429")

        with patch("courselens_worker.llm._chat", side_effect=_chat), \
                patch("time.sleep") as slept, \
                self.assertRaises(LLMError):
            judge_lecture_quality("k", subtitle_sample=_SUBTITLE_SAMPLE)
        self.assertEqual(attempts["n"], 1)
        slept.assert_not_called()

    def test_both_modes_none_raises_value_error(self):
        with self.assertRaises(ValueError):
            judge_lecture_quality("k")


class RunnerBranchTests(unittest.TestCase):
    def _job(self, payload):
        return {
            "schema": JOB_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "task_id": "0123456789abcdef0123456789abcdef",
            "job_kind": "quality_judge",
            "input_hash": "0" * 64,
            "pipeline": {"version": "quality-judge-v1"},
            "payload": payload,
            "secrets": {"deepseek_api_key": "k"},
        }

    def test_branch_reports_outputs_metrics_and_ledger(self):
        report = {
            "schema_version": 1,
            "subtitle": {"sample_size": 2, "targets_total": 2,
                         "findings": list(_SUBTITLE_FINDINGS["findings"])},
            "summary": None,
        }

        def fake_judge(api_key, *, subtitle_sample, summary_pack, glossary, usage_sink):
            self.assertEqual(subtitle_sample, _SUBTITLE_SAMPLE)
            self.assertIsNone(summary_pack)
            self.assertEqual(glossary, ("链式聚合",))
            usage_sink.append({"prompt_tokens": 7, "completion_tokens": 3})
            return report

        with patch("courselens_worker.llm.judge_lecture_quality", side_effect=fake_judge):
            result = process_job(self._job({
                "title": "t", "subtitle_sample": _SUBTITLE_SAMPLE,
                "glossary": ["链式聚合"],
            }))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["outputs"]["quality_judge"], report)
        self.assertEqual(result["metrics"]["deepseek_tokens"], 10)
        self.assertEqual(result["metrics"]["transcript_segments"], 0)
        self.assertEqual(result["warnings"], [])

    def test_branch_counts_transcript_and_marks_mode_invalid(self):
        report = {"schema_version": 1, "subtitle": None, "summary": None}

        def fake_judge(api_key, **kwargs):
            return report

        with patch("courselens_worker.llm.judge_lecture_quality", side_effect=fake_judge):
            result = process_job(self._job({"summary_pack": _SUMMARY_PACK}))
        self.assertEqual(result["warnings"], ["judge_mode_invalid"])
        self.assertEqual(result["metrics"]["transcript_segments"], len(_SEGMENTS))

    def test_branch_flags_adjudication_degradation(self):
        report = {
            "schema_version": 1,
            "subtitle": {"sample_size": 2, "targets_total": 2,
                         "findings": list(_SUBTITLE_FINDINGS["findings"])},
            "summary": None,
            "stages": {"subtitle": {"flash_findings": 1, "adjudicated": False}},
        }

        def fake_judge(api_key, **kwargs):
            return report

        with patch("courselens_worker.llm.judge_lecture_quality", side_effect=fake_judge):
            result = process_job(self._job({"subtitle_sample": _SUBTITLE_SAMPLE}))
        self.assertEqual(result["warnings"], ["judge_adjudication_failed"])
        self.assertEqual(result["outputs"]["quality_judge"], report)

    def test_branch_clean_cascade_has_no_degradation_warning(self):
        report = {
            "schema_version": 1,
            "subtitle": {"sample_size": 2, "targets_total": 2, "findings": []},
            "summary": None,
            "stages": {"subtitle": {"flash_findings": 0, "adjudicated": False}},
        }

        def fake_judge(api_key, **kwargs):
            return report

        with patch("courselens_worker.llm.judge_lecture_quality", side_effect=fake_judge):
            result = process_job(self._job({"subtitle_sample": _SUBTITLE_SAMPLE}))
        self.assertEqual(result["warnings"], [], "零筛出未裁≠降级，不打警告")

    def test_branch_returns_boundary_rulings_and_bills_usage(self):
        report = {
            "schema_version": 1,
            "subtitle": {"sample_size": 2, "targets_total": 2, "findings": []},
            "summary": None,
            "stages": {"subtitle": {"flash_findings": 0, "adjudicated": False}},
        }
        rulings = {"rulings": [], "malformed_pairs": 0, "unruled_pairs": 0}
        seen: dict = {}

        def fake_judge(api_key, **kwargs):
            return report

        def fake_boundary(api_key, *, pairs, glossary, usage_sink):
            seen["pairs"] = pairs
            seen["glossary"] = glossary
            usage_sink.append({"prompt_tokens": 6, "completion_tokens": 4})
            return rulings

        with patch("courselens_worker.llm.judge_lecture_quality", side_effect=fake_judge), \
                patch("courselens_worker.llm.judge_term_boundaries",
                      side_effect=fake_boundary):
            result = process_job(self._job({
                "subtitle_sample": _SUBTITLE_SAMPLE,
                "glossary": ["链式聚合"],
                "term_boundary": [{"wrong": "a", "right": "b", "signal_count": 2}],
            }))
        self.assertEqual(result["outputs"]["term_boundary_rulings"], rulings)
        self.assertEqual(seen["pairs"], [{"wrong": "a", "right": "b", "signal_count": 2}])
        self.assertEqual(seen["glossary"], ("链式聚合",))
        self.assertEqual(result["metrics"]["deepseek_tokens"], 10, "裁决账并同一账本")
        self.assertEqual(result["warnings"], [])

    def test_branch_boundary_failure_is_closed_warning_not_task_failure(self):
        report = {
            "schema_version": 1,
            "subtitle": {"sample_size": 2, "targets_total": 2, "findings": []},
            "summary": None,
            "stages": {"subtitle": {"flash_findings": 0, "adjudicated": False}},
        }

        def fake_judge(api_key, **kwargs):
            return report

        def failing_boundary(api_key, **kwargs):
            raise LLMError("AI request failed: ConnectionError")

        with patch("courselens_worker.llm.judge_lecture_quality", side_effect=fake_judge), \
                patch("courselens_worker.llm.judge_term_boundaries",
                      side_effect=failing_boundary):
            result = process_job(self._job({
                "subtitle_sample": _SUBTITLE_SAMPLE,
                "term_boundary": [{"wrong": "a", "right": "b"}],
            }))
        self.assertEqual(result["status"], "completed", "裁决失败不拖垮质检任务")
        self.assertEqual(result["warnings"], ["judge_boundary_failed"])
        self.assertNotIn("term_boundary_rulings", result["outputs"])

    def test_branch_ignores_non_list_boundary_key(self):
        report = {
            "schema_version": 1,
            "subtitle": {"sample_size": 2, "targets_total": 2, "findings": []},
            "summary": None,
            "stages": {"subtitle": {"flash_findings": 0, "adjudicated": False}},
        }

        def fake_judge(api_key, **kwargs):
            return report

        def fail_if_called(api_key, **kwargs):
            raise AssertionError("非 list 边界键不得触发裁决调用")

        with patch("courselens_worker.llm.judge_lecture_quality", side_effect=fake_judge), \
                patch("courselens_worker.llm.judge_term_boundaries",
                      side_effect=fail_if_called):
            result = process_job(self._job({
                "subtitle_sample": _SUBTITLE_SAMPLE,
                "term_boundary": "垃圾形状",
            }))
        self.assertNotIn("term_boundary_rulings", result["outputs"])
        self.assertEqual(result["warnings"], [])

    def test_branch_lets_llm_errors_fail_honestly(self):
        def fake_judge(api_key, **kwargs):
            raise LLMError("AI request failed: ConnectionError")

        with patch("courselens_worker.llm.judge_lecture_quality", side_effect=fake_judge):
            with self.assertRaises(LLMError):
                process_job(self._job({"subtitle_sample": _SUBTITLE_SAMPLE}))


class BoundaryRulingTests(unittest.TestCase):
    """THINK-LADDER-2 设计 B：记忆边界候选一次 flash 裁决（worker 面）。"""

    _PAIRS = [
        {"wrong": "链式巨合", "right": "链式聚合", "signal_count": 2},
        {"wrong": "凝崁化", "right": "凝胶化", "signal_count": 1},
        {"wrong": "深问", "right": "追问", "signal_count": 1},
    ]
    _RULINGS = {
        "rulings": [
            {"wrong": "深问", "right": "追问",
             "ruling": "unsure", "reason_code": "not_in_glossary"},
            {"wrong": "凝崁化", "right": "凝胶化",
             "ruling": "valid", "reason_code": "glossary_match"},
            {"wrong": "无关对", "right": "编造",
             "ruling": "valid", "reason_code": "glossary_match"},
            {"wrong": "链式巨合", "right": "链式聚合",
             "ruling": "bogus", "reason_code": "glossary_match"},
        ],
    }

    def _run(self, pairs, payloads, *, glossary=()):
        calls = {"n": 0}

        def _chat(api_key, messages, **kwargs):
            from courselens_worker import llm as llm_module

            with llm_module._USAGE_LOCK:
                llm_module._CALL_LOG.append({
                    "prompt_tokens": 10, "completion_tokens": 5,
                    "reasoning_tokens": 0, "prompt_cache_hit_tokens": 0,
                    "latency_ms": 10, "thinking": None,
                })
            index = min(calls["n"], len(payloads) - 1)
            calls["n"] += 1
            return payloads[index]

        usage: list[dict] = []
        lines: list[str] = []
        with patch("courselens_worker.llm._chat", side_effect=_chat), \
                patch("courselens_worker.llm._emit_telemetry", side_effect=lines.append), \
                patch("time.sleep"):
            result = judge_term_boundaries(
                "k", pairs=pairs, glossary=glossary, usage_sink=usage)
        return result, calls, usage, lines

    def test_prompt_bounded_with_closed_enums(self):
        self.assertLessEqual(len(_JUDGE_BOUNDARY_PROMPT), 300)
        for needle in (
            "逐对判断", "零自由文本",
            "ruling=valid/invalid/unsure",
            "reason_code=glossary_match/glossary_conflict/not_in_glossary/"
            "insufficient_context", "right", "wrong",
        ):
            self.assertIn(needle, _JUDGE_BOUNDARY_PROMPT)

    def test_pair_validation_drops_malformed_and_caps(self):
        raw = [
            {"wrong": "a", "right": "b"},
            "垃圾对",
            {"wrong": "", "right": "b"},
            {"wrong": "a"},
            {"wrong": "c", "right": "d", "signal_count": "非整数也忽略"},
        ] + [{"wrong": f"w{i}", "right": f"r{i}"} for i in range(_JUDGE_BOUNDARY_MAX_PAIRS + 2)]
        pairs, malformed = _validate_boundary_pairs(raw)
        self.assertEqual(len(pairs), _JUDGE_BOUNDARY_MAX_PAIRS, "帽 10 截断")
        self.assertEqual(malformed, 3, "非 dict/空 wrong/缺 right 各计一")
        self.assertEqual(pairs[0], {"wrong": "a", "right": "b"})

    def test_zero_valid_pairs_makes_no_call(self):
        result, calls, usage, lines = self._run([{"wrong": "", "right": "x"}], ["垃圾一"])
        self.assertEqual(calls["n"], 0, "零良构对零调用")
        self.assertEqual(
            result, {"rulings": [], "malformed_pairs": 1, "unruled_pairs": 0})
        self.assertEqual(usage, [])
        self.assertEqual(
            [line for line in lines if line.startswith("stage=term-boundary")],
            ["stage=term-boundary ruled=0 unruled=0 malformed_pairs=1"])

    def test_happy_path_rulings_echo_order_and_usage(self):
        result, calls, usage, lines = self._run(
            self._PAIRS, [json.dumps(self._RULINGS)], glossary=("链式聚合",))
        self.assertEqual(calls["n"], 1, "恰一次 flash 裁决")
        self.assertEqual(
            [item["wrong"] for item in result["rulings"]],
            ["凝崁化", "深问"],
            "输出按请求顺序（响应乱序重排；bogus ruling 条目丢弃）")
        self.assertEqual(
            result["rulings"][1],
            {"wrong": "深问", "right": "追问",
             "ruling": "unsure", "reason_code": "not_in_glossary"})
        self.assertEqual(result["malformed_pairs"], 0)
        self.assertEqual(result["unruled_pairs"], 1, "bogus ruling 条目=该对落 unruled")
        self.assertEqual([record.get("stage") for record in usage], ["boundary"])
        final = [line for line in lines if line.startswith("stage=term-boundary ")]
        self.assertEqual(
            final, ["stage=term-boundary ruled=2 unruled=1 malformed_pairs=0"])

    def test_shape_invalid_both_attempts_raises_llmerror(self):
        with self.assertRaises(LLMError):
            self._run(self._PAIRS, ["垃圾一", "垃圾二"])

    def test_429_never_retries(self):
        attempts = {"n": 0}

        def _chat(api_key, messages, **kwargs):
            attempts["n"] += 1
            raise LLMError("AI request returned HTTP 429")

        with patch("courselens_worker.llm._chat", side_effect=_chat), \
                patch("time.sleep") as slept:
            with self.assertRaises(LLMError):
                judge_term_boundaries("k", pairs=self._PAIRS)
        self.assertEqual(attempts["n"], 1)
        slept.assert_not_called()

    def test_salvage_recovers_rulings_objects(self):
        raw = (
            "好的，裁决如下 {\"rulings\": [{\"wrong\": \"a\", \"right\": \"b\", "
            "\"ruling\": \"unsure\", \"reason_code\": \"insufficient_context\"}]}"
        )
        recovered = _salvage_judge_findings(raw, key="rulings")
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["rulings"][0]["ruling"], "unsure")
        self.assertIsNone(_salvage_judge_findings("完全没有对象", key="rulings"))


class JudgeCoRunTests(unittest.TestCase):
    """N2-2a：summary run 内 judge 共位（additive judge_request，worker 腿）。"""

    _TRANSCRIPT = [
        {"index": i, "start_ms": i * 30_000, "end_ms": (i + 1) * 30_000,
         "text": f"第{i}段内容。"}
        for i in range(4)
    ]
    _SUMMARY = {
        "markdown": "本讲讲链式聚合与凝胶化。",
        "chapters": [
            {"title": "开场", "start_ms": 0, "summary": "链式聚合"},
            {"title": "凝胶化", "start_ms": 30_000, "summary": "临界转化率"},
        ],
        "key_takeaways": ["链式聚合逐步进行", "凝胶化有临界转化率"],
        "knowledge_points": [],
    }

    def _summary_job(self, payload):
        return {
            "schema": JOB_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "task_id": "0123456789abcdef0123456789abcdef",
            "job_kind": "summary",
            "input_hash": "0" * 64,
            "pipeline": {"version": "summary-v1"},
            "payload": payload,
            "secrets": {"deepseek_api_key": "k"},
        }

    def _run_summary_job(self, payload, *, judge_report=None, judge_error=None):
        captured: dict = {}
        usage: list[dict] = []

        def fake_summary(api_key, **kwargs):
            return dict(self._SUMMARY)

        def fake_judge(api_key, **kwargs):
            captured.update(kwargs)
            kwargs["usage_sink"].append({"prompt_tokens": 7, "completion_tokens": 3})
            if judge_error is not None:
                raise judge_error
            return judge_report if judge_report is not None else {
                "schema_version": 1, "subtitle": None,
                "summary": {"targets_total": 3, "findings": []},
                "stages": {"summary": {"flash_findings": 0, "adjudicated": False}},
            }

        with patch("courselens_worker.llm.create_summary", side_effect=fake_summary), \
                patch("courselens_worker.llm.judge_lecture_quality",
                      side_effect=fake_judge):
            result = process_job(self._summary_job(payload))
        return result, captured, usage

    def test_co_run_builds_both_inputs_from_run_products(self):
        payload = {
            "title": "t",
            "transcript": self._TRANSCRIPT,
            "glossary": ["链式聚合"],
            "judge_request": {
                "segment_indices": [1, 3, 1, "x", True, 99, -1],
                "transcript_digest": "0" * 64,
            },
        }
        result, captured, _usage = self._run_summary_job(payload)
        self.assertEqual(result["status"], "completed")
        self.assertIn("quality_judge", result["outputs"])
        sample = captured["subtitle_sample"]
        self.assertEqual(
            [segment["index"] for segment in sample["segments"]],
            [1, 3], "非 int/越界/重复索引全部丢弃，去重保序")
        self.assertEqual(sample["total_segments"], 4)
        pack = captured["summary_pack"]
        self.assertEqual(pack["markdown"], "本讲讲链式聚合与凝胶化。")
        self.assertEqual(len(pack["chapters"]), 2)
        self.assertEqual(
            sorted(pack["chapters"][0]), ["start_ms", "summary", "title"],
            "chapters 三键与客户端 judge 载荷同形")
        self.assertEqual(pack["key_takeaways"], self._SUMMARY["key_takeaways"])
        self.assertEqual(
            [segment["index"] for segment in pack["transcript"]], [0, 1, 2, 3])
        self.assertEqual(captured["glossary"], ("链式聚合",))
        self.assertEqual(
            result["metrics"]["deepseek_tokens"], 10, "共位质检账并任务账本")

    def test_co_run_skips_silently_without_material(self):
        payload = {
            "title": "t",
            "transcript": [],
            "judge_request": {"segment_indices": [0, 1]},
            "summary": {},
        }

        def fake_summary(api_key, **kwargs):
            return {"markdown": "", "chapters": [], "key_takeaways": [],
                    "knowledge_points": []}

        def fail_if_called(api_key, **kwargs):
            raise AssertionError("无材料不得触发质检调用")

        with patch("courselens_worker.llm.create_summary", side_effect=fake_summary), \
                patch("courselens_worker.llm.judge_lecture_quality",
                      side_effect=fail_if_called):
            result = process_job(self._summary_job(payload))
        self.assertEqual(result["status"], "completed")
        self.assertNotIn("quality_judge", result["outputs"])
        self.assertEqual(result["warnings"], [], "零材料静默跳过（漏斗兜底照旧）")

    def test_co_run_failure_warns_and_summary_survives(self):
        payload = {
            "title": "t",
            "transcript": self._TRANSCRIPT,
            "judge_request": {"segment_indices": [0]},
        }
        result, _captured, usage = self._run_summary_job(
            payload, judge_error=LLMError("AI request failed: ConnectionError"))
        self.assertEqual(result["status"], "completed", "共位失败不拖垮总结")
        self.assertIn("summary", result["outputs"])
        self.assertEqual(result["warnings"], ["judge_co_run_failed"])
        self.assertNotIn("quality_judge", result["outputs"])

    def test_no_judge_request_keeps_legacy_shape(self):
        payload = {"title": "t", "transcript": self._TRANSCRIPT}
        result, _captured, _usage = self._run_summary_job(payload)
        self.assertNotIn("quality_judge", result["outputs"])
        self.assertEqual(result["warnings"], [])


class ProtocolRoundtripTests(unittest.TestCase):
    """协议层 quality_judge 加性注册：job intake 与 result 两端闭集都放行
    （PKG-A 申报的共享协议面恰域扩展）。"""

    def test_job_roundtrip_accepts_quality_judge(self):
        worker_priv, worker_pub = generate_box_keypair()
        result_pub, _result_priv = generate_box_keypair()
        sealed = seal_job({
            "task_id": "0123456789abcdef0123456789abcdef",
            "job_kind": "quality_judge",
            "created_at": time.time(),
            "expires_at": time.time() + 600,
            "result_public_key": result_pub,
            "pipeline": {"version": "quality-judge-v1"},
            "requested_outputs": ["quality_judge"],
            "payload": {"title": "t", "subtitle_sample": _SUBTITLE_SAMPLE},
            "secrets": {},
        }, worker_pub)
        opened = open_job(sealed, worker_priv)
        self.assertEqual(opened["job_kind"], "quality_judge")

    def test_result_roundtrip_accepts_quality_judge(self):
        _result_priv, result_pub = generate_box_keypair()
        sign_priv, sign_pub = generate_signing_keypair()
        result = {
            "schema": RESULT_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "task_id": "0123456789abcdef0123456789abcdef",
            "job_kind": "quality_judge",
            "input_hash": "a" * 64,
            "status": "completed",
        }
        sealed = seal_result(result, result_pub, sign_priv)
        opened = open_result(
            sealed, result_private_key=_result_priv,
            worker_signing_public_key=sign_pub,
            expected_task_id=result["task_id"],
            expected_input_hash=result["input_hash"],
        )
        self.assertEqual(opened["job_kind"], "quality_judge")


if __name__ == "__main__":
    unittest.main()
