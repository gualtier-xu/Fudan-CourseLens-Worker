"""P11-CONTRACT-1 PKG-A：LLM-as-judge 离线质检（worker 侧）钉面。

覆盖：两条提示词 ≤300 帽+底线词+闭集枚举、thinking env 闭集覆写与输出帽
配对、_salvage_judge_findings 两级抢救、字幕/总结 mode 闭集校验逐项（越界
target/dimension/code/severity/position 丢弃+计数、附加字段剥除=零自由文本、
数量帽 warn 优先截断）、单 mode null、双 mode 无效 fail-closed、usage_sink
流水（失败尝试不丢账）、429/授权类不入重试集、传输失败原样上抛（无本地降
级）、遥测仅计数零内容、runner 分支（outputs/metrics/warnings/诚实失败）、
协议层 quality_judge 加性注册 roundtrip（job intake 与 result 两端闭集）。
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
    _JUDGE_FINDINGS_CAP,
    _JUDGE_SUBTITLE_PROMPT,
    _JUDGE_SUMMARY_PROMPT,
    _resolve_judge_thinking,
    _salvage_judge_findings,
    _validate_judge_subtitle,
    _validate_judge_summary,
    judge_lecture_quality,
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
    """按 system 提示词分流的替身：各自重放末项供重试；每调用落一条 usage
    流水（与 test_review_views 同法），供 usage_sink 账面断言。"""
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
        if system == _JUDGE_SUBTITLE_PROMPT:
            payloads = subtitle_payloads or ["垃圾一"]
            index = min(calls["subtitle"], len(payloads) - 1)
            calls["subtitle"] += 1
            return payloads[index]
        assert system == _JUDGE_SUMMARY_PROMPT, system[:40]
        payloads = summary_payloads or ["垃圾一"]
        index = min(calls["summary"], len(payloads) - 1)
        calls["summary"] += 1
        return payloads[index]

    return _chat, calls


class PromptPinTests(unittest.TestCase):
    def test_prompts_bounded_with_bottom_lines_and_closed_enums(self):
        for prompt, needles in (
            (_JUDGE_SUBTITLE_PROMPT, (
                "没有问题输出空 findings", "禁止编造输入外的问题", "只依据输入",
                "零自由文本", "segment", "term_fidelity", "readability",
                "glossary_violation", "homophone_suspect", "term_inconsistent",
                "broken_flow", "garbled", "warn", "info",
            )),
            (_JUDGE_SUMMARY_PROMPT, (
                "没有问题输出空 findings", "禁止编造输入外的问题", "只依据输入",
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

    def test_thinking_tier_scales_output_cap(self):
        seen: dict = {}

        def _chat(api_key, messages, *, max_tokens=8192, thinking=None):
            seen["max_tokens"] = max_tokens
            seen["thinking"] = thinking
            return json.dumps({"findings": []})

        with patch("courselens_worker.llm._chat", side_effect=_chat):
            judge_lecture_quality("k", subtitle_sample=_SUBTITLE_SAMPLE)
        self.assertEqual(seen["max_tokens"], 8192)
        self.assertEqual(seen["thinking"], {"type": "disabled"})
        with patch("courselens_worker.llm._chat", side_effect=_chat), \
                patch.dict(os.environ, {JUDGE_THINKING_ENV: "high"}, clear=False):
            judge_lecture_quality("k", subtitle_sample=_SUBTITLE_SAMPLE)
        self.assertEqual(seen["max_tokens"], 16384)
        self.assertEqual(seen["thinking"], {"type": "enabled", "reasoning_effort": "high"})


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
        self.assertEqual(calls, {"subtitle": 1, "summary": 1}, "每讲恰 1-2 次调用")
        self.assertEqual(len(usage), 2, "每调用一条 usage 流水")
        final = [line for line in lines if line.startswith("stage=quality-judge ")]
        self.assertEqual(len(final), 1, lines)
        self.assertIn("subtitle_findings=1", final[0])
        self.assertIn("summary_findings=2", final[0])
        self.assertIn("findings_truncated=0", final[0])
        self.assertIn("findings_invalid_dropped=0", final[0])
        for leaked in ("链式聚合", "凝胶化", "本讲讲", "k"):
            self.assertNotIn(leaked, final[0], "遥测零内容零账号值")

    def test_glossary_travels_data_channel_and_empty_omits_key(self):
        capture: dict = {}

        def _chat(api_key, messages, *, max_tokens=8192, thinking=None):
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

    def test_single_mode_only_makes_one_call(self):
        report, calls, _, _ = self._run([json.dumps(_SUBTITLE_FINDINGS)])
        self.assertIsNotNone(report["subtitle"])
        self.assertIsNone(report["summary"])
        self.assertEqual(calls, {"subtitle": 1, "summary": 0})

    def test_one_mode_invalid_nulls_it_and_other_survives(self):
        report, calls, usage, lines = self._run(
            [json.dumps(_SUBTITLE_FINDINGS), "垃圾一"], summary_pack=_SUMMARY_PACK)
        self.assertIsNotNone(report["subtitle"])
        self.assertIsNone(report["summary"], "形状两败置 null，不拖垮另一 mode")
        self.assertEqual(calls["summary"], 2, "attempts=2")
        self.assertEqual(len(usage), 3, "失败尝试也落账")
        retries = [line for line in lines if line.startswith("stage=quality-judge-retry")]
        self.assertEqual(len(retries), 1, lines)
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

    def test_branch_lets_llm_errors_fail_honestly(self):
        def fake_judge(api_key, **kwargs):
            raise LLMError("AI request failed: ConnectionError")

        with patch("courselens_worker.llm.judge_lecture_quality", side_effect=fake_judge):
            with self.assertRaises(LLMError):
                process_job(self._job({"subtitle_sample": _SUBTITLE_SAMPLE}))


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
