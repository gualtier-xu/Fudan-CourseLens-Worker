"""SUBTITLE-DEEP-1 Phase A: term-position deep correction pins.

Covers the term-only validation gate, revision stamping, window packing,
checkpoint resume trust, response cache, runner chaining, and the telemetry
discipline (counters only, never subtitle content).
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from courselens_worker.formats import normalize_segments
from courselens_worker.glossary import resolve_course_terms
from courselens_worker.llm import (
    LLMError,
    TERM_APPLIED_STATUS,
    TERM_PROOFREAD_VERSION,
    _CORRECTION_STATUSES,
    _deep_op_allowed,
    _normalized_terms,
    term_proofread_segments,
)
from courselens_worker.runner import _apply_term_stage

TERMS = ("能带图", "能带", "电势", "费米能级", "小信号", "PN结")


def seg(start_ms, end_ms, text, **extra):
    value = {"start_ms": start_ms, "end_ms": end_ms, "text": text}
    value.update(extra)
    return value


def run_term(segments, responses, *, terms=TERMS, prior=None, cache=None, ppt_pages=None):
    """Run term_proofread_segments with mocked _chat; returns (result, payloads)."""
    payloads = []

    def fake_chat(api_key, messages, **kwargs):
        payloads.append(json.loads(messages[1]["content"]))
        if isinstance(responses, list):
            return responses[min(len(payloads) - 1, len(responses) - 1)]
        return responses

    with patch("courselens_worker.llm._chat", side_effect=fake_chat):
        result = term_proofread_segments(
            "secret", segments, terms=terms,
            prior_checkpoint=prior, cache=cache, ppt_pages=ppt_pages,
        )
    return result, payloads


class TermGateTests(unittest.TestCase):
    def test_normalized_terms_dedupe_and_ignore_short(self):
        self.assertEqual(
            _normalized_terms(("能带图", "能带图", " 电势 ", "x")),
            ["能带图", "电势"],
        )

    def test_deep_gate_allows_homophone_and_punct_ops(self):
        normalized = _normalized_terms(TERMS)
        # 同音/近音修正（含闭集外常用词）放行
        self.assertTrue(_deep_op_allowed("电视", "电势", normalized))
        self.assertTrue(_deep_op_allowed("肺敏能级", "费米能级", normalized))
        # 标点位：正文逐字相同、只增闭集标点（整段重标点是模型自然形态）
        self.assertTrue(_deep_op_allowed("然后我们能看", "然后，我们能看", normalized))
        self.assertTrue(_deep_op_allowed(
            "啊这个讲了半天很复杂嗯很复杂",
            "啊，这个讲了半天很复杂，嗯，很复杂。",
            normalized,
        ))
        # 内容位伴随标点：去标点漂移 ≤4 放行
        self.assertTrue(_deep_op_allowed(
            "平衡态它就代表肺敏能级它的位置",
            "平衡态它就代表费米能级，它的位置",
            normalized,
        ))
        # 拒绝：内容漂移超帽
        self.assertFalse(_deep_op_allowed("能耐", "能带图结构说明文字很长", normalized))
        # 拒绝：标点位用了非闭集字符
        self.assertFalse(_deep_op_allowed("我们看", "我们《看》", normalized))
        # 拒绝：删原有标点
        self.assertFalse(_deep_op_allowed("然后，我们能看", "然后我们能看", normalized))
        # 拒绝：同字标点连用（叠标点伪影）
        self.assertFalse(_deep_op_allowed(
            "所以你们那个绩点还是挺重要的",
            "所以，，，你们那个绩点还是挺重要的。",
            normalized,
        ))

    def test_deep_correction_without_terms_still_applies(self):
        # v3：terms 为空照跑（纯标点/常用词路径）
        primary = [seg(0, 1000, "就能得到我的这样一个能耐图")]
        response = json.dumps([{"id": "t0", "old": "能耐图", "new": "能带图"}])
        result, payloads = run_term(primary, response, terms=())
        self.assertEqual(result[0]["text"], "就能得到我的这样一个能带图")
        self.assertEqual(len(payloads), 1)

    def test_punct_only_op_applies(self):
        primary = [seg(0, 1000, "然后我们能看电势它就是这个")]
        response = json.dumps([{"id": "t0", "old": "然后我们能看", "new": "然后，我们能看"}])
        result, _ = run_term(primary, response, terms=())
        self.assertEqual(result[0]["text"], "然后，我们能看电势它就是这个")
        self.assertEqual(result[0]["correction"], TERM_APPLIED_STATUS)

    def test_term_replacement_applies_and_stamps_revision(self):
        primary = [seg(0, 1000, "就能得到我的这样一个能耐图")]
        response = json.dumps([{"id": "t0", "old": "能耐图", "new": "能带图"}])
        result, _ = run_term(primary, response)
        self.assertEqual(result[0]["text"], "就能得到我的这样一个能带图")
        self.assertEqual(result[0]["correction"], TERM_APPLIED_STATUS)
        self.assertEqual(result[0]["term_revision"], TERM_PROOFREAD_VERSION)
        self.assertIn(result[0]["correction"], _CORRECTION_STATUSES)

    def test_multiple_term_ops_apply_per_segment(self):
        primary = [seg(0, 1000, "电场就能得到电视电视然后再得到能耐图")]
        response = json.dumps([
            {"id": "t0", "old": "电视电视", "new": "电势电势"},
            {"id": "t0", "old": "能耐图", "new": "能带图"},
        ])
        result, _ = run_term(primary, response)
        self.assertEqual(result[0]["text"], "电场就能得到电势电势然后再得到能带图")
        self.assertEqual(result[0]["correction"], TERM_APPLIED_STATUS)

    def test_generic_correction_applies_v3_expansion(self):
        # v3 扩权（总控补充行）：任意误识位修正放行，不再要求闭集术语位
        primary = [seg(0, 1000, "这个器件很有意思")]
        response = json.dumps([{"id": "t0", "old": "很有意思", "new": "挺有意思"}])
        result, _ = run_term(primary, response)
        self.assertEqual(result[0]["text"], "这个器件挺有意思")
        self.assertEqual(result[0]["correction"], TERM_APPLIED_STATUS)

    def test_term_correction_drops_token_identity(self):
        primary = [seg(
            0, 1000, "电视是两点零",
            tokens=[[ "电", 0]], segment_id="seg-1", source_hash="a" * 64,
            provenance={"producer": "p"}, lang="zh",
        )]
        response = json.dumps([{"id": "t0", "old": "电视", "new": "电势"}])
        result, _ = run_term(primary, response)
        self.assertEqual(result[0]["text"], "电势是两点零")
        self.assertNotIn("tokens", result[0])
        self.assertNotIn("segment_id", result[0])
        self.assertEqual(result[0]["source_hash"], "a" * 64)
        self.assertEqual(result[0]["term_revision"], TERM_PROOFREAD_VERSION)

    def test_uncorrected_segment_has_no_correction_key(self):
        primary = [seg(0, 1000, "没有术语问题的句子", tokens=[["t", 0]], segment_id="s")]
        result, _ = run_term(primary, "[]")
        self.assertNotIn("correction", result[0])
        self.assertEqual(result[0]["tokens"], [["t", 0]])
        self.assertEqual(result[0]["segment_id"], "s")

    def test_ambiguous_old_is_rejected(self):
        primary = [seg(0, 1000, "能耐图和能耐图")]
        response = json.dumps([{"id": "t0", "old": "能耐图", "new": "能带图"}])
        result, _ = run_term(primary, response)
        self.assertEqual(result[0]["correction"], "rejected-ambiguous")

    def test_unknown_id_is_ignored(self):
        primary = [seg(0, 1000, "能耐图在这里")]
        response = json.dumps([{"id": "t99", "old": "能耐图", "new": "能带图"}])
        result, _ = run_term(primary, response)
        self.assertEqual(result[0]["text"], "能耐图在这里")
        self.assertNotIn("correction", result[0])

    def test_protected_number_change_is_rejected(self):
        primary = [seg(0, 1000, "电压是5伏的能耐图")]
        response = json.dumps([{"id": "t0", "old": "5伏", "new": "6伏"}])
        result, _ = run_term(primary, response)
        self.assertEqual(result[0]["text"], "电压是5伏的能耐图")

    def test_truncated_array_recovers_completed_objects(self):
        # 截断抢救：未闭合数组中已完成的提案逐个回收（各过验证门）
        primary = [seg(0, 1000, "就能得到能耐图"), seg(2000, 3000, "第二句原文")]
        truncated = '[{"id": "t0", "old": "能耐图", "new": "能带图"}, {"id": "t1", "ol'
        result, _ = run_term(primary, truncated)
        self.assertEqual(result[0]["text"], "就能得到能带图")
        self.assertEqual(result[1]["text"], "第二句原文")

    def test_chatty_response_array_is_salvaged(self):
        primary = [seg(0, 1000, "就能得到能耐图")]
        chatty = "好的，以下是修正提案：\n" + json.dumps(
            [{"id": "t0", "old": "能耐图", "new": "能带图"}]) + "\n请审核。"
        result, _ = run_term(primary, chatty)
        self.assertEqual(result[0]["text"], "就能得到能带图")

    def test_salvage_still_fails_closed_without_array(self):
        with self.assertRaises(LLMError):
            run_term([seg(0, 1000, "原文")], "这段文字里没有任何数组结构")

    def test_hard_window_splits_into_halves_and_recovers(self):
        # 自适应分窗：整窗确定性失败时对半递归（≥4 段才分），分片成功即恢复
        primary = [seg(index * 2000, index * 2000 + 1000, f"第{index}句能耐图") for index in range(8)]
        calls = {"n": 0}

        def flaky_chat(api_key, messages, **kwargs):
            calls["n"] += 1
            payload = json.loads(messages[1]["content"])
            if len(payload) > 2:
                raise LLMError("reasoning budget blowout")
            fixes = []
            for item in payload:
                if "能耐图" in item["text"]:
                    fixes.append({"id": item["id"], "old": "能耐图", "new": "能带图"})
            return json.dumps(fixes)

        with (
            patch("courselens_worker.llm._chat", side_effect=flaky_chat),
            patch("courselens_worker.llm.time.sleep"),
        ):
            result = term_proofread_segments("secret", primary, terms=TERMS)
        self.assertEqual(
            calls["n"], 13,
            "8段片3败+两个4段片各3败+四个2段片各1成（每片重试梯=3）",
        )
        fixed = sum(1 for item in result if item.get("correction") == TERM_APPLIED_STATUS)
        self.assertEqual(fixed, 8, "全部段完成术语修正")
        self.assertTrue(all("能带图" in item["text"] for item in result))

    def test_composed_double_punct_is_rejected(self):
        # 合成校验：两条各自合法的单逗号提案顺序应用不得叠出同字标点
        primary = [seg(0, 1000, "所以你们那个绩点还是挺重要的特别是对于")]
        response = json.dumps([
            {"id": "t0", "old": "所以你们", "new": "所以，你们"},
            {"id": "t0", "old": "你们那个", "new": "，你们那个"},
        ])
        result, _ = run_term(primary, response)
        self.assertNotIn("，，", result[0]["text"])

    def test_non_array_response_raises_llm_error(self):
        with self.assertRaises(LLMError):
            run_term([seg(0, 1000, "原文")], json.dumps({"index": 0}))

    def test_empty_terms_still_corrects_v3(self):
        # v3：空术语表不再旁路（标点/常用词路径）；无提案时原样返回
        primary = [seg(0, 1000, "能耐图")]
        result, payloads = run_term(primary, "[]", terms=())
        self.assertEqual(result[0]["text"], "能耐图")
        self.assertEqual(len(payloads), 1)
        self.assertNotIn("correction", result[0])


class TermWindowTests(unittest.TestCase):
    def _segments(self, count):
        return [seg(index * 2000, index * 2000 + 1000, f"第{index}句能耐图") for index in range(count)]

    def test_windows_pack_by_segment_cap_with_overlap(self):
        # v4 重叠窗：owned 核=推进单位，wire=核+前瞻 4 条（相邻窗重叠 4 段）
        from courselens_worker.llm import _TERM_WINDOW_SEGMENTS, _term_windows

        windows = _term_windows(self._segments(_TERM_WINDOW_SEGMENTS + 3), None)
        self.assertEqual([len(window["owned"]) for window in windows], [_TERM_WINDOW_SEGMENTS, 3])
        # 23 段总数：w0 wire=核20+前瞻3（剩余仅3），w1 wire=owned 3
        self.assertEqual([len(window["wire"]) for window in windows], [_TERM_WINDOW_SEGMENTS + 3, 3])
        # 重叠正确性：w0 wire 尾=前瞻段（属 w1 owned），w1 owned 从 t20 起
        self.assertEqual(windows[0]["wire"][-1]["id"], "t22")
        self.assertEqual(windows[1]["owned"][0]["id"], f"t{_TERM_WINDOW_SEGMENTS}")

    def test_owned_window_drops_lookahead_ops(self):
        # interior-wins：指向重叠前瞻段的提案被静默忽略，owned 段照常应用
        from courselens_worker.llm import _term_windows

        primary = self._segments(22)
        windows = _term_windows(primary, None)
        wire = windows[0]["wire"]
        owned_ids = {entry["id"] for entry in windows[0]["owned"]}
        lookahead_id = wire[-1]["id"]
        self.assertNotIn(lookahead_id, owned_ids)
        response = json.dumps([
            {"id": "t0", "old": "能耐图", "new": "能带图"},
            {"id": lookahead_id, "old": "能耐图", "new": "能带图"},
        ])
        from courselens_worker.llm import _apply_term_ops

        out = _apply_term_ops(
            wire, json.loads(response), _normalized_terms(TERMS), owned_ids=owned_ids,
        )
        self.assertEqual(len(out), 20, "输出只含 owned 段（重叠段不重复产出）")
        self.assertIn("能带图", out[0]["text"])

    def test_wire_payload_carries_ids_and_slide(self):
        pages = [{"created_sec": 0, "text": "课件术语 费米能级"}]
        primary = [seg(0, 1000, "能耐图")]
        _, payloads = run_term(primary, "[]", ppt_pages=pages)
        self.assertEqual(payloads[0][0]["id"], "t0")
        self.assertEqual(payloads[0][0]["slide"], "课件术语 费米能级")

    def test_cache_hit_skips_chat(self):
        primary = [seg(0, 1000, "就能得到能耐图")]
        cache: dict[str, str] = {}
        calls = {"n": 0}

        def chat_once(api_key, messages, **kwargs):
            calls["n"] += 1
            return json.dumps([{"id": "t0", "old": "能耐图", "new": "能带图"}])

        with patch("courselens_worker.llm._chat", side_effect=chat_once):
            result = term_proofread_segments("secret", primary, terms=TERMS, cache=cache)
        self.assertEqual(result[0]["text"], "就能得到能带图")
        self.assertEqual(calls["n"], 1)
        with patch(
            "courselens_worker.llm._chat",
            side_effect=AssertionError("cache hit must not call chat"),
        ):
            cached = term_proofread_segments("secret", primary, terms=TERMS, cache=cache)
        self.assertEqual(cached[0]["text"], "就能得到能带图")
        self.assertEqual(calls["n"], 1, "第二遍必须命中缓存零调用")


class TermCheckpointTests(unittest.TestCase):
    def test_resume_trusted_only_when_proofread_completed(self):
        primary = [seg(0, 1000, "能耐图")]
        trusted_prior = {
            "term_proofread_revision": TERM_PROOFREAD_VERSION,
            "proofread_completed_windows": 2,
            "proofread_total_windows": 2,
            "term_proofread_completed_windows": 1,
            "term_proofread_total_windows": 1,
            "term_proofread_segments": [seg(0, 1000, "能带图", correction=TERM_APPLIED_STATUS)],
        }
        result, payloads = run_term(primary, "[]", prior=trusted_prior)
        self.assertEqual(payloads, [], "信任检查点零重复调用")
        self.assertEqual(result[0]["text"], "能带图")

    def test_resume_rejected_when_proofread_incomplete(self):
        primary = [seg(0, 1000, "能耐图")]
        stale_prior = {
            "term_proofread_revision": TERM_PROOFREAD_VERSION,
            "proofread_completed_windows": 1,
            "proofread_total_windows": 2,
            "term_proofread_completed_windows": 1,
            "term_proofread_total_windows": 1,
            "term_proofread_segments": [seg(0, 1000, "旧结果")],
        }
        result, payloads = run_term(primary, "[]", prior=stale_prior)
        self.assertEqual(len(payloads), 1, "词级校对未完成的检查点必须重跑")
        self.assertEqual(result[0]["text"], "能耐图")

    def test_resume_rejected_on_revision_mismatch(self):
        primary = [seg(0, 1000, "能耐图")]
        stale_prior = {
            "term_proofread_revision": "term-deep-v0",
            "proofread_completed_windows": 1,
            "proofread_total_windows": 1,
            "term_proofread_completed_windows": 1,
            "term_proofread_total_windows": 1,
            "term_proofread_segments": [seg(0, 1000, "旧版本结果")],
        }
        result, _ = run_term(primary, "[]", prior=stale_prior)
        self.assertEqual(result[0]["text"], "能耐图")

    def test_checkpoint_writes_carry_term_state(self):
        written = []
        primary = [seg(0, 1000, "能耐图")]
        response = json.dumps([{"id": "t0", "old": "能耐图", "new": "能带图"}])
        with patch("courselens_worker.llm._chat", return_value=response):
            term_proofread_segments(
                "secret", primary, terms=TERMS, checkpoint=written.append,
            )
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["stage"], "term_proofread")
        self.assertEqual(written[0]["term_proofread_revision"], TERM_PROOFREAD_VERSION)
        self.assertEqual(written[0]["term_proofread_completed_windows"], 1)


class TermTelemetryTests(unittest.TestCase):
    def test_final_line_carries_counters_without_content(self):
        lines = []
        primary = [seg(0, 1000, "就能得到能耐图")]
        response = json.dumps([{"id": "t0", "old": "能耐图", "new": "能带图"}])
        with (
            patch("courselens_worker.llm._chat", return_value=response),
            patch("courselens_worker.llm._emit_telemetry", side_effect=lines.append),
        ):
            result, _ = run_term(primary, response)
        finals = [line for line in lines if line.startswith("stage=term-proofread ")]
        self.assertEqual(len(finals), 1)
        self.assertIn("windows=1/1", finals[0])
        self.assertIn("terms=6", finals[0])
        self.assertIn("applied-term=1", finals[0])
        for segment in result:
            self.assertNotIn(str(segment["text"]), finals[0])


class ResolveCourseTermsTests(unittest.TestCase):
    def test_payload_glossary_and_env_file_merge_deduped(self):
        with tempfile.TemporaryDirectory() as temporary:
            term_file = Path(temporary) / "terms.txt"
            term_file.write_text("费米能级\n\n电势\n", encoding="utf-8")
            with patch.dict(os.environ, {"COURSELENS_TERM_GLOSSARY_FILE": str(term_file)}):
                resolved = resolve_course_terms({"glossary": ["能带图", "能带图", ""]})
        self.assertEqual(resolved, ("能带图", "费米能级", "电势"))

    def test_absent_sources_return_empty(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_course_terms({}), ())
            self.assertEqual(resolve_course_terms(None), ())


class ThinkingTierTests(unittest.TestCase):
    """H-SUBDEEP-SUP3：思考档参数透传、缓存键档位隔离、usage 落账。"""

    def test_thinking_field_is_forwarded_to_chat(self):
        payloads = []
        captured_kwargs = []

        def fake_chat(api_key, messages, **kwargs):
            payloads.append(json.loads(messages[1]["content"]))
            captured_kwargs.append(kwargs.get("thinking"))
            return "[]"

        segments = [seg(0, 1000, "能耐图")]
        with patch("courselens_worker.llm._chat", side_effect=fake_chat):
            term_proofread_segments(
                "secret", segments, terms=TERMS,
                thinking={"type": "enabled", "reasoning_effort": "low"},
            )
        self.assertEqual(
            captured_kwargs, [{"type": "enabled", "reasoning_effort": "low"}]
        )

    def test_default_tier_is_disabled_v4(self):
        captured = []

        def fake_chat(api_key, messages, **kwargs):
            captured.append(kwargs.get("thinking"))
            return "[]"

        with patch("courselens_worker.llm._chat", side_effect=fake_chat):
            term_proofread_segments("secret", [seg(0, 1000, "原文")], terms=TERMS)
        self.assertEqual(
            captured[0], {"type": "disabled"},
            "v4 缺省=关思考（成本 1/16，质量由提示词+示例库补齐）",
        )

    def test_env_overrides_tier_with_closed_set(self):
        captured = []

        def fake_chat(api_key, messages, **kwargs):
            captured.append(kwargs.get("thinking"))
            return "[]"

        cases = {
            "provider-default": None,
            "default": None,
            "disabled": {"type": "disabled"},
            "high": {"type": "enabled", "reasoning_effort": "high"},
            "low": {"type": "enabled", "reasoning_effort": "low"},
        }
        for raw, expected in cases.items():
            with (
                patch.dict(os.environ, {"COURSELENS_TERM_THINKING": raw}),
                patch("courselens_worker.llm._chat", side_effect=fake_chat),
            ):
                term_proofread_segments("secret", [seg(0, 1000, "原文")], terms=TERMS)
            self.assertEqual(captured[-1], expected, raw)
        # 无效值回退模块缺省（disabled），绝不崩
        with (
            patch.dict(os.environ, {"COURSELENS_TERM_THINKING": "bogus"}),
            patch("courselens_worker.llm._chat", side_effect=fake_chat),
        ):
            term_proofread_segments("secret", [seg(0, 1000, "原文")], terms=TERMS)
        self.assertEqual(captured[-1], {"type": "disabled"})

    def test_cache_key_is_isolated_per_tier(self):
        calls = {"n": 0}
        cache: dict[str, str] = {}

        def fake_chat(api_key, messages, **kwargs):
            calls["n"] += 1
            return "[]"

        segments = [seg(0, 1000, "能耐图")]
        # 同一文本两个档位必须各自真实调用（缓存键含档位）。
        with patch("courselens_worker.llm._chat", side_effect=fake_chat):
            term_proofread_segments(
                "secret", segments, terms=TERMS, cache=cache,
                thinking={"type": "enabled", "reasoning_effort": "low"},
            )
            term_proofread_segments(
                "secret", segments, terms=TERMS, cache=cache,
                thinking={"type": "disabled"},
            )
        self.assertEqual(calls["n"], 2)
        # 同档同文本复跑命中缓存。
        with patch("courselens_worker.llm._chat", side_effect=fake_chat):
            term_proofread_segments(
                "secret", segments, terms=TERMS, cache=cache,
                thinking={"type": "enabled", "reasoning_effort": "low"},
            )
        self.assertEqual(calls["n"], 2)

    def test_usage_sink_records_call_counters(self):
        def fake_chat(api_key, messages, **kwargs):
            # 模拟 _chat 落账
            with patch("courselens_worker.llm._USAGE_LOCK"):
                from courselens_worker import llm as llm_mod

                llm_mod._CALL_LOG.append({
                    "prompt_tokens": 100, "completion_tokens": 50,
                    "reasoning_tokens": 30, "prompt_cache_hit_tokens": 64,
                    "latency_ms": 900, "thinking": None,
                })
            return "[]"

        usage: list[dict] = []
        with patch("courselens_worker.llm._chat", side_effect=fake_chat):
            term_proofread_segments(
                "secret", [seg(0, 1000, "原文")], terms=TERMS, usage_sink=usage,
            )
        self.assertEqual(len(usage), 1)
        self.assertEqual(usage[0]["reasoning_tokens"], 30)
        self.assertEqual(usage[0]["completion_tokens"], 50)

    def test_apply_term_stage_aggregates_deep_usage(self):
        value = {"segments": [seg(0, 1000, "就能得到能耐图")]}
        response = json.dumps([{"id": "t0", "old": "能耐图", "new": "能带图"}])

        def fake_chat(api_key, messages, **kwargs):
            from courselens_worker import llm as llm_mod

            llm_mod._CALL_LOG.append({
                "prompt_tokens": 100, "completion_tokens": 50,
                "reasoning_tokens": 30, "prompt_cache_hit_tokens": 64,
                "latency_ms": 900, "thinking": None,
            })
            return response

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("courselens_worker.llm._chat", side_effect=fake_chat),
        ):
            _apply_term_stage(
                value, api_key="k",
                payload={"glossary": ["能带图"]},
                checkpoint_writer=None, warnings=[],
            )
        self.assertEqual(value["segments"][0]["text"], "就能得到能带图")
        usage = value["deep_usage"]
        self.assertEqual(usage["calls"], 1)
        self.assertEqual(usage["reasoning_tokens"], 30)
        self.assertEqual(usage["prompt_cache_hit_tokens"], 64)
        self.assertEqual(usage["latency_seconds"], 0.9)


class ApplyTermStageTests(unittest.TestCase):
    def _value(self):
        return {"segments": [seg(0, 1000, "就能得到能耐图")], "metrics": {}}

    def test_runner_wiring_passes_task_level_cache(self):
        """N15 接线钉：生产调用点把任务级 cache dict 传入 term_proofread_segments
        （自适应分窗/重试的同键二连调用第二跳零网络）；每任务一个新 dict，
        跨任务不共享。缓存命中语义本体由 TermWindowTests.test_cache_hit_skips_chat 钉。"""
        value = self._value()
        captured: list = []

        def _fake_term(api_key, segments, **kwargs):
            captured.append(kwargs.get("cache"))
            return list(segments)

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("courselens_worker.llm.term_proofread_segments", side_effect=_fake_term),
        ):
            _apply_term_stage(value, api_key="k", payload={}, checkpoint_writer=None, warnings=[])
            _apply_term_stage(value, api_key="k", payload={}, checkpoint_writer=None, warnings=[])
        self.assertEqual(len(captured), 2)
        self.assertIsInstance(captured[0], dict)
        self.assertIsInstance(captured[1], dict)
        self.assertIsNot(captured[0], captured[1])

    def test_skips_without_api_key(self):
        value = self._value()
        with patch.dict(os.environ, {}, clear=True):
            _apply_term_stage(value, api_key="", payload={}, checkpoint_writer=None, warnings=[])
            self.assertEqual(value["segments"][0]["text"], "就能得到能耐图")
            _apply_term_stage(
                value, api_key="k", payload={"glossary": ["能带图"]},
                checkpoint_writer=None, warnings=[],
            )
        self.assertEqual(value["segments"][0]["text"], "就能得到能耐图")

    def test_empty_payload_runs_v3_and_collects_audit(self):
        # v3：无术语表照跑（标点/常用词路径），审计账随 value.deep_audit 透出
        value = self._value()
        response = json.dumps([{"id": "t0", "old": "能耐图", "new": "能带图"}])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("courselens_worker.llm._chat", return_value=response),
        ):
            _apply_term_stage(value, api_key="k", payload={}, checkpoint_writer=None, warnings=[])
        self.assertEqual(value["segments"][0]["text"], "就能得到能带图")
        audit = value["deep_audit"]
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["before"], "就能得到能耐图")
        self.assertEqual(audit[0]["after"], "就能得到能带图")
        self.assertEqual(audit[0]["start_ms"], 0)

    def test_env_kill_switch_disables_stage(self):
        value = self._value()

        def fail_chat(api_key, messages, **kwargs):
            raise AssertionError("switch off must not call the LLM")

        with (
            patch.dict(os.environ, {"COURSELENS_TERM_PROOFREAD": "0"}),
            patch("courselens_worker.llm._chat", side_effect=fail_chat),
        ):
            _apply_term_stage(
                value, api_key="k",
                payload={"glossary": ["能带图"]},
                checkpoint_writer=None, warnings=[],
            )
        self.assertEqual(value["segments"][0]["text"], "就能得到能耐图")

    def test_payload_glossary_terms_correct_segments(self):
        value = self._value()
        response = json.dumps([{"id": "t0", "old": "能耐图", "new": "能带图"}])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("courselens_worker.llm._chat", return_value=response),
        ):
            _apply_term_stage(
                value, api_key="k",
                payload={"glossary": ["能带图"]},
                checkpoint_writer=None, warnings=[],
            )
        self.assertEqual(value["segments"][0]["text"], "就能得到能带图")
        self.assertEqual(value["segments"][0]["term_revision"], TERM_PROOFREAD_VERSION)

    def test_llm_failure_degrades_with_warning_and_keeps_segments(self):
        value = self._value()
        warnings: list[str] = []
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("courselens_worker.llm._chat", side_effect=LLMError("boom")),
            patch("courselens_worker.llm.time.sleep"),
        ):
            _apply_term_stage(
                value, api_key="k",
                payload={"glossary": ["能带图"]},
                checkpoint_writer=None, warnings=warnings,
            )
        self.assertEqual(warnings, ["term_proofread_degraded"])
        self.assertEqual(value["segments"][0]["text"], "就能得到能耐图")

    def test_checkpoint_preserves_prior_state_keys(self):
        value = self._value()
        written = []
        prior = {"completed_chunks": 3, "proofread_segments": [{"text": "x"}], "stage": "proofread"}
        response = json.dumps([{"id": "t0", "old": "能耐图", "new": "能带图"}])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("courselens_worker.llm._chat", return_value=response),
        ):
            _apply_term_stage(
                value, api_key="k",
                payload={"glossary": ["能带图"], "checkpoint": prior},
                checkpoint_writer=written.append, warnings=[],
            )
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["completed_chunks"], 3)
        self.assertIn("proofread_segments", written[0])
        self.assertEqual(written[0]["stage"], "term_proofread")


class FormatsTermRevisionTests(unittest.TestCase):
    def test_term_revision_survives_normalization(self):
        normalized = normalize_segments([
            seg(0, 1000, "能带图", term_revision=TERM_PROOFREAD_VERSION, correction=TERM_APPLIED_STATUS),
        ])
        self.assertEqual(normalized[0]["term_revision"], TERM_PROOFREAD_VERSION)


# ---- V4NONTHINK-1 新钉面：重叠窗已在上（TermWindowTests），以下为
# 分歧跨度（suspects）/示例库检索/讲内一致性锚定/金样本回归。 ----


class SuspectsTests(unittest.TestCase):
    def test_homophone_span_becomes_anchored_triple(self):
        from courselens_worker.llm import _disagreement_suspects

        suspects = _disagreement_suspects("平衡态它就代表肺敏能级", "平衡态它就代表费米能级")
        self.assertEqual(suspects, ["代表|肺敏|费米"])

    def test_pure_punct_and_one_sided_spans_are_skipped(self):
        from courselens_worker.llm import _disagreement_suspects

        self.assertEqual(_disagreement_suspects("你好，世界", "你好世界"), [])
        self.assertEqual(_disagreement_suspects("你好世界", "你好，世界"), [])
        # 单侧空（增删）不进裁决
        self.assertEqual(_disagreement_suspects("我们开始上课", "我们马上开始上课"), [])

    def test_long_blocks_and_cap(self):
        from courselens_worker.llm import _SUSPECT_MAX_PER_SEGMENT, _disagreement_suspects

        left = "甲" * 20
        self.assertEqual(_disagreement_suspects(f"前缀{left}后缀", "前缀乙" * 1 + "后缀"), [])
        text = "一二三四五六七八" * 4
        alt = "一二三四五六七八" * 4
        # 构造 4 个短分歧：逐词替换
        left = "AA甲BB甲CC甲DD甲EE"
        right = "AA乙BB乙CC乙DD乙EE"
        suspects = _disagreement_suspects(left, right)
        self.assertEqual(len(suspects), _SUSPECT_MAX_PER_SEGMENT)

    def test_alt_segments_feed_wire_and_prompt_mode(self):
        primary = [seg(0, 1000, "平衡态它就代表肺敏能级")]
        alts = [seg(0, 1000, "平衡态它就代表费米能级")]
        captured_prompts = []
        payloads = []

        def fake_chat(api_key, messages, **kwargs):
            captured_prompts.append(messages[0]["content"])
            payloads.append(json.loads(messages[1]["content"]))
            return "[]"

        with patch("courselens_worker.llm._chat", side_effect=fake_chat):
            term_proofread_segments("secret", primary, terms=TERMS, alt_segments=alts)
        self.assertEqual(payloads[0][0].get("suspects"), ["代表|肺敏|费米"])
        self.assertIn("分歧候选", captured_prompts[0])
        self.assertIn("裁决", captured_prompts[0])

    def test_no_alternates_keeps_plain_prompt(self):
        captured_prompts = []

        def fake_chat(api_key, messages, **kwargs):
            captured_prompts.append(messages[0]["content"])
            return "[]"

        with patch("courselens_worker.llm._chat", side_effect=fake_chat):
            term_proofread_segments("secret", [seg(0, 1000, "原文")], terms=TERMS)
        self.assertNotIn("suspects", captured_prompts[0])


class ExampleLibraryTests(unittest.TestCase):
    def test_library_has_ten_generalized_families(self):
        from courselens_worker.llm import _EXAMPLE_LIBRARY

        self.assertEqual(len(_EXAMPLE_LIBRARY), 10)
        families = {entry["family"] for entry in _EXAMPLE_LIBRARY}
        self.assertEqual(len(families), 10)
        joined = json.dumps(_EXAMPLE_LIBRARY, ensure_ascii=False)
        # 通用化纪律：零半导体专名（能带/费米/掺杂/电视 不入示例库）
        for banned in ("能带", "费米", "掺杂", "电势"):
            self.assertNotIn(banned, joined)

    def test_default_retrieval_and_triggers(self):
        from courselens_worker.llm import _select_examples

        default = _select_examples("纯中文窗口没有触发", ())
        self.assertEqual(
            [entry["family"] for entry in default],
            ["term-homophone", "punct", "cross-segment", "protect-number"],
        )
        latin = _select_examples("窗口里有 DNA 和 for 循环", ())
        self.assertEqual(latin[-1]["family"], "mixed-latin")
        demix = _select_examples("他做的不对我写的不对他说得对", ())
        self.assertEqual(demix[-1]["family"], "de-mixing")
        both = _select_examples("for 循环他做的不对我写的不对", ())
        self.assertEqual(len(both), 6)

    def test_course_examples_first_and_cap(self):
        from courselens_worker.llm import _select_examples

        course = (
            {"input": [{"id": "c0", "text": "课程专属示例"}], "ops": []},
        )
        picked = _select_examples("普通窗口", course)
        self.assertEqual(picked[0]["input"][0]["text"], "课程专属示例")

    def test_render_shape_matches_output_contract(self):
        from courselens_worker.llm import _render_example

        text = _render_example({
            "input": [{"id": "e0", "text": "原文"}],
            "ops": [{"id": "e0", "old": "原", "new": "改"}],
        })
        self.assertIn("示例：输入", text)
        self.assertIn("输出", text)
        self.assertIn('"id":"e0"', text)
        self.assertIn('"old":"原"', text)

    def test_prompt_examples_change_system_prompt_and_cache_key(self):
        # 同载荷不同示例检索面（latin 触发）必须各自真实调用（缓存键含提示指纹）
        calls = {"n": 0}
        cache: dict[str, str] = {}

        def fake_chat(api_key, messages, **kwargs):
            calls["n"] += 1
            return "[]"

        segments = [seg(0, 1000, "看这段 for 循环代码能耐图")]
        with patch("courselens_worker.llm._chat", side_effect=fake_chat):
            term_proofread_segments("secret", segments, terms=TERMS, cache=cache)
        self.assertEqual(calls["n"], 1)
        with patch("courselens_worker.llm._chat", side_effect=fake_chat):
            term_proofread_segments("secret", segments, terms=TERMS, cache=cache)
        self.assertEqual(calls["n"], 1, "同窗同提示命中缓存")

    def test_resolve_course_examples_conforming_only(self):
        from courselens_worker.glossary import resolve_course_examples

        payload = {
            "examples": [
                {"input": [{"id": "e0", "text": "文本"}], "ops": [{"id": "e0", "old": "原", "new": "改"}]},
                {"input": [{"id": "e0"}], "ops": []},  # 缺 text → 整条丢弃
                "garbage",
                {"input": [], "ops": []},  # 空 input → 丢弃
            ],
        }
        resolved = resolve_course_examples(payload)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["ops"][0]["new"], "改")
        self.assertEqual(resolve_course_examples({}), ())
        self.assertEqual(resolve_course_examples(None), ())


class ConsistencyAnchorTests(unittest.TestCase):
    def test_trusted_mapping_patches_same_shape_residual(self):
        from courselens_worker.llm import _consistency_anchor_pass

        segments = [seg(0, 1000, "这里电视出现了第三次的残留")]
        audit = [
            {"before": "电视是两点零", "after": "电势是两点零"},
            {"before": "电视虽然高", "after": "电势虽然高"},
        ]
        stats = _consistency_anchor_pass(segments, audit)
        self.assertEqual(stats["trusted_mappings"], 1)
        self.assertEqual(stats["applied_segments"], 1)
        self.assertEqual(segments[0]["text"], "这里电势出现了第三次的残留")
        self.assertEqual(segments[0]["correction"], TERM_APPLIED_STATUS)

    def test_single_occurrence_mapping_is_not_trusted(self):
        from courselens_worker.llm import _consistency_anchor_pass

        segments = [seg(0, 1000, "电视残留在这里")]
        audit = [{"before": "电视是两点零", "after": "电势是两点零"}]
        stats = _consistency_anchor_pass(segments, audit)
        self.assertEqual(stats["trusted_mappings"], 0)
        self.assertEqual(segments[0]["text"], "电视残留在这里")

    def test_naive_single_char_and_inclusion_mappings_are_rejected(self):
        # 禁 naive 单字映射（视→势 毁文本教训）+ 禁包含关系映射（能带→能带图）
        from courselens_worker.llm import _consistency_anchor_pass

        counts: dict = {}
        from courselens_worker.llm import _extract_anchor_pairs

        _extract_anchor_pairs("电视电视", "电势电势", counts)
        self.assertNotIn(("视", "势"), counts)
        self.assertEqual(counts.get(("电视", "电势")), 2)
        inclusion: dict = {}
        _extract_anchor_pairs("能带能带", "能带图能带图", inclusion)
        self.assertEqual(
            [pair for pair in inclusion if "能带图" in pair[1]],
            [],
            "包含关系映射不取",
        )

    def test_pure_punct_diff_yields_no_mapping(self):
        from courselens_worker.llm import _extract_anchor_pairs

        counts: dict = {}
        _extract_anchor_pairs("然后我们能看", "然后，我们能看。", counts)
        self.assertEqual(counts, {})

    def test_protected_and_gates_still_guard_anchor_application(self):
        from courselens_worker.llm import _consistency_anchor_pass

        # 可信映射正常应用（电视→电势，来自两条真实形态审计 diff）
        segments = [seg(0, 1000, "这里电视残留")]
        audit = [
            {"before": "电视是两点零", "after": "电势是两点零"},
            {"before": "电视虽然高", "after": "电势虽然高"},
        ]
        stats = _consistency_anchor_pass(segments, audit)
        self.assertEqual(stats["applied_segments"], 1)
        self.assertEqual(segments[0]["text"], "这里电势残留")
        # 内容漂移超帽的种子映射经确定性门拒绝（fail-closed，零改动）
        segments2 = [seg(0, 1000, "这个流程要运行")]
        stats2 = _consistency_anchor_pass(
            segments2, [], seeded=[["运行", "运行的系统流程特别复杂", 2]],
        )
        self.assertEqual(stats2["applied_segments"], 0)
        self.assertEqual(segments2[0]["text"], "这个流程要运行")

    def test_seeded_checkpoint_mappings_survive_resume(self):
        from courselens_worker.llm import _consistency_anchor_pass

        segments = [seg(0, 1000, "电视残留")]
        stats = _consistency_anchor_pass(segments, [], seeded=[["电视", "电势", 2]])
        self.assertEqual(stats["applied_segments"], 1)
        self.assertEqual(segments[0]["text"], "电势残留")

    def test_long_block_mapping_is_skipped(self):
        from courselens_worker.llm import _consistency_anchor_pass

        segments = [seg(0, 1000, "甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲")]
        audit = [{
            "before": "乙" * 20,
            "after": "丙" * 20,
        }]
        # 差异块 20 字 > 帽 16 → 不成映射
        stats = _consistency_anchor_pass(segments, audit)
        self.assertEqual(stats["trusted_mappings"], 0)

    def test_checkpoint_carries_anchor_mappings(self):
        written = []
        audit: list[dict] = []
        primary = [
            seg(0, 1000, "电视是两点零"), seg(2000, 3000, "电视虽然高"),
            seg(4000, 5000, "电视残留第三处"),
        ]
        responses = json.dumps([
            {"id": "t0", "old": "电视", "new": "电势"},
            {"id": "t1", "old": "电视", "new": "电势"},
        ])
        with patch("courselens_worker.llm._chat", return_value=responses):
            out = term_proofread_segments(
                "secret", primary, terms=TERMS, checkpoint=written.append,
                audit_sink=audit,
            )
        self.assertTrue(written)
        self.assertIn("term_anchor_mappings", written[-1])
        self.assertIn(["电视", "电势", 2], written[-1]["term_anchor_mappings"])
        # 锚定补漏：第三处同形残留被整讲映射补上（返回结果面），且入审计账
        self.assertEqual(out[2]["text"], "电势残留第三处")
        self.assertEqual(len(audit), 3)


# ---- 件8：金样本回归钉（35 条真实审计 diff 按族固化，纯本地 mock _chat） ----
# 来源：SUP3 tier_matrix 审计抽评 20 条（high/none 档）+ SUBDEEP Phase C 人工
# 抽评 20 条 + 文档在案真实案例。每条=(段文本, 提案) → 断言验证门判定。
# 保护验证门不被未来改动回归；gate verdict 是唯一断言面（LLM 质量属评测面）。
GOLDEN_SAMPLES = [
    # --- 族1 术语同音（4 条；tier_matrix high 档审计实证）---
    ("所以一个降落的电视要承上一个负q的话", "电视", "电势", "applied"),
    ("平衡态它就代表肺敏能级它的位置", "肺敏能级", "费米能级", "applied"),
    ("它的扩散跟票移它是已经相互平衡了", "票移", "漂移", "applied"),
    ("就能得到我的这样一个能耐图", "能耐图", "能带图", "applied"),
    # --- 族2 跨段截断（3 条；SUBDEEP 残留主因家族）---
    ("平衡态它就代表肺敏能", "肺敏能", "费米能", "applied"),
    ("这一段我们讲空间联合区的问题", "空间联合区", "空间电荷区", "applied"),
    ("能耐图和能耐图都在这里", "能耐图", "能带图", "rejected-ambiguous"),
    # --- 族3 数字单位（4 条）---
    ("电压是5伏的能耐图", "5伏", "6伏", "rejected-protected"),
    ("温度升高了23摄氏度然后我们看图", "温度升高了23摄氏度", "温度升高了23摄氏度，", "applied"),
    ("压强是1.01乘十的五次方帕", "1.01乘", "1.01乘以", "applied"),
    ("这个值是二十三", "二十三", "三十三", "rejected-protected"),
    # --- 族4 否定保护（3 条）---
    ("注意这不是扩散而是漂移", "这不是扩散", "这不是扩散，", "applied"),
    ("他不来了我们开始吧", "不来了", "来了", "rejected-protected"),
    ("没有电流的时候能带是平的", "没有电流", "没有电流，", "applied"),
    # --- 族5 叠标点（3 条；Phase C 叠标点三连环家族）---
    ("所以你们那个绩点还是挺重要的", "所以你们", "所以，，你们", "rejected-term"),
    ("然后我们能看电势它就是这个", "然后我们能看", "然后，我们能看", "applied"),
    ("好的，那我们继续看下一页", "好的，", "好的。", "rejected-term"),
    # --- 族6 标点闭集（4 条）---
    ("我们看书中的这段话就行", "这段话", "《这段话》", "rejected-term"),
    ("注意下面的公式推导", "注意下面的公式推导", "注意：下面的公式推导。", "applied"),
    ("然后我们看下一页的内容", "下一页", "下一页—", "rejected-term"),
    ("问题是什么呢问题是我们没有准备", "什么呢问题", "什么呢？问题", "applied"),
    # --- 族7 长度漂移（3 条）---
    ("这个器件很有意思", "很有意思", "有意思多了而且特别特别长", "rejected-term"),
    ("这个器件很有意思", "很有意思", "挺有意思", "applied"),
    ("就能得到我的这样一个能耐图", "能耐图", "能带图结构说明文字", "rejected-term"),
    # --- 族8 人名与译名克制（3 条；文科通识家族）---
    ("我是历史系的王浩然今天讲经济史", "王浩然", "王浩然，", "applied"),
    ("正如爱恩斯坦所说的那样", "爱恩斯坦", "爱因斯坦", "applied"),
    ("接下来有请李四教授发言", "李四教授", "李四教授。", "applied"),
    # --- 族9 中英夹杂与标点混合（2 条）---
    ("接下来看for循环里面的边界条件", "接下来看for循环里面的边界条件", "接下来看 for 循环里面的边界条件。", "applied"),
    ("这个参数等于零点五然后看输出", "等于零点五", "等于 0.5", "rejected-protected"),
    # --- 族10 审计实测误改防线（5 条；SUP3 none 档失败案/乘上误改案/tier_matrix 实证）---
    ("降落的电视要承上一个负q", "承上", "乘上", "applied"),
    ("从这个它里面飘就是扩散走了电荷", "它里面飘", "它里面漂", "applied"),
    ("肺敏能力一定是占一根的这种状态", "肺敏能力", "费米能级", "applied"),
    ("电场就能得到电视电视然后再再负q一下", "电视电视", "电势电势", "applied"),
    ("什么情况下会有电流呢哎就是说我这个能耐弯曲的情况下", "能耐弯曲", "能带弯曲", "applied"),
    ("这就是电子的能量乘以负q啊所谓的耐度啊", "耐度", "能带", "applied"),
]


class GoldenSampleTests(unittest.TestCase):
    """35 条真实审计 diff 按族过验证门：verdict 是唯一断言面。"""

    def test_thirty_five_golden_samples_hit_expected_verdicts(self):
        from courselens_worker.llm import _apply_term_ops

        self.assertEqual(len(GOLDEN_SAMPLES), 35)
        normalized = _normalized_terms(TERMS)
        mismatches = []
        for index, (text, old, new, expected) in enumerate(GOLDEN_SAMPLES):
            chunk = [{
                "id": "t0",
                "start_ms": 0,
                "end_ms": 1000,
                "text": text,
                "primary": {"start_ms": 0, "end_ms": 1000, "text": text},
            }]
            ops = [{"id": "t0", "old": old, "new": new}]
            out = _apply_term_ops(chunk, ops, normalized)
            status = out[0].get("correction")
            applied = status == TERM_APPLIED_STATUS
            if expected == "applied":
                ok = applied and new in out[0]["text"]
            else:
                ok = not applied and status == expected and out[0]["text"] == text
            if not ok:
                mismatches.append((index, expected, status, out[0]["text"]))
        self.assertEqual(mismatches, [], "金样本判定失配")

    def test_golden_samples_end_to_end_through_full_stage(self):
        # 端到端：35 条经完整 term_proofread_segments（mock _chat 逐条喂提案），
        # 判定一致且 applied 条带 v4 版本戳（锚定补漏不越权新增改动）。
        from courselens_worker.llm import _apply_term_ops  # noqa: F401 保证门已导入

        for text, old, new, expected in GOLDEN_SAMPLES:
            segments = [seg(0, 1000, text)]
            response = json.dumps([{"id": "t0", "old": old, "new": new}])
            with patch("courselens_worker.llm._chat", return_value=response):
                out = term_proofread_segments("secret", segments, terms=TERMS)
            status = out[0].get("correction")
            if expected == "applied":
                self.assertEqual(status, TERM_APPLIED_STATUS, (text, old))
                self.assertEqual(out[0]["term_revision"], TERM_PROOFREAD_VERSION)
                self.assertIn(new, out[0]["text"])
            else:
                self.assertEqual(status, expected, (text, old))
                self.assertEqual(out[0]["text"], text)
                self.assertNotIn("term_revision", out[0])


if __name__ == "__main__":
    unittest.main()
