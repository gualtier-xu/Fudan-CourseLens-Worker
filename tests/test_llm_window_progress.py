"""N8：LLM 窗环 progress 发射（worker 侧补，客户端零改）。

夜14 定谳：客户端消费面已通（signed progress 信封 completed/total→percent），
但 llm.py 三个窗环零发射——summary/proofread/term 阶段学生看冻结进度条（N21
活体以最重形式现世：看完 10 分钟 OCR 后任务直接失败的观感同源）。本钉锁：
①三函数逐批发射且序列单调、终值=total；②summary 降级跳过的窗也计数推进
（进度诚实=含跳过）+merge 终值（total 含 merge 位）；③runner 接线把回调
传入（summary/term 两面 capture 钉）。
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from courselens_worker import llm as llm_mod
from courselens_worker.llm import REVIEW_VIEWS_ENV
from courselens_worker.runner import (
    _apply_term_stage,
    _process_materialized_job,
)


def _segment(index: int) -> dict:
    return {
        "start_ms": index * 1000,
        "end_ms": (index + 1) * 1000,
        "text": f"第{index}段原文",
    }


def _assert_monotonic(testcase, emissions, final_total: int) -> None:
    testcase.assertTrue(emissions, "至少一次发射")
    completed_values = [completed for _, completed, _ in emissions]
    testcase.assertEqual(completed_values, sorted(completed_values), "序列必须单调")
    for stage, completed, total in emissions:
        testcase.assertIn(stage, {"summary", "proofread", "term_proofread"})
        testcase.assertLessEqual(completed, total)
    testcase.assertEqual(emissions[-1][1], final_total, "终值 completed=total")
    testcase.assertEqual(emissions[-1][2], final_total)


class ProofreadProgressTests(unittest.TestCase):
    def test_batch_emissions_are_monotonic_and_reach_total(self):
        emissions: list[tuple[str, int, int]] = []
        with patch("courselens_worker.llm._chat", return_value="[]"):
            llm_mod.proofread_segments(
                "sk-synthetic",
                [_segment(index) for index in range(45)],
                [_segment(index) for index in range(45)],
                progress=lambda stage, completed, total: emissions.append(
                    (stage, completed, total)
                ),
            )
        # 45 段 → 3 窗（20 段/窗），批 [0,1] 后 (2,3)、批 [2] 后 (3,3)。
        _assert_monotonic(self, emissions, 3)
        self.assertEqual(emissions, [("proofread", 2, 3), ("proofread", 3, 3)])


class TermProgressTests(unittest.TestCase):
    def test_batch_emissions_are_monotonic_and_reach_total(self):
        emissions: list[tuple[str, int, int]] = []

        def fake_chat(api_key, messages, **kwargs):
            return json.dumps([{"id": "t0", "old": "能耐图", "new": "能带图"}])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("courselens_worker.llm._chat", side_effect=fake_chat),
        ):
            llm_mod.term_proofread_segments(
                "sk-synthetic",
                [_segment(index) for index in range(45)],
                terms=("能带图",),
                progress=lambda stage, completed, total: emissions.append(
                    (stage, completed, total)
                ),
            )
        _assert_monotonic(self, emissions, emissions[-1][2])
        for stage, _, _ in emissions:
            self.assertEqual(stage, "term_proofread")


class SummaryProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        llm_mod.reset_usage()

    def test_windows_and_merge_emit_and_reach_total_including_merge_slot(self):
        emissions: list[tuple[str, int, int]] = []

        def fake_chat(api_key, messages, **kwargs):
            if messages[0]["content"] == llm_mod._SUMMARY_MERGE_PROMPT:
                return json.dumps({"markdown": "combined", "chapters": []})
            return json.dumps({"markdown": "part", "chapters": []})

        with (
            patch.dict(os.environ, {REVIEW_VIEWS_ENV: "off"}),
            patch("courselens_worker.llm._chat", side_effect=fake_chat),
        ):
            llm_mod.create_summary(
                "sk-synthetic",
                title="t",
                transcript=[_segment(index) for index in range(130)],
                ppt_pages=[],
                progress=lambda stage, completed, total: emissions.append(
                    (stage, completed, total)
                ),
            )
        # 130 段 → 2 窗（120 段/窗）同批 → 窗批 (2,3)；merge 完成 → 终值 (3,3)
        # （total 含 merge 位，与 completed_chunks=total_chunks+1 语义对齐）。
        _assert_monotonic(self, emissions, 3)
        self.assertEqual(emissions[-1], ("summary", 3, 3))

    def test_skipped_windows_still_advance_progress(self):
        emissions: list[tuple[str, int, int]] = []

        def fake_chat(api_key, messages, **kwargs):
            if messages[0]["content"] == llm_mod._SUMMARY_MERGE_PROMPT:
                return json.dumps({"markdown": "combined", "chapters": []})
            if "第120段" in messages[1]["content"]:
                # 窗口级重试预算盖坏形不盖传输异常（传输 LLMError 立即穿透）：
                # 坏形×穷尽 → 该窗降级跳过（SUMMARY-FIX-1 语义）。
                return "not-json-at-all"
            return json.dumps({"markdown": "part", "chapters": []})

        with (
            patch.dict(os.environ, {REVIEW_VIEWS_ENV: "off"}),
            patch("courselens_worker.llm._chat", side_effect=fake_chat),
        ):
            llm_mod.create_summary(
                "sk-synthetic",
                title="t",
                transcript=[_segment(index) for index in range(130)],
                ppt_pages=[],
                progress=lambda stage, completed, total: emissions.append(
                    (stage, completed, total)
                ),
            )
        # 同批两窗：一窗坏形穷尽跳过，进度仍推进到 (2,3)（诚实含跳过）。
        self.assertEqual(emissions[0], ("summary", 2, 3))
        _assert_monotonic(self, emissions, 3)


class RunnerWiringTests(unittest.TestCase):
    """runner 三处把既有 progress 回调传入（capture 钉）。"""

    def test_summary_branch_passes_progress_to_create_summary(self):
        seen: dict = {}

        def _fake_summary(api_key, *, title, transcript, ppt_pages, prior_checkpoint,
                          checkpoint, usage_sink=None, **kwargs):
            seen["progress"] = kwargs.get("progress")
            return {"model": "deepseek-flash", "markdown": "笔记", "chapters": []}

        job = {
            "job_kind": "summary",
            "task_id": "task-progress-sum",
            "input_hash": "0" * 64,
            "pipeline": {"version": "test-v2"},
            "payload": {
                "title": "测试课程",
                "transcript": [_segment(index) for index in range(3)],
                "slides": [],
            },
            "secrets": {"deepseek_api_key": "sk-synthetic"},
        }
        with patch(
            "courselens_worker.ocr.process_slides",
            side_effect=lambda slides, **kw: ([], {}),
        ), patch("courselens_worker.llm.create_summary", side_effect=_fake_summary):
            _process_materialized_job(job)
        self.assertIsNotNone(seen["progress"])
        seen["progress"]("summary", 1, 2)

    def test_apply_term_stage_passes_progress_through(self):
        seen: dict = {}

        def _fake_term(api_key, segments, **kwargs):
            seen["progress"] = kwargs.get("progress")
            return list(segments)

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("courselens_worker.llm.term_proofread_segments", side_effect=_fake_term),
        ):
            _apply_term_stage(
                {"segments": [_segment(0)]},
                api_key="k",
                payload={},
                checkpoint_writer=None,
                warnings=[],
                progress=lambda stage, completed, total: None,
            )
        self.assertIsNotNone(seen["progress"])


if __name__ == "__main__":
    unittest.main()
