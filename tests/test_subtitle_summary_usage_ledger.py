"""RR-ACCOUNT2-1：字幕/总结任务族的真实 token 消耗必须随结果上报（AS6 记账链）。

WINIT-1 实测缺口：0dafe42 补齐 question/answer 后，字幕（词级校对窗+term 深校
对段）与总结（窗口+合并）任务的 deepseek_tokens 仍恒 0——usage_sink 收了账但
从不进 result.metrics，客户端 _record_task_usage 无数可落。本钉锁住「两个任务
族的 result.metrics 汇总各自链路真实 token」这一行为（纯观测计数，prompt+
completion 与 answer/deep_usage 同口径；各分支增量聚合防组合任务互相覆盖）。
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from courselens_worker import llm as llm_mod
from courselens_worker.runner import _process_materialized_job


def _usage_record(prompt: int, completion: int) -> dict:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "latency_ms": 12,
        "thinking": None,
    }


def _subtitle_job() -> dict:
    return {
        "job_kind": "subtitle",
        "task_id": "task-usage-sub",
        "input_hash": "0" * 64,
        "pipeline": {"version": "test-v2"},
        "payload": {
            "mode": "automatic",
            "media": {"start_seconds": 0, "duration_seconds": 60},
        },
        "secrets": {"deepseek_api_key": "sk-synthetic"},
    }


def _summary_job() -> dict:
    return {
        "job_kind": "summary",
        "task_id": "task-usage-sum",
        "input_hash": "0" * 64,
        "pipeline": {"version": "test-v2"},
        "payload": {
            "title": "测试课程",
            "transcript": [
                {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"text-{index}"}
                for index in range(130)
            ],
            "slides": [],
        },
        "secrets": {"deepseek_api_key": "sk-synthetic"},
    }


class SubtitleUsageLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        llm_mod.reset_usage()

    def test_subtitle_job_reports_chain_token_usage(self) -> None:
        """字幕任务（kind=subtitle）：term 深校对窗的真实 token 进 result.metrics。"""

        def _fake_chat(api_key, messages, **kwargs):
            with llm_mod._USAGE_LOCK:
                llm_mod._CALL_LOG.append(_usage_record(100, 50))
            return "[]"  # 术语窗合法空修正：不改动任何段

        transcribe_value = {
            "mode": "automatic",
            "segments": [
                {"start_ms": 0, "end_ms": 2000, "text": "第一段"},
                {"start_ms": 2000, "end_ms": 4000, "text": "第二段"},
            ],
            "raw_sensevoice": [],
            "raw_paraformer": [],
            "metrics": {"asr_seconds": 1.0},
        }
        with patch.dict(
            "os.environ", {"SENSEVOICE_MODEL_DIR": "s", "PARAFORMER_MODEL_DIR": "p"}
        ), \
                patch("courselens_worker.asr.transcribe", return_value=transcribe_value), \
                patch("courselens_worker.llm._chat", side_effect=_fake_chat):
            result = _process_materialized_job(_subtitle_job())

        # 2 段 → 1 个术语窗 → 一次调用 100+50（与 answer/deep_usage 同口径）。
        self.assertEqual(result["metrics"]["deepseek_tokens"], 150)
        # outputs 里的 deep_usage 与 metrics 同账本（字幕链统一账本）。
        self.assertEqual(result["outputs"]["subtitle"]["deep_usage"]["calls"], 1)

    def test_word_level_proofread_drains_windows_into_usage_sink(self) -> None:
        """词级校对窗（transcribe 内部链）：usage_sink 逐窗收账、全局流水清零。"""
        sink: list[dict] = []

        def _fake_chat(api_key, messages, **kwargs):
            with llm_mod._USAGE_LOCK:
                llm_mod._CALL_LOG.append(_usage_record(30, 20))
            return "[]"  # 校对窗合法空修正

        with patch("courselens_worker.llm._chat", side_effect=_fake_chat):
            result = llm_mod.proofread_segments(
                "sk-synthetic",
                [{"start_ms": 0, "end_ms": 1000, "text": "粗识别一"},
                 {"start_ms": 1000, "end_ms": 2000, "text": "粗识别二"}],
                [{"start_ms": 0, "end_ms": 1000, "text": "精识别一"},
                 {"start_ms": 1000, "end_ms": 2000, "text": "精识别二"}],
                usage_sink=sink,
            )

        self.assertEqual(len(result), 2)
        self.assertEqual(
            sum(record["prompt_tokens"] + record["completion_tokens"] for record in sink),
            50,
        )
        # 已收进账本的流水不再留在全局日志里（不重不漏）。
        self.assertEqual(llm_mod.drain_call_log(), [])

    def test_subtitle_job_without_llm_reports_zero(self) -> None:
        """无 Key 的字幕任务零调用：如实计 0（不是「无记录」）。"""
        job = _subtitle_job()
        job["secrets"] = {}
        with patch.dict(
            "os.environ", {"SENSEVOICE_MODEL_DIR": "s", "PARAFORMER_MODEL_DIR": "p"}
        ), \
                patch(
                    "courselens_worker.asr.transcribe",
                    return_value={
                        "mode": "fallback",
                        "segments": [],
                        "raw_sensevoice": [],
                        "metrics": {},
                    },
                ):
            result = _process_materialized_job(job)
        self.assertEqual(result["metrics"]["deepseek_tokens"], 0)


class SummaryUsageLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        llm_mod.reset_usage()

    def test_summary_job_reports_chain_token_usage(self) -> None:
        """总结任务（kind=summary）：runner 传 usage_sink 并把链账汇总进 metrics。"""
        seen: dict = {}

        def _fake_summary(api_key, *, title, transcript, ppt_pages, prior_checkpoint,
                          checkpoint, usage_sink=None, **kwargs):
            seen["sink_passed"] = usage_sink is not None
            if usage_sink is not None:
                usage_sink.extend([_usage_record(300, 150), _usage_record(200, 100)])
            return {"model": "deepseek-chat", "markdown": "笔记", "chapters": []}

        with patch(
            "courselens_worker.ocr.process_slides",
            side_effect=lambda slides, **kw: ([], {}),
        ), \
                patch("courselens_worker.llm.create_summary", side_effect=_fake_summary):
            result = _process_materialized_job(_summary_job())

        self.assertTrue(seen["sink_passed"])
        self.assertEqual(result["metrics"]["deepseek_tokens"], 750)
        self.assertEqual(result["outputs"]["summary"]["markdown"], "笔记")

    def test_create_summary_extends_usage_sink_with_real_records(self) -> None:
        """真实 create_summary：窗口+合并（含失败尝试）的流水进 usage_sink。"""
        sink: list[dict] = []

        def _fake_chat(api_key, messages, **kwargs):
            with llm_mod._USAGE_LOCK:
                llm_mod._CALL_LOG.append(_usage_record(100, 50))
            if messages[0]["content"] == llm_mod._SUMMARY_MERGE_PROMPT:
                return json.dumps({"markdown": "combined", "chapters": []})
            return json.dumps({"markdown": "part", "chapters": []})

        with patch("courselens_worker.llm._chat", side_effect=_fake_chat):
            value = llm_mod.create_summary(
                "k",
                title="t",
                transcript=[
                    {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"text-{index}"}
                    for index in range(130)
                ],
                ppt_pages=[],
                usage_sink=sink,
            )

        self.assertEqual(value["markdown"], "combined")
        # 130 段 → 2 窗 + 1 合并 = 3 次调用，全部落账。
        self.assertEqual(len(sink), 3)
        self.assertEqual(
            sum(record["prompt_tokens"] + record["completion_tokens"] for record in sink),
            450,
        )


if __name__ == "__main__":
    unittest.main()
