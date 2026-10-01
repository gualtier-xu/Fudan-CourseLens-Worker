"""RR-PARK-1 P2：answer 任务的真实 token 消耗必须随结果上报（AS6 记账链）。

月账 deepseek_tokens 恒 0 的 worker 侧断点：answer 分支只记 evidence_count，
_chat 的每次调用 usage 流水从未进 metrics。本钉锁住「answer 分支 drain 调用
流水并聚合 prompt+completion」这一行为（纯观测计数，与 deep_usage 同口径）。
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from courselens_worker import llm as llm_mod
from courselens_worker.runner import process_job


def _answer_job() -> dict:
    return {
        "schema": "courselens.worker-job.v1",
        "protocol_version": 1,
        "task_id": "task-usage-1",
        "job_kind": "learning_pack",
        "input_hash": "0" * 64,
        "requested_outputs": ["answer"],
        "pipeline": {"version": "bookmark-answer-v1", "llm": "deepseek-flash"},
        "payload": {
            "query": "解释这一段",
            "evidence": [{"citation_id": "r1", "text": "证据"}],
        },
        "secrets": {"deepseek_api_key": "sk-synthetic"},
    }


class AnswerUsageLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        llm_mod.reset_usage()

    def test_answer_job_reports_aggregated_token_usage(self) -> None:
        def _fake_chat(api_key, messages, **kwargs):
            with llm_mod._USAGE_LOCK:
                llm_mod._CALL_LOG.append({
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "reasoning_tokens": 0,
                    "prompt_cache_hit_tokens": 0,
                    "latency_ms": 12,
                    "thinking": None,
                })
            return json.dumps(
                {"answer": "回答", "grounded": True, "citations": ["r1"]},
                ensure_ascii=False,
            )

        with patch("courselens_worker.llm._chat", side_effect=_fake_chat):
            result = process_job(_answer_job())

        self.assertEqual(result["metrics"]["deepseek_tokens"], 150)
        self.assertEqual(result["metrics"]["evidence_count"], 1)

    def test_declined_answer_without_llm_call_reports_zero(self) -> None:
        # 无 evidence 时 answer_question 直接诚实拒绝、零调用：计 0（如实
        # 观测，不是「无记录」；行级 NULL 语义仍留给历史行）。
        job = _answer_job()
        job["payload"]["evidence"] = []
        result = process_job(job)
        self.assertEqual(result["metrics"]["deepseek_tokens"], 0)


class AnswerCourseTermsPassthroughTests(unittest.TestCase):
    """RR-P6MEM-1：payload 课程记忆术语表透传 answer_question 并实报注入数。

    注入数是客户端学生可见标注（本回答已应用课程记忆 N 条）的唯一事实源：
    没带 glossary=无计数（客户端零标注）；带了就如实报数（去空去重后）。
    """

    def _run(self, payload_extra: dict) -> tuple[dict, list[dict]]:
        job = _answer_job()
        job["payload"].update(payload_extra)
        captured: list[dict] = []

        def _fake_chat(api_key, messages, **kwargs):
            captured.append(messages)
            return json.dumps(
                {"answer": "回答", "grounded": True, "citations": ["r1"]},
                ensure_ascii=False,
            )

        with patch("courselens_worker.llm._chat", side_effect=_fake_chat):
            result = process_job(job)
        return result, captured

    def test_payload_glossary_reaches_answer_and_reports_count(self) -> None:
        result, captured = self._run({"glossary": ["费米能级", "量子力学", " "]})
        self.assertEqual(result["metrics"]["course_memory_terms"], 2)
        user = captured[0][1]["content"]
        self.assertEqual(json.loads(user)["course_terms"], ["费米能级", "量子力学"])

    def test_no_glossary_keeps_legacy_shape_and_no_metric(self) -> None:
        result, captured = self._run({})
        self.assertNotIn("course_memory_terms", result["metrics"])
        self.assertNotIn("course_terms", json.loads(captured[0][1]["content"]))


if __name__ == "__main__":
    unittest.main()
