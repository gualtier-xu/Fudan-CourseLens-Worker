"""H1-FORENSICS-1/N21 回归钉：summary/chapters 分支的课程记忆计数必须随任务完成上报。

活体事故（run 37139804938/37140805939）：计数块在分支 metrics 局部名绑定之前
引用 ``metrics["course_memory_terms"]``——自 09-29 起客户端对带课程记忆词的
课程恒注入 ``payload.glossary``（application.py 入队侧），OCR 刚完成后的毫秒级
纯 CPU 区确定性抛 UnboundLocalError，带 PPT 的总结在新钉代全灭（1497 页 OCR
白付后 2ms 死）。修法=计数块整体下移到 metrics 构造之后；本文件四钉锁住：
①payload glossary 计数+完成；②env 术语文件分源计数；③chapters 同验；
④重试带 checkpoint（活体 attempt-2 形态）同样完成，且计数在 deepseek_tokens
增量聚合后幸存（:737 整体重绑定抹计数的潜伏语义一并封死）。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from courselens_worker import llm as llm_mod
from courselens_worker.glossary import TERM_GLOSSARY_ENV
from courselens_worker.runner import _process_materialized_job


def _fake_summary(api_key, *, title, transcript, ppt_pages, prior_checkpoint,
                  checkpoint, usage_sink=None, **kwargs):
    if usage_sink is not None:
        usage_sink.extend([
            {"prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 0,
             "prompt_cache_hit_tokens": 0, "latency_ms": 1, "thinking": None},
        ])
    return {"model": "deepseek-flash", "markdown": "笔记", "chapters": []}


def _summary_job(kind: str, payload_extra: dict) -> dict:
    payload = {
        "title": "测试课程",
        "transcript": [
            {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"text-{index}"}
            for index in range(3)
        ],
        "slides": [],
    }
    payload.update(payload_extra)
    return {
        "job_kind": kind,
        "task_id": f"task-memory-{kind}",
        "input_hash": "0" * 64,
        "pipeline": {"version": "test-v2"},
        "payload": payload,
        "secrets": {"deepseek_api_key": "sk-synthetic"},
    }


class SummaryMemoryMetricsTests(unittest.TestCase):
    """分支内 metrics 计数块的位置约束：必须在 metrics 构造之后。"""

    def setUp(self) -> None:
        llm_mod.reset_usage()

    def test_summary_with_course_memory_glossary_counts_and_completes(self):
        """H1 精确复现形：payload.glossary 在场时 summary 必须完成且计数=1。"""
        job = _summary_job("summary", {"glossary": ["示例词"]})
        with patch(
            "courselens_worker.ocr.process_slides",
            side_effect=lambda slides, **kw: ([], {}),
        ), patch("courselens_worker.llm.create_summary", side_effect=_fake_summary):
            result = _process_materialized_job(job)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["metrics"]["course_memory_terms"], 1)
        self.assertNotIn("env_glossary_terms", result["metrics"])

    def test_summary_with_env_glossary_file_counts_env_terms(self):
        """env 术语文件词单独计数（分源口径），payload 无 glossary 时不冒充记忆注入。"""
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8")
        try:
            handle.write("热力学第一定律\n傅里叶变换\n")
            handle.close()
            job = _summary_job("summary", {})
            with patch.dict("os.environ", {TERM_GLOSSARY_ENV: handle.name}), patch(
                "courselens_worker.ocr.process_slides",
                side_effect=lambda slides, **kw: ([], {}),
            ), patch("courselens_worker.llm.create_summary", side_effect=_fake_summary):
                result = _process_materialized_job(job)
        finally:
            os.unlink(handle.name)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["metrics"]["env_glossary_terms"], 2)
        self.assertNotIn("course_memory_terms", result["metrics"])

    def test_chapters_with_memory_glossary_counts_and_completes(self):
        """chapters 分支与 summary 共用同一计数块，同验。"""
        job = _summary_job("chapters", {"glossary": ["词一", "词二", "词一"]})
        with patch(
            "courselens_worker.ocr.process_slides",
            side_effect=lambda slides, **kw: ([], {}),
        ), patch("courselens_worker.llm.create_summary", side_effect=_fake_summary):
            result = _process_materialized_job(job)
        self.assertEqual(result["status"], "completed")
        # 去重保序：重复词只计一次。
        self.assertEqual(result["metrics"]["course_memory_terms"], 2)

    def test_retry_with_checkpoint_payload_keeps_memory_counts(self):
        """活体 attempt-2 形态：重试载荷携带 checkpoint（OCR 已全量）时同路径
        必须完成；计数与后置 deepseek_tokens 增量聚合共存（重绑定抹计数封死）。"""
        job = _summary_job("summary", {
            "glossary": ["示例词"],
            "checkpoint": {
                "ppt_pages": [{"page_index": 0, "created_sec": 0}],
                "ppt_skipped": {},
            },
        })
        with patch("courselens_worker.llm.create_summary", side_effect=_fake_summary):
            result = _process_materialized_job(job)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["metrics"]["course_memory_terms"], 1)
        # 同一 metrics 字典上后置聚合不抹计数（潜伏 bug 二连钉）。
        self.assertEqual(result["metrics"]["deepseek_tokens"], 15)
        self.assertEqual(
            result["outputs"]["ppt_pages"], [{"page_index": 0, "created_sec": 0}]
        )


if __name__ == "__main__":
    unittest.main()
