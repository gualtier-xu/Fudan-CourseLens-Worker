import json
import os
import unittest
from unittest.mock import patch

from courselens_worker.llm import (
    PROOFREAD_PAIRING,
    REVIEW_VIEWS_ENV,
    answer_question,
    create_summary,
    proofread_segments,
)


class LLMCheckpointTests(unittest.TestCase):
    def test_answer_question_rejects_unreferenced_or_empty_evidence(self):
        self.assertFalse(answer_question("key", query="q", evidence=[])["grounded"])

    def test_answer_question_preserves_only_known_citation_ids(self):
        with patch("courselens_worker.llm._chat", return_value='{"answer":"回答","grounded":true,"citations":["r1","fake"]}'):
            value = answer_question("key", query="q", evidence=[{"citation_id": "r1", "text": "证据"}])
        self.assertTrue(value["grounded"])
        self.assertEqual(value["citations"], ["r1"])

    def test_answer_prompt_requires_full_answer_and_solution_approach(self):
        """C⑩ 用户拍板：完整答案+每题解题思路；提示词必须承载该要求。"""
        with patch("courselens_worker.llm._chat", return_value='{"answer":"回答","grounded":true,"citations":["r1"]}') as chat:
            answer_question("key", query="q", evidence=[{"citation_id": "r1", "text": "证据"}])
        system = chat.call_args.args[1][0]["content"]
        self.assertIn("完整答案", system)
        self.assertIn("解题思路", system)
        self.assertEqual(chat.call_args.kwargs.get("max_tokens"), 8192,
                         "完整答案+思路需要更大的输出预算（8192）")

    def test_proofread_resumes_after_completed_window(self):
        source = [
            {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"文本{index}"}
            for index in range(25)
        ]
        prior_segments = [
            {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"已{index}"}
            for index in range(20)
        ]
        response = json.dumps([
            {"id": f"p{index}", "old": f"文本{index}", "new": f"修正{index}"}
            for index in range(20, 25)
        ])
        checkpoints = []
        with patch("courselens_worker.llm._chat", return_value=response) as chat:
            result = proofread_segments(
                "secret",
                source,
                source,
                prior_checkpoint={
                    "proofread_pairing": PROOFREAD_PAIRING,
                    "proofread_completed_windows": 1,
                    "proofread_segments": prior_segments,
                },
                checkpoint=checkpoints.append,
            )
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(len(result), 25)
        self.assertEqual(result[-1]["text"], "修正24")
        self.assertEqual(result[-1]["correction"], "applied")
        self.assertEqual(result[0]["text"], "已0")
        self.assertEqual(checkpoints[-1]["proofread_completed_windows"], 2)
        self.assertEqual(checkpoints[-1]["proofread_pairing"], PROOFREAD_PAIRING)

    def test_proofread_legacy_checkpoint_restarts_without_trusting_old_segments(self):
        source = [
            {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"文本{index}"}
            for index in range(5)
        ]
        checkpoints = []
        with patch("courselens_worker.llm._chat", return_value=json.dumps([])) as chat:
            result = proofread_segments(
                "secret",
                source,
                source,
                prior_checkpoint={
                    "proofread_completed_windows": 1,
                    "proofread_segments": [
                        {"start_ms": 0, "end_ms": 1000, "text": "stale-legacy-rewrite"}
                    ],
                },
                checkpoint=checkpoints.append,
            )
        self.assertEqual(chat.call_count, 1)
        self.assertEqual([item["text"] for item in result], [f"文本{index}" for index in range(5)])
        self.assertNotIn("stale-legacy-rewrite", [item["text"] for item in result])
        self.assertEqual(checkpoints[-1]["proofread_pairing"], PROOFREAD_PAIRING)

    def test_proofread_resume_passes_slide_context_for_remaining_windows(self):
        source = [
            {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"文本{index}"}
            for index in range(25)
        ]
        prior_segments = [
            {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"已{index}"}
            for index in range(20)
        ]
        payloads = []

        def fake_chat(api_key, messages, **kwargs):
            payloads.append(json.loads(messages[1]["content"]))
            return json.dumps([])

        checkpoints = []
        with patch("courselens_worker.llm._chat", side_effect=fake_chat) as chat:
            proofread_segments(
                "secret",
                source,
                source,
                prior_checkpoint={
                    "proofread_pairing": PROOFREAD_PAIRING,
                    "proofread_completed_windows": 1,
                    "proofread_segments": prior_segments,
                },
                checkpoint=checkpoints.append,
                ppt_pages=[{"created_sec": 0, "text": "幻灯片术语"}],
            )
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(payloads[0][0]["id"], "p20")
        self.assertEqual(payloads[0][0]["slide"], "幻灯片术语")
        self.assertEqual(checkpoints[-1]["proofread_pairing"], PROOFREAD_PAIRING)

    def test_summary_resumes_map_windows_before_final_merge(self):
        transcript = [
            {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"text-{index}"}
            for index in range(240)
        ]
        first_part = {"markdown": "part one", "chapters": []}
        second_part = {"markdown": "part two", "chapters": []}
        final = {
            "markdown": "combined",
            "chapters": [{"title": "chapter", "start_ms": 120000, "summary": "summary"}],
        }
        checkpoints = []
        # 本钉面=窗口/合并 resume 语义（视图派生属 test_review_views.py），
        # 关掉派生调用保持既有调用数语义。
        with patch(
            "courselens_worker.llm._chat",
            side_effect=[json.dumps(second_part), json.dumps(final)],
        ) as chat, patch.dict(os.environ, {REVIEW_VIEWS_ENV: "off"}):
            result = create_summary(
                "secret",
                title="title",
                transcript=transcript,
                ppt_pages=[],
                prior_checkpoint={
                    "summary_completed_windows": 1,
                    "summary_parts": [first_part],
                },
                checkpoint=checkpoints.append,
            )
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(result["markdown"], "combined")
        self.assertEqual(result["chapters"][0]["start_ms"], 120000)
        self.assertEqual(checkpoints[-1]["summary_completed_windows"], 2)


    def test_summary_plan_drift_resumes_from_first_mismatched_window(self):
        """夜10-C T24：窗口计划漂移（证据包换内容）→ 从首个不一致窗重跑，
        之前的字幕窗计数保留；不重复调用未变的窗口。"""
        transcript = [
            {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"t{index}"}
            for index in range(240)
        ]
        first_part = {"markdown": "kept part", "chapters": []}
        second_part = {"markdown": "re-run part", "chapters": []}
        final = {"markdown": "combined", "chapters": []}
        packet_v2 = {
            "usable": True,
            "items": [
                {"kind": "document_page", "citation_id": "c-v2", "label": "v2 页",
                 "text": "second version page", "page": 1},
            ],
        }
        with (
            patch(
                "courselens_worker.llm._chat",
                side_effect=[json.dumps(second_part), json.dumps(final)],
            ) as chat,
            patch.dict(os.environ, {REVIEW_VIEWS_ENV: "off"}),
        ):
            result = create_summary(
                "secret",
                title="t",
                transcript=transcript,
                ppt_pages=[],
                evidence_packet=packet_v2,
                prior_checkpoint={
                    "summary_completed_windows": 2,
                    "summary_parts": [first_part, {"markdown": "stale", "chapters": []}],
                    # 旧计划：两窗都是字幕窗；新计划第二窗变成证据窗 → 从窗 2 重跑
                    "summary_window_plan": ["transcript", "transcript"],
                },
            )
        # 恰一次窗口调用（第 2 窗重跑）+ 一次合并调用；第 1 窗（计划一致）不重跑
        self.assertEqual(chat.call_count, 2)
        sent = json.loads(chat.call_args_list[0][0][1][1]["content"])
        self.assertIn("second version page", json.dumps(sent, ensure_ascii=False)[:200000])
        self.assertNotIn("kept part", json.dumps(sent), "重跑窗不携带旧第一窗内容")
        self.assertEqual(result["markdown"], "combined")
        self.assertEqual(len(result.get("chapters") or []), 0)

    def test_summary_checkpoint_records_the_window_plan(self):
        """N7A：新 checkpoint 记窗口计划；旧 checkpoint（无计划）行为不变。"""
        transcript = [
            {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"t{index}"}
            for index in range(240)
        ]
        first_part = {"markdown": "part one", "chapters": []}
        checkpoints = []
        with patch("courselens_worker.llm._chat",
                   return_value=json.dumps({"markdown": "m", "chapters": []})):
            create_summary("secret", title="t", transcript=transcript, ppt_pages=[],
                           prior_checkpoint={"summary_completed_windows": 1,
                                             "summary_parts": [first_part]},
                           checkpoint=checkpoints.append)
        self.assertEqual(checkpoints[-1]["summary_window_plan"], ["transcript", "transcript"])
        self.assertEqual(checkpoints[-1]["summary_evidence_windows"], 0,
                         "没有证据包就没有文档窗")
        # 旧 checkpoint 少了整个计划键时，计数语义与历史一致（继续往后跑）
        self.assertEqual(checkpoints[-1]["summary_completed_windows"], 2)

    def test_proofread_window_retries_transient_bad_json(self):
        """N8-B U5：单窗瞬时坏响应（空 content/坏 JSON）在窗口级重试后痊愈。"""
        source = [
            {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"文本{index}"}
            for index in range(5)
        ]
        responses = ["", "oops not json", json.dumps([
            {"id": "p0", "old": "文本0", "new": "修正0"}
        ])]
        with patch("courselens_worker.llm._chat", side_effect=responses) as chat:
            result = proofread_segments("secret", source, source)
        self.assertEqual(chat.call_count, 3)
        self.assertEqual(result[0]["text"], "修正0")
        self.assertEqual(result[0]["correction"], "applied")

    def test_proofread_window_raises_after_bounded_retries(self):
        """N8-B U5：重试穷尽仍坏→照旧抛 LLMError（G7 降级语义不变）。"""
        source = [
            {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"文本{index}"}
            for index in range(5)
        ]
        with patch("courselens_worker.llm._chat", return_value="persistent-garbage") as chat:
            with self.assertRaises(Exception):
                proofread_segments("secret", source, source)
        self.assertEqual(chat.call_count, 3)



if __name__ == "__main__":
    unittest.main()
