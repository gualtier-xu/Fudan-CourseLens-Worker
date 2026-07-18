import json
import unittest
from unittest.mock import patch

from courselens_worker.llm import create_summary, proofread_segments


class LLMCheckpointTests(unittest.TestCase):
    def test_proofread_resumes_after_completed_window(self):
        source = [
            {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"raw-{index}"}
            for index in range(25)
        ]
        prior_segments = [
            {"start_ms": index * 1000, "end_ms": (index + 1) * 1000, "text": f"done-{index}"}
            for index in range(20)
        ]
        response = json.dumps([
            {"index": index, "text": f"fixed-{index}"} for index in range(20, 25)
        ])
        checkpoints = []
        with patch("courselens_worker.llm._chat", return_value=response) as chat:
            result = proofread_segments(
                "secret",
                source,
                source,
                prior_checkpoint={
                    "proofread_completed_windows": 1,
                    "proofread_segments": prior_segments,
                },
                checkpoint=checkpoints.append,
            )
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(len(result), 25)
        self.assertEqual(result[-1]["text"], "fixed-24")
        self.assertEqual(checkpoints[-1]["proofread_completed_windows"], 2)

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
        with patch(
            "courselens_worker.llm._chat",
            side_effect=[json.dumps(second_part), json.dumps(final)],
        ) as chat:
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


if __name__ == "__main__":
    unittest.main()
