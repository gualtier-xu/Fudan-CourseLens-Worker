"""Time-local OCR terminology context for Standard-mode subtitle correction."""

import json
import os
import unittest
from unittest.mock import patch

from courselens_worker.llm import PROOFREAD_PAIRING, proofread_segments
from courselens_worker.runner import _process_materialized_job


ENV = {"SENSEVOICE_MODEL_DIR": "sensevoice", "PARAFORMER_MODEL_DIR": "paraformer"}


def seg(start_ms, end_ms, text):
    return {"start_ms": start_ms, "end_ms": end_ms, "text": text}


def transcribe_value(mode="automatic", segments=None):
    return {
        "mode": mode,
        "segments": list(segments if segments is not None else [seg(0, 2000, "这个决策叔算法很经典")]),
        "raw_sensevoice": [],
        "raw_paraformer": [],
        "metrics": {},
    }


def recording_transcribe(value, calls):
    def fake(job, *, checkpoint=None, **kwargs):
        if checkpoint is not None:
            checkpoint({
                "completed_chunks": 1,
                "total_chunks": 1,
                "mode": "automatic",
                "raw_sensevoice": [],
                "raw_paraformer": [],
            })
        calls.append("asr")
        return value

    return fake


def pack_job(requested, *, proofread=True, slides=None, prior=None, kind="learning_pack"):
    # automatic 策略：DeepSeek Key 在 secrets 中存在与否决定回退行为
    return {
        "job_kind": kind,
        "task_id": "task-1",
        "input_hash": "hash-1",
        "pipeline": {"version": "test-v2"},
        "requested_outputs": list(requested),
        "payload": {
            "mode": "automatic",
            "media": {"start_seconds": 0, "duration_seconds": 60},
            "transcript": [],
            "slides": list(slides or []),
            **({"checkpoint": dict(prior)} if prior is not None else {}),
        },
        "secrets": {"deepseek_api_key": "secret" if proofread else ""},
    }


def run_pack(job, checkpoints):
    """Run one materialized job with faked transcribe/OCR/model seams.

    Returns (result, events, wire_payloads).  ``events`` records the stage
    order plus every proofread-checkpoint value merged by the runner.
    """
    events = []
    wire_payloads = []

    def fake_transcribe(job, *, proofread, checkpoint, **kwargs):
        events.append("asr")
        if checkpoint is not None:
            checkpoint({
                "completed_chunks": 1,
                "total_chunks": 1,
                "mode": "automatic",
                "raw_sensevoice": [],
                "raw_paraformer": [],
            })
        if proofread is None:
            return transcribe_value("automatic")
        segments = proofread(
            [seg(0, 2000, "这个决策叔算法很经典")],
            [seg(0, 2000, "这个决策叔算法很经典")],
            dict(job["payload"].get("checkpoint") or {}),
            lambda value: events.append(("proofread_checkpoint", dict(value))),
        )
        return transcribe_value("automatic", segments)

    def fake_chat(api_key, messages, **kwargs):
        events.append("chat")
        wire_payloads.append(json.loads(messages[1]["content"]))
        return json.dumps([{"id": "p0", "old": "决策叔", "new": "决策树"}])

    def fake_slides(slides_arg, *, progress, prior_checkpoint, checkpoint, **kwargs):
        events.append("ocr")
        pages = [
            {"page_num": index + 1, "created_sec": index, "text": f"幻灯片{index}术语"}
            for index in range(len(slides_arg))
        ]
        checkpoint({
            "stage": "ocr",
            "completed_chunks": len(pages),
            "total_chunks": len(pages),
            "ocr_completed_items": len(pages),
            "ppt_pages": pages,
            "ppt_skipped": {},
        })
        return pages, {}

    with patch.dict(os.environ, ENV), \
            patch("courselens_worker.asr.transcribe", side_effect=fake_transcribe), \
            patch("courselens_worker.ocr.process_slides", side_effect=fake_slides), \
            patch("courselens_worker.llm._chat", side_effect=fake_chat):
        result = _process_materialized_job(job, checkpoint_writer=checkpoints.append)
    return result, events, wire_payloads


class RunnerOrderTests(unittest.TestCase):
    def test_ocr_runs_once_before_standard_proofread_and_feeds_slide_context(self):
        checkpoints = []
        job = pack_job(
            ["subtitle", "ocr"],
            slides=[{"page_num": 1, "created_sec": 1, "source": {}}],
        )
        result, events, payloads = run_pack(job, checkpoints)
        self.assertEqual(events[:2], ["ocr", "asr"])
        self.assertEqual(events.count("ocr"), 1)
        # SUBTITLE-DEEP-1 v3：词级校对后术语/深校对层无条件跟进——第二次 chat
        # 属 term 窗口（id 空间 t0..），复用的 p0 提案在那里被忽略。
        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0][0]["slide"], "幻灯片0术语")
        self.assertEqual(result["outputs"]["subtitle"]["segments"][0]["text"], "这个决策树算法很经典")
        self.assertEqual(result["outputs"]["ppt_pages"][0]["text"], "幻灯片0术语")

    def test_subtitle_stage_checkpoints_retain_ocr_fields(self):
        checkpoints = []
        job = pack_job(
            ["subtitle", "ocr"],
            slides=[{"page_num": 1, "created_sec": 1, "source": {}}],
        )
        _, events, _ = run_pack(job, checkpoints)
        asr_checkpoint = checkpoints[0]
        self.assertEqual(asr_checkpoint["completed_chunks"], 1)
        self.assertEqual(asr_checkpoint["ocr_completed_items"], 1)
        self.assertEqual(asr_checkpoint["ppt_pages"][0]["text"], "幻灯片0术语")
        self.assertEqual(asr_checkpoint["ppt_skipped"], {})
        proofread_checkpoint = next(
            value
            for item in events
            if isinstance(item, tuple) and item[0] == "proofread_checkpoint"
            for value in [item[1]]
        )
        self.assertEqual(proofread_checkpoint["proofread_pairing"], PROOFREAD_PAIRING)
        self.assertEqual(proofread_checkpoint["ppt_pages"][0]["text"], "幻灯片0术语")
        self.assertEqual(proofread_checkpoint["ocr_completed_items"], 1)

    def test_ocr_checkpoint_retains_prior_subtitle_state(self):
        prior = {
            "completed_chunks": 3,
            "total_chunks": 9,
            "mode": "automatic",
            "raw_sensevoice": [seg(0, 500, "旧参考")],
            "raw_paraformer": [seg(0, 500, "旧主文本")],
            "pcm_fingerprint": "a" * 64,
        }
        checkpoints = []
        job = pack_job(
            ["subtitle", "ocr"],
            slides=[{"page_num": 1, "created_sec": 0, "source": {}}],
            prior=prior,
        )
        _, _, _ = run_pack(job, checkpoints)
        ocr_checkpoint = checkpoints[0]
        self.assertEqual(ocr_checkpoint["stage"], "ocr")
        self.assertEqual(ocr_checkpoint["ocr_completed_items"], 1)
        # The carried subtitle progress wins the shared chunk counters and
        # keeps the raw segments resumable without duplication.
        self.assertEqual(ocr_checkpoint["completed_chunks"], 3)
        self.assertEqual(ocr_checkpoint["total_chunks"], 9)
        self.assertEqual(ocr_checkpoint["mode"], "automatic")
        self.assertEqual(ocr_checkpoint["pcm_fingerprint"], prior["pcm_fingerprint"])
        self.assertEqual(ocr_checkpoint["raw_paraformer"], prior["raw_paraformer"])

    def test_completed_slides_are_not_reprocessed_on_resume(self):
        prior = {
            "ocr_completed_items": 1,
            "ppt_pages": [{"page_num": 1, "created_sec": 0, "text": "幻灯片术语"}],
            "ppt_skipped": {},
        }
        checkpoints = []
        calls = []
        job = pack_job(
            ["subtitle", "ocr"],
            slides=[{"page_num": 1, "created_sec": 0, "source": {}}],
            prior=prior,
        )
        with patch.dict(os.environ, ENV), \
                patch(
                    "courselens_worker.asr.transcribe",
                    side_effect=recording_transcribe(transcribe_value(), calls),
                ), \
                patch("courselens_worker.ocr._fetch_page") as fetch, \
                patch("courselens_worker.llm._chat", return_value="[]"):
            result = _process_materialized_job(job, checkpoint_writer=checkpoints.append)
        fetch.assert_not_called()
        self.assertEqual(result["outputs"]["ppt_pages"], prior["ppt_pages"])
        self.assertEqual(checkpoints[0]["ppt_pages"], prior["ppt_pages"])
        self.assertEqual(checkpoints[0]["ocr_completed_items"], 1)

    def test_summary_checkpoints_and_pages_survive_reorder(self):
        checkpoints = []
        job = pack_job(
            ["subtitle", "summary", "ocr"],
            slides=[{"page_num": 1, "created_sec": 0, "source": {}}],
        )

        def fake_summary(api_key, *, title, transcript, ppt_pages, prior_checkpoint, checkpoint,
                         usage_sink=None, progress=None):
            checkpoint({
                "stage": "summary",
                "completed_chunks": 1,
                "total_chunks": 2,
                "summary_completed_windows": 1,
                "summary_parts": [{"markdown": "部分", "chapters": []}],
            })
            return {"model": "deepseek-chat", "markdown": "笔记", "chapters": []}

        with patch.dict(os.environ, ENV), \
                patch(
                    "courselens_worker.asr.transcribe",
                    return_value=transcribe_value(),
                ), \
                patch("courselens_worker.ocr.process_slides", return_value=(
                    [{"page_num": 1, "created_sec": 0, "text": "幻灯片术语"}], {}
                )) as slides_mock, \
                patch("courselens_worker.llm.create_summary", side_effect=fake_summary):
            result = _process_materialized_job(job, checkpoint_writer=checkpoints.append)
        self.assertEqual(slides_mock.call_count, 1)
        self.assertEqual(result["outputs"]["summary"]["markdown"], "笔记")
        self.assertEqual(result["outputs"]["ppt_pages"][0]["text"], "幻灯片术语")
        summary_checkpoint = next(
            value for value in checkpoints if value.get("stage") == "summary"
        )
        self.assertEqual(summary_checkpoint["ppt_pages"][0]["text"], "幻灯片术语")
        self.assertEqual(summary_checkpoint["ocr_completed_items"], 1)

    def test_standalone_subtitle_keeps_checkpoints_without_ocr_fields(self):
        checkpoints = []
        job = pack_job(["subtitle"], kind="subtitle")
        with patch.dict(os.environ, ENV), \
                patch(
                    "courselens_worker.asr.transcribe",
                    return_value=transcribe_value(),
                ) as transcribe_mock, \
                patch("courselens_worker.llm._chat", return_value="[]"):
            result = _process_materialized_job(job, checkpoint_writer=checkpoints.append)
        transcribe_mock.assert_called_once()
        self.assertEqual(result["outputs"]["subtitle"]["mode"], "automatic")
        # SUBTITLE-DEEP-1 v3：深校对层无条件跟进，term 检查点替代「零检查点」
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]["stage"], "term_proofread")
        self.assertNotIn("ppt_pages", checkpoints[0])
        self.assertNotIn("ocr_completed_items", checkpoints[0])
        self.assertNotIn("ppt_pages", result["outputs"])

    def test_standalone_subtitle_checkpoints_stay_unwrapped(self):
        checkpoints = []
        calls = []
        job = pack_job(["subtitle"], kind="subtitle")
        with patch.dict(os.environ, ENV), \
                patch(
                    "courselens_worker.asr.transcribe",
                    side_effect=recording_transcribe(transcribe_value(), calls),
                ), \
                patch("courselens_worker.llm._chat", return_value="[]"):
            _process_materialized_job(job, checkpoint_writer=checkpoints.append)
        # SUBTITLE-DEEP-1 v3：asr 块检查点之后多一条 term 深校对检查点，
        # 两者都不携带 OCR 字段（unwrapped 语义保持）。
        self.assertEqual(len(checkpoints), 2)
        self.assertEqual(checkpoints[0]["completed_chunks"], 1)
        self.assertEqual(checkpoints[1]["stage"], "term_proofread")
        for checkpoint in checkpoints:
            self.assertNotIn("ppt_pages", checkpoint)
            self.assertNotIn("ocr_completed_items", checkpoint)

    def test_fallback_proofread_path_runs_ocr_once_without_proofread_context(self):
        checkpoints = []
        job = pack_job(
            ["subtitle", "ocr"],
            proofread=False,
            slides=[{"page_num": 1, "created_sec": 0, "source": {}}],
        )
        result, events, payloads = run_pack(job, checkpoints)
        self.assertEqual(events[:2], ["ocr", "asr"])
        self.assertEqual(payloads, [])
        self.assertEqual(result["outputs"]["subtitle"]["mode"], "automatic")
        self.assertEqual(checkpoints[0]["ocr_completed_items"], 1)


class SlideContextWireTests(unittest.TestCase):
    def run_proofread(self, primary, sensevoice, pages, response="[]"):
        payloads = []

        def fake_chat(api_key, messages, **kwargs):
            payloads.append(json.loads(messages[1]["content"]))
            return response

        with patch("courselens_worker.llm._chat", side_effect=fake_chat):
            result = proofread_segments("secret", sensevoice, primary, ppt_pages=pages)
        return result, payloads

    def test_active_slide_not_a_future_slide(self):
        pages = [
            {"created_sec": 30, "text": "未来术语"},
            {"created_sec": 2, "text": "当前术语"},
        ]
        _, payloads = self.run_proofread([seg(0, 4000, "第一句")], [seg(0, 4000, "参考")], pages)
        self.assertEqual(payloads[0][0]["slide"], "当前术语")

    def test_latest_slide_at_or_before_midpoint_wins(self):
        pages = [
            {"created_sec": 1, "text": "旧术语"},
            {"created_sec": 3, "text": "新术语"},
        ]
        _, payloads = self.run_proofread([seg(0, 8000, "第一句")], [seg(0, 8000, "参考")], pages)
        self.assertEqual(payloads[0][0]["slide"], "新术语")

    def test_slide_context_is_bounded_and_normalized(self):
        pages = [{"created_sec": 0, "text": "术  语\n" + "长" * 400}]
        _, payloads = self.run_proofread([seg(0, 2000, "第一句")], [seg(0, 2000, "参考")], pages)
        slide = payloads[0][0]["slide"]
        self.assertEqual(len(slide), 200)
        self.assertNotIn("\n", slide)
        self.assertNotIn("  ", slide)

    def test_no_usable_context_omits_the_field(self):
        cases = [
            None,
            [],
            [{"created_sec": 999, "text": "未来术语"}],
            [{"created_sec": 0, "text": "   "}],
            ["junk"],
        ]
        for pages in cases:
            _, payloads = self.run_proofread([seg(0, 2000, "第一句")], [seg(0, 2000, "参考")], pages)
            self.assertEqual(
                payloads[0][0],
                {"id": "p0", "start_ms": 0, "end_ms": 2000, "text": "第一句", "alt": "参考"},
            )

    def test_slide_terminology_replacement_passes_existing_validators(self):
        pages = [{"created_sec": 0, "text": "第3章 决策树算法"}]
        response = json.dumps([{"id": "p0", "old": "决策叔", "new": "决策树"}])
        result, _ = self.run_proofread(
            [seg(0, 2000, "这个决策叔算法很经典")],
            [seg(0, 2000, "这个决策叔算法很经典")],
            pages,
            response,
        )
        self.assertEqual(result[0]["text"], "这个决策树算法很经典")
        self.assertEqual(result[0]["correction"], "applied")

    def test_slide_alone_cannot_authorize_a_protected_change(self):
        pages = [{"created_sec": 0, "text": "温度是26摄氏度"}]
        response = json.dumps([{"id": "p0", "old": "25", "new": "26"}])
        result, _ = self.run_proofread(
            [seg(0, 2000, "温度是25摄氏度")],
            [seg(0, 2000, "温度是25摄氏度")],
            pages,
            response,
        )
        self.assertEqual(result[0]["text"], "温度是25摄氏度")
        self.assertEqual(result[0]["correction"], "rejected-protected")


if __name__ == "__main__":
    unittest.main()
