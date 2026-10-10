"""Deterministic, evidence-grounded Lecture IR view (worker side)."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from courselens_worker.lecture_ir import build_lecture_ir
from courselens_worker import llm as courselens_llm
from courselens_worker.runner import _process_materialized_job
from shared.evidence_contract import (
    CONTRACT_ID,
    NAMESPACE_SEGMENT,
    NAMESPACE_SLIDE_ENTITY,
    NAMESPACE_SLIDE_EVENT,
    NAMESPACE_SOURCE,
    NAMESPACE_UNIT,
    compute_id,
    validate_document,
)

_PROVENANCE = {"producer": "test-producer", "model": "test-model", "config_hash": "0123456789abcdef"}
_DECK_ID = "deck-1"
_SLIDE_SHA_PREFIX = "c" * 64
_SOURCE_IDENTITY = {
    "kind": "recording",
    "origin": "external_import",
    "title": None,
    "duration_ms": 600_000,
    "source_sha256": "d" * 64,
}
_DOC_SOURCE_ID = compute_id(NAMESPACE_SOURCE, _SOURCE_IDENTITY)


def stamp_segment_id(start_ms, end_ms, text):
    return compute_id(NAMESPACE_SEGMENT, {
        "source_id": _DOC_SOURCE_ID,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "text": text,
        "lang": None,
        "no_speech": False,
        "producer": _PROVENANCE["producer"],
        "model": _PROVENANCE["model"],
        "config_hash": _PROVENANCE["config_hash"],
    })


def identified_segment(start_ms, end_ms, text):
    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "text": text,
        "segment_id": stamp_segment_id(start_ms, end_ms, text),
    }


def slide_ids(created_sec, content_sha):
    entity = compute_id(NAMESPACE_SLIDE_ENTITY, {
        "source_id": _DOC_SOURCE_ID,
        "deck_id": _DECK_ID,
        "page": 1,
        "content_sha256": content_sha,
    })
    event = compute_id(NAMESPACE_SLIDE_EVENT, {
        "entity": entity,
        "start_ms": created_sec * 1000,
        "end_ms": None,
    })
    return entity, event


def identified_page(created_sec, text="幻灯片", content_sha=None, **extra):
    page = {
        "page_num": created_sec + 1,
        "created_sec": created_sec,
        "text": text,
        "source_sha256": content_sha or _SLIDE_SHA_PREFIX,
    }
    entity, event = slide_ids(page["created_sec"], page["source_sha256"])
    page.update({"entity_id": entity, "event_id": event, "deck_id": _DECK_ID})
    page.update(extra)
    return page


def chapter(title, start_ms, summary="章节概述"):
    return {"title": title, "start_ms": start_ms, "summary": summary}


def empty_view():
    return {"contract": CONTRACT_ID, "sections": [], "knowledge_units": [], "key_moments": []}


class BuilderUnitTests(unittest.TestCase):
    def test_empty_and_legacy_inputs_degrade_to_empty_view(self):
        self.assertEqual(build_lecture_ir(), empty_view())
        self.assertEqual(build_lecture_ir([], [], []), empty_view())
        # Legacy shapes: no contract IDs anywhere, yet anchors exist.
        legacy = build_lecture_ir(
            transcript=[{"start_ms": 0, "end_ms": 2000, "text": "旧文本"}],
            chapters=[chapter("第一章", 0)],
            ppt_pages=[{"page_num": 1, "created_sec": 0, "text": "旧幻灯片"}],
        )
        self.assertEqual(legacy, empty_view())

    def test_section_end_comes_from_next_anchor_then_evidence_end(self):
        transcript = [
            identified_segment(0, 30_000, "第一段"),
            identified_segment(60_000, 120_000, "第二段"),
        ]
        view = build_lecture_ir(
            transcript=transcript,
            chapters=[chapter("第一章", 0), chapter("第二章", 60_000)],
            ppt_pages=[identified_page(30)],
        )
        self.assertEqual(
            [item["time"] for item in view["sections"]],
            [{"start_ms": 0, "end_ms": 60_000}, {"start_ms": 60_000, "end_ms": 120_000}],
        )

    def test_unfounded_or_invalid_chapter_anchors_fail_closed(self):
        transcript = [identified_segment(0, 60_000, "只有一段")]
        view = build_lecture_ir(
            transcript=transcript,
            chapters=[
                chapter("无锚章节", 30_000),
                chapter("负锚", -1),
                chapter("假锚", True),
                chapter("坏锚", "0"),
                chapter("开头", 0),
            ],
            ppt_pages=[],
        )
        self.assertEqual(len(view["sections"]), 1)
        self.assertEqual(view["sections"][0]["title"], "开头")
        self.assertEqual(view["sections"][0]["time"], {"start_ms": 0, "end_ms": 60_000})

    def test_repeated_slide_events_are_preserved_and_duplicates_collapse(self):
        # Same slide content shown twice: two events, one entity.
        pages = [
            identified_page(1, content_sha=_SLIDE_SHA_PREFIX),
            identified_page(40, content_sha=_SLIDE_SHA_PREFIX),
            # A literal duplicate occurrence (identical identity) collapses.
            identified_page(1, content_sha=_SLIDE_SHA_PREFIX),
        ]
        view = build_lecture_ir(ppt_pages=pages)
        self.assertEqual(len(view["key_moments"]), 2)
        entities = {item["content"]["entity_id"] for item in view["key_moments"]}
        self.assertEqual(len(entities), 1)
        self.assertEqual(
            [item["time"]["start_ms"] for item in view["key_moments"]],
            [1_000, 40_000],
        )

    def test_key_moment_intervals_partition_the_evidence_range(self):
        pages = [identified_page(1), identified_page(10), identified_page(20)]
        view = build_lecture_ir(ppt_pages=pages)
        self.assertEqual(
            [item["time"] for item in view["key_moments"]],
            [
                {"start_ms": 1_000, "end_ms": 10_000},
                {"start_ms": 10_000, "end_ms": 20_000},
                {"start_ms": 20_000, "end_ms": 20_000},
            ],
        )

    def test_knowledge_units_split_when_the_active_slide_changes(self):
        transcript = [
            identified_segment(0, 8_000, "第一题"),
            identified_segment(9_000, 15_000, "还是第一题"),
            identified_segment(20_000, 30_000, "第二题"),
        ]
        pages = [identified_page(0, content_sha="a" * 64), identified_page(18, content_sha="b" * 64)]
        view = build_lecture_ir(
            transcript=transcript,
            chapters=[chapter("整课", 0)],
            ppt_pages=pages,
        )
        self.assertEqual(len(view["knowledge_units"]), 2)
        self.assertEqual(view["knowledge_units"][0]["time"], {"start_ms": 0, "end_ms": 15_000})
        self.assertEqual(view["knowledge_units"][1]["time"], {"start_ms": 20_000, "end_ms": 30_000})
        self.assertEqual(
            [len(unit["spans"]) for unit in view["knowledge_units"]], [2, 1]
        )

    def test_segments_without_identifiers_are_never_cited(self):
        transcript = [
            identified_segment(0, 10_000, "有身份"),
            {"start_ms": 10_000, "end_ms": 20_000, "text": "被改写丢失身份"},
        ]
        view = build_lecture_ir(
            transcript=transcript,
            chapters=[chapter("开头", 0)],
            ppt_pages=[],
        )
        section = view["sections"][0]
        self.assertEqual(section["spans"], [{"kind": "segment", "id": transcript[0]["segment_id"]}])
        self.assertEqual(len(view["knowledge_units"]), 1)
        self.assertEqual(len(view["knowledge_units"][0]["spans"]), 1)

    def test_malformed_evidence_identifiers_are_ignored(self):
        transcript = [{
            "start_ms": 0, "end_ms": 1000, "text": "坏身份",
            "segment_id": "seg:ZZZZ",
        }]
        pages = [{
            "page_num": 1, "created_sec": 0, "text": "页",
            "event_id": "slevt:ABCDEF012345", "entity_id": "slent:short",
        }]
        view = build_lecture_ir(transcript=transcript, chapters=[chapter("开头", 0)], ppt_pages=pages)
        self.assertEqual(view, empty_view())

    def test_zero_width_section_at_evidence_end_keeps_anchor_evidence(self):
        pages = [identified_page(60)]
        view = build_lecture_ir(ppt_pages=pages, chapters=[chapter("结尾", 60_000)])
        self.assertEqual(len(view["sections"]), 1)
        self.assertEqual(view["sections"][0]["time"], {"start_ms": 60_000, "end_ms": 60_000})
        self.assertEqual(
            view["sections"][0]["spans"],
            [{"kind": "slide_event", "id": pages[0]["event_id"]}],
        )

    def test_ids_are_stable_under_input_reordering(self):
        transcript = [
            identified_segment(0, 10_000, "甲"),
            identified_segment(10_000, 20_000, "乙"),
            identified_segment(20_000, 30_000, "丙"),
        ]
        pages = [identified_page(0, content_sha="a" * 64), identified_page(5, content_sha="b" * 64)]
        chapters = [chapter("第一章", 0), chapter("第二章", 10_000)]
        ordered = build_lecture_ir(transcript=transcript, chapters=chapters, ppt_pages=pages)
        reordered = build_lecture_ir(
            transcript=list(reversed(transcript)),
            chapters=list(reversed(chapters)),
            ppt_pages=list(reversed(pages)),
        )
        self.assertEqual(ordered, reordered)

    def test_generated_chapter_prose_never_becomes_evidence(self):
        summary = "模型生成的独有句子标记"
        transcript = [identified_segment(0, 60_000, "讲解")]
        view = build_lecture_ir(
            transcript=transcript,
            chapters=[chapter("章节", 0, summary=summary)],
            ppt_pages=[],
        )
        self.assertNotIn(summary, json.dumps(view, ensure_ascii=False))
        evidence_ids = {transcript[0]["segment_id"]}
        for unit in view["sections"] + view["knowledge_units"] + view["key_moments"]:
            self.assertTrue(unit["spans"])
            for span in unit["spans"]:
                self.assertIn(span["id"], evidence_ids)


class ContractConformanceTests(unittest.TestCase):
    def test_emitted_units_validate_inside_an_evidence_v1_document(self):
        transcript = [
            identified_segment(0, 10_000, "甲"),
            identified_segment(10_000, 20_000, "乙"),
        ]
        pages = [identified_page(0, content_sha="a" * 64), identified_page(15, content_sha="b" * 64)]
        view = build_lecture_ir(
            transcript=transcript,
            chapters=[chapter("第一章", 0), chapter("第二章", 15_000)],
            ppt_pages=pages,
        )
        document = {
            "contract": CONTRACT_ID,
            "source": {"id": _DOC_SOURCE_ID, **_SOURCE_IDENTITY},
            "fingerprints": dict(_PROVENANCE),
            "speech": {"segments": [
                {
                    "id": item["segment_id"],
                    "start_ms": item["start_ms"],
                    "end_ms": item["end_ms"],
                    "text": item["text"],
                }
                for item in transcript
            ]},
            "slides": {
                "entities": [
                    {
                        "id": page["entity_id"],
                        "deck_id": _DECK_ID,
                        "page": 1,
                        "content_sha256": page["source_sha256"],
                    }
                    for page in pages
                ],
                "events": [
                    {
                        "id": page["event_id"],
                        "entity": page["entity_id"],
                        "start_ms": page["created_sec"] * 1000,
                        "end_ms": None,
                    }
                    for page in pages
                ],
            },
            "units": (
                view["sections"] + view["knowledge_units"] + view["key_moments"]
            ),
        }
        normalized = validate_document(document)
        self.assertEqual(
            {unit["id"] for unit in normalized["units"]},
            {unit["id"] for unit in document["units"]},
        )
        for unit in normalized["units"]:
            self.assertIn(unit["kind"], {"section", "knowledge_unit", "key_moment"})
            self.assertTrue(unit["spans"])


def summary_job(kind="summary", requested=None, transcript=None, slides=None, prior=None):
    payload = {
        "title": "测试课程",
        "transcript": list(transcript or []),
        "slides": list(slides or []),
    }
    if prior is not None:
        payload["checkpoint"] = dict(prior)
    job = {
        "job_kind": kind,
        "task_id": "task-ir",
        "input_hash": "hash-ir",
        "pipeline": {"version": "test-v2"},
        "payload": payload,
        "secrets": {"deepseek_api_key": "secret"},
    }
    if requested is not None:
        job["requested_outputs"] = list(requested)
    return job


class RunnerSeamTests(unittest.TestCase):
    def run_summary_family(self, job, chapters):
        pages = [identified_page(0, content_sha="a" * 64)]

        def fake_slides(slides_arg, *, progress, prior_checkpoint, checkpoint, **kwargs):
            return pages, {}

        def fake_summary(api_key, *, title, transcript, ppt_pages, prior_checkpoint, checkpoint,
                         evidence_packet=None, course_context=None, usage_sink=None,
                         progress=None):
            # evidence_packet/course_context 是 N7A 的加性参数：显式接住以钉住调用签名。
            # usage_sink 是 RR-ACCOUNT2-1 的加性参数：任务级 token 账本由 runner 汇总。
            return {"model": "deepseek-chat", "markdown": "笔记", "chapters": chapters}

        with patch("courselens_worker.ocr.process_slides", side_effect=fake_slides), \
                patch("courselens_worker.llm.create_summary", side_effect=fake_summary):
            return _process_materialized_job(job), pages

    def test_learning_pack_llm_failure_returns_llm_pending_with_outputs(self):
        """夜10-C 第九波任务2：55fbf220 实锤面——learning_pack 的 LLM 段失败
        同享 llm_pending 降级（OCR/转写产物随回执交还），不再整单失败。"""
        job = summary_job(
            kind="learning_pack",
            requested=["ocr", "summary"],
            transcript=[identified_segment(0, 60_000, "讲解")],
            slides=[{"source": {}}],
        )
        with patch("courselens_worker.ocr.process_slides",
                   side_effect=lambda slides, **kw: ([identified_page(0)], {})),                 patch("courselens_worker.llm.create_summary",
                      side_effect=courselens_llm.LLMError("AI request failed: ConnectionError")):
            result = _process_materialized_job(job)
        self.assertEqual(result["status"], "llm_pending")
        pending = result["outputs"]["llm_pending"]
        self.assertEqual(pending["reason_code"], "llm_remote_failed")
        self.assertEqual(pending["job_kind"], "learning_pack")
        self.assertEqual(len(pending["ppt_pages"]), 1, "已完成 OCR 页随回执交还")
        self.assertEqual(len(pending["transcript"]), 1)
        self.assertIn("llm_pending_remote_failed", result["warnings"])

    def test_summary_llm_failure_returns_llm_pending_receipt(self):
        """夜10-C 第七波②：远端 LLM 段闭集失败 → llm_pending 回执（非 LLM
        产物随回执交还客户端领回），不再整单失败。"""
        job = summary_job(kind="summary", transcript=[identified_segment(0, 60_000, "讲解")])
        with patch("courselens_worker.ocr.process_slides", side_effect=lambda slides, **kw: ([], {})),                 patch("courselens_worker.llm.create_summary",
                      side_effect=courselens_llm.LLMError("AI request failed: ConnectionError")):
            result = _process_materialized_job(job)
        self.assertEqual(result["status"], "llm_pending")
        pending = result["outputs"]["llm_pending"]
        self.assertEqual(pending["title"], "测试课程")
        self.assertEqual(pending["transcript"], [identified_segment(0, 60_000, "讲解")])
        self.assertEqual(pending["reason_code"], "llm_remote_failed")
        self.assertIn("llm_pending_remote_failed", result["warnings"])

    def test_summary_job_attaches_the_additive_view(self):
        transcript = [identified_segment(0, 60_000, "讲解")]
        job = summary_job(kind="summary", transcript=transcript, slides=[{"source": {}}])
        result, pages = self.run_summary_family(job, [chapter("第一章", 0)])
        outputs = result["outputs"]
        self.assertEqual(set(outputs), {"ppt_pages", "summary", "lecture_ir"})
        view = outputs["lecture_ir"]
        self.assertEqual(view["contract"], CONTRACT_ID)
        self.assertEqual(len(view["sections"]), 1)
        self.assertEqual(view["sections"][0]["title"], "第一章")
        self.assertEqual(len(view["key_moments"]), 1)
        # Pre-existing outputs stay byte-identical to the produced values.
        self.assertEqual(outputs["ppt_pages"], pages)
        self.assertEqual(outputs["summary"]["markdown"], "笔记")

    def test_chapters_job_attaches_the_additive_view(self):
        transcript = [identified_segment(0, 60_000, "讲解")]
        job = summary_job(kind="chapters", transcript=transcript, slides=[{"source": {}}])
        result, _ = self.run_summary_family(job, [chapter("第一章", 0)])
        self.assertEqual(set(result["outputs"]), {"ppt_pages", "chapters", "lecture_ir"})
        self.assertEqual(len(result["outputs"]["lecture_ir"]["sections"]), 1)

    def test_learning_pack_attaches_the_view_for_summary_outputs(self):
        transcript = [identified_segment(0, 60_000, "讲解")]
        prior = {"ppt_pages": [identified_page(0, content_sha="a" * 64)], "ppt_skipped": {}}
        job = summary_job(
            kind="learning_pack",
            requested=["summary"],
            transcript=transcript,
            prior=prior,
        )
        result, _ = self.run_summary_family(job, [chapter("第一章", 0)])
        self.assertIn("lecture_ir", result["outputs"])
        self.assertEqual(len(result["outputs"]["lecture_ir"]["key_moments"]), 1)

    def test_learning_pack_without_summary_family_has_no_lecture_ir(self):
        job = summary_job(
            kind="learning_pack",
            requested=["ocr"],
            transcript=[identified_segment(0, 60_000, "讲解")],
            slides=[{"source": {}}],
        )
        result, _ = self.run_summary_family(job, [])
        self.assertNotIn("lecture_ir", result["outputs"])

    def test_subtitle_only_job_has_no_lecture_ir(self):
        job = {
            "job_kind": "subtitle",
            "task_id": "task-ir",
            "input_hash": "hash-ir",
            "pipeline": {"version": "test-v2"},
            "payload": {
                "mode": "automatic",
                "media": {"start_seconds": 0, "duration_seconds": 60},
                "transcript": [],
            },
            "secrets": {"deepseek_api_key": "secret"},
        }
        with patch.dict("os.environ", {"SENSEVOICE_MODEL_DIR": "s", "PARAFORMER_MODEL_DIR": "p"}), \
                patch("courselens_worker.asr.transcribe") as transcribe_mock:
            transcribe_mock.return_value = {
                "mode": "automatic",
                "segments": [],
                "raw_sensevoice": [],
                "raw_paraformer": [],
                "metrics": {},
            }
            result = _process_materialized_job(job)
        self.assertNotIn("lecture_ir", result["outputs"])


class KnowledgePointProjectionTests(unittest.TestCase):
    """N7A：多源知识点投影成 evidence.v1 稳定单元（引用可解析、时间不编造）。"""

    def setUp(self):
        self.segment_id = f"{NAMESPACE_SEGMENT}:0123456789ab"
        self.event_id = f"{NAMESPACE_SLIDE_EVENT}:0123456789ab"
        self.transcript = [{
            "segment_id": self.segment_id, "start_ms": 0, "end_ms": 4000, "text": "动力学",
        }]
        self.pages = [{
            "event_id": self.event_id, "page_num": 1, "created_sec": 2, "text": "第三章",
        }]
        self.packet = {
            "contract": "courselens.course-knowledge.v1",
            "items": [
                {"citation_id": "ckc:000000000001", "kind": "transcript",
                 "source_id": self.segment_id, "revision_id": "a" * 16,
                 "content_hash": "b" * 64, "locator": {"start_ms": 0, "end_ms": 4000},
                 "label": "00:00 动力学"},
                {"citation_id": "ckc:000000000002", "kind": "document_page",
                 "source_id": "doc-x", "revision_id": "c" * 16,
                 "content_hash": "d" * 64, "locator": {"page": 3}, "label": "第3页 讲义"},
            ],
        }

    def test_points_citing_transcript_become_units_with_real_intervals(self):
        point = {"title": "凝胶点", "text": "p_c≈0.8。",
                 "citation_ids": ["ckc:000000000001", "ckc:000000000002"]}
        view = build_lecture_ir(transcript=self.transcript, chapters=[], ppt_pages=self.pages,
                                knowledge_points=[point], evidence_packet=self.packet)
        self.assertEqual(view["knowledge_points_projected"], 1)
        self.assertEqual(view["knowledge_points_skipped"], 0)
        projections = [unit for unit in view["knowledge_units"] if unit["content"]
                       and unit["content"].get("text") == "p_c≈0.8。"]
        self.assertEqual(len(projections), 1)
        unit = projections[0]
        self.assertEqual(unit["kind"], "knowledge_unit")
        self.assertEqual(unit["time"], {"start_ms": 0, "end_ms": 4000},
                         "区间取自被引字幕，不自造")
        self.assertEqual(unit["spans"], [{"kind": "segment", "id": self.segment_id}])
        self.assertEqual(unit["title"], "凝胶点")
        # 文档页没有时间锚，走 contract evidence_refs 而不是 span
        refs = unit["content"]["evidence_refs"]
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["citation_id"], "ckc:000000000002")
        self.assertEqual(refs[0]["locator"], {"page": 3})

    def test_points_without_any_citable_span_are_skipped_and_counted(self):
        point = {"title": "仅文档", "text": "讲义第3页写了。",
                 "citation_ids": ["ckc:000000000002"]}
        view = build_lecture_ir(transcript=self.transcript, chapters=[], ppt_pages=self.pages,
                                knowledge_points=[point], evidence_packet=self.packet)
        self.assertEqual(view["knowledge_points_projected"], 0)
        self.assertEqual(view["knowledge_points_skipped"], 1,
                         "不编造时间轴：没有可解析引用就不出单元")

    def test_slide_citations_resolve_through_the_event_window(self):
        packet = {
            "contract": "courselens.course-knowledge.v1",
            "items": [{"citation_id": "ckc:000000000003", "kind": "slide",
                       "source_id": self.event_id, "revision_id": "e" * 16,
                       "content_hash": "f" * 64, "locator": {"page": 1}, "label": "第1页"}],
        }
        view = build_lecture_ir(
            transcript=self.transcript, chapters=[], ppt_pages=self.pages,
            knowledge_points=[{"title": "幻灯", "text": "第三章。",
                               "citation_ids": ["ckc:000000000003"]}],
            evidence_packet=packet)
        self.assertEqual(view["knowledge_points_projected"], 1)
        unit = next(unit for unit in view["knowledge_units"]
                    if unit["spans"] and unit["spans"][0]["kind"] == "slide_event")
        # 幻灯事件的时间窗与该 key_moment 一致：到下一个事件或证据末尾为止，
        # 是既有推导规则，不是为知识点另造一套时间。
        moment = next(moment for moment in view["key_moments"]
                      if moment["spans"][0]["id"] == self.event_id)
        self.assertEqual(unit["time"], moment["time"])
        self.assertEqual(unit["time"], {"start_ms": 2000, "end_ms": 4000})

    def test_unknown_citation_is_dropped_without_breaking_the_view(self):
        view = build_lecture_ir(
            transcript=self.transcript, chapters=[], ppt_pages=self.pages,
            knowledge_points=[{"title": "悬空", "text": "引用了包外。",
                               "citation_ids": ["ckc:ffffffffffff"]}],
            evidence_packet=self.packet)
        self.assertEqual(view["knowledge_points_projected"], 0)
        self.assertEqual(view["knowledge_points_skipped"], 1)

    def test_projection_is_stable_across_input_order(self):
        points = [
            {"title": "乙", "text": "第二次讲到的。", "citation_ids": ["ckc:000000000001"]},
            {"title": "甲", "text": "先讲到的。", "citation_ids": ["ckc:000000000001"]},
        ]
        forward = build_lecture_ir(transcript=self.transcript, ppt_pages=self.pages,
                                   knowledge_points=points, evidence_packet=self.packet)
        backward = build_lecture_ir(transcript=self.transcript, ppt_pages=self.pages,
                                    knowledge_points=list(reversed(points)),
                                    evidence_packet=self.packet)
        self.assertEqual(
            sorted(unit["id"] for unit in forward["knowledge_units"]),
            sorted(unit["id"] for unit in backward["knowledge_units"]),
            "单元身份与输入顺序无关",
        )

    def test_view_stays_four_keys_without_knowledge_points(self):
        view = build_lecture_ir(transcript=self.transcript, chapters=[], ppt_pages=self.pages)
        self.assertEqual(sorted(view), ["contract", "key_moments", "knowledge_units", "sections"],
                         "不传知识点的旧调用拿到的是与历史逐键相同的视图")


if __name__ == "__main__":
    unittest.main()
