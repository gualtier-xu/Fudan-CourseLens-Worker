from __future__ import annotations

import hashlib
import sys
import types
import unittest
from unittest.mock import patch

# The private Windows CI installs the real Worker OCR stack; lighter runtimes
# (the public mirror unit job) only provide the signing dependencies, so these
# contracts skip there instead of failing to import.
try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

# numpy and rapidocr are imported by the OCR module itself; inert stand-ins
# keep the import contract testable wherever only Pillow is present.
try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    _numpy = types.ModuleType("numpy")
    _numpy.asarray = lambda value: value
    sys.modules.setdefault("numpy", _numpy)
try:
    import rapidocr_onnxruntime  # noqa: F401
except ModuleNotFoundError:
    _rapid = types.ModuleType("rapidocr_onnxruntime")
    _rapid.RapidOCR = object
    sys.modules.setdefault("rapidocr_onnxruntime", _rapid)

from courselens_worker.source import SourceSecurityError  # noqa: E402

if Image is not None:
    import io

    from PIL import ImageDraw

    def _png_bytes(color: str) -> bytes:
        # V2 四级流水（NIGHT5-U4）后夹具必须带内容：纯色帧会被无特征规则
        # 当作空白页在 OCR 前落刀。底色保留 color 语义，叠加一块对比色版面。
        buffer = io.BytesIO()
        image = Image.new("RGB", (320, 180), color)
        draw = ImageDraw.Draw(image)
        draw.rectangle([20, 20, 220, 70], fill=(20, 90, 160) if color == "white" else (240, 200, 60))
        draw.rectangle([20, 90, 260, 104], fill=(60, 62, 66))
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    PNG = _png_bytes("white")
    PNG_ALT = _png_bytes("black")
    HTML = b"<!doctype html><html><body>authorization expired</body></html>"
else:
    PNG = b""
    PNG_ALT = b""
    HTML = b"<!doctype html><html><body>authorization expired</body></html>"

DECK = {"deck_id": "deck-a1b2c3d4e5f6", "source_id": "src:0123456789ab"}


def _slides(count: int) -> list[dict]:
    return [
        {"page_num": index + 1, "created_sec": 0, "source": {"url": f"https://example.invalid/{index}"}}
        for index in range(count)
    ]


@unittest.skipIf(Image is None, "Pillow runtime required for slide OCR tests")
class SlideOcrToleranceTests(unittest.TestCase):
    def _run(self, bodies: list, *, prior: dict | None = None, deck: dict | None = None):
        from courselens_worker.ocr import process_slides

        responses = {f"https://example.invalid/{index}": body for index, body in enumerate(bodies)}
        def fake_fetch(source):
            value = responses[str(source.get("url"))]
            if isinstance(value, Exception):
                raise value
            return value

        slides = [
            {
                "page_num": index + 1,
                "created_sec": index * 10,
                "source": {"url": f"https://example.invalid/{index}"},
                **({"deck": deck} if deck else {}),
            }
            for index in range(len(bodies))
        ]
        with patch("courselens_worker.ocr.fetch_bytes", side_effect=fake_fetch), \
             patch("courselens_worker.ocr._dhash", return_value="ab01ef01ab01ef01"), \
             patch("courselens_worker.ocr._engine", return_value=lambda image: ([["box", "text"]], 0.1)):
            pages, skipped = process_slides(
                slides,
                progress=lambda *_args: None,
                prior_checkpoint=prior,
            )
        return pages, skipped

    def test_non_image_slide_is_skipped_without_failing_the_batch(self):
        pages, skipped = self._run([PNG, HTML])
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["text"], "text")
        self.assertEqual(skipped, {"html_body": 1})

    def test_all_slides_unavailable_still_returns_without_raising(self):
        pages, skipped = self._run([HTML, HTML])
        self.assertEqual(pages, [])
        self.assertEqual(skipped, {"html_body": 2})

    def test_empty_body_is_classified_as_empty(self):
        pages, skipped = self._run([b"", PNG])
        self.assertEqual(len(pages), 1)
        self.assertEqual(skipped, {"empty": 1})

    def test_bom_and_whitespace_prefixed_html_is_classified(self):
        pages, skipped = self._run([b"\xef\xbb\xbf  \r\n" + HTML, b"  \t" + HTML])
        self.assertEqual(pages, [])
        self.assertEqual(skipped, {"html_body": 2})

    def test_json_bodies_are_classified_without_guessing_upstream_errors(self):
        pages, skipped = self._run([b'{"error": "synthetic"}', b'[{"status": "synthetic"}]'])
        self.assertEqual(pages, [])
        self.assertEqual(skipped, {"json_body": 2})

    def test_unsupported_binary_stays_unidentified(self):
        pages, skipped = self._run([b"\x00\xff\xfe\x10\x20" * 6])
        self.assertEqual(pages, [])
        self.assertEqual(skipped, {"unidentified_image": 1})

    def test_text_bodies_never_reach_the_ocr_engine(self):
        from courselens_worker.ocr import process_slides

        with patch("courselens_worker.ocr.fetch_bytes", side_effect=lambda source: HTML), \
             patch(
                 "courselens_worker.ocr._engine",
                 side_effect=AssertionError("engine must not run on text bodies"),
             ):
            pages, skipped = process_slides(
                _slides(1), progress=lambda *_args: None
            )
        self.assertEqual(pages, [])
        self.assertEqual(skipped, {"html_body": 1})

    def _run_sources(self, slides: list[dict], fetch_handler):
        from courselens_worker.ocr import process_slides

        with patch("courselens_worker.ocr.fetch_bytes", side_effect=fetch_handler), \
             patch("courselens_worker.ocr._dhash", return_value="ab01ef01ab01ef01"), \
             patch("courselens_worker.ocr._engine", return_value=lambda image: ([["box", "text"]], 0.1)):
            pages, skipped = process_slides(slides, progress=lambda *_args: None)
        return pages, skipped

    def test_alternate_route_rescues_a_connection_failure(self):
        def handler(source):
            if "webvpn" in str(source.get("url")):
                return PNG
            raise SourceSecurityError("source request failed: OSError")

        slides = [{"page_num": 1, "created_sec": 0, "source": {
            "url": "https://media.example.edu/slide.jpg", "headers": {},
            "_alternate_source": {"url": "https://webvpn.fudan.edu.cn/https/slide.jpg", "headers": {}},
        }}]
        pages, skipped = self._run_sources(slides, handler)
        self.assertEqual(len(pages), 1)
        self.assertEqual(skipped, {})

    def test_alternate_route_rescues_an_html_body(self):
        def handler(source):
            if "webvpn" in str(source.get("url")):
                return PNG
            return HTML

        slides = [{"page_num": 1, "created_sec": 0, "source": {
            "url": "https://media.example.edu/slide.jpg", "headers": {},
            "_alternate_source": {"url": "https://webvpn.fudan.edu.cn/https/slide.jpg", "headers": {}},
        }}]
        pages, skipped = self._run_sources(slides, handler)
        self.assertEqual(len(pages), 1)
        self.assertEqual(skipped, {})

    def test_double_failure_keeps_the_primary_closed_set_reason(self):
        def handler(source):
            raise SourceSecurityError("source image returned HTTP 403")

        slides = [{"page_num": 1, "created_sec": 0, "source": {
            "url": "https://media.example.edu/slide.jpg", "headers": {},
            "_alternate_source": {"url": "https://webvpn.fudan.edu.cn/https/slide.jpg", "headers": {}},
        }}]
        pages, skipped = self._run_sources(slides, handler)
        self.assertEqual(pages, [])
        self.assertEqual(skipped, {"image_http_error": 1})

    def test_alternate_text_body_still_skips_as_text(self):
        def handler(source):
            return HTML

        slides = [{"page_num": 1, "created_sec": 0, "source": {
            "url": "https://media.example.edu/slide.jpg", "headers": {},
            "_alternate_source": {"url": "https://webvpn.fudan.edu.cn/https/slide.jpg", "headers": {}},
        }}]
        pages, skipped = self._run_sources(slides, handler)
        self.assertEqual(pages, [])
        self.assertEqual(skipped, {"html_body": 1})

    def test_fetch_failure_maps_to_closed_set_reason(self):
        pages, skipped = self._run([
            SourceSecurityError("source image returned HTTP 403"),
        ])
        self.assertEqual(pages, [])
        self.assertEqual(skipped, {"image_http_error": 1})

    def test_repeated_content_is_one_entity_with_separate_events(self):
        pages, skipped = self._run([PNG, PNG], deck=DECK)
        self.assertEqual(len(pages), 2)
        self.assertEqual(skipped, {})
        self.assertEqual(pages[0]["source_sha256"], pages[1]["source_sha256"])
        self.assertEqual(pages[0]["entity_id"], pages[1]["entity_id"])
        self.assertNotEqual(pages[0]["event_id"], pages[1]["event_id"])
        self.assertEqual([page["page_num"] for page in pages], [1, 2])
        for page in pages:
            self.assertRegex(page["entity_id"], r"^slent:[0-9a-f]{12}$")
            self.assertRegex(page["event_id"], r"^slevt:[0-9a-f]{12}$")
            self.assertEqual(page["deck_id"], DECK["deck_id"])
            self.assertEqual(page["dhash"], "ab01ef01ab01ef01")

    def test_identical_dhash_with_distinct_content_is_not_merged(self):
        pages, skipped = self._run([PNG, PNG_ALT], deck=DECK)
        self.assertEqual(len(pages), 2)
        self.assertEqual(skipped, {})
        self.assertNotEqual(pages[0]["source_sha256"], pages[1]["source_sha256"])
        self.assertNotEqual(pages[0]["entity_id"], pages[1]["entity_id"])

    def test_identity_is_deterministic_across_runs(self):
        first = self._run([PNG, PNG_ALT], deck=DECK)[0]
        second = self._run([PNG, PNG_ALT], deck=DECK)[0]
        self.assertEqual(
            [(page["entity_id"], page["event_id"]) for page in first],
            [(page["entity_id"], page["event_id"]) for page in second],
        )

    def test_unscoped_slides_carry_no_identity(self):
        pages, skipped = self._run([PNG, PNG])
        self.assertEqual(len(pages), 2)
        self.assertEqual(skipped, {})
        for page in pages:
            self.assertNotIn("entity_id", page)
            self.assertNotIn("event_id", page)
            self.assertNotIn("deck_id", page)

    def test_checkpoint_resume_keeps_events_without_duplicates(self):
        expected, _ = self._run([PNG, PNG_ALT], deck=DECK)
        prior = {
            "ocr_completed_items": 1,
            "ppt_pages": [dict(expected[0])],
            "ppt_skipped": {},
        }
        pages, skipped = self._run([PNG, PNG_ALT], deck=DECK, prior=prior)
        self.assertEqual(pages, expected)
        self.assertEqual(skipped, {})
        self.assertEqual([page["page_num"] for page in pages], [1, 2])

    def test_legacy_checkpoint_rows_gain_derivable_identity(self):
        prior = {
            "ocr_completed_items": 1,
            "ppt_pages": [{
                "page_num": 1,
                "created_sec": 0,
                "text": "text",
                "dhash": "ab01ef01ab01ef01",
                "source_sha256": hashlib.sha256(PNG).hexdigest(),
            }],
            "ppt_skipped": {},
        }
        pages, _ = self._run([PNG, PNG_ALT], deck=DECK, prior=prior)
        self.assertIn("entity_id", pages[0])
        self.assertIn("event_id", pages[0])
        fresh, _ = self._run([PNG, PNG_ALT], deck=DECK)
        self.assertEqual(pages[0]["entity_id"], fresh[0]["entity_id"])
        self.assertEqual(pages[0]["event_id"], fresh[0]["event_id"])

    def test_legacy_rows_without_content_digest_stay_unidentified(self):
        prior = {
            "ocr_completed_items": 1,
            "ppt_pages": [{"page_num": 1, "created_sec": 0, "text": "text", "dhash": "x"}],
            "ppt_skipped": {},
        }
        pages, _ = self._run([PNG, PNG_ALT], deck=DECK, prior=prior)
        self.assertNotIn("entity_id", pages[0])
        self.assertIn("entity_id", pages[1])

    def test_skip_counts_survive_checkpoint_resume(self):
        prior = {
            "ocr_completed_items": 1,
            "ppt_pages": [],
            "ppt_skipped": {"unidentified_image": 1, "html_body": 1},
        }
        pages, skipped = self._run([HTML, HTML], prior=prior)
        self.assertEqual(pages, [])
        self.assertEqual(skipped, {"unidentified_image": 1, "html_body": 2})

    def test_checkpoint_payload_carries_skip_counts(self):
        from courselens_worker.ocr import process_slides

        checkpoints: list[dict] = []
        with patch(
            "courselens_worker.ocr.fetch_bytes",
            side_effect=lambda source: HTML,
        ), patch("courselens_worker.ocr._dhash", return_value="ab01ef01ab01ef01"):
            process_slides(
                _slides(1),
                progress=lambda *_args: None,
                checkpoint=checkpoints.append,
            )
        self.assertTrue(checkpoints)
        self.assertEqual(checkpoints[-1]["ppt_skipped"], {"html_body": 1})


class RunnerSlideWarningTests(unittest.TestCase):
    """Runner wiring is importable and testable without the OCR stack."""

    def _summary_job(self, slides: list) -> dict:
        return {
            "task_id": "0123456789abcdef0123456789abcdef",
            "job_kind": "summary",
            "input_hash": "a" * 64,
            "pipeline": {"version": "v2"},
            "payload": {"title": "", "transcript": ["t"], "slides": slides},
            "secrets": {},
        }

    def _run_summary(self, job: dict, slides_result):
        from unittest.mock import Mock

        from courselens_worker.runner import _process_materialized_job

        ocr_stub = types.ModuleType("courselens_worker.ocr")
        ocr_stub.process_slides = Mock(return_value=slides_result)
        llm_stub = types.ModuleType("courselens_worker.llm")
        llm_stub.create_summary = Mock(
            return_value={"markdown": "m", "chapters": [], "model": "deepseek-chat"}
        )
        # 夜10-C 第七波②：runner 现从 llm 模块同源导入 LLMError（llm_pending 面）
        llm_stub.LLMError = type("LLMError", (Exception,), {})
        with patch.dict(
            sys.modules,
            {"courselens_worker.ocr": ocr_stub, "courselens_worker.llm": llm_stub},
        ):
            result = _process_materialized_job(job)
        return result, ocr_stub.process_slides, llm_stub.create_summary

    def test_summary_job_degrades_with_warning_and_skip_metrics(self):
        result, slides_call, summary_call = self._run_summary(
            self._summary_job([{"source": {}}]), ([], {"unidentified_image": 1})
        )
        self.assertTrue(slides_call.called)
        self.assertTrue(summary_call.called)
        self.assertEqual(result["outputs"]["ppt_pages"], [])
        self.assertEqual(result["metrics"]["slides_skipped"], {"unidentified_image": 1})
        self.assertEqual(result["warnings"], ["slides_skipped"])

    def test_clean_summary_job_keeps_empty_warning_list(self):
        result, slides_call, _summary_call = self._run_summary(
            self._summary_job([]), (["prior-page"], {})
        )
        self.assertFalse(slides_call.called)
        self.assertEqual(result["outputs"]["ppt_pages"], [])
        self.assertEqual(result["warnings"], [])
        self.assertNotIn("slides_skipped", result["metrics"])


if __name__ == "__main__":
    unittest.main()
