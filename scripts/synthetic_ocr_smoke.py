"""Exercise RapidOCR with a slide generated inside the Actions runner."""

from __future__ import annotations

import io
import json
import time

from PIL import Image, ImageDraw, ImageFont

from courselens_worker import ocr


SYNTHETIC_LABEL = "COURSE LENS SYNTHETIC SLIDE 2026"


def _slide_bytes() -> bytes:
    image = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            64,
        )
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((60, 60, 1220, 660), outline="#204a87", width=8)
    draw.text((120, 270), SYNTHETIC_LABEL, fill="black", font=font)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def main() -> int:
    raw = _slide_bytes()
    checkpoints: list[dict] = []
    progress: list[tuple[str, int, int]] = []
    original_fetch = ocr.fetch_bytes
    started = time.monotonic()
    try:
        ocr.fetch_bytes = lambda source: raw
        pages = ocr.process_slides(
            [
                {"page_num": 1, "created_sec": 0, "source": {"synthetic": True}},
                {"page_num": 2, "created_sec": 5, "source": {"synthetic": True}},
            ],
            progress=lambda stage, completed, total: progress.append(
                (stage, completed, total)
            ),
            checkpoint=checkpoints.append,
        )
    finally:
        ocr.fetch_bytes = original_fetch
    if len(pages) != 1:
        raise RuntimeError("exact duplicate synthetic slides were not deduplicated")
    if not str(pages[0].get("text") or "").strip():
        raise RuntimeError("RapidOCR returned no text for the generated slide")
    if not checkpoints or checkpoints[-1].get("ocr_completed_items") != 2:
        raise RuntimeError("OCR checkpoint did not cover every generated slide")
    print(json.dumps({
        "schema": "synthetic-ocr-smoke.v1",
        "sample_origin": "generated-in-runner",
        "input_pages": 2,
        "deduplicated_pages": len(pages),
        "recognized_characters": len(str(pages[0].get("text") or "")),
        "progress_events": len(progress),
        "checkpoints": len(checkpoints),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
