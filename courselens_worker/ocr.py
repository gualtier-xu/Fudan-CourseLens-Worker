"""Bounded, single-threaded OCR for generic slide images."""

from __future__ import annotations

import hashlib
import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from .source import fetch_bytes

_OCR_LOCAL = threading.local()


def _dhash(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8))
    pixels = np.asarray(gray)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _engine() -> RapidOCR:
    value = getattr(_OCR_LOCAL, "engine", None)
    if value is None:
        value = RapidOCR()
        _OCR_LOCAL.engine = value
    return value


def _ocr_page(index: int, item: dict[str, Any], raw: bytes) -> dict[str, Any] | None:
    if not raw:
        return None
    with Image.open(io.BytesIO(raw)) as opened:
        image = opened.convert("RGB")
    fingerprint = _dhash(image)
    result, _elapsed = _engine()(np.asarray(image))
    lines = []
    for row in result or []:
        if len(row) >= 2 and str(row[1]).strip():
            lines.append(str(row[1]).strip())
    return {
        "page_num": int(item.get("page_num") or index + 1),
        "created_sec": int(item.get("created_sec") or 0),
        "text": "\n".join(lines),
        "dhash": fingerprint,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    }


def process_slides(
    slides: list[dict[str, Any]],
    *,
    progress: Callable[[str, int, int], None],
    prior_checkpoint: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    prior = dict(prior_checkpoint or {})
    output: list[dict[str, Any]] = list(prior.get("ppt_pages") or [])
    seen: set[str] = {
        str(item.get("dhash") or "") for item in output if item.get("dhash")
    }
    total = len(slides)
    completed = max(0, min(total, int(prior.get("ocr_completed_items") or 0)))
    prefetch = max(1, min(20, int(os.environ.get("COURSELENS_IMAGE_PREFETCH") or 16)))
    concurrency = max(1, min(2, int(os.environ.get("COURSELENS_OCR_CONCURRENCY") or 1)))
    for batch_start in range(completed, total, prefetch):
        batch_end = min(total, batch_start + prefetch)
        indices = list(range(batch_start, batch_end))
        with ThreadPoolExecutor(max_workers=min(prefetch, len(indices)), thread_name_prefix="image-fetch") as fetch_pool:
            fetched = {
                index: fetch_pool.submit(fetch_bytes, dict(slides[index].get("source") or {}))
                for index in indices
            }
            raw_pages = {index: fetched[index].result() for index in indices}
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="ocr") as ocr_pool:
            recognized = {
                index: ocr_pool.submit(_ocr_page, index, slides[index], raw_pages[index])
                for index in indices
            }
            for index in indices:
                page = recognized[index].result()
                if page and str(page.get("dhash") or "") not in seen:
                    seen.add(str(page["dhash"]))
                    output.append(page)
                progress("ocr", index + 1, total)
                should_checkpoint = (index + 1) % 5 == 0 or index + 1 == total
                if checkpoint is not None and should_checkpoint:
                    checkpoint({
                        "stage": "ocr",
                        "completed_chunks": index + 1,
                        "total_chunks": total,
                        "ocr_completed_items": index + 1,
                        "ppt_pages": output,
                    })
        raw_pages.clear()
    return output
