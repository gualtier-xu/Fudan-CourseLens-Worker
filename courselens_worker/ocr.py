"""Bounded, single-threaded OCR for generic slide images."""

from __future__ import annotations

import hashlib
import io
from typing import Any, Callable

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from .source import fetch_bytes


def _dhash(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8))
    pixels = np.asarray(gray)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def process_slides(
    slides: list[dict[str, Any]],
    *,
    progress: Callable[[str, int, int], None],
    prior_checkpoint: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    engine = RapidOCR()
    prior = dict(prior_checkpoint or {})
    output: list[dict[str, Any]] = list(prior.get("ppt_pages") or [])
    seen: set[str] = {
        str(item.get("dhash") or "") for item in output if item.get("dhash")
    }
    total = len(slides)
    completed = max(0, min(total, int(prior.get("ocr_completed_items") or 0)))
    for index in range(completed, total):
        item = slides[index]
        raw = fetch_bytes(dict(item.get("source") or {}))
        if raw:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            fingerprint = _dhash(image)
            if fingerprint not in seen:
                seen.add(fingerprint)
                result, _elapsed = engine(np.asarray(image))
                lines = []
                for row in result or []:
                    if len(row) >= 2 and str(row[1]).strip():
                        lines.append(str(row[1]).strip())
                output.append({
                    "page_num": int(item.get("page_num") or index + 1),
                    "created_sec": int(item.get("created_sec") or 0),
                    "text": "\n".join(lines),
                    "dhash": fingerprint,
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                })
            del raw, image
        progress("ocr", index + 1, total)
        if checkpoint is not None:
            checkpoint({
                "stage": "ocr",
                "completed_chunks": index + 1,
                "total_chunks": total,
                "ocr_completed_items": index + 1,
                "ppt_pages": output,
            })
    return output
