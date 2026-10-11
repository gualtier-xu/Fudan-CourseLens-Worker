"""Bounded, single-threaded OCR for generic slide images."""

from __future__ import annotations

import hashlib
import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np
from PIL import Image, UnidentifiedImageError
from rapidocr_onnxruntime import RapidOCR

from shared.evidence_contract import (
    NAMESPACE_SLIDE_ENTITY,
    NAMESPACE_SLIDE_EVENT,
    compute_id,
)

from .junk_filter import (
    JUNK_PAGE_SKIP_REASON,
    adjudicate_deck,
    chrome_verdict,
    featureless_verdict,
    frame_features,
    is_junk_page,
)
from .source import fetch_bytes, safe_source_error_code

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


def _fetch_page(source: dict[str, Any]) -> tuple[bytes, str]:
    """Fetch one slide; a failure is a closed-set skip reason, never fatal.

    Slide hosts intermittently reject the session's primary transport while
    serving the other one, so a source may carry an ``_alternate_source``
    used for one bounded retry — for connection failures and for text
    bodies (an authorization page is a transport failure in disguise).
    When both attempts fail, the primary closed-set reason is kept.
    """
    primary = dict(source or {})
    alternate = primary.pop("_alternate_source", None)
    has_alternate = isinstance(alternate, dict) and bool(alternate)
    try:
        raw = fetch_bytes(primary)
    except Exception as exc:  # the URL, headers, and body must stay private
        if not has_alternate:
            return b"", safe_source_error_code(exc) or "fetch_failed"
        try:
            return fetch_bytes(dict(alternate)), ""
        except Exception:
            return b"", safe_source_error_code(exc) or "fetch_failed"
    if has_alternate and _text_body_kind(raw):
        try:
            alternate_raw = fetch_bytes(dict(alternate))
        except Exception:
            return raw, ""
        if not _text_body_kind(alternate_raw):
            return alternate_raw, ""
    return raw, ""


def _text_body_kind(raw: bytes) -> str:
    """Classify a short in-memory prefix as a non-image text body.

    The check is deliberately conservative: only an explicit HTML doctype or
    ``<html`` tag, or a leading JSON object/array marker, is classified; any
    other body stays ``None`` so Pillow decides, and no upstream error kind is
    ever guessed from the content.
    """
    prefix = raw[:512].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html"):
        return "html_body"
    if prefix[:1] in (b"{", b"["):
        return "json_body"
    return ""


def _ocr_page(index: int, item: dict[str, Any], raw: bytes) -> tuple[dict[str, Any] | None, str]:
    """Recognize one slide; a bad page degrades to a closed-set skip reason.

    A platform slide URL can serve a non-image body (for example an
    authorization page) or a format this Pillow build cannot identify.  One
    such page must skip quietly instead of failing the whole summary job.
    """
    if not raw:
        return None, "empty"
    text_kind = _text_body_kind(raw)
    if text_kind:
        return None, text_kind
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            image = opened.convert("RGB")
    except UnidentifiedImageError:
        return None, "unidentified_image"
    except Exception:
        return None, "decode_failed"
    features = None
    try:
        features = frame_features(image)
        # V2 四级流水前两级（NIGHT5-U4）：黑名单命中 / 无特征页 / chrome 结构门
        # 都在 OCR 之前落刀，统一计入 junk_page 闭集原因；过滤器故障绝不致批。
        if (
            is_junk_page(image)
            or featureless_verdict(features)
            or chrome_verdict(features)
        ):
            return None, JUNK_PAGE_SKIP_REASON
    except Exception:
        features = None  # a filter fault must not fail the batch; the page stays content
    try:
        fingerprint = _dhash(image)
        result, _elapsed = _engine()(np.asarray(image))
    except Exception:
        return None, "ocr_failed"
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
        "_junk": features,
    }, ""


def process_slides(
    slides: list[dict[str, Any]],
    *,
    progress: Callable[[str, int, int], None],
    prior_checkpoint: dict[str, Any] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return (recognized pages, skip counts by closed-set reason).

    Every successfully recognized occurrence is kept.  Exact repeated
    content (equal ``source_sha256``) is one logical slide entity observed
    at several distinct timeline events, so a repeat is never classified as
    a skipped or error page and identical-looking perceptual hashes never
    discard or merge content — ``dhash`` stays a diagnostic field only.
    The one perceptual skip reason is ``junk_page``: the fixed blacklist
    plus the V2 featureless/chrome/semantic stages (``junk_filter``) drop
    junk screenshots there, each counted under the same closed-set reason.
    Entity/event IDs are deterministic evidence.v1 digests over the deck
    scope and content; they are attached only when derivable from the deck
    scope and stored nonsecret fields.
    """
    prior = dict(prior_checkpoint or {})
    output: list[dict[str, Any]] = list(prior.get("ppt_pages") or [])
    skipped: dict[str, int] = dict(prior.get("ppt_skipped") or {})
    deck = next(
        (dict(item.get("deck") or {}) for item in slides if isinstance(item.get("deck"), dict)),
        {},
    )
    entity_by_content: dict[str, str] = {}
    first_page_by_content: dict[str, int] = {}

    def _attach_identity(page: dict[str, Any]) -> None:
        content = str(page.get("source_sha256") or "")
        if not deck or not content:
            return
        page_num = max(1, int(page.get("page_num") or 0))
        first_page = first_page_by_content.setdefault(content, page_num)
        entity_id = entity_by_content.get(content)
        if entity_id is None:
            entity_id = compute_id(NAMESPACE_SLIDE_ENTITY, {
                "source_id": str(deck.get("source_id") or ""),
                "deck_id": str(deck.get("deck_id") or ""),
                "page": first_page,
                "content_sha256": content,
            })
            entity_by_content[content] = entity_id
        page["entity_id"] = entity_id
        page["event_id"] = compute_id(NAMESPACE_SLIDE_EVENT, {
            "entity": entity_id,
            "start_ms": int(page.get("created_sec") or 0) * 1000,
            "end_ms": None,
        })
        page["deck_id"] = str(deck.get("deck_id") or "")

    for page in output:
        _attach_identity(page)
    total = len(slides)
    completed = max(0, min(total, int(prior.get("ocr_completed_items") or 0)))
    # V2 后两级（③家族聚簇/④语义裁决）的存活页旗标随行收集；旗标不进检查点。
    screened: list[tuple[dict[str, Any], dict[str, object] | None]] = []
    prefetch = max(1, min(20, int(os.environ.get("COURSELENS_IMAGE_PREFETCH") or 16)))
    concurrency = max(1, min(2, int(os.environ.get("COURSELENS_OCR_CONCURRENCY") or 1)))
    for batch_start in range(completed, total, prefetch):
        batch_end = min(total, batch_start + prefetch)
        indices = list(range(batch_start, batch_end))
        with ThreadPoolExecutor(max_workers=min(prefetch, len(indices)), thread_name_prefix="image-fetch") as fetch_pool:
            fetched = {
                index: fetch_pool.submit(_fetch_page, dict(slides[index].get("source") or {}))
                for index in indices
            }
            raw_pages = {index: fetched[index].result() for index in indices}
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="ocr") as ocr_pool:
            recognized = {
                index: ocr_pool.submit(_ocr_page, index, slides[index], raw_pages[index][0])
                for index in indices
            }
            for index in indices:
                fetch_reason = raw_pages[index][1]
                page, ocr_reason = recognized[index].result()
                reason = fetch_reason or ocr_reason
                if page is None:
                    if reason:
                        skipped[reason] = int(skipped.get(reason) or 0) + 1
                else:
                    screened.append((page, page.pop("_junk", None)))
                    _attach_identity(page)
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
                        "ppt_skipped": skipped,
                    })
        raw_pages.clear()
    # V2 后两级（NIGHT5-U4）：家族聚簇与语义裁决只作用于本轮新幸存页；
    # 语义词从不单独落刀，必须伴随聚簇或弱 chrome 形状证据。历史检查点页
    # 在其所在轮次已完成裁决，不重复进入。
    flagged = [(page, features) for page, features in screened if features is not None]
    if flagged:
        records = [{**features, "text": str(page.get("text") or "")} for page, features in flagged]
        verdicts = adjudicate_deck(records)
        killed = {id(page) for (page, _features), kill in zip(flagged, verdicts) if kill}
        if killed:
            output[:] = [page for page in output if id(page) not in killed]
            skipped[JUNK_PAGE_SKIP_REASON] = int(skipped.get(JUNK_PAGE_SKIP_REASON) or 0) + len(killed)
    return output, {name: count for name, count in skipped.items() if count > 0}
