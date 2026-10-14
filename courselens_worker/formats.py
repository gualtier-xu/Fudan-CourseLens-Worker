"""Derived subtitle output formats."""

from __future__ import annotations

from typing import Any


def _stamp(milliseconds: int, separator: str) -> str:
    value = max(0, int(milliseconds))
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


# Additive evidence.v1 fields carried through normalization instead of being
# dropped.  Anchors, text, and these keys are the accepted segment envelope;
# identity/secret validation stays with shared/evidence_contract.py and the
# client compatibility seam.  "correction" is the bounded proofread status
# emitted by llm.py (applied / rejected-* / unpaired); it records how the
# text was produced and is not part of any evidence identity.
_EVIDENCE_KEYS = (
    "segment_id",
    "evidence_id",
    "source_hash",
    "provenance",
    "tokens",
    "lang",
    "correction",
    "term_revision",
)


def normalize_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort, drop empty text, and repair missing or nonpositive durations.

    Valid overlaps between segments are preserved (evidence.v1 §4): one
    segment is never clamped to the previous segment's end.
    夜10-C 边界加固：非 dict 行与不可解析起始时间戳的行整行跳过；结束时间戳
    不可解析时按缺省修复（start+1000），绝不因异常文稿形态崩溃。
    """
    output: list[dict[str, Any]] = []
    keyed: list[tuple[int, dict[str, Any]]] = []
    for item in segments:
        if not isinstance(item, dict):
            continue
        try:
            order_key = int(item.get("start_ms") or 0)
        except (TypeError, ValueError):
            continue
        keyed.append((order_key, item))
    keyed.sort(key=lambda pair: pair[0])
    for order_key, item in keyed:
        text = " ".join(str(item.get("text") or "").split()).strip()
        if not text:
            continue
        start = max(0, order_key)
        try:
            raw_end = int(item.get("end_ms") or start + 1000)
        except (TypeError, ValueError):
            raw_end = start + 1000
        end = max(start + 200, raw_end)
        cleaned = {"start_ms": start, "end_ms": end, "text": text}
        for key in _EVIDENCE_KEYS:
            value = item.get(key)
            if value is not None:
                cleaned[key] = value
        output.append(cleaned)
    return output


def to_srt(segments: list[dict[str, Any]]) -> str:
    blocks = []
    for index, item in enumerate(normalize_segments(segments), start=1):
        blocks.append(
            f"{index}\n{_stamp(item['start_ms'], ',')} --> {_stamp(item['end_ms'], ',')}\n{item['text']}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(segments: list[dict[str, Any]]) -> str:
    blocks = ["WEBVTT"]
    for item in normalize_segments(segments):
        blocks.append(
            f"{_stamp(item['start_ms'], '.')} --> {_stamp(item['end_ms'], '.')}\n{item['text']}"
        )
    return "\n\n".join(blocks) + "\n"
