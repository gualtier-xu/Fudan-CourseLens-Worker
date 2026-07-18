"""Derived subtitle output formats."""

from __future__ import annotations

from typing import Any


def _stamp(milliseconds: int, separator: str) -> str:
    value = max(0, int(milliseconds))
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def normalize_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    previous_end = 0
    for item in sorted(segments, key=lambda value: int(value.get("start_ms") or 0)):
        text = " ".join(str(item.get("text") or "").split()).strip()
        if not text:
            continue
        start = max(previous_end, int(item.get("start_ms") or 0))
        end = max(start + 200, int(item.get("end_ms") or start + 1000))
        output.append({"start_ms": start, "end_ms": end, "text": text})
        previous_end = end
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
