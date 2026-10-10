"""Deterministic Lecture IR view over already-produced course evidence.

Builds an additive ``lecture_ir`` result view — ordered ``section``,
``knowledge_unit``, and ``key_moment`` units — strictly downstream of the
transcript segments, summary chapters, and recognized ``ppt_pages`` the
runner already holds.  Every unit follows the accepted ``evidence.v1``
generic unit schema: a content-addressed ``unit:<12 hex>`` contract ID, an
absolute ``time`` interval, and nonempty ``spans`` that cite only existing
speech-segment (``seg:``) or slide-event (``slevt:``) IDs.  Generated
chapter summaries, notes, quiz items, and answers never become spans.

Rules that keep the view honest and reproducible:

- Slide-event evidence is never merged away: every recognized occurrence is
  its own ``key_moment``; only identical event identity (same ``slevt:``
  ID, e.g. checkpoint duplicates) collapses.
- A section exists only where a chapter anchor coincides with a real
  evidence anchor (a segment start or a slide time).  Its end is the next
  valid anchor or the transcript/slide evidence end.  Chapters with
  missing, negative, non-integer, or unfounded anchors are dropped —
  timing is never fabricated.
- Units whose interval contains no citable evidence (segments can lose
  their ``seg:`` ID when proofreading rewrites the text, and pages without
  a derivable deck scope carry no ``slevt:`` ID) are omitted; empty or
  legacy inputs therefore degrade to an empty view.

No model call, checkpoint family, wire change, or requested-output mode is
introduced here; the runner attaches the returned dict additively.
"""

from __future__ import annotations

import re
from typing import Any

from shared.evidence_contract import (
    CONTRACT_ID,
    NAMESPACE_SLIDE_ENTITY,
    NAMESPACE_SLIDE_EVENT,
    NAMESPACE_SEGMENT,
    NAMESPACE_UNIT,
    compute_id,
)

# Upper bound on cited spans per unit so a long lecture cannot inflate one
# result payload; the earliest evidence in canonical order wins.
_MAX_UNIT_SPANS = 64

_SEGMENT_ID_RE = re.compile(rf"^{NAMESPACE_SEGMENT}:[0-9a-f]{{12}}$")
_ENTITY_ID_RE = re.compile(rf"^{NAMESPACE_SLIDE_ENTITY}:[0-9a-f]{{12}}$")
_EVENT_ID_RE = re.compile(rf"^{NAMESPACE_SLIDE_EVENT}:[0-9a-f]{{12}}$")


def _as_ms(value: Any) -> int | None:
    """Coerce a millisecond anchor: int or integral finite float, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _contract_id(value: Any, pattern: re.Pattern[str]) -> str | None:
    return value if isinstance(value, str) and pattern.match(value) else None


def _collect_segments(
    transcript: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], set[int], int]:
    """Return citable segments (sorted, deduplicated), all valid segment
    anchor starts, and the transcript end in milliseconds.

    Anchors and the end cover every well-anchored segment regardless of
    ID loss; only segments still carrying a well-formed ``seg:`` ID are
    citable by units.
    """
    by_id: dict[str, dict[str, Any]] = {}
    starts: set[int] = set()
    end = 0
    for item in transcript or []:
        if not isinstance(item, dict):
            continue
        start = _as_ms(item.get("start_ms"))
        stop = _as_ms(item.get("end_ms"))
        if start is None or stop is None or start < 0 or stop < start:
            continue
        starts.add(start)
        end = max(end, stop)
        segment_id = _contract_id(item.get("segment_id"), _SEGMENT_ID_RE)
        if segment_id is None or segment_id in by_id:
            continue
        by_id[segment_id] = {
            "id": segment_id,
            "start_ms": start,
            "end_ms": stop,
        }
    ordered = sorted(by_id.values(), key=lambda value: (value["start_ms"], value["end_ms"], value["id"]))
    return ordered, starts, end


def _collect_events(
    ppt_pages: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Return citable slide events (sorted, deduplicated) and their times.

    Repeated slide content yields distinct events and is preserved; only a
    literally duplicated event ID (identical occurrence) collapses.  Event
    times mirror the OCR producer: ``created_sec`` seconds since media
    start, where a missing value means the deck's first slide at time 0.
    """
    by_id: dict[str, dict[str, Any]] = {}
    starts: set[int] = set()
    for item in ppt_pages or []:
        if not isinstance(item, dict):
            continue
        created = _as_ms(item.get("created_sec"))
        created = 0 if created is None or created < 0 else created
        start_ms = created * 1000
        starts.add(start_ms)
        event_id = _contract_id(item.get("event_id"), _EVENT_ID_RE)
        if event_id is None or event_id in by_id:
            continue
        by_id[event_id] = {
            "id": event_id,
            "entity_id": _contract_id(item.get("entity_id"), _ENTITY_ID_RE),
            "page_num": _as_ms(item.get("page_num")),
            "start_ms": start_ms,
        }
    ordered = sorted(by_id.values(), key=lambda value: (value["start_ms"], value["id"]))
    return ordered, starts


def _in_range(moment: int, start: int, end: int) -> bool:
    """Half-open membership, closed only for a zero-width interval."""
    if start == end:
        return moment == start
    return start <= moment < end


def _active_entity(events: list[dict[str, Any]], at_ms: int) -> str | None:
    """Entity ID of the latest slide event at or before ``at_ms``.

    ``events`` are sorted by (start, id), so the last eligible entry is the
    deterministic winner.
    """
    best: dict[str, Any] | None = None
    for event in events:
        if event["start_ms"] > at_ms:
            break
        best = event
    return best["entity_id"] if best is not None else None


def _unit(
    kind: str,
    title: str | None,
    time_range: dict[str, int],
    spans: list[dict[str, str]],
    content: dict[str, Any] | None,
) -> dict[str, Any]:
    """One evidence.v1 unit with its content-addressed contract ID."""
    unit: dict[str, Any] = {
        "kind": kind,
        "title": title,
        "time": time_range,
        "spans": spans,
        "content": content,
    }
    unit["id"] = compute_id(
        NAMESPACE_UNIT,
        {
            "kind": kind,
            "title": title,
            "time": time_range,
            "spans": spans,
            "content": content,
        },
    )
    return unit


def _range_spans(
    segments: list[dict[str, Any]],
    events: list[dict[str, Any]],
    start: int,
    end: int,
) -> list[dict[str, str]]:
    """Citable segment/slide-event references inside [start, end), capped."""
    collected: list[tuple[int, str, str, dict[str, str]]] = []
    for segment in segments:
        if _in_range(segment["start_ms"], start, end):
            collected.append(
                (segment["start_ms"], "segment", segment["id"],
                 {"kind": "segment", "id": segment["id"]})
            )
    for event in events:
        if _in_range(event["start_ms"], start, end):
            collected.append(
                (event["start_ms"], "slide_event", event["id"],
                 {"kind": "slide_event", "id": event["id"]})
            )
    collected.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in collected[:_MAX_UNIT_SPANS]]


def _build_sections(
    chapters: list[dict[str, Any]] | None,
    segments: list[dict[str, Any]],
    events: list[dict[str, Any]],
    segment_starts: set[int],
    event_starts: set[int],
    evidence_end: int,
) -> list[dict[str, Any]]:
    """One section per unique, evidence-founded chapter anchor.

    The end is the next valid anchor or the evidence end; sections whose
    interval holds no citable evidence are dropped.
    """
    anchors = segment_starts | event_starts
    valid: list[tuple[int, str | None]] = []
    seen: set[tuple[int, str | None]] = set()
    for chapter in chapters or []:
        if not isinstance(chapter, dict):
            continue
        start = _as_ms(chapter.get("start_ms"))
        if start is None or start < 0 or start > evidence_end or start not in anchors:
            continue
        title_value = chapter.get("title")
        title = title_value.strip() if isinstance(title_value, str) else ""
        title = title or None
        key = (start, title)
        if key in seen:
            continue
        seen.add(key)
        valid.append(key)
    valid.sort(key=lambda item: (item[0], item[1] or ""))
    sections: list[dict[str, Any]] = []
    for index, (start, title) in enumerate(valid):
        end = valid[index + 1][0] if index + 1 < len(valid) else evidence_end
        spans = _range_spans(segments, events, start, end)
        if not spans:
            continue
        sections.append(_unit("section", title, {"start_ms": start, "end_ms": end}, spans, None))
    return sections


def _build_knowledge_units(
    sections: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group citable speech inside each section by the active slide entity.

    A new group starts when the slide active at the segment midpoint
    changes; with no slides yet, one group covers the section's citable
    speech.
    """
    units: list[dict[str, Any]] = []
    for section in sections:
        start = section["time"]["start_ms"]
        end = section["time"]["end_ms"]
        groups: list[tuple[str | None, list[dict[str, Any]]]] = []
        for segment in segments:
            if not _in_range(segment["start_ms"], start, end):
                continue
            entity = _active_entity(events, (segment["start_ms"] + segment["end_ms"]) // 2)
            if groups and groups[-1][0] == entity:
                groups[-1][1].append(segment)
            else:
                groups.append((entity, [segment]))
        for entity, members in groups:
            members = members[:_MAX_UNIT_SPANS]
            spans = [{"kind": "segment", "id": member["id"]} for member in members]
            time_range = {
                "start_ms": members[0]["start_ms"],
                "end_ms": members[-1]["end_ms"],
            }
            content = {"entity_id": entity} if entity else None
            units.append(_unit("knowledge_unit", None, time_range, spans, content))
    units.sort(key=lambda unit: (unit["time"]["start_ms"], unit["id"]))
    return units


def _build_key_moments(
    events: list[dict[str, Any]],
    evidence_end: int,
) -> list[dict[str, Any]]:
    """One key moment per slide occurrence, repeated content included."""
    units: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        end = events[index + 1]["start_ms"] if index + 1 < len(events) else evidence_end
        content: dict[str, Any] = {}
        if event["entity_id"] is not None:
            content["entity_id"] = event["entity_id"]
        if event["page_num"] is not None:
            content["page_num"] = event["page_num"]
        units.append(_unit(
            "key_moment",
            None,
            {"start_ms": event["start_ms"], "end_ms": end},
            [{"kind": "slide_event", "id": event["id"]}],
            content or None,
        ))
    return units


def _event_windows(events: list[dict[str, Any]], evidence_end: int) -> dict[str, dict[str, int]]:
    """Slide-event id → its interval, derived the same way key moments do it."""
    windows: dict[str, dict[str, int]] = {}
    for index, event in enumerate(events):
        end = events[index + 1]["start_ms"] if index + 1 < len(events) else evidence_end
        windows[event["id"]] = {"start_ms": event["start_ms"], "end_ms": max(event["start_ms"], end)}
    return windows


def _project_knowledge_points(
    knowledge_points: list[dict[str, Any]] | None,
    evidence_packet: dict[str, Any] | None,
    segments: list[dict[str, Any]],
    events: list[dict[str, Any]],
    evidence_end: int,
) -> tuple[list[dict[str, Any]], int]:
    """Project validated knowledge points into evidence.v1 ``knowledge_unit``s.

    A unit is emitted only when at least one of its citations resolves to a
    locally-held ``seg:``/``slevt|slent:`` identity, because every evidence.v1
    unit must carry a non-empty, resolvable span set and a real interval —
    timing is never fabricated.  Citations of document/assessment kinds have
    no speech anchor, so they travel as course-knowledge ``evidence_refs``
    inside ``content`` instead of becoming spans.  Points that resolve to no
    citable span are skipped and counted (their content still reaches the
    client through ``summary.knowledge_points``).
    """
    points = knowledge_points if isinstance(knowledge_points, list) else []
    if not points:
        return [], 0
    items = {
        str(item.get("citation_id")): item
        for item in ((evidence_packet or {}).get("items") or [])
        if isinstance(item, dict) and item.get("citation_id")
    }
    if not items:
        return [], len(points)
    segments_by_id = {segment["id"]: segment for segment in segments}
    event_windows = _event_windows(events, evidence_end)
    units: list[dict[str, Any]] = []
    skipped = 0
    for point in points:
        if not isinstance(point, dict):
            continue
        text = str(point.get("text") or "").strip()
        if not text:
            continue
        citable: list[tuple[int, int, int, dict[str, str]]] = []
        references: list[dict[str, Any]] = []
        for citation_id in point.get("citation_ids") or []:
            item = items.get(str(citation_id))
            if item is None:
                continue
            kind = str(item.get("kind"))
            source_id = str(item.get("source_id"))
            if kind == "transcript" and source_id in segments_by_id:
                segment = segments_by_id[source_id]
                citable.append((segment["start_ms"], segment["end_ms"], 0,
                                {"kind": "segment", "id": source_id}))
                continue
            if kind == "slide" and source_id in event_windows:
                window = event_windows[source_id]
                citable.append((window["start_ms"], window["end_ms"], 1,
                                {"kind": "slide_event", "id": source_id}))
                continue
            references.append({
                "citation_id": str(citation_id),
                "kind": kind,
                "source_id": source_id,
                "revision_id": str(item.get("revision_id") or ""),
                "content_hash": str(item.get("content_hash") or ""),
                "locator": dict(item.get("locator") or {}),
                "label": str(item.get("label") or ""),
            })
        if not citable:
            skipped += 1
            continue
        citable.sort(key=lambda value: (value[0], value[1], value[2], value[3]["id"]))
        start = min(value[0] for value in citable)
        end = max(value[1] for value in citable)
        content: dict[str, Any] = {"text": text}
        if references:
            content["evidence_refs"] = references[:_MAX_UNIT_SPANS]
        if point.get("conflict") is True:
            content["conflict"] = True
        units.append(_unit(
            "knowledge_unit",
            str(point.get("title") or "").strip()[:60] or None,
            {"start_ms": start, "end_ms": end},
            [value[3] for value in citable[:_MAX_UNIT_SPANS]],
            content,
        ))
    return units, skipped


def build_lecture_ir(
    transcript: list[dict[str, Any]] | None = None,
    chapters: list[dict[str, Any]] | None = None,
    ppt_pages: list[dict[str, Any]] | None = None,
    knowledge_points: list[dict[str, Any]] | None = None,
    evidence_packet: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the additive Lecture IR view from already-produced evidence.

    Total over arbitrary inputs: anything malformed, unfounded, or
    uncitable is dropped, and empty or legacy inputs yield the empty view
    ``{"contract", "sections": [], "knowledge_units": [], "key_moments": []}``.
    ``knowledge_points``/``evidence_packet`` are optional and additive: without
    them the view is exactly the pre-existing one.
    """
    segments, segment_starts, transcript_end = _collect_segments(transcript)
    events, event_starts = _collect_events(ppt_pages)
    evidence_end = max(transcript_end, max(event_starts, default=0))
    sections = _build_sections(
        chapters, segments, events, segment_starts, event_starts, evidence_end
    )
    knowledge_units = _build_knowledge_units(sections, segments, events)
    view = {
        "contract": CONTRACT_ID,
        "sections": sections,
        "knowledge_units": knowledge_units,
        "key_moments": _build_key_moments(events, evidence_end),
    }
    if knowledge_points:
        # 只在真的处理过多源知识点时报账：不传知识点的旧调用拿到的是与历史
        # 逐键相同的视图（空输入仍等于四键空视图）。
        projected, skipped = _project_knowledge_points(
            knowledge_points, evidence_packet, segments, events, evidence_end
        )
        if projected:
            view["knowledge_units"] = sorted(
                knowledge_units + projected,
                key=lambda unit: (unit["time"]["start_ms"], unit["id"]),
            )
        view["knowledge_points_projected"] = len(projected)
        view["knowledge_points_skipped"] = skipped
    return view
