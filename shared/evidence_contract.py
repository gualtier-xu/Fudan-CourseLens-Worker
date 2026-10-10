"""CourseLens evidence contract v1 (``evidence.v1``).

Deterministic, standard-library-only JSON contract for the subtitle/AI
foundation batch.  It defines immutable source identity and provenance,
speech/ASR evidence with absolute millisecond anchors and optional token
time points, correction records and display cues, ``SlideEntity`` versus
occurrence-level ``SlideEvent``, generic ``EvidenceUnit`` references
(sections, knowledge units, key moments, notes, QA, quiz items), checkpoint
interruption metadata, and the empty measurement-table catalog used by the
synthetic benchmark fixture.

Semantics:

- Immutable evidence: source identity, raw speech segments, slide entities
  and events, correction records, checkpoint records.  Corrections never
  mutate the evidence they target.
- Regenerable views: display cues, evidence units (notes, answers, quizzes
  are views over evidence, not evidence), and measurement values.
- IDs are content/provenance addressed ``<namespace>:<12 hex>`` digests over
  canonical identity fields; they never depend on display-array position.

This module deliberately does not touch ``shared/protocol/*``
(``job.v2``/``result.v2``/``control.v2``/``sealed.v2``), SRT/VTT exports,
``summary``/``chapters``/``ppt_pages``, or any provider/UI code.  Normative
text and compatibility rules: ``docs/evidence-contract-v1.md``.  Synthetic
benchmark fixture: ``tests/fixtures/evidence_benchmark_v1.json``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re

CONTRACT_ID = "evidence.v1"
_ID_DIGEST_CHARS = 12

NAMESPACE_SOURCE = "src"
NAMESPACE_SEGMENT = "seg"
NAMESPACE_CORRECTION = "cor"
NAMESPACE_CUE = "cue"
NAMESPACE_SLIDE_ENTITY = "slent"
NAMESPACE_SLIDE_EVENT = "slevt"
NAMESPACE_UNIT = "unit"
NAMESPACE_CHECKPOINT = "chk"
NAMESPACE_MEASUREMENT = "mtr"

_ID_RE = re.compile(r"^(src|seg|cor|cue|slent|slevt|unit|chk|mtr):[0-9a-f]{12}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONFIG_HASH_RE = re.compile(r"^[0-9a-f]{12,64}$")
_LANG_RE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")

SOURCE_KINDS = ("recording", "live", "slide_deck", "document")
SOURCE_ORIGINS = ("native_upload", "live_capture", "external_import")
# ``needs_alignment`` is the "alignment required" state from the accepted
# U0 synthesis ledger; ``timing_preserved`` keeps target anchors; ``rejected``
# records are audit-only and must never back cues or units.
CORRECTION_STATES = ("timing_preserved", "needs_alignment", "rejected")
CORRECTION_ACTORS = ("user", "llm", "rule")
UNIT_KINDS = ("section", "knowledge_unit", "key_moment", "note", "qa", "quiz_item")
UNIT_SPAN_KINDS = ("segment", "correction", "slide_event")
CUE_SOURCE_KINDS = ("segment", "correction")
CHECKPOINT_STATUSES = ("complete", "interrupted")
MEASUREMENT_DIRECTIONS = ("lower_is_better", "higher_is_better", "information")

# Credential-shaped strings rejected in any document string value.  Strings
# that are exactly a contract ID or a 64-hex digest are exempt (hashes are
# legitimate provenance); the patterns below target key material instead.
SECRET_VALUE_PATTERNS = (
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "provider key literal (sk-...)"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"), "GitHub token"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    (re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{16,}"), "bearer credential"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key block"),
    (
        re.compile(
            r"(?i)(api[_-]?key|secret|passwd|password|credential|bearer\s+token)\s*[:=]\s*\S+"
        ),
        "credential assignment",
    ),
)
_SECRET_KEY_RE = re.compile(r"(?i)(secret|passwd|password|api[_-]?key|apikey|credential)")


class EvidenceContractError(ValueError):
    """Closed validation failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str, path: str = "") -> None:
        super().__init__(f"{code}: {message}" + (f" (at {path})" if path else ""))
        self.code = code
        self.message = message
        self.path = path


def canonical_json(value) -> str:
    """Canonical JSON text: sorted keys, compact separators, finite numbers."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def compute_id(namespace: str, identity: dict) -> str:
    """Deterministic, position-independent ID over canonical identity fields."""
    payload = canonical_json(identity).encode("utf-8")
    return f"{namespace}:{hashlib.sha256(payload).hexdigest()[:_ID_DIGEST_CHARS]}"


def _fail(code: str, message: str, path: str) -> None:
    raise EvidenceContractError(code, message, path)


def _check(condition: bool, code: str, message: str, path: str) -> None:
    if not condition:
        raise EvidenceContractError(code, message, path)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _safe_canonical(value) -> str:
    try:
        return canonical_json(value)
    except (TypeError, ValueError):
        return repr(value)


def _sort_key(obj, fields):
    """Total, type-tolerant sort key so array order is canonical under any
    input permutation; garbage field values still sort deterministically and
    are rejected later by validation."""
    parts = []
    for field in fields:
        value = obj.get(field) if isinstance(obj, dict) else None
        if _is_int(value):
            parts.append((0, value, ""))
        elif isinstance(value, str):
            parts.append((1, 0, value))
        elif value is None:
            parts.append((2, 0, ""))
        else:
            parts.append((3, 0, _safe_canonical(value)))
    parts.append((4, 0, _safe_canonical(obj) if isinstance(obj, dict) else repr(obj)))
    return tuple(parts)


def _require_ms(value, path: str) -> int:
    _check(_is_int(value), "type_invalid", "expected integer milliseconds", path)
    _check(value >= 0, "anchor_negative", "millisecond anchor must be >= 0", path)
    return value


def _validate_optional_anchor_pair(start, end, path: str, duration_ms):
    """Both-or-neither anchors; non-negative; start <= end; within duration."""
    if start is None and end is None:
        return None
    _check(
        start is not None and end is not None,
        "value_invalid",
        "start_ms/end_ms must be set together",
        path,
    )
    _require_ms(start, path + ".start_ms")
    _require_ms(end, path + ".end_ms")
    if start > end:
        _fail("anchor_descending", "start_ms must be <= end_ms", path)
    if duration_ms is not None and end > duration_ms:
        _fail("anchor_out_of_range", "anchor exceeds source duration_ms", path)
    return (start, end)


# ---------------------------------------------------------------------------
# Identity fields per object type.  These are the only fields hashed into
# IDs; regenerable or additive fields are excluded so views can change
# without churning evidence identity.


def _source_identity(source: dict) -> dict:
    return {
        "kind": source.get("kind"),
        "origin": source.get("origin"),
        "title": source.get("title"),
        "duration_ms": source.get("duration_ms"),
        "source_sha256": source.get("source_sha256"),
    }


def _fingerprint_scope(fingerprints: dict) -> dict:
    """Producer/model/config provenance folded into evidence identities."""
    return {
        "producer": fingerprints.get("producer"),
        "model": fingerprints.get("model"),
        "config_hash": fingerprints.get("config_hash"),
    }


def _segment_identity(segment: dict, source_id: str, fingerprint_scope: dict) -> dict:
    identity = {
        "source_id": source_id,
        "start_ms": segment.get("start_ms"),
        "end_ms": segment.get("end_ms"),
        "text": segment.get("text"),
        "lang": segment.get("lang"),
        "no_speech": segment.get("no_speech"),
    }
    identity.update(fingerprint_scope)
    return identity


def _correction_identity(correction: dict) -> dict:
    return {
        "target": correction.get("target"),
        "state": correction.get("state"),
        "text": correction.get("text"),
        "start_ms": correction.get("start_ms"),
        "end_ms": correction.get("end_ms"),
        "actor": correction.get("actor"),
        "reason": correction.get("reason"),
    }


def _cue_identity(cue: dict) -> dict:
    return {
        "start_ms": cue.get("start_ms"),
        "end_ms": cue.get("end_ms"),
        "lines": cue.get("lines"),
        "derived_from": cue.get("derived_from"),
        "lang": cue.get("lang"),
    }


def _slide_entity_identity(entity: dict, source_id: str) -> dict:
    return {
        "source_id": source_id,
        "deck_id": entity.get("deck_id"),
        "page": entity.get("page"),
        "content_sha256": entity.get("content_sha256"),
    }


def _slide_event_identity(event: dict) -> dict:
    return {
        "entity": event.get("entity"),
        "start_ms": event.get("start_ms"),
        "end_ms": event.get("end_ms"),
    }


def _unit_identity(unit: dict) -> dict:
    return {
        "kind": unit.get("kind"),
        "title": unit.get("title"),
        "time": unit.get("time"),
        "spans": unit.get("spans"),
        "content": unit.get("content"),
    }


def _checkpoint_identity(checkpoint: dict, source_id: str) -> dict:
    return {
        "source_id": source_id,
        "stage": checkpoint.get("stage"),
        "seq": checkpoint.get("seq"),
        "status": checkpoint.get("status"),
        "position_ms": checkpoint.get("position_ms"),
        "payload_sha256": checkpoint.get("payload_sha256"),
        "detail": checkpoint.get("detail"),
    }


def _measurement_identity(row: dict) -> dict:
    return {
        "family": row.get("family"),
        "metric": row.get("metric"),
        "unit": row.get("unit"),
        "direction": row.get("direction"),
        "definition": row.get("definition"),
        "dims": row.get("dims"),
        "scope": row.get("scope"),
    }


# ---------------------------------------------------------------------------
# Empty measurement catalog.  Definitions only; values stay null until a
# benchmark run fills them.  No thresholds or values are invented here.


METRICS = (
    {"family": "asr_text", "metric": "cer", "unit": "ratio", "direction": "lower_is_better",
     "definition": "character error rate = (substitutions + deletions + insertions) / reference character count; rows with zero reference characters stay null"},
    {"family": "asr_text", "metric": "wer", "unit": "ratio", "direction": "lower_is_better",
     "definition": "word error rate = (substitutions + deletions + insertions) / reference whitespace-delimited word count; rows with zero reference words stay null"},
    {"family": "asr_text", "metric": "terminology_accuracy", "unit": "ratio", "direction": "higher_is_better",
     "definition": "share of scope term-list occurrences transcribed with an exact surface match in the hypothesis"},
    {"family": "asr_text", "metric": "numeric_accuracy", "unit": "ratio", "direction": "higher_is_better",
     "definition": "share of numeric items (integers, decimals, percentages, scientific notation) transcribed exactly"},
    {"family": "asr_text", "metric": "unit_accuracy", "unit": "ratio", "direction": "higher_is_better",
     "definition": "share of scope unit-list items (SI units, currencies, percentages) transcribed exactly"},
    {"family": "asr_text", "metric": "negation_accuracy", "unit": "ratio", "direction": "higher_is_better",
     "definition": "share of negated scopes (e.g. 不 / 无 / 未 / not / without) whose polarity is preserved in the hypothesis"},
    {"family": "asr_text", "metric": "unsupported_addition_rate", "unit": "ratio", "direction": "lower_is_better",
     "definition": "hypothesis content with no reference support divided by total hypothesis content"},
    {"family": "alignment", "metric": "boundary_mae_ms", "unit": "ms", "direction": "lower_is_better",
     "definition": "mean absolute error in milliseconds between hypothesis anchors and reference anchors over time-matched spans"},
    {"family": "alignment", "metric": "boundary_p95_ms", "unit": "ms", "direction": "lower_is_better",
     "definition": "95th percentile absolute anchor error in milliseconds over time-matched spans"},
    {"family": "alignment", "metric": "speech_coverage", "unit": "ratio", "direction": "higher_is_better",
     "definition": "voiced reference duration covered by non-empty hypotheses divided by total voiced reference duration"},
    {"family": "alignment", "metric": "alignment_failure_rate", "unit": "ratio", "direction": "lower_is_better",
     "definition": "corrected spans still in state needs_alignment after the realignment pass divided by all corrected spans"},
    {"family": "presentation", "metric": "suber", "unit": "ratio", "direction": "lower_is_better",
     "definition": "subtitle error rate: edit distance over cue blocks and line segmentations normalized by reference cue length"},
    {"family": "presentation", "metric": "cue_duration_violation_rate", "unit": "ratio", "direction": "lower_is_better",
     "definition": "display cues whose duration exceeds the presentation limit divided by total display cues"},
    {"family": "presentation", "metric": "cue_cps_p95", "unit": "chars_per_second", "direction": "lower_is_better",
     "definition": "95th percentile of characters per second across display cues (one unit per CJK character, one per latin word)"},
    {"family": "presentation", "metric": "cue_line_width_violation_rate", "unit": "ratio", "direction": "lower_is_better",
     "definition": "cue lines whose rendered width exceeds the display limit divided by total cue lines"},
    {"family": "retrieval", "metric": "retrieval_recall_at_k", "unit": "ratio", "direction": "higher_is_better",
     "definition": "reference-relevant evidence units retrieved within the top k divided by all relevant units; k is recorded in scope"},
    {"family": "retrieval", "metric": "retrieval_mrr", "unit": "score", "direction": "higher_is_better",
     "definition": "mean reciprocal rank of the first reference-relevant evidence unit across queries"},
    {"family": "retrieval", "metric": "retrieval_evidence_sufficiency", "unit": "ratio", "direction": "higher_is_better",
     "definition": "retrieval answers citing at least the minimum required evidence units divided by all retrieval answers"},
    {"family": "qa", "metric": "qa_faithfulness", "unit": "ratio", "direction": "higher_is_better",
     "definition": "answers fully entailed by their cited evidence units divided by all non-refusal answers"},
    {"family": "qa", "metric": "qa_refusal_rate", "unit": "ratio", "direction": "information",
     "definition": "answers refused for insufficient evidence divided by all questions; the target rate is product policy, so direction is informational"},
    {"family": "quiz", "metric": "quiz_validity", "unit": "ratio", "direction": "higher_is_better",
     "definition": "quiz items that are answerable from cited evidence with a single determinate answer divided by all generated items"},
    {"family": "quiz", "metric": "quiz_answerability", "unit": "ratio", "direction": "higher_is_better",
     "definition": "quiz items a prepared student can answer from the cited evidence alone divided by all generated items"},
    {"family": "quiz", "metric": "quiz_correctness", "unit": "ratio", "direction": "higher_is_better",
     "definition": "generated answers matching adjudicated reference answers divided by answerable items"},
    {"family": "performance", "metric": "rtf", "unit": "ratio", "direction": "lower_is_better",
     "definition": "real-time factor = producer wall-clock processing time divided by media duration"},
    {"family": "performance", "metric": "rss_peak_mb", "unit": "MB", "direction": "lower_is_better",
     "definition": "peak resident set size of the producing process in mebibytes"},
    {"family": "performance", "metric": "disk_delta_mb", "unit": "MB", "direction": "lower_is_better",
     "definition": "net durable bytes written by the producing run divided by 1 MiB"},
    {"family": "cost", "metric": "api_tokens_total", "unit": "count", "direction": "information",
     "definition": "sum of provider-reported input and output counts for the producing run; usage counts only, never credential material"},
    {"family": "cost", "metric": "api_cost_usd", "unit": "usd", "direction": "information",
     "definition": "provider-reported billing estimate in US dollars for the producing run"},
    {"family": "checkpoint", "metric": "checkpoint_size_bytes", "unit": "bytes", "direction": "information",
     "definition": "largest serialized checkpoint payload in bytes across pipeline stages for the producing run"},
    {"family": "checkpoint", "metric": "recovery_time_s", "unit": "s", "direction": "lower_is_better",
     "definition": "wall-clock seconds from interruption detection to resumed equivalent progress"},
)

METRIC_INDEX = {(spec["family"], spec["metric"]): spec for spec in METRICS}


def empty_measurement_tables() -> list:
    """Canonical empty measurement rows (value null) for benchmark fixtures."""
    rows = []
    for spec in METRICS:
        row = {
            "family": spec["family"],
            "metric": spec["metric"],
            "unit": spec["unit"],
            "direction": spec["direction"],
            "definition": spec["definition"],
            "value": None,
            "dims": {},
            "scope": None,
        }
        row["id"] = compute_id(NAMESPACE_MEASUREMENT, _measurement_identity(row))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Normalization: defaults plus canonical array order.  Array sorting uses
# identity-safe keys only, so any input permutation normalizes identically.


def normalize_document(document):
    """Return a canonical deep copy with defaults filled and arrays sorted.

    Envelope problems raise immediately; per-object validation happens in
    :func:`validate_document`.  Token point order inside a segment is
    deliberately not re-sorted: it is evidence about the producer and is
    validated as non-descending instead.
    """
    if not isinstance(document, dict):
        _fail("not_json_object", "document must be a JSON object", "")
    if "contract" not in document:
        _fail("contract_missing", "missing contract field", "contract")
    if document["contract"] != CONTRACT_ID:
        _fail(
            "contract_unsupported",
            f"unsupported contract {document['contract']!r}; expected {CONTRACT_ID!r}",
            "contract",
        )
    doc = copy.deepcopy(document)

    doc.setdefault("speech", {})
    doc.setdefault("corrections", [])
    doc.setdefault("cues", [])
    doc.setdefault("slides", {})
    doc.setdefault("units", [])
    doc.setdefault("checkpoints", [])
    doc.setdefault("fingerprints", {})
    doc.setdefault("measurements", {})

    speech = doc["speech"]
    if isinstance(speech, dict):
        speech.setdefault("segments", [])
        segments = speech["segments"]
        if isinstance(segments, list):
            for segment in segments:
                if isinstance(segment, dict):
                    segment.setdefault("lang", None)
                    segment.setdefault("tokens", None)
                    segment.setdefault("confidence", None)
                    segment.setdefault("no_speech", False)
                    segment.setdefault("source_hash", None)
            segments.sort(key=lambda s: _sort_key(s, ("start_ms", "end_ms", "text", "lang")))
    corrections = doc["corrections"]
    if isinstance(corrections, list):
        corrections.sort(
            key=lambda c: _sort_key(
                c, ("target", "state", "start_ms", "end_ms", "text", "actor", "reason")
            )
        )
    cues = doc["cues"]
    if isinstance(cues, list):
        cues.sort(key=lambda c: _sort_key(c, ("start_ms", "end_ms", "lines", "derived_from")))
    slides = doc["slides"]
    if isinstance(slides, dict):
        slides.setdefault("entities", [])
        slides.setdefault("events", [])
        entities = slides["entities"]
        if isinstance(entities, list):
            entities.sort(
                key=lambda e: _sort_key(e, ("deck_id", "page", "content_sha256"))
            )
        events = slides["events"]
        if isinstance(events, list):
            events.sort(key=lambda e: _sort_key(e, ("start_ms", "end_ms", "entity")))
    units = doc["units"]
    if isinstance(units, list):
        units.sort(key=lambda u: _sort_key(u, ("kind", "title", "time", "spans", "content")))
    checkpoints = doc["checkpoints"]
    if isinstance(checkpoints, list):
        checkpoints.sort(
            key=lambda c: _sort_key(
                c, ("stage", "seq", "status", "position_ms", "payload_sha256", "detail")
            )
        )
    measurements = doc["measurements"]
    if isinstance(measurements, dict):
        measurements.setdefault("tables", [])
        tables = measurements["tables"]
        if isinstance(tables, list):
            tables.sort(
                key=lambda m: _sort_key(
                    m, ("family", "metric", "unit", "direction", "definition", "dims", "scope")
                )
            )
    return doc


# ---------------------------------------------------------------------------
# Validation.  Field checks run before ID verification so malformed anchors
# and unsupported states are reported even when the ID also no longer matches.


def _check_object_id(obj: dict, namespace: str, identity: dict, seen_ids: dict, path: str) -> None:
    provided = obj.get("id")
    _check(isinstance(provided, str), "field_required", "id is required", path + ".id")
    if not _ID_RE.match(provided) or provided.split(":")[0] != namespace:
        _fail(
            "id_malformed",
            f"id must match {namespace}:<12 lowercase hex>",
            path + ".id",
        )
    expected = compute_id(namespace, identity)
    if provided != expected:
        _fail(
            "id_mismatch",
            f"id does not match its identity fields (expected {expected})",
            path + ".id",
        )
    seen = seen_ids.setdefault(namespace, set())
    if provided in seen:
        _fail("duplicate_id", "duplicate object id within its namespace", path + ".id")
    seen.add(provided)


def _scan_secrets(value, path: str) -> None:
    """Reject credential-shaped strings and non-finite numbers anywhere."""
    if isinstance(value, str):
        if _ID_RE.match(value) or _SHA256_RE.match(value):
            return
        for pattern, label in SECRET_VALUE_PATTERNS:
            if pattern.search(value):
                _fail("secret_like", f"string looks like {label}", path)
    elif isinstance(value, bool):
        return
    elif isinstance(value, float):
        if not math.isfinite(value):
            _fail("value_invalid", "non-finite number is not valid JSON data", path)
    elif isinstance(value, dict):
        for key, item in value.items():
            key_path = path + "." + str(key)
            if isinstance(key, str) and _SECRET_KEY_RE.search(key):
                _fail("secret_like_key", f"field name {key!r} looks like a credential store", key_path)
            _scan_secrets(item, key_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_secrets(item, f"{path}[{index}]")


def _validate_source(source, seen_ids: dict) -> None:
    path = "source"
    _check(isinstance(source, dict), "field_required", "source object is required", path)
    _check(
        source.get("kind") in SOURCE_KINDS,
        "value_unsupported",
        f"source.kind must be one of {SOURCE_KINDS}",
        path + ".kind",
    )
    _check(
        source.get("origin") in SOURCE_ORIGINS,
        "value_unsupported",
        f"source.origin must be one of {SOURCE_ORIGINS}",
        path + ".origin",
    )
    title = source.get("title")
    _check(
        title is None or (isinstance(title, str) and title.strip() != ""),
        "value_invalid",
        "source.title must be a non-empty string or null",
        path + ".title",
    )
    duration = source.get("duration_ms")
    if duration is not None:
        _require_ms(duration, path + ".duration_ms")
    sha = source.get("source_sha256")
    _check(
        sha is None or (isinstance(sha, str) and _SHA256_RE.match(sha)),
        "value_invalid",
        "source.source_sha256 must be 64 lowercase hex or null",
        path + ".source_sha256",
    )
    if source.get("origin") == "external_import":
        _check(
            isinstance(sha, str) and bool(_SHA256_RE.match(sha)),
            "field_required",
            "external_import sources must carry source_sha256 (import must not discard provenance)",
            path + ".source_sha256",
        )
    _check_object_id(source, NAMESPACE_SOURCE, _source_identity(source), seen_ids, path)


def _validate_fingerprints(fingerprints) -> None:
    path = "fingerprints"
    _check(isinstance(fingerprints, dict), "type_invalid", "fingerprints must be an object", path)
    if not fingerprints:
        return
    producer = fingerprints.get("producer")
    _check(
        isinstance(producer, str) and producer.strip() != "",
        "field_required",
        "fingerprints.producer is required when fingerprints are present",
        path + ".producer",
    )
    model = fingerprints.get("model")
    _check(
        model is None or isinstance(model, str),
        "type_invalid",
        "fingerprints.model must be a string or null",
        path + ".model",
    )
    config_hash = fingerprints.get("config_hash")
    _check(
        config_hash is None
        or (isinstance(config_hash, str) and _CONFIG_HASH_RE.match(config_hash)),
        "value_invalid",
        "fingerprints.config_hash must be 12-64 lowercase hex or null",
        path + ".config_hash",
    )
    media_sha = fingerprints.get("media_sha256")
    _check(
        media_sha is None or (isinstance(media_sha, str) and _SHA256_RE.match(media_sha)),
        "value_invalid",
        "fingerprints.media_sha256 must be 64 lowercase hex or null",
        path + ".media_sha256",
    )


def _validate_tokens(tokens, start_ms: int, end_ms: int, path: str) -> None:
    if tokens is None:
        return
    _check(isinstance(tokens, list), "type_invalid", "tokens must be an array or null", path)
    previous_start = None
    for index, token in enumerate(tokens):
        tpath = f"{path}[{index}]"
        _check(
            isinstance(token, list) and len(token) == 3,
            "type_invalid",
            "token must be [text, start_ms, end_ms|null]",
            tpath,
        )
        token_text, token_start, token_end = token
        _check(
            isinstance(token_text, str) and token_text != "",
            "value_invalid",
            "token text must be a non-empty string",
            tpath + "[0]",
        )
        _require_ms(token_start, tpath + "[1]")
        if token_start < start_ms or token_start > end_ms:
            _fail("anchor_out_of_range", "token start outside segment anchors", tpath + "[1]")
        if token_end is not None:
            _require_ms(token_end, tpath + "[2]")
            if token_end < token_start:
                _fail("anchor_descending", "token end_ms must be >= start_ms", tpath + "[2]")
            if token_end > end_ms:
                _fail("anchor_out_of_range", "token end outside segment anchors", tpath + "[2]")
        if previous_start is not None and token_start < previous_start:
            _fail("anchor_descending", "token start times must be non-descending", tpath + "[1]")
        previous_start = token_start


def _validate_speech(speech, source: dict, fingerprints: dict, duration_ms, seen_ids: dict) -> dict:
    path = "speech"
    segments_by_id = {}
    _check(isinstance(speech, dict), "type_invalid", "speech must be an object", path)
    segments = speech.get("segments")
    _check(
        isinstance(segments, list),
        "type_invalid",
        "speech.segments must be an array",
        path + ".segments",
    )
    fingerprint_scope = _fingerprint_scope(fingerprints)
    source_id = source.get("id")
    source_sha = source.get("source_sha256")
    for index, segment in enumerate(segments):
        spath = f"{path}.segments[{index}]"
        _check(isinstance(segment, dict), "type_invalid", "segment must be an object", spath)
        start = _require_ms(segment.get("start_ms"), spath + ".start_ms")
        end = _require_ms(segment.get("end_ms"), spath + ".end_ms")
        if start > end:
            _fail("anchor_descending", "segment start_ms must be <= end_ms", spath)
        if duration_ms is not None and end > duration_ms:
            _fail("anchor_out_of_range", "segment end_ms exceeds source duration_ms", spath)
        text = segment.get("text")
        _check(isinstance(text, str), "type_invalid", "segment.text must be a string", spath + ".text")
        no_speech = segment.get("no_speech")
        _check(
            isinstance(no_speech, bool),
            "type_invalid",
            "segment.no_speech must be a boolean",
            spath + ".no_speech",
        )
        if no_speech and text != "":
            _fail("value_invalid", "segment with no_speech=true must have empty text", spath + ".text")
        lang = segment.get("lang")
        _check(
            lang is None or (isinstance(lang, str) and _LANG_RE.match(lang)),
            "value_invalid",
            "segment.lang must be a language tag or null",
            spath + ".lang",
        )
        confidence = segment.get("confidence")
        if confidence is not None:
            _check(
                isinstance(confidence, (int, float))
                and not isinstance(confidence, bool)
                and math.isfinite(confidence)
                and 0.0 <= confidence <= 1.0,
                "value_invalid",
                "segment.confidence must be a finite number in [0, 1] or null",
                spath + ".confidence",
            )
        source_hash = segment.get("source_hash")
        _check(
            source_hash is None
            or (isinstance(source_hash, str) and _SHA256_RE.match(source_hash)),
            "value_invalid",
            "segment.source_hash must be 64 lowercase hex or null",
            spath + ".source_hash",
        )
        if (
            source_hash is not None
            and isinstance(source_sha, str)
            and source_hash != source_sha
        ):
            _fail(
                "value_invalid",
                "segment.source_hash must match source.source_sha256",
                spath + ".source_hash",
            )
        _validate_tokens(segment.get("tokens"), start, end, spath + ".tokens")
        identity = _segment_identity(segment, source_id, fingerprint_scope)
        _check_object_id(segment, NAMESPACE_SEGMENT, identity, seen_ids, spath)
        segments_by_id[segment["id"]] = segment
    return segments_by_id


def _validate_corrections(corrections, segments_by_id: dict, duration_ms, seen_ids: dict) -> dict:
    path = "corrections"
    _check(isinstance(corrections, list), "type_invalid", "corrections must be an array", path)
    by_id = {}
    for index, correction in enumerate(corrections):
        cpath = f"{path}[{index}]"
        _check(isinstance(correction, dict), "type_invalid", "correction must be an object", cpath)
        target = correction.get("target")
        _check(
            isinstance(target, str),
            "type_invalid",
            "correction.target must be a string",
            cpath + ".target",
        )
        target_segment = segments_by_id.get(target)
        _check(
            target_segment is not None,
            "reference_missing",
            "correction.target does not match a segment id in this document",
            cpath + ".target",
        )
        state = correction.get("state")
        _check(
            state in CORRECTION_STATES,
            "status_unsupported",
            f"correction.state must be one of {CORRECTION_STATES}",
            cpath + ".state",
        )
        text = correction.get("text")
        _check(
            text is None or isinstance(text, str),
            "type_invalid",
            "correction.text must be a string or null",
            cpath + ".text",
        )
        anchors = _validate_optional_anchor_pair(
            correction.get("start_ms"), correction.get("end_ms"), cpath, duration_ms
        )
        actor = correction.get("actor")
        _check(
            actor is None or actor in CORRECTION_ACTORS,
            "value_unsupported",
            f"correction.actor must be one of {CORRECTION_ACTORS} or null",
            cpath + ".actor",
        )
        reason = correction.get("reason")
        _check(
            reason is None or isinstance(reason, str),
            "type_invalid",
            "correction.reason must be a string or null",
            cpath + ".reason",
        )
        if state == "timing_preserved":
            if anchors is not None and (
                anchors[0] != target_segment["start_ms"] or anchors[1] != target_segment["end_ms"]
            ):
                _fail(
                    "state_conflict",
                    "timing_preserved keeps the target anchors; use needs_alignment for moved anchors",
                    cpath,
                )
            if text is None or text == target_segment["text"]:
                _fail("empty_correction", "timing_preserved must change the text", cpath)
        elif state == "needs_alignment":
            anchors_changed = anchors is not None and (
                anchors[0] != target_segment["start_ms"] or anchors[1] != target_segment["end_ms"]
            )
            text_changed = text is not None and text != target_segment["text"]
            if not (anchors_changed or text_changed):
                _fail(
                    "empty_correction",
                    "needs_alignment must change anchors or text",
                    cpath,
                )
        # rejected: audit-only record of a proposal that must not be applied;
        # no further constraints beyond the checks above.
        identity = _correction_identity(correction)
        _check_object_id(correction, NAMESPACE_CORRECTION, identity, seen_ids, cpath)
        by_id[correction["id"]] = correction
    return by_id


def _validate_cues(cues, segments_by_id: dict, corrections_by_id: dict, duration_ms, seen_ids: dict) -> None:
    path = "cues"
    _check(isinstance(cues, list), "type_invalid", "cues must be an array", path)
    for index, cue in enumerate(cues):
        qpath = f"{path}[{index}]"
        _check(isinstance(cue, dict), "type_invalid", "cue must be an object", qpath)
        start = _require_ms(cue.get("start_ms"), qpath + ".start_ms")
        end = _require_ms(cue.get("end_ms"), qpath + ".end_ms")
        if start > end:
            _fail("anchor_descending", "cue start_ms must be <= end_ms", qpath)
        if duration_ms is not None and end > duration_ms:
            _fail("anchor_out_of_range", "cue end_ms exceeds source duration_ms", qpath)
        lines = cue.get("lines")
        _check(
            isinstance(lines, list) and 1 <= len(lines) <= 8,
            "value_invalid",
            "cue.lines must be an array of 1-8 strings",
            qpath + ".lines",
        )
        for line_index, line in enumerate(lines):
            _check(
                isinstance(line, str) and line != "",
                "value_invalid",
                "cue lines must be non-empty strings",
                f"{qpath}.lines[{line_index}]",
            )
        derived = cue.get("derived_from")
        _check(
            isinstance(derived, dict),
            "field_required",
            "cue.derived_from is required (cues are derived views, never authoritative)",
            qpath + ".derived_from",
        )
        kind = derived.get("kind")
        _check(
            kind in CUE_SOURCE_KINDS,
            "value_unsupported",
            f"cue.derived_from.kind must be one of {CUE_SOURCE_KINDS}",
            qpath + ".derived_from.kind",
        )
        ref = derived.get("id")
        _check(
            isinstance(ref, str),
            "type_invalid",
            "cue.derived_from.id must be a string",
            qpath + ".derived_from.id",
        )
        if kind == "segment":
            _check(
                ref in segments_by_id,
                "reference_missing",
                "cue.derived_from.id does not match a segment id in this document",
                qpath + ".derived_from.id",
            )
        else:
            correction = corrections_by_id.get(ref)
            _check(
                correction is not None,
                "reference_missing",
                "cue.derived_from.id does not match a correction id in this document",
                qpath + ".derived_from.id",
            )
            if correction["state"] == "rejected":
                _fail(
                    "invalid_derivation",
                    "display cues must not derive from rejected corrections",
                    qpath + ".derived_from.id",
                )
        lang = cue.get("lang")
        _check(
            lang is None or (isinstance(lang, str) and _LANG_RE.match(lang)),
            "value_invalid",
            "cue.lang must be a language tag or null",
            qpath + ".lang",
        )
        identity = _cue_identity(cue)
        _check_object_id(cue, NAMESPACE_CUE, identity, seen_ids, qpath)


def _validate_slides(slides, source: dict, duration_ms, seen_ids: dict) -> dict:
    path = "slides"
    _check(isinstance(slides, dict), "type_invalid", "slides must be an object", path)
    entities = slides.get("entities")
    _check(isinstance(entities, list), "type_invalid", "slides.entities must be an array", path + ".entities")
    events = slides.get("events")
    _check(isinstance(events, list), "type_invalid", "slides.events must be an array", path + ".events")
    source_id = source.get("id")
    entities_by_id = {}
    for index, entity in enumerate(entities):
        epath = f"{path}.entities[{index}]"
        _check(isinstance(entity, dict), "type_invalid", "slide entity must be an object", epath)
        deck_id = entity.get("deck_id")
        _check(
            isinstance(deck_id, str) and deck_id != "",
            "value_invalid",
            "slide entity deck_id must be a non-empty string",
            epath + ".deck_id",
        )
        page = entity.get("page")
        _check(
            _is_int(page) and page >= 1,
            "value_invalid",
            "slide entity page must be an integer >= 1",
            epath + ".page",
        )
        content_sha = entity.get("content_sha256")
        _check(
            content_sha is None
            or (isinstance(content_sha, str) and _SHA256_RE.match(content_sha)),
            "value_invalid",
            "slide entity content_sha256 must be 64 lowercase hex or null",
            epath + ".content_sha256",
        )
        title = entity.get("title")
        _check(
            title is None or isinstance(title, str),
            "type_invalid",
            "slide entity title must be a string or null",
            epath + ".title",
        )
        region = entity.get("region")
        if region is not None:
            _check(
                isinstance(region, dict),
                "type_invalid",
                "slide entity region must be an object or null",
                epath + ".region",
            )
            for key in ("x", "y", "w", "h"):
                _check(
                    _is_int(region.get(key)),
                    "type_invalid",
                    f"region.{key} must be an integer",
                    epath + f".region.{key}",
                )
            _check(
                region["x"] >= 0 and region["y"] >= 0,
                "value_invalid",
                "region x/y must be >= 0",
                epath + ".region",
            )
            _check(
                region["w"] >= 1 and region["h"] >= 1,
                "value_invalid",
                "region w/h must be >= 1",
                epath + ".region",
            )
        identity = _slide_entity_identity(entity, source_id)
        _check_object_id(entity, NAMESPACE_SLIDE_ENTITY, identity, seen_ids, epath)
        entities_by_id[entity["id"]] = entity
    events_by_id = {}
    for index, event in enumerate(events):
        vpath = f"{path}.events[{index}]"
        _check(isinstance(event, dict), "type_invalid", "slide event must be an object", vpath)
        entity_ref = event.get("entity")
        _check(
            isinstance(entity_ref, str) and entity_ref in entities_by_id,
            "reference_missing",
            "slide event entity does not match a slide entity id in this document",
            vpath + ".entity",
        )
        start = _require_ms(event.get("start_ms"), vpath + ".start_ms")
        if duration_ms is not None and start > duration_ms:
            _fail("anchor_out_of_range", "slide event start_ms exceeds source duration_ms", vpath + ".start_ms")
        end = event.get("end_ms")
        if end is not None:
            _require_ms(end, vpath + ".end_ms")
            if start > end:
                _fail("anchor_descending", "slide event start_ms must be <= end_ms", vpath)
            if duration_ms is not None and end > duration_ms:
                _fail("anchor_out_of_range", "slide event end_ms exceeds source duration_ms", vpath + ".end_ms")
        identity = _slide_event_identity(event)
        _check_object_id(event, NAMESPACE_SLIDE_EVENT, identity, seen_ids, vpath)
        events_by_id[event["id"]] = event
    return events_by_id


def _validate_units(units, segments_by_id: dict, corrections_by_id: dict, events_by_id: dict, duration_ms, seen_ids: dict) -> None:
    path = "units"
    _check(isinstance(units, list), "type_invalid", "units must be an array", path)
    collections = {
        "segment": segments_by_id,
        "correction": corrections_by_id,
        "slide_event": events_by_id,
    }
    for index, unit in enumerate(units):
        upath = f"{path}[{index}]"
        _check(isinstance(unit, dict), "type_invalid", "unit must be an object", upath)
        kind = unit.get("kind")
        _check(
            kind in UNIT_KINDS,
            "value_unsupported",
            f"unit.kind must be one of {UNIT_KINDS}",
            upath + ".kind",
        )
        title = unit.get("title")
        _check(
            title is None or (isinstance(title, str) and title.strip() != ""),
            "value_invalid",
            "unit.title must be a non-empty string or null",
            upath + ".title",
        )
        time_range = unit.get("time")
        if time_range is not None:
            _check(
                isinstance(time_range, dict),
                "type_invalid",
                "unit.time must be an object or null",
                upath + ".time",
            )
            _validate_optional_anchor_pair(
                time_range.get("start_ms"), time_range.get("end_ms"), upath + ".time", duration_ms
            )
        spans = unit.get("spans")
        _check(
            isinstance(spans, list) and len(spans) >= 1,
            "value_invalid",
            "unit.spans must be a non-empty array of evidence references",
            upath + ".spans",
        )
        seen_refs = set()
        for span_index, span in enumerate(spans):
            ref_path = f"{upath}.spans[{span_index}]"
            _check(isinstance(span, dict), "type_invalid", "span reference must be an object", ref_path)
            span_kind = span.get("kind")
            _check(
                span_kind in UNIT_SPAN_KINDS,
                "value_unsupported",
                f"unit span kind must be one of {UNIT_SPAN_KINDS}",
                ref_path + ".kind",
            )
            span_id = span.get("id")
            _check(
                isinstance(span_id, str),
                "type_invalid",
                "unit span id must be a string",
                ref_path + ".id",
            )
            collection = collections[span_kind]
            _check(
                span_id in collection,
                "reference_missing",
                f"unit span does not match a {span_kind} id in this document",
                ref_path + ".id",
            )
            if span_kind == "correction" and collection[span_id]["state"] == "rejected":
                _fail(
                    "invalid_derivation",
                    "evidence units must not cite rejected corrections",
                    ref_path + ".id",
                )
            ref_key = (span_kind, span_id)
            _check(
                ref_key not in seen_refs,
                "duplicate_reference",
                "duplicate span reference within the unit",
                ref_path,
            )
            seen_refs.add(ref_key)
        content = unit.get("content")
        _check(
            content is None or isinstance(content, dict),
            "type_invalid",
            "unit.content must be an object or null",
            upath + ".content",
        )
        identity = _unit_identity(unit)
        _check_object_id(unit, NAMESPACE_UNIT, identity, seen_ids, upath)


def _validate_checkpoints(checkpoints, source_id: str, duration_ms, seen_ids: dict) -> None:
    path = "checkpoints"
    _check(isinstance(checkpoints, list), "type_invalid", "checkpoints must be an array", path)
    for index, checkpoint in enumerate(checkpoints):
        kpath = f"{path}[{index}]"
        _check(isinstance(checkpoint, dict), "type_invalid", "checkpoint must be an object", kpath)
        stage = checkpoint.get("stage")
        _check(
            isinstance(stage, str) and stage != "",
            "value_invalid",
            "checkpoint.stage must be a non-empty string",
            kpath + ".stage",
        )
        seq = checkpoint.get("seq")
        _check(
            _is_int(seq) and seq >= 0,
            "value_invalid",
            "checkpoint.seq must be an integer >= 0",
            kpath + ".seq",
        )
        status = checkpoint.get("status")
        _check(
            status in CHECKPOINT_STATUSES,
            "status_unsupported",
            f"checkpoint.status must be one of {CHECKPOINT_STATUSES}",
            kpath + ".status",
        )
        position = checkpoint.get("position_ms")
        if position is not None:
            _require_ms(position, kpath + ".position_ms")
            if duration_ms is not None and position > duration_ms:
                _fail(
                    "anchor_out_of_range",
                    "checkpoint position exceeds source duration_ms",
                    kpath + ".position_ms",
                )
        payload_sha = checkpoint.get("payload_sha256")
        _check(
            payload_sha is None
            or (isinstance(payload_sha, str) and _SHA256_RE.match(payload_sha)),
            "value_invalid",
            "checkpoint.payload_sha256 must be 64 lowercase hex or null",
            kpath + ".payload_sha256",
        )
        detail = checkpoint.get("detail")
        _check(
            detail is None or isinstance(detail, str),
            "type_invalid",
            "checkpoint.detail must be a string or null",
            kpath + ".detail",
        )
        identity = _checkpoint_identity(checkpoint, source_id)
        _check_object_id(checkpoint, NAMESPACE_CHECKPOINT, identity, seen_ids, kpath)


def _validate_measurements(measurements, seen_ids: dict) -> None:
    path = "measurements"
    _check(isinstance(measurements, dict), "type_invalid", "measurements must be an object", path)
    tables = measurements.get("tables")
    _check(
        isinstance(tables, list),
        "type_invalid",
        "measurements.tables must be an array",
        path + ".tables",
    )
    for index, row in enumerate(tables):
        mpath = f"{path}.tables[{index}]"
        _check(isinstance(row, dict), "type_invalid", "measurement row must be an object", mpath)
        family = row.get("family")
        metric = row.get("metric")
        _check(
            isinstance(family, str) and isinstance(metric, str),
            "type_invalid",
            "measurement family/metric must be strings",
            mpath,
        )
        spec = METRIC_INDEX.get((family, metric))
        _check(spec is not None, "metric_unknown", f"unknown measurement {family}/{metric}", mpath)
        _check(
            row.get("unit") == spec["unit"],
            "value_invalid",
            f"measurement unit must be {spec['unit']!r}",
            mpath + ".unit",
        )
        _check(
            row.get("direction") == spec["direction"],
            "value_invalid",
            f"measurement direction must be {spec['direction']!r}",
            mpath + ".direction",
        )
        _check(
            row.get("definition") == spec["definition"],
            "value_invalid",
            "measurement definition must match the catalog entry",
            mpath + ".definition",
        )
        value = row.get("value")
        if value is not None:
            _check(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value),
                "value_invalid",
                "measurement value must be a finite number or null",
                mpath + ".value",
            )
        dims = row.get("dims")
        _check(isinstance(dims, dict), "type_invalid", "measurement dims must be an object", mpath + ".dims")
        for dim_key, dim_value in dims.items():
            _check(
                isinstance(dim_key, str) and dim_key != "",
                "value_invalid",
                "measurement dim keys must be non-empty strings",
                mpath + ".dims",
            )
            _check(
                isinstance(dim_value, str) and dim_value != "",
                "value_invalid",
                "measurement dim values must be non-empty strings",
                mpath + ".dims",
            )
        scope = row.get("scope")
        _check(
            scope is None or (isinstance(scope, str) and scope != ""),
            "value_invalid",
            "measurement scope must be a non-empty string or null",
            mpath + ".scope",
        )
        identity = _measurement_identity(row)
        _check_object_id(row, NAMESPACE_MEASUREMENT, identity, seen_ids, mpath)


def validate_document(document):
    """Validate an evidence.v1 document and return its canonical normal form.

    Raises :class:`EvidenceContractError` with a stable ``code`` on the first
    violation.  The input is never mutated.
    """
    doc = normalize_document(document)
    _scan_secrets(doc, "$")
    seen_ids = {}
    source = doc.get("source")
    _validate_source(source, seen_ids)
    fingerprints = doc["fingerprints"]
    _validate_fingerprints(fingerprints)
    duration_ms = source.get("duration_ms")
    segments_by_id = _validate_speech(doc["speech"], source, fingerprints, duration_ms, seen_ids)
    corrections_by_id = _validate_corrections(doc["corrections"], segments_by_id, duration_ms, seen_ids)
    _validate_cues(doc["cues"], segments_by_id, corrections_by_id, duration_ms, seen_ids)
    events_by_id = _validate_slides(doc["slides"], source, duration_ms, seen_ids)
    _validate_units(
        doc["units"], segments_by_id, corrections_by_id, events_by_id, duration_ms, seen_ids
    )
    _validate_checkpoints(doc["checkpoints"], source["id"], duration_ms, seen_ids)
    _validate_measurements(doc["measurements"], seen_ids)
    return doc


def assign_ids(document):
    """Fill every object id from its identity fields, then validate.

    Fixture generation and tests use this to obtain canonical IDs.  Objects
    already carrying IDs are re-stamped; validation never invents or mutates
    IDs, it only verifies them.
    """
    doc = normalize_document(document)
    source = doc.get("source")
    if isinstance(source, dict):
        source["id"] = compute_id(NAMESPACE_SOURCE, _source_identity(source))
        source_id = source["id"]
        fingerprints = doc.get("fingerprints")
        fingerprint_scope = _fingerprint_scope(
            fingerprints if isinstance(fingerprints, dict) else {}
        )
        speech = doc.get("speech")
        if isinstance(speech, dict) and isinstance(speech.get("segments"), list):
            for segment in speech["segments"]:
                if isinstance(segment, dict):
                    segment["id"] = compute_id(
                        NAMESPACE_SEGMENT,
                        _segment_identity(segment, source_id, fingerprint_scope),
                    )
        corrections = doc.get("corrections")
        if isinstance(corrections, list):
            for correction in corrections:
                if isinstance(correction, dict):
                    correction["id"] = compute_id(
                        NAMESPACE_CORRECTION, _correction_identity(correction)
                    )
        cues = doc.get("cues")
        if isinstance(cues, list):
            for cue in cues:
                if isinstance(cue, dict):
                    cue["id"] = compute_id(NAMESPACE_CUE, _cue_identity(cue))
        slides = doc.get("slides")
        if isinstance(slides, dict):
            entities = slides.get("entities")
            if isinstance(entities, list):
                for entity in entities:
                    if isinstance(entity, dict):
                        entity["id"] = compute_id(
                            NAMESPACE_SLIDE_ENTITY, _slide_entity_identity(entity, source_id)
                        )
            events = slides.get("events")
            if isinstance(events, list):
                for event in events:
                    if isinstance(event, dict):
                        event["id"] = compute_id(
                            NAMESPACE_SLIDE_EVENT, _slide_event_identity(event)
                        )
        units = doc.get("units")
        if isinstance(units, list):
            for unit in units:
                if isinstance(unit, dict):
                    unit["id"] = compute_id(NAMESPACE_UNIT, _unit_identity(unit))
        checkpoints = doc.get("checkpoints")
        if isinstance(checkpoints, list):
            for checkpoint in checkpoints:
                if isinstance(checkpoint, dict):
                    checkpoint["id"] = compute_id(
                        NAMESPACE_CHECKPOINT, _checkpoint_identity(checkpoint, source_id)
                    )
        measurements = doc.get("measurements")
        if isinstance(measurements, dict) and isinstance(measurements.get("tables"), list):
            for row in measurements["tables"]:
                if isinstance(row, dict):
                    row["id"] = compute_id(NAMESPACE_MEASUREMENT, _measurement_identity(row))
    return validate_document(doc)


def to_json(document) -> str:
    """Canonical JSON representation of a validated document."""
    return canonical_json(validate_document(document))


def from_json(text):
    """Parse strict JSON text and validate it as an evidence.v1 document."""
    def _reject_constant(name):
        _fail("value_invalid", f"{name} is not valid JSON data", "")

    def _no_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("duplicate_key", f"duplicate JSON key {key!r}", "")
            result[key] = value
        return result

    try:
        document = json.loads(
            text, parse_constant=_reject_constant, object_pairs_hook=_no_duplicate_keys
        )
    except json.JSONDecodeError as error:
        _fail("value_invalid", f"invalid JSON: {error.msg}", "")
    return validate_document(document)
