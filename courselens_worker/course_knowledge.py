"""有界、引用闭包的多源 evidence packet 摄入（worker 侧）。

客户端（N7K）为每讲构建 evidence packet：字幕关键窗口、PPT 页、课次/课程级
文档页、Assessment IR。本模块只做两件事：

1. 校验并规范化 packet —— 引用 ID 闭集、定位键闭集、content_hash 与正文一致、
   课程绑定、条目/字符硬上限；不合规条目逐条丢弃并记闭集原因码，绝不静默放行。
2. 校验模型返回的 knowledge point —— 只能引用包内 citation，越界/空引用/超长
   一律整条拒绝。

包内一切文本都当「数据」对待：文档与题目正文是不可信证据，不是指令；模型只被
允许回合同字段与包内 citation。

## 为什么不 import shared.course_knowledge_contract

worker 走的是"发布镜像"链：`scripts/worker_mirror_allowlist.json` 是显式文件清单，
新合同文件不在其中，import 它会让线上 worker 直接 ImportError。因此这里**只镜像**
合同的闭集常量与 `compute_id` 算法（各 3-12 行），并由
`worker/tests/test_course_knowledge.py` 直接 import 真合同断言两者逐项等价——
运行时零依赖，漂移由测试当场抓住。

## 与 N7K 的接口期望（写入合同的引用必须能被本 job 验证）

- `transcript` / `slide` 条目可以**不带 text**：本地按 `source_id`（`seg:` /
  `slevt|slent:`）从本 job 自己的 transcript / ppt_pages 里取回正文并核对 hash。
  取不回正文的引用整条丢弃（`reference_missing`）——模型不能引用它没看过的内容。
- `document_page` / `assessment_item` / `bookmark` 条目**必须带 text**：这些正文
  不在 job payload 里，只能由包提供。
- 条目形状二选一：`{"ref": {...7键...}, "text": "..."}` 或 7 键平铺在条目上再加
  `text`。`ref` 内多出的键会被忽略，`ref` 的闭集与合同一致。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

CONTRACT_ID = "courselens.course-knowledge.v1"

# 镜像 shared/course_knowledge_contract.py 的闭集（测试钉等价）。
EVIDENCE_KINDS = ("transcript", "slide", "document_page", "assessment_item", "bookmark")
DOCUMENT_KINDS = ("document_page", "assessment_item", "bookmark")
TIME_KINDS = ("transcript", "slide")
NAMESPACE_CITATION = "ckc"
LOCATOR_FIELDS = {
    "transcript": ("start_ms", "end_ms"),
    "bookmark": ("start_ms", "end_ms"),
    "slide": ("page",),
    "document_page": ("page",),
    "assessment_item": ("question_no",),
}
CONTENT_HASH_RE = re.compile(r"^[0-9a-f]{32,64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{12,64}$")
CITATION_ID_RE = re.compile(r"^ckc:[0-9a-f]{12}$")
SEGMENT_ID_RE = re.compile(r"^seg:[0-9a-f]{12}$")
SLIDE_ID_RE = re.compile(r"^(slevt|slent):[0-9a-f]{12}$")

# 硬上限：包大就截断并记 coverage，绝不无界送云。
MAX_PACKET_ITEMS = 48
MAX_PACKET_CHARS = 48_000
MAX_ITEM_CHARS = 2_000
MAX_ITEMS_PER_WINDOW = 8
MAX_WINDOW_CHARS = 6_000
MAX_COURSE_CONTEXT_KEYS = 16
MAX_COURSE_CONTEXT_CHARS = 200

MAX_KNOWLEDGE_POINTS = 24
MAX_TOPIC_CANDIDATES = 12
MAX_CITATION_IDS_PER_POINT = 8
MAX_POINT_TITLE_CHARS = 60
MAX_POINT_TEXT_CHARS = 400
MAX_TOPIC_CHARS = 40
MAX_REJECT_SAMPLES = 16

# 闭集拒绝原因码。前 8 个与合同的错误码同名同义，后 3 个是本模块的边界判定。
REJECT_CODES = (
    "not-a-dict",
    "kind_unsupported",
    "cross_course_reference",
    "field_required",
    "hash_malformed",
    "id_malformed",
    "id_mismatch",
    "reference_missing",
    "duplicate_citation",
    "empty_text",
    "value_invalid",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _compute_id(namespace: str, identity: dict[str, Any]) -> str:
    payload = _canonical_json(identity).encode("utf-8")
    return f"{namespace}:{hashlib.sha256(payload).hexdigest()[:12]}"


def citation_id_for(ref: dict[str, Any]) -> str:
    """镜像合同的 citation_id 派生（测试与 `citation_id_for` 逐项比对）。"""
    return _compute_id(NAMESPACE_CITATION, {
        "kind": ref.get("kind"),
        "source_id": ref.get("source_id"),
        "revision_id": ref.get("revision_id"),
        "content_hash": ref.get("content_hash"),
        "locator": ref.get("locator"),
    })


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _hash_matches(content_hash: str, text: str) -> bool:
    """content_hash 必须落在正文 sha256 的对应长度前缀上。"""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[: len(content_hash)] == content_hash or digest == content_hash


def _reference_index(transcript: list[dict[str, Any]] | None,
                     ppt_pages: list[dict[str, Any]] | None) -> dict[str, str]:
    """本 job 已持有的可核对正文：seg/slevt/slent ID → 文本。"""
    index: dict[str, str] = {}
    for item in transcript or []:
        if not isinstance(item, dict):
            continue
        segment_id = item.get("segment_id")
        text = item.get("text")
        if isinstance(segment_id, str) and isinstance(text, str) and text.strip():
            index.setdefault(segment_id, text)
    for item in ppt_pages or []:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        for key in ("event_id", "entity_id"):
            value = item.get(key)
            if isinstance(value, str) and value:
                index.setdefault(value, text)
    return index


class _Rejector:
    """按闭集原因码逐条记账；样本有上限，计数无上限。"""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.samples: list[dict[str, str]] = []

    def __call__(self, label: str, code: str) -> None:
        self.counts[code] = self.counts.get(code, 0) + 1
        if len(self.samples) < MAX_REJECT_SAMPLES:
            self.samples.append({"item": str(label)[:64], "reason": code})

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _empty_packet(course_id: str, sub_id: str, *, provided: bool) -> dict[str, Any]:
    return {
        "contract": CONTRACT_ID,
        "course_id": course_id,
        "sub_id": sub_id,
        "items": [],
        "citations": {},
        "coverage": {},
        "dropped": {"items": 0, "chars": 0, "reasons": {}, "truncated_items": 0},
        "rejected": [],
        "rejected_counts": {},
        "references_resolved": 0,
        "provided": provided,
        "usable": False,
    }


def _packet_ref(entry: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """取出条目里的引用与正文；两种约定形状都接受。"""
    nested = entry.get("ref")
    if isinstance(nested, dict):
        ref = dict(nested)
        text = entry.get("text")
    else:
        allowed = {"citation_id", "kind", "source_id", "revision_id", "content_hash",
                   "locator", "label", "course_id"}
        ref = {key: value for key, value in entry.items() if key in allowed}
        text = entry.get("text")
    if not isinstance(text, str) or not text.strip():
        text = entry.get("excerpt")  # 客户端 evidence_packet 用 excerpt 装正文
    return ref, (text if isinstance(text, str) else "")


def _validate_ref(ref: dict[str, Any], course_id: str, reject: _Rejector,
                  label: str) -> dict[str, Any] | None:
    """按合同闭集校验一个引用；返回规范化引用或 None。

    包可能只带引用**视图**（citation_id/kind/locator/label + 摘要），也可能带
    完整身份（source_id/revision_id/content_hash）。两种都收：出现任一身份字段
    就要求三件齐备（半套身份是坏包），齐备时才做 hash 与派生核对。
    """
    kind = ref.get("kind")
    if kind not in EVIDENCE_KINDS:
        reject(label, "kind_unsupported")
        return None
    ref_course = ref.get("course_id")
    if ref_course not in (None, "") and str(ref_course) != str(course_id):
        reject(label, "cross_course_reference")
        return None
    label_value = ref.get("label")
    if label_value is not None and not isinstance(label_value, str):
        reject(label, "value_invalid")
        return None
    locator = ref.get("locator")
    if not isinstance(locator, dict):
        reject(label, "field_required")
        return None
    anchor_keys = LOCATOR_FIELDS[kind]
    for key in locator:
        if key not in anchor_keys:
            reject(label, "value_invalid")
            return None
    canonical_locator: dict[str, int] = {}
    for key in anchor_keys:
        value = locator.get(key)
        if not _is_int(value):
            reject(label, "field_required")
            return None
        canonical_locator[key] = value
    if "start_ms" in canonical_locator:
        if canonical_locator["start_ms"] < 0 or canonical_locator["start_ms"] > canonical_locator["end_ms"]:
            reject(label, "value_invalid")
            return None
    elif "page" in canonical_locator and canonical_locator["page"] < 1:
        reject(label, "value_invalid")
        return None
    elif "question_no" in canonical_locator and canonical_locator["question_no"] < 1:
        reject(label, "value_invalid")
        return None
    source_id = ref.get("source_id")
    revision_id = ref.get("revision_id")
    content_hash = ref.get("content_hash")
    full_identity = any(value not in (None, "") for value in (source_id, revision_id, content_hash))
    canonical: dict[str, Any] = {
        "kind": kind,
        "locator": canonical_locator,
        "label": label_value if isinstance(label_value, str) else "",
    }
    if full_identity:
        if not source_id or not revision_id or not content_hash:
            reject(label, "field_required")
            return None
        if kind == "transcript" and not isinstance(source_id, str):
            reject(label, "reference_missing")
            return None
        if isinstance(source_id, str) and kind == "transcript" and not SEGMENT_ID_RE.match(source_id):
            reject(label, "reference_missing")
            return None
        if isinstance(source_id, str) and kind == "slide" and not SLIDE_ID_RE.match(source_id):
            reject(label, "reference_missing")
            return None
        if not isinstance(source_id, str) or len(source_id) > 128:
            reject(label, "value_invalid")
            return None
        if not isinstance(revision_id, str) or not REVISION_RE.match(revision_id):
            reject(label, "hash_malformed")
            return None
        if not isinstance(content_hash, str) or not CONTENT_HASH_RE.match(content_hash):
            reject(label, "hash_malformed")
            return None
        canonical.update({
            "source_id": source_id, "revision_id": revision_id,
            "content_hash": content_hash,
        })
        derived = citation_id_for(canonical)
        canonical["citation_id"] = derived
        canonical["identity"] = "full"
        provided = ref.get("citation_id")
        if provided not in (None, ""):
            if not isinstance(provided, str) or not CITATION_ID_RE.match(provided):
                reject(str(provided)[:64], "id_malformed")
                return None
            if provided != derived:
                reject(provided, "id_mismatch")
                return None
        return canonical
    # 视图形态：只有 citation_id 可用，无法复核派生（不假装核过）。
    provided = ref.get("citation_id")
    if not isinstance(provided, str) or not CITATION_ID_RE.match(provided):
        reject(str(provided or "")[:64], "id_malformed")
        return None
    canonical.update({"citation_id": provided, "identity": "view"})
    return canonical


def _packet_items(packet: dict[str, Any]) -> list[Any]:
    """条目列表：兼容 ``items`` 与客户端现产的 ``evidence``。"""
    for key in ("items", "evidence"):
        value = packet.get(key)
        if isinstance(value, list):
            return value
    return []


def normalize_evidence_packet(
    packet: Any,
    *,
    course_id: str = "",
    sub_id: str = "",
    transcript: list[dict[str, Any]] | None = None,
    ppt_pages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """校验并规范化一个 evidence packet。

    整体不可用时返回 ``usable=False`` 的空包（调用方据此降级到旧 summary）；
    单条不合规则逐条丢弃并记原因码，其余条目照常可用。
    """
    expected_course = str(course_id or "").strip()
    expected_sub = str(sub_id or "").strip()
    if not isinstance(packet, dict) or packet.get("contract") != CONTRACT_ID:
        return _empty_packet(expected_course, expected_sub, provided=isinstance(packet, dict))
    packet_course = str(packet.get("course_id") or "").strip()
    if not packet_course or (expected_course and packet_course != expected_course):
        reject = _Rejector()
        reject("packet", "cross_course_reference")
        value = _empty_packet(expected_course, expected_sub, provided=True)
        value["rejected"] = reject.samples
        value["rejected_counts"] = reject.counts
        return value
    references = _reference_index(transcript, ppt_pages)
    reject = _Rejector()
    items: list[dict[str, Any]] = []
    citations: dict[str, dict[str, Any]] = {}
    coverage: dict[str, dict[str, int]] = {}
    seen_identity: set[tuple[str, str]] = set()
    total_chars = 0
    dropped_items = 0
    dropped_chars = 0
    references_resolved = 0
    integrity_verified = 0
    integrity_unavailable = 0

    for entry in _packet_items(packet):
        if not isinstance(entry, dict):
            reject("", "not-a-dict")
            continue
        ref, text = _packet_ref(entry)
        label = str(ref.get("source_id") or ref.get("citation_id") or "")
        canonical = _validate_ref(ref, packet_course, reject, label)
        if canonical is None:
            continue
        kind = canonical["kind"]
        resolved = text.strip()
        resolved_locally = False
        if canonical["identity"] == "full":
            # 完整身份：字幕/幻灯的正文本地可取，且能当场核对 hash。
            if not resolved and kind in TIME_KINDS:
                resolved = str(references.get(canonical["source_id"]) or "").strip()
                resolved_locally = True
            if not resolved:
                reject(canonical["citation_id"],
                       "reference_missing" if kind in TIME_KINDS else "empty_text")
                continue
            if not _hash_matches(canonical["content_hash"], resolved):
                reject(canonical["citation_id"], "hash_malformed")
                continue
            if resolved_locally:
                references_resolved += 1
            integrity_verified += 1
        else:
            # 视图形态：正文就是包给的摘要，无法核对 hash（如实计数，不假称已核）。
            if not resolved:
                reject(canonical["citation_id"], "empty_text")
                continue
            integrity_unavailable += 1
        if len(resolved) > MAX_ITEM_CHARS:
            resolved = resolved[:MAX_ITEM_CHARS]
        identity = (canonical["identity"], canonical["citation_id"])
        if canonical["citation_id"] in citations or identity in seen_identity:
            reject(canonical["citation_id"], "duplicate_citation")
            continue
        if len(items) >= MAX_PACKET_ITEMS or total_chars + len(resolved) > MAX_PACKET_CHARS:
            dropped_items += 1
            dropped_chars += len(resolved)
            continue
        seen_identity.add(identity)
        item = {
            "citation_id": canonical["citation_id"],
            "kind": kind,
            "source_id": canonical.get("source_id", ""),
            "revision_id": canonical.get("revision_id", ""),
            "content_hash": canonical.get("content_hash", ""),
            "locator": canonical["locator"],
            "label": canonical["label"],
            "text": resolved,
            "identity": canonical["identity"],
        }
        items.append(item)
        citations[item["citation_id"]] = item
        total_chars += len(resolved)
        bucket = coverage.setdefault(kind, {"items": 0, "chars": 0})
        bucket["items"] += 1
        bucket["chars"] += len(resolved)
    packet_dropped = packet.get("dropped") if isinstance(packet.get("dropped"), dict) else {}
    packet_dropped_items = packet_dropped.get("items") if _is_int(packet_dropped.get("items")) else 0
    packet_dropped_chars = packet_dropped.get("chars") if _is_int(packet_dropped.get("chars")) else 0
    packet_reasons = packet_dropped.get("reasons") if isinstance(packet_dropped.get("reasons"), dict) else {}
    return {
        "contract": CONTRACT_ID,
        "course_id": packet_course,
        "sub_id": expected_sub,
        "items": items,
        "citations": citations,
        "coverage": coverage,
        "dropped": {
            "items": dropped_items + int(packet_dropped_items or 0),
            "chars": dropped_chars + int(packet_dropped_chars or 0),
            "truncated_items": dropped_items,
            "reasons": {str(key)[:40]: int(value) for key, value in packet_reasons.items()
                        if _is_int(value)},
        },
        "rejected": reject.samples,
        "rejected_counts": reject.counts,
        "references_resolved": references_resolved,
        "integrity_verified": integrity_verified,
        "integrity_unavailable": integrity_unavailable,
        "provided": True,
        "usable": bool(items),
    }


def packet_windows(packet: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """把非时间锚定条目（文档页/题目/书签）切成有界窗口。

    字幕/幻灯条目不进窗口：它们的正文已在既有 transcript/ppt 窗口里投喂过，
    重复加载只是多花 token。
    """
    windows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    chars = 0
    for item in packet.get("items") or []:
        if item.get("kind") not in DOCUMENT_KINDS:
            continue
        size = len(str(item.get("text") or ""))
        if current and (len(current) >= MAX_ITEMS_PER_WINDOW or chars + size > MAX_WINDOW_CHARS):
            windows.append(current)
            current = []
            chars = 0
        current.append(item)
        chars += size
    if current:
        windows.append(current)
    return windows


def evidence_index(packet: dict[str, Any]) -> list[dict[str, Any]]:
    """给合并调用的引用索引：只给身份与位置，让模型能按 citation_id 引用。"""
    index = []
    for item in packet.get("items") or []:
        index.append({
            "citation_id": item["citation_id"],
            "kind": item["kind"],
            "label": item["label"],
            "locator": dict(item.get("locator") or {}),
        })
    return index


def validate_course_context(value: Any) -> dict[str, str]:
    """课程上下文透传：只留短标量，限量限长，不引入第二套结构。"""
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for key, item in value.items():
        if len(result) >= MAX_COURSE_CONTEXT_KEYS:
            break
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            continue
        text = str(item).strip()[:MAX_COURSE_CONTEXT_CHARS]
        if text:
            result[str(key)[:40]] = text
    return result


def validate_knowledge_points(
    value: Any,
    citations: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """校验模型返回的 knowledge points；只保留全部引用都在包内的条目。"""
    accepted: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    written: set[str] = set()

    def reject(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    if not isinstance(value, list):
        return [], {"rejected": 0, "reasons": {}, "not_a_list": True}
    for entry in value:
        if not isinstance(entry, dict):
            reject("not-a-dict")
            continue
        title = str(entry.get("title") or "").strip()
        text = str(entry.get("text") or entry.get("summary") or "").strip()
        raw_ids = entry.get("citation_ids")
        if not isinstance(raw_ids, list):
            raw_ids = entry.get("citations")
        if not title:
            reject("empty-title")
            continue
        if not text:
            reject("empty-text")
            continue
        if not isinstance(raw_ids, list) or not raw_ids:
            reject("no-citation")
            continue
        known: list[str] = []
        unknown = False
        for raw_id in raw_ids:
            citation_id = str(raw_id)
            if citation_id in citations:
                if citation_id not in known:
                    known.append(citation_id)
            else:
                unknown = True
        # 引用了包外 citation 就整条拒绝：只留合法引用会把「基于不存在证据的
        # 主张」洗成有据可查，这是 fail-closed 必须挡住的一步。
        if unknown:
            reject("unknown-citation")
            continue
        if not known:
            reject("no-citation")
            continue
        known = known[:MAX_CITATION_IDS_PER_POINT]
        bounded_title = title[:MAX_POINT_TITLE_CHARS]
        bounded_text = text[:MAX_POINT_TEXT_CHARS]
        dedupe = f"{bounded_title}|{bounded_text}"
        if dedupe in written:
            reject("duplicate")
            continue
        written.add(dedupe)
        point: dict[str, Any] = {
            "title": bounded_title,
            "text": bounded_text,
            "citation_ids": known,
        }
        if entry.get("conflict") is True:
            # 冲突并列是诚实输出：两处证据说法不同就如实标注，不做裁决。
            point["conflict"] = True
        accepted.append(point)
        if len(accepted) >= MAX_KNOWLEDGE_POINTS:
            break
    return accepted, {"rejected": sum(reasons.values()), "reasons": reasons, "not_a_list": False}


def validate_topic_candidates(value: Any) -> list[str]:
    """主题候选：只要非空短字符串，去重、限量；非字符串条目按形状错误丢弃。"""
    result: list[str] = []
    if not isinstance(value, list):
        return result
    for entry in value:
        if not isinstance(entry, str):
            continue
        text = entry.strip()[:MAX_TOPIC_CHARS]
        if not text or text in result:
            continue
        result.append(text)
        if len(result) >= MAX_TOPIC_CANDIDATES:
            break
    return result


def coverage_summary(packet: dict[str, Any]) -> dict[str, Any]:
    """来源覆盖：投喂了什么、丢了多少，一并如实上报。"""
    return {
        "items": len(packet.get("items") or []),
        "kinds": {kind: dict(bucket) for kind, bucket in sorted((packet.get("coverage") or {}).items())},
        "dropped": dict(packet.get("dropped") or {}),
        "rejected": dict(packet.get("rejected_counts") or {}),
    }
