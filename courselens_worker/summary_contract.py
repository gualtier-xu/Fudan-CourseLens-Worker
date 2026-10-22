"""V010-LEARN-IMPL-P0-1 期 0 合同冻结（LEARN-D2 §4.6 数据契约包；shared
contract gate，权=product-v010learn-d2-result-20261021.md）。

三契约的 worker 侧加法演进——schema 冻结唯一权威=本模块闭集常量与校验函数：

1. timestamp_summary v3：加法 ``schema_ver=3`` 标记 + 可选 ``doc`` 字段（完整
   总结文档承载：overview/sections/terms/summary_notes）。v2 既有字段
   （markdown/chapters/key_takeaways/assessment_events/takeaway_anchors 等）
   字段名与结构零变化；旧消费者按「未知字段忽略」降级纪律零感知。
2. chapters_v2 板块：大板块切片数据结构（aggregation+items[source 溯源]）。
   板块对齐闭集=首块 0:00、≥3 块、每块 ≥180s（5-20 分钟目标、3 分钟地板）、
   回翻不成块（start_ms 严格递增）；不合闭集=整键省略（前端走旧 chapters，
   绝不半残）。期 0 只落 semantic_fallback 路径（幻灯片检测 slide_alignment
   源另立 worker 包，LEARN-D2 §5 风险 4）；source 闭集已含 slides/upload
   为该包预留。
3. events confidence/source：assessment_events 每项加法增 ``confidence``
   （high|low 确证分级）与 ``source``（transcription|manual 来源标注）；
   ``end_ms?`` 可选窗为校验器保留（期 0 合并链引句无时间窗，缺省合法）。

confidence 语义（两载体同一纪律，期 2 消费勿混淆）：
- ``high``=三证齐（规则命中+引句证据+句簇时间戳，雷达规则道），或用户手建
  （manual，期 2 预留）；
- ``low``=LLM 合并链候选（引句已验原文但分类未经规则确证）——与雷达
  source=llm→status=unconfirmed 的既有展示纪律同轴（B8 低置信候选不上轴）。

坏形状降级纪律（LEARN-D2 §5 风险 2）：doc/chapters_v2 任何形状不合=不产
字段→前端走旧态诚实呈现，绝不渲染半残文档。全部纯函数，零 I/O 零网络。

V010-CONTENT-IMPL-1（CONTENT-R1 兑现道，2026-10-22）三期 1 增量：
1. 章节数量纪律 ``enforce_chapter_count``（CONTENT-R1 实测 44 例 chapters
   0-59 失控）：按讲次时长比例区间帽（≥3 地板、20 帽=同族产品实证常数），
   超限按时间近邻合并（最小 start 间隙、并列取最左），≤帽原样透传（幂等）。
2. 边界感知贪心（CONTENT-R1 C1）：build_chapters_v2 的 5 分钟定时闭合改为
   ±2 分钟浮动窗内「文本重合度最低点」闭合——字符二元组 Dice 相似度
   （零新模型零新 LLM 调用），仍守 3 分钟地板/20 分钟帽/首块 0:00；
   浮动窗内无候选（稀疏转写大间隙）=回退旧定时贪心；并列取离目标最近。
3. LLM 结构化 doc 增量 ``coerce_llm_doc``（期 0 预留的「期 1 LLM 结构化
   输出」）：合并链 doc 候选逐点锚白名单收敛+逐段清洗后过
   validate_summary_doc 同一道门；逐级丢弃（点→段→整文档），全不可用=
   None（调用方走本地 build_summary_doc 兜底），落键形状恒过门=禁半残。
"""

from __future__ import annotations

from typing import Any

# ---- 期 0 schema 冻结闭集（唯一权威；schema 变更走协议包流程）--------------
SCHEMA_VER = 3
DOC_SECTION_SOURCES = ("slides", "upload", "semantic")
DOC_AGGREGATIONS = ("slide_alignment", "semantic_fallback")
EVENT_CONFIDENCES = ("high", "low")
EVENT_SOURCES = ("transcription", "manual")

# 板块对齐闭集（LEARN-D2 §4.6：5-20 分钟目标、3 分钟地板）。
_BLOCK_FLOOR_MS = 180_000   # 3 分钟地板（硬门：任一块不足=降级省键）
_BLOCK_TARGET_MS = 300_000  # 5 分钟切片目标（贪心闭合阈值）
_BLOCK_CAP_MS = 1_200_000   # 20 分钟目标帽（软目标超限提前闭合，不硬门）
# V010-CONTENT-IMPL-1（CONTENT-R1 C1 边界感知贪心）：浮动窗=目标时刻 ±2 分钟，
# 窗内取文本重合度最低点闭合（零新调用）；窗下界恰为 3 分钟地板（切点永不
# 破地板），上界钳在 20 分钟帽内（浮动延伸不吃帽）。
_BLOCK_FLOAT_MS = 120_000
# 章节数量纪律（CONTENT-R1 B2 实证 0-59 失控）：帽=时长//地板与常数帽取小。
_CHAPTER_COUNT_CAP = 20
_CHAPTER_MERGED_SUMMARY_CAP = 500
_DOC_MIN_SECTIONS = 3       # ≥3 块硬门
_MAX_BLOCKS = 500

_OVERVIEW_MAX_PARAS = 4
_OVERVIEW_PARA_CAP = 400
_TITLE_CAP = 120
_SUMMARY_CAP = 2_000
_POINT_CAP = 500
_TERM_CAP = 64
_NOTE_CAP = 120
_NOTES_MAX = 5  # 合同「3-5 条小结」；key_takeaways(3-6) 头部截取


def _clean_text(value: Any, cap: int) -> str:
    return str(value or "").strip()[:cap]


def _valid_ms(value: Any) -> int | None:
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return None
    return ms if ms >= 0 else None


# ---- doc 校验（期 1 LLM 结构化输出与本模块本地构建共用的同一道门）----------


def _validate_doc_section(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    start_ms = _valid_ms(item.get("start_ms"))
    title = _clean_text(item.get("title"), _TITLE_CAP)
    source = str(item.get("source") or "")
    if start_ms is None or not title or source not in DOC_SECTION_SOURCES:
        return None
    raw_points = item.get("points")
    if not isinstance(raw_points, list):
        return None
    points: list[dict[str, Any]] = []
    for point in raw_points[:64]:
        if not isinstance(point, dict):
            return None
        text = _clean_text(point.get("text"), _POINT_CAP)
        anchor = _valid_ms(point.get("anchor_ms"))
        if not text or anchor is None:
            return None
        points.append({"text": text, "anchor_ms": anchor})
    if not points:
        return None
    raw_formulas = item.get("formulas")
    if raw_formulas is None:
        raw_formulas = []
    if not isinstance(raw_formulas, list):
        return None
    formulas: list[dict[str, Any]] = []
    for formula in raw_formulas[:64]:
        if not isinstance(formula, dict):
            return None
        latex = _clean_text(formula.get("latex"), 300)
        if not latex:
            return None
        entry: dict[str, Any] = {"latex": latex}
        anchor = _valid_ms(formula.get("anchor_ms"))
        if anchor is not None:
            entry["anchor_ms"] = anchor
        caption = _clean_text(formula.get("caption"), 200)
        if caption:
            entry["caption"] = caption
        formulas.append(entry)
    section: dict[str, Any] = {
        "start_ms": start_ms,
        "title": title,
        "source": source,
        "points": points,
        "formulas": formulas,
    }
    raw_example = item.get("example")
    if raw_example is not None:
        if not isinstance(raw_example, dict):
            return None
        example_text = _clean_text(raw_example.get("text"), _POINT_CAP)
        if not example_text:
            return None
        example: dict[str, Any] = {"text": example_text}
        anchor = _valid_ms(raw_example.get("anchor_ms"))
        if anchor is not None:
            example["anchor_ms"] = anchor
        section["example"] = example
    return section


def validate_summary_doc(value: Any) -> dict[str, Any] | None:
    """doc 字段闭集校验：形状不合返回 None（坏形状降级=不产 doc 字段）。

    overview 接受单字符串或纯文本段落数组（LEARN-D2 §4.6 合同两形态）；
    sections 非空且逐块过形状门（板级闭集由 chapters_v2 校验器承担，doc 侧
    只保文档结构完整）。terms/summary_notes 缺省=空列表（诚实空态）。"""
    if not isinstance(value, dict):
        return None
    raw_overview = value.get("overview")
    if isinstance(raw_overview, str):
        overview: str | list[str] = raw_overview.strip()[:8_000]
        if not overview:
            return None
    elif isinstance(raw_overview, list):
        paras = [
            _clean_text(par, _OVERVIEW_PARA_CAP)
            for par in raw_overview[:_OVERVIEW_MAX_PARAS]
            if isinstance(par, str) and str(par).strip()
        ]
        if not paras:
            return None
        overview = paras
    else:
        return None
    raw_sections = value.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        return None
    sections: list[dict[str, Any]] = []
    for item in raw_sections[:_MAX_BLOCKS]:
        section = _validate_doc_section(item)
        if section is None:
            return None
        sections.append(section)
    raw_terms = value.get("terms")
    if raw_terms is None:
        raw_terms = []
    if not isinstance(raw_terms, list):
        return None
    terms: list[dict[str, Any]] = []
    for term in raw_terms[:_TERM_CAP]:
        if not isinstance(term, dict):
            return None
        name = _clean_text(term.get("name"), 60)
        if not name:
            return None
        terms.append({
            "symbol": _clean_text(term.get("symbol"), 60),
            "name": name,
            "note": _clean_text(term.get("note"), 200),
        })
    raw_notes = value.get("summary_notes")
    if raw_notes is None:
        raw_notes = []
    if not isinstance(raw_notes, list):
        return None
    notes = [
        _clean_text(note, _NOTE_CAP)
        for note in raw_notes[:8]
        if isinstance(note, str) and str(note).strip()
    ]
    if len(notes) != len(raw_notes[:8]):
        return None
    return {
        "overview": overview,
        "sections": sections,
        "terms": terms,
        "summary_notes": notes,
    }


# ---- chapters_v2 板块校验（板级对齐闭集在此硬门）---------------------------


def validate_chapters_v2(value: Any) -> dict[str, Any] | None:
    """chapters_v2 板块闭集校验：首块 0:00、≥3 块、块间 ≥180s（3 分钟地板）、
    回翻不成块（start_ms 严格递增）。任一不合=None（整键省略，旧 chapters
    兜底）。末块时长无下一块边界可证=豁免地板检查（构建器以转写尾探针兜底）。"""
    if not isinstance(value, dict):
        return None
    aggregation = str(value.get("aggregation") or "")
    if aggregation not in DOC_AGGREGATIONS:
        return None
    raw_items = value.get("items")
    if not isinstance(raw_items, list):
        return None
    items: list[dict[str, Any]] = []
    for item in raw_items[:_MAX_BLOCKS]:
        if not isinstance(item, dict):
            return None
        title = _clean_text(item.get("title"), _TITLE_CAP)
        start_ms = _valid_ms(item.get("start_ms"))
        source = str(item.get("source") or "")
        if not title or start_ms is None or source not in DOC_SECTION_SOURCES:
            return None
        items.append({
            "title": title,
            "start_ms": start_ms,
            "summary": _clean_text(item.get("summary"), _SUMMARY_CAP),
            "source": source,
        })
    if len(items) < _DOC_MIN_SECTIONS:
        return None
    if items[0]["start_ms"] != 0:
        return None
    previous = items[0]["start_ms"]
    for item in items[1:]:
        if item["start_ms"] <= previous:  # 回翻/同刻=不成块
            return None
        if item["start_ms"] - previous < _BLOCK_FLOOR_MS:
            return None
        previous = item["start_ms"]
    return {"aggregation": aggregation, "items": items}


# ---- 章节边界感知贪心（V010-CONTENT-IMPL-1，CONTENT-R1 C1）-----------------


def _char_bigrams(text: str) -> set[str]:
    cleaned = "".join(ch for ch in str(text or "").lower() if ch.isalnum())
    return {cleaned[index:index + 2] for index in range(len(cleaned) - 1)}


def _dice_similarity(left: str, right: str) -> float:
    left_grams = _char_bigrams(left)
    right_grams = _char_bigrams(right)
    if not left_grams or not right_grams:
        return 0.0
    return 2.0 * len(left_grams & right_grams) / (len(left_grams) + len(right_grams))


def _boundary_similarity(segments: list[tuple[int, int, str]], index: int) -> float:
    """候选切点的相邻文本重合度：切点前最多 2 段 vs 切点起最多 2 段（各截
    200 字）。最低=话题边界最强；与 llm._takeaway_anchor_ms 同族字符二元组
    Dice（本模块零依赖 llm，独立同式实现）。"""
    before = "".join(part[2] for part in segments[max(0, index - 2):index])[-200:]
    after = "".join(part[2] for part in segments[index:index + 2])[:200]
    return _dice_similarity(before, after)


def _find_block_boundary(
    segments: list[tuple[int, int, str]], index: int, block_start: int,
) -> int | None:
    """下一块起点=浮动窗 [目标−2min, min(目标+2min, 20 分钟帽)] 内文本重合度
    最低的段起点（并列取离目标最近、再取最早）；窗内无候选（稀疏转写大间隙）
    =回退旧定时贪心（首个 ≥ 目标的段起点）；无更多段=None（尾块）。"""
    total = len(segments)
    target = block_start + _BLOCK_TARGET_MS
    low = target - _BLOCK_FLOAT_MS
    high = min(target + _BLOCK_FLOAT_MS, block_start + _BLOCK_CAP_MS)
    candidates = [
        position for position in range(index + 1, total)
        if low <= segments[position][0] <= high
    ]
    if candidates:
        return min(
            candidates,
            key=lambda position: (
                _boundary_similarity(segments, position),
                abs(segments[position][0] - target),
                segments[position][0],
            ),
        )
    for position in range(index + 1, total):
        if segments[position][0] >= target:
            return position
    return None


# ---- 期 0 semantic_fallback 构建器（确定性本地切片，零 LLM 调用）-----------


def _block_title(chapters: list[dict[str, Any]], block_text: str) -> tuple[str, str]:
    """板块名=幻灯片标题优先（期 0=LLM 章节名按时间落入）、转写短语兜底。"""
    for chapter in chapters:
        title = _clean_text(chapter.get("title"), _TITLE_CAP)
        if title:
            return title, _clean_text(chapter.get("summary"), _SUMMARY_CAP)
    head = _clean_text(block_text, _TITLE_CAP).split("。", 1)[0].split("，", 1)[0]
    return (head or "本讲内容"), _clean_text(block_text, 160)


def build_chapters_v2(
    chapters: list[dict[str, Any]], transcript: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """期 0 semantic_fallback 板块切片：转写段时间轴边界感知贪心（CONTENT-R1
    C1：5 分钟目标 ±2 分钟浮动窗内取文本重合度最低点闭合，零 LLM 调用；稀疏
    转写回退定时贪心），LLM 章节名/摘要按时间落入对应块（幻灯片标题优先、
    转写短语兜底）；尾块不足地板并入前块；随后过 validate_chapters_v2 硬门，
    不合= None（前端走旧 chapters）。转写缺席（纯课件讲次）=None（诚实省键，
    不虚构板块）。"""
    segments: list[tuple[int, int, str]] = []
    for item in transcript or []:
        if not isinstance(item, dict):
            continue
        start = _valid_ms(item.get("start_ms"))
        text = _clean_text(item.get("text"), 400)
        if start is None or not text:
            continue
        end = _valid_ms(item.get("end_ms"))
        segments.append((start, end if end is not None and end > start else start, text))
    if not segments:
        return None
    segments.sort(key=lambda trio: trio[0])
    usable_chapters = sorted(
        (
            item for item in (chapters or [])
            if isinstance(item, dict) and _clean_text(item.get("title"), _TITLE_CAP)
        ),
        key=lambda item: int(item.get("start_ms") or 0),
    )
    blocks: list[tuple[int, list[tuple[int, int, str]]]] = []
    block_start = 0  # 首块 0:00（板块铺满时间轴，闭集首块定义）
    index = 0
    while index < len(segments):
        boundary = _find_block_boundary(segments, index, block_start)
        if boundary is None:
            blocks.append((block_start, segments[index:]))
            break
        blocks.append((block_start, segments[index:boundary]))
        block_start = segments[boundary][0]
        index = boundary
    # 尾块不足地板=并入前块（回翻不成块纪律的对偶：宁少一块不立碎块）。
    if len(blocks) >= 2:
        last_start, last_parts = blocks[-1]
        tail_ms = last_parts[-1][1] - last_start
        if tail_ms < _BLOCK_FLOOR_MS:
            prev_start, prev_parts = blocks[-2]
            blocks[-2] = (prev_start, prev_parts + last_parts)
            blocks.pop()
    block_starts = [block[0] for block in blocks]
    items: list[dict[str, Any]] = []
    remaining = list(usable_chapters)
    for index, (block_start, parts) in enumerate(blocks):
        block_text = " ".join(part[2] for part in parts)
        next_start = block_starts[index + 1] if index + 1 < len(block_starts) else None
        # 时间落入本块（下一块起点前）的章节归本块取名；消费过的不再复用。
        owned = [
            chapter for chapter in remaining
            if block_start <= int(chapter.get("start_ms") or 0)
            and (next_start is None or int(chapter.get("start_ms") or 0) < next_start)
        ]
        for chapter in owned:
            remaining.remove(chapter)
        title, summary = _block_title(owned, block_text)
        items.append({
            "title": title,
            "start_ms": block_start,
            "summary": summary or _clean_text(block_text, 160),
            "source": "semantic",
        })
    return validate_chapters_v2({"aggregation": "semantic_fallback", "items": items})


# ---- 章节数量纪律（V010-CONTENT-IMPL-1，CONTENT-R1 B2）---------------------


def enforce_chapter_count(
    chapters: list[dict[str, Any]], duration_ms: int,
) -> list[dict[str, Any]]:
    """LLM 合并链章节数量纪律：区间帽=max(≥3 地板, min(时长//3 分钟地板, 20))。
    实测 44 例 0-59 失控（与质量无关，最大产物 0 章节）——超帽=反复「时间最
    近邻合并」（最小 start 间隙对，并列取最左）直至 ≤ 帽；≤帽按 start_ms 稳
    定排序原样透传（键集零变化，幂等）。纯本地确定性规则，零新调用；章节数
    据不删字（合并摘要拼接截帽），锚取合并对首项（白名单值不丢形）。"""
    items = [item for item in (chapters or []) if isinstance(item, dict)]
    if not items:
        return []
    items.sort(key=lambda item: int(item.get("start_ms") or 0))
    duration = max(0, int(duration_ms or 0))
    cap = max(_DOC_MIN_SECTIONS, min(duration // _BLOCK_FLOOR_MS, _CHAPTER_COUNT_CAP))
    merged = list(items)
    while len(merged) > cap:
        best_index = 0
        best_gap: int | None = None
        for position in range(len(merged) - 1):
            gap = (
                int(merged[position + 1].get("start_ms") or 0)
                - int(merged[position].get("start_ms") or 0)
            )
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_index = position
        first, second = merged[best_index], merged[best_index + 1]
        merged[best_index] = {
            "title": _clean_text(first.get("title"), _TITLE_CAP),
            "start_ms": int(first.get("start_ms") or 0),
            "summary": (
                _clean_text(first.get("summary"), 2_000).strip()
                + "；"
                + _clean_text(second.get("summary"), 2_000).strip()
            )[:_CHAPTER_MERGED_SUMMARY_CAP],
        }
        del merged[best_index + 1]
    return merged


# ---- 期 1 LLM 结构化 doc 增量（V010-CONTENT-IMPL-1；走同一校验门）----------


def coerce_llm_doc(raw: Any, valid_anchors: set[int]) -> dict[str, Any] | None:
    """合并链 LLM doc 候选收敛器：逐级清洗后过 validate_summary_doc 同一道门
    （期 0 docstring 预留的「期 1 LLM 结构化输出」兑现）。清洗规则：
    - 锚完整性：sections 的 start_ms/points 的 anchor_ms 必须在转写/课件锚
      白名单内（QSCOREFIX-1④ 同纪律：白名单外=不可跳转=宁丢不编）；
    - source 收敛：合并链候选语义恒为 ``semantic``（ provenance 是调用方已知
      事实，不信任 LLM 自报）；
    - 逐级丢弃：坏点丢点、空段丢段、全空= None（调用方回退本地
      build_summary_doc 组装）；落键形状恒过 validate_summary_doc=禁半残。"""
    if not isinstance(raw, dict) or not isinstance(valid_anchors, set):
        return None
    raw_sections = raw.get("sections")
    if not isinstance(raw_sections, list):
        return None
    sections: list[dict[str, Any]] = []
    for item in raw_sections[:_MAX_BLOCKS]:
        if not isinstance(item, dict):
            continue
        start_ms = _valid_ms(item.get("start_ms"))
        title = _clean_text(item.get("title"), _TITLE_CAP)
        if start_ms is None or start_ms not in valid_anchors or not title:
            continue
        raw_points = item.get("points")
        if not isinstance(raw_points, list):
            continue
        points: list[dict[str, Any]] = []
        for point in raw_points[:16]:
            if not isinstance(point, dict):
                continue
            text = _clean_text(point.get("text"), _POINT_CAP)
            anchor = _valid_ms(point.get("anchor_ms"))
            if not text or anchor is None or anchor not in valid_anchors:
                continue
            points.append({"text": text, "anchor_ms": anchor})
        if not points:
            continue
        formulas: list[dict[str, Any]] = []
        raw_formulas = item.get("formulas")
        if isinstance(raw_formulas, list):
            for formula in raw_formulas[:16]:
                if not isinstance(formula, dict):
                    continue
                latex = _clean_text(formula.get("latex"), 300)
                if not latex:
                    continue
                entry: dict[str, Any] = {"latex": latex}
                anchor = _valid_ms(formula.get("anchor_ms"))
                if anchor is not None and anchor in valid_anchors:
                    entry["anchor_ms"] = anchor
                caption = _clean_text(formula.get("caption"), 200)
                if caption:
                    entry["caption"] = caption
                formulas.append(entry)
        section: dict[str, Any] = {
            "start_ms": start_ms,
            "title": title,
            "source": "semantic",
            "points": points,
            "formulas": formulas,
        }
        raw_example = item.get("example")
        if isinstance(raw_example, dict):
            example_text = _clean_text(raw_example.get("text"), _POINT_CAP)
            if example_text:
                example: dict[str, Any] = {"text": example_text}
                anchor = _valid_ms(raw_example.get("anchor_ms"))
                if anchor is not None and anchor in valid_anchors:
                    example["anchor_ms"] = anchor
                section["example"] = example
        sections.append(section)
    if not sections:
        return None
    return validate_summary_doc({
        "overview": raw.get("overview") if raw.get("overview") is not None else "",
        "sections": sections,
        "terms": raw.get("terms") if raw.get("terms") is not None else [],
        "summary_notes": raw.get("summary_notes") if raw.get("summary_notes") is not None else [],
    })


# ---- doc 本地构建（期 0：零新调用，从已过门的结构化产物忠实组装）-----------


def build_summary_doc(
    markdown: str, chapters_v2: dict[str, Any] | None,
    key_takeaways: list[str],
) -> dict[str, Any] | None:
    """期 0 doc 构建：sections=chapters_v2 板块 1:1 忠实映射（points=板块摘要
    锚在块起点），summary_notes=key_takeaways 头部截取（3-5 条小结合同），
    overview=笔记开头段落（诚原子集，非概念地图——LLM 结构化 overview 由
    期 1 增量承担），terms=空列表（glossary 为纯词串，强配 symbol/note=
    虚构，禁）。chapters_v2 缺席=整链降级返回 None（绝不半残文档）。"""
    if not isinstance(chapters_v2, dict) or not chapters_v2.get("items"):
        return None
    paragraphs: list[str] = []
    for para in str(markdown or "").split("\n\n"):
        text = " ".join(
            line.strip().lstrip("#").lstrip("-").lstrip("*").lstrip(">").strip()
            for line in para.splitlines()
            if line.strip()
        ).strip()
        if text:
            paragraphs.append(text[:_OVERVIEW_PARA_CAP])
        if len(paragraphs) >= _OVERVIEW_MAX_PARAS:
            break
    if not paragraphs:
        return None
    sections = [
        {
            "start_ms": int(item["start_ms"]),
            "title": item["title"],
            "source": item["source"],
            "points": [{
                "text": item["summary"] or item["title"],
                "anchor_ms": int(item["start_ms"]),
            }],
            "formulas": [],
        }
        for item in chapters_v2["items"]
    ]
    notes = [
        _clean_text(note, _NOTE_CAP) for note in (key_takeaways or [])[:_NOTES_MAX]
        if str(note or "").strip()
    ]
    return validate_summary_doc({
        "overview": paragraphs,
        "sections": sections,
        "terms": [],
        "summary_notes": notes,
    })


# ---- events 加法演进（v2 字段零变化，只增 confidence/source）---------------


def annotate_assessment_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并链考核事件加法标注：逐项补 confidence/source（缺省=low/合并链候选
    语义，见模块头 confidence 语义节）。输入项原样复制不回写；end_ms 保留
    合法缺省（合并链引句无时间窗）。非 dict 项原样透传由上游拒绝计数。"""
    annotated: list[dict[str, Any]] = []
    for item in events:
        if not isinstance(item, dict):
            annotated.append(item)
            continue
        merged = dict(item)
        confidence = str(merged.get("confidence") or "low")
        merged["confidence"] = confidence if confidence in EVENT_CONFIDENCES else "low"
        source = str(merged.get("source") or "transcription")
        merged["source"] = source if source in EVENT_SOURCES else "transcription"
        annotated.append(merged)
    return annotated
