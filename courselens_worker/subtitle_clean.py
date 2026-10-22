"""Subtitle display cleaning (worker leg): Tier1 deterministic + Tier2 op gate.

字幕呈现 v2 道1 清洗腿（V010-SUBTITLE-D1；规格唯一出处=archive
V010-SUBTITLE-UX-R1 §2，字段级契约=shared/subtitle_display_contract.py，
本模块是其 worker 生产端，契约本体零改动）：

- **Tier1（零 LLM token、默认生效）**：确定性闭集垫词删除。对段文本做
  闭集整词扫描（宁脏勿错守卫内置），产出 deletion-only 的 clean 视图
  （来源 ``tier1_rule``）；单段删词比超 ``CLEAN_TIER1_MAX_DEL_RATIO``
  即整段放弃（保留原文）。
- **Tier2（deletion-only LLM 轻清洗）**：校对窗顺带产出的 ``{"id","del"}``
  删除提案的确定性验证器——闭集词表命中（或纯重复串）+ 保护形零触碰
  + 每段 ops ≤ 4 + 删词比 ≤ ``CLEAN_TIER2_MAX_DEL_RATIO``，任何一门
  不过即整段回退 Tier1（fail-closed，绝不报错、绝不改写）。
- **忠实层（构造性保证）**：本模块只计算 clean 视图三元组
  （clean_text/clean_source/clean_del_ratio），从不写 ``text``；
  VTT/SRT/导出/证据层恒 verbatim（契约传输矩阵义务）。
- **生产端省键**：无删除（clean 与 text 同值）时整体不落三键（宁缺勿错）；
  落键前经 shared 契约 ``validate_display_cue_v2`` schema 校验，失败即弃。

边界规则成文（钉面=worker/tests/test_subtitle_clean.py）：

- 句尾问询垫词与句中垫词形态：单边后界守卫——垫词后接句读/空白/文本尾
  才删（句首问询垫词如「对吧我们开始」保留）。
- 裸指示词（这个/那个/然后）：双边整词跨度守卫——前后皆句读/边界才删，
  「这个公式」的指示用法与「我们讲这个」的句尾指代一律保留。
- 纯语气词：单边后界守卫；表与客户端整形 SHAPE_FILLER_CHARS 同表。
- 保护形（数字/单位/公式/否定，llm._PROTECTED_FORM_RE）：与删除跨度部分
  重叠的候选直接弃用；完整含于具名闭集删除跨度内的保护形豁免（规格具名
  词表「对不对」的「不」随整词放行）；删后保护形多重集必须恰为删前减去
  豁免集（组合出新保护形如「三那个十→三十」即拒，宁脏勿错）。
- 删后 content-core 非空（至少一个字母/数字/CJK 字符）、不产生句首标点
  （结果首部句读折叠）、删词比按 Tier 帽复核。

遥测：全部为计数与闭集词（rejected 原因闭集），零提示词/零字幕内容
（runner._progress 纪律同 llm/asr）。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable

from shared.subtitle_display_contract import (
    CLEAN_TIER1_MAX_DEL_RATIO,
    CLEAN_TIER2_MAX_DEL_RATIO,
    CLEAN_MAX_OPS_PER_SEGMENT,
    clean_budget_ok,
    clean_del_ratio_of,
    validate_clean_ops,
    validate_display_cue_v2,
)

# ---- Tier1 闭集词表（规格 §2 具名闭集首版；走查后按误删率扩缩） ----------
# 守卫模式：any=形态本身无歧义，句中直接删；after=单边后界（后接句读/
# 空白/文本尾才删）；whole=双边整词跨度（前后皆句读/边界）。
# 句尾问询垫词（单边后界：句首问询垫词如「对吧我们开始」保留）。
TAG_QUESTION_FILLERS = (
    "对吧", "对不对", "是不是", "行不行", "好不好", "好吗", "懂吧", "明白吧",
)
# 句中垫词（仅垫词形态；形态本身无歧义，句中直接删——fixture 金样本
# 「这个公式就是说我们…」对齐）。
MID_SENTENCE_FILLERS = ("就是说", "然后呢", "这个呢", "那个呢", "的话呢", "然后的话")
# 裸指示垫词（歧义面最大，可作实义指示词/指代宾语；仅双边整词跨度才删，
# 「这个公式」的指示用法与「我们讲这个」的句尾指代一律保留）。
BARE_DEMONSTRATIVE_FILLERS = ("这个", "那个", "然后")
# 纯语气词（与 src AS13 SHAPE_FILLER_CHARS 同表，客户端整形口径对齐；
# 单边后界）。
PARTICLE_FILLERS = tuple("呃嗯啊哦噢哎唉欸诶嘿哟唷咦哇喔咯嘞呣唔嘛啦呗")

_GUARD_ANY = "any"
_GUARD_AFTER = "after"
_GUARD_WHOLE = "whole"

# 长度优先扫描序（「然后的话」先于「然后」、「明白吧」先于单字）。
_FILLER_RULES: tuple[tuple[str, str], ...] = tuple(sorted(
    (
        *(((form, _GUARD_AFTER) for form in TAG_QUESTION_FILLERS)),
        *(((form, _GUARD_ANY) for form in MID_SENTENCE_FILLERS)),
        *(((form, _GUARD_WHOLE) for form in BARE_DEMONSTRATIVE_FILLERS)),
        *(((form, _GUARD_AFTER) for form in PARTICLE_FILLERS)),
    ),
    key=lambda item: (-len(item[0]), item[0]),
))
_FILLER_LOOKUP: dict[str, str] = dict(_FILLER_RULES)

# Tier2 验证用闭集并集（闭集类别命中即放行，位置守卫属 Tier1 选择逻辑）。
TIER2_CLOSED_SET = frozenset(
    (*TAG_QUESTION_FILLERS, *MID_SENTENCE_FILLERS,
     *BARE_DEMONSTRATIVE_FILLERS, *PARTICLE_FILLERS)
)

# 句读与空白（判定整词跨度/句首标点折叠）。
_JUNCTURE_CHARS = frozenset("，。？！、；：,.!?;:…—·（）()「」『』《》〈〉\"'")
# 否定字：垫词紧邻前为否定字时不删（「不对吧」≠「对吧」垫词形态）。
_NEGATION_CHARS = frozenset("不没未无非别勿莫")

_MAX_REPEAT_UNIT_CHARS = 8

# Tier2 拒因闭集（遥测词，零内容）。
TIER2_REJECT_REASONS = (
    "empty", "shape", "closed_set", "protected", "ratio", "contract", "content_core",
)


def _is_juncture(text: str, index: int) -> bool:
    """index 处是否句读/空白/文本尾边界。"""
    if index >= len(text):
        return True
    char = text[index]
    return char in _JUNCTURE_CHARS or char.isspace()


def _at_span_start(text: str, index: int) -> bool:
    """index 处是否句读/空白/文本头边界（整词跨度左界）。"""
    if index <= 0:
        return True
    char = text[index - 1]
    return char in _JUNCTURE_CHARS or char.isspace()


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """保护形跨度（llm._PROTECTED_FORM_RE；迟绑定同 glossary 惯例防环）。"""
    from .llm import _PROTECTED_FORM_RE

    return [(match.start(), match.end()) for match in _PROTECTED_FORM_RE.finditer(text)]


def _protected_form_count(text: str) -> Counter:
    from .llm import _PROTECTED_FORM_RE

    return Counter(match.group(0) for match in _PROTECTED_FORM_RE.finditer(text))


def _overlaps(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def _contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def _is_repeat_candidate(text: str, fragment: str) -> bool:
    """纯重复串类别：相邻口吃重复的单份拷贝删除保其一，语义不变。

    「我们我们讲」删一份「我们」→「我们讲」。要求 fragment（1..8 字、
    零句读）首个出现位与紧随其后的等长同串相邻；不相邻的重复是修辞
    （「我们，我们讲」）不是口吃，不删。
    """
    if not fragment or len(fragment) > _MAX_REPEAT_UNIT_CHARS:
        return False
    if any(char in _JUNCTURE_CHARS or char.isspace() for char in fragment):
        return False
    first = text.find(fragment)
    if first < 0:
        return False
    return text.find(fragment, first + len(fragment)) == first + len(fragment)


def _fold_spacing(cleaned: str) -> str:
    """删除接缝的空白/重复句读折叠与首部句读折叠（只删不增，子序列性保持）。

    重复句读折叠只对普通句读符生效（……省略号与——破折号的成对排版
    形态保留）；「我们讲，然后，收尾」删垫词后的「，，」缝由此收敛。
    """
    cleaned = re.sub(r"[ \t\r\n]+", " ", cleaned)
    cleaned = re.sub(r"([，。？！、；：,.!?;:])[ \t]*\1+", r"\1", cleaned)
    cleaned = cleaned.strip()
    return cleaned.lstrip("".join(_JUNCTURE_CHARS) + " ").strip()


def _has_content_core(text: str) -> bool:
    return any(char.isalnum() for char in text)


def _splice(text: str, spans: list[tuple[int, int]]) -> str:
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _guard_cleaned(
    text: str,
    spans: list[tuple[int, int]],
    max_del_ratio: float,
) -> tuple[str, str] | None:
    """共享守卫链：保护形门 → 拼接 → 折叠 → content-core → 帽复核。

    返回 cleaned 或 None（=保护形/守卫/删词比帽拒绝，fail-closed 保留原文）。
    豁免集=完整含于删除跨度内的保护形（具名闭集整删放行，如「对不对」
    的「不」）；部分重叠在候选选择阶段已由调用方弃用，此处兜底复核。
    """
    protected = _protected_spans(text)
    waived: list[str] = []
    for start, end in protected:
        hit = any(_overlaps(start, end, span[0], span[1]) for span in spans)
        if not hit:
            continue
        if any(_contains(span, (start, end)) for span in spans):
            waived.append(text[start:end])
            continue
        return None
    cleaned = _fold_spacing(_splice(text, spans))
    if not cleaned or not _has_content_core(cleaned):
        return None
    expected = _protected_form_count(text) - Counter(waived)
    if _protected_form_count(cleaned) != expected:
        return None
    ratio = clean_del_ratio_of(text, cleaned)
    if not clean_budget_ok(_budget_source(max_del_ratio), ratio):
        return None
    return cleaned


def _budget_source(max_del_ratio: float) -> str:
    return "tier1_rule" if max_del_ratio >= CLEAN_TIER1_MAX_DEL_RATIO else "tier2_llm"


def tier1_candidate_spans(text: str) -> list[tuple[int, int]]:
    """闭集整词扫描（宁脏勿错守卫内置；跨度按序、互不重叠）。

    通用否定守卫：紧邻前或紧邻后为否定字的候选一律不删——「不对吧」的
    「对吧」、「就是说不清楚」的「就是说」与「啊不是」的「啊」都处在
    否定语义边界上，删除会孤儿化/翻转否定（fail-closed 保原文）。
    """
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        match: tuple[int, str, str] | None = None
        for form, guard in _FILLER_RULES:
            if text.startswith(form, index):
                match = (index, form, guard)
                break
        if match is None:
            index += 1
            continue
        start, form, guard = match
        end = start + len(form)
        after_ok = _is_juncture(text, end)
        if guard == _GUARD_ANY:
            allowed = True
        elif guard == _GUARD_AFTER:
            allowed = after_ok
        else:
            allowed = _at_span_start(text, start) and after_ok
        negation_adjacent = (start > 0 and text[start - 1] in _NEGATION_CHARS) or (
            end < len(text) and text[end] in _NEGATION_CHARS
        )
        if (
            allowed
            and not negation_adjacent
            and not any(_overlaps(start, end, span[0], span[1]) for span in spans)
        ):
            spans.append((start, end))
            index = end
        else:
            index += 1
    # 保护形部分重叠的候选弃用（整含豁免在守卫链处理）。
    kept: list[tuple[int, int]] = []
    for span in spans:
        partial = any(
            _overlaps(span[0], span[1], prot[0], prot[1])
            and not _contains(span, prot)
            for prot in _protected_spans(text)
        )
        if not partial:
            kept.append(span)
    return kept


def tier1_clean_text(text: str) -> str | None:
    """Tier1 确定性闭集垫词删除；无可清洗或守卫拒绝返回 None（保留原文）。"""
    if not isinstance(text, str) or not text.strip():
        return None
    spans = tier1_candidate_spans(text)
    if not spans:
        return None
    return _guard_cleaned(text, spans, CLEAN_TIER1_MAX_DEL_RATIO)


def validate_tier2_dels(text: str, dels: list[Any]) -> tuple[str | None, str | None]:
    """Tier2 删除提案确定性验证门；返回 (cleaned, reject_reason)。

    门序（规格 §2 ①-⑤ + 生产端词表资产）：形状预门 → 闭集/纯重复命中 →
    shared.validate_clean_ops（逐字出现/不重叠/ops 帽/保护形幸存；比率帽
    由本函数对折叠后终态清洗文本统一复核，保证拒因与落键 ratio 同源）→
    content-core → 保护形多重集恒等 → 删词比 ≤20% 硬帽。任何一门不过
    返回 (None, 闭集拒因)，调用方整段回退 Tier1。
    """
    if not isinstance(text, str) or not text.strip():
        return None, "empty"
    if not isinstance(dels, list) or not dels:
        return None, "shape"
    fragments: list[str] = []
    for item in dels:
        if not isinstance(item, str) or not item.strip():
            return None, "shape"
        stripped = item.strip()
        if stripped not in TIER2_CLOSED_SET and not _is_repeat_candidate(text, stripped):
            return None, "closed_set"
        if stripped not in fragments:
            fragments.append(stripped)
    if len(fragments) > CLEAN_MAX_OPS_PER_SEGMENT:
        return None, "shape"
    spans: list[tuple[int, int]] = []
    for fragment in fragments:
        start = text.find(fragment)
        if start < 0:
            return None, "closed_set"
        spans.append((start, start + len(fragment)))
    # 否定边界守卫（与 Tier1 同源）：实际删除发生位的紧邻前/后为否定字 →
    # 拒。「他是不是不能来」的 A-not-A 疑问形态与「不对吧」的否定边界，
    # 删除会翻转/孤儿化否定语义（宁脏勿错，整段回退 Tier1）。
    for start, end in spans:
        if (start > 0 and text[start - 1] in _NEGATION_CHARS) or (
            end < len(text) and text[end] in _NEGATION_CHARS
        ):
            return None, "protected"
    # 保护形：部分重叠任一提案跨度 → 整段拒（fail-closed）。
    for prot_start, prot_end in _protected_spans(text):
        if any(
            _overlaps(prot_start, prot_end, span[0], span[1])
            and not _contains(span, (prot_start, prot_end))
            for span in spans
        ):
            return None, "protected"
    waived_forms = [
        text[prot_start:prot_end]
        for prot_start, prot_end in _protected_spans(text)
        if any(_contains(span, (prot_start, prot_end)) for span in spans)
    ]
    # 幸存保护形 = 删前多重集 − 豁免集；组合出新保护形由下方恒等复核兜底。
    surviving = list((_protected_form_count(text) - Counter(waived_forms)).elements())
    try:
        cleaned = validate_clean_ops(
            text,
            [{"id": f"c{index}", "del": fragment} for index, fragment in enumerate(fragments)],
            protected_forms=surviving,
        )
    except Exception:
        return None, "contract"
    cleaned = _fold_spacing(cleaned)
    if not cleaned or not _has_content_core(cleaned):
        return None, "content_core"
    if _protected_form_count(cleaned) != _protected_form_count(text) - Counter(waived_forms):
        return None, "protected"
    ratio = clean_del_ratio_of(text, cleaned)
    if not clean_budget_ok("tier2_llm", ratio):
        return None, "ratio"
    return cleaned, None


def apply_display_cleaning(
    segments: list[dict[str, Any]],
    clean_plan: list[dict[str, Any]] | None,
    *,
    telemetry: "dict[str, int] | None" = None,
) -> dict[str, int]:
    """终界清洗：Tier2 锚定计划复验优先，Tier1 兜底；就地附加契约三元组。

    ``clean_plan`` 项形 ``{"start_ms","end_ms","dels":[...]}``（校对窗验证
    通过的删除计划，锚定随链稳定的 (start_ms,end_ms)）。每段：计划命中且
    对最终文本复验通过 → ``tier2_llm``；否则 Tier1 闭集扫描 → ``tier1_rule``；
    都无删除 → 不落键（宁缺勿错）。落键前经 shared 契约 schema 校验
    （失败即弃键，绝不抛错）。``text`` 恒不写（忠实层构造性保证）。
    返回闭集遥测计数。
    """
    stats = {
        "segments": 0,
        "tier1_applied": 0,
        "tier2_applied": 0,
        "tier2_fallback_tier1": 0,
        "tier2_plan_unmatched": 0,
        "no_cleaning": 0,
    }
    plan_by_anchor: dict[tuple[int, int], list[Any]] = {}
    for entry in clean_plan or []:
        if not isinstance(entry, dict):
            continue
        try:
            anchor = (int(entry.get("start_ms") or 0), int(entry.get("end_ms") or 0))
        except (TypeError, ValueError):
            continue
        dels = entry.get("dels")
        if isinstance(dels, list) and dels and anchor not in plan_by_anchor:
            plan_by_anchor[anchor] = dels
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = segment.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        stats["segments"] += 1
        anchor = (
            int(segment.get("start_ms") or 0),
            int(segment.get("end_ms") or 0),
        )
        cleaned: str | None = None
        source: str | None = None
        dels = plan_by_anchor.pop(anchor, None)
        if dels is not None:
            cleaned, reason = validate_tier2_dels(text, dels)
            if cleaned is not None:
                source = "tier2_llm"
                stats["tier2_applied"] += 1
            else:
                stats["tier2_fallback_tier1"] += 1
        if cleaned is None:
            cleaned = tier1_clean_text(text)
            if cleaned is not None:
                source = "tier1_rule"
                stats["tier1_applied"] += 1
        if cleaned is None or source is None or cleaned == text:
            stats["no_cleaning"] += 1
            continue
        trio = {
            "clean_text": cleaned,
            "clean_source": source,
            "clean_del_ratio": clean_del_ratio_of(text, cleaned),
        }
        probe = dict(segment)
        probe.update(trio)
        try:
            validate_display_cue_v2(probe)
        except Exception:
            stats["no_cleaning"] += 1
            continue
        segment.update(trio)
    stats["tier2_plan_unmatched"] = len(plan_by_anchor)
    if telemetry is not None:
        telemetry.update(stats)
    return stats
