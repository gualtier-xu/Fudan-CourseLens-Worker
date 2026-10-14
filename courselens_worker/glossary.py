"""字幕术语纠错·规则版（N5A-P3）：课程词表的确定性后处理，零 token、零外联。

词表来自本讲课件 OCR 文本的高频中文术语（2-8 字、词频≥3、封顶 200 词），
OCR 缺席时只退课程名。字幕里与词表词编辑距离≤1 的错拼变体——不在词表、
不触发受保护形、全讲出现≥2 次——整讲统一替换回词表词（讲内一致性优先），
被改写的段落标记闭集校对状态 ``applied-glossary``。无证据不成对、不替换。
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

GLOSSARY_MAX_TERMS = 200
GLOSSARY_MIN_FREQ = 3
GLOSSARY_APPLIED_STATUS = "applied-glossary"
# SUBTITLE-DEEP-1：术语闭集的运行时注入面。课程术语表可经 job payload 的可选
# ``glossary`` 键（客户端接线留桩，缺省缺席=旧行为）或操作员环境变量文件
# （每行一词）进入术语位深校对与热词文件；两者都缺席时术语层整体跳过。
TERM_GLOSSARY_ENV = "COURSELENS_TERM_GLOSSARY_FILE"
_MIN_TERM_CHARS = 2
_MAX_TERM_CHARS = 8
_EDIT_RADIUS = 1
_MIN_VARIANT_OCCURRENCES = 2
_TERM_RE = re.compile(r"[\u4e00-\u9fa5]{2,8}")
# 高频停用串闭集：OCR 页眉页脚与常见虚词组合，不进词表。
_STOPWORDS = frozenset({
    "的了的", "一个", "我们", "你们", "他们", "这个", "那个", "什么", "可以",
    "就是", "但是", "然后", "所以", "因为", "如果", "现在", "时候", "这样",
    "没有", "不是", "一下", "大家", "自己", "这个", "那么", "这些", "那些",
    "第几讲", "上一页", "下一页", "第一页", "课件", "讲义", "第页共",
})


def _run_substring_counts(text: str, counts: Counter, *, min_freq: int | None = None) -> None:
    """中文串内滑窗枚举 2-8 字子串并计数（术语不一定是分词边界）。"""
    for run in re.findall(r"[\u4e00-\u9fa5]{2,40}", str(text or "")):
        size = len(run)
        for length in range(_MIN_TERM_CHARS, min(_MAX_TERM_CHARS, size) + 1):
            for start in range(0, size - length + 1):
                counts[run[start:start + length]] += 1


def build_glossary(pages: list[dict[str, Any]] | None, course_title: str = "") -> tuple[str, ...]:
    """从课件页 OCR 文本构造课程词表；OCR 缺席时退课程名（不展开）。

    频次≥3 的子串入候选，依（频次↓、长度↓、字典序）排序后封顶。刻意不做
    包含剪枝：重复句式语料里「上下文短语」与「真术语」频次难分，剪枝会误杀
    真术语；多留重叠候选只影响词表体积（封顶 200），不影响配对安全性——
    配对还要求变体全讲出现≥2 次且不触发受保护形。
    """
    counts: Counter = Counter()
    for page in pages or []:
        _run_substring_counts((page or {}).get("text"), counts)
    candidates = [
        term for term, freq in counts.items()
        if freq >= GLOSSARY_MIN_FREQ and term not in _STOPWORDS
    ]
    if not candidates:
        # 课程名兜底：整串术语（滑窗取最长若干，不做频次门槛）
        title_counts: Counter = Counter()
        _run_substring_counts(course_title, title_counts)
        candidates = [term for term in title_counts if term not in _STOPWORDS]
    candidates.sort(key=lambda term: (-counts[term], -len(term), term))
    return tuple(candidates[:GLOSSARY_MAX_TERMS])


def _within_edit_distance(left: str, right: str, radius: int = _EDIT_RADIUS) -> bool:
    """同长=替换数；长度差 1=插入/删除。O(n) 双指针。"""
    if left == right:
        return False
    if abs(len(left) - len(right)) > radius:
        return False
    if len(left) == len(right):
        return sum(1 for a, b in zip(left, right) if a != b) <= radius
    short, long = (left, right) if len(left) < len(right) else (right, left)
    index = 0
    while index < len(short) and short[index] == long[index]:
        index += 1
    return short[index:] == long[index + 1:]


def _candidate_variants(segment_text: str, glossary: tuple[str, ...]):
    """产出一个中文串里的全部 (variant, term) 配对候选。

    串取「不限长中文连续段」——字幕句常常整句无标点，若按 2-8 字切段，
    跨越切点的错拼串就永远配不上词表词。
    """
    glossary_set = set(glossary)
    for run in re.findall(r"[\u4e00-\u9fa5]{2,}", segment_text):
        for size in range(_MAX_TERM_CHARS + 1, _MIN_TERM_CHARS - 1, -1):
            for start in range(0, len(run) - size + 1):
                piece = run[start:start + size]
                if piece in glossary_set:
                    continue
                for term in glossary:
                    if abs(len(term) - len(piece)) > _EDIT_RADIUS:
                        continue
                    if _MIN_TERM_CHARS <= len(piece) <= _MAX_TERM_CHARS + 1 and _within_edit_distance(piece, term):
                        yield piece, term


def resolve_course_terms(payload: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Resolve the course term closed set: payload ``glossary`` + env file.

    去重保序、去空、封顶 GLOSSARY_MAX_TERMS；两个来源都缺席返回空元组，
    调用方据此跳过术语层（零行为变化的缺省路径）。
    """
    values: list[str] = []
    raw = (payload or {}).get("glossary")
    if isinstance(raw, list):
        values.extend(str(item).strip() for item in raw)
    env_path = os.environ.get(TERM_GLOSSARY_ENV, "").strip()
    if env_path:
        try:
            lines = Path(env_path).read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        values.extend(line.strip() for line in lines)
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return tuple(deduped[:GLOSSARY_MAX_TERMS])


# V4NONTHINK-1 件2：课程记忆示例 payload 桩（SUBTITLE-DEEP-1 包B 沉淀后接入的
# 读侧合同）。形状=与通用示例库同构的最小单元：{"input": [{"id","text"}...],
# "ops": [{"id","old","new"}...]}。conforming-only fail-closed：任一条目不成形
# 即整条丢弃（坏示例进不了提示词）；缺席=空元组=零行为变化。
def resolve_course_examples(payload: dict[str, Any] | None = None) -> tuple[dict[str, Any], ...]:
    raw = (payload or {}).get("examples")
    if not isinstance(raw, list):
        return ()
    resolved: list[dict[str, Any]] = []
    for item in raw[:GLOSSARY_MAX_TERMS]:
        if not isinstance(item, dict):
            continue
        raw_input = item.get("input")
        raw_ops = item.get("ops")
        if not isinstance(raw_input, list) or not raw_input or not isinstance(raw_ops, list):
            continue
        shaped_input: list[dict[str, str]] = []
        shaped_ops: list[dict[str, str]] = []
        valid = True
        for entry in raw_input:
            entry_id = str(entry.get("id") or "").strip() if isinstance(entry, dict) else ""
            entry_text = str(entry.get("text") or "").strip() if isinstance(entry, dict) else ""
            if not entry_id or not entry_text:
                valid = False
                break
            shaped_input.append({"id": entry_id, "text": entry_text})
        if valid:
            for op in raw_ops:
                op_id = str(op.get("id") or "").strip() if isinstance(op, dict) else ""
                op_old = str(op.get("old") or "") if isinstance(op, dict) else ""
                op_new = str(op.get("new") or "") if isinstance(op, dict) else ""
                if not op_id or not op_old or not op_new:
                    valid = False
                    break
                shaped_ops.append({"id": op_id, "old": op_old, "new": op_new})
        if valid:
            resolved.append({"input": shaped_input, "ops": shaped_ops})
    return tuple(resolved)


def apply_glossary(segments: list[dict[str, Any]], glossary: tuple[str, ...]) -> list[dict[str, Any]]:
    """讲内一致的术语纠错；改动段标记 applied-glossary（第八校对态）。"""
    if not glossary or not segments:
        return segments
    from .llm import _protected_change  # 迟绑定：llm 顶层已导入本模块

    # 第一遍：统计每个错拼变体的全讲出现次数；只接受能唯一映射到词表词的
    # 串（等距多词=无证据不成对）。
    piece_terms: dict[str, set[str]] = {}
    piece_counts: Counter = Counter()
    for segment in segments:
        text = str(segment.get("text") or "")
        for piece, term in _candidate_variants(text, glossary):
            piece_terms.setdefault(piece, set()).add(term)
            piece_counts[piece] += text.count(piece)
    trusted = {
        piece: next(iter(terms))
        for piece, terms in piece_terms.items()
        if len(terms) == 1 and piece_counts[piece] >= _MIN_VARIANT_OCCURRENCES
    }
    if not trusted:
        return segments

    # 第二遍：整讲统一替换；受保护形变化一律回退。
    output: list[dict[str, Any]] = []
    for segment in segments:
        text = str(segment.get("text") or "")
        replaced = text
        for piece, term in trusted.items():
            replaced = replaced.replace(piece, term)
        if replaced == text or _protected_change(text, replaced, ""):
            output.append(segment)
            continue
        updated = dict(segment)
        updated["text"] = replaced
        updated["correction"] = GLOSSARY_APPLIED_STATUS
        output.append(updated)
    return output
