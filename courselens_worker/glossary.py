"""字幕术语纠错·规则版（N5A-P3）：课程词表的确定性后处理，零 token、零外联。

词表来自本讲课件 OCR 文本的高频中文术语（2-8 字、词频≥3、封顶 200 词），
OCR 缺席时只退课程名。字幕里与词表词编辑距离≤1 的错拼变体——不在词表、
不触发受保护形、全讲出现≥2 次——整讲统一替换回词表词（讲内一致性优先），
被改写的段落标记闭集校对状态 ``applied-glossary``。无证据不成对、不替换。

RXFIX-1（2026-10-29，37355 半器课型误替换根修）加两道准入守卫：讲内
「优势形」（≥GLOSSARY_DOMINANT_FORM_MIN_OCCURRENCES 次=本讲真实词形）与
「源语料保护形」（课件 OCR/课程名语料含该串=课程权威写法）一律不按错拼
变体处理——修正确的、不伤原对的（真实损伤签名：电流→电平/电路→电平/
效应→效应管/基极→栅极整讲系统性改写）。
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
# RXFIX-1（2026-10-29，37355 半器课型术语系统性误替换根修）：讲内「优势形」
# 守卫阈值。实测损伤签名=正确术语在讲内高频（电流 382-384/电路 35/效应
# 104 次仍被整讲替换成词表近形），而合法错拼变体通常 ≤3 次；讲内 ≥8 次的
# 形即本讲真实词形，编辑距离再近也绝不改写（宁漏勿错=本模块既有哲学，
# 错误替换的代价远高于漏纠：一次误替换毁整讲物理语义）。
GLOSSARY_DOMINANT_FORM_MIN_OCCURRENCES = 8
# RXESCAPE-1（2026-10-30，685785 残留伤型根修）：提示词注入面近形影子过滤比
# ——低证课件词自身转写 <8 次且存在编辑距离 ≤1 的转写优势形近形词（≥8 次且
# ≥4 倍于己）时不入「术语写法以 glossary 表为准」权威清单（37355 实证：
# 电平〔转写 2 次〕vs 电流〔150 次〕同表注入=模型侧近形「规范化」偏置通道；
# 4 倍比与优势形阈值同源，宁漏勿错）。
_GLOSSARY_PROMPT_SHADOW_RATIO = 4
_TERM_RE = re.compile(r"[\u4e00-\u9fa5]{2,8}")
# 高频停用串闭集：OCR 页眉页脚与常见虚词组合，不进词表。
_STOPWORDS = frozenset({
    "的了的", "一个", "我们", "你们", "他们", "这个", "那个", "什么", "可以",
    "就是", "但是", "然后", "所以", "因为", "如果", "现在", "时候", "这样",
    "没有", "不是", "一下", "大家", "自己", "这个", "那么", "这些", "那些",
    "第几讲", "上一页", "下一页", "第一页", "课件", "讲义", "第页共",
})


class GlossaryTerms(tuple):
    """词表元组 + 源语料证据底座（RXFIX-1）。

    行为与普通元组逐位兼容（``in``/迭代/``==()`` 全不变）；多携带一个
    ``source_corpus``=构造词表时的全部源文本（课件 OCR 页文本+课程名）。
    配对侧用它做「源语料保护形」守卫：出现在课件语料里的串=课程真实
    写法，绝不是错拼变体。普通元组（payload/env 词表、旧调用方、测试）
    无此属性 → 守卫不生效 → 零行为变化。
    """

    source_corpus: str = ""

    def __new__(cls, terms, source_corpus: str = "") -> "GlossaryTerms":
        self = super().__new__(cls, terms)
        self.source_corpus = str(source_corpus or "")
        return self


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

    RXFIX-1：返回 GlossaryTerms（附 source_corpus=全部页文本+课程名），
    配对侧据此拦截「源语料保护形」与「讲内优势形」，根修最小对立体误替换
    （电流→电平/基极→栅极族）。
    """
    corpus_parts: list[str] = [str(course_title or "")] if str(course_title or "").strip() else []
    counts: Counter = Counter()
    for page in pages or []:
        page_text = str((page or {}).get("text") or "")
        corpus_parts.append(page_text)
        _run_substring_counts(page_text, counts)
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
    return GlossaryTerms(candidates[:GLOSSARY_MAX_TERMS], "\n".join(corpus_parts))


def _source_corpus(glossary: tuple[str, ...]) -> str:
    """词表的源语料全文（GlossaryTerms 专用；普通元组=空=守卫不生效）。"""
    return str(getattr(glossary, "source_corpus", "") or "")


def _piece_blocked_as_variant(piece: str, source_corpus: str, occurrences: int) -> bool:
    """RXFIX-1 配对守卫：该串不得按错拼变体处理（两道守卫命中任一即禁）。

    ①优势形守卫：讲内出现 ≥GLOSSARY_DOMINANT_FORM_MIN_OCCURRENCES 次的
    串=本讲真实词形（37355 实证：电流 382 次/效应 104 次仍被整讲改写）；
    ②源语料守卫：串出现在课件 OCR/课程名语料=课程权威写法，绝不是变体。
    """
    if occurrences >= GLOSSARY_DOMINANT_FORM_MIN_OCCURRENCES:
        return True
    return bool(source_corpus) and piece in source_corpus


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


def _is_han(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _piece_always_embedded(piece: str, text: str) -> bool:
    """RXESCAPE-1 零证片段守卫的形态判据：piece 在 text 中的每次出现都前后
    紧邻汉字（无一处处于中文串边界=行首/行尾/标点/拉丁邻接）→ 该串只是更长
    真实词的内部片段（「电中性区」中的「电中」），不是词，绝不按错拼变体
    改写。存在任一边界出现 → 非伪影（正常词形），由既有守卫裁量。"""
    if not piece or not text:
        return False
    start = 0
    seen = False
    while True:
        index = text.find(piece, start)
        if index < 0:
            break
        seen = True
        left = text[index - 1] if index > 0 else ""
        right = text[index + len(piece)] if index + len(piece) < len(text) else ""
        if not (_is_han(left) and _is_han(right)):
            return False
        start = index + 1
    return seen


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
    """讲内一致的术语纠错；改动段标记 applied-glossary（第八校对态）。

    RXFIX-1：配对准入加双守卫——讲内优势形（≥8 次）与源语料保护形（课件
    OCR/课程名含该串）一律不按错拼变体处理（37355 术语系统性误替换根修）。
    """
    if not glossary or not segments:
        return segments
    from .llm import _protected_change  # 迟绑定：llm 顶层已导入本模块

    source_corpus = _source_corpus(glossary)
    # 第一遍：统计每个错拼变体的全讲出现次数；只接受能唯一映射到词表词的
    # 串（等距多词=无证据不成对）；优势形/源语料形禁配对（RXFIX-1）。
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
        if len(terms) == 1
        and piece_counts[piece] >= _MIN_VARIANT_OCCURRENCES
        and not _piece_blocked_as_variant(piece, source_corpus, piece_counts[piece])
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


# V010-WORKER-QUAL-1（2026-10-26，W3 术语讲内混用）：总结 markdown 的讲内
# 同文种变体统一——与 apply_glossary 同一 trusted-pairing 纪律（唯一映射+
# 全文出现≥2），作用面=单条文本（markdown/doc 字段）。跨文种混用（Kirk↔科克）
# 编辑距离无法配对，走提示词统一指令（llm._STEM_TEACHING_RULES）。
_UNIFY_TEXT_MIN_OCCURRENCES = 2


def unify_text_terms(
    text: str,
    glossary: tuple[str, ...],
    *,
    source_corpus: str | None = None,
    evidence_text: str | None = None,
) -> str:
    """单条文本的讲内术语变体统一（确定性、零 token、零外联）。

    只在中文连续段内配对（LaTeX/代码/英文天然不在替换面）；变体须唯一映射到
    词表词且全文出现≥2 次（一次出现宁留原样=宁漏勿错删）。glossary 空=原样
    返回（零行为变化）。

    RXFIX-1：双守卫与 apply_glossary 同源——优势形（≥8 次）与源语料保护形
    （课件/课程名含该串）一律不改写。``source_corpus`` 显式传参供 create_summary
    透传（其内部已把词表展平成普通列表，属性会丢失）；缺省读词表自带属性。

    RXFIX-3（2026-10-29，669766 掷1 残留伤型根修）：``evidence_text``=讲内转写
    全文时两道升级——
    ①优势形守卫计频口径改为讲内转写（docstring「讲内」语义的真实现：老师真实
    说的词形以转写为准；修 RXFIX-1 实现按「被统一文本内部」计频的口径错位，
    该错位下提示词偏置先稀释 md 内词形、守卫随即失守）；
    ③方向守卫（新增）：piece 转写有证（>0）而目标词转写零证（=0）→ 禁替换
    （课件专属词永不「改正」老师真实说出的词；37355 实证伤型=电平〔课件词，
    转写 0 次〕整改电路/水平/电导/电阻〔口语词，转写 6/…次〕）。合法纠错向
    不受影响：OCR 字形变体（变換）转写零证、目标（变换）转写有证，方向守卫
    不触发。``evidence_text`` 缺省 None=零行为变化（兼容旧调用方与测试）。"""
    if not text or not glossary:
        return str(text or "")
    corpus = str(source_corpus) if source_corpus else _source_corpus(glossary)
    evidence = str(evidence_text or "")
    # RXESCAPE-1（2026-10-30，653998 掷2 活体实证补强）：无课件语料（payload/
    # env 词表、transcript-only 讲）时以讲内转写为保护语料兜底——老师真实说出
    # 的词形绝不被低证近形课件词「改正」。掷2 残留 31 处即此通道：电源/电池/
    # 电光/电路族全部转写有证（4/2/5/5 次）而低于优势形阈值、目标「电平」转
    # 写偶证 2 次令方向守卫失效、真实词形带串边界令片段守卫失效。合法纠错不
    # 受影响：OCR 字形变体/模型错拼（变換、厄早效应）从不在转写中出现=不受
    # 保护，照常统一（语音= ground truth，词形纠错只修「没被说出来的写法」）。
    if not corpus and evidence:
        corpus = evidence
    source = str(text)
    counts: Counter = Counter()
    piece_terms: dict[str, set[str]] = {}
    for run in re.findall(r"[\u4e00-\u9fa5]{2,}", source):
        for piece, term in _candidate_variants(run, glossary):
            # RXESCAPE-1（2026-10-30，653998 掷2 重放实证）：摘要面只做同长替换
            # 配对——插入/删除向（piece 比 term 长 1，如「极型层→型层」删「极」、
            # 「转换→换」类）在本面只产生长词内部伪影删除，无实证合法纠错形
            # （本面合法修复全部同长：变換→变换/厄早效应→厄利效应/电子→电平）；
            # 字幕面 apply_glossary 保留插入/删除（ASR 漏字向，其自有守卫裁量）。
            if len(piece) != len(term):
                continue
            piece_terms.setdefault(piece, set()).add(term)
            counts[piece] += run.count(piece)
    evidence_counts: dict[str, int] = {}
    if evidence:
        for piece, piece_terms_set in piece_terms.items():
            evidence_counts[piece] = evidence.count(piece)
            for term in piece_terms_set:
                evidence_counts.setdefault(term, evidence.count(term))
    trusted = {}
    for piece, piece_terms_set in piece_terms.items():
        if len(piece_terms_set) != 1:
            continue
        term = next(iter(piece_terms_set))
        if counts[piece] < _UNIFY_TEXT_MIN_OCCURRENCES:
            continue
        if _piece_blocked_as_variant(piece, corpus, counts[piece]):
            continue
        if evidence:
            # RXFIX-3 ①优势形守卫计频口径：讲内（转写）≥8=老师真实词形，绝不改写。
            if evidence_counts.get(piece, 0) >= GLOSSARY_DOMINANT_FORM_MIN_OCCURRENCES:
                continue
            # RXFIX-3 ③方向守卫：口语有证词不被转写零证词「改正」。
            if evidence_counts.get(piece, 0) > 0 and evidence_counts.get(term, 0) == 0:
                continue
            # RXESCAPE-1 ④零证片段守卫：piece 在被统一文本中的每次出现都严格
            # 嵌入更长中文串（前后紧邻汉字、无一处处于串边界）=滑窗内部伪影，
            # 禁改写。典型伤型=课件近形词的内部片段改写正确词——685785 实证：
            # 电中性区→电平性区×2（piece「电中」转写 0 次）；653998 活体复掷
            # 再实证（2026-10-30 22:1x 掷）：光电转换→光电平换（piece「电转」
            # 、「电平」转写偶证 2 次——嵌入形态本身即判据，不要求双零证）。
            # 既有优势形（<8）与方向守卫（piece 0 证不触发）对该形态不设防。
            # 合法纠错向不受影响：字形/近形变体（变換→变换、厄早效应→厄利效
            # 应）在文本中存在串边界出现（行首/标点/拉丁邻接），守卫不触发——
            # 判别特征=伪影只以嵌入形存在（宁漏勿错：纯嵌入形的真变体让渡）。
            if _piece_always_embedded(piece, source):
                continue
        trusted[piece] = term
    if not trusted:
        return source
    replaced = source
    for piece, term in trusted.items():
        replaced = replaced.replace(piece, term)
    return replaced


def _evidence_dominant_pieces(evidence: str, glossary_set: set[str]) -> dict[str, int]:
    """转写内出现 ≥GLOSSARY_DOMINANT_FORM_MIN_OCCURRENCES 次的词形闭集
    （RXESCAPE-1 注入影子用）。影子候选不限于词表内近形——653998 活体实证
    （2026-10-30 22:1x 掷）：payload 课程记忆词表含「电平」而缺「电流」，
    仅查表内兄弟漏影，「电平」以权威名义入表=模型侧 58 处近形清洗通道。"""
    counts: Counter = Counter()
    for run in re.findall(r"[\u4e00-\u9fa5]{2,}", evidence):
        for size in range(_MIN_TERM_CHARS, _MAX_TERM_CHARS + 1):
            for start in range(0, len(run) - size + 1):
                counts[run[start:start + size]] += 1
    return {
        piece: count
        for piece, count in counts.items()
        if count >= GLOSSARY_DOMINANT_FORM_MIN_OCCURRENCES and piece not in glossary_set
    }


def filter_prompt_terms(terms, evidence_text: str | None) -> tuple[str, ...]:
    """提示词注入面的近形影子词过滤（RXESCAPE-1，模型侧偏置通道收口）。

    背景：注入面把词表以「术语写法以 glossary 表为准」的权威清单带给生成
    模型；与讲内优势口语词编辑距离 ≤1 的低证课件词（37355 实证：电平〔转写
    2 次〕vs 电流〔150 次〕）会被模型当成「权威写法」对口语词做近形「规范
    化」——重灾三讲的生成侧替换即此通道（RXFIX-3 只滤了转写零证词，转写偶
    证的近形课件词仍入表）。影子候选=①词表内编辑距离 ≤1 的转写优势形兄弟，
    ②转写自身 ≥GLOSSARY_DOMINANT_FORM_MIN_OCCURRENCES 次的词形（653998 活
    体实证：payload 词表含「电平」缺「电流」，仅查表内兄弟漏影）。自身即优
    势形（≥8 次）的词恒保留；被影子剔除只损失「权威写法提示」收益、无改写
    方向（保守侧=宁漏勿错；副作用=与高频常用词近形的低证术语让渡注入，
    如「变换」vs 高频「变化」，由守卫与 gate_F 监控）。机械 unify 兜底面
    不受影响（其自有优势形/方向/零证片段三重守卫）。``evidence_text`` 空=
    原样返回（零行为变化）。
    """
    term_list = [str(term).strip() for term in (terms or ()) if str(term).strip()]
    evidence = str(evidence_text or "")
    if not term_list or not evidence:
        return tuple(term_list)
    counts = {term: evidence.count(term) for term in term_list}
    dominant_pieces = _evidence_dominant_pieces(evidence, set(term_list))

    def _shadowed(term: str) -> bool:
        if counts[term] >= GLOSSARY_DOMINANT_FORM_MIN_OCCURRENCES:
            return False
        threshold = _GLOSSARY_PROMPT_SHADOW_RATIO * counts[term]
        for sibling in term_list:
            if sibling != term and counts[sibling] >= GLOSSARY_DOMINANT_FORM_MIN_OCCURRENCES \
                    and counts[sibling] >= threshold and _within_edit_distance(term, sibling):
                return True
        for piece, count in dominant_pieces.items():
            if count >= threshold and _within_edit_distance(term, piece):
                return True
        return False

    return tuple(term for term in term_list if not _shadowed(term))
