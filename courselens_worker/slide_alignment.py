"""Courseware (slide) alignment for summary generation — V010-SLIDEALIGN-1.

转写源结构性缺公式（老师不念公式；BANQI-QUAL U1 实锤：零公式讲次幻灯数学行
57-61）。本模块把讲次的幻灯 OCR 素材整理成「课件侧第二源」投喂总结链：

- 去重：完全重复帧（同 ``source_sha256``/``dhash``/归一化文本）折叠为一个
  逻辑页（``courseware_plan.v1`` 的 exact-duplicate 折叠同规则），消除同一页
  连续投喂几十次的稀释与膨胀；
- 公式抽取：本地闭集正则检出幻灯 OCR 文本中的数学行，作为逐页/讲级公式单
  （纯本地计算，零外发；进入 LLM 通道的只有与现网 ppt_pages 同源的 OCR 文本）；
- 对齐：讲级起步（整副 deck 属一讲，目录即讲级粒度），窗口级按 ``created_sec``
  落窗、窗外页近邻兜底（不丢内容）。

Fail-closed 契约：无可用页/全 junk → ``build_slide_pack`` 返回 ``None``，
调用方按无 slide 源的既有路径逐位执行；本模块的任何异常都不允许改变总结链
的成败语义。
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# 讲级 deck/公式单尺寸帽：prompt 经济（去重后典型讲次 deck ≈23-82 页）。
MAX_DECK_PAGES = 120
MAX_FORMULAS = 48
# 单页文本与单条公式行的字符帽（OCR 页文本长尾截断，只影响提示词数据面）。
PAGE_TEXT_CAP = 400
FORMULA_LINE_CAP = 160

# 数学行检测闭集（与验收尺 formula_gap.py 同族字符集 + worker 常见数学记号）。
# 这是「行级疑似公式」的召回网：精确性由下游 LLM 消费面把关（提示词明确
# 「疑似行需结合上下文取舍」），本层宁多勿漏。
MATH_LINE_RE = re.compile(
    r"[∫∑∏√∂∇≈≤≥±∝∞ΔλμσΩτφθεδγβπα]|=\s*[a-zA-Z]|[a-zA-Z]\s*=\s*|[×÷]|\^"
)
# LaTeX 命令行（OCR 偶尔直接吐出 LaTeX 片段，如 \frac）。
LATEXISH_LINE_RE = re.compile(
    r"\\(frac|sum|int|sqrt|alpha|beta|gamma|delta|mu|sigma|phi|tau|omega|partial|left|right)"
)
# 归一化去重键的噪声字符（空白/常见 OCR 条anti噪声）——仅用于去重键，不改文本。
_DEDUPE_STRIP_RE = re.compile(r"\s+")


def _dedupe_key(page: dict[str, Any]) -> str:
    """Exact-duplicate identity: content sha → dhash → normalized text hash."""
    sha = str(page.get("source_sha256") or "").strip().lower()
    if sha:
        return "sha:" + sha
    dhash = str(page.get("dhash") or "").strip().lower()
    if dhash:
        return "dhash:" + dhash
    text = _DEDUPE_STRIP_RE.sub("", str(page.get("text") or ""))
    if text:
        return "text:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    return ""


def dedupe_pages(ppt_pages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Collapse exact-duplicate slide frames into one logical page per content.

    保留首个出现的逻辑页（最早 ``created_sec``——输入按 capture 序）；后续
    同内容出现只累计 ``repeat_count``（诊断字段，不改对齐）。无身份键（无
    sha/dhash/文本）的页按独立页保留（fail-open 保内容：去重绝不丢内容页）。
    """
    deck: list[dict[str, Any]] = []
    position_by_key: dict[str, int] = {}
    for raw in ppt_pages or []:
        if not isinstance(raw, dict):
            continue  # 畸形行按垃圾丢弃（生产输入恒为 dict；防御面不保内容）
        page = raw
        key = _dedupe_key(page)
        if key and key in position_by_key:
            kept = deck[position_by_key[key]]
            kept["repeat_count"] = int(kept.get("repeat_count") or 0) + 1
            continue
        created_sec = max(0, int(page.get("created_sec") or 0))
        text = str(page.get("text") or "").strip()
        entry = {
            "page_num": max(1, int(page.get("page_num") or len(deck) + 1)),
            "created_sec": created_sec,
            "text": text[:PAGE_TEXT_CAP],
            "repeat_count": 0,
        }
        if key:
            position_by_key[key] = len(deck)
        deck.append(entry)
    return deck


def extract_formula_lines(text: str) -> list[str]:
    """Return math-looking lines from one slide's OCR text (closed-set regex)."""
    lines: list[str] = []
    for line in str(text or "").splitlines():
        candidate = line.strip()
        if not candidate or len(candidate) > FORMULA_LINE_CAP:
            continue
        if MATH_LINE_RE.search(candidate) or LATEXISH_LINE_RE.search(candidate):
            lines.append(candidate[:FORMULA_LINE_CAP])
    return lines


def document_pages_to_ppt_pages(
    document_pages: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """用户导入材料页（客户端已提取+已对齐）→ ppt_pages 兼容形状。

    V010-SHUJI-SLIDES-1 S2：平台课件腿缺失的讲次（现网 ppt_pages=0），用户
    自有导入材料（courseware 文档，学习库 ``learning_document_pages`` ×
    ``document_alignments`` 投影）可作 slide pack 第二源。行契约：
    ``{page_num, text, text_hash|source_sha256, visual_hash|dhash, start_ms}``。

    - ``created_sec = start_ms // 1000``（缺失/畸形/负值 → 0；窗口指派按
      近邻兜底归窗，内容不丢——与平台 OCR 页 0 秒页同语义）；
    - 身份键优先 ``source_sha256``/``text_hash``（页面文本身份），次
      ``dhash``/``visual_hash``；同文页由既有 exact-duplicate 折叠收敛；
    - fail-closed：非 dict 行、无文本行丢弃（文本 deck 对它们无消费价值）；
      全空输入 → ``[]``（调用方按无 slide 源路径执行）；
    - 讲位映射（哪份文档归哪一讲）由导入面对齐层负责，本层只做逐页形状
      变换，绝不跨讲硬塞。
    """
    pages: list[dict[str, Any]] = []
    for raw in document_pages or []:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        sha = str(raw.get("source_sha256") or raw.get("text_hash") or "").strip().lower()
        dhash = str(raw.get("dhash") or raw.get("visual_hash") or "").strip().lower()
        try:
            start_ms = int(float(raw.get("start_ms") or 0))
        except (TypeError, ValueError):
            start_ms = 0
        try:
            page_num = int(float(raw.get("page_num") or 0))
        except (TypeError, ValueError):
            page_num = 0
        pages.append({
            "page_num": max(1, page_num or len(pages) + 1),
            "created_sec": max(0, start_ms // 1000),
            "text": text,
            "source_sha256": sha,
            "dhash": dhash,
        })
    return pages


def build_document_slide_pack(
    document_pages: list[dict[str, Any]] | None,
    *,
    max_deck_pages: int = MAX_DECK_PAGES,
    max_formulas: int = MAX_FORMULAS,
) -> dict[str, Any] | None:
    """用户导入材料 → 讲级 slide pack；不可用 → ``None``（fail-closed）。

    与 ``build_slide_pack`` 同形状同帽（deck ≤``MAX_DECK_PAGES`` 页、公式单
    ≤``MAX_FORMULAS`` 条），下游窗口指派/提示词接线零特判。
    """
    pages = document_pages_to_ppt_pages(document_pages)
    if not pages:
        return None
    return build_slide_pack(
        pages, max_deck_pages=max_deck_pages, max_formulas=max_formulas,
    )


def build_slide_pack(
    ppt_pages: list[dict[str, Any]] | None,
    *,
    max_deck_pages: int = MAX_DECK_PAGES,
    max_formulas: int = MAX_FORMULAS,
) -> dict[str, Any] | None:
    """Build the lecture-level slide pack, or ``None`` when nothing is usable.

    形状：``{"deck": [{page_num, created_sec, text, repeat_count, formulas}],
    "formulas": [讲级去重公式行], "deck_pages": N, "raw_frames": M}``。
    空输入/全空文本 → ``None``（fail-closed：调用方按无 slide 源路径执行）。
    """
    deck = dedupe_pages(ppt_pages)
    usable = [page for page in deck if page["text"]]
    if not usable:
        return None
    if len(usable) > max(1, int(max_deck_pages)):
        usable = usable[: max(1, int(max_deck_pages))]
    seen_formulas: set[str] = set()
    formula_sheet: list[str] = []
    for page in usable:
        page["formulas"] = extract_formula_lines(page["text"])
        for line in page["formulas"]:
            key = _DEDUPE_STRIP_RE.sub("", line)
            if key and key not in seen_formulas:
                seen_formulas.add(key)
                if len(formula_sheet) < max(1, int(max_formulas)):
                    formula_sheet.append(line)
    return {
        "deck": usable,
        "formulas": formula_sheet,
        "deck_pages": len(usable),
        "raw_frames": len(ppt_pages or []),
    }


def _window_slice(pages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Compact per-window payload; ``None`` when the window got no pages."""
    if not pages:
        return None
    formulas: list[str] = []
    seen: set[str] = set()
    for page in pages:
        for line in page.get("formulas") or []:
            key = _DEDUPE_STRIP_RE.sub("", str(line))
            if key and key not in seen:
                seen.add(key)
                formulas.append(str(line))
    return {
        "deck": [
            {
                "page_num": int(page.get("page_num") or 0),
                "created_sec": max(0, int(page.get("created_sec") or 0)),
                "text": str(page.get("text") or ""),
            }
            for page in pages
        ],
        "formulas": formulas,
    }


def assign_pack_windows(
    slide_pack: dict[str, Any] | None,
    bounds: list[tuple[int, int]],
) -> list[dict[str, Any] | None]:
    """Globally assign deck pages to transcript windows, preserving content.

    ``bounds`` 为各转写窗口的 ``(lower_ms, upper_ms)``（与调用方窗口切分同序）。
    页归窗：``created_sec`` 毫秒化落入某窗区间归该窗；区间外页（0 秒页/讲末页）
    按与各窗区间的最小边界距离近邻归入唯一窗（讲级兜底——0 秒课件首页与讲末
    尾页不丢，这与 raw 帧按窗过滤丢边界页的行为刻意不同）。保持各窗内时间序。
    任一窗无页 → 该窗位 ``None``（fail-closed：绝不造空 slide 源）。
    """
    if not isinstance(slide_pack, dict):
        return [None] * len(bounds)
    deck = [page for page in (slide_pack.get("deck") or []) if isinstance(page, dict)]
    if not deck:
        return [None] * len(bounds)
    ranges = [(max(0, int(lower)), max(0, int(upper))) for lower, upper in bounds]
    buckets: list[list[dict[str, Any]]] = [[] for _ in ranges]
    for page in deck:
        moment = max(0, int(page.get("created_sec") or 0)) * 1000
        target = -1
        for index, (lower, upper) in enumerate(ranges):
            if lower <= moment <= upper:
                target = index
                break
        if target < 0 and ranges:
            target = min(
                range(len(ranges)),
                key=lambda index: min(
                    abs(moment - ranges[index][0]), abs(moment - ranges[index][1])
                ),
            )
        if target >= 0:
            buckets[target].append(page)
    return [_window_slice(bucket) for bucket in buckets]
