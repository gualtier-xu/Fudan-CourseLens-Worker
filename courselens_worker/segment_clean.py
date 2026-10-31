"""Segment-level deterministic subtitle cleaning (A7-ROUTER-P1).

段级确定性清洗两拍（消费 A3-SUBREAD-2 findings 实证规则，参数全部来自
scratch-personal/night1031-a3subread/repair_rules.json C1-C7，禁凭空造规则）：

**Part 1 ``clean_segments``（只删段、不改字、不增段）**——落 worker 转写链
``transcribe`` 之后、term 深校对之前（garble/filler 段不烧 term 深校对 token；
Tier2 清洗计划锚失配即整段回退 Tier1，fail-closed 既有语义不变）：

- C4 假名/谚文主导段整段删（规则原文「非预期字符集（假名/谚文）整段删」；
  守卫=脚本主导判据：假名/谚文字数 > 汉字数的段才删，混合正常中文段保留）。
- C3 相邻同文段去重（「保留首段，时间戳取并集」；证据键随首段，被并段
  文本留档 raw_* 层）。
- C2 filler-only 段清洗（「单段≤2 汉字且∈闭集且邻段间隔<1s→删」；闭集=A3
  实证集（拉丁词大小写不敏感）；「连续≥3 段 filler-only 保留中间 1 段作节奏」；
  双邻间隔都 <1s 才删（宁漏勿错），首末段缺邻=不删（尾部归 C1）。
- C1 课末闲聊尾段截断（「以收束语锚点+其后段 filler 密度>70% 判定尾段并
  截断；保留锚点前正常答疑」）：收束语锚点闭集取规则原文具名形（「那这个
  这样」「下课」——「OK，下课」由「下课」子串覆盖）；锚点须落在讲次时长
  ≥80% 处（课末语义的时刻化，防「下课之前…」中段误锚）；锚点后 ≥5 段且
  低信息密度 ≥70% 才截断。低信息=filler-only ∪ ≤3 字 ∪ 脚本 garble（A3
  「密度低、信息零」的碎段构成；逐段误计在多段尾窗内互相抵消）。

**Part 2 ``flag_segments``（只加闭集标记键、零删改）**——落终界（ct-punc /
display clean 之后），键经导入面 ``sanitize_evidence_metadata`` 有界留存：

- C4b 连续英文短行低置信标记（「连续英文短行无上下文语义→标记低置信」；
  只标不删——英文短行可能是合法术语，删=越权；守卫=无数字/运算符且 3-10
  token（A3 实证杂线 3-9 token 形态），公式与 ≤2 token 合法术语形态不标）
  → ``low_confidence_en``。
- C6 无终止标点段合并评估标记（「段尾无标点且语义未尽（不以的/了/是/和/
  与/或结尾）→与下段合并评估」；只标不并——合并=结构性操作归重清洗批
  人工评估）→ ``merge_candidate``；「与下段」的时刻化=间隔 ≤5s（与展示链
  折叠窗 _SHAPE_FOLD_WINDOW_MS 同源）。
- C7 大间隙段标记（「仅打点标记，供播放器跳转，不删段」）→ ``long_gap_ms``。

忠实层与遥测：Part 1 不写任何文本、不动 ``raw_*`` 键与证据键（C3 并段保留
首段证据身份）；Part 2 零删改；统计只含计数与闭集词，零字幕内容。全函数
纯确定性、零 LLM、零外联；无 A3 证据之外的任何词表内联。
"""

from __future__ import annotations

import re
from typing import Any

# ---- C2 filler-only 闭集（A3-SUBREAD-2 repair_rules.json C2 实证集）--------
# 拉丁词按小写比对；非拉丁字符逐字比对。
_FILLER_TOKENS = frozenset({
    "嗯", "啊", "呃", "哦", "噢", "对", "好", "行", "是", "不", "来", "啦",
    "耶", "唉", "哎", "嘛", "吧", "呢", "哈", "裤", "胖", "走",
    "yeah", "ok", "the", "x",
})
_EDGE_PUNCT = "。！？，、；：,.!?;:…—·~～\"'（）()「」『』《》〈〉 \u00a0\t"
_C2_MAX_GAP_MS = 1000
_C2_RUN_KEEP_MIN = 3

# ---- C2b 课间静默封顶段群（A2-SUBREAD-1 清洗规则 1/2 实证）------------------
# 源 SRT 30s 封顶把课间/午休静默切成连续长时长贫文本段（7 讲 18 群实证：
# 连续 ≥3 段、单段 29.5-30.5s、文本 ≤4 字）——整群删，群间大间隙由 C7 标记
# 接力（删后邻段 gap>30s 自动落 long_gap_ms）。
_C2B_MIN_DURATION_MS = 25_000
_C2B_MAX_DURATION_MS = 35_000
_C2B_MAX_CHARS = 4
_C2B_RUN_MIN = 3
_C2B_RUN_GAP_MS = 2_000

# ---- C1 收束语锚点闭集（repair_rules.json C1 具名形）------------------------
# 「OK，下课」由「下课」子串覆盖；「就这样」单形中段歧义不入（锚点仅取
# C1 原文具名的复合形）。
_C1_ANCHORS = ("那这个就这样", "下课")
_C1_TAIL_POSITION_RATIO = 0.8
_C1_MIN_TAIL_SEGMENTS = 5
_C1_FILLER_DENSITY = 0.7
_C1_LOW_INFO_MAX_CHARS = 3

# ---- C4 假名/谚文脚本范围（「非预期字符集」）--------------------------------
_KANA_HANGUL_RE = re.compile(r"[\u3040-\u30ff\uac00-\ud7af]")
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")

# ---- C6/C7/C4b 标记参数 ------------------------------------------------------
_C6_TERMINAL_PUNCT = frozenset("。！？，、；：,.!?;:…—~～")
_C6_TRAILING_EXEMPT = frozenset("的了是和与或")
_C6_MERGE_MAX_GAP_MS = 5000  # 与展示链 _SHAPE_FOLD_WINDOW_MS 同源
_C7_LONG_GAP_MS = 30000
_C4B_MAX_TOKENS = 10

# 闭集拒因（遥测词，零内容）。
CLEAN_REASONS = ("c1_tail", "c2_filler", "c3_dup", "c4_script")


def _norm_text(segment: dict[str, Any]) -> str:
    """段文本规整视图（去边缘标点/空白；判定用，不回写）。"""
    return str(segment.get("text") or "").strip().strip(_EDGE_PUNCT).strip()


def _is_filler_only(segment: dict[str, Any]) -> bool:
    """C2 判据：规整后整段 ∈ A3 filler 闭集（拉丁词大小写不敏感）。

    闭集成员即守卫（A3 集含 yeah/OK/the 等 2-4 字母拉丁形，「≤2 汉字」约束
    已由闭集本身编码——集外任何多字中文段一律不算 filler）。
    """
    text = _norm_text(segment)
    if not text:
        return False
    return text in _FILLER_TOKENS or text.lower() in _FILLER_TOKENS


def _is_script_garble(segment: dict[str, Any]) -> bool:
    """C4 判据：假名/谚文字数 > 汉字数（脚本主导=garble 段）。"""
    text = str(segment.get("text") or "")
    return len(_KANA_HANGUL_RE.findall(text)) > len(_HAN_RE.findall(text))


def _is_low_information(segment: dict[str, Any]) -> bool:
    """C1 尾窗低信息判据：filler-only ∪ ≤3 字 ∪ 脚本 garble。"""
    if _is_filler_only(segment) or _is_script_garble(segment):
        return True
    return len(_norm_text(segment)) <= _C1_LOW_INFO_MAX_CHARS


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _row_key(row: dict[str, Any]) -> tuple[int, int, int]:
    """稳定序键：(start_ms, end_ms, 原序)——同刻段保持输入序。"""
    return (
        _int_or(row.get("start_ms"), 0),
        _int_or(row.get("end_ms"), 0),
        row.get("_input_order", 0),
    )


def _c3_dup_indexes(rows: list[dict[str, Any]]) -> set[int]:
    """C3：相邻文本完全相同的后继段下标集（并段目标；首段保留）。"""
    drop: set[int] = set()
    for index in range(1, len(rows)):
        prev_text = str(rows[index - 1].get("text") or "").strip()
        text = str(rows[index].get("text") or "").strip()
        if prev_text and text == prev_text:
            drop.add(index)
    return drop


def _c2_drop_indexes(rows: list[dict[str, Any]]) -> set[int]:
    """C2：filler-only 可删下标集（双邻间隔守卫 + ≥3 连串保留中位节奏段）。"""
    filler = [index for index, row in enumerate(rows) if _is_filler_only(row)]
    # 连续 filler-only 连串（序列相邻，无论间隔）。
    runs: list[list[int]] = []
    for index in filler:
        if runs and runs[-1][-1] == index - 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    keep_rhythm: set[int] = set()
    for run in runs:
        if len(run) >= _C2_RUN_KEEP_MIN:
            keep_rhythm.add(run[(len(run) - 1) // 2])
    drop: set[int] = set()
    for index in filler:
        if index in keep_rhythm:
            continue
        row = rows[index]
        start = _int_or(row.get("start_ms"), 0)
        end = _int_or(row.get("end_ms"), 0)
        if index > 0:
            prev_end = _int_or(rows[index - 1].get("end_ms"), 0)
            if start - prev_end >= _C2_MAX_GAP_MS:
                continue
        else:
            continue  # 首段缺前邻=不删（宁漏勿错）
        if index + 1 < len(rows):
            next_start = _int_or(rows[index + 1].get("start_ms"), 0)
            if next_start - end >= _C2_MAX_GAP_MS:
                continue
        else:
            continue  # 末段缺后邻=不删（尾部归 C1）
        drop.add(index)
    return drop


def _is_silence_cap(segment: dict[str, Any]) -> bool:
    """C2b 判据：封顶时长（25-35s）且规整文本 ≤4 字的贫文本段。"""
    start = _int_or(segment.get("start_ms"), 0)
    end = _int_or(segment.get("end_ms"), 0)
    if not (_C2B_MIN_DURATION_MS <= end - start <= _C2B_MAX_DURATION_MS):
        return False
    return 0 < len(_norm_text(segment)) <= _C2B_MAX_CHARS


def _c2b_drop_indexes(rows: list[dict[str, Any]]) -> set[int]:
    """C2b：连续 ≥3 段封顶贫文本段成群 → 整群删（课间/午休静默群）。"""
    cap = [index for index, row in enumerate(rows) if _is_silence_cap(row)]
    drop: set[int] = set()
    run: list[int] = []
    for index in cap:
        if run:
            gap = _int_or(rows[index].get("start_ms"), 0) - _int_or(rows[run[-1]].get("end_ms"), 0)
            if index == run[-1] + 1 and 0 <= gap <= _C2B_RUN_GAP_MS:
                run.append(index)
                continue
            if len(run) >= _C2B_RUN_MIN:
                drop.update(run)
            run = []
        run.append(index)
    if len(run) >= _C2B_RUN_MIN:
        drop.update(run)
    return drop


def _c1_tail_start(rows: list[dict[str, Any]]) -> int:
    """C1：尾段截断起点（首删行下标）；无满足锚点返回 len(rows)。"""
    if not rows:
        return 0
    lecture_end = max(_int_or(row.get("end_ms"), 0) for row in rows)
    for index in range(len(rows) - 1, -1, -1):
        text = str(rows[index].get("text") or "")
        anchor = next(
            (item for item in _C1_ANCHORS if item in text), None
        )
        if anchor is None:
            continue
        if _int_or(rows[index].get("start_ms"), 0) < lecture_end * _C1_TAIL_POSITION_RATIO:
            continue  # 锚点不在课末时段（「下课之前…」中段形态）→ 该锚不用
        tail = rows[index + 1:]
        if len(tail) < _C1_MIN_TAIL_SEGMENTS:
            continue
        density = sum(1 for row in tail if _is_low_information(row)) / len(tail)
        if density >= _C1_FILLER_DENSITY:
            return index + 1
    return len(rows)


def clean_segments(
    segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Part 1 段级确定性清洗：C4→C3→C2→C1，只删段并段、不改字。

    全部判据在输入行上一次性分类（删除决策互不级联，顺序无关、幂等）；
    C3 并段把并段集的 end_ms 取并到首段。返回 (清洗后段列表, 闭集计数)。
    非 dict 行原样跳过（与 normalize_segments 边界一致）。
    """
    stats = {f"c{key}_dropped": 0 for key in ("1", "2", "2b", "3", "4")}
    rows = [dict(row) for row in segments if isinstance(row, dict)]
    for order, row in enumerate(rows):
        row["_input_order"] = order
    rows.sort(key=_row_key)

    c4_drop = {index for index, row in enumerate(rows) if _is_script_garble(row)}
    c3_drop = _c3_dup_indexes(rows)
    c2_drop = _c2_drop_indexes(rows)
    c2b_drop = _c2b_drop_indexes(rows)
    tail_start = _c1_tail_start(rows)
    c1_drop = set(range(tail_start, len(rows)))

    # C3 并段时间戳并集：后继同文段的 end_ms 并回首段。
    merged_end: dict[int, int] = {}
    for index in sorted(c3_drop):
        head = index - 1
        while head in c3_drop:
            head -= 1
        merged_end[head] = max(
            merged_end.get(head, _int_or(rows[head].get("end_ms"), 0)),
            _int_or(rows[index].get("end_ms"), 0),
        )
    for head, end in merged_end.items():
        if end > _int_or(rows[head].get("end_ms"), 0):
            rows[head]["end_ms"] = end

    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row.pop("_input_order", None)
        if index in c1_drop:
            stats["c1_dropped"] += 1
            continue
        if index in c4_drop:
            stats["c4_dropped"] += 1
            continue
        if index in c2b_drop:
            stats["c2b_dropped"] += 1
            continue
        if index in c2_drop:
            stats["c2_dropped"] += 1
            continue
        if index in c3_drop:
            stats["c3_dropped"] += 1
            continue
        output.append(row)
    return output, stats


def flag_segments(segments: list[dict[str, Any]]) -> dict[str, int]:
    """Part 2 终界标记（C4b/C6/C7）：就地加闭集键，零删改零文本变化。

    返回闭集计数；键 ``low_confidence_en``/``merge_candidate``/``long_gap_ms``
    经导入面 sanitize_evidence_metadata 有界留存（未知安全加性键通道）。
    """
    stats = {"c4b_flagged": 0, "c6_flagged": 0, "c7_flagged": 0}
    rows = [row for row in segments if isinstance(row, dict)]
    for index, row in enumerate(rows):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        # C4b：无汉字、无数字/运算符、3-10 个空白 token 的英文短行 → 低置信
        # 标记（A3 杂线 3-9 token 实证形：mode age transfer / tHe local Body）。
        # 守卫（正当使用零误伤）：含数字或 =+*/ 的行=公式/单位形态不标；
        # ≤2 token=合法英文术语不标。
        if (
            not _HAN_RE.search(text)
            and re.search(r"[A-Za-z]", text)
            and not re.search(r"[0-9=+*/]", text)
            and 3 <= len(text.split()) <= _C4B_MAX_TOKENS
        ):
            row["low_confidence_en"] = 1
            stats["c4b_flagged"] += 1
        # C7：与前段间隔 >30s → 大间隙标记（供播放器跳转）。
        if index > 0:
            gap = _int_or(row.get("start_ms"), 0) - _int_or(rows[index - 1].get("end_ms"), 0)
            if gap > _C7_LONG_GAP_MS:
                row["long_gap_ms"] = gap
                stats["c7_flagged"] += 1
        # C6：段尾无终止标点、不以免挂尾字结尾、与下段间隔 ≤5s → 合并评估。
        if index + 1 < len(rows):
            last = text[-1] if text else ""
            next_start = _int_or(rows[index + 1].get("start_ms"), 0)
            gap_next = next_start - _int_or(row.get("end_ms"), 0)
            if (
                last
                and last not in _C6_TERMINAL_PUNCT
                and last not in _C6_TRAILING_EXEMPT
                and 0 <= gap_next <= _C6_MERGE_MAX_GAP_MS
            ):
                row["merge_candidate"] = 1
                stats["c6_flagged"] += 1
    return stats
