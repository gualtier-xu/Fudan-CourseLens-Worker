"""字幕数字/日期类近音错字守卫（ASR-CORRECT-R1）。

shared 协议域（与 ``evidence_contract``/``subtitle_display_contract`` 同域）：
纯标准库、确定性、零第三方依赖、零 I/O；worker（总结链日期交叉验证）与
客户端（字幕/搜索展示标注、radar 纠偏）共同消费。

问题（ASR-CORRECT-R1 任务书 §实锤案例，用户原话「这不是老师的口误，是你
ASR 的错误！而且经过 AI 处理后根本看不出来有问题」）：近音数字错字穿透
全部下游——字幕原文误导（搜「期中」看到「八月三号」）、AI 总结隐式交叉
推断出正确日期反而把错误「洗白」、搜索/radar 面同源污染。

产品纪律（任务书 §S2，冻结）：**字幕原文永不静默改写**——本模块只产出
「存疑批注」（additive 键 ``date_note``，原话保留+批注呈现），绝不改写
``text``；修正价值以「老师自纠说法为准」的人话批注呈现（人情味文案，
像耐心的同学在解释，不技术责备）。

检测面（v1 两器，全部确定性零 LLM token）：

- **D1 自纠锚**（任务书 S1-① 强信号）：日期主张句（「X月X号/日」）±3 段
  内出现老师自我纠正信号（说错/讲错/更正/口误）且纠正句含月份主张时，
  产出 ``self_corrected`` 批注（含 ``corrected_claim`` 纠正值，如
  「八月三号」→「11月3号」）；纠正句无月份时仍产出纯存疑批注（宁注勿漏，
  批注本身不改写、误注无害）。
- **D2 星期一致性**：全月日主张（或其 ±5 段窗内）存在「周X/星期X」主张，
  且调用方给出版本年 ``year`` 时，历法核验该日是否确为周X；矛盾=识别
  存疑批注。无 ``year`` 时本器静默跳过（绝不臆测年份）。
  已知年份来源：讲次标题日期前缀（如「2026-10-19第4-7节」→2026）由
  消费方解析后传入；radar 侧自带 lecture_date/semester 锚不经此路。

误报盾（S1 全量 79 句人工核判校准）：序数/编号/公式「二号粒子/十六号
元素/十四号米」类形态不构成月日主张（须「月」字在场）→ 不触发；「第五周」
类周次表述不匹配星期词闭集（周后须紧跟星期名）→ 不触发；「说错」窗口外
（±3 之外）的主张不标。

批注形状（additive 键值，契约外键按 evidence.v1 加法容忍惯例，不归
``subtitle.display.v2`` 闭集管辖）::

    {"kind": "self_corrected" | "weekday_conflict",
     "claim": "八月三号",
     "corrected_claim": "11月3号" | null,
     "start_ms": 9752848, "end_ms": 9759153,
     "note": "<人情味一句话批注>"}

消费方义务：呈现层把批注与人话文案一并可见（修正不静默）；任何消费方
不得以批注为据改写证据层文本。worker 总结链消费
``summary_prompt_suffix``（把纠正值与「须提字幕原文差异」义务注入窗
prompt）；radar 消费 ``corrected_claim`` 做候选纠偏（原句 quote 恒
verbatim，纠偏只作用于解析候选，仍过既有合理性门）。

遥测：本模块零遥测零 I/O；消费方遥测沿用各自既有纪律。
钉面=tests/test_subtitle_date_guard.py（含 685785 实锤样例回归钉）。
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any

__all__ = [
    "DATE_NOTE_KEY",
    "GUARD_VERSION",
    "date_claims",
    "annotate_segments",
    "apply_date_notes",
    "summary_prompt_suffix",
    "note_start_ms",
    "note_end_ms",
]

# additive 键名与版本（消费方按版本兼容；算法升级只影响新批注，存量
# 批注随下一次标注刷新，无落盘态=无迁移面）。
DATE_NOTE_KEY = "date_note"
GUARD_VERSION = 1

# ---- 形态闭集 --------------------------------------------------------------

# 全月日主张：中文数字月（一~十二，含「十一」「十二」近音高危形态）+
# 中文/阿拉伯日 + 号/日 后缀。须「月」字在场——序数/编号/公式类
# 「二号粒子」「十六号元素」不构成主张（S1 误报盾①）。
_CN_MD_RE = re.compile(
    r"(?P<month>十[一二]?|[一二三四五六七八九]|[一二三四五六七八九十]{2,3})月"
    r"(?P<day>三十[一]?|[一二三四五六七八九十]{1,3})[号日]"
)
_AR_MD_RE = re.compile(r"(?P<month>\d{1,2})月(?P<day>\d{1,2})[号日]")

# 老师自我纠正信号闭集（口语实录形态；「不是X是Y」形态含混过高，v1 不收）。
_SELF_CORRECT_RE = re.compile(r"说错|讲错|更正|口误")

# 纠正句月份主张（老师改口常说阿拉伯数字——ASR 对清晰重读的改口转写更准，
# 实锤案例「11月，说错了。」即阿数形态）。
_CORRECT_MONTH_RE = re.compile(
    r"(\d{1,2}|十[一二]?|[一二三四五六七八九]|[一二三四五六七八九十]{2,3})月"
)

# 星期主张闭集（周后紧跟星期名才匹配；「第五周」「下周有」类周次/周泛指
# 不匹配=S1 误报盾②）。
_WEEKDAY_RE = re.compile(r"周([一二三四五六日天])|星期([一二三四五六日天])|礼拜([一二三四五六日天])")
_WEEKDAY_INDEX = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_WEEKDAY_NAME = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")

# 窗口（S1 实测校准：实锤案例 主张→自纠 Δ3 段、主张→星期印证 Δ5 段）。
_CORRECT_WINDOW = 3
_WEEKDAY_WINDOW = 5

_CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_num(text: str) -> int | None:
    """中文数字→整数（支持 1~99 的口播日期形态；解析失败返回 None）。"""
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text in _CN_DIGIT:
        return _CN_DIGIT[text]
    if text == "十":
        return 10
    if text.startswith("十"):
        rest = _CN_DIGIT.get(text[1:])
        return 10 + rest if rest is not None else None
    if text.endswith("十"):
        head = _CN_DIGIT.get(text[:-1])
        return head * 10 if head is not None else None
    if "十" in text:
        head, _, tail = text.partition("十")
        h, t = _CN_DIGIT.get(head), _CN_DIGIT.get(tail)
        if h is not None and t is not None:
            return h * 10 + t
    return None


def _claim_parts(match: re.Match[str]) -> tuple[int, int, str, str] | None:
    """匹配→(month, day, claim 原文, 日后缀)；非法日期（2月30号）返回 None。"""
    month_raw, day_raw = match.group("month"), match.group("day")
    month, day = _cn_num(month_raw), _cn_num(day_raw)
    if month is None or day is None:
        return None
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return None
    claim = match.group(0)
    suffix = claim[-1]
    return month, day, claim, suffix


def date_claims(text: str) -> list[tuple[int, int, str, str]]:
    """一段文本中的全部月日主张（按出现序）：(month, day, claim, suffix)。"""
    out: list[tuple[int, int, str, str]] = []
    for match in _AR_MD_RE.finditer(text):
        parts = _claim_parts(match)
        if parts:
            out.append(parts)
    for match in _CN_MD_RE.finditer(text):
        parts = _claim_parts(match)
        if parts and not any(p[2] == parts[2] for p in out):
            out.append(parts)
    return out


def _text_of(item: Any) -> str:
    return str(item.get("text") or "") if isinstance(item, dict) else ""


def _start_ms_of(item: Any) -> int:
    try:
        return int(item.get("start_ms") or 0) if isinstance(item, dict) else 0
    except (TypeError, ValueError):
        return 0


def _end_ms_of(item: Any) -> int:
    try:
        return int(item.get("end_ms") or 0) if isinstance(item, dict) else 0
    except (TypeError, ValueError):
        return 0


def _self_corrected_note(
    segments: list[Any], index: int, month: int, day: int, claim: str, suffix: str,
) -> dict[str, Any] | None:
    """D1：主张句 ±3 段内找老师自纠与纠正月份（实锤形态「11月，说错了。」）。"""
    lo = max(0, index - _CORRECT_WINDOW)
    hi = min(len(segments), index + _CORRECT_WINDOW + 1)
    for j in range(lo, hi):
        if j == index:
            continue
        text = _text_of(segments[j])
        if not text or not _SELF_CORRECT_RE.search(text):
            continue
        # 纠正句含完整月日主张（「不是X，是十月六号」）时优先整体采用。
        corrected_full = date_claims(text)
        if corrected_full:
            m2, d2, claim2, _ = corrected_full[0]
            if (m2, d2) == (month, day):
                continue  # 老师复述同值=非纠错，继续扫窗
            return {
                "kind": "self_corrected",
                "claim": claim,
                "corrected_claim": f"{m2}月{d2}号",
                "note": (
                    f"日期存疑：字幕可能把这里听错了——原文写作「{claim}」，"
                    f"老师随后改口「{claim2}」，请以改口为准。"
                ),
            }
        month_match = _CORRECT_MONTH_RE.search(text)
        corrected_month = _cn_num(month_match.group(1)) if month_match else None
        if corrected_month is not None:
            if corrected_month == month:
                if j < index:
                    # 改口句在主张之前且月份一致：这句主张是老师改口后的正确
                    # 复述，不是识别错误（UXCHECK-1 走查误报根修；与上方
                    # 「复述同值=非纠错」同一设计意图的月-only 形态补全）。
                    return None
                # 主张在前、月份相同的「说错了」在后：真·日后纠形态，落纯存疑。
            else:
                corrected_claim = f"{corrected_month}月{day}{suffix}"
                note_text = (
                    f"日期存疑：字幕可能把这里听错了——原文写作「{claim}」，"
                    f"老师随后改口「{month_match.group(0)}」"
                    f"（{_cn_month_name(month)}与"
                    f"{_cn_month_name(corrected_month)}读音相近），"
                    f"请以改口为准（{corrected_month}月{day}日）。"
                )
                return {
                    "kind": "self_corrected",
                    "claim": claim,
                    "corrected_claim": corrected_claim,
                    "note": note_text,
                }
        # 自纠信号在场但无月份主张：仍产出纯存疑批注（宁注勿漏，不改写）。
        return {
            "kind": "self_corrected",
            "claim": claim,
            "corrected_claim": None,
            "note": (
                f"日期存疑：字幕这里的日期可能听错了——老师随后当场改口，"
                "但没提到具体是哪天，请结合老师后面的说法理解。"
            ),
        }
    return None


def _cn_month_name(month: int) -> str:
    """月份口语名（近音对照展示用；11→「十一」）。"""
    if month == 10:
        return "十"
    if 11 <= month <= 19:
        return "十" + "一二三四五六七八九"[month - 11]
    if 1 <= month <= 9:
        return "一二三四五六七八九"[month - 1]
    if month == 20:
        return "二十"
    return str(month)


def _weekday_conflict_note(
    segments: list[Any], index: int, month: int, day: int, claim: str, year: int,
) -> dict[str, Any] | None:
    """D2：主张句（含 ±5 段窗）内的星期主张与历法矛盾→存疑批注。"""
    lo = max(0, index - _WEEKDAY_WINDOW)
    hi = min(len(segments), index + _WEEKDAY_WINDOW + 1)
    for j in range(lo, hi):
        text = _text_of(segments[j])
        if not text:
            continue
        weekday_match = _WEEKDAY_RE.search(text)
        if not weekday_match:
            continue
        claimed_weekday = _WEEKDAY_INDEX.get(
            next(g for g in weekday_match.groups() if g)
        )
        if claimed_weekday is None:
            continue
        try:
            actual = _dt.date(year, month, day).weekday()
        except ValueError:
            return None
        if actual == claimed_weekday:
            return None
        return {
            "kind": "weekday_conflict",
            "claim": claim,
            "corrected_claim": None,
            "note": (
                f"日期存疑：字幕可能把这里听错了——说「{claim}」是"
                f"「{_WEEKDAY_NAME[claimed_weekday]}」，但 {year} 年那天其实是"
                f"{_WEEKDAY_NAME[actual]}，请以老师说法为准。"
            ),
        }
    return None


def annotate_segments(
    segments: list[Any], *, year: int | None = None,
) -> list[dict[str, Any] | None]:
    """按输入序返回批注列表（对齐输入长度；无批注位=None）。

    同段同时命中 D1/D2 时保留 D1（自纠锚含纠正值，优先级更高）；批注附
    ``start_ms``/``end_ms`` 锚（消费方按时间对齐搜索命中）。
    """
    notes: list[dict[str, Any] | None] = [None] * len(segments)
    for index, segment in enumerate(segments):
        text = _text_of(segment)
        if not text:
            continue
        for month, day, claim, suffix in date_claims(text):
            note = _self_corrected_note(segments, index, month, day, claim, suffix)
            if note is None and year is not None:
                note = _weekday_conflict_note(segments, index, month, day, claim, year)
            if note is not None:
                note["start_ms"] = _start_ms_of(segment)
                note["end_ms"] = _end_ms_of(segment)
                notes[index] = note
                break
    return notes


def apply_date_notes(
    segments: list[Any], *, year: int | None = None,
) -> list[dict[str, Any]]:
    """返回新列表：命中段附 ``date_note`` additive 键，其余段与原值逐位不变。

    纯函数：不改输入、绝不触碰 ``text``（证据层 verbatim 纪律）。
    """
    notes = annotate_segments(segments, year=year)
    if not any(notes):
        return [dict(segment) for segment in segments]
    out: list[dict[str, Any]] = []
    for segment, note in zip(segments, notes):
        row = dict(segment)
        if note is not None:
            row[DATE_NOTE_KEY] = note
        out.append(row)
    return out


def note_start_ms(note: dict[str, Any] | None) -> int:
    try:
        return int(note.get("start_ms") or 0) if isinstance(note, dict) else 0
    except (TypeError, ValueError):
        return 0


def note_end_ms(note: dict[str, Any] | None) -> int:
    try:
        return int(note.get("end_ms") or 0) if isinstance(note, dict) else 0
    except (TypeError, ValueError):
        return 0


def summary_prompt_suffix(notes: list[dict[str, Any] | None]) -> str:
    """worker 总结链日期交叉验证后缀（仅当批注在场；空批注=空串零增字）。

    修正不静默义务成文进 prompt：笔记引用纠正值时必须带「字幕原文作…」
    括注——直接回应「经过 AI 处理后根本看不出来有问题」的用户痛点。
    """
    present = [note for note in notes if note]
    if not present:
        return ""
    lines = [
        "日期存疑标注（修正不静默）：本窗字幕中以下日期可能被语音识别听错："
    ]
    for note in present:
        claim = str(note.get("claim") or "")
        corrected = note.get("corrected_claim")
        if corrected:
            lines.append(
                f"- 字幕作「{claim}」，老师随后改口为「{corrected}」。"
                f"若笔记引用该日期，写成「{corrected}（字幕原文作「{claim}」，"
                "是识别错误，以老师改口为准）」，禁止只写正确日期而不提差异。"
            )
        else:
            lines.append(
                f"- 字幕作「{claim}」，老师当场有改口；笔记引用该日期时注明"
                "「以老师最新说法为准」。"
            )
    return "\n".join(lines)
