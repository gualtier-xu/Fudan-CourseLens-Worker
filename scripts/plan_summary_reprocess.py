# -*- coding: utf-8 -*-
"""V010-WORKER-QUAL-1 S4：总结存量重处理计划生成器（只读，零 LLM，零外联）。

背景（CONTENT-QUAL-1 取证 + 用户 2026-10-25 令「总结文字质量」）：worker
七项根修（W1-W7）落地后，旧代产物中的缺陷形态（latex 残渣/占位 caption/
元话语/页码噪声/近重复章节/terms 全空）不会自愈——受影响讲次需按本计划
重新生成总结。处理语义变更重处理=处理新鲜度律义务；重处理成本候用户批准
后执行（本脚本绝不入队、绝不外呼）。

用法（本机操作员）：
    python plan_summary_reprocess.py --db <learning.db 路径> [--out plan.json]
        [--min-defects 1] [--price-file prices.json]

产出：受影响讲次计划（sub_id/课程/缺陷计数/窗口数/token 与成本估算，
谷价+高峰双档）。执行路径（候批后）=既有客户端流
``application.enqueue_summary(course_id, sub_id, force=True)`` 逐讲触发
（或等价的客户端批量面）；DeepSeek 谷价窗=工作日 18:00-次日 09:00 + 周末
节假日全天（DEEPECON-R1 定谳：高峰仅周一至五 9-12/14-18）。

成本模型（DEEPCOST 定谳沿用，假设显式化）：
- 总结链（N 窗+合并+review_views）token ≈ 39K × (windows/9)，input:output
  ≈ 74:26（8 窗讲次标定，线性折算）；
- flash 价（$/M）：输入未命中 高峰 0.30 / 谷 0.15；输出 高峰 1.20 / 谷 0.60；
  汇率 7.2；
- 谷价窗执行再乘 0.5 已含；judge 抽检可选项另计 ≈¥0.02/讲（谷）。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from courselens_worker.summary_contract import (  # noqa: E402
    sane_formula_caption,
    sane_latex,
    strip_meta_discourse,
)

# DEEPCOST 定谳常数（product-deepcost1-result-20261010.md；假设显式可覆写）。
BASE_SUMMARY_TOKENS = 39_000      # 8 窗+合并+review_views 标定讲
BASE_SUMMARY_WINDOWS = 9          # 8 转写窗 + 1 合并
INPUT_SHARE = 0.74
USD_CNY = 7.2
PRICE_USD_PER_M = {
    "input_miss_peak": 0.30, "input_miss_valley": 0.15,
    "output_peak": 1.20, "output_valley": 0.60,
}

_NOISE_POINT_RE = (
    re.compile(r"^第\s*\d+\s*页(标题|内容|目录)?\s*[：:]"),
    re.compile(r"^p\d+\s*[–—~\-]\s*p?\d+", re.IGNORECASE),
)


def _norm(value: str) -> str:
    return "".join(
        ch for ch in str(value or "").lower()
        if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"
    )


def _dice(left: str, right: str) -> float:
    def grams(text: str) -> set[str]:
        cleaned = _norm(text)
        return {cleaned[i:i + 2] for i in range(max(0, len(cleaned) - 1))}
    left_grams, right_grams = grams(left), grams(right)
    if not left_grams or not right_grams:
        return 0.0
    return 2.0 * len(left_grams & right_grams) / (len(left_grams) + len(right_grams))


def audit_artifact(data: dict) -> dict:
    """七项守卫原样复用（与 worker 校验器同一闭集）：缺陷计数。

    F9-FIX-1 包①②扩审计（2026-10-29，判据=评判道 F9-1-R2 同口径）：
    chapter_start_nonmonotonic / takeaway_zero_anchor / chapter_summary_empty /
    hollow_content 四键随判据级计数入计划——存量旧代产物（CONTENT-IMPL-1
    f87b1ca1 与 WORKER-QUAL-1 17419761 之前生成）不会自愈，重处理清单须
    覆盖时间轴完整性与泄漏面，边界=候用户批准分批执行（处理新鲜度律）。"""
    md = str(data.get("markdown") or "")
    overview = data.get("overview")
    if overview is None:
        overview = (data.get("doc") or {}).get("overview")
    overview_text = (
        overview if isinstance(overview, str)
        else " ".join(str(p) for p in overview) if isinstance(overview, list) else ""
    )
    doc = data.get("doc") or {}
    unsane = placeholders = noise = formulas_total = 0
    for section in doc.get("sections") or []:
        for formula in section.get("formulas") or []:
            formulas_total += 1
            if not sane_latex(formula.get("latex")):
                unsane += 1
            caption = formula.get("caption")
            if caption and not sane_formula_caption(caption):
                placeholders += 1
        for point in section.get("points") or []:
            if any(p.search(str(point.get("text") or "")) for p in _NOISE_POINT_RE):
                noise += 1
    chapters = data.get("chapters") or []
    near_dup = sum(
        1 for i in range(len(chapters) - 1)
        if _dice(chapters[i].get("title"), chapters[i + 1].get("title")) >= 0.75
    )
    # F9-FIX-1 包②扩审计：时间轴完整性（F9 扫描同口径=None/0 皆计零锚）。
    anchors = data.get("takeaway_anchors") or []
    zero_anchors = sum(1 for value in anchors if value is None or value == 0)
    nonmonotonic = sum(
        1 for i in range(len(chapters) - 1)
        if int(chapters[i].get("start_ms") or 0) > int(chapters[i + 1].get("start_ms") or 0)
    )
    chapter_summary_empty = sum(
        1 for item in chapters if not str(item.get("summary") or "").strip()
    )
    takeaways = [str(item or "").strip() for item in (data.get("key_takeaways") or [])]
    hollow = int(not chapters and not any(takeaways))
    meta_leak = bool(
        ("判定：" in md and "证据：" in md)
        or ("判定：" in overview_text and "证据：" in overview_text)
    )
    first_line = md.splitlines()[0] if md else ""
    return {
        "meta_leak": int(meta_leak),
        "formulas": formulas_total,
        "unsane_latex": unsane,
        "placeholder_captions": placeholders,
        "noise_points": noise,
        "near_dup_chapter_pairs": near_dup,
        "doc_terms": len(doc.get("terms") or []) if doc else None,
        "markdown_head_needs_strip": int(
            strip_meta_discourse(first_line) != first_line
        ),
        "chapter_start_nonmonotonic": nonmonotonic,
        "takeaway_zero_anchor": zero_anchors,
        "chapter_summary_empty": chapter_summary_empty,
        "hollow_content": hollow,
    }


def estimate_cost(windows: int) -> dict:
    scale = max(1, windows) / BASE_SUMMARY_WINDOWS
    tokens = BASE_SUMMARY_TOKENS * scale
    input_tokens = tokens * INPUT_SHARE
    output_tokens = tokens * (1 - INPUT_SHARE)

    def cost(price_in: float, price_out: float) -> float:
        usd = (
            input_tokens / 1e6 * price_in
            + output_tokens / 1e6 * price_out
        )
        return round(usd * USD_CNY, 3)

    return {
        "windows": windows,
        "tokens_total": int(tokens),
        "cny_valley": cost(
            PRICE_USD_PER_M["input_miss_valley"], PRICE_USD_PER_M["output_valley"],
        ),
        "cny_peak": cost(
            PRICE_USD_PER_M["input_miss_peak"], PRICE_USD_PER_M["output_peak"],
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True, help="learning.db 路径（只读打开）")
    parser.add_argument("--out", default="", help="计划 JSON 输出路径（缺省 stdout）")
    parser.add_argument("--min-defects", type=int, default=1,
                        help="入计划门槛：缺陷类目数（缺省 1=任一缺陷即入）")
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT a.sub_id, a.course_id, a.content_json, "
            "(SELECT COUNT(*) FROM transcript_segments t WHERE t.sub_id=a.sub_id) "
            "FROM ai_artifacts a WHERE a.kind='timestamp_summary' AND a.status='ready' "
            "ORDER BY a.sub_id, a.updated_at DESC"
        ).fetchall()
    finally:
        conn.close()

    latest: dict[int, tuple] = {}
    for sub_id, course_id, content_json, segment_count in rows:
        if sub_id in latest:
            continue
        try:
            data = json.loads(content_json)
        except json.JSONDecodeError:
            continue
        latest[sub_id] = (course_id, data, int(segment_count or 0))

    plan: list[dict] = []
    for sub_id, (course_id, data, segment_count) in sorted(latest.items()):
        audit = audit_artifact(data)
        defect_kinds = sum([
            audit["meta_leak"], int(audit["unsane_latex"] > 0),
            int(audit["placeholder_captions"] > 0), int(audit["noise_points"] > 0),
            int(audit["near_dup_chapter_pairs"] > 0),
            # F9-FIX-1 包①②扩审计：时间轴完整性与结构空心面入计划判据。
            int(audit["chapter_start_nonmonotonic"] > 0),
            int(audit["takeaway_zero_anchor"] > 0),
            int(audit["chapter_summary_empty"] > 0),
            audit["hollow_content"],
        ])
        if defect_kinds < max(1, args.min_defects):
            continue
        windows = max(1, math.ceil(segment_count / 120)) + 1  # +合并位
        entry = {
            "sub_id": sub_id,
            "course_id": course_id,
            "defect_kinds": defect_kinds,
            **audit,
            "estimate": estimate_cost(windows),
        }
        plan.append(entry)

    total_valley = round(sum(e["estimate"]["cny_valley"] for e in plan), 2)
    total_peak = round(sum(e["estimate"]["cny_peak"] for e in plan), 2)
    result = {
        "generated_for": "V010-WORKER-QUAL-1 S4（候用户批准成本后执行）",
        "execution_path": "application.enqueue_summary(course_id, sub_id, force=True) 逐讲；谷价窗=工作日18:00-09:00+周末全天",
        "pricing_assumptions": {
            "model": "deepseek-flash", "usd_cny": USD_CNY,
            "base_tokens": BASE_SUMMARY_TOKENS, **PRICE_USD_PER_M,
        },
        "lecture_count": len(plan),
        "cny_total_valley": total_valley,
        "cny_total_peak": total_peak,
        "judge_sampling_optional_cny_valley": round(len(plan) * 0.02, 2),
        "plan": plan,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
        print(f"plan written: {args.out} ({len(plan)} lectures, "
              f"valley≈¥{total_valley} peak≈¥{total_peak})")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
