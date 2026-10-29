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

import re
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
# BANQI-FIX1（685785 三连败定谳，2026-10-23）：doc 候选锚拯救阈值=chapters 锚
# 拯救（llm QSCOREFIX-1④ / RR-QWIN-1 同族）同值同义；本模块零依赖 llm 的独立
# 同式实现（与 _boundary_similarity 同族字符二元组 Dice）。fail-closed 语义
# 零放宽：白名单外锚先派生、派生不达阈值宁可丢段/丢点，绝不编锚。
_DOC_ANCHOR_MIN_SCORE = 0.22

# ---- V010-WORKER-QUAL-1（2026-10-26，用户令「总结文字质量」七项根修）--------
# 文字质量确定性守卫闭集：全部零 LLM、逐条目 fail-closed（丢条不丢段、丢段
# 不丢档），判据=CONTENT-QUAL-1 全库取证的真实缺陷形态（680954/27738/39178）。

# W1 公式残渣门（新生成守卫；存量=重处理方案另批）：latex 字段四类显然坏式
# 逐条丢弃——中文入式（\beta不宜太高）、悬垂尾符（nco =）、双等号（IB==70\mu A）、
# 自指（Icbo =(1+β)Icbo，左式单标识符在右式标识符集中原样再现，len≥2 防
# v=v_0+at 误伤）。合法式（f_T=1/(2π\tau_F)）零触碰。
_LATEX_CJK_RE = re.compile(r"[\u4e00-\u9fa5]")
_LATEX_DANGLING_TAIL_RE = re.compile(r"[=+*,;:\\/\-]\s*$")
_LATEX_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LATEX_SELF_REF_MIN_TOKEN = 2


def sane_latex(latex: Any) -> bool:
    """公式 latex 字段健全性门：显然坏式=False（丢弃该条，不影响所在段）。"""
    text = str(latex or "").strip()
    if not text:
        return False
    if _LATEX_CJK_RE.search(text):
        return False
    if text.startswith("=") or "==" in text:
        return False
    if _LATEX_DANGLING_TAIL_RE.search(text):
        return False
    if "=" in text:
        left, right = text.split("=", 1)
        left_idents = _LATEX_IDENT_RE.findall(left)
        if len(left_idents) == 1 and len(left_idents[0]) >= _LATEX_SELF_REF_MIN_TOKEN:
            if left_idents[0] in set(_LATEX_IDENT_RE.findall(right)):
                return False
    return True


# W2 caption 占位过滤：模型把输入描述「课件公式行」逐条回填为 caption
# （680954 实测 39/39）。闭集匹配占位词=丢 caption 键保 latex（不丢整条）。
_FORMULA_CAPTION_PLACEHOLDERS = frozenset({"课件公式行", "课件公式", "公式行", "公式"})


def sane_formula_caption(caption: Any) -> bool:
    """caption 占位判定：命中占位闭集（或纯占位词+页码尾）=False。"""
    text = str(caption or "").strip()
    if not text:
        return False
    if text in _FORMULA_CAPTION_PLACEHOLDERS:
        return False
    base = re.sub(r"[\s\d#．.、]+$", "", text)
    return base not in _FORMULA_CAPTION_PLACEHOLDERS


# W4 要点页码噪声：纯页码/目录/标题枚举句无结论价值（CONTENT-QUAL-1 W4 实
# 样「第 5 页标题：…」「p688–p689 为第二章目录」）。闭集保守匹配=丢点不丢段。
_NOISE_POINT_PATTERNS = (
    re.compile(r"^第\s*\d+\s*页(标题|内容|目录)?\s*[：:]"),
    re.compile(r"^第\s*\d+\s*页$"),
    re.compile(r"^p\d+\s*[–—~\-]\s*p?\d+", re.IGNORECASE),
    re.compile(r"^(课件|幻灯|slide)\s*(第\s*\d+\s*页)?\s*[（(]?(标题|目录)", re.IGNORECASE),
)


def is_noise_point_text(text: Any) -> bool:
    """要点文本=纯页码/目录/标题枚举句判定（保守闭集，宁漏勿错删）。"""
    candidate = str(text or "").strip()
    if not candidate:
        return False
    return any(pattern.search(candidate) for pattern in _NOISE_POINT_PATTERNS)


# W5 元话语剥除（守卫层；主修=提示词判定内部化）：「判定：…；证据：…」是
# 课型自门控的内部结构词，禁止上屏（27738 实测概览开头直出）。闭集限定：
# 只剥「判定+证据」成对结构（含全/半角冒号），证据段以句读为界（防吞同句
# 正文），成对结构不齐=不动（正文正常讨论「判定」的讲次零误伤）。
# V010-REPROC-QUALFOLLOWUP-1 扩边（2026-10-26 六讲重处理实录归纳）：模型
# 实际发射的形态族超出原三形——「判定：短词（括注）。〔证据：…。〕」括注形
# （括注内依据/证据段分号配对不再必需）与判定句+证据句连写形，且元话语可落
# 在课程头之后的引用行（659504 实录 `> 判定：…（依据 …）`）——剥除窗口从
# 首行扩到头 _META_SCAN_HEAD_LINES 行。行界纪律：首行保留原宽松形（行中
# 60 字内先导词不阻剥，既有行为逐位保留）；行 2+ 只接「锚定形」——判定：
# 必须行首（容 H1/引用前缀），正文句中提及（「正文提到判定：…」）零误伤
# （既有钉边界原样保留）。黏连无句号界定的前缀不动（保守闭集纪律不变）。
_META_PAREN_RE = re.compile(
    r"[（(]\s*判定\s*[:：][^（）()]{0,120}[；;]\s*证据\s*[:：][^（）()]{0,200}[）)]"
)
_META_LINE_RE = re.compile(
    r"^\s*#{0,6}[^\n]{0,60}判定\s*[:：][^；;\n]{0,120}[；;]\s*证据\s*[:：]"
    r"[^；;\n。]{0,200}[。；;.]?\s*$"
)
_META_PREFIX_RE = re.compile(
    r"^\s*#{0,6}\s*判定\s*[:：][^；;\n]{0,120}[；;]\s*证据\s*[:：]"
    r"[^；;\n。]{0,200}[。；;.]?\s*"
)
_META_LINE_ANCHORED_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:>[^\S\n]*)*判定\s*[:：][^；;\n]{0,120}[；;]\s*证据\s*[:：]"
    r"[^；;\n。]{0,200}[。；;.]?\s*$"
)
_META_PREFIX_ANCHORED_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:>[^\S\n]*)*判定\s*[:：][^；;\n]{0,120}[；;]\s*证据\s*[:：]"
    r"[^；;\n。]{0,200}[。；;.]?\s*"
)
_META_SCAN_HEAD_LINES = 6
# 形四（括注形，自带行首锚定）：判定：短词（括注）。〔证据：…。〕——短词=
# 判定值（如「推导型」「本讲为推导型理工内容」≤16 字且无句读/括号）；括注=
# 依据/证据段（无嵌套括号）；句号/行界为元话语段右界，防吞同句正文。
_META_PAREN_KW_RE = re.compile(
    r"^(?P<lead>\s*(?:#{1,6}\s*)?(?:>[^\S\n]*)*)"
    r"判定\s*[:：]\s*[^（()）\n。；;]{1,16}\s*"
    r"[（(][^（）()\n]{0,200}[）)]\s*[。.]?"
    r"(?:\s*证据\s*[:：][^；;\n。]{0,200}[。.]?)?"
)


# F9-FIX-1 包①（内省括注剥除，452284 实录「（此处有口语化表达，未形成完整
# 结论）」上屏正文）：模型对讲者口语的自评括注是处理管线内部话语，禁止上屏。
# 闭集保守匹配=「此处」起头 + 自评关键词（口语/未形成/未展开/听不清）——
# 编辑性澄清括注（「（此处指上升沿）」F9 评判 5 处/4 讲归打磨=内容有益零误伤）
# 与其他「此处」括注（无自评关键词）一律不动。
_INTROSPECTIVE_PAREN_RE = re.compile(
    r"（此处[^（）()\n]*(?:口语|未形成|未展开|听不清)[^（）()\n]*）"
)


def strip_introspective_parens(markdown: Any) -> str:
    """内省括注剥除（F9-FIX-1 包①）：模型自评括注（对讲者口语的元评论）
    从正文剥离；闭集关键词守卫，编辑性澄清括注零误伤。"""
    return _INTROSPECTIVE_PAREN_RE.sub("", str(markdown or ""))


def _strip_keyword_paren_meta(line: str) -> str:
    """形四/五剥除：整行=元话语删行（含 H1/引用前缀形）；句号界定的前缀段
    只剥元话语段留正文；黏连无句号界定（「判定：X（Y）正文」）不动。"""
    match = _META_PAREN_KW_RE.match(line)
    if not match:
        return line
    rest = line[match.end():]
    if not rest.strip():
        return ""
    if match.group(0).rstrip().endswith(("。", ".")):
        return match.group("lead") + rest.lstrip()
    return line


def _strip_head_line(line: str, *, allow_loose: bool) -> str:
    """单行元话语剥除。allow_loose=首行宽松闭集（既有三形原样）；行 2+ 只接
    行首锚定形（_META_LINE_ANCHORED/_META_PAREN_KW/_META_PREFIX_ANCHORED）。"""
    cleaned = _META_PAREN_RE.sub("", line).rstrip()
    stripped = cleaned.strip()
    if _META_LINE_ANCHORED_RE.match(stripped):
        return ""
    if allow_loose and _META_LINE_RE.match(stripped):
        return ""
    cleaned = _strip_keyword_paren_meta(cleaned)
    if allow_loose:
        return _META_PREFIX_RE.sub("", cleaned, count=1)
    return _META_PREFIX_ANCHORED_RE.sub("", cleaned, count=1)


def strip_meta_discourse(markdown: Any) -> str:
    """头部元话语剥除（守卫层五形态闭集）：①H1 尾「（判定：…；证据：…）」
    括注；②整行判定元话语（分号配对形）；③行首判定前缀；④判定：短词（括注）
    形；⑤判定句+证据句连写形。窗口=头 6 行（元话语可落课程头之后），行 2+
    仅锚定形，证据段以句读为界，同句正文零吞并。"""
    text = str(markdown or "")
    lines = text.splitlines()
    for index, line in enumerate(lines[:_META_SCAN_HEAD_LINES]):
        if not line.strip():
            continue
        lines[index] = _strip_head_line(line, allow_loose=(index == 0))
    return "\n".join(lines)


# ---- V010-TUNING-RXFIX-1（TB-P1-1 生成侧根修，2026-10-29）：分隔符垃圾洗除
# F9EVAL/TB 实采（15 讲产物全量扫）：『。；』589 处（列表项句号后游离分号，
# v1 产物「……繁琐公式。；教师提供邮箱联系方式」形）+『：；；；』30 处（空
# 列表项，「……剧增）：；；；」形）贯通总结/闪卡/搜索三学生可见面。生成侧
# 单点根修（非渲染侧补丁）=定稿守卫族同位洗除，下游派生自净。闭集保守规则：
#   ① {。；;} 二连以上跑段 → 归并为单分隔符（跑内含句号归『。』否则『；』）；
#   ② 行尾『：；』空项形（冒号后仅分隔符=该列表项无内容）→ 剥分隔符留冒号；
#   ③ LaTeX/行内码跨度豁免（\(…\)/\[…\]/$$…$$/`…`——半角分号是合法字符）；
#   ④ 单分隔符（正常中文列表「a；b」）零触碰。零内容增益/删减，纯标点整形。
_SEPARATOR_JUNK_RUN_RE = re.compile(r"[。；;]{2,}")
_SEPARATOR_TAIL_EMPTY_RE = re.compile(r"([：:])[；;。]+(?=\s*(?:\n|$))")
_SEPARATOR_PROTECTED_SPAN_RE = re.compile(
    r"(\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|`[^`\n]*`)", re.DOTALL
)


def _strip_separator_junk_span(text: str) -> str:
    def _collapse(match: re.Match[str]) -> str:
        return "。" if "。" in match.group(0) else "；"

    cleaned = _SEPARATOR_JUNK_RUN_RE.sub(_collapse, text)
    return _SEPARATOR_TAIL_EMPTY_RE.sub(r"\1", cleaned)


def strip_separator_junk(text: Any) -> str:
    """分隔符垃圾洗除（TB-P1-1 生成侧根修）：『。；』『；；；』族闭集整形。

    生成产物（markdown/chapters/takeaways/doc 文本字段）定稿时统一过洗：
    跑段归并+行尾空项剥除，LaTeX/行内码跨度豁免，单分隔符零触碰——下游
    闪卡/搜索派生自净，存量 v1 脏产物走重生成路径（BATCH-35）。"""
    value = str(text or "")
    if not (_SEPARATOR_JUNK_RUN_RE.search(value) or _SEPARATOR_TAIL_EMPTY_RE.search(value)):
        return value
    parts = _SEPARATOR_PROTECTED_SPAN_RE.split(value)
    return "".join(
        part if index % 2 else _strip_separator_junk_span(part)
        for index, part in enumerate(parts)
    )


# W7 相邻近重复章节合并门：39178 实测「线性电源的结构与原理」(7:17) 与
# 「线性电源：结构与原理」(7:51) 连排。精确归一化去重（llm 链既有）只接同键，
# 标点/虚词差异漏过——本门用与边界感知贪心同族 Dice 相似度，相邻对归一化标题
# 相似度≥阈值=合并（保留先者锚/要点，后者摘要拼接）。
_NEAR_DUP_TITLE_MIN_SCORE = 0.75


def _title_dice(left: str, right: str) -> float:
    def bigrams(value: str) -> set[str]:
        cleaned = "".join(
            ch for ch in str(value or "").lower()
            if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"
        )
        return {cleaned[i:i + 2] for i in range(max(0, len(cleaned) - 1))}

    left_grams, right_grams = bigrams(left), bigrams(right)
    if not left_grams or not right_grams:
        return 0.0
    return 2.0 * len(left_grams & right_grams) / (len(left_grams) + len(right_grams))


def merge_near_duplicate_titles(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """相邻近重复标题合并（就地语义、返回新列表与合并次数）。

    仅相邻对判定（Dice≥0.75）；合并=保留先者全量字段、后者 ``summary``/
    ``points``/``formulas`` 类列表字段拼接进先者（与 enforce_chapter_count
    同式：内容零删改逐字保留、锚取先者）。输入须已按 start_ms 升序。"""
    merged: list[dict[str, Any]] = []
    merge_count = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if merged and title:
            previous = merged[-1]
            previous_title = str(previous.get("title") or "").strip()
            if previous_title and _title_dice(previous_title, title) >= _NEAR_DUP_TITLE_MIN_SCORE:
                for list_key in ("points", "formulas"):
                    if isinstance(item.get(list_key), list):
                        previous[list_key] = list(previous.get(list_key) or []) + list(item[list_key])
                previous_summary = str(previous.get("summary") or "").strip()
                item_summary = str(item.get("summary") or "").strip()
                if item_summary:
                    combined = (previous_summary + "；" + item_summary).strip("；")
                    previous["summary"] = combined[:_CHAPTER_MERGED_SUMMARY_CAP]
                merge_count += 1
                continue
        if isinstance(item, dict):
            merged.append(dict(item))
    return merged, merge_count



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
        # W1 公式残渣门：显然坏式逐条丢弃（不丢整段；共享门=双路径同守卫）。
        if not sane_latex(latex):
            continue
        entry: dict[str, Any] = {"latex": latex}
        anchor = _valid_ms(formula.get("anchor_ms"))
        if anchor is not None:
            entry["anchor_ms"] = anchor
        caption = _clean_text(formula.get("caption"), 200)
        # W2 占位过滤：「课件公式行」类占位词不上屏（保 latex 丢 caption）。
        if caption and sane_formula_caption(caption):
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


def _derive_doc_anchor_ms(
    text: str, pool: list[tuple[int, str]] | None,
) -> int | None:
    """doc 候选锚拯救（BANQI-FIX1）：title/point 文本对转写锚池取字符二元组
    Dice 最优，低于阈值=None。与 chapters 锚拯救（llm._takeaway_anchor_ms，
    QSCOREFIX-1④）同源同阈同纪律——锚错比锚缺更伤「AI 说的话可信吗」的信任
    闭环；``pool`` 缺省 None=无拯救（调用方不传=逐位旧行为，零回归面）。"""
    if not pool:
        return None
    grams = _char_bigrams(text)
    if not grams:
        return None
    best_ms: int | None = None
    best_score = 0.0
    for start_ms, candidate_text in pool:
        candidate_grams = _char_bigrams(candidate_text)
        if not candidate_grams:
            continue
        score = 2.0 * len(grams & candidate_grams) / (len(grams) + len(candidate_grams))
        if score > best_score:
            best_score = score
            best_ms = int(start_ms)
    if best_score < _DOC_ANCHOR_MIN_SCORE:
        return None
    return best_ms


def coerce_llm_doc(
    raw: Any,
    valid_anchors: set[int],
    anchor_text_pool: list[tuple[int, str]] | None = None,
    drop_stats: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    """合并链 LLM doc 候选收敛器：逐级清洗后过 validate_summary_doc 同一道门
    （期 0 docstring 预留的「期 1 LLM 结构化输出」兑现）。清洗规则：
    - 锚完整性：sections 的 start_ms/points 的 anchor_ms 必须在转写/课件锚
      白名单内（QSCOREFIX-1④ 同纪律：白名单外=不可跳转=宁丢不编）；
      BANQI-FIX1 增量：白名单外先走与 chapters 同源的锚拯救（title/point 文本
      对 ``anchor_text_pool`` 派生真锚，阈值 _DOC_ANCHOR_MIN_SCORE），派生失败
      维持既有 fail-closed 丢段/丢点——685785 三连败实证：LLM chapters 20/20
      全过而 doc 候选三连全拒=doc 路径缺同款拯救（点锚取自渲染时间戳的确定性
      离格）是对合法内容的系统性误拒，非门太严。
    - source 收敛：合并链候选语义恒为 ``semantic``（ provenance 是调用方已知
      事实，不信任 LLM 自报）；
    - 逐级丢弃：坏点丢点、空段丢段、全空= None（调用方回退本地
      build_summary_doc 组装）；落键形状恒过 validate_summary_doc=禁半残。
    - ``drop_stats``（可选，闭集计数器）：raw_missing/sections_dropped_title/
      sections_dropped_start/sections_anchor_rescued/sections_dropped_points/
      points_dropped_text/points_dropped_anchor/points_anchor_rescued/
      points_dropped_noise/sections_empty/sections_near_dup_merged/
      formulas_dropped_unsane/
      captions_dropped_placeholder/validate_rejected/accepted_sections——拒因
      归档遥测（夜10-C 同纪律：计数器零内容），供 685785 重生成轮实证落点条款。
    - V010-WORKER-QUAL-1 文字质量守卫（全部逐条目 fail-closed）：W4 纯页码/
      目录枚举点丢弃；W1 残渣 latex 逐条丢弃；W2 占位 caption 丢键保式；
      W5 overview 元话语剥除；W7 相邻近重复段标题合并。"""
    stats = drop_stats if isinstance(drop_stats, dict) else None

    def _bump(key: str) -> None:
        if stats is not None:
            stats[key] = stats.get(key, 0) + 1

    if not isinstance(raw, dict) or not isinstance(valid_anchors, set):
        _bump("raw_missing")
        return None
    raw_sections = raw.get("sections")
    if not isinstance(raw_sections, list):
        _bump("raw_missing")
        return None
    sections: list[dict[str, Any]] = []
    for item in raw_sections[:_MAX_BLOCKS]:
        if not isinstance(item, dict):
            continue
        start_ms = _valid_ms(item.get("start_ms"))
        title = _clean_text(item.get("title"), _TITLE_CAP)
        if not title:
            _bump("sections_dropped_title")
            continue
        if start_ms is None or start_ms not in valid_anchors:
            derived = _derive_doc_anchor_ms(title, anchor_text_pool)
            if derived is None:
                _bump("sections_dropped_start")
                continue
            start_ms = derived
            _bump("sections_anchor_rescued")
        raw_points = item.get("points")
        if not isinstance(raw_points, list):
            _bump("sections_dropped_points")
            continue
        points: list[dict[str, Any]] = []
        for point in raw_points[:16]:
            if not isinstance(point, dict):
                continue
            text = _clean_text(point.get("text"), _POINT_CAP)
            if not text:
                _bump("points_dropped_text")
                continue
            # W4 要点噪声门：纯页码/目录/标题枚举句无结论价值=丢点不丢段。
            if is_noise_point_text(text):
                _bump("points_dropped_noise")
                continue
            anchor = _valid_ms(point.get("anchor_ms"))
            if anchor is None or anchor not in valid_anchors:
                derived = _derive_doc_anchor_ms(text, anchor_text_pool)
                if derived is None:
                    _bump("points_dropped_anchor")
                    continue
                anchor = derived
                _bump("points_anchor_rescued")
            points.append({"text": text, "anchor_ms": anchor})
        if not points:
            _bump("sections_dropped_points")
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
                # W1 公式残渣门：显然坏式逐条丢弃（680954 实测 ~20% 残渣形态）。
                if not sane_latex(latex):
                    _bump("formulas_dropped_unsane")
                    continue
                entry: dict[str, Any] = {"latex": latex}
                anchor = _valid_ms(formula.get("anchor_ms"))
                if anchor is not None and anchor in valid_anchors:
                    entry["anchor_ms"] = anchor
                caption = _clean_text(formula.get("caption"), 200)
                # W2 占位过滤：「课件公式行」类占位词不上屏（保 latex 丢 caption）。
                if caption and sane_formula_caption(caption):
                    entry["caption"] = caption
                elif caption:
                    _bump("captions_dropped_placeholder")
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
        _bump("sections_empty")
        return None
    # W7 相邻近重复段标题合并（39178「线性电源的结构与原理/线性电源：结构与
    # 原理」连排形态）：逐对 Dice≥0.75 合并，保先者锚、列表字段拼接。
    sections, near_dup_merged = merge_near_duplicate_titles(sections)
    if near_dup_merged and stats is not None:
        stats["sections_near_dup_merged"] = (
            stats.get("sections_near_dup_merged", 0) + near_dup_merged
        )
    # W5 元话语剥除（守卫层）：overview 承载自 markdown 首段，继承同款剥除。
    # F9-FIX-1 包①：内省括注（模型自评元评论）同族不上屏。
    raw_overview_value = raw.get("overview")
    if isinstance(raw_overview_value, str):
        raw_overview_value = strip_introspective_parens(
            strip_meta_discourse(raw_overview_value)
        )
    elif isinstance(raw_overview_value, list):
        raw_overview_value = [
            strip_introspective_parens(strip_meta_discourse(par))
            for par in raw_overview_value
        ]
    validated = validate_summary_doc({
        "overview": raw_overview_value if raw_overview_value is not None else "",
        "sections": sections,
        "terms": raw.get("terms") if raw.get("terms") is not None else [],
        "summary_notes": raw.get("summary_notes") if raw.get("summary_notes") is not None else [],
    })
    if validated is None:
        # 逐级清洗后仍被整门拒=overview/terms/summary_notes 形状坏（sections
        # 已全部过形状门）——诚实省键走本地兜底，计数归档拒因。
        _bump("validate_rejected")
        return None
    if stats is not None:
        stats["accepted_sections"] = len(validated["sections"])
    return validated


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
