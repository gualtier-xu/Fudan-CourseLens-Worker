"""CourseLens subtitle display contract v2 (``subtitle.display.v2``).

字幕呈现 v2 的数据契约与契约钉共享模块（V010-SUBTITLE-C2 落地）。位于
``shared/`` 协议域——与 ``shared/evidence_contract.py`` 同域；worker（Tier2
清洗腿注入）与客户端（Tier1 整形产出、overlay 渲染消费）共同消费。纯标准
库、确定性、零第三方依赖、零 I/O（镜像仓同步面惯例与 evidence_contract 一致）。

冻结契约（规格唯一出处=archive V010-SUBTITLE-UX-R1 结果文件 §1/§2/§3.1；
本模块为字段级唯一权威，消费道不得另行手抄）：

字段级契约（全部为展示层**加法键**；证据层 verbatim 一字不动）：

- ``clean_text: str`` —— 整洁视图文本。删除式衍生：必须是同 cue ``text``
  的按序子序列（deletion-only，绝不增改）；可与 ``text`` 同值（=未清洗
  透传）。
- ``clean_source: str`` —— 清洗来源，闭集 ``CLEAN_SOURCES``：
  ``tier1_rule``（AS13 确定性闭集，零 token）/ ``tier2_llm``（worker LLM
  deletion-only 轻清洗）。
- ``clean_del_ratio: float`` —— 删词比（删除字符数/原文字符数，区间
  [0,1]）。必须与 ``(len(text)-len(clean_text))/len(text)`` 导出值一致
  （``ratio_mismatch`` 即非法）；元数据诚实性=契约而非惯例。
- 三键**捆绑相生**（all-or-none）：任一在场则三键必须齐备且全部合法；
  否则生产端必须整体省键（``sanitize_display_fields`` 执行）。
- ``math_block: {block_id: str, part: str}`` —— 跨 cue 公式块边界标记
  （hold-back）。键集闭集 ``MATH_BLOCK_KEYS``；``part`` 闭集
  ``MATH_BLOCK_PARTS``：``whole``（公式在本 cue 内完整闭合）/
  ``open``（含开定界符未闭合，块首）/ ``hold``（块中续挂）/
  ``close``（含闭合定界符，块尾，此后整条公式可上屏）。
- ``block_id`` 为非空字符串，同一讲次内稳定；同块 cue 在流内必须连续
  （``block_discontiguous`` 非法），闭合后同 id 不得复现
  （``block_reopened`` 非法）。

公式定界符闭集（与 frontend math-render.js / testbench math_cue_guard.mjs
同表，字节级钉在契约钉）：

- ``MATH_OPENERS`` = ``\\[…\\]`` 与 ``$$…$$``（display）+ ``\\(…\\)``
  （inline）；**单 ``$`` 永不识别**（防正文金额误吞）。
- hold-back 越窗兜底参数：``MATH_HOLD_BACK_MAX_CUES=6`` /
  ``MATH_HOLD_BACK_MAX_CHARS=300``；越窗放弃=不 emit 标记、按纯文本碎 cue
  呈现（与 §3.1 越窗放弃合并语义一致）。

worker Tier2 清洗 op 形（``validate_clean_ops``）：

- op = ``{"id": str, "del": str}``，键集**恰为** ``CLEAN_OP_KEYS``
  （多键少键皆非法）；``del`` 逐字出现于该段文本；ops 间不重叠；
  每段 ops ≤ ``CLEAN_MAX_OPS_PER_SEGMENT``；删除比超 ``max_del_ratio``
  即非法（该段整体回退=生产端义务）。
- 保护形零触碰：``protected_forms`` 由调用方传入（worker 侧 =
  ``llm._PROTECTED_FORM_RE`` 命中串；shared 不得 import worker），任一
  保护形在清洗结果中消失即 ``protected_form_altered`` 非法。

Tier 删词比预算（``clean_budget_ok``）：Tier1 单 cue 上限
``CLEAN_TIER1_MAX_DEL_RATIO=0.3``（超限该 cue 保留原文整句）；Tier2 单段
上限 ``CLEAN_TIER2_MAX_DEL_RATIO=0.2``（超限该段整体回退）。schema 校验
只锁 [0,1] 值域；预算门=生产端策略，经本助手函数统一执行。

schema 闭集与 fail-closed 省键规则（规范性文本）：

- v2 键空间闭集 = ``DISPLAY_V2_KEYS`` ∪ ``{"math_block"}``；本契约只裁
  判这些键。其余加法键（其他系统所有）按 evidence.v1 加法容忍惯例不归
  本契约管辖。
- **消费端降级规则（唯一权威实现=``display_text``/``math_block_of``，
  绝不抛错）**：``clean_text`` 缺席/非 str/空串/非删除式子序列 → 整洁
  视图不可用，回退 ``text`` 原文；``math_block`` 缺席/畸形（键集不对、
  part 越集、block_id 空）→ 按无公式语义纯文本渲染。元数据
  （clean_source/clean_del_ratio）畸形只影响元数据本身，绝不阻塞呈现。
  一句话：**坏形状降级到 verbatim，绝不报错、绝不吞文本、绝无开关。**
- **生产端省键规则（唯一权威实现=``sanitize_display_fields``，绝不抛
  错）**：三键不齐、类型不合法、非删除式、ratio 与导出值不符 → 干净地
  省掉全部 v2 键；math_block 形状不合法 → 省掉该键。宁缺勿错。

传输矩阵（义务条款，改行为属各消费道、非本契约道）：

- 证据层（learning_store transcript_segments、evidence.v1 文档）：恒
  verbatim，v2 键永不进入。
- worker 段 → 客户端整形（subtitle_reader AS13）：单源行全字段透传，
  v2 键随行；合并行 v2 键传播=道2（整形扩展）职责；整形输出新增 v2 键
  属整形口径变化 → 道2 须递增 ``DISPLAY_CACHE_FORMAT``（VTT-PERF 落盘
  缓存失效义务）。
- ``/subtitles/segments`` JSON 面：v2 键随 shaped cue 行透传给 overlay。
- ``/subtitles/file`` VTT 面与全部导出面：恒 verbatim（``_vtt_bytes``
  只写 ``text``），v2 键不序列化——忠实导出语义冻结。
- overlay 渲染消费（道3）：整洁视图取 ``display_text(cue)``；公式块
  hold-back 状态机以 ``math_block_of``/``MATH_OPENERS``/越窗参数为唯一
  契约出处；held 段不上屏，``open``/``hold`` 必有后续 ``close``
  （``validate_math_block_sequence`` 保证）。

错误面：``SubtitleDisplayContractError(code, message, path)``，code 闭集
见各校验函数；与 evidence_contract 错误风格同构。契约钉=
``tests/test_subtitle_display_contract.py``；fixture 样例=
``tests/testbench/fixtures/subtitle_display_v2/subtitle_display_v2_samples.json``
（含公式跨 cue hold-back 构造样例）。本模块落地零行为改动：现网消费者
未接线，行为逐位不变。
"""

from __future__ import annotations

CONTRACT_ID = "subtitle.display.v2"

# ---- 清洗元数据闭集 -------------------------------------------------------
CLEAN_SOURCES = ("tier1_rule", "tier2_llm")
CLEAN_TIER1_MAX_DEL_RATIO = 0.3
CLEAN_TIER2_MAX_DEL_RATIO = 0.2

# 展示层 v2 加法键闭集（捆绑语义见模块 docstring）。
DISPLAY_V2_KEYS = ("clean_text", "clean_source", "clean_del_ratio")
_MATH_BLOCK_KEY = "math_block"
# 全部 v2 键（含公式块标记对象键）。
DISPLAY_V2_ALL_KEYS = DISPLAY_V2_KEYS + (_MATH_BLOCK_KEY,)

# ---- 公式块标记闭集 -------------------------------------------------------
MATH_BLOCK_KEYS = ("block_id", "part")
MATH_BLOCK_PARTS = ("whole", "open", "hold", "close")
MATH_BLOCK_ID_MAX_CHARS = 128

# 公式定界符闭集（与 math-render.js / math_cue_guard.mjs 同表）：
# (open, close, display)；单 "$" 永不识别。
MATH_OPENERS = (
    ("\\[", "\\]", True),
    ("$$", "$$", True),
    ("\\(", "\\)", False),
)

# hold-back 越窗兜底参数（与 testbench 同参）。
MATH_HOLD_BACK_MAX_CUES = 6
MATH_HOLD_BACK_MAX_CHARS = 300

# ---- worker Tier2 清洗 op 契约 -------------------------------------------
CLEAN_OP_KEYS = ("id", "del")
CLEAN_MAX_OPS_PER_SEGMENT = 4

_RATIO_EPS = 1e-6
_BLOCK_ID_RE_ERROR = "block_id must be a non-empty string of <=128 chars"


class SubtitleDisplayContractError(ValueError):
    """契约校验失败（code 闭集、与 evidence_contract 错误风格同构）。"""

    def __init__(self, code: str, message: str, path: str = ""):
        super().__init__(f"{path or CONTRACT_ID}: [{code}] {message}")
        self.code = code
        self.message = message
        self.path = path


def _fail(code: str, message: str, path: str) -> None:
    raise SubtitleDisplayContractError(code, message, path)


def _check(condition: bool, code: str, message: str, path: str) -> None:
    if not condition:
        _fail(code, message, path)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# 删除式与删词比
# ---------------------------------------------------------------------------

def clean_text_is_deletion_only(clean: object, text: object) -> bool:
    """clean 是否为 text 的按序子序列（deletion-only 形状判定）。"""
    if not (isinstance(clean, str) and isinstance(text, str)):
        return False
    if not clean:
        return False
    if len(clean) > len(text):
        return False
    index = 0
    for char in clean:
        index = text.find(char, index)
        if index < 0:
            return False
        index += 1
    return True


def clean_del_ratio_of(text: object, clean: object) -> float:
    """删词比导出定义：删除字符数/原文字符数；原文字符数 0 时为 0.0。"""
    if not (isinstance(text, str) and isinstance(clean, str)) or not text:
        return 0.0
    return (len(text) - len(clean)) / len(text)


def clean_budget_ok(clean_source: object, clean_del_ratio: object) -> bool:
    """按 Tier 预算门判定删词比（生产端策略的统一执行点）。"""
    if clean_source == "tier1_rule":
        cap = CLEAN_TIER1_MAX_DEL_RATIO
    elif clean_source == "tier2_llm":
        cap = CLEAN_TIER2_MAX_DEL_RATIO
    else:
        return False
    return _is_number(clean_del_ratio) and 0.0 <= float(clean_del_ratio) <= cap + 1e-9


# ---------------------------------------------------------------------------
# 公式定界符（闭集扫描助手，供整形检测与钉测试共用）
# ---------------------------------------------------------------------------

def unclosed_math_opener(text: object) -> tuple[str, str] | None:
    """平衡扫描：返回文本中最早未闭合公式的 (open, closer)，全闭合为 None。

    语义与 testbench math_cue_guard 状态机一致：按最早开定界符配对其后的
    闭合定界符；单 ``$`` 不在闭集，永不配对。
    """
    source = text if isinstance(text, str) else str(text or "")
    position = 0
    while True:
        hit: tuple[int, str, str] | None = None
        for opener, closer, _display in MATH_OPENERS:
            found = source.find(opener, position)
            if found >= 0 and (hit is None or found < hit[0]):
                hit = (found, opener, closer)
        if hit is None:
            return None
        found, opener, closer = hit
        close_at = source.find(closer, found + len(opener))
        if close_at < 0:
            return (opener, closer)
        position = close_at + len(closer)


# ---------------------------------------------------------------------------
# cue 级 schema 校验（v2 键空间闭集；其余键不归本契约管辖）
# ---------------------------------------------------------------------------

def _clean_trio_validate(cue: dict, path: str) -> None:
    present = [key for key in DISPLAY_V2_KEYS if key in cue]
    if not present:
        return
    cpath = f"{path}.clean"
    _check(
        len(present) == len(DISPLAY_V2_KEYS),
        "field_required",
        f"clean keys are all-or-none: missing {sorted(set(DISPLAY_V2_KEYS) - set(present))}",
        cpath,
    )
    text = cue.get("text")
    _check(
        isinstance(text, str) and text != "",
        "text_missing",
        "clean keys require a non-empty string cue.text",
        cpath,
    )
    clean = cue["clean_text"]
    _check(
        isinstance(clean, str) and clean != "",
        "value_invalid",
        "clean_text must be a non-empty string",
        cpath + ".clean_text",
    )
    _check(
        clean_text_is_deletion_only(clean, text),
        "not_deletion_only",
        "clean_text must be an in-order subsequence of text (deletion-only)",
        cpath + ".clean_text",
    )
    _check(
        cue["clean_source"] in CLEAN_SOURCES,
        "value_unsupported",
        f"clean_source must be one of {CLEAN_SOURCES}",
        cpath + ".clean_source",
    )
    ratio = cue["clean_del_ratio"]
    _check(
        _is_number(ratio) and 0.0 <= float(ratio) <= 1.0,
        "value_invalid",
        "clean_del_ratio must be a number in [0, 1]",
        cpath + ".clean_del_ratio",
    )
    expected = clean_del_ratio_of(text, clean)
    _check(
        abs(float(ratio) - expected) <= _RATIO_EPS,
        "ratio_mismatch",
        f"clean_del_ratio must equal derived deletion ratio {expected!r}",
        cpath + ".clean_del_ratio",
    )


def _math_block_shape_valid(block: object) -> bool:
    return (
        isinstance(block, dict)
        and set(block.keys()) == set(MATH_BLOCK_KEYS)
        and isinstance(block.get("block_id"), str)
        and 0 < len(block["block_id"]) <= MATH_BLOCK_ID_MAX_CHARS
        and block.get("part") in MATH_BLOCK_PARTS
    )


def validate_display_cue_v2(cue: object) -> None:
    """校验单条 shaped/display cue 的 v2 加法键（schema 闭集，违规抛错）。

    仅裁判 v2 键空间；证据层键（index/start_ms/end_ms/text 等）与其他
    加法键由 evidence.v1 及其消费方管辖，本函数不越权。
    """
    _check(isinstance(cue, dict), "type_invalid", "cue must be an object", "cue")
    _clean_trio_validate(cue, "cue")
    block = cue.get(_MATH_BLOCK_KEY)
    if block is not None:
        _check(
            _math_block_shape_valid(block),
            "value_invalid",
            "math_block must be an object with exactly keys "
            f"{MATH_BLOCK_KEYS}, a 1..{MATH_BLOCK_ID_MAX_CHARS} char block_id "
            f"and part in {MATH_BLOCK_PARTS}",
            f"cue.{_MATH_BLOCK_KEY}",
        )


# ---------------------------------------------------------------------------
# 流级公式块序列校验（跨 cue hold-back 边界规则）
# ---------------------------------------------------------------------------

def validate_math_block_sequence(cues: object) -> None:
    """校验 cue 流内公式块标记序列的合法形状（违规抛错）。

    合法块 = ``[whole]`` 或 ``[open, hold*, close]``（至少 open+close）；
    同块 cue 必须流内连续；闭合块 id 不得复现。未闭合公式在越窗放弃后
    不得 emit 标记（纯文本呈现），故 ``[open]`` 或 ``[open, hold...]``
    无 close 即非法。
    """
    _check(isinstance(cues, list), "type_invalid", "cues must be a list", "cues")
    blocks: dict[str, list[str]] = {}
    order: list[str] = []
    previous_id: str | None = None
    for index, cue in enumerate(cues):
        _check(isinstance(cue, dict), "type_invalid", "cue must be an object", f"cues[{index}]")
        block = cue.get(_MATH_BLOCK_KEY)
        if block is None:
            previous_id = None
            continue
        qpath = f"cues[{index}].{_MATH_BLOCK_KEY}"
        validate_display_cue_v2(cue)
        block_id = block["block_id"]
        part = block["part"]
        _check(block_id not in blocks or previous_id == block_id,
               "block_discontiguous",
               f"math block {block_id!r} must be contiguous in the cue stream", qpath)
        _check(
            not (block_id in blocks and blocks[block_id][-1] == "close"),
            "block_reopened",
            f"math block {block_id!r} was closed and must not reappear", qpath,
        )
        if block_id not in blocks:
            blocks[block_id] = []
            order.append(block_id)
        blocks[block_id].append(part)
        previous_id = block_id
    for block_id in order:
        parts = blocks[block_id]
        legal = parts == ["whole"] or (
            len(parts) >= 2
            and parts[0] == "open"
            and parts[-1] == "close"
            and all(part == "hold" for part in parts[1:-1])
        )
        _check(legal, "block_sequence_invalid",
               f"math block {block_id!r} parts {parts} must be [whole] or "
               "[open, hold..., close]", f"cues.math_block[{block_id!r}]")


# ---------------------------------------------------------------------------
# 消费端降级（唯一权威实现；绝不抛错）
# ---------------------------------------------------------------------------

def display_text(cue: object) -> str:
    """整洁视图取值：clean_text 合法取 clean，否则回退 verbatim text。"""
    if not isinstance(cue, dict):
        return ""
    text = cue.get("text")
    base = text if isinstance(text, str) else ""
    clean = cue.get("clean_text")
    if (
        isinstance(clean, str)
        and clean != ""
        and isinstance(text, str)
        and text != ""
        and clean_text_is_deletion_only(clean, text)
    ):
        return clean
    return base


def math_block_of(cue: object) -> dict | None:
    """公式块标记取值：形状合法返回拷贝，否则 None（按纯文本渲染）。"""
    if isinstance(cue, dict):
        block = cue.get(_MATH_BLOCK_KEY)
        if _math_block_shape_valid(block):
            return dict(block)
    return None


# ---------------------------------------------------------------------------
# 生产端省键（唯一权威实现；绝不抛错）
# ---------------------------------------------------------------------------

def sanitize_display_fields(cue: object) -> dict:
    """返回 cue 拷贝：v2 键仅在全部合法时保留，否则整体省键（宁缺勿错）。"""
    if not isinstance(cue, dict):
        return {}
    out = dict(cue)
    try:
        _clean_trio_validate(out, "cue")
    except SubtitleDisplayContractError:
        for key in DISPLAY_V2_KEYS:
            out.pop(key, None)
    if not _math_block_shape_valid(out.get(_MATH_BLOCK_KEY)):
        out.pop(_MATH_BLOCK_KEY, None)
    return out


# ---------------------------------------------------------------------------
# worker Tier2 清洗 op 校验（形状门+保护形零触碰+预算门）
# ---------------------------------------------------------------------------

def validate_clean_ops(
    text: object,
    ops: object,
    protected_forms: object = (),
    max_del_ratio: float | None = None,
) -> str:
    """校验 deletion-only 清洗 op 列表并返回清洗结果文本（违规抛错）。

    门（规格 §2 ①③④⑤ 的形状子集；②闭集词表命中属生产端词表资产）：
    op 键集恰为 CLEAN_OP_KEYS；del 逐字出现；ops 不重叠；每段
    ops ≤ CLEAN_MAX_OPS_PER_SEGMENT；protected_forms（调用方传入的保护
    形命中串）零触碰；删除比 ≤ max_del_ratio（传入 Tier 上限）。
    """
    _check(isinstance(text, str) and text != "", "text_missing",
           "clean ops require a non-empty string segment text", "ops")
    _check(isinstance(ops, list), "type_invalid", "ops must be a list", "ops")
    _check(len(ops) <= CLEAN_MAX_OPS_PER_SEGMENT, "ops_limit",
           f"at most {CLEAN_MAX_OPS_PER_SEGMENT} ops per segment", "ops")
    spans: list[tuple[int, int]] = []
    for index, op in enumerate(ops):
        qpath = f"ops[{index}]"
        _check(isinstance(op, dict), "type_invalid", "op must be an object", qpath)
        _check(set(op.keys()) == set(CLEAN_OP_KEYS), "op_keys_invalid",
               f"op keys must be exactly {CLEAN_OP_KEYS}", qpath)
        _check(isinstance(op["id"], str) and op["id"] != "", "value_invalid",
               "op.id must be a non-empty string", qpath + ".id")
        fragment = op["del"]
        _check(isinstance(fragment, str) and fragment != "", "value_invalid",
               "op.del must be a non-empty string", qpath + ".del")
        start = text.find(fragment)
        _check(start >= 0, "del_absent",
               "op.del must appear verbatim in the segment text", qpath + ".del")
        spans.append((start, start + len(fragment)))
    spans.sort()
    for (start_a, end_a), (start_b, _end_b) in zip(spans, spans[1:]):
        _check(start_b >= end_a, "del_overlap", "ops must not overlap", "ops")
    cleaned_parts: list[str] = []
    cursor = 0
    for start, end in spans:
        cleaned_parts.append(text[cursor:start])
        cursor = end
    cleaned_parts.append(text[cursor:])
    cleaned = "".join(cleaned_parts)
    forms = list(protected_forms)
    for index, form in enumerate(forms):
        qpath = f"protected_forms[{index}]"
        _check(isinstance(form, str) and form != "", "value_invalid",
               "protected form must be a non-empty string", qpath)
        _check(form in text, "protected_form_absent",
               "protected form does not appear in the segment text", qpath)
        _check(form in cleaned, "protected_form_altered",
               "protected form must survive cleaning verbatim", qpath)
    if max_del_ratio is not None:
        ratio = clean_del_ratio_of(text, cleaned)
        _check(ratio <= float(max_del_ratio) + 1e-9, "del_ratio_exceeded",
               f"deletion ratio {ratio!r} exceeds budget {max_del_ratio!r}", "ops")
    return cleaned
