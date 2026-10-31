"""A7-ROUTER-P1 段级确定性清洗：C1-C4 删并截断 + C4b/C6/C7 标记双向变异。

规则源=A3-SUBREAD-2 实证（segment_clean 模块头成文）；判据单位=伤害单位
（lessons#57）：每个删除/标记规则必须同时证明 ①该删/该标的命中（错形拦截
向）②不该删/不该标的一律不碰（正当使用零误伤向）。纯本地零外联零 LLM。
"""

from __future__ import annotations

from courselens_worker.segment_clean import clean_segments, flag_segments


def _seg(start_ms: int, end_ms: int, text: str, **extra) -> dict:
    row = {"start_ms": start_ms, "end_ms": end_ms, "text": text}
    row.update(extra)
    return row


def _texts(rows: list[dict]) -> list[str]:
    return [str(row.get("text") or "") for row in rows]


# ---- C2 filler-only 段清洗 ---------------------------------------------------


def test_c2_filler_only_deleted_with_gap_guard():
    rows = [
        _seg(0, 1000, "我们讲这个器件的等效电路"),
        _seg(1100, 1500, "嗯"),  # 双邻间隔 <1s → 删
        _seg(2000, 3000, "它的跨导是"),
    ]
    cleaned, stats = clean_segments(rows)
    assert _texts(cleaned) == ["我们讲这个器件的等效电路", "它的跨导是"]
    assert stats["c2_dropped"] == 1


def test_c2_latin_filler_case_insensitive():
    rows = [
        _seg(0, 1000, "好，那我们看下一页"),
        _seg(1050, 1400, "yeah"),
        _seg(1600, 2000, "OK，继续"),
    ]
    cleaned, stats = clean_segments(rows)
    assert _texts(cleaned) == ["好，那我们看下一页", "OK，继续"]
    assert stats["c2_dropped"] == 1


def test_c2_isolated_filler_with_large_gap_kept():
    # 零误伤向：双邻间隔 ≥1s 的孤立 filler 不删（宁漏勿错守卫）。
    rows = [
        _seg(0, 1000, "我们讲这个器件的等效电路"),
        _seg(1100, 1500, "嗯"),
        _seg(4000, 5000, "它的跨导是"),
    ]
    cleaned, stats = clean_segments(rows)
    assert _texts(cleaned) == ["我们讲这个器件的等效电路", "嗯", "它的跨导是"]
    assert stats["c2_dropped"] == 0


def test_c2_boundary_filler_kept():
    # 零误伤向：首段/末段缺邻不删（尾部归 C1 管辖）。
    rows = [_seg(0, 500, "嗯"), _seg(600, 2000, "开始上课")]
    cleaned, _ = clean_segments(rows)
    assert _texts(cleaned) == ["嗯", "开始上课"]


def test_c2_run_of_three_keeps_middle():
    # 连续 ≥3 段 filler-only 保留中间 1 段作节奏。
    rows = [
        _seg(0, 1000, "先看这个图"),
        _seg(1100, 1400, "嗯"),
        _seg(1500, 1800, "啊"),
        _seg(1900, 2200, "呃"),
        _seg(2300, 4000, "这个电路是共源极"),
    ]
    cleaned, stats = clean_segments(rows)
    assert _texts(cleaned) == ["先看这个图", "啊", "这个电路是共源极"]
    assert stats["c2_dropped"] == 2


def test_c2_content_segments_never_deleted():
    # 零误伤向：应答词/短实义段（集外形态、多字段）一律不删。
    rows = [
        _seg(0, 1000, "对，这个公式成立"),
        _seg(1100, 1500, "好的"),
        _seg(1600, 2000, "是吧，我们继续看"),
        _seg(2100, 4000, "下一章讲阈值电压"),
    ]
    cleaned, stats = clean_segments(rows)
    assert stats["c2_dropped"] == 0
    assert len(cleaned) == 4


# ---- C3 相邻重复段去重 --------------------------------------------------------


def test_c3_adjacent_duplicate_merged():
    rows = [
        _seg(0, 2000, "咱这个好像是摩托罗拉那，", evidence_id="ev-1"),
        _seg(2100, 4000, "咱这个好像是摩托罗拉那，", evidence_id="ev-2"),
        _seg(4100, 6000, "然后我们看它的负载线"),
    ]
    cleaned, stats = clean_segments(rows)
    assert len(cleaned) == 2
    assert stats["c3_dropped"] == 1
    assert cleaned[0]["end_ms"] == 4000  # 时间戳取并集
    assert cleaned[0].get("evidence_id") == "ev-1"  # 证据身份随首段
    assert _texts(cleaned)[1] == "然后我们看它的负载线"


def test_c3_non_adjacent_and_near_identical_kept():
    # 零误伤向：非相邻同文=修辞复现；一字之差≠完全相同。
    rows = [
        _seg(0, 1000, "这个公式很重要"),
        _seg(1100, 2000, "我们看下一页"),
        _seg(2100, 3000, "这个公式很重要"),
        _seg(3100, 4000, "这个公式很重要啊"),
    ]
    cleaned, stats = clean_segments(rows)
    assert stats["c3_dropped"] == 0
    assert len(cleaned) == 4


# ---- C4 假名/谚文整段删 -------------------------------------------------------


def test_c4_kana_dominated_segment_deleted():
    rows = [
        _seg(0, 1000, "我们接着讲工艺流程"),
        _seg(1100, 1500, "スたレか"),
        _seg(1600, 3000, "这一步是淀积"),
    ]
    cleaned, stats = clean_segments(rows)
    assert _texts(cleaned) == ["我们接着讲工艺流程", "这一步是淀积"]
    assert stats["c4_dropped"] == 1


def test_c4_hangul_dominated_segment_deleted():
    rows = [
        _seg(0, 1000, "안녕하세요 까"),
        _seg(1100, 3000, "继续看氧化步骤"),
    ]
    cleaned, stats = clean_segments(rows)
    assert _texts(cleaned) == ["继续看氧化步骤"]
    assert stats["c4_dropped"] == 1


def test_c4_mixed_legit_segment_kept():
    # 零误伤向：汉字主导+个别假名字符的正常段不删。
    rows = [_seg(0, 2000, "这个叫サンダ？不对，是衬底偏压效应")]
    cleaned, stats = clean_segments(rows)
    assert stats["c4_dropped"] == 0
    assert len(cleaned) == 1


# ---- C1 课末闲聊尾段截断 ------------------------------------------------------


def _tail_lecture(anchor_text: str = "行，那这个就这样。") -> list[dict]:
    """课末尾段标本：锚点位于讲次时长 ≥80% 处（终界含尾段 32.4s，锚 26s）。

    授课段文本互异（防触发 C3 相邻同文去重，隔离 C1 判定面）。
    """
    rows = [_seg(0, 4000, "今天我们讲版图设计")]
    for page, base in enumerate(range(5000, 24000, 3000), start=3):
        rows.append(_seg(base, base + 900, f"第{page}页讲接触孔的画法"))
    rows.append(_seg(26000, 26400, anchor_text))
    for index, token in enumerate(["嗯", "啊", "胖", "x", "OK", "啦"]):
        base = 26600 + index * 1000
        rows.append(_seg(base, base + 800, token))
    return rows


def test_c1_tail_truncated_after_anchor():
    rows = _tail_lecture()
    cleaned, stats = clean_segments(rows)
    assert stats["c1_dropped"] == 6
    assert cleaned[-1]["text"] == "行，那这个就这样。"  # 锚点保留
    assert "第3页讲接触孔的画法" in _texts(cleaned)  # 锚点前正常授课段全保留


def test_c1_mid_lecture_anchor_ignored():
    # 零误伤向：锚点在中段（<80% 时长，「下课之前…」形态）不触发。
    rows = [
        _seg(0, 4000, "下课之前我们还要讲一页"),
        _seg(4100, 8000, "这一页是接触孔"),
        _seg(8100, 9000, "中段小结一句"),
        # 孤立 filler（间隔 ≥1s，不触发 C2，隔离 C1 判定）。
        _seg(9100, 9400, "嗯"),
        _seg(10600, 10900, "啊"),
        _seg(12100, 12400, "呃"),
        _seg(13600, 13900, "哦"),
        _seg(15100, 15400, "哎"),
        _seg(16600, 20000, "然后正式讲下一节"),
    ]
    cleaned, stats = clean_segments(rows)
    assert stats["c1_dropped"] == 0
    assert len(cleaned) == 9


def test_c1_real_content_after_anchor_not_truncated():
    # 零误伤向：锚点后是正常答疑内容（低信息密度 <70%）→ 不截断。
    rows = _tail_lecture()[:8]  # 引言+7 授课段（至 23000+900）
    rows.append(_seg(26000, 26400, "行，那这个就这样。"))
    for index in range(6):
        base = 26600 + index * 1000
        rows.append(
            _seg(base, base + 900, f"有同学问第{index}章的作业，下周三之前交到课程邮箱")
        )
    cleaned, stats = clean_segments(rows)
    assert stats["c1_dropped"] == 0
    assert len(cleaned) == 15


def test_c1_small_tail_not_truncated():
    # 零误伤向：锚点后不足 5 段不判（样本太小）。
    rows = _tail_lecture()[:8]
    rows.append(_seg(26000, 26400, "行，那这个就这样。"))
    rows.append(_seg(26600, 27400, "嗯"))
    rows.append(_seg(27600, 28400, "啊"))
    cleaned, stats = clean_segments(rows)
    assert stats["c1_dropped"] == 0


# ---- 幂等性与边界 -------------------------------------------------------------


def test_clean_segments_idempotent():
    rows = _tail_lecture()
    once, _ = clean_segments(rows)
    twice, stats = clean_segments(once)
    assert _texts(twice) == _texts(once)
    assert all(value == 0 for value in stats.values())


def test_clean_segments_skips_non_dict_rows():
    cleaned, _ = clean_segments(["not-a-dict", _seg(0, 1000, "正常段")])
    assert _texts(cleaned) == ["正常段"]


def test_clean_segments_empty_path():
    cleaned, stats = clean_segments([])
    assert cleaned == []
    assert stats["c1_dropped"] == 0


# ---- Part 2 标记（C4b/C6/C7）：只标不删不改文 ---------------------------------


def test_flags_added_and_text_untouched():
    # C6 按 A3 谓词（段尾无终止标点+不以的了是和与或结尾+邻段 ≤5s）对 1/2/3
    # 行均成立（1364 段/库 的高发面=标记本义）；此处断言关键命中不漏。
    rows = [
        _seg(0, 1000, "第一段讲完了"),
        _seg(36000, 37000, "我们休息一下再回来"),  # gap 35s → C7；尾字「来」→ C6
        _seg(37500, 38500, "刚才说到跨导"),  # 尾字「导」无终止标点 → C6
        _seg(38600, 39600, "mode age transfer"),  # C4b 杂线（尾字 r 亦 C6）
        _seg(39700, 40700, "下一节讲版图。"),
    ]
    texts_before = _texts(rows)
    stats = flag_segments(rows)
    assert _texts(rows) == texts_before  # 文本零改写
    assert len(rows) == 5  # 零删段
    assert stats["c7_flagged"] == 1
    assert rows[1]["long_gap_ms"] == 35000
    assert stats["c6_flagged"] == 3
    assert rows[2]["merge_candidate"] == 1
    assert "merge_candidate" not in rows[4]  # 终止标点不标
    assert stats["c4b_flagged"] == 1
    assert rows[3]["low_confidence_en"] == 1


def test_flags_false_positive_guards():
    # 零误伤向：公式/单位形态、合法英文术语不标低置信；免挂尾字/终标点不标合并。
    rows = [
        _seg(0, 1000, "μCox = 100 μA/V²"),  # 数字+运算符 → 不标低置信
        _seg(1100, 2000, "CMOS inverter"),  # 2 token 合法术语 → 不标低置信
        _seg(2100, 3000, "这个电路是这个节点的"),  # 尾字「的」→ 不标合并
        _seg(3100, 4000, "公式成立。"),  # 终止标点 → 不标合并
        _seg(4100, 4200, "收尾。"),
        _seg(4300, 4400, "uh, x Growth in one point nine six y"),  # 9 token 杂线 → 标
    ]
    stats = flag_segments(rows)
    assert "low_confidence_en" not in rows[0]
    assert "low_confidence_en" not in rows[1]
    assert "merge_candidate" not in rows[2]
    assert "merge_candidate" not in rows[3]
    assert "long_gap_ms" not in rows[1]
    assert rows[5]["low_confidence_en"] == 1
    assert stats["c7_flagged"] == 0


def test_flag_segments_empty_path():
    assert flag_segments([]) == {"c4b_flagged": 0, "c6_flagged": 0, "c7_flagged": 0}


# ---- C2b 课间静默封顶段群（A2-SUBREAD-1 规则 1/2）---------------------------


def test_c2b_silence_cap_run_deleted():
    rows = [
        _seg(0, 2000, "我们先回顾一下上节课的内容要点"),
        _seg(30000, 60000, "嗯"),
        _seg(60000, 90000, "the"),
        _seg(90000, 120000, "对"),
        _seg(120500, 150000, "正式内容：这一段讲钝化层的生长工艺与缺陷密度关系。"),
    ]
    cleaned, stats = clean_segments(rows)
    assert stats["c2b_dropped"] == 3
    assert _texts(cleaned) == ["我们先回顾一下上节课的内容要点", "正式内容：这一段讲钝化层的生长工艺与缺陷密度关系。"]


def test_c2b_short_run_and_substantive_kept():
    # 零误伤向：成群不足 3 段不删；封顶时长但文本承载充分不删。
    rows = [
        _seg(30000, 60000, "嗯"),
        _seg(70000, 100000, "the"),
        _seg(105000, 135000, "这一段虽然时长封顶但文本是有实际内容的长句不能删掉。"),
    ]
    cleaned, stats = clean_segments(rows)
    assert stats["c2b_dropped"] == 0
    assert len(cleaned) == 3
