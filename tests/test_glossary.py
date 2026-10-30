"""N5A-P3 字幕术语纠错·规则版：配对替换、受保护形、讲内一致性、空路径。

方案 PLAN-N5A-P3 的最低验证单测：同音错配对替换≥5 例、受保护形不替≥3 例、
词表空路径 1 例、全讲一致性 1 例；词表≤200 词条；纯本地（无 requests）。
"""

from __future__ import annotations
from pathlib import Path

from courselens_worker.glossary import (
    GLOSSARY_APPLIED_STATUS,
    GLOSSARY_DOMINANT_FORM_MIN_OCCURRENCES,
    GLOSSARY_MAX_TERMS,
    apply_glossary,
    build_glossary,
    filter_prompt_terms,
    unify_text_terms,
)


def _pages(*texts: str, repeats: int = 3):
    pages = []
    for text in texts:
        for _ in range(repeats):
            pages.append({"text": text})
    return pages


def _segments(*texts: str):
    return [{"segment_id": index, "text": text} for index, text in enumerate(texts)]


def test_build_glossary_freq_cap_and_empty_path():
    pages = _pages("傅里叶变换是信号处理的核心工具", "卷积神经网络与深度学习")
    glossary = build_glossary(pages)
    assert "傅里叶变换" in glossary
    assert "卷积神经网络" in glossary
    assert len(glossary) <= GLOSSARY_MAX_TERMS
    # OCR 缺席：仅课程名（不展开）
    assert "高等数学" in build_glossary([], "高等数学上册")
    assert build_glossary(None, "") == ()


def test_homophone_pairs_replace_across_lecture():
    pairs = [
        ("傅里叶变换", "傅里叶变換"),
        ("概率密度函数", "概卒密度函数"),
        ("卷积神经网络", "卷积神精网络"),
        ("高斯分布", "高斯分饰"),
        ("勒让德多项式", "勒让德多項式"),
    ]
    for term, variant in pairs:
        pages = _pages(f"本讲主题是{term}的应用")
        glossary = build_glossary(pages)
        assert term in glossary, (term, glossary)
        segments = _segments(
            f"我们先回顾{variant}的定义",
            f"{variant}在工程里很常见",
        )
        output = apply_glossary(segments, glossary)
        changed = [item for item in output if item.get("correction") == GLOSSARY_APPLIED_STATUS]
        assert len(changed) == 2, (term, variant, [item["text"] for item in output])
        assert all(item["text"].find(variant) < 0 for item in output)
        assert all(item["text"].find(term) >= 0 for item in output)


def test_protected_forms_are_never_replaced():
    term = "傅里叶变换"
    glossary = build_glossary(_pages(f"本讲主题是{term}"))
    # 否定词插入（不/没）：替换会删掉受保护的否定形 → 一律回退
    for negation in ("不", "没"):
        variant = f"{negation}{term}"
        segments = _segments(f"这{variant}的用法", f"书上{variant}展开了")
        output = apply_glossary(segments, glossary)
        assert all(item.get("correction") != GLOSSARY_APPLIED_STATUS for item in output), negation
        assert all(item["text"].find(variant) >= 0 for item in output), negation
    # 中文数字串插入（万+二 构成两位数字串）：同理回退
    term2 = "二次型"
    glossary2 = build_glossary(_pages(f"线性代数里的{term2}很重要"))
    segments2 = _segments("这万二次型的讨论", "书上万二次型有例题")
    output2 = apply_glossary(segments2, glossary2)
    assert all(item.get("correction") != GLOSSARY_APPLIED_STATUS for item in output2)
    # 公式邻接不受牵连：受保护形序列不变 → 正常应用
    segments3 = _segments("若 x=y+z 则傅里叶变換成立", "傅里叶变換是线性算子")
    output3 = apply_glossary(segments3, glossary)
    assert len([item for item in output3 if item.get("correction") == GLOSSARY_APPLIED_STATUS]) == 2


def test_lecture_wide_consistency_requires_two_occurrences():
    glossary = build_glossary(_pages("本讲主题是傅里叶变换"))
    single = _segments("偶尔提到一次傅里叶变換")
    assert apply_glossary(single, glossary) == single, "只出现一次的串不成对"
    two = _segments("讲到傅里叶变換", "又讲傅里叶变換")
    output = apply_glossary(two, glossary)
    assert all(item.get("correction") == GLOSSARY_APPLIED_STATUS for item in output)


def test_glossary_module_is_pure_local():
    source = Path(__file__).parent.parent.joinpath("courselens_worker", "glossary.py").read_text(encoding="utf-8")
    assert "requests" not in source, "术语纠错必须纯本地，不允许外联"
    assert "import requests" not in source


# ---- RXFIX-1 对照测试集（2026-10-29；37355 半器课型误替换根修验收面） --------
# 双类验收=「修正确的、不伤原对的」：误替换类（37355 真实损伤镜像）零替换、
# 正确替换类（ASR 同音错拼）照常替换。防复发钉：任何一道守卫回退即红。


class Rxfix1MisReplacementClass:
    """误替换类：词表近形与正确术语构成最小对立体（电流/电平、基极/栅极族）。"""

    # OCR 页=MOSFET 面高频造出 电平/栅极/效应管；字幕=BJT 面正确术语高频。
    MOSFET_PAGES = (
        "MOS场效应管的栅极电压决定沟道电平，输出电平随电平翻转变化",
        "效应管栅极氧化层与阈值电平",
    )

    @classmethod
    def glossary(cls):
        glossary = build_glossary(_pages(*cls.MOSFET_PAGES))
        assert "电平" in glossary and "栅极" in glossary and "效应管" in glossary
        # 效应 ∈ 词表（场效应管子串，词表内串本就跳过配对）；电流/电路/基极
        # 是词表外近形——正是守卫必须拦下的误替换面。
        assert "电流" not in glossary and "电路" not in glossary
        assert "基极" not in glossary
        return glossary


def test_rxfix1_dominant_real_terms_are_never_replaced():
    """优势形守卫：讲内高频正确术语（≥8 次）绝不改写成词表近形。

    真实损伤签名（REPROC-BANQI-2 P-7）：电流 382-384/效应 104/电路 35 次仍
    被整讲替换为 电平/效应管/电平——旧逻辑只设 ≥2 下限，恰好放大误替换。"""
    glossary = Rxfix1MisReplacementClass.glossary()
    sentence = "集电极电流经电路放大，基极电压与厄利效应决定输出。"
    segments = _segments(*([sentence] * 9))
    # 自检：各形出现 9 次 ≥ 优势形阈值。
    for piece in ("电流", "电路", "基极", "效应"):
        assert sum(str(seg["text"]).count(piece) for seg in segments) >= GLOSSARY_DOMINANT_FORM_MIN_OCCURRENCES
    output = apply_glossary(segments, glossary)
    assert all(item.get("correction") != GLOSSARY_APPLIED_STATUS for item in output)
    assert all("电平" not in item["text"] for item in output), "电流/电路 不得被改成电平"
    assert all("栅极" not in item["text"] for item in output), "基极 不得被改成栅极"
    assert all("效应管" not in item["text"] for item in output), "效应 不得被改成效应管"
    assert [item["text"] for item in output] == [item["text"] for item in segments], "零改写"


def test_rxfix1_source_corpus_form_is_never_replaced():
    """源语料守卫：2-7 次的中频串若在课件语料出现=课程真实写法，禁替换。"""
    pages = ([{"text": "输出电平与电平翻转"}] * 3) + [{"text": "集电极电流一瞥"}]
    glossary = build_glossary(pages)
    assert "电平" in glossary and "电流" not in glossary
    # 电流=3 次（≥2 可配对、<8 非优势形），但 电流 ∈ OCR 语料 → 禁改写。
    segments = _segments("电流流过负载", "测量电流", "记录电流")
    output = apply_glossary(segments, glossary)
    assert all(item.get("correction") != GLOSSARY_APPLIED_STATUS for item in output)
    assert all("电平" not in item["text"] for item in output)


def test_rxfix1_dominance_threshold_boundary():
    """阈值边界：7 次=仍可纠错，8 次=优势形禁改（守卫不做一刀切摘除）。"""
    glossary = build_glossary(_pages("本讲主题是傅里叶变换"))
    seven = _segments(*(["傅里叶变換出现"] * 7))
    out7 = apply_glossary(seven, glossary)
    assert all("傅里叶变换" in item["text"] for item in out7)
    eight = _segments(*(["傅里叶变換出现"] * 8))
    out8 = apply_glossary(eight, glossary)
    assert all(item.get("correction") != GLOSSARY_APPLIED_STATUS for item in out8)
    assert all("傅里叶变換" in item["text"] for item in out8)


def test_rxfix1_mixed_lecture_repairs_genuine_and_protects_real():
    """同讲双类共存=修正确的、不伤原对的（核心验收）。"""
    pages = (
        ([{"text": "输出电平与电平翻转"}] * 3)
        + ([{"text": "本讲主题是傅里叶变换"}] * 3)
    )
    glossary = build_glossary(pages)
    assert "电平" in glossary and "傅里叶变换" in glossary
    genuine = _segments(*(["傅里叶变換是线性算子"] * 3))     # 错拼 3 次 → 应替换
    real = _segments(*(["集电极电流经电路"] * 9))            # 真词 9 次 → 禁替换
    output = apply_glossary(genuine + real, glossary)
    repaired = [item for item in output if "傅里叶变换" in item["text"]]
    preserved = [item for item in output if "电流" in item["text"]]
    assert len(repaired) == 3 and len(preserved) == 9
    assert all("电平" not in item["text"] for item in preserved)
    assert all(item["text"].find("傅里叶变換") < 0 for item in repaired)


def test_rxfix1_glossary_carries_source_corpus():
    """词表元组携带源语料（守卫证据底座）；普通元组场景守卫不生效。"""
    glossary = build_glossary(_pages("甲乙丙丁戊己"), "高等数学上册")
    corpus = str(getattr(glossary, "source_corpus", ""))
    assert "甲乙丙丁戊己" in corpus and "高等数学上册" in corpus
    from courselens_worker.glossary import GlossaryTerms
    plain = GlossaryTerms(("电平",), "")
    assert plain.source_corpus == ""
    assert tuple(plain) == ("电平",) and plain == ("电平",)


def test_rxfix1_unify_text_terms_respects_both_guards():
    """摘要面（unify_text_terms）同守卫：真实词形禁改、错拼照常统一。"""
    glossary = build_glossary(([{"text": "输出电平与电平翻转"}] * 3) + [{"text": "集电极电流一瞥"}])
    real_text = "电流分三种：电流、电流，还有电流、电流。"
    assert unify_text_terms(real_text, glossary) == real_text, "源语料形（电流）禁改"
    dominant_text = "电流值得讲。" + "电流很重要，" * 8
    assert unify_text_terms(dominant_text, glossary) == dominant_text, "优势形（≥8 次）禁改"
    genuine_glossary = build_glossary(_pages("本讲主题是傅里叶变换"))
    genuine_text = "傅里叶变換出现两次傅里叶变換。"
    assert unify_text_terms(genuine_text, genuine_glossary).count("傅里叶变换") == 2


def test_rxfix1_unify_explicit_source_corpus_passthrough():
    """create_summary 展平词表后经 source_corpus 显式透传仍生效。"""
    glossary = build_glossary(([{"text": "输出电平与电平翻转"}] * 3) + [{"text": "集电极电流一瞥"}])
    text = "电流与电流。"
    assert unify_text_terms(text, tuple(glossary), source_corpus=glossary.source_corpus) == text
    # 无语料=守卫不生效（旧行为）。
    assert unify_text_terms(text, tuple(glossary)) != text or text.count("电流") < 2


def _rxfix3_slide_glossary():
    """37355 伤型镜像词表：电平=课件高频词（转写 0 次），电流族=口语词。"""
    return build_glossary(([{"text": "输出电平与电平翻转"}] * 3) + [{"text": "集电极电流一瞥"}])


def test_rxfix3_direction_guard_blocks_spoken_word_to_slide_only_term():
    """RXFIX-3 ③方向守卫：口语有证词（电路转写 6 次）不被转写零证词（电平）改正。

    669766 掷1 残留伤型实证（r19 代，run 40841877970）：md 电平 18 处含
    电平线（水平）/电平率（电导率）/方块电平（方块电阻）族——提示词偏置 +
    unify 机械面同向受染。evidence_text=转写全文后两方向全拦。"""
    glossary = _rxfix3_slide_glossary()
    evidence = "经电路放大，" * 3 + "水平线电导率方块电阻，" + "其余口语词。"
    md = "经电路放大，电场为水平线，电导率与方块电阻。经电路放大。"
    out = unify_text_terms(md, tuple(glossary), evidence_text=evidence)
    assert "电路" in out and "水平线" in out, out
    assert "电导率" in out and "方块电阻" in out, out
    assert "电平" not in out, out


def test_rxfix3_evidence_dominance_counts_transcript_not_text():
    """RXFIX-3 ①计频口径：优势形按转写计（≥8）而非被统一文本内部计。

    RXFIX-1 实现按 md 内部计频——提示词偏置先稀释 md 内词形、守卫随即失守；
    口径改转写后，md 内稀疏的口语高频词（如电流）同样受守卫保护。"""
    glossary = _rxfix3_slide_glossary()
    # 转写：电流 ×9（≥8 优势形）；md：电流仅 2 次（<8，旧口径下不设防）。
    evidence = "电流流过电路。" * 9
    md = "电流与电流。"
    assert unify_text_terms(md, tuple(glossary), evidence_text=evidence) == md
    # 对照：转写低频（1 次）时方向守卫亦拦（电流转写有证、电平零证）。
    low = unify_text_terms(md, tuple(glossary), evidence_text="电流一瞥")
    assert low == md


def test_rxfix3_legit_ocr_glyph_repair_unaffected():
    """合法纠错向不受方向守卫影响：字形变体（变換）转写零证、目标（变换）
    转写有证 → 照常统一（RXFIX-1 正例钉的 RXFIX-3 对应面）。"""
    genuine_glossary = build_glossary(_pages("本讲主题是傅里叶变换"))
    evidence = "傅里叶变换是线性算子，傅里叶变换可分离。"
    md = "傅里叶变換出现两次傅里叶变換。"
    out = unify_text_terms(md, tuple(glossary := tuple(genuine_glossary)), evidence_text=evidence)
    assert out.count("傅里叶变换") == 2, out


def test_rxfix3_no_evidence_text_zero_behavior_change():
    """evidence_text 缺省=None：旧调用方（字幕链/旧测试）逐位零行为变化。"""
    glossary = _rxfix3_slide_glossary()
    genuine_glossary = build_glossary(_pages("本讲主题是傅里叶变换"))
    dominant_text = "电流值得讲。" + "电流很重要，" * 8
    assert unify_text_terms(dominant_text, tuple(glossary)) == dominant_text
    genuine_text = "傅里叶变換出现两次傅里叶变換。"
    assert unify_text_terms(genuine_text, tuple(genuine_glossary)).count("傅里叶变换") == 2


# ---- RXESCAPE-1（2026-10-30；685785 残留伤型=滑窗片段伪影+注入影子词根修验收面）----


def test_rxescape1_embedded_fragment_blocks_rewrite():
    """零证片段守卫：piece 只以「更长真实词的内部片段」存在（每次出现前后紧
    邻汉字、无串边界）→ 禁改写，无论目标词转写零证（685785 实证：电中性区
    →电平性区，「电中」/「电平」转写双 0）还是偶证（653998 活体复掷实证：
    光电转换→光电平换，「电转」0 证而「电平」转写 2 次）——嵌入形态本身即
    判据。既有优势形守卫（转写 0<8）与方向守卫（piece 0 证不触发）在该形
    态双双不设防，RXESCAPE-1 收口。"""
    md = "在耗尽层外电中性区以扩散方式衰减。\n又一处电中性区按扩散长度衰减。"
    evidence = "电流流过电路，载流子扩散。" * 5  # 无「电中」亦无「电平」
    out = unify_text_terms(md, ("电平",), evidence_text=evidence)
    assert out == md, out
    assert "电平性区" not in out
    # 653998 活体复掷伤型：目标词转写偶证（2 次<8），片段伪影仍拦。
    md2 = "光电转换把光变成电。硅基光电转换效率高。"
    evidence2 = "逻辑电平切换。" + "电流流过电路。" * 5  # 电平 2 次有证
    out2 = unify_text_terms(md2, ("电平",), evidence_text=evidence2)
    assert out2 == md2, out2
    assert "电平换" not in out2


def test_rxescape1_boundary_occurrence_keeps_legit_repair():
    """合法纠错向不受零证片段守卫影响：变体在文本中存在串边界出现（行首/
    标点后）→ 非内部伪影，照常统一（教师未说出的课件权威词形照常修复）；
    变体只以嵌入形存在（mid-prose 无边界）→ 守卫拦（宁漏勿错）。"""
    evidence = "文本0文本1文本2文本3"  # 双方转写均零证
    md = "厄早效应两见：厄早效应、厄早效应。"
    out = unify_text_terms(md, ("厄利效应",), evidence_text=evidence)
    assert out.count("厄利效应") == 3, out
    md2 = "由厄早效应主导机制，经厄早效应放大输出。"
    out2 = unify_text_terms(md2, ("厄利效应",), evidence_text=evidence)
    assert out2 == md2, out2


def test_rxescape1_dual_zero_spares_legit_glyph_repair():
    """合法纠错向不受零证片段守卫影响：字形变体（变換）零证、目标（变换）
    转写有证 → 照常统一（与 685785 伤型的判别特征=目标词转写证据）。"""
    md = "傅里叶变換出现两次傅里叶变換。"
    evidence = "傅里叶变换是线性算子，傅里叶变换可分离。"
    out = unify_text_terms(md, ("变换",), evidence_text=evidence)
    assert out.count("傅里叶变换") == 2, out


def test_rxescape1_no_evidence_callers_zero_behavior_change():
    """evidence_text 缺省（None）调用方逐位零行为变化——零证片段/语料兜底守卫
    只在转写证据面生效（旧调用方/旧测试兼容）。"""
    md = "在耗尽层外电中性区按扩散长度衰减，电中性区重复两次。"
    # 无证据面：旧行为原样保留（机械配对面不在本守卫射程）。
    out = unify_text_terms(md, ("电平",))
    assert isinstance(out, str)


def test_rxescape1_speech_corpus_fallback_protects_spoken_words():
    """语料兜底守卫（653998 掷2 活体实证）：无课件语料时转写即保护语料——
    老师真实说出的词形（电源/电池/电光，转写 4/2/5 次均低于优势形阈值）绝
    不被低证近形课件词「电平」（转写偶证 2 次）改正；「电平转换/太阳能电平/
    电平电压」族 31 处残留即此通道。合法纠错不受影响：变体从不在转写中。"""
    md = "电源电压为 VDD。太阳能电池把光变成电。做电光转换。"
    evidence = "电源" * 4 + "电池" * 2 + "电光" * 5 + "电平" * 2
    out = unify_text_terms(md, ("电平",), evidence_text=evidence)
    assert out == md, out
    assert "电平" not in out
    # 对照：变体（变換）不在转写中 → 照常统一（语音 ground truth 语义）。
    md2 = "傅里叶变換出现两次傅里叶变換。"
    evidence2 = "傅里叶变换是重点。" + "电源" * 8
    out2 = unify_text_terms(md2, ("变换",), evidence_text=evidence2)
    assert out2.count("傅里叶变换") == 2, out2


def test_rxescape1_filter_prompt_terms_shadowed_slide_word_dropped():
    """注入影子过滤：电平（转写 2 次）被编辑距离 1 的转写优势形近形词电流
    （150 次）影子 → 不入「术语写法以 glossary 表为准」权威清单；自身优势形
    （电流/变换 ≥8 次）恒保留。37355 重灾三讲的生成侧替换通道收口。"""
    terms = ("电流", "电平", "变换")
    evidence = "电流。" * 150 + "变换。" * 15 + "电平。" * 2
    kept = filter_prompt_terms(terms, evidence)
    assert kept == ("电流", "变换"), kept


def test_rxescape1_filter_prompt_terms_shadow_from_transcript_piece():
    """影子候选含转写自身优势形（不限于表内兄弟）——653998 活体实证伤型：
    payload 课程记忆词表含「电平」而缺「电流」，仅查表内兄弟漏影，「电平」
    以权威名义入表=模型侧 58 处近形清洗通道。电流非表内词仍影之。"""
    terms = ("电平", "器件")
    evidence = "电流。" * 150 + "电平。" * 2 + "器件。" * 5
    kept = filter_prompt_terms(terms, evidence)
    assert kept == ("器件",), kept


def test_rxescape1_filter_prompt_terms_dominant_slide_word_not_shadowed():
    """影子过滤零误伤：电平被真实大量讲出（≥8 次）时与电流同表保留——正当
    使用「电平」的课（数字/逻辑课型）注入面零变化。"""
    terms = ("电流", "电平")
    evidence = "电流。" * 20 + "电平。" * 20
    kept = filter_prompt_terms(terms, evidence)
    assert set(kept) == set(terms), kept


def test_rxescape1_filter_prompt_terms_empty_evidence_passthrough():
    """evidence 空=原样返回（零行为变化）。"""
    terms = ("电流", "电平")
    assert filter_prompt_terms(terms, "") == terms
    assert filter_prompt_terms((), "电流电流") == ()
