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
