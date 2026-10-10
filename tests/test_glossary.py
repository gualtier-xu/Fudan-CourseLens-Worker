"""N5A-P3 字幕术语纠错·规则版：配对替换、受保护形、讲内一致性、空路径。

方案 PLAN-N5A-P3 的最低验证单测：同音错配对替换≥5 例、受保护形不替≥3 例、
词表空路径 1 例、全讲一致性 1 例；词表≤200 词条；纯本地（无 requests）。
"""

from __future__ import annotations
from pathlib import Path

from courselens_worker.glossary import (
    GLOSSARY_APPLIED_STATUS,
    GLOSSARY_MAX_TERMS,
    apply_glossary,
    build_glossary,
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
