# -*- coding: utf-8 -*-
"""L1a 思考档显式化钉（DEEPCOST-FIX-1，生成式钉同族形态）。

历史教训：v1 词级校对链曾是全 worker 唯一未显式传 ``thinking`` 的调用面
（提供商缺省=思考档 enabled/high，思考 token 按输出价 4× 计；DEEPCOST-1
调研定谳为账单头号单点，实测基线 5 窗 completion 的 99.5% 是思考 token）。
本钉闭掉「缺省思考」复发的逃脱通道：

1. 全 worker 生产模块（courselens_worker/*.py）AST 扫描：每个 ``_chat``
   调用点必须显式携带 ``thinking`` 关键字（档位值由各链策略自定，钉只管
   「显式」——未来要开思考的链必须写出 ``thinking=…`` 让评审可见）；
2. ``_chat`` 本体 default-safe：漏传=显式 disabled，不再裸奔提供商缺省
   （兜动态分发类 AST 不可见调用面）。
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "courselens_worker"


def _chat_call_sites(tree: ast.AST):
    """Yield (module, lineno, node) for every call whose target is _chat."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "_chat":
            yield node
        elif isinstance(func, ast.Attribute) and func.attr == "_chat":
            yield node


class LlmThinkingPinTests(unittest.TestCase):
    """「所有 _chat 调用点必须显式 thinking」生成式钉。"""

    def test_every_chat_call_site_passes_thinking_explicitly(self) -> None:
        modules = sorted(PACKAGE.glob("*.py"))
        self.assertTrue(modules, "worker package modules must exist")
        checked = 0
        for module in modules:
            tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
            for node in _chat_call_sites(tree):
                keywords = {kw.arg for kw in node.keywords}
                self.assertIn(
                    "thinking",
                    keywords,
                    f"{module.name}:{node.lineno} _chat 调用缺显式 thinking 参数"
                    "（L1a 钉：漏传=提供商缺省思考档，成本 16×；请显式传档）",
                )
                checked += 1
        # 生成式钉自证：扫描面非空（_chat 至少有本体定义与真实调用点）。
        self.assertGreaterEqual(checked, 8, f"unexpectedly few _chat call sites: {checked}")

    def test_chat_default_is_disabled_not_provider_default(self) -> None:
        source = (PACKAGE / "llm.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="llm.py")
        default_safe = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            targets = [t for t in node.targets if isinstance(t, ast.Subscript)]
            if not targets:
                continue
            subscript = targets[0]
            if not (isinstance(subscript.value, ast.Name) and subscript.value.id == "payload"):
                continue
            if not (isinstance(subscript.slice, ast.Constant) and subscript.slice.value == "thinking"):
                continue
            value = node.value
            if not isinstance(value, ast.IfExp):
                continue
            fallback = value.orelse
            if (
                isinstance(fallback, ast.Dict)
                and len(fallback.keys) == 1
                and isinstance(fallback.keys[0], ast.Constant)
                and fallback.keys[0].value == "type"
                and isinstance(fallback.values[0], ast.Constant)
                and fallback.values[0].value == "disabled"
            ):
                default_safe = True
        self.assertTrue(
            default_safe,
            "_chat 本体必须保持 default-safe：payload[\"thinking\"] 缺省回退"
            " {\"type\": \"disabled\"}（L1a 钉；改回裸缺省=思考档成本复发）",
        )


if __name__ == "__main__":
    unittest.main()
