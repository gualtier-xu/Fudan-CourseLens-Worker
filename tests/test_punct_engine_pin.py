from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PunctEnginePinGuardTests(unittest.TestCase):
    """N10 回归逃脱守卫：标点引擎依赖必须钉在部署面（代代缺失的逃脱在案）。

    历史教训：ct-punc 引擎依赖曾代代缺失而单测面全绿（测试用注入式假引擎，
    生产静默回落无标点恢复，R1-N10 定谳）；本守卫闭掉「依赖再漂移」的逃脱
    通道：requirements 必须声明引擎真实传递依赖、引擎本体禁入（多消费者同
    解析必 ResolutionImpossible，R1-N17）、生产与门驱动两处引擎钉一致。
    """

    def test_requirements_declare_punct_engine_transitives(self):
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        for pin in ("jieba==", "kaldi-native-fbank==", "librosa==", "PyYAML==",
                    "scipy==", "sentencepiece=="):
            self.assertIn(pin, text)
        # 引擎本体不得进 requirements：其元数据 numpy<=1.26.4 与库钉
        # numpy==2.2.6 同解析必冲突，且本文件被多个工作流/私仓 CI 直接安装。
        self.assertNotRegex(text, r"(?m)^funasr-onnx")

    def test_engine_pin_is_consistent_across_install_faces(self):
        process = (ROOT / ".github" / "workflows" / "process.yml").read_text(encoding="utf-8")
        proof = (ROOT / "scripts" / "punct_inference_proof.py").read_text(encoding="utf-8")
        match = re.search(r"funasr-onnx==(\d+\.\d+\.\d+)", process)
        self.assertIsNotNone(match, "process.yml must declare the engine pin")
        engine_pin = f"funasr-onnx=={match.group(1)}"
        self.assertIn(engine_pin, proof)
        # 生产分步安装与真推理门驱动必须同配方：核心集先行、引擎 --no-deps。
        self.assertIn("--no-deps", proof)
        self.assertIn("requirements-core", proof)

    def test_process_runtime_install_splits_funasr_no_deps(self):
        # N10 案 a：先装去 funasr 的核心集、再 --no-deps 单装引擎本体。
        workflow = (ROOT / ".github" / "workflows" / "process.yml").read_text(encoding="utf-8")
        self.assertIn("requirements-core.txt", workflow)
        self.assertIn("--no-deps", workflow)
        self.assertLess(
            workflow.index("pip install --requirement /tmp/requirements-core.txt"),
            workflow.index("pip install --no-deps funasr-onnx"),
        )

    def test_ci_runs_real_punct_inference_gate(self):
        # N10 合并门（fail-closed）：无真推理证据的标点引擎不进 pin。
        source = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("  punct-inference:", source)
        self.assertIn("scripts/punct_inference_proof.py", source)


if __name__ == "__main__":
    unittest.main()
