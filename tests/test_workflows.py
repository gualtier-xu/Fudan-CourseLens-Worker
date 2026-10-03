from __future__ import annotations

import unittest
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_repository_attributes_force_lf_for_all_text(self):
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("* text=auto eol=lf", attributes)

    def test_production_process_installs_media_tools_before_worker(self):
        workflow = (ROOT / ".github" / "workflows" / "process.yml").read_text(encoding="utf-8")
        self.assertRegex(workflow, r"sed -i '/dl\\.google\\.com/d'")
        self.assertLess(
            workflow.index("sed -i '/dl\\.google\\.com/d'"),
            workflow.index("sudo apt-get update"),
        )
        install = workflow.index("sudo apt-get install --no-install-recommends --yes curl ffmpeg")
        process = workflow.index("name: Process encrypted job")
        self.assertLess(install, process)
        self.assertIn("COURSELENS_WORKFLOW_PROFILE: process-v1", workflow)

    def test_all_actions_are_pinned_to_full_commit_sha(self):
        for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
            source = path.read_text(encoding="utf-8")
            for action in re.findall(r"uses:\s*([^\s]+)", source):
                with self.subTest(path=path.name, action=action):
                    self.assertRegex(action, r"^[^@]+@[0-9a-f]{40}$")

    def test_mirror_policy_runs_base_validator_without_secrets(self):
        source = (ROOT / ".github" / "workflows" / "mirror-policy.yml").read_text(encoding="utf-8")
        self.assertIn("pull_request_target", source)
        self.assertIn("persist-credentials: false", source)
        self.assertIn("trusted/scripts/check_generated_mirror.py", source)
        self.assertNotIn("secrets.", source)

    def test_ci_runs_gitleaks_protocol_and_boundary_jobs(self):
        source = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn('PYTHONDONTWRITEBYTECODE: "1"', source)
        self.assertIn("  unit:\n", source)
        self.assertIn("gitleaks/gitleaks-action@ff98106e4c7b2bc287b24eaf42907196329070c7", source)
        self.assertIn("protocol:", source)
        self.assertIn("boundary:", source)

    def test_public_release_controller_separates_source_and_publish_apps(self):
        source = (ROOT / ".github" / "workflows" / "publish-mirror.yml").read_text(encoding="utf-8")
        self.assertIn("environment: worker-mirror-release", source)
        self.assertIn("if: github.ref == 'refs/heads/main'", source)
        self.assertIn("permissions:\n  contents: read", source)
        self.assertIn("WORKER_MIRROR_SOURCE_APP_PRIVATE_KEY", source)
        self.assertIn("WORKER_MIRROR_PUBLISHER_APP_PRIVATE_KEY", source)
        self.assertIn("merge-base --is-ancestor origin/main HEAD", source)
        self.assertNotIn("merge-base --is-ancestor HEAD origin/main", source)
        self.assertIn("repository: gualtier-xu-co/Fudan-CourseLens-Private", source)
        self.assertIn("repositories: Fudan-CourseLens-Private", source)
        self.assertIn("repositories: Fudan-CourseLens-Worker", source)
        self.assertIn("persist-credentials: false", source)
        self.assertNotIn("pull_request:", source)

    def test_llm_workflow_is_minimal_and_media_free(self):
        # N1：纯 LLM 任务（summary/chapters/quality_judge/answer-only/带 PPT 的
        # OCR 总结）走精简工作流，免付 apt/ffmpeg/ASR 模型缓存三步媒体环境。
        source = (ROOT / ".github" / "workflows" / "llm.yml").read_text(encoding="utf-8")
        self.assertIn("COURSELENS_WORKFLOW_PROFILE: llm-v1", source)
        self.assertIn("environment: courselens-worker", source)
        # 媒体面三步（apt 媒体工具、模型缓存、模型安装）与模型根 env 全缺席：
        # 媒体 kind 误入本工作流时按既有闭集语义 fail-closed。
        self.assertNotIn("apt-get", source)
        self.assertNotIn("actions/cache", source)
        self.assertNotIn("install_models", source)
        self.assertNotIn("COURSELENS_MODEL_ROOT", source)
        # 精简装机组与库钉同版（快速通道 pip 集，OCR 腿 rapidocr 在内）。
        for pin in ("PyNaCl==1.6.2", "requests==2.34.2", "zstandard==0.25.0",
                    "numpy==2.2.6", "Pillow==12.3.0", "rapidocr-onnxruntime==1.4.4"):
            self.assertIn(pin, source)


if __name__ == "__main__":
    unittest.main()
