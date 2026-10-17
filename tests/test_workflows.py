from __future__ import annotations

import sys
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
        # N5：ffmpeg 快路径一探即过（runner 预装 curl，装列表去 curl）；冷
        # 路径保持 sed 源清理→update→install 原序列且仍在 worker 步之前。
        self.assertRegex(workflow, r"command -v ffmpeg >/dev/null 2>&1 \|\| \{")
        self.assertNotIn("--yes curl ffmpeg", workflow)
        self.assertRegex(workflow, r"sed -i '/dl\\.google\\.com/d'")
        self.assertLess(
            workflow.index("sed -i '/dl\\.google\\.com/d'"),
            workflow.index("sudo apt-get update"),
        )
        install = workflow.index("sudo apt-get install --no-install-recommends --yes ffmpeg")
        process = workflow.index("name: Process encrypted job")
        self.assertLess(install, process)
        self.assertIn("COURSELENS_WORKFLOW_PROFILE: process-v1", workflow)

    def test_echo_concurrency_group_carries_task_id(self):
        """N19：echo 并发组与 process/llm 同形（带 task_id 维度），同仓并发
        echo 不再互斥排队。"""
        workflow = (ROOT / ".github" / "workflows" / "echo.yml").read_text(encoding="utf-8")
        expected = "group: courselens-compute-${{ github.repository_id }}-${{ inputs.task_id }}"
        self.assertIn(expected, workflow)
        for name in ("process.yml", "llm.yml"):
            sibling = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
            self.assertIn(expected, sibling, f"{name} 应为同形并发组")

    def test_ocr_concurrency_fallback_default_matches_bootstrap_pin(self):
        """N9（夜14-R1）：三个工作流的 OCR 并发回退默认与客户端 bootstrap 钉值
        一致（2）；执行仓 Variables 仍可下调，代码侧帽 min(2, env)（ocr.py）
        不变。"""
        expected = "COURSELENS_OCR_CONCURRENCY: ${{ vars.COURSELENS_OCR_CONCURRENCY || '2' }}"
        for name in ("process.yml", "llm.yml", "cloud-daily.yml"):
            source = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
            with self.subTest(workflow=name):
                self.assertIn(expected, source)
                self.assertNotIn(
                    "COURSELENS_OCR_CONCURRENCY: ${{ vars.COURSELENS_OCR_CONCURRENCY || '1' }}",
                    source,
                )

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
        # 精简装机组与库钉同版（快速通道 pip 集，OCR 腿 rapidocr 在内；
        # D12-FIX-1：platform_session 硬导入族 Crypto+curl_cffi 补入八钉——
        # 缺任一即 pure-LLM 作业 ModuleNotFoundError，D-20261009-12 先例）。
        for pin in ("PyNaCl==1.6.2", "requests==2.34.2", "zstandard==0.25.0",
                    "numpy==2.2.6", "Pillow==12.3.0", "rapidocr-onnxruntime==1.4.4",
                    "pycryptodome==3.23.0", "curl-cffi==0.15.0"):
            self.assertIn(pin, source)

    def test_llm_install_pins_exist_in_requirements(self):
        """D12-FIX-1 防漂移钉：llm.yml 精简装机行的每个 name==version 都必须
        逐字存在于 worker/requirements.txt（权威钉源）。六钉漏
        pycryptodome/curl-cffi 先例（D-20261009-12）的回归门——新任务若往
        llm.yml 加钉而忘了 requirements（或反之），此处即红。"""
        source = (ROOT / ".github" / "workflows" / "llm.yml").read_text(encoding="utf-8")
        install_line = next(
            line.strip() for line in source.splitlines()
            if line.strip().startswith("run: python -m pip install ")
        )
        pins = install_line.removeprefix("run: python -m pip install ").split()
        self.assertGreaterEqual(len(pins), 8)
        requirements = (
            ROOT / "requirements.txt"
        ).read_text(encoding="utf-8").splitlines()
        for pin in pins:
            with self.subTest(pin=pin):
                self.assertIn(pin, requirements)

    def test_platform_session_import_family_is_installable_from_llm_pins(self):
        """D12-FIX-1 闭包钉：platform_session 模块级第三方导入族必须全部可由
        llm.yml 装机行满足——以 ast 静态取导入并按导入名↔发行名映射核对，
        防再漏（D-20261009-12 先例：Crypto/curl_cffi 双缺致 pure-LLM 全灭）。"""
        import ast

        session_tree = ast.parse(
            (ROOT / "courselens_worker" / "platform_session.py").read_text(
                encoding="utf-8"
            )
        )
        third_party = set()
        for node in session_tree.body:  # 仅模块级导入（D-20261009-12 死因面）
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else ([node.module] if node.module else [])
            )
            for name in names:
                top = name.split(".")[0]
                if top not in sys.stdlib_module_names and top != "courselens_worker":
                    third_party.add(top)
        # platform_session 的第三方模块级闭包恰为三件（相对导入 .source 不计）。
        self.assertEqual(third_party, {"Crypto", "curl_cffi", "requests"})
        source = (ROOT / ".github" / "workflows" / "llm.yml").read_text(encoding="utf-8")
        install_line = next(
            line.strip() for line in source.splitlines()
            if line.strip().startswith("run: python -m pip install ")
        )
        pinned_names = {
            pin.split("==")[0] for pin in
            install_line.removeprefix("run: python -m pip install ").split()
        }
        # 导入名→发行名（PyPI 名）映射闭集。
        import_to_dist = {"Crypto": "pycryptodome", "curl_cffi": "curl-cffi", "requests": "requests"}
        for module_name in sorted(third_party):
            with self.subTest(module=module_name):
                self.assertIn(import_to_dist[module_name], pinned_names)


if __name__ == "__main__":
    unittest.main()
