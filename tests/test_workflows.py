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
        install = workflow.index("sudo apt-get install --no-install-recommends --yes curl ffmpeg")
        process = workflow.index("name: Process encrypted job")
        self.assertLess(install, process)

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
        self.assertIn("merge-base --is-ancestor HEAD origin/main", source)
        self.assertIn("repository: gualtier-xu/Fudan-CourseLens-Private", source)
        self.assertIn("repositories: Fudan-CourseLens-Private", source)
        self.assertIn("repositories: Fudan-CourseLens", source)
        self.assertIn("persist-credentials: false", source)
        self.assertNotIn("pull_request:", source)


if __name__ == "__main__":
    unittest.main()
