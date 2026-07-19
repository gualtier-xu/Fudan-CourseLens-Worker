from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_production_process_installs_media_tools_before_worker(self):
        workflow = (ROOT / ".github" / "workflows" / "process.yml").read_text(encoding="utf-8")
        install = workflow.index("sudo apt-get install --no-install-recommends --yes curl ffmpeg")
        process = workflow.index("name: Process encrypted job")
        self.assertLess(install, process)


if __name__ == "__main__":
    unittest.main()
