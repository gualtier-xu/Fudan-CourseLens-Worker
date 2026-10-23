"""Local pre-publish gate for the public mirror boundary rules.

The full boundary scan (including its git-history leg) only runs inside
the public mirror CI, so a private tree can stay completely green while
shipping a string the public gate rejects — exactly how the junk-filter
blacklist label ``icourse-notice-page`` slipped through a 294-test local
run and red-lined the mirror publish.  This module runs the real scanner
module's tree legs against the current tree on every local test run, so
the violation surfaces here, before a release ever ships.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

WORKER_ROOT = Path(__file__).resolve().parents[1]
SCANNER_PATH = WORKER_ROOT / "scripts" / "check_public_boundary.py"


def _load_scanner():
    spec = importlib.util.spec_from_file_location("check_public_boundary", SCANNER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"boundary scanner not found at {SCANNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicBoundaryTreeGateTests(unittest.TestCase):
    def test_current_tree_passes_public_boundary_tree_legs(self):
        scanner = _load_scanner()
        self.assertEqual(scanner.tree_failures(), [])

    def test_detector_still_bites_on_platform_host_pattern(self):
        scanner = _load_scanner()
        planted = scanner._violations("sample.py", 'host = "icourse-notice.example"')
        self.assertIn("sample.py: platform host", planted)
        self.assertEqual(planted, ["sample.py: platform host"])


if __name__ == "__main__":
    unittest.main()
