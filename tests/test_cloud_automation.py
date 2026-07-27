from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from nacl.secret import SecretBox

from courselens_worker.cloud_automation import (
    STATE_SCHEMA,
    _empty_state,
    _open_state,
    _reset_daily_budget,
    _rules_need_ai,
    _seal_state,
)


_INSTALL_MODELS_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "install_models.py"
)
_INSTALL_MODELS_SPEC = importlib.util.spec_from_file_location(
    "courselens_install_models", _INSTALL_MODELS_PATH
)
assert _INSTALL_MODELS_SPEC is not None and _INSTALL_MODELS_SPEC.loader is not None
install_models = importlib.util.module_from_spec(_INSTALL_MODELS_SPEC)
_INSTALL_MODELS_SPEC.loader.exec_module(install_models)


class CloudAutomationTests(unittest.TestCase):
    @staticmethod
    def _model_archive(path: Path, member: tarfile.TarInfo, content: bytes = b"") -> None:
        with tarfile.open(path, "w:bz2") as archive:
            archive.addfile(member, io.BytesIO(content) if member.isfile() else None)

    def test_model_extraction_is_python310_compatible_and_rejects_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "model.tar.bz2"
            member = tarfile.TarInfo("model/tokens.txt")
            member.size = 6
            self._model_archive(archive, member, b"token\n")
            destination = root / "models"
            install_models._safe_extract(archive, destination)
            self.assertEqual((destination / "model" / "tokens.txt").read_bytes(), b"token\n")

            link_archive = root / "link.tar.bz2"
            link = tarfile.TarInfo("model/link")
            link.type = tarfile.SYMTYPE
            link.linkname = "../outside"
            self._model_archive(link_archive, link)
            with self.assertRaisesRegex(RuntimeError, "unsafe member"):
                install_models._safe_extract(link_archive, destination)

    def test_state_is_encrypted_and_tamper_rejected(self):
        key = os.urandom(SecretBox.KEY_SIZE)
        state = _empty_state()
        state["seen"] = {"course": ["lecture"]}
        raw = _seal_state(state, key)
        self.assertNotIn(b"course", raw)
        self.assertEqual(_open_state(raw, key)["schema"], STATE_SCHEMA)
        envelope = json.loads(raw)
        ciphertext = bytearray(base64.b64decode(envelope["ciphertext"]))
        ciphertext[-1] ^= 1
        envelope["ciphertext"] = base64.b64encode(ciphertext).decode()
        with self.assertRaises(Exception):
            _open_state(json.dumps(envelope).encode(), key)

    def test_budget_resets_on_beijing_date_change(self):
        state = _empty_state()
        state["budget"] = {"date": "2000-01-01", "lectures": 2, "runner_minutes": 300, "deepseek_tokens": 100000}
        value = _reset_daily_budget(state)
        self.assertEqual(value["lectures"], 0)
        self.assertEqual(value["runner_minutes"], 0)
        self.assertEqual(value["deepseek_tokens"], 0)

    def test_ai_key_is_required_only_for_rules_that_need_ai(self):
        self.assertFalse(_rules_need_ai({"rules": [{"discovery_only": True}]}))
        self.assertFalse(_rules_need_ai({
            "rules": [{"discovery_only": False, "subtitle_mode": "fast"}],
        }))
        self.assertTrue(_rules_need_ai({
            "rules": [{"discovery_only": False, "subtitle_mode": "standard"}],
        }))
        self.assertTrue(_rules_need_ai({
            "rules": [{"discovery_only": False, "summary": True}],
        }))

    def test_workflow_has_24_half_hour_schedules_and_no_pull_request_secrets(self):
        workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "cloud-daily.yml"
        text = workflow.read_text(encoding="utf-8")
        self.assertEqual(text.count('- cron: "30 '), 24)
        self.assertNotIn("pull_request:", text)
        self.assertIn("COURSELENS_CLOUD_ENABLED == 'true'", text)
        self.assertIn("actions: write", text)

    def test_cloud_workflows_and_runtime_do_not_support_smtp(self):
        root = Path(__file__).resolve().parents[1]
        paths = [
            root / "courselens_worker" / "cloud_automation.py",
            root / ".github" / "workflows" / "cloud-daily.yml",
            root / ".github" / "workflows" / "cloud-verify.yml",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
        self.assertNotIn("smtp", combined)
        self.assertNotIn("emailmessage", combined)

    def test_cloud_workflows_use_the_macos_route_and_os_scoped_models(self):
        root = Path(__file__).resolve().parents[1]
        daily = (root / ".github" / "workflows" / "cloud-daily.yml").read_text(
            encoding="utf-8"
        )
        verify = (root / ".github" / "workflows" / "cloud-verify.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("runs-on: macos-15", daily)
        self.assertIn("runs-on: macos-15", verify)
        self.assertIn("${{ runner.os }}-asr-models", daily)
        self.assertIn("brew list ffmpeg", daily)
        self.assertNotIn("apt-get", daily)


if __name__ == "__main__":
    unittest.main()
