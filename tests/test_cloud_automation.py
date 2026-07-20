from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from nacl.secret import SecretBox

from courselens_worker.cloud_automation import (
    STATE_SCHEMA,
    _empty_state,
    _open_state,
    _reset_daily_budget,
    _rules_need_ai,
    _seal_state,
    _send_mail,
)


class CloudAutomationTests(unittest.TestCase):
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

    def test_disabled_email_does_not_open_network_connection(self):
        with patch("smtplib.SMTP_SSL") as smtp:
            result = _send_mail(
                {"enabled": False}, "", "", "",
                subject="test", code="ok", counts={}, elapsed=1, budget={},
            )
        self.assertTrue(result)
        smtp.assert_not_called()

    def test_email_includes_course_names_only_after_explicit_opt_in(self):
        client = MagicMock()
        context = MagicMock()
        context.__enter__.return_value = client
        with patch("smtplib.SMTP_SSL", return_value=context):
            result = _send_mail(
                {"enabled": True, "provider": "gmail", "include_course_names": True},
                "sender@example.com", "app-password", "recipient@example.com",
                subject="test", code="ok",
                counts={"discovered": 1, "processed": 1, "failed": 0},
                elapsed=1, budget={}, course_names=["课程 A\nInjected: no"],
            )
        self.assertTrue(result)
        message = client.send_message.call_args.args[0]
        body = message.get_content()
        self.assertIn("courses=", body)
        self.assertIn("课程 A Injected: no", body)

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


if __name__ == "__main__":
    unittest.main()
