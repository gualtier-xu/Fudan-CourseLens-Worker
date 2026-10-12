from __future__ import annotations

import base64
import contextlib
import importlib.util
import io
import json
import os
import re
import tarfile
import tempfile
import time
import unittest
import unittest.mock
from datetime import datetime
from pathlib import Path
from nacl.secret import SecretBox
from nacl.signing import VerifyKey

from courselens_worker import cloud_automation
from courselens_worker.cloud_automation import (
    CHECKPOINT_KEEP_LIMIT,
    GATE_TZ,
    OUTPUT_BUNDLE,
    PENDING_TTL_SECONDS,
    RULES_SCHEMA,
    STATE_SCHEMA,
    _dedupe_key,
    _empty_state,
    _open_state,
    _reset_daily_budget,
    _rules_need_ai,
    _seal_state,
)
from courselens_worker.platform_session import PlatformSessionError


_INSTALL_MODELS_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "install_models.py"
)
_INSTALL_MODELS_SPEC = importlib.util.spec_from_file_location(
    "courselens_install_models", _INSTALL_MODELS_PATH
)
assert _INSTALL_MODELS_SPEC is not None and _INSTALL_MODELS_SPEC.loader is not None
install_models = importlib.util.module_from_spec(_INSTALL_MODELS_SPEC)
_INSTALL_MODELS_SPEC.loader.exec_module(install_models)


def _signing_keypair():
    from nacl.signing import SigningKey
    private = SigningKey.generate()
    return base64.b64encode(bytes(private)).decode(), base64.b64encode(bytes(private.verify_key)).decode()


def make_rules(*, rules=None, config_hash="d" * 64, account_id="2020001", budget=None):
    return {
        "schema": RULES_SCHEMA,
        "config_hash": config_hash,
        "account_id": account_id,
        "budget": budget or {"max_lectures": 2, "max_runner_minutes": 300, "max_deepseek_tokens": 100000},
        "rules": rules if rules is not None else [],
    }


def make_course_rule(course_id="36941", *, baseline=None, priority=50):
    return {
        "course_id": course_id,
        "priority": priority,
        "max_lecture_minutes": 240,
        "selection_generation": 1,
        "baseline": list(baseline or []),
    }


class FakeConnector:
    def __init__(self, courses, slides=None, schedule_rows=None):
        self.courses = courses
        self.closed = False
        self.slides = slides if slides is not None else []
        # SMART-SCHED gate leg plus structural-cap counters: a gate-skip run
        # must cost exactly one schedule GET and zero discovery/media/LLM work.
        self.schedule_rows = list(schedule_rows or [])
        self.schedule_calls = 0
        self.discover_calls = 0
        self.media_calls = 0
        self.slide_calls = 0

    def today_schedule_rows(self):
        self.schedule_calls += 1
        return list(self.schedule_rows)

    def discover_authorized_courses(self):
        self.discover_calls += 1
        return self.courses

    def media_source(self, course_id, sub_id):
        self.media_calls += 1
        return {"kind": "synthetic"}

    def slide_sources(self, course_id, sub_id):
        self.slide_calls += 1
        return list(self.slides)

    def close(self):
        self.closed = True


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self._payload = payload
        self.content = json.dumps(payload).encode("utf-8")

    def json(self):
        return self._payload


class FakeGitHubState:
    """Replaces the GitHub artifact API with an in-memory store."""

    def __init__(self):
        self.artifacts = []
        self.next_id = 1
        self.deleted = []

    def request(self, method, path, **kwargs):
        if method == "GET" and path == "/actions/artifacts":
            return FakeResponse({"artifacts": [item for item in self.artifacts if not item.get("expired")]})
        if method == "GET" and path.endswith("/zip"):
            artifact_id = int(path.split("/")[-2])
            item = next(entry for entry in self.artifacts if entry["id"] == artifact_id)
            archive = io.BytesIO()
            import zipfile
            with zipfile.ZipFile(archive, "w") as bundle:
                for name, raw in item["files"].items():
                    bundle.writestr(name, raw)
            response = FakeResponse({})
            response.content = archive.getvalue()
            return response
        if method == "DELETE" and "/actions/artifacts/" in path:
            artifact_id = int(path.rstrip("/").split("/")[-1])
            self.deleted.append(artifact_id)
            self.artifacts = [item for item in self.artifacts if item["id"] != artifact_id]
            return FakeResponse({}, status_code=204)
        raise AssertionError(f"unexpected request {method} {path}")

    def upload_state(self, raw):
        self.artifacts.append({
            "id": self.next_id, "name": f"courselens-cloud-state-{self.next_id}-1",
            "created_at": f"2026-09-11T0{self.next_id}:00:00Z", "expired": False,
            "files": {"state.box.json": raw.decode("utf-8")},
        })
        self.next_id += 1


class WorkerEnvTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._previous_cwd = os.getcwd()
        os.chdir(self._tmp.name)
        self.github_state = FakeGitHubState()
        self.signing_private, self.signing_public = _signing_keypair()
        self.state_key = base64.b64encode(bytes(range(SecretBox.KEY_SIZE))).decode("ascii")
        from courselens_worker.protocol import generate_box_keypair
        result_private, result_public = generate_box_keypair()
        self.result_private = result_private
        self._patches = [
            unittest.mock.patch.object(cloud_automation, "_github_request", self.github_state.request),
            unittest.mock.patch.dict(os.environ, {
                "GITHUB_TOKEN": "synthetic-token",
                "GITHUB_REPOSITORY": "synthetic-owner/courselens-worker-synthetic",
                "COURSELENS_CLOUD_STATE_KEY": self.state_key,
                "COURSELENS_CLOUD_RESULT_PUBLIC_KEY": result_public,
                "WORKER_SIGNING_PRIVATE_KEY": self.signing_private,
                "COURSELENS_CLOUD_DEEPSEEK_API_KEY": "sk-synthetic",
                # v3 run binding defaults: verified protocol/config, scheduled run.
                "COURSELENS_CLOUD_EXPECTED_PROTOCOL_VERSION": RULES_SCHEMA,
                "COURSELENS_CLOUD_EXPECTED_CONFIG_HASH": "d" * 64,
                "COURSELENS_CLOUD_MANUAL_DISPATCH": "false",
                "COURSELENS_CLOUD_DISPATCH_CONFIG_HASH": "",
                "COURSELENS_CLOUD_ENABLED_FLAG": "true",
            }),
        ]
        for patch in self._patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self._patches):
            patch.stop()
        os.chdir(self._previous_cwd)
        self._tmp.cleanup()

    def set_rules(self, rules):
        os.environ["COURSELENS_CLOUD_RULES_JSON"] = json.dumps(rules)

    def read_persisted_state(self):
        path = Path(".work/cloud-state/state.box.json")
        if not path.exists():
            return None
        key = base64.b64decode(self.state_key)
        return _open_state(path.read_bytes(), key)


import unittest.mock  # noqa: F401  (re-exported for WorkerEnvTestCase patches)


class WorkflowStaticTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    def setUp(self):
        self.daily = (self.root / ".github" / "workflows" / "cloud-daily.yml").read_text(encoding="utf-8")
        self.verify = (self.root / ".github" / "workflows" / "cloud-verify.yml").read_text(encoding="utf-8")

    def test_daily_declares_the_fixed_weekday_grid_and_nightly_fallback(self):
        crons = re.findall(r'- cron: "([^"]+)"', self.daily)
        self.assertEqual(crons, [
            "15 1 * * 1-5", "10 2 * * 1-5", "10 3 * * 1-5", "5 4 * * 1-5",
            "0 5 * * 1-5", "45 6 * * 1-5", "40 7 * * 1-5", "40 8 * * 1-5",
            "35 9 * * 1-5", "30 10 * * 1-5", "0 14 * * *",
        ])
        # Event fan-out stays closed: the grid is static, never env-driven.
        self.assertNotIn("COURSELENS_CLOUD_CRON", self.daily)
        self.assertNotIn("github.event.schedule", self.daily)

    def test_daily_gate_hook_is_acceptance_only_and_inert_on_schedule(self):
        self.assertIn("gate_fake_time:", self.daily)
        self.assertIn("COURSELENS_CLOUD_GATE_FAKE_TIME: ${{ inputs.gate_fake_time || '' }}", self.daily)

    def test_workflows_never_expose_secrets_to_pull_requests_and_pin_by_full_sha(self):
        for text in (self.daily, self.verify):
            self.assertNotIn("pull_request", text)
            self.assertIn("environment: courselens-worker", text)
            self.assertIn("contents: read", text)
            self.assertIn("actions: write", text)
            self.assertIn("cancel-in-progress: false", text)
            for action in re.findall(r"uses:\s*(\S+)", text):
                self.assertRegex(action, r"@[0-9a-f]{40}$", f"action not pinned by full SHA: {action}")
        self.assertIn("timeout-minutes:", self.daily)
        self.assertIn("timeout-minutes:", self.verify)

    def test_result_retention_is_30_days_and_only_state_keeps_the_90_day_ceiling(self):
        # 结果 artifact = 30 天；只有最新一份加密续跑状态保留 90 天上限
        self.assertIn("retention-days: 30", self.daily)
        self.assertEqual(self.daily.count("retention-days: 90"), 1)
        self.assertEqual(self.daily.count("retention-days: 30"), 1)
        # 90 天只属于状态 artifact 的上传步骤，不回到结果步骤
        result_block = self.daily[self.daily.index("courselens-cloud-result"):self.daily.index("courselens-cloud-state")]
        self.assertIn("retention-days: 30", result_block)
        self.assertNotIn("retention-days: 90", result_block)
        state_block = self.daily[self.daily.index("courselens-cloud-state"):]
        self.assertIn("retention-days: 90", state_block)
        self.assertIn("rm -rf .work", self.daily)
        self.assertIn("rm -rf .work", self.verify)

    def test_verify_receives_signing_key_for_signed_evidence(self):
        self.assertIn("WORKER_SIGNING_PRIVATE_KEY: ${{ secrets.WORKER_SIGNING_PRIVATE_KEY }}", self.verify)

    def test_cloud_workflows_and_runtime_do_not_support_smtp(self):
        paths = [
            self.root / "courselens_worker" / "cloud_automation.py",
            self.root / ".github" / "workflows" / "cloud-daily.yml",
            self.root / ".github" / "workflows" / "cloud-verify.yml",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
        self.assertNotIn("smtp", combined)
        self.assertNotIn("emailmessage", combined)

    def test_cloud_workflows_use_the_macos_route_and_os_scoped_models(self):
        self.assertIn("runs-on: macos-15", self.daily)
        self.assertIn("runs-on: macos-15", self.verify)
        self.assertIn("${{ runner.os }}-asr-models", self.daily)
        self.assertIn("brew list ffmpeg", self.daily)
        self.assertNotIn("apt-get", self.daily)


class ScheduleGateTests(unittest.TestCase):
    """SMART-SCHED 智能门单元钉：激活矩阵/伪造时刻 fail-closed/窗匹配/fail-open/账面预检。

    门是预过滤器不是裁决器：秒退必须持肯定性空证据，一切不确定一律过门
    （fail-open）；账面预检只读 completed/pending/baseline，恰一次语义零弱化。
    """

    def _moment(self, hour, minute):
        # 2026-10-06 is a Tuesday; the day window is 08:30-19:00 Beijing.
        return datetime(2026, 10, 6, hour, minute, tzinfo=GATE_TZ)

    def _rules(self, baseline=None):
        return make_rules(rules=[make_course_rule("36941", baseline=baseline or [])])

    def _state(self, *completed_pairs):
        state = _empty_state()
        seen = {}
        rules = self._rules()
        for course_id, sub_id in completed_pairs:
            seen.setdefault(course_id, []).append(_dedupe_key(rules, course_id, sub_id))
        state["seen"] = seen
        return state

    @staticmethod
    def _row(sub_id="s-1", *, date="2026-10-06", hour=9, minute=40, course_id="36941"):
        return {
            "course_id": course_id, "sub_id": sub_id, "date": date,
            "end_precise": datetime(2026, 10, 6, hour, minute, tzinfo=GATE_TZ),
        }

    def test_activation_matrix_bounds_gate_to_daytime_schedule_runs(self):
        self.assertFalse(cloud_automation._gate_active("schedule", self._moment(8, 29)))
        self.assertTrue(cloud_automation._gate_active("schedule", self._moment(8, 30)))
        self.assertTrue(cloud_automation._gate_active("schedule", self._moment(18, 59)))
        self.assertFalse(cloud_automation._gate_active("schedule", self._moment(19, 0)))
        # Nightly fallback, manual dispatch, and unknown kinds stay bypassed.
        self.assertFalse(cloud_automation._gate_active("schedule", self._moment(22, 5)))
        self.assertFalse(cloud_automation._gate_active("manual", self._moment(10, 10)))
        self.assertFalse(cloud_automation._gate_active("", self._moment(10, 10)))

    def test_acceptance_gate_needs_fake_time_inside_the_day_window(self):
        with unittest.mock.patch.dict(os.environ, {"COURSELENS_CLOUD_GATE_FAKE_TIME": "2026-10-06T10:10:00+08:00"}):
            self.assertTrue(cloud_automation._gate_active("acceptance", self._moment(10, 10)))
        with unittest.mock.patch.dict(os.environ, {"COURSELENS_CLOUD_GATE_FAKE_TIME": ""}):
            self.assertFalse(cloud_automation._gate_active("acceptance", self._moment(10, 10)))
        # Outside the day window the acceptance hook stays inert (22:05 走全管线).
        with unittest.mock.patch.dict(os.environ, {"COURSELENS_CLOUD_GATE_FAKE_TIME": "2026-10-06T22:05:00+08:00"}):
            self.assertFalse(cloud_automation._gate_active("acceptance", self._moment(22, 5)))

    def test_gate_now_parses_fake_time_and_fails_closed_without_guessing(self):
        with unittest.mock.patch.dict(os.environ, {"COURSELENS_CLOUD_GATE_FAKE_TIME": "2026-10-06T10:10:00+08:00"}):
            self.assertEqual(cloud_automation._gate_now("acceptance"), self._moment(10, 10))
        with unittest.mock.patch.dict(os.environ, {"COURSELENS_CLOUD_GATE_FAKE_TIME": ""}):
            self.assertIsNone(cloud_automation._gate_now("acceptance"))
        for raw in ("not-a-time", "2026-10-06T10:10:00"):
            with unittest.mock.patch.dict(os.environ, {"COURSELENS_CLOUD_GATE_FAKE_TIME": raw}):
                with self.assertRaises(cloud_automation.CloudAutomationError) as caught:
                    cloud_automation._gate_now("acceptance")
                self.assertEqual(caught.exception.code, "cloud_gate_time_invalid")
        # Manual bypasses even with a well-formed value present; schedule
        # decides only inside the day window.
        with unittest.mock.patch.dict(os.environ, {"COURSELENS_CLOUD_GATE_FAKE_TIME": "2026-10-06T10:10:00+08:00"}):
            self.assertIsNone(cloud_automation._gate_now("manual"))
        self.assertIsNone(cloud_automation._gate_now("schedule", now=self._moment(22, 5)))
        self.assertEqual(
            cloud_automation._gate_now("schedule", now=self._moment(8, 30)),
            self._moment(8, 30),
        )

    def test_window_match_skips_only_on_positive_empty_evidence(self):
        rules = self._rules()
        state = self._state()
        moment = self._moment(10, 10)
        connector = FakeConnector([])
        gate_pass, stats = cloud_automation._gate_decision(connector, rules, state, moment)
        self.assertEqual((gate_pass, stats["rows"], stats["matched"]), (False, 0, 0))
        # 窗 = 实际启动 - 30min ± 20min → 09:20..10:00；窗外的结束时刻不构成命中。
        connector.schedule_rows = [
            self._row(sub_id="s-early", hour=9, minute=15),
            self._row(sub_id="s-late", hour=10, minute=1),
            self._row(sub_id="s-otherday", date="2026-10-05", hour=9, minute=40),
            self._row(sub_id="s-unselected", course_id="777"),
        ]
        gate_pass, stats = cloud_automation._gate_decision(connector, rules, state, moment)
        self.assertEqual((gate_pass, stats["rows"], stats["matched"]), (False, 4, 0))
        connector.schedule_rows.append(self._row(sub_id="s-9"))
        gate_pass, stats = cloud_automation._gate_decision(connector, rules, state, moment)
        self.assertTrue(gate_pass)
        self.assertEqual(stats["matched"], 1)

    def test_window_edges_are_inclusive_twenty_minute_tolerance(self):
        rules = self._rules()
        state = self._state()
        moment = self._moment(10, 10)
        # 窗 = 09:20..10:00（实际启动 10:10 - 30min ± 20min），两端含。
        for hour, minute, expected in ((9, 19, False), (9, 20, True), (10, 0, True), (10, 1, False)):
            connector = FakeConnector([], schedule_rows=[self._row(hour=hour, minute=minute)])
            gate_pass, _ = cloud_automation._gate_decision(connector, rules, state, moment)
            self.assertEqual(gate_pass, expected, f"end {hour:02d}:{minute:02d} expected pass={expected}")

    def test_uncertain_rows_are_hits_fail_open(self):
        rules = self._rules()
        connector = FakeConnector([], schedule_rows=[
            {"course_id": "36941", "sub_id": "s-1", "date": "2026-10-06", "end_precise": None},
            {"course_id": "36941", "sub_id": "s-2", "date": "", "end_precise": None},
        ])
        gate_pass, stats = cloud_automation._gate_decision(
            connector, rules, self._state(), self._moment(10, 10))
        self.assertTrue(gate_pass)
        self.assertEqual(stats["uncertain"], 2)

    def test_unreadable_schedule_fails_open(self):
        class FailingConnector:
            def today_schedule_rows(self):
                raise PlatformSessionError("platform_course_request_failed")

        gate_pass, stats = cloud_automation._gate_decision(
            FailingConnector(), self._rules(), self._state(), self._moment(10, 10))
        self.assertTrue(gate_pass)
        self.assertEqual(stats["rows"], -1)

    def test_ledger_precheck_skips_only_when_every_hit_is_known(self):
        rules = self._rules()
        moment = self._moment(10, 10)
        hit = self._row(sub_id="s-1")
        # 身份不可绑（缺 sub_id）→ 过门。
        connector = FakeConnector([], schedule_rows=[{**hit, "sub_id": ""}])
        gate_pass, _ = cloud_automation._gate_decision(connector, rules, self._state(), moment)
        self.assertTrue(gate_pass)
        # 基线排除 → 已知 → 秒退。
        baseline_rules = self._rules(baseline=["s-1"])
        connector = FakeConnector([], schedule_rows=[hit])
        gate_pass, stats = cloud_automation._gate_decision(connector, baseline_rules, self._state(), moment)
        self.assertFalse(gate_pass)
        self.assertEqual(stats["known"], 1)
        # 已完成 → 已知 → 秒退。
        connector = FakeConnector([], schedule_rows=[hit])
        gate_pass, stats = cloud_automation._gate_decision(
            connector, rules, self._state(("36941", "s-1")), moment)
        self.assertFalse(gate_pass)
        self.assertEqual(stats["known"], 1)
        # 在途 pending → 已知 → 秒退。
        state = self._state()
        state["pending"] = [{
            "key": _dedupe_key(rules, "36941", "s-1"), "course_id": "36941", "sub_id": "s-1",
            "expires_at": time.time() + 3600,
        }]
        connector = FakeConnector([], schedule_rows=[hit])
        gate_pass, stats = cloud_automation._gate_decision(connector, rules, state, moment)
        self.assertFalse(gate_pass)
        self.assertEqual(stats["known"], 1)
        # 未知线索 → 过门；任一未知行胜过全部已知行。
        connector = FakeConnector([], schedule_rows=[hit])
        gate_pass, _ = cloud_automation._gate_decision(connector, rules, self._state(), moment)
        self.assertTrue(gate_pass)
        unknown = self._row(sub_id="s-2")
        connector = FakeConnector([], schedule_rows=[hit, unknown])
        gate_pass, stats = cloud_automation._gate_decision(
            connector, rules, self._state(("36941", "s-1")), moment)
        self.assertTrue(gate_pass)
        self.assertEqual(stats["known"], 1)


class CloudAutomationTests(WorkerEnvTestCase):
    def test_model_extraction_is_python310_compatible_and_rejects_links(self):
        root = Path(self._tmp.name)
        archive = root / "model.tar.bz2"
        member = tarfile.TarInfo("model/tokens.txt")
        member.size = 6
        with tarfile.open(archive, "w:bz2") as bundle:
            bundle.addfile(member, io.BytesIO(b"token\n"))
        destination = root / "models"
        install_models._safe_extract(archive, destination)
        self.assertEqual((destination / "model" / "tokens.txt").read_bytes(), b"token\n")

        link_archive = root / "link.tar.bz2"
        link = tarfile.TarInfo("model/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../outside"
        with tarfile.open(link_archive, "w:bz2") as bundle:
            bundle.addfile(link)
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

    def test_ai_key_is_required_for_any_selected_course(self):
        # v3 固定包：任何选中课程都含 AI 总结/章节，必须配置 Key。
        self.assertTrue(_rules_need_ai({"rules": [make_course_rule()]}))
        self.assertFalse(_rules_need_ai({"rules": []}))

    def test_rules_envelope_must_be_v3_with_trusted_config_hash(self):
        self.set_rules({"schema": "cloud-automation.v2", "config_hash": "d" * 64})
        with self.assertRaises(cloud_automation.CloudAutomationError) as caught:
            cloud_automation._rules()
        self.assertEqual(caught.exception.code, "cloud_rules_invalid")
        self.set_rules({"schema": RULES_SCHEMA, "config_hash": ""})
        with self.assertRaises(cloud_automation.CloudAutomationError):
            cloud_automation._rules()
        oversize = make_course_rule(baseline=[f"l-{index:04d}" for index in range(401)])
        self.set_rules(make_rules(rules=[oversize]))
        with self.assertRaises(cloud_automation.CloudAutomationError) as caught:
            cloud_automation._rules()
        self.assertEqual(caught.exception.code, "cloud_rules_invalid")
        self.set_rules(make_rules())
        self.assertEqual(cloud_automation._rules()["schema"], RULES_SCHEMA)

    def test_dedupe_key_drops_global_config_and_binds_bundle_and_pipeline(self):
        rules = make_rules()
        base = _dedupe_key(rules, "36941", "l-1")
        self.assertEqual(base, _dedupe_key(rules, "36941", "l-1"))
        self.assertNotEqual(base, _dedupe_key(rules, "36941", "l-2"))
        # A pure settings change (new config hash, budgets, priorities) must
        # never change the completed-work identity.
        self.assertEqual(base, _dedupe_key(
            make_rules(config_hash="e" * 64, budget={
                "max_lectures": 5, "max_runner_minutes": 600, "max_deepseek_tokens": 10,
            }),
            "36941", "l-1",
        ))
        self.assertNotEqual(base, _dedupe_key(make_rules(account_id="2020002"), "36941", "l-1"))

    def test_verify_seals_signed_matching_evidence_without_processing(self):
        # Verification with no selected rules runs without an AI key: the
        # connection check is skipped entirely (no key env in this test).
        os.environ["COURSELENS_CLOUD_DEEPSEEK_API_KEY"] = ""
        self.set_rules(make_rules())
        connector = FakeConnector([])
        with unittest.mock.patch.object(
            cloud_automation, "cloud_session_from_environment", return_value=connector,
        ):
            self.assertEqual(cloud_automation.verify(), 0)
        state = self.read_persisted_state()
        record = state["verification"]
        self.assertEqual(record["config_hash"], "d" * 64)
        self.assertEqual(record["protocol"], RULES_SCHEMA)
        message = json.dumps(
            {key: record[key] for key in ("config_hash", "protocol", "verified_at")},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        VerifyKey(base64.b64decode(self.signing_public)).verify(
            message, bytes.fromhex(record["receipt"]),
        )
        self.assertTrue(connector.closed)

    def test_state_artifact_loss_starts_wallclock_chain_not_counter(self):
        # REVOKE-REVISION-FIX 生命周期钉（E2E-2v5 定谳缺陷，修前红）：revoke/
        # erase/artifact 过期清掉仓内 state artifact 后，新链首件的 revision
        # 必须落在墙钟毫秒域——客户端导入闸的历史计数器天花板（如 prior=2，
        # automation_last_cloud_state 持久化）才能放行新链首件。修前计数器
        # 首件 revision=1 ≤ prior 被静默拒收，撤销→重启用生命周期内 verify
        # 翻转被砖 N 次派发（N=prior）。
        os.environ["COURSELENS_CLOUD_DEEPSEEK_API_KEY"] = ""
        self.set_rules(make_rules())
        floor = int(time.time() * 1000)
        connector = FakeConnector([])
        with unittest.mock.patch.object(
            cloud_automation, "cloud_session_from_environment", return_value=connector,
        ):
            self.assertEqual(cloud_automation.verify(), 0)
        state = self.read_persisted_state()
        self.assertGreaterEqual(state["revision"], floor)

    def test_revision_chain_continues_over_legacy_counter_state(self):
        # 跨代连续性：旧计数器 state（历史值 5）载入后，下一件 revision 仍
        # 严格递增并跳入墙钟毫秒域——计数器→墙钟代际切换不产生可被客户端
        # 天花板拒收的回退值。
        key = base64.b64decode(self.state_key)
        floor = int(time.time() * 1000)
        state = _empty_state()
        state["revision"] = 5
        cloud_automation._persist_state(state, key)
        self.assertGreater(state["revision"], 5)
        self.assertGreaterEqual(state["revision"], floor)

    def test_revision_stays_strictly_monotonic_when_clock_jumps_back(self):
        # 墙钟回拨/同毫秒连写不破坏链内严格递增：max(墙钟毫秒, prior+1) 地板。
        key = base64.b64decode(self.state_key)
        state = _empty_state()
        cloud_automation._persist_state(state, key)
        first = state["revision"]
        with unittest.mock.patch.object(
            cloud_automation.time, "time", return_value=(first - 3_600_000) / 1000.0,
        ):
            cloud_automation._persist_state(state, key)
        self.assertEqual(state["revision"], first + 1)

    @staticmethod
    def _stage_names(output: str) -> list:
        return [
            line.split()[0][len("stage="):]
            for line in output.splitlines()
            if line.startswith("stage=")
        ]

    def test_verify_emits_closed_set_stage_lines_without_ai_key(self):
        # CLOUDVERIFY-FIX P1：无 Key 形态探针跳过 → ai_probe 行缺席（缺席=
        # 跳过，不落假完成行）；已过腿各落一行闭集 stage 行。
        os.environ["COURSELENS_CLOUD_DEEPSEEK_API_KEY"] = ""
        self.set_rules(make_rules())
        connector = FakeConnector([])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), unittest.mock.patch.object(
            cloud_automation, "cloud_session_from_environment", return_value=connector,
        ):
            self.assertEqual(cloud_automation.verify(), 0)
        self.assertEqual(self._stage_names(buffer.getvalue()), ["platform_ok", "verified"])

    def test_verify_emits_ai_probe_stage_line_and_pins_probe_args(self):
        # CLOUDVERIFY-FIX P1+P0 联合走查：探针真实运行时三行齐落
        # （platform_ok → ai_probe → verified），且探针参数=关思考+32 预算。
        self.set_rules(make_rules())
        connector = FakeConnector([])
        captured = []

        def fake_chat(api_key, messages, **kwargs):
            captured.append(kwargs)
            return "OK"

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), unittest.mock.patch.object(
            cloud_automation, "cloud_session_from_environment", return_value=connector,
        ), unittest.mock.patch.object(cloud_automation, "_chat", side_effect=fake_chat):
            self.assertEqual(cloud_automation.verify(), 0)
        self.assertEqual(
            self._stage_names(buffer.getvalue()), ["platform_ok", "ai_probe", "verified"],
        )
        self.assertEqual(captured[0]["thinking"], {"type": "disabled"})
        self.assertEqual(captured[0]["max_tokens"], 32)

    def run_daily_with(self, courses, *, process=None, prefill_state=None, slides=None, env=None, connector=None):
        # The runtime pops one-shot secret env vars; restore them per run.
        os.environ["COURSELENS_CLOUD_STATE_KEY"] = self.state_key
        os.environ["COURSELENS_CLOUD_DEEPSEEK_API_KEY"] = "sk-synthetic"
        rules = make_rules(rules=[make_course_rule("36941")])
        self.set_rules(rules)
        if prefill_state is not None:
            key = base64.b64decode(self.state_key)
            self.github_state.upload_state(_seal_state(prefill_state, key))
        if connector is None:
            connector = FakeConnector(courses, slides=slides)
        processed = []
        if process is not None:
            def process_wrapper(job, *, checkpoint_writer=None):
                processed.append(job)
                return process(job, checkpoint_writer=checkpoint_writer)
        else:
            def process_wrapper(job, *, checkpoint_writer=None):
                processed.append(job)
                return {
                    "outputs": {"subtitle": {"mode": "automatic"}},
                    "metrics": {"elapsed_seconds": 1.0},
                }
        overrides = dict(env or {})
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.dict(os.environ, overrides))
            stack.enter_context(unittest.mock.patch.object(
                cloud_automation, "cloud_session_from_environment", return_value=connector,
            ))
            stack.enter_context(unittest.mock.patch.object(cloud_automation, "process_job", process_wrapper))
            stack.enter_context(unittest.mock.patch.object(
                cloud_automation, "usage_snapshot", return_value={"total_tokens": 0},
            ))
            code = cloud_automation.run_daily()
        # Simulate the workflow's always() upload of the incremental state.
        state_path = Path(".work/cloud-state/state.box.json")
        if state_path.exists():
            self.github_state.upload_state(state_path.read_bytes())
        return code, processed, connector

    def test_exact_opt_in_and_playback_availability_gate_candidates(self):
        courses = [
            {"course_id": "36941", "lectures": [
                {"sub_id": "l-1", "has_playback": True},
                {"sub_id": "l-2", "has_playback": False},
            ]},
            {"course_id": "99999", "lectures": [{"sub_id": "l-9", "has_playback": True}]},
        ]
        code, processed, _ = self.run_daily_with(courses)
        self.assertEqual(code, 0)
        self.assertEqual(len(processed), 1)
        expected_key = _dedupe_key(make_rules(rules=[make_course_rule("36941")]), "36941", "l-1")
        unexpected_key = _dedupe_key(make_rules(rules=[make_course_rule("36941")]), "36941", "l-2")
        other_course_key = _dedupe_key(make_rules(rules=[make_course_rule("36941")]), "99999", "l-9")
        state = self.read_persisted_state()
        seen = json.dumps(state["seen"])
        self.assertIn(expected_key, seen)
        self.assertNotIn(unexpected_key, seen)
        self.assertNotIn(other_course_key, seen)

    def test_gate_skip_run_meets_four_structural_caps_and_exits_zero(self):
        courses = [{"course_id": "36941", "lectures": [{"sub_id": "l-1", "has_playback": True}]}]
        fixed = datetime(2026, 10, 6, 10, 10, tzinfo=GATE_TZ)
        with unittest.mock.patch.object(cloud_automation, "_gate_now", return_value=fixed):
            code, processed, connector = self.run_daily_with(
                courses, env={"COURSELENS_CLOUD_TRIGGER_KIND": "schedule"})
        self.assertEqual(code, 0)
        self.assertEqual(processed, [])
        # 四上限：≤1 个新 GET（恰 1 个月表请求）、零目录发现、零媒体、零 LLM。
        self.assertEqual(connector.schedule_calls, 1)
        self.assertEqual(connector.discover_calls, 0)
        self.assertEqual(connector.media_calls, 0)
        self.assertEqual(connector.slide_calls, 0)
        state = self.read_persisted_state()
        self.assertEqual(state["last_run"]["code"], "cloud_gate_skipped")
        self.assertEqual(state["last_run"]["counts"]["failed"], 0)
        self.assertEqual(state["last_run"]["counts"]["discovered"], 0)
        self.assertEqual(int(state["budget"]["deepseek_tokens"]), 0)

    def test_gate_pass_run_falls_through_to_full_pipeline(self):
        # 有课过门：窗内命中未知线索 → 过门 → 既有候选链原样处理（恰一次账照旧）。
        courses = [{"course_id": "36941", "lectures": [{"sub_id": "l-1", "has_playback": True}]}]
        connector = FakeConnector(courses, schedule_rows=[{
            "course_id": "36941", "sub_id": "l-1", "date": "2026-10-06",
            "end_precise": datetime(2026, 10, 6, 9, 40, tzinfo=GATE_TZ),
        }])
        fixed = datetime(2026, 10, 6, 10, 10, tzinfo=GATE_TZ)
        with unittest.mock.patch.object(cloud_automation, "_gate_now", return_value=fixed):
            code, processed, connector = self.run_daily_with(
                courses, env={"COURSELENS_CLOUD_TRIGGER_KIND": "schedule"}, connector=connector)
        self.assertEqual(code, 0)
        self.assertEqual(connector.schedule_calls, 1)
        self.assertEqual(connector.discover_calls, 1)
        self.assertEqual(len(processed), 1)
        state = self.read_persisted_state()
        self.assertEqual(state["last_run"]["code"], "cloud_daily_completed")

    def test_gate_fails_open_when_schedule_read_fails(self):
        # UIS 失败 fail-open：月表读取失败 → 过门 → 全管线（现状同款降级，绝不因读不到课表而漏）。
        courses = [{"course_id": "36941", "lectures": [{"sub_id": "l-1", "has_playback": True}]}]
        connector = FakeConnector(courses)

        def broken_schedule_rows():
            connector.schedule_calls += 1
            raise PlatformSessionError("platform_course_request_failed")

        connector.today_schedule_rows = broken_schedule_rows
        fixed = datetime(2026, 10, 6, 10, 10, tzinfo=GATE_TZ)
        with unittest.mock.patch.object(cloud_automation, "_gate_now", return_value=fixed):
            code, processed, connector = self.run_daily_with(
                courses, env={"COURSELENS_CLOUD_TRIGGER_KIND": "schedule"}, connector=connector)
        self.assertEqual(code, 0)
        self.assertEqual(connector.schedule_calls, 1)
        self.assertEqual(connector.discover_calls, 1)
        self.assertEqual(len(processed), 1)
        state = self.read_persisted_state()
        self.assertEqual(state["last_run"]["code"], "cloud_daily_completed")

    def test_acceptance_fake_time_gate_skips_empty_day_without_discovery(self):
        courses = [{"course_id": "36941", "lectures": [{"sub_id": "l-1", "has_playback": True}]}]
        code, processed, connector = self.run_daily_with(courses, env={
            "COURSELENS_CLOUD_TRIGGER_KIND": "acceptance",
            "COURSELENS_CLOUD_GATE_FAKE_TIME": "2026-10-06T10:10:00+08:00",
        })
        self.assertEqual(code, 0)
        self.assertEqual(processed, [])
        self.assertEqual(connector.schedule_calls, 1)
        self.assertEqual(connector.discover_calls, 0)

    def test_acceptance_fake_time_outside_day_window_keeps_full_pipeline(self):
        # fake=22:05（夜窗兜底时点）：门不激活 → 既有全管线原样。
        courses = [{"course_id": "36941", "lectures": [{"sub_id": "l-1", "has_playback": True}]}]
        code, processed, connector = self.run_daily_with(courses, env={
            "COURSELENS_CLOUD_TRIGGER_KIND": "acceptance",
            "COURSELENS_CLOUD_GATE_FAKE_TIME": "2026-10-06T22:05:00+08:00",
        })
        self.assertEqual(code, 0)
        self.assertEqual(connector.schedule_calls, 0)
        self.assertEqual(connector.discover_calls, 1)
        self.assertEqual(len(processed), 1)

    def test_acceptance_malformed_fake_time_fails_closed_before_any_school_call(self):
        os.environ["COURSELENS_CLOUD_STATE_KEY"] = self.state_key
        os.environ["COURSELENS_CLOUD_DEEPSEEK_API_KEY"] = "sk-synthetic"
        self.set_rules(make_rules(rules=[make_course_rule("36941")]))
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.dict(os.environ, {
                "COURSELENS_CLOUD_TRIGGER_KIND": "acceptance",
                "COURSELENS_CLOUD_GATE_FAKE_TIME": "not-a-time",
            }))
            login = stack.enter_context(unittest.mock.patch.object(
                cloud_automation, "cloud_session_from_environment"))
            code = cloud_automation.run_daily()
        login.assert_not_called()
        self.assertEqual(code, 1)
        state = self.read_persisted_state()
        self.assertEqual(state["last_run"]["code"], "cloud_gate_time_invalid")
        self.assertEqual(state["last_run"]["counts"]["failed"], 1)

    def test_selection_baseline_excludes_playable_and_releases_later_lectures(self):
        """基线语义：选择时刻已可播放的讲次被排除；未列出/新讲次可播即合格。"""
        def run_with(courses, baseline):
            os.environ["COURSELENS_CLOUD_STATE_KEY"] = self.state_key
            os.environ["COURSELENS_CLOUD_DEEPSEEK_API_KEY"] = "sk-synthetic"
            self.set_rules(make_rules(rules=[make_course_rule("36941", baseline=baseline)]))
            connector = FakeConnector(courses)
            processed = []
            def process(job, *, checkpoint_writer=None):
                processed.append(job)
                return {"outputs": {}, "metrics": {}}
            with contextlib.ExitStack() as stack:
                stack.enter_context(unittest.mock.patch.object(
                    cloud_automation, "cloud_session_from_environment", return_value=connector,
                ))
                stack.enter_context(unittest.mock.patch.object(cloud_automation, "process_job", process))
                stack.enter_context(unittest.mock.patch.object(
                    cloud_automation, "usage_snapshot", return_value={"total_tokens": 0},
                ))
                code = cloud_automation.run_daily()
            state_path = Path(".work/cloud-state/state.box.json")
            if state_path.exists():
                self.github_state.upload_state(state_path.read_bytes())
            return code, processed

        courses = [{"course_id": "36941", "lectures": [
            {"sub_id": "l-old", "has_playback": True},
            {"sub_id": "l-new", "has_playback": True},
            {"sub_id": "l-future", "has_playback": False},
        ]}]
        code, processed = run_with(courses, baseline=["l-old"])
        self.assertEqual(code, 0)
        # l-old excluded by the baseline; l-new processed; l-future unplayable.
        self.assertEqual(len(processed), 1)

        # A listed lecture that could not play becomes eligible when it can:
        # the next window discovers l-future now playable, and it is not in
        # the baseline, so it runs.
        courses_later = [{"course_id": "36941", "lectures": [
            {"sub_id": "l-future", "has_playback": True},
        ]}]
        code, processed = run_with(courses_later, baseline=["l-old"])
        self.assertEqual(code, 0)
        self.assertEqual(len(processed), 1)

        # A fresh generation that lists everything playable excludes all of it.
        code, processed = run_with(courses_later, baseline=["l-future"])
        self.assertEqual(code, 0)
        self.assertEqual(len(processed), 0)

    def test_fixed_bundle_outputs_and_checkpoint_resume(self):
        courses = [{"course_id": "36941", "lectures": [{"sub_id": "l-1", "has_playback": True}]}]
        checkpoint_seen = {}

        def first_attempt(job, *, checkpoint_writer=None):
            self.assertEqual(job["requested_outputs"], list(OUTPUT_BUNDLE))
            if checkpoint_writer is not None:
                checkpoint_writer({
                    "stage": "ocr", "completed_chunks": 2, "total_chunks": 4,
                    "ocr_completed_items": 2, "ppt_pages": [{"page_num": 1}],
                })
            checkpoint_seen["saved"] = True
            raise RuntimeError("synthetic timeout after checkpoint")

        code, processed, _ = self.run_daily_with(courses, process=first_attempt)
        self.assertEqual(code, 1)
        self.assertEqual(len(processed), 1)
        state = self.read_persisted_state()
        saved = dict((state.get("checkpoints") or {}).get(
            _dedupe_key(make_rules(rules=[make_course_rule("36941")]), "36941", "l-1")
        ) or {})
        self.assertTrue(saved, "检查点必须保存在加密状态里")
        self.assertEqual(saved.get("ocr_completed_items"), 2)

        def resume_attempt(job, *, checkpoint_writer=None):
            prior = dict((job.get("payload") or {}).get("checkpoint") or {})
            checkpoint_seen["resumed"] = prior
            self.assertEqual(prior.get("ocr_completed_items"), 2,
                             "重试必须携带已保存的阶段检查点")
            # Media URLs are reacquired fresh for every attempt.
            self.assertEqual(job["payload"]["media"], {"kind": "synthetic"})
            return {"outputs": {"subtitle": {"mode": "automatic"}}, "metrics": {}}

        code, processed, _ = self.run_daily_with(courses, process=resume_attempt)
        self.assertEqual(code, 0)
        self.assertIn("resumed", checkpoint_seen)
        state = self.read_persisted_state()
        self.assertEqual(state.get("checkpoints"), {},
                         "成功后活动检查点必须清空")

    def test_session_expiry_exhaustion_keeps_chunk_progress_and_resumes_at_16(self):
        """第四十一案：分块中期会话作废（重试耗尽）后，16/27 必须保留并可续跑。

        实测链路＝客户端重启→重登录→学校会话作废→runner 在 16/27 块处报
        platform_course_request_failed、整任务 exit 1。此钉验证失败不留残影：
        检查点（含已识别段与指纹链）仍加密留在状态里、该讲不进 seen/pending
        可重跑，下一次尝试把 completed_chunks=16 交回任务，而不是从 0 重来。
        """
        courses = [{"course_id": "36941", "lectures": [{"sub_id": "l-1", "has_playback": True}]}]
        seen = {}
        partial = {
            "stage": "asr",
            "mode": "automatic",
            "completed_chunks": 16,
            "total_chunks": 27,
            "raw_sensevoice": [{"start_ms": 0, "end_ms": 600000, "text": "已识别片段"}],
            "raw_paraformer": [{"start_ms": 0, "end_ms": 600000, "text": "已识别片段"}],
            "pcm_fingerprint": "a" * 64,
        }
        item_key = _dedupe_key(make_rules(rules=[make_course_rule("36941")]), "36941", "l-1")

        def expired_attempt(job, *, checkpoint_writer=None):
            checkpoint_writer(dict(partial))
            raise PlatformSessionError("platform_course_request_failed")

        code, processed, _ = self.run_daily_with(courses, process=expired_attempt)
        self.assertEqual(code, 1, "会话作废仍是失败运行（重试窗口内自愈才不失败）")
        self.assertEqual(len(processed), 1)
        state = self.read_persisted_state()
        saved = dict((state.get("checkpoints") or {}).get(item_key) or {})
        self.assertEqual(saved.get("completed_chunks"), 16, "已完成的 16 块必须留在检查点里")
        self.assertEqual(saved.get("total_chunks"), 27)
        self.assertEqual(saved.get("raw_sensevoice"), partial["raw_sensevoice"])
        self.assertEqual(saved.get("pcm_fingerprint"), "a" * 64, "指纹链随检查点保留，续跑身份可复现")
        self.assertEqual(dict(state.get("seen") or {}).get("36941") or [], [],
                         "失败项绝不能被记成已完成")
        self.assertEqual(state.get("pending") or [], [], "失败项的 pending 必须清掉才可重跑")

        def resume_attempt(job, *, checkpoint_writer=None):
            prior = dict((job.get("payload") or {}).get("checkpoint") or {})
            seen["prior"] = prior
            return {"outputs": {"subtitle": {"mode": "automatic"}}, "metrics": {}}

        code, processed, _ = self.run_daily_with(courses, process=resume_attempt)
        self.assertEqual(code, 0)
        self.assertEqual(seen["prior"].get("completed_chunks"), 16, "续跑从最后完成块开始，不归零")
        self.assertEqual(seen["prior"].get("total_chunks"), 27)
        self.assertEqual(seen["prior"].get("pcm_fingerprint"), "a" * 64)
        self.assertEqual(self.read_persisted_state().get("checkpoints"), {},
                         "成功后活动检查点必须清空")

    def test_manual_dispatch_rejected_while_paused_before_login(self):
        def rejected():
            raise AssertionError("login attempted despite paused manual dispatch")

        code, processed, _ = self.run_daily_with(
            [{"course_id": "36941", "lectures": [{"sub_id": "l-1", "has_playback": True}]}],
            env={
                "COURSELENS_CLOUD_MANUAL_DISPATCH": "true",
                "COURSELENS_CLOUD_DISPATCH_CONFIG_HASH": "d" * 64,
                "COURSELENS_CLOUD_ENABLED_FLAG": "false",
            },
        )
        self.assertEqual(code, 1)
        state = self.read_persisted_state()
        self.assertEqual(state["last_run"]["code"], "cloud_dispatch_paused")

    def test_protocol_or_config_mismatch_rejected_before_login(self):
        for env, expected_code in (
            ({"COURSELENS_CLOUD_EXPECTED_PROTOCOL_VERSION": "cloud-automation.v2"},
             "cloud_protocol_mismatch"),
            ({"COURSELENS_CLOUD_EXPECTED_CONFIG_HASH": "e" * 64},
             "cloud_config_mismatch"),
            ({"COURSELENS_CLOUD_EXPECTED_CONFIG_HASH": ""},
             "cloud_config_mismatch"),
        ):
            with self.subTest(code=expected_code):
                code, _processed, _ = self.run_daily_with(
                    [{"course_id": "36941", "lectures": [{"sub_id": "l-1", "has_playback": True}]}],
                    env=env,
                )
                self.assertEqual(code, 1)
                state = self.read_persisted_state()
                self.assertEqual(state["last_run"]["code"], expected_code)

    def test_manual_dispatch_requires_matching_config_hash(self):
        code, _processed, _ = self.run_daily_with(
            [{"course_id": "36941", "lectures": [{"sub_id": "l-1", "has_playback": True}]}],
            env={
                "COURSELENS_CLOUD_MANUAL_DISPATCH": "true",
                "COURSELENS_CLOUD_DISPATCH_CONFIG_HASH": "0" * 64,
                "COURSELENS_CLOUD_ENABLED_FLAG": "true",
            },
        )
        self.assertEqual(code, 1)
        state = self.read_persisted_state()
        self.assertEqual(state["last_run"]["code"], "cloud_config_mismatch")

    def test_completed_lectures_never_run_twice(self):
        courses = [{"course_id": "36941", "lectures": [{"sub_id": "l-1", "has_playback": True}]}]
        code, processed, _ = self.run_daily_with(courses)
        self.assertEqual(len(processed), 1)
        code, processed, _ = self.run_daily_with(courses)
        self.assertEqual(code, 0)
        self.assertEqual(processed, [])
        state = self.read_persisted_state()
        self.assertEqual(state["last_run"]["counts"]["skipped_pending"], 1)

    def test_already_pending_item_is_skipped_until_ttl_expires(self):
        item_key = _dedupe_key(
            make_rules(rules=[make_course_rule("36941")]), "36941", "l-1",
        )
        pending_state = _empty_state()
        pending_state["pending"] = [{
            "key": item_key, "course_id": "36941", "sub_id": "l-1",
            "expires_at": time.time() + PENDING_TTL_SECONDS,
        }]
        courses = [{"course_id": "36941", "lectures": [{"sub_id": "l-1", "has_playback": True}]}]
        code, processed, _ = self.run_daily_with(courses, prefill_state=pending_state)
        self.assertEqual(code, 0)
        self.assertEqual(processed, [])
        state = self.read_persisted_state()
        self.assertEqual(state["last_run"]["counts"]["skipped_pending"], 1)

        expired_state = _empty_state()
        expired_state["pending"] = [{
            "key": item_key, "course_id": "36941", "sub_id": "l-1", "expires_at": 1.0,
        }]
        code, processed, _ = self.run_daily_with(courses, prefill_state=expired_state)
        self.assertEqual(len(processed), 1)

    def test_budget_exhaustion_defers_without_marking_completed(self):
        os.environ["COURSELENS_CLOUD_RULES_JSON"] = json.dumps(make_rules(
            budget={"max_lectures": 1, "max_runner_minutes": 300, "max_deepseek_tokens": 100000},
            rules=[make_course_rule("36941")],
        ))
        connector = FakeConnector([{"course_id": "36941", "lectures": [
            {"sub_id": "l-1", "has_playback": True},
            {"sub_id": "l-2", "has_playback": True},
        ]}])
        processed = []

        def process(job, *, checkpoint_writer=None):
            processed.append(job)
            return {"outputs": {"subtitle": {"mode": "automatic"}}, "metrics": {}}

        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.object(
                cloud_automation, "cloud_session_from_environment", return_value=connector,
            ))
            stack.enter_context(unittest.mock.patch.object(cloud_automation, "process_job", process))
            stack.enter_context(unittest.mock.patch.object(
                cloud_automation, "usage_snapshot", return_value={"total_tokens": 0},
            ))
            code = cloud_automation.run_daily()
        self.assertEqual(code, 0)
        self.assertEqual(len(processed), 1)
        state = self.read_persisted_state()
        self.assertEqual(state["last_run"]["code"], "budget_exhausted")
        self.assertEqual(state["last_run"]["counts"]["deferred"], 1)
        self.assertEqual(state["last_run"]["counts"]["processed"], 1)
        # Deferred item is not in seen: it stays retryable.
        deferred_key = _dedupe_key(make_rules(
            budget={"max_lectures": 1, "max_runner_minutes": 300, "max_deepseek_tokens": 100000},
            rules=[make_course_rule("36941")],
        ), "36941", "l-2")
        self.assertNotIn(deferred_key, json.dumps(state["seen"]))
        self.assertEqual(state["circuits"]["budget"]["state"], "open")

    def test_known_credential_rejection_opens_auth_circuit_immediately(self):
        self.set_rules(make_rules())
        os.environ["COURSELENS_CLOUD_STATE_KEY"] = self.state_key

        def rejected():
            raise PlatformSessionError("platform_ticket_rejected")

        with unittest.mock.patch.object(
            cloud_automation, "cloud_session_from_environment", side_effect=rejected,
        ):
            cloud_automation.run_daily()
        # Simulate the workflow's always() upload of the incremental state.
        self.github_state.upload_state(
            Path(".work/cloud-state/state.box.json").read_bytes()
        )
        state = self.read_persisted_state()
        self.assertEqual(state["circuits"]["authentication"]["state"], "open")
        # A second run must not even attempt a login while the circuit is open.
        os.environ["COURSELENS_CLOUD_STATE_KEY"] = self.state_key
        self.set_rules(make_rules())
        with unittest.mock.patch.object(
            cloud_automation, "cloud_session_from_environment",
            side_effect=AssertionError("login attempted despite open auth circuit"),
        ):
            code = cloud_automation.run_daily()
        self.assertEqual(code, 1)
        state = self.read_persisted_state()
        self.assertEqual(state["last_run"]["code"], "authentication_circuit_open")

    def test_transient_processing_failure_keeps_item_retryable_and_opens_only_ai_circuit(self):
        from courselens_worker.llm import LLMError
        courses = [{"course_id": "36941", "lectures": [{"sub_id": "l-1", "has_playback": True}]}]

        def transient_fail(job, *, checkpoint_writer=None):
            raise LLMError("DeepSeek HTTP 503 unavailable")

        code, processed, _ = self.run_daily_with(courses, process=transient_fail)
        # Transient AI failures degrade honestly: the run reports completion
        # with a degraded circuit and the item stays retryable.
        self.assertEqual(code, 0)
        self.assertEqual(len(processed), 1)
        state = self.read_persisted_state()
        self.assertNotIn(_dedupe_key(
            make_rules(rules=[make_course_rule("36941")]), "36941", "l-1",
        ), json.dumps(state["seen"]))
        self.assertEqual(state["circuits"]["deepseek"]["state"], "degraded")
        self.assertEqual(state["circuits"]["authentication"]["state"], "closed")
        # The next run retries the same lecture (failure is not completion).
        code, processed, _ = self.run_daily_with(courses)
        self.assertEqual(code, 0)
        self.assertEqual(len(processed), 1)

    def test_cloud_state_and_results_never_touch_transient_paths_after_run(self):
        courses = [{"course_id": "36941", "lectures": [{"sub_id": "l-1", "has_playback": True}]}]
        self.run_daily_with(courses)
        state = self.read_persisted_state()
        raw = (Path(".work/cloud-state/state.box.json")).read_bytes()
        self.assertNotIn(b"2020001", raw)
        self.assertNotIn(b"l-1", raw)
        self.assertEqual(state["schema"], STATE_SCHEMA)

    def test_success_result_envelope_carries_verified_courseware_plan(self):
        slides = [
            {"page_num": 1, "created_sec": 30, "record_id": "101",
             "source": {"url": "https://slides.invalid/1.img"}},
            {"page_num": 2, "created_sec": 60, "record_id": "102",
             "source": {"url": "https://slides.invalid/2.img"}},
            {"page_num": 3, "created_sec": 90, "record_id": "103",
             "source": {"url": "https://slides.invalid/3.img"}},
        ]
        courses = [{"course_id": "36941", "lectures": [{"sub_id": "l-1", "has_playback": True}]}]

        def process(job, *, checkpoint_writer=None):
            return {
                "outputs": {"ppt_pages": [
                    {"page_num": 1, "created_sec": 30, "source_sha256": "a" * 64,
                     "dhash": "0" * 16, "text": "第1页 标题"},
                    {"page_num": 2, "created_sec": 60, "source_sha256": "b" * 64,
                     "dhash": "1" * 16, "text": "第2页 内容"},
                    {"page_num": 3, "created_sec": 90, "source_sha256": "b" * 64,
                     "dhash": "1" * 16, "text": "第2页 内容"},
                ]},
                "metrics": {"elapsed_seconds": 1.0, "slides_skipped": {"empty": 1}},
            }

        code, _processed, _connector = self.run_daily_with(
            courses, process=process, slides=slides,
        )
        self.assertEqual(code, 0)
        results_dir = Path(".work/cloud-results")
        envelopes = sorted(results_dir.glob("*.box.json"))
        self.assertEqual(len(envelopes), 1)
        envelope = json.loads(envelopes[0].read_text(encoding="utf-8"))
        from courselens_worker.protocol import open_result
        result = open_result(
            envelope, self.result_private, self.signing_public,
            expected_task_id=str(envelope.get("task_id")),
            expected_input_hash=str(envelope.get("input_hash")),
        )
        outputs = result["outputs"]
        plan = outputs.get("courseware_plan")
        self.assertIsInstance(plan, dict)
        self.assertEqual(plan["schema"], "courseware_plan.v1")
        self.assertEqual(plan["course_id"], "36941")
        self.assertEqual(plan["sub_id"], "l-1")
        # Only exact duplicates collapse; counts stay auditable.
        self.assertEqual(plan["counts"]["kept"], 2)
        self.assertEqual(plan["counts"]["exact_duplicates"], 1)
        self.assertEqual(plan["counts"]["skipped"], 1)
        # The plan carries no URL, cookie, OCR body, thumbnail, or image bytes.
        rendered = json.dumps(plan, ensure_ascii=False)
        self.assertNotIn("slides.invalid", rendered)
        self.assertNotIn("pptimgurl", rendered)
        self.assertNotIn("http", rendered)
        self.assertNotIn("标题", rendered)
        self.assertNotIn("dhash", rendered)
        # The recorded digest matches the plan exactly.
        from courselens_worker.protocol import canonical_json, sha256_hex
        self.assertEqual(
            outputs["courseware_plan_digest"],
            sha256_hex(canonical_json(plan)),
        )
        # Record ids survived from the slide inventory (time-guarded).
        record_ids = {entry["record_id"] for entry in plan["entries"]}
        self.assertEqual(record_ids, {"101", "102"})


class VerifyAiProbeTests(unittest.TestCase):
    """CLOUDVERIFY-FIX（2026-10-05）探针参数钉（DIAG 提案 1 验收①）。

    背景：deepseek-flash（09-21 e6cb675 起）为思考档模型，_chat 缺省不带
    thinking 字段=提供商默认 enabled/high，思考 token 计入 max_tokens 预算
    → 原 max_tokens=8 探针被思考耗尽、content 空 → deepseek_verification_
    failed（verify workflow run 37288805260 根因）。钉面经完整 _chat 传输链：
    探针载荷必须显式 thinking={"type": "disabled"} + max_tokens=32；思考档
    usage 记账形态（reasoning_tokens>0）+content="OK" 通过；截断空 content
    按既有闭集码诚实失败。
    """

    def setUp(self) -> None:
        from courselens_worker import llm as llm_mod

        self._llm = llm_mod
        llm_mod.reset_usage()

    def tearDown(self) -> None:
        self._llm.reset_usage()

    def _probe_payload(self, content, usage=None):
        response = unittest.mock.Mock()
        response.status_code = 200
        response.headers = {}
        response.json.return_value = {
            "choices": [{"message": {"content": content}}],
            "usage": usage if usage is not None else {
                "prompt_tokens": 10, "completion_tokens": 32, "total_tokens": 42,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        }
        session = unittest.mock.Mock()
        session.post.return_value = response
        with unittest.mock.patch.object(self._llm, "_http_session", return_value=session):
            cloud_automation._verify_ai("sk-synthetic")
        return session.post.call_args.kwargs["json"]

    def test_probe_pins_thinking_disabled_and_32_token_budget(self):
        payload = self._probe_payload("OK")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["max_tokens"], 32)

    def test_probe_passes_with_thinking_tier_usage_shape(self):
        # 模拟思考档 200 响应记账形态（reasoning_tokens>0）：探针判定只看
        # content，usage 不再能把 8 token 预算挤爆（载荷已关思考+32 预算）。
        usage = {
            "prompt_tokens": 10, "completion_tokens": 32, "total_tokens": 42,
            "completion_tokens_details": {"reasoning_tokens": 29},
        }
        self._probe_payload("OK", usage=usage)  # must not raise

    def test_probe_truncated_empty_content_fails_closed(self):
        with self.assertRaises(cloud_automation.CloudAutomationError) as caught:
            self._probe_payload("")
        self.assertEqual(caught.exception.code, "deepseek_verification_failed")


if __name__ == "__main__":
    unittest.main()


class PlatformCodeCollapseTests(unittest.TestCase):
    """U11 挑战塌缩扩集：已登记平台码原样上抛，未知串仍塌缩。"""

    def test_precise_codes_survive_and_unknown_collapses(self):
        from courselens_worker.cloud_automation import (
            _PLATFORM_PRECISE_CODES,
            collapse_platform_code,
        )

        self.assertIn("platform_challenge_required", _PLATFORM_PRECISE_CODES)
        self.assertGreaterEqual(len(_PLATFORM_PRECISE_CODES), 18)
        self.assertEqual(collapse_platform_code("platform_challenge_required"), "platform_challenge_required")
        self.assertEqual(collapse_platform_code("platform_media_missing"), "platform_media_missing")
        self.assertEqual(collapse_platform_code("platform_something_new"), "platform_session_failed")
