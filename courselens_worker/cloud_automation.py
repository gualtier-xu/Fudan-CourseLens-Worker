"""Unattended, bounded cloud discovery and learning-pack processing.

All platform access is delegated to ``platform_session``.  This module keeps
only encrypted incremental state and encrypted result envelopes in Actions
artifacts.  Public logs contain closed-set codes, counts, stages and timing.
"""

from __future__ import annotations

import base64
import io
import json
import os
import secrets
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
from nacl.secret import SecretBox

from .llm import LLMError, _chat, reset_usage, usage_snapshot
from .platform_session import PlatformSessionError, cloud_session_from_environment
from .protocol import PROTOCOL_VERSION, RESULT_SCHEMA, canonical_json, seal_result, sha256_hex
from .runner import process_job, safe_worker_error_detail


STATE_SCHEMA = "cloud.state.v1"
RULES_SCHEMA = "cloud-automation.v1"
RESULT_PREFIX = "courselens-cloud-result-"
STATE_PREFIX = "courselens-cloud-state-"


class CloudAutomationError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _required(name: str, *, pop: bool = False) -> str:
    value = (os.environ.pop(name, "") if pop else os.environ.get(name, "")).strip()
    if not value:
        raise CloudAutomationError("cloud_configuration_missing")
    return value


def _github_request(method: str, path: str, **kwargs) -> requests.Response:
    token = _required("GITHUB_TOKEN")
    repository = _required("GITHUB_REPOSITORY")
    expected = tuple(kwargs.pop("expected", (200,)))
    response = requests.request(
        method,
        f"https://api.github.com/repos/{repository}{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
            "User-Agent": "Fudan-CourseLens-Cloud/1",
        },
        timeout=45,
        **kwargs,
    )
    if response.status_code not in expected:
        raise CloudAutomationError("github_state_unavailable")
    return response


def _state_key() -> bytes:
    try:
        value = base64.b64decode(_required("COURSELENS_CLOUD_STATE_KEY", pop=True), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise CloudAutomationError("cloud_state_key_invalid") from exc
    if len(value) != SecretBox.KEY_SIZE:
        raise CloudAutomationError("cloud_state_key_invalid")
    return value


def _empty_state() -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "revision": 0,
        "seen": {},
        "budget": {"date": "", "lectures": 0, "runner_minutes": 0.0, "deepseek_tokens": 0},
        "circuits": {
            "authentication": {"state": "closed", "failures": 0},
            "deepseek": {"state": "closed", "failures": [], "retry_after": 0},
            "platform": {"state": "closed", "failures": 0},
            "budget": {"state": "closed", "failures": 0},
        },
        "pending": [],
        "updated_at": 0,
    }


def _open_state(raw: bytes, key: bytes) -> dict[str, Any]:
    try:
        envelope = json.loads(raw.decode("utf-8"))
        ciphertext = base64.b64decode(str(envelope["ciphertext"]).encode("ascii"), validate=True)
        plaintext = SecretBox(key).decrypt(ciphertext)
        value = json.loads(plaintext.decode("utf-8"))
    except (ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CloudAutomationError("cloud_state_rejected") from exc
    if not isinstance(value, dict) or value.get("schema") != STATE_SCHEMA:
        raise CloudAutomationError("cloud_state_rejected")
    return value


def _seal_state(value: dict[str, Any], key: bytes) -> bytes:
    ciphertext = bytes(SecretBox(key).encrypt(canonical_json(value)))
    return canonical_json({
        "schema": "cloud.state.box.v1",
        "encoding": "secretbox+base64",
        "sha256": sha256_hex(ciphertext),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    })


def _load_previous_state(key: bytes) -> dict[str, Any]:
    try:
        payload = _github_request(
            "GET", "/actions/artifacts", params={"per_page": 100}, expected=(200,)
        ).json()
        candidates = [
            item for item in list(payload.get("artifacts") or [])
            if str(item.get("name") or "").startswith(STATE_PREFIX) and not item.get("expired")
        ]
        if not candidates:
            return _empty_state()
        latest = max(candidates, key=lambda item: str(item.get("created_at") or ""))
        response = _github_request(
            "GET", f"/actions/artifacts/{int(latest['id'])}/zip", expected=(200,)
        )
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            raw = archive.read("state.box.json")
        return _open_state(raw, key)
    except CloudAutomationError:
        raise
    except (KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise CloudAutomationError("cloud_state_rejected") from exc


def _prune_previous_states() -> None:
    try:
        payload = _github_request(
            "GET", "/actions/artifacts", params={"per_page": 100}, expected=(200,)
        ).json()
        candidates = sorted(
            [
                item for item in list(payload.get("artifacts") or [])
                if str(item.get("name") or "").startswith(STATE_PREFIX) and not item.get("expired")
            ],
            key=lambda item: str(item.get("created_at") or ""),
            reverse=True,
        )
        for item in candidates[1:]:
            _github_request(
                "DELETE", f"/actions/artifacts/{int(item['id'])}", expected=(204, 404)
            )
    except (CloudAutomationError, KeyError, ValueError):
        return


def _rules() -> dict[str, Any]:
    raw = _required("COURSELENS_CLOUD_RULES_JSON", pop=True)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CloudAutomationError("cloud_rules_invalid") from exc
    raw = ""
    if not isinstance(value, dict) or value.get("schema") != RULES_SCHEMA:
        raise CloudAutomationError("cloud_rules_invalid")
    return value


def _today() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")


def _reset_daily_budget(state: dict[str, Any]) -> dict[str, Any]:
    budget = dict(state.get("budget") or {})
    if budget.get("date") != _today():
        budget = {"date": _today(), "lectures": 0, "runner_minutes": 0.0, "deepseek_tokens": 0}
        state["budget"] = budget
        dict(state.get("circuits") or {}).setdefault("budget", {})["state"] = "closed"
    return budget


def _verify_ai(api_key: str) -> None:
    if not api_key:
        return
    value = _chat(api_key, [
        {"role": "system", "content": "Reply with OK only."},
        {"role": "user", "content": "Connection check"},
    ], max_tokens=8)
    if value.strip().upper() != "OK":
        raise CloudAutomationError("deepseek_verification_failed")


def _rules_need_ai(rules: dict[str, Any]) -> bool:
    return any(
        not bool(item.get("discovery_only", True)) and (
            str(item.get("subtitle_mode") or "fast") == "standard"
            or bool(item.get("summary"))
            or bool(item.get("chapters"))
        )
        for item in list(rules.get("rules") or [])
        if isinstance(item, dict)
    )


def _expiring_result_count() -> int:
    try:
        payload = _github_request(
            "GET", "/actions/artifacts", params={"per_page": 100}, expected=(200,)
        ).json()
        now = time.time()
        threshold = now + 3 * 86400
        count = 0
        for item in list(payload.get("artifacts") or []):
            if not str(item.get("name") or "").startswith(RESULT_PREFIX) or item.get("expired"):
                continue
            expires_at = str(item.get("expires_at") or "")
            if not expires_at:
                continue
            try:
                expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
            if now < expiry <= threshold:
                count += 1
        return count
    except (CloudAutomationError, ValueError, TypeError):
        return 0


def verify() -> int:
    started = time.monotonic()
    rules = _rules()
    api_key = os.environ.pop("COURSELENS_CLOUD_DEEPSEEK_API_KEY", "").strip()
    connector = None
    try:
        connector = cloud_session_from_environment()
        if _rules_need_ai(rules) and not api_key:
            raise CloudAutomationError("deepseek_key_missing")
        _verify_ai(api_key)
        if os.environ.get("COURSELENS_CLOUD_RESET_CIRCUIT") == "true":
            key = _state_key()
            state = _load_previous_state(key)
            for name in ("authentication", "deepseek", "platform", "budget"):
                state.setdefault("circuits", {})[name] = {
                    "state": "closed", "failures": [] if name == "deepseek" else 0,
                    "retry_after": 0, "last_error_code": "",
                }
            state["revision"] = int(state.get("revision") or 0) + 1
            state["updated_at"] = time.time()
            _prune_previous_states()
            root = Path(".work") / "cloud-state"
            root.mkdir(parents=True, exist_ok=True)
            (root / "state.box.json").write_bytes(_seal_state(state, key))
        print(f"stage=verified elapsed={round(time.monotonic() - started, 2)}", flush=True)
        return 0
    finally:
        api_key = ""
        if connector is not None:
            connector.session.close()


def _result_envelope(
    *, course: dict[str, Any], lecture: dict[str, Any], outputs: dict[str, Any],
    metrics: dict[str, Any], config_hash: str,
) -> dict[str, Any]:
    task_id = secrets.token_hex(16)
    input_hash = sha256_hex(canonical_json({
        "course_id": str(course.get("course_id") or ""),
        "sub_id": str(lecture.get("sub_id") or ""),
        "config_hash": config_hash,
    }))
    result = {
        "schema": RESULT_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "task_id": task_id,
        "job_kind": "learning_pack",
        "input_hash": input_hash,
        "pipeline_fingerprint": RULES_SCHEMA,
        "status": "completed",
        "outputs": {
            **outputs,
            "cloud_catalog": {
                "course_id": str(course.get("course_id") or ""),
                "title": str(course.get("title") or ""),
                "teacher": str(course.get("teacher") or ""),
                "term": str(course.get("term") or ""),
                "department": str(course.get("department") or ""),
                "lecture": dict(lecture),
            },
        },
        "metrics": dict(metrics or {}),
        "warnings": [],
    }
    envelope = seal_result(
        result,
        _required("COURSELENS_CLOUD_RESULT_PUBLIC_KEY"),
        _required("WORKER_SIGNING_PRIVATE_KEY"),
    )
    envelope["input_hash"] = input_hash
    return envelope


def _write_result(envelope: dict[str, Any]) -> None:
    root = Path(".work") / "cloud-results"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{secrets.token_hex(16)}.box.json"
    destination.write_text(
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def run_daily() -> int:
    started = time.monotonic()
    key = _state_key()
    state = _load_previous_state(key)
    _prune_previous_states()
    rules = _rules()
    api_key = os.environ.pop("COURSELENS_CLOUD_DEEPSEEK_API_KEY", "").strip()
    limits = dict(rules.get("budget") or {})
    budget = _reset_daily_budget(state)
    circuits = dict(state.get("circuits") or {})
    counts = {
        "discovered": 0, "processed": 0, "failed": 0, "deferred": 0,
        "expiring_results": _expiring_result_count(),
    }
    code = "cloud_daily_completed"
    connector = None
    try:
        auth = dict(circuits.get("authentication") or {})
        if auth.get("state") == "open":
            raise CloudAutomationError("authentication_circuit_open")
        connector = cloud_session_from_environment()
        if connector is None:
            raise CloudAutomationError("platform_connection_failed")
        auth.update({"state": "closed", "failures": 0, "last_error_code": ""})
        circuits["authentication"] = auth
        catalog = connector.discover_authorized_courses()
        circuits["platform"] = {
            "state": "closed", "failures": 0, "last_error_code": "",
        }
        rule_map = {str(item.get("course_id") or ""): dict(item) for item in list(rules.get("rules") or [])}
        seen = {str(key): list(value or []) for key, value in dict(state.get("seen") or {}).items()}
        candidates: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for course in catalog:
            course_id = str(course.get("course_id") or "")
            known = set(str(value) for value in seen.get(course_id, []))
            rule = rule_map.get(course_id, {"course_id": course_id, "discovery_only": True, "only_new": True, "priority": 0})
            for lecture in list(course.get("lectures") or []):
                sub_id = str(lecture.get("sub_id") or "")
                if not sub_id or sub_id in known or not lecture.get("has_playback"):
                    continue
                counts["discovered"] += 1
                candidates.append((int(rule.get("priority") or 0), course, lecture, rule))
        candidates.sort(key=lambda item: (-item[0], str(item[2].get("date") or ""), str(item[2].get("sub_id") or "")))
        maximum = max(1, int(limits.get("max_lectures") or 2))
        runner_stop = max(0.0, float(limits.get("max_runner_minutes") or 300) - 30.0)
        token_limit = max(0, int(limits.get("max_deepseek_tokens") or 100000))
        for _priority, course, lecture, rule in candidates:
            course_id = str(course.get("course_id") or "")
            sub_id = str(lecture.get("sub_id") or "")
            current_runner_minutes = float(budget.get("runner_minutes") or 0) + (time.monotonic() - started) / 60.0
            if budget["lectures"] >= maximum or current_runner_minutes >= runner_stop or budget["deepseek_tokens"] >= token_limit:
                counts["deferred"] += 1
                circuits["budget"] = {"state": "open", "failures": 0, "last_error_code": "budget_exhausted"}
                code = "budget_exhausted"
                continue
            if rule.get("discovery_only"):
                _write_result(_result_envelope(
                    course=course, lecture=lecture, outputs={}, metrics={"discovery_only": True},
                    config_hash=str(rules.get("config_hash") or ""),
                ))
                budget["lectures"] += 1
                seen.setdefault(course_id, []).append(sub_id)
                continue
            deepseek_circuit = dict(circuits.get("deepseek") or {})
            needs_ai = (
                str(rule.get("subtitle_mode") or "fast") == "standard"
                or bool(rule.get("summary")) or bool(rule.get("chapters"))
            )
            if (
                needs_ai and deepseek_circuit.get("state") == "open"
                and float(deepseek_circuit.get("retry_after") or 0) > time.time()
            ):
                counts["deferred"] += 1
                code = "deepseek_circuit_open"
                continue
            duration = int(lecture.get("duration_seconds") or 0)
            if duration and duration > int(rule.get("max_lecture_minutes") or 240) * 60:
                counts["deferred"] += 1
                continue
            requested = ["subtitle"]
            if rule.get("ocr"):
                requested.append("ocr")
            if rule.get("summary"):
                requested.append("summary")
            if rule.get("chapters"):
                requested.append("chapters")
            reset_usage()
            job = {
                "schema": "job.v2", "protocol_version": PROTOCOL_VERSION,
                "task_id": secrets.token_hex(16), "job_kind": "learning_pack",
                "input_hash": "", "result_public_key": _required("COURSELENS_CLOUD_RESULT_PUBLIC_KEY"),
                "payload": {
                    "mode": str(rule.get("subtitle_mode") or "fast"),
                    "title": str(course.get("title") or ""),
                    "media": connector.media_source(course_id, sub_id),
                    "slides": connector.slide_sources(course_id, sub_id) if set(requested).intersection({"ocr", "summary", "chapters"}) else [],
                },
                "requested_outputs": requested,
                "secrets": {"deepseek_api_key": api_key},
                "pipeline": {"version": RULES_SCHEMA},
            }
            job["input_hash"] = sha256_hex(canonical_json({key: value for key, value in job.items() if key != "input_hash"}))
            try:
                result = process_job(job)
                usage = usage_snapshot()
                metrics = dict(result.get("metrics") or {})
                metrics["deepseek_tokens"] = int(usage.get("total_tokens") or 0)
                _write_result(_result_envelope(
                    course=course, lecture=lecture, outputs=dict(result.get("outputs") or {}),
                    metrics=metrics, config_hash=str(rules.get("config_hash") or ""),
                ))
                budget["lectures"] += 1
                budget["deepseek_tokens"] += int(usage.get("total_tokens") or 0)
                counts["processed"] += 1
                seen.setdefault(course_id, []).append(sub_id)
                circuits["deepseek"] = {"state": "closed", "failures": [], "retry_after": 0, "last_error_code": ""}
            except Exception as exc:
                counts["failed"] += 1
                reason = safe_worker_error_detail(exc)
                if isinstance(exc, LLMError):
                    now = time.time()
                    deepseek = dict(circuits.get("deepseek") or {})
                    message = str(exc)
                    if "HTTP 401" in message or "HTTP 403" in message or "does not contain an AI API key" in message:
                        deepseek.update({
                            "failures": [now], "last_error_code": "deepseek_auth_failed",
                            "state": "open", "retry_after": 0,
                        })
                        circuits["deepseek"] = deepseek
                        code = "deepseek_auth_failed"
                        continue
                    failures = [float(value) for value in deepseek.get("failures") or [] if now - float(value) < 86400]
                    failures.append(now)
                    deepseek.update({
                        "failures": failures[-5:], "last_error_code": "deepseek_transient",
                        "state": "open" if len(failures) >= 5 else "degraded",
                        "retry_after": now + 6 * 3600 if len(failures) >= 5 else 0,
                    })
                    circuits["deepseek"] = deepseek
                    code = "deepseek_transient"
                else:
                    code = reason or "cloud_processing_failed"
        state["seen"] = seen
    except PlatformSessionError as exc:
        code = str(exc) if str(exc) in {
            "platform_auth_failed", "platform_ticket_rejected", "platform_session_rejected",
            "platform_connection_failed", "platform_course_request_failed",
        } else "platform_session_failed"
        if code in {"platform_auth_failed", "platform_ticket_rejected", "platform_session_rejected"}:
            auth = dict(circuits.get("authentication") or {})
            failures = int(auth.get("failures") or 0) + 1
            auth.update({
                "failures": failures, "state": "open" if failures >= 3 else "degraded",
                "last_error_code": code,
            })
            circuits["authentication"] = auth
        else:
            platform = dict(circuits.get("platform") or {})
            platform.update({
                "state": "degraded",
                "failures": int(platform.get("failures") or 0) + 1,
                "last_error_code": code,
            })
            circuits["platform"] = platform
        counts["failed"] += 1
    except CloudAutomationError as exc:
        code = exc.code
        counts["failed"] += 1
    finally:
        elapsed = time.monotonic() - started
        budget["runner_minutes"] = round(float(budget.get("runner_minutes") or 0) + elapsed / 60.0, 3)
        state.update({
            "schema": STATE_SCHEMA,
            "revision": int(state.get("revision") or 0) + 1,
            "budget": budget,
            "circuits": circuits,
            "last_run": {"code": code, "counts": counts, "elapsed_seconds": round(elapsed, 2)},
            "updated_at": time.time(),
        })
        root = Path(".work") / "cloud-state"
        root.mkdir(parents=True, exist_ok=True)
        (root / "state.box.json").write_bytes(_seal_state(state, key))
        api_key = ""
        if connector is not None:
            connector.session.close()
        print(
            f"stage=complete code={code} discovered={counts['discovered']} "
            f"processed={counts['processed']} failed={counts['failed']} deferred={counts['deferred']} "
            f"elapsed={round(elapsed, 2)}",
            flush=True,
        )
    return 0 if code in {
        "cloud_daily_completed", "budget_exhausted", "deepseek_transient",
        "deepseek_circuit_open",
    } else 1


def main() -> int:
    reset_usage()
    try:
        return verify() if os.environ.get("COURSELENS_CLOUD_VERIFY_ONLY") == "1" else run_daily()
    except Exception as exc:
        code = exc.code if isinstance(exc, CloudAutomationError) else safe_worker_error_detail(exc) or "cloud_worker_failed"
        print(f"cloud_worker_failed code={code}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
