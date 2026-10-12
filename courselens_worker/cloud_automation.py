"""Opt-in, bounded cloud discovery and learning-pack processing.

All platform access is delegated to ``platform_session``.  This module keeps
only encrypted incremental state and encrypted result envelopes in Actions
artifacts.  Public logs contain closed-set codes, counts, stages and timing.

The ``cloud-automation.v3`` contract fixes the schedule on the client side
(weekday class-grid cron points shortly after each standard period ends plus
a nightly 22:00 Asia/Shanghai fallback).  Each daytime scheduled run first
passes a bounded schedule gate (SMART-SCHED): with positive empty evidence —
no selected course ends within the window and no uncertain rows — the run
skips discovery honestly (``cloud_gate_skipped``); every uncertain or
unknown-ledger row passes the gate (fail-open).  The worker trusts
only the exact signed rules envelope and its config hash.  Every run is bound
to the exact expected protocol and verified config values before any school
login; manual dispatch stays closed while automation is paused.  One selected
course is one fixed output bundle (subtitle ASR, slide OCR, AI summary and
chapters, Lecture IR/evidence); eligibility is decided by the client-captured
selection baseline, while the completed-work identity binds account scope,
course, lecture, output-bundle version and pipeline version so unrelated
config changes never replay finished lectures.  Per-lecture stage progress is
checkpointed into the encrypted state artifact so a retry reacquires fresh
media URLs and resumes completed stages instead of restarting.

The courseware plan (``courseware_plan.v1``) is derived during the OCR image
pass results: it records which capture events to keep and in which order.
It carries opaque identifiers, hashes, labels and decisions only — never a
``pptimgurl``, signed URL, cookie, OCR body, thumbnail, or image bytes.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import secrets
import sys
import time
import zipfile
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
from nacl.secret import SecretBox
from nacl.signing import SigningKey

from .llm import LLMError, _chat, reset_usage, usage_snapshot
from .platform_session import PlatformSessionError, cloud_session_from_environment
from .protocol import PROTOCOL_VERSION, RESULT_SCHEMA, canonical_json, seal_result, sha256_hex
from .runner import process_job, safe_worker_error_detail


STATE_SCHEMA = "cloud.state.v1"
RULES_SCHEMA = "cloud-automation.v3"
RESULT_PREFIX = "courselens-cloud-result-"
STATE_PREFIX = "courselens-cloud-state-"

# The v3 fixed output bundle: every selected course promises exactly these
# outputs; quiz generation stays a manual, local action.
OUTPUT_BUNDLE = ("subtitle", "ocr", "summary", "chapters")
OUTPUT_BUNDLE_VERSION = "unattended-full.v1"

# A pending record from a crashed run blocks reprocessing for one day, then
# the item becomes retryable again.
PENDING_TTL_SECONDS = 24 * 3600

# Bounded active-lecture checkpoints: newest wins, older entries drop so the
# encrypted state artifact stays small even across long candidate lists.
CHECKPOINT_KEEP_LIMIT = 4

# SMART-SCHED（2026-10-06）智能门常量：cron 网格点 = 节末 + 30 分钟，门把
# 「实际启动 - 30 分钟」视为本窗末，±20 分钟吸收 GitHub schedule 的常规延迟。
# 门仅在白天课窗（08:30-19:00 北京时间）对 schedule 触发生效；22:00 夜窗
# 兜底、manual 与未带伪造时刻的验收一律旁路（操作者意图优先，现状不回归）。
GATE_OFFSET_MIN = 30
GATE_TOLERANCE_MIN = 20
GATE_TZ = ZoneInfo("Asia/Shanghai")
GATE_DAY = (dtime(8, 30), dtime(19, 0))

# Plan bounds and frozen ordering gate (courseware_plan.v1).
PLAN_SCHEMA = "courseware_plan.v1"
PLAN_POLICY_VERSION = 1
PLAN_MAX_ENTRIES = 5000
PLAN_LABEL_COVERAGE_MIN = 0.9
PLAN_RECORD_ID_MAX_LENGTH = 64
PLAN_PAGE_LABEL_RE = re.compile(r"\b(\d{1,3})\s*/\s*(\d{1,3})\b")
PLAN_CHINESE_PAGE_LABEL_RE = re.compile(r"第\s*(\d{1,3})\s*页")


class CloudAutomationError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class _GateSkipSignal(RuntimeError):
    """Internal: the schedule gate proved this run would be an empty scan.

    Raised only on positive empty evidence (no selected course ends near this
    window and no uncertain rows).  Never counts as a failure; the encrypted
    state still records the run code and honest runner minutes.
    """


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
        "checkpoints": {},
        "budget": {"date": "", "lectures": 0, "runner_minutes": 0.0, "deepseek_tokens": 0},
        "circuits": {
            "authentication": {"state": "closed", "failures": 0},
            "deepseek": {"state": "closed", "failures": [], "retry_after": 0},
            "platform": {"state": "closed", "failures": 0},
            "budget": {"state": "closed", "failures": 0},
        },
        "pending": [],
        "verification": {},
        "updated_at": 0,
    }


def _bounded_checkpoints(value: Any) -> dict[str, Any]:
    """Keep only fresh, newest checkpoints so the state artifact stays small."""
    now = time.time()
    live = {
        str(key): dict(item)
        for key, item in dict(value or {}).items()
        if isinstance(item, dict)
        and now - float(item.get("saved_at") or 0) < PENDING_TTL_SECONDS
    }
    if len(live) <= CHECKPOINT_KEEP_LIMIT:
        return live
    keep = sorted(
        live.items(),
        key=lambda item: float(item[1].get("saved_at") or 0),
    )[-CHECKPOINT_KEEP_LIMIT:]
    return dict(keep)


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
    if not str(value.get("config_hash") or ""):
        raise CloudAutomationError("cloud_rules_invalid")
    total_baseline = 0
    for item in list(value.get("rules") or []):
        if not isinstance(item, dict):
            raise CloudAutomationError("cloud_rules_invalid")
        baseline = list(item.get("baseline") or [])
        if len(baseline) > 400:
            raise CloudAutomationError("cloud_rules_invalid")
        if any(not str(row or "").strip() or len(str(row)) > PLAN_RECORD_ID_MAX_LENGTH for row in baseline):
            raise CloudAutomationError("cloud_rules_invalid")
        total_baseline += len(baseline)
    if total_baseline > 2000:
        raise CloudAutomationError("cloud_rules_invalid")
    return value


def _verify_run_binding(rules: dict[str, Any], *, check_dispatch: bool) -> None:
    """Bind this run to the exact verified protocol/config before any login.

    The workflow passes the published repo variables and the dispatch inputs
    through the environment; a mismatch (or a manual dispatch while
    automation is paused) fails closed before the school session is touched.
    """
    expected_protocol = os.environ.get("COURSELENS_CLOUD_EXPECTED_PROTOCOL_VERSION", "").strip()
    if expected_protocol != RULES_SCHEMA:
        raise CloudAutomationError("cloud_protocol_mismatch")
    expected_hash = os.environ.get("COURSELENS_CLOUD_EXPECTED_CONFIG_HASH", "").strip()
    if not expected_hash or expected_hash != str(rules.get("config_hash") or ""):
        raise CloudAutomationError("cloud_config_mismatch")
    if not check_dispatch:
        return
    if os.environ.get("COURSELENS_CLOUD_MANUAL_DISPATCH", "").strip().lower() == "true":
        dispatch_hash = os.environ.get("COURSELENS_CLOUD_DISPATCH_CONFIG_HASH", "").strip()
        if dispatch_hash != str(rules.get("config_hash") or ""):
            raise CloudAutomationError("cloud_config_mismatch")
        enabled = os.environ.get("COURSELENS_CLOUD_ENABLED_FLAG", "").strip().lower()
        if enabled != "true":
            raise CloudAutomationError("cloud_dispatch_paused")


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
    # CLOUDVERIFY-FIX（2026-10-05）：deepseek-flash（e6cb675 起）为思考档模型，
    # 传参前 _chat 缺省不带 thinking 字段=提供商默认 enabled/high，思考 token 计入
    # max_tokens 预算 → 原 max_tokens=8 全被思考耗尽、content 为空，探针误判
    # deepseek_verification_failed（verify workflow 两连败根因）。显式关思考
    # （llm.TERM_THINKING 同款合法形态，字面量内联防 env 覆写）+ 预算提至 32
    # （"OK" 仅 2-3 token，余量覆盖分词差异）；闭集比较语义不动。
    # （L1a/DEEPCOST-FIX-1 后 _chat 漏传缺省=disabled；此处显式形态由
    # test_llm_thinking_pin AST 钉守护，保留。）
    value = _chat(api_key, [
        {"role": "system", "content": "Reply with OK only."},
        {"role": "user", "content": "Connection check"},
    ], max_tokens=32, thinking={"type": "disabled"})
    if value.strip().upper() != "OK":
        raise CloudAutomationError("deepseek_verification_failed")


def _emit_verify_stage(stage: str, elapsed: float) -> None:
    # CLOUDVERIFY-FIX P1：verify 各腿闭集 stage 行——只落闭集词与秒，零口令、
    # 零课程名、零账号值（与 runner._progress / llm telemetry 同纪律）。
    print(f"stage={stage} elapsed={round(elapsed, 2)}", flush=True)


def _rules_need_ai(rules: dict[str, Any]) -> bool:
    # v3 固定包：任何选中课程都包含 AI 总结/章节，因此只要存在选中规则就
    # 必须 配置 DeepSeek Key（客户端在上传时同样 fail closed）。
    return any(
        isinstance(item, dict) and str(item.get("course_id") or "").strip()
        for item in list(rules.get("rules") or [])
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


def _verification_record(rules: dict[str, Any]) -> dict[str, Any]:
    """Signed verification evidence, stored inside the encrypted state."""
    record = {
        "config_hash": str(rules.get("config_hash") or ""),
        "protocol": RULES_SCHEMA,
        "verified_at": time.time(),
    }
    signing_bytes = base64.b64decode(_required("WORKER_SIGNING_PRIVATE_KEY"), validate=True)
    signature = SigningKey(signing_bytes).sign(canonical_json(record)).signature
    record["receipt"] = signature.hex()
    return record


def _persist_state(state: dict[str, Any], key: bytes) -> None:
    # REVOKE-REVISION-FIX（2026-10-08，E2E-2v5 定谳缺陷根修）：revision 原为
    # artifact 内计数器（prior+1）。revoke/erase/artifact 过期清掉仓内状态
    # artifact 后计数器归零重启，而客户端导入闸按持久化 prior 单调比较、低
    # 于天花板静默拒收（src/runtime/automation.py `_import_cloud_state`），
    # 撤销→重启用生命周期内 verify 翻转被砖 N 次派发（N=prior）。改为墙钟
    # 毫秒基序数：单调性不再依附可删 artifact 的存续，新链首件即大于任何
    # 历史计数器值；``max(prior + 1)`` 地板保证同链严格递增（时钟回拨/同
    # 毫秒连写安全）。客户端比较语义零改动，同链 staleness 防护不弱化。
    state.update({
        "schema": STATE_SCHEMA,
        "revision": max(int(time.time() * 1000), int(state.get("revision") or 0) + 1),
        "updated_at": time.time(),
    })
    root = Path(".work") / "cloud-state"
    root.mkdir(parents=True, exist_ok=True)
    (root / "state.box.json").write_bytes(_seal_state(state, key))


def _plan_page_label(text: str) -> str:
    """Normalize ``11/60`` / ``第3页`` labels from recognized OCR text.

    Only the normalized label survives into the plan — never the OCR body.
    An empty string means the capture carries no believable page label.
    """
    value = str(text or "")
    match = PLAN_PAGE_LABEL_RE.search(value)
    if match:
        current, total = int(match.group(1)), int(match.group(2))
        if 1 <= current <= total <= 999:
            return f"{current}/{total}"
        return ""
    match = PLAN_CHINESE_PAGE_LABEL_RE.search(value)
    if match:
        current = int(match.group(1))
        if 1 <= current <= 999:
            return f"{current}"
    return ""


def _plan_record_id(value: Any) -> str:
    """Bounded opaque capture record id; anything URL-like is dropped."""
    record_id = str(value or "").strip()
    if (
        not record_id
        or len(record_id) > PLAN_RECORD_ID_MAX_LENGTH
        or any(char.isspace() for char in record_id)
        or "://" in record_id
        or "@" in record_id
    ):
        return ""
    return record_id


def build_courseware_plan(
    *,
    course_id: str,
    sub_id: str,
    pages: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    skipped_total: int = 0,
) -> dict[str, Any]:
    """Derive the bounded keep/order plan from one OCR image pass.

    Collapse rule: only exact ``source_sha256`` duplicates collapse.  Every
    distinct content variant is kept — same-label variants with any visual
    difference stay protected as ``annotated_candidate`` and unlabeled
    variants stay ``unknown``; classification is metadata and never deletes
    or replaces a page.  Ordering uses page labels only behind the frozen
    coverage/denominator/conflict gate, and otherwise preserves first-capture
    order.  The plan holds opaque identifiers, hashes, labels and decisions —
    no URLs, cookies, OCR text, thumbnails, or image bytes.
    """
    inventory_by_num: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(inventory or []):
        inventory_by_num[max(1, int((item or {}).get("page_num") or index + 1))] = {
            "record_id": _plan_record_id((item or {}).get("record_id")),
            "created_sec": max(0, int((item or {}).get("created_sec") or 0)),
        }
    entries: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_sha: dict[str, int] = {}
    ordinal_at_time: dict[int, int] = {}
    last_by_label: dict[str, dict[str, Any]] = {}
    for page in list(pages or []):
        page = dict(page or {})
        page_num = max(1, int(page.get("page_num") or 0))
        created_sec = max(0, int(page.get("created_sec") or 0))
        source_sha256 = str(page.get("source_sha256") or "")
        if not source_sha256:
            continue
        row = inventory_by_num.get(page_num) or {}
        # The opaque record id joins only when the capture-time guard agrees,
        # so a moved inventory row can never silently re-anchor an entry.
        record_id = str(row.get("record_id") or "") if row.get("created_sec") == created_sec else ""
        ordinal = ordinal_at_time.get(created_sec, 0)
        ordinal_at_time[created_sec] = ordinal + 1
        label = _plan_page_label(page.get("text") or "")
        keeper_position = seen_sha.get(source_sha256)
        if keeper_position is not None:
            excluded.append({
                "kept_position": keeper_position,
                "capture_time": created_sec,
                "capture_ordinal": ordinal,
                "source_sha256": source_sha256,
                "reason": "exact_duplicate",
            })
            keeper = entries[keeper_position - 1]
            keeper["duplicate_count"] = int(keeper.get("duplicate_count") or 0) + 1
            continue
        annotation = {"class": "unknown", "confidence": 0.0}
        keep_reason = "distinct_capture"
        version_of = 0
        base = last_by_label.get(label) if label else None
        if base is not None:
            version_of = int(base.get("position") or 0)
            if str(page.get("dhash") or "") == str(base.get("dhash") or ""):
                annotation = {"class": "clean_candidate", "confidence": 0.8}
            else:
                annotation = {"class": "annotated_candidate", "confidence": 0.55}
            keep_reason = "version_variant_retained"
        entry = {
            "position": len(entries) + 1,
            "record_id": record_id,
            "capture_time": created_sec,
            "capture_ordinal": ordinal,
            "source_sha256": source_sha256,
            "page_label": label,
            "page_label_source": "ocr_text" if label else "",
            "annotation": annotation,
            "keep_reason": keep_reason,
            "version_of_position": version_of,
            "duplicate_count": 0,
        }
        entries.append(entry)
        seen_sha[source_sha256] = entry["position"]
        if label:
            last_by_label[label] = entry
        if len(entries) + len(excluded) > PLAN_MAX_ENTRIES:
            raise CloudAutomationError("courseware_plan_too_large")
    # Ordering: page labels win only behind the frozen gate; otherwise the
    # first-capture order stands.  Gate = denominator labels on nearly every
    # kept entry, one shared denominator, and the labeled set forming a
    # complete, conflict-free 1..M run (no repeated page number) so the
    # reorder is provably safe.
    labeled = [
        entry for entry in entries
        if "/" in str(entry.get("page_label") or "")
    ]
    mode = "capture_order"
    confidence = 0.0
    if (
        entries
        and len(labeled) / len(entries) >= PLAN_LABEL_COVERAGE_MIN
    ):
        denominators = {int(str(entry["page_label"]).split("/")[1]) for entry in labeled}
        numbers = sorted(
            int(str(entry["page_label"]).split("/")[0]) for entry in labeled
        )
        complete = numbers == list(range(1, len(numbers) + 1))
        if len(denominators) == 1 and complete and next(iter(denominators)) == len(numbers):
            mode = "page_label"
            confidence = 0.9
    if mode == "page_label":
        ordered = sorted(
            entries,
            key=lambda entry: int(str(entry["page_label"]).split("/")[0]),
        )
    else:
        ordered = entries
    for entry in ordered:
        # capture_position is the stable first-capture identity; duplicate
        # and version references always point at it, independent of the
        # chosen output ordering.
        entry["capture_position"] = entry["position"]
    for position, entry in enumerate(ordered, start=1):
        entry["output_position"] = position
    inventory_digest = sha256_hex(canonical_json([
        {"record_id": row.get("record_id") or "", "created_sec": row.get("created_sec") or 0}
        for _num, row in sorted(inventory_by_num.items())
    ]))
    return {
        "schema": PLAN_SCHEMA,
        "policy_version": PLAN_POLICY_VERSION,
        "pipeline": RULES_SCHEMA,
        "course_id": str(course_id or ""),
        "sub_id": str(sub_id or ""),
        "inventory_digest": inventory_digest,
        "entries": [
            {key: entry[key] for key in (
                "output_position", "capture_position", "record_id",
                "capture_time", "capture_ordinal",
                "source_sha256", "page_label", "page_label_source", "annotation",
                "keep_reason", "version_of_position", "duplicate_count",
            )}
            for entry in ordered
        ],
        "excluded": [
            {key: item[key] for key in (
                "kept_position", "capture_time", "capture_ordinal",
                "source_sha256", "reason",
            )}
            for item in excluded
        ],
        "ordering": {"mode": mode, "confidence": confidence},
        "counts": {
            "input_events": len(inventory_by_num),
            "recognized": len(list(pages or [])),
            "kept": len(entries),
            "exact_duplicates": len(excluded),
            "skipped": max(0, int(skipped_total or 0)),
        },
    }


def verify() -> int:
    started = time.monotonic()
    rules = _rules()
    # Protocol/config binding is rejected before any school login.
    _verify_run_binding(rules, check_dispatch=False)
    api_key = os.environ.pop("COURSELENS_CLOUD_DEEPSEEK_API_KEY", "").strip()
    connector = None
    try:
        if _rules_need_ai(rules) and not api_key:
            raise CloudAutomationError("deepseek_key_missing")
        # CLOUDVERIFY-FIX P1：每腿完成即落一行闭集 stage 行，任一腿失败时
        # 日志直读腿位（已过行+缺席行）。platform_ok=平台复合腿（UIS 认证+
        # 票据跟单；子腿失败面由失败码 connection_stage 后缀直读，成功面到达
        # 本行即两子腿已过）；ai_probe 行缺席=探针跳过（无选中课程/无 Key）。
        leg_started = time.monotonic()
        connector = cloud_session_from_environment()
        _emit_verify_stage("platform_ok", time.monotonic() - leg_started)
        if api_key:
            leg_started = time.monotonic()
            _verify_ai(api_key)
            _emit_verify_stage("ai_probe", time.monotonic() - leg_started)
        key = _state_key()
        state = _load_previous_state(key)
        if os.environ.get("COURSELENS_CLOUD_RESET_CIRCUIT") == "true":
            for name in ("authentication", "deepseek", "platform", "budget"):
                state.setdefault("circuits", {})[name] = {
                    "state": "closed", "failures": [] if name == "deepseek" else 0,
                    "retry_after": 0, "last_error_code": "",
                }
        state["verification"] = _verification_record(rules)
        _prune_previous_states()
        _persist_state(state, key)
        print(f"stage=verified elapsed={round(time.monotonic() - started, 2)}", flush=True)
        return 0
    finally:
        api_key = ""
        if connector is not None:
            connector.close()


def _result_envelope(
    *, dedupe_key: str, course: dict[str, Any], lecture: dict[str, Any],
    outputs: dict[str, Any], metrics: dict[str, Any], config_hash: str,
) -> dict[str, Any]:
    task_id = secrets.token_hex(16)
    result = {
        "schema": RESULT_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "task_id": task_id,
        "job_kind": "learning_pack",
        "input_hash": dedupe_key,
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
    envelope["input_hash"] = dedupe_key
    return envelope


def _write_result(envelope: dict[str, Any]) -> None:
    root = Path(".work") / "cloud-results"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{secrets.token_hex(16)}.box.json"
    destination.write_text(
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _dedupe_key(rules: dict[str, Any], course_id: str, sub_id: str) -> str:
    """Stable completed-work identity: account scope, course, lecture, output
    bundle version and pipeline version.

    Global configuration (budgets, priorities, rule sets, the config hash)
    deliberately stays out: a settings change must never replay finished
    lectures.  Selection-generation eligibility and the pending/completed
    ledgers remain the only replay controls.
    """
    return sha256_hex(canonical_json({
        "schema": RULES_SCHEMA,
        "account_id": str(rules.get("account_id") or ""),
        "course_id": str(course_id),
        "sub_id": str(sub_id),
        "output_bundle": OUTPUT_BUNDLE_VERSION,
        "pipeline": RULES_SCHEMA,
    }))


def _completed_keys(state: dict[str, Any]) -> set[str]:
    seen = dict(state.get("seen") or {})
    keys: set[str] = set()
    for value in seen.values():
        keys.update(str(item) for item in list(value or []))
    return keys


def _active_pending_keys(state: dict[str, Any]) -> set[str]:
    now = time.time()
    pending = [
        item for item in list(state.get("pending") or [])
        if float(item.get("expires_at") or 0) > now
    ]
    state["pending"] = pending
    return {str(item.get("key") or "") for item in pending}


# U11 挑战塌缩扩集：平台层已登记的全部精确码原样上抛（原五码扩到全闭集），
# 未登记的未知串才塌缩为 platform_session_failed——挑战/媒体/分页类失败不再
# 被吞成一句泛化话，学生看到的提示能对上真实原因。
_PLATFORM_PRECISE_CODES = frozenset({
    "platform_auth_failed", "platform_ticket_rejected", "platform_session_rejected",
    "platform_connection_failed", "platform_course_request_failed",
    "platform_auth_context_missing", "platform_auth_method_missing",
    "platform_challenge_required", "platform_course_context_missing",
    "platform_credentials_missing", "platform_key_rejected", "platform_media_missing",
    "platform_redirect_rejected", "platform_ticket_missing",
    "platform_slide_pagination_stalled", "platform_slide_payload_invalid",
    "platform_slide_record_storm", "platform_slide_response_too_large",
})


def collapse_platform_code(base_code: str) -> str:
    return str(base_code) if str(base_code) in _PLATFORM_PRECISE_CODES else "platform_session_failed"


def _auth_rejection(code: str) -> bool:
    return code in {
        "platform_auth_failed", "platform_ticket_rejected", "platform_session_rejected",
    }


def _gate_active(trigger_kind: str, now_bj: datetime) -> bool:
    """Whether the pre-filter gate decides this run at ``now_bj`` (Beijing).

    The gate exists only for the fixed daytime class grid: scheduled runs
    inside the morning-to-evening window, and acceptance runs carrying an
    explicit fake Beijing time (an E2E stand-in for a grid point).  Manual
    dispatches, the nightly 22:00 fallback, and acceptance runs outside the
    day window always keep the full pipeline.
    """
    kind = str(trigger_kind or "").strip()
    if kind == "schedule":
        return GATE_DAY[0] <= now_bj.time() < GATE_DAY[1]
    if kind == "acceptance":
        fake = os.environ.get("COURSELENS_CLOUD_GATE_FAKE_TIME", "").strip()
        return bool(fake) and GATE_DAY[0] <= now_bj.time() < GATE_DAY[1]
    return False


def _gate_now(trigger_kind: str, *, now: datetime | None = None) -> datetime | None:
    """Effective gate decision time, or None when the gate is bypassed.

    Acceptance runs decide only with an explicit fake Beijing time; a
    malformed or naive value fails closed (``cloud_gate_time_invalid``)
    instead of silently re-anchoring to the real clock.  Scheduled runs use
    the actual startup moment; GitHub schedule delay is absorbed by the
    window tolerance, not by guessing.
    """
    kind = str(trigger_kind or "").strip()
    if kind == "schedule":
        moment = now or datetime.now(GATE_TZ)
    elif kind == "acceptance":
        raw = os.environ.get("COURSELENS_CLOUD_GATE_FAKE_TIME", "").strip()
        if not raw:
            return None
        try:
            moment = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise CloudAutomationError("cloud_gate_time_invalid") from exc
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise CloudAutomationError("cloud_gate_time_invalid")
        moment = moment.astimezone(GATE_TZ)
    else:
        return None
    return moment if _gate_active(kind, moment) else None


def _gate_decision(
    connector: Any, rules: dict[str, Any], state: dict[str, Any], now_bj: datetime,
) -> tuple[bool, dict[str, int]]:
    """One bounded pre-filter question: would the full pipeline be empty?

    The gate is a pre-filter, never an arbiter: skipping requires positive
    empty evidence — no selected course ends inside this window and no
    uncertain rows — and every uncertain or unknown-ledger row passes the
    gate (fail-open).  It reads the once-per-day ledgers only and never
    writes seen/pending/baseline, so exactly-once semantics are untouched.
    """
    stats = {"rows": 0, "matched": 0, "uncertain": 0, "known": 0}
    try:
        rows = connector.today_schedule_rows()
    except PlatformSessionError:
        # Fail-open: unreadable schedule data never skips a run.
        return True, {**stats, "rows": -1}
    stats["rows"] = len(rows)
    expected_end = now_bj - timedelta(minutes=GATE_OFFSET_MIN)
    low = expected_end - timedelta(minutes=GATE_TOLERANCE_MIN)
    high = expected_end + timedelta(minutes=GATE_TOLERANCE_MIN)
    today = now_bj.date().isoformat()
    rule_map = {
        str(item.get("course_id") or ""): dict(item)
        for item in list(rules.get("rules") or [])
    }
    matched: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("course_id") or "") not in rule_map:
            continue
        date = str(row.get("date") or "")
        if date and date != today:
            continue
        end = row.get("end_precise")
        if end is None:
            # Uncertain rows are hits: fail-open, never guess.
            stats["uncertain"] += 1
            matched.append(row)
            continue
        if low <= end <= high:
            matched.append(row)
    stats["matched"] = len(matched)
    if not matched:
        return False, stats
    completed = _completed_keys(state)
    pending = _active_pending_keys(state)
    for row in matched:
        sub_id = str(row.get("sub_id") or "")
        if not sub_id:
            return True, stats
        rule = rule_map[str(row.get("course_id") or "")]
        baseline = {
            str(item) for item in list(rule.get("baseline") or [])
            if str(item or "").strip()
        }
        if sub_id in baseline:
            stats["known"] += 1
            continue
        item_key = _dedupe_key(rules, str(row.get("course_id") or ""), sub_id)
        if item_key in completed or item_key in pending:
            stats["known"] += 1
            continue
        return True, stats
    # Every matched row is already completed, pending, or baseline-excluded:
    # positive evidence the pipeline would process nothing new.
    return False, stats


def _emit_gate_stage(outcome: str, stats: dict[str, int], elapsed: float) -> None:
    # Closed-set gate line: counts and the outcome word only — never course
    # names, account values, or raw payloads (same discipline as
    # _emit_verify_stage / runner._progress).
    print(
        f"stage=gate outcome={outcome} rows={stats.get('rows', 0)} "
        f"matched={stats.get('matched', 0)} uncertain={stats.get('uncertain', 0)} "
        f"known={stats.get('known', 0)} elapsed={round(elapsed, 2)}",
        flush=True,
    )


def run_daily() -> int:
    started = time.monotonic()
    rules = _rules()
    key = _state_key()
    state = _load_previous_state(key)
    state["checkpoints"] = _bounded_checkpoints(state.get("checkpoints"))
    _prune_previous_states()
    api_key = os.environ.pop("COURSELENS_CLOUD_DEEPSEEK_API_KEY", "").strip()
    limits = dict(rules.get("budget") or {})
    budget = _reset_daily_budget(state)
    circuits = dict(state.get("circuits") or {})
    counts = {
        "discovered": 0, "processed": 0, "failed": 0, "deferred": 0,
        "skipped_pending": 0, "expiring_results": _expiring_result_count(),
    }
    code = "cloud_daily_completed"
    connector = None
    try:
        # Protocol/config/dispatch binding is rejected before any school
        # login; a manual dispatch while automation is paused stays closed.
        # The closed-set reason lands in the encrypted state for the client.
        _verify_run_binding(rules, check_dispatch=True)
        auth = dict(circuits.get("authentication") or {})
        if auth.get("state") == "open":
            raise CloudAutomationError("authentication_circuit_open")
        if _rules_need_ai(rules) and not api_key:
            # v3 fixed bundle needs AI; fail before consuming a school session.
            raise CloudAutomationError("deepseek_key_missing")
        # SMART-SCHED gate moment resolves before any school login: a
        # malformed acceptance fake time fails closed without touching UIS.
        gate_started = time.monotonic()
        gate_moment = _gate_now(os.environ.get("COURSELENS_CLOUD_TRIGGER_KIND", ""))
        connector = cloud_session_from_environment()
        if connector is None:
            raise CloudAutomationError("platform_connection_failed")
        circuits["authentication"] = {"state": "closed", "failures": 0, "last_error_code": ""}
        if gate_moment is not None:
            # SMART-SCHED 智能门：同进程同登录的预过滤器（非裁决器）。秒退仅
            # 持「本窗末 ±20 分钟确认无选中课程下课且无不确定行」的肯定性空
            # 证据；一切不确定一律过门（fail-open）。过门即落入既有候选链，
            # 预算帽、熔断、恰一次账零新增语义；秒退不计 failed，runner 分钟
            # 在 finally 诚实入账。
            gate_pass, gate_stats = _gate_decision(connector, rules, state, gate_moment)
            _emit_gate_stage(
                "pass" if gate_pass else "skip", gate_stats, time.monotonic() - gate_started,
            )
            if not gate_pass:
                raise _GateSkipSignal()
        catalog = connector.discover_authorized_courses()
        circuits["platform"] = {
            "state": "closed", "failures": 0, "last_error_code": "",
        }
        rule_map = {str(item.get("course_id") or ""): dict(item) for item in list(rules.get("rules") or [])}
        completed = _completed_keys(state)
        pending_keys = _active_pending_keys(state)
        candidates: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for course in catalog:
            course_id = str(course.get("course_id") or "")
            # Exact course opt-in: unselected courses are never processed.
            if course_id not in rule_map:
                continue
            rule = rule_map[course_id]
            baseline = {
                str(item) for item in list(rule.get("baseline") or [])
                if str(item or "").strip()
            }
            for lecture in list(course.get("lectures") or []):
                sub_id = str(lecture.get("sub_id") or "")
                # Playback gate: a listed lecture that cannot play at this
                # window becomes eligible exactly when playback appears.
                if not sub_id or not lecture.get("has_playback"):
                    continue
                # Selection baseline: lectures already playable at the
                # selection instant stay excluded for this generation.
                if sub_id in baseline:
                    continue
                requested = list(OUTPUT_BUNDLE)
                item_key = _dedupe_key(rules, course_id, sub_id)
                if item_key in completed or item_key in pending_keys:
                    # Completed or already-pending work never runs twice.
                    counts["skipped_pending"] += 1
                    continue
                counts["discovered"] += 1
                candidates.append((int(rule.get("priority") or 0), course, lecture, rule, item_key, requested))
        candidates.sort(key=lambda item: (
            -item[0], str(item[2].get("date") or ""), str(item[2].get("sub_id") or "")
        ))
        maximum = max(1, int(limits.get("max_lectures") or 2))
        runner_stop = max(0.0, float(limits.get("max_runner_minutes") or 300) - 30.0)
        token_limit = max(0, int(limits.get("max_deepseek_tokens") or 100000))

        def remember_pending(course_id: str, sub_id: str, item_key: str) -> None:
            pending = [
                item for item in list(state.get("pending") or [])
                if float(item.get("expires_at") or 0) > time.time()
            ]
            pending.append({
                "key": item_key, "course_id": str(course_id), "sub_id": str(sub_id),
                "expires_at": time.time() + PENDING_TTL_SECONDS,
            })
            state["pending"] = pending
            _persist_state(state, key)

        def forget_pending(item_key: str) -> None:
            state["pending"] = [
                item for item in list(state.get("pending") or [])
                if str(item.get("key") or "") != item_key
            ]

        for _priority, course, lecture, rule, item_key, requested in candidates:
            course_id = str(course.get("course_id") or "")
            sub_id = str(lecture.get("sub_id") or "")
            current_runner_minutes = float(budget.get("runner_minutes") or 0) + (time.monotonic() - started) / 60.0
            if budget["lectures"] >= maximum or current_runner_minutes >= runner_stop or budget["deepseek_tokens"] >= token_limit:
                # Deferred items stay retryable: never marked completed.
                counts["deferred"] += 1
                circuits["budget"] = {"state": "open", "failures": 0, "last_error_code": "budget_exhausted"}
                code = "budget_exhausted"
                continue
            deepseek_circuit = dict(circuits.get("deepseek") or {})
            if (
                deepseek_circuit.get("state") == "open"
                and float(deepseek_circuit.get("retry_after") or 0) > time.time()
            ):
                counts["deferred"] += 1
                code = "deepseek_circuit_open"
                continue
            duration = int(lecture.get("duration_seconds") or 0)
            if duration and duration > int(rule.get("max_lecture_minutes") or 240) * 60:
                counts["deferred"] += 1
                continue
            remember_pending(course_id, sub_id, item_key)
            reset_usage()
            # Media URLs are always reacquired fresh for this attempt; a
            # saved checkpoint resumes completed stages without restarting.
            checkpoints = dict(state.get("checkpoints") or {})
            saved_checkpoint = dict(checkpoints.get(item_key) or {})
            slides = connector.slide_sources(course_id, sub_id)
            slides_meta = [
                {
                    "record_id": item.get("record_id"),
                    "created_sec": item.get("created_sec"),
                    "page_num": item.get("page_num"),
                }
                for item in slides
                if isinstance(item, dict)
            ]
            payload = {
                "mode": "automatic",
                "title": str(course.get("title") or ""),
                "media": connector.media_source(course_id, sub_id),
                "slides": slides,
            }
            if saved_checkpoint:
                payload["checkpoint"] = saved_checkpoint
            job = {
                "schema": "job.v2", "protocol_version": PROTOCOL_VERSION,
                "task_id": secrets.token_hex(16), "job_kind": "learning_pack",
                "input_hash": "", "result_public_key": _required("COURSELENS_CLOUD_RESULT_PUBLIC_KEY"),
                "payload": payload,
                "requested_outputs": requested,
                "secrets": {"deepseek_api_key": api_key},
                "pipeline": {"version": RULES_SCHEMA},
            }
            job["input_hash"] = sha256_hex(canonical_json({key: value for key, value in job.items() if key != "input_hash"}))

            def checkpoint_writer(value: dict[str, Any], _item_key: str = item_key) -> None:
                # One encrypted active-lecture checkpoint inside the existing
                # state artifact: bounded, newest-wins, sealed with the state.
                updated = dict(state.get("checkpoints") or {})
                updated[_item_key] = {**dict(value or {}), "saved_at": time.time()}
                state["checkpoints"] = _bounded_checkpoints(updated)
                _persist_state(state, key)

            try:
                result = process_job(job, checkpoint_writer=checkpoint_writer)
                outputs = dict(result.get("outputs") or {})
                pages = list(outputs.get("ppt_pages") or [])
                if pages and slides_meta:
                    skipped_counts = dict(
                        (result.get("metrics") or {}).get("slides_skipped") or {}
                    )
                    plan = build_courseware_plan(
                        course_id=course_id, sub_id=sub_id,
                        pages=pages, inventory=slides_meta,
                        skipped_total=sum(
                            int(value) for value in skipped_counts.values()
                            if isinstance(value, (int, float))
                        ),
                    )
                    outputs["courseware_plan"] = plan
                    outputs["courseware_plan_digest"] = sha256_hex(canonical_json(plan))
                usage = usage_snapshot()
                metrics = dict(result.get("metrics") or {})
                metrics["deepseek_tokens"] = int(usage.get("total_tokens") or 0)
                _write_result(_result_envelope(
                    dedupe_key=item_key, course=course, lecture=lecture,
                    outputs=outputs,
                    metrics=metrics, config_hash=str(rules.get("config_hash") or ""),
                ))
                budget["lectures"] += 1
                budget["deepseek_tokens"] += int(usage.get("total_tokens") or 0)
                forget_pending(item_key)
                # Success clears this lecture's checkpoint: the encrypted
                # result now carries everything the client needs.
                updated = dict(state.get("checkpoints") or {})
                updated.pop(item_key, None)
                state["checkpoints"] = updated
                state.setdefault("seen", {}).setdefault(course_id, []).append(item_key)
                completed.add(item_key)
                counts["processed"] += 1
                circuits["deepseek"] = {"state": "closed", "failures": [], "retry_after": 0, "last_error_code": ""}
                _persist_state(state, key)
            except Exception as exc:
                # A failed item stays retryable: pending clears, seen stays
                # untouched, and the saved checkpoint (if any) survives so a
                # retry resumes completed stages.  One lecture failure never
                # erases other safe candidates.
                forget_pending(item_key)
                _persist_state(state, key)
                counts["failed"] += 1
                reason = safe_worker_error_detail(exc)
                if isinstance(exc, CloudAutomationError):
                    code = exc.code
                    continue
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
    except _GateSkipSignal:
        # 智能门肯定性空证据秒退：零 LLM、零媒体、零候选扫描、不计 failed；
        # state 照常落盘（last_run.code 可观测 + runner 分钟诚实记账）。
        code = "cloud_gate_skipped"
    except PlatformSessionError as exc:
        base_code = str(exc)
        code = safe_worker_error_detail(exc) if base_code in _PLATFORM_PRECISE_CODES else "platform_session_failed"
        if _auth_rejection(base_code):
            # A known credential or identity rejection opens the auth circuit
            # immediately; the next scheduled login is prevented.
            circuits["authentication"] = {
                "state": "open", "failures": int((dict(circuits.get("authentication") or {})).get("failures") or 0) + 1,
                "last_error_code": code,
            }
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
            "budget": budget,
            "circuits": circuits,
            "last_run": {"code": code, "counts": counts, "elapsed_seconds": round(elapsed, 2)},
        })
        _persist_state(state, key)
        api_key = ""
        if connector is not None:
            connector.close()
        print(
            f"stage=complete code={code} discovered={counts['discovered']} "
            f"processed={counts['processed']} failed={counts['failed']} deferred={counts['deferred']} "
            f"skipped_pending={counts['skipped_pending']} "
            f"elapsed={round(elapsed, 2)}",
            flush=True,
        )
    return 0 if code in {
        "cloud_daily_completed", "budget_exhausted", "deepseek_transient",
        "deepseek_circuit_open", "cloud_gate_skipped",
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
