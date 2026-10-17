"""Opt-in, bounded cloud discovery and learning-pack processing.

All platform access is delegated to ``platform_session``.  This module keeps
only encrypted incremental state and encrypted result envelopes in Actions
artifacts.  Public logs contain closed-set codes, counts, stages and timing.

The ``cloud-automation.v4`` contract fixes the schedule on the client side
(weekday class-grid cron points shortly after each standard period ends plus
a nightly 22:00 Asia/Shanghai fallback).  Each daytime scheduled run first
passes a bounded schedule gate (SMART-SCHED): with positive empty evidence —
no selected course ends within the window and no uncertain rows — the run
skips discovery honestly (``cloud_gate_skipped``); every uncertain or
unknown-ledger row passes the gate (fail-open).  The worker trusts
only the exact signed rules envelope and its config hash.  v4 makes the
cascade semantics explicit: every playable lecture of a selected course is
eligible — including the lectures already playable when the course was
enabled (enabling promises the full history) — while each rule carries a
client-computed ``cascade_skip`` list (lectures whose local artifacts are
already complete) so the budget is spent only on lectures that still lack
outputs.  A legacy ``v3`` envelope (an old client) keeps the historical
baseline-exclusion semantics; the schema field carries the semantics, so a
missing field is never a signal.  The completed-work identity and every
output stamp stay bound to the frozen processing-pipeline identity across
the protocol bump, so finished lectures and imported summaries never replay
because of an upgrade.  Every run is bound
to the exact expected protocol and verified config values before any school
login; manual dispatch stays closed while automation is paused.  One selected
course is one fixed output bundle (subtitle ASR, slide OCR, AI summary and
chapters, Lecture IR/evidence); per-lecture stage progress is
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
RULES_SCHEMA = "cloud-automation.v4"
# 协议载体闭集（V010-CASCADE-1）：v4=显式级联语义（cascade_skip=客户端申明
# 本地产物已齐的讲次，预算只花在缺产物讲次上）；v3=旧信封（老客户端），
# 回退 baseline 排除旧语义不回归（设计 R1）。语义由 schema 字段显式承载，
# 字段缺席永远不是信号。
_RULES_SCHEMAS = ("cloud-automation.v3", "cloud-automation.v4")
# 恰一次账与产物盖章的管线身份（冻结，V010-CASCADE-1）：v3→v4 只反转「哪些
# 讲次进候选」的选择语义，处理管线与输出包（unattended-full.v1）未变——
# dedupe 键、job/result 的 pipeline 盖章、courseware_plan.pipeline 都锚在
# 本身份上，已完成讲次与已导入总结永不因协议升级重跑/重复导入。
PIPELINE_IDENTITY = "cloud-automation.v3"
RESULT_PREFIX = "courselens-cloud-result-"
STATE_PREFIX = "courselens-cloud-state-"

# The v3 fixed output bundle: every selected course promises exactly these
# outputs; quiz generation stays a manual, local action.
OUTPUT_BUNDLE = ("subtitle", "ocr", "summary", "chapters")
OUTPUT_BUNDLE_VERSION = "unattended-full.v1"

# 长讲时长上限的闭集顶（V010-SUBFIX-1）：与客户端 max_lecture_minutes 校验
# 上界（5..600）同源，兜底平台目录里的异常时长（如未停止的全天流）。
LECTURE_DURATION_CAP_MAX_MINUTES = 600

# A pending record from a crashed run blocks reprocessing for one day, then
# the item becomes retryable again.
PENDING_TTL_SECONDS = 24 * 3600

# Bounded active-lecture checkpoints: newest wins, older entries drop so the
# encrypted state artifact stays small even across long candidate lists.
CHECKPOINT_KEEP_LIMIT = 4

# 内容生命周期三态闭集（V010-CONTENTLIFE-1-A，设计权威
# product-v010contentlife-result-20261017.md §2）：讲次级 media_state ∈
# {current, mismatch_deferred, stale_reprocessing}。签名=平台目录声明时长
# （duration_seconds，每窗免费重枚举自带），变化阈值 ≥2s 吸收平台秒级抖动
# （设计 §2.2 转移图「签名变化≥2s」）。台账全部 additive 落加密 state 且
# 有界封顶（completed ≤2000 / mismatch ≤400 / 观测 ≤400，与 envelope 同规）；
# 老状态缺键=空账，缺字段永远不是信号。
MEDIA_STATE_CURRENT = "current"
MEDIA_STATE_MISMATCH_DEFERRED = "mismatch_deferred"
MEDIA_STATE_STALE_REPROCESSING = "stale_reprocessing"
# worker 侧确定性失败闭集码（MEDIAMISMATCH-1 产物，runner._ASR_ERROR_CODES
# 映射「authorized media duration mismatch」）：mismatch 台账的唯一入口。
MEDIA_MISMATCH_CODE = "media_duration_mismatch"
MEDIA_SIGNATURE_DELTA_SECONDS = 2
COMPLETED_MEDIA_KEEP_LIMIT = 2000
MISMATCH_MEDIA_KEEP_LIMIT = 400
MEDIA_OBSERVATIONS_KEEP_LIMIT = 400

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


def _signature_changed(previous: int, current: int) -> bool:
    """设计 §2.2：声明时长签名变化 ≥2s 才算平台侧真变化；秒级抖动不复活
    不重排队，缺声明（0）无从比对一律不判（缺字段永远不是信号）。"""
    return (
        int(previous or 0) > 0 and int(current or 0) > 0
        and abs(int(current) - int(previous)) >= MEDIA_SIGNATURE_DELTA_SECONDS
    )


def _bounded_media_ledger(value: Any, limit: int, *, stamp_field: str) -> dict[str, Any]:
    """有界签名台账：按时间戳 newest-wins，老条目丢弃保 state artifact 小。"""
    items = [
        (str(key), dict(row))
        for key, row in dict(value or {}).items()
        if isinstance(row, dict)
    ]
    if len(items) <= limit:
        return dict(items)
    items.sort(key=lambda item: float(item[1].get(stamp_field) or 0))
    return dict(items[-limit:])


def _bounded_media_observations(value: Any) -> list[dict[str, Any]]:
    items = [dict(row) for row in list(value or []) if isinstance(row, dict)]
    return items[-MEDIA_OBSERVATIONS_KEEP_LIMIT:]


def _record_media_observation(
    state: dict[str, Any], *, course_id: str, sub_id: str,
    media_state: str, declared_duration_seconds: int,
) -> None:
    """讲次级观测入既有加密 state（additive 有界，设计 §2.4）：客户端
    _import_cloud_state 消费为 catalog media_state 与「老师更新了」通知面。
    闭集字段（sub_id/state/声明秒数/观测时刻），零课程名零媒体值。"""
    observations = _bounded_media_observations(state.get("media_observations"))
    observations.append({
        "course_id": str(course_id or ""),
        "sub_id": str(sub_id or ""),
        "media_state": str(media_state),
        "declared_duration_seconds": int(declared_duration_seconds or 0),
        "observed_at": time.time(),
    })
    state["media_observations"] = _bounded_media_observations(observations)


def _prune_media_ledgers(state: dict[str, Any]) -> None:
    """内容生命周期台账封顶（设计 §2.6）：只在有内容时写键，老状态零扰动。"""
    completed = _bounded_media_ledger(
        state.get("completed_media"), COMPLETED_MEDIA_KEEP_LIMIT, stamp_field="recorded_at",
    )
    if completed:
        state["completed_media"] = completed
    mismatch = _bounded_media_ledger(
        state.get("mismatch_media"), MISMATCH_MEDIA_KEEP_LIMIT, stamp_field="first_detected_at",
    )
    if mismatch:
        state["mismatch_media"] = mismatch
    observations = _bounded_media_observations(state.get("media_observations"))
    if observations:
        state["media_observations"] = observations


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
    if not isinstance(value, dict) or str(value.get("schema") or "") not in _RULES_SCHEMAS:
        raise CloudAutomationError("cloud_rules_invalid")
    if not str(value.get("config_hash") or ""):
        raise CloudAutomationError("cloud_rules_invalid")
    total_baseline = 0
    total_skipped = 0
    for item in list(value.get("rules") or []):
        if not isinstance(item, dict):
            raise CloudAutomationError("cloud_rules_invalid")
        baseline = list(item.get("baseline") or [])
        skipped = list(item.get("cascade_skip") or [])
        # cascade_skip 与 baseline 同形同界：有界不透明 sub_id 列表
        # （≤400/课程、总量 ≤2000），校验面零新增。
        if len(baseline) > 400 or len(skipped) > 400:
            raise CloudAutomationError("cloud_rules_invalid")
        if any(
            not str(row or "").strip() or len(str(row)) > PLAN_RECORD_ID_MAX_LENGTH
            for row in baseline + skipped
        ):
            raise CloudAutomationError("cloud_rules_invalid")
        total_baseline += len(baseline)
        total_skipped += len(skipped)
    if total_baseline > 2000 or total_skipped > 2000:
        raise CloudAutomationError("cloud_rules_invalid")
    return value


def _selection_skip_ids(schema: str, rule: dict[str, Any]) -> set[str]:
    """Per-course already-known lecture ids, by explicit envelope schema.

    v4 cascade semantics: the client declares which lectures already have
    complete local artifacts (``cascade_skip``).  A legacy v3 envelope keeps
    the historical baseline-exclusion semantics.  The schema field carries
    the semantics; a missing field is never a signal.
    """
    field = "cascade_skip" if str(schema) == RULES_SCHEMA else "baseline"
    return {
        str(row).strip()
        for row in list(rule.get(field) or [])
        if str(row or "").strip()
    }


def _verify_run_binding(rules: dict[str, Any], *, check_dispatch: bool) -> None:
    """Bind this run to the exact verified protocol/config before any login.

    The workflow passes the published repo variables and the dispatch inputs
    through the environment; a mismatch (or a manual dispatch while
    automation is paused) fails closed before the school session is touched.
    """
    expected_protocol = os.environ.get("COURSELENS_CLOUD_EXPECTED_PROTOCOL_VERSION", "").strip()
    # The variable and the envelope are written by the same client upload, so
    # they must name the same schema: a v3 client keeps env=v3 + v3 envelope,
    # a v4 client env=v4 + v4 envelope.  Any cross pairing fails closed.
    if expected_protocol != str(rules.get("schema") or ""):
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
        "protocol": str(rules.get("schema") or RULES_SCHEMA),
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
        "pipeline": PIPELINE_IDENTITY,
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
        "pipeline_fingerprint": PIPELINE_IDENTITY,
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
    bundle version and the frozen processing-pipeline identity.

    Global configuration (budgets, priorities, rule sets, the config hash)
    deliberately stays out: a settings change must never replay finished
    lectures.  The pipeline identity is frozen across the v3→v4 selection-
    semantics bump (V010-CASCADE-1): upgrading the client must never turn the
    completed ledger into replays or duplicate summary imports.  Selection-
    generation eligibility and the pending/completed ledgers remain the only
    replay controls.
    """
    return sha256_hex(canonical_json({
        "schema": PIPELINE_IDENTITY,
        "account_id": str(rules.get("account_id") or ""),
        "course_id": str(course_id),
        "sub_id": str(sub_id),
        "output_bundle": OUTPUT_BUNDLE_VERSION,
        "pipeline": PIPELINE_IDENTITY,
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
    writes seen/pending, so exactly-once semantics are untouched.  Since
    V010-SUBFIX-1 a baseline lecture is not "known" unless the envelope says
    so: v4 cascade_skip declares client-complete lectures; a legacy v3
    envelope keeps the baseline-exclusion semantics.  Completed, pending,
    mismatch-deferred and schema-declared lectures are all known work this
    window will not do.
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
    schema = str(rules.get("schema") or "")
    rule_map = {
        str(item.get("course_id") or ""): dict(item)
        for item in list(rules.get("rules") or [])
    }
    skip_map = {
        course_id: _selection_skip_ids(schema, rule)
        for course_id, rule in rule_map.items()
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
    # 内容生命周期三态语义（V010-CONTENTLIFE-1-A）：mismatch_deferred 讲次
    # 本窗不会被 selection 段入队（零入队零配额烧），门面计入已知——不算
    # 未知工作、不烧门、不入失败面；stale_reprocessing（已摘账）对门不可见，
    # 正是本窗要复活的工作，照旧过门。
    mismatch_media = dict(state.get("mismatch_media") or {})
    for row in matched:
        sub_id = str(row.get("sub_id") or "")
        if not sub_id:
            return True, stats
        # 信封申明已知的讲次（v4 cascade_skip=客户端产物已齐 / v3 baseline=
        # 旧排除契约）与已完成或在途一样不构成本窗工作：正性空证据的一部分。
        # 语义按 schema 走（_selection_skip_ids），字段缺席永远不是信号。
        if sub_id in skip_map.get(str(row.get("course_id") or ""), set()):
            stats["known"] += 1
            continue
        item_key = _dedupe_key(rules, str(row.get("course_id") or ""), sub_id)
        if item_key in mismatch_media:
            stats["known"] += 1
            continue
        if item_key in completed or item_key in pending:
            stats["known"] += 1
            continue
        return True, stats
    # Every matched row is already completed or pending: positive evidence the
    # pipeline would process nothing new.
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
    _prune_media_ledgers(state)
    _prune_previous_states()
    api_key = os.environ.pop("COURSELENS_CLOUD_DEEPSEEK_API_KEY", "").strip()
    limits = dict(rules.get("budget") or {})
    budget = _reset_daily_budget(state)
    circuits = dict(state.get("circuits") or {})
    counts = {
        "discovered": 0, "processed": 0, "failed": 0, "deferred": 0,
        "skipped_pending": 0, "skipped_cascade": 0, "skipped_mismatch": 0,
        "expiring_results": _expiring_result_count(),
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
        envelope_schema = str(rules.get("schema") or "")
        completed = _completed_keys(state)
        pending_keys = _active_pending_keys(state)
        # 内容生命周期台账（V010-CONTENTLIFE-1-A）：completed 签名账 + mismatch
        # 台账的窗内工作副本；每次变更即刻回写 state（窗内逐项 persist 可见）。
        completed_media = dict(state.get("completed_media") or {})
        mismatch_media = dict(state.get("mismatch_media") or {})
        candidates: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for course in catalog:
            course_id = str(course.get("course_id") or "")
            # Exact course opt-in: unselected courses are never processed.
            if course_id not in rule_map:
                continue
            rule = rule_map[course_id]
            # v4 显式级联语义（V010-CASCADE-1）：cascade_skip=客户端从本地
            # catalog 真值算出的「产物已齐」讲次，预算只花在缺产物讲次上；
            # v3 旧信封回退 baseline 排除旧语义（老客户端契约不回归）。语义
            # 由信封 schema 显式承载，字段缺席永远不是信号。
            skip_ids = _selection_skip_ids(envelope_schema, rule)
            for lecture in list(course.get("lectures") or []):
                sub_id = str(lecture.get("sub_id") or "")
                # Playback gate: a listed lecture that cannot play at this
                # window becomes eligible exactly when playback appears.
                if not sub_id or not lecture.get("has_playback"):
                    continue
                if sub_id in skip_ids:
                    # 客户端申明产物已齐（v4）或启用基线（v3 旧语义）：不进
                    # 候选、不消耗预算；诚实计数供 run 卡与验收对账。
                    counts["skipped_cascade"] += 1
                    continue
                requested = list(OUTPUT_BUNDLE)
                item_key = _dedupe_key(rules, course_id, sub_id)
                # 内容生命周期两段比对（V010-CONTENTLIFE-1-A，设计 §2.3-1）：
                # 本窗重枚举自带每讲新鲜声明时长，检测成本=0、零新增平台调用。
                # ① mismatch 台账：签名未变=确定性失败仍在 → 零入队零配额烧
                #   （skipped_mismatch 诚实计数）；签名已变（平台修正/换媒体）
                #   → 摘账复活转 stale_reprocessing 自动重排队+通知事件。
                # ② 已完成讲次签名比对：「老师更新了」检测——声明时长变化 ≥2s
                #   → 摘账标 stale 自动重排队+通知事件；无基线（存量完成讲）
                #   首窗只补记不判，绝不因补基线而重跑。
                declared = int(lecture.get("duration_seconds") or 0)
                stale_requeued = False
                mm_record = dict(mismatch_media.get(item_key) or {})
                if mm_record:
                    if _signature_changed(int(mm_record.get("declared_duration_seconds") or 0), declared):
                        mismatch_media.pop(item_key, None)
                        state["mismatch_media"] = mismatch_media
                        _record_media_observation(
                            state, course_id=course_id, sub_id=sub_id,
                            media_state=MEDIA_STATE_STALE_REPROCESSING,
                            declared_duration_seconds=declared,
                        )
                        stale_requeued = True
                    else:
                        mm_record["last_checked_at"] = time.time()
                        mismatch_media[item_key] = mm_record
                        state["mismatch_media"] = mismatch_media
                        counts["skipped_mismatch"] += 1
                        continue
                elif item_key in completed:
                    recorded = int(dict(
                        completed_media.get(item_key) or {}
                    ).get("declared_duration_seconds") or 0)
                    if _signature_changed(recorded, declared):
                        seen_map = dict(state.get("seen") or {})
                        seen_map[course_id] = [
                            key for key in list(seen_map.get(course_id) or [])
                            if key != item_key
                        ]
                        state["seen"] = seen_map
                        completed.discard(item_key)
                        completed_media.pop(item_key, None)
                        state["completed_media"] = completed_media
                        _record_media_observation(
                            state, course_id=course_id, sub_id=sub_id,
                            media_state=MEDIA_STATE_STALE_REPROCESSING,
                            declared_duration_seconds=declared,
                        )
                        stale_requeued = True
                    elif not recorded and declared:
                        completed_media[item_key] = {
                            "declared_duration_seconds": declared,
                            "recorded_at": time.time(),
                        }
                        state["completed_media"] = completed_media
                if item_key in completed or item_key in pending_keys:
                    # Completed or already-pending work never runs twice.
                    counts["skipped_pending"] += 1
                    continue
                counts["discovered"] += 1
                candidates.append((
                    int(rule.get("priority") or 0), course, lecture, rule,
                    item_key, requested, stale_requeued,
                ))
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

        for _priority, course, lecture, rule, item_key, requested, stale_requeued in candidates:
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
            # V010-SUBFIX-1 长讲上限（SUBDIAG-1 定谳）：上限的保护面=「整个单窗
            # 预算都装不下的讲次」——预算只在讲次间检查，开工后装不下=24h
            # pending 停滞。实测处理时长远小于讲次时长（4.5h 连堂讲在同一
            # Actions 环全管线 ≈29min），旧默认 240min 曾把 267-270min 真实连堂
            # 课型永久 defer。有效上限=max(规则上限, 单窗预算)，再以闭集顶
            # 600min 兜底异常时长；规则值仍为下限（显式配置不被架空）。
            duration_cap_minutes = min(
                max(
                    int(rule.get("max_lecture_minutes") or 240),
                    int(limits.get("max_runner_minutes") or 300),
                ),
                LECTURE_DURATION_CAP_MAX_MINUTES,
            )
            if duration and duration > duration_cap_minutes * 60:
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
                "pipeline": {"version": PIPELINE_IDENTITY},
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
                # 内容生命周期落账（设计 §2.3-1）：成功=签名入账回 current；
                # 复活讲成功额外写 current 观测，客户端及时清 stale/deferred 标。
                completed_media[item_key] = {
                    "declared_duration_seconds": int(duration or 0),
                    "recorded_at": time.time(),
                }
                state["completed_media"] = completed_media
                if mismatch_media.pop(item_key, None) is not None:
                    state["mismatch_media"] = mismatch_media
                if stale_requeued:
                    _record_media_observation(
                        state, course_id=course_id, sub_id=sub_id,
                        media_state=MEDIA_STATE_CURRENT,
                        declared_duration_seconds=int(duration or 0),
                    )
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
                if reason == MEDIA_MISMATCH_CODE:
                    # 确定性媒体签名不符（MEDIAMISMATCH-1 闭集码）→ 讲次置
                    # mismatch_deferred：后续窗零入队零配额烧，签名一变自动
                    # 复活转正（设计 §2.2 转移图）；本次尝试仍诚实计 failed，
                    # 此后不再烧——「deferred 不算失败」指门面与后续窗语义。
                    mismatch_media[item_key] = {
                        "declared_duration_seconds": int(duration or 0),
                        "first_detected_at": time.time(),
                        "last_checked_at": time.time(),
                    }
                    state["mismatch_media"] = mismatch_media
                    _record_media_observation(
                        state, course_id=course_id, sub_id=sub_id,
                        media_state=MEDIA_STATE_MISMATCH_DEFERRED,
                        declared_duration_seconds=int(duration or 0),
                    )
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
        _prune_media_ledgers(state)
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
            f"skipped_mismatch={counts['skipped_mismatch']} "
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
