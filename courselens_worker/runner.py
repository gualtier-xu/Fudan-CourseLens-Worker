"""Production entrypoint for one encrypted GitHub Actions job."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .mailbox import IssueMailbox
from .punct import apply_ct_punc, ct_punc_mode
from .protocol import (
    CONTROL_SCHEMA,
    PROTOCOL_VERSION,
    RESULT_SCHEMA,
    PROCESS_CANARY_FIXTURE_BYTES,
    PROCESS_CANARY_FIXTURE_RECORDS,
    PROCESS_CANARY_FIXTURE_SHA256,
    PROCESS_CANARY_PIPELINE,
    PROCESS_CANARY_SCHEMA,
    open_job,
    seal_control,
    seal_result,
    validate_task_id,
)


class WorkerError(RuntimeError):
    pass


_ASR_ERROR_CODES = {
    "authorized media duration probe timed out": "duration_probe_timeout",
    "authorized media duration could not be determined": "duration_probe_failed",
    "configured ASR model directory is incomplete": "model_incomplete",
    "configured ASR token file is missing": "model_tokens_missing",
    "unsupported ASR backend": "unsupported_backend",
    "unsupported subtitle mode": "unsupported_mode",
    "media start is invalid": "invalid_start",
    "media duration is missing or outside the supported range": "invalid_duration",
    "checkpoint subtitle mode does not match the job": "checkpoint_mode_mismatch",
    "authorized media decode timed out": "media_decode_timeout",
    "ffmpeg could not decode the authorized media stream": "media_decode_failed",
    "authorized media request returned HTTP 401": "media_http_401",
    "authorized media request returned HTTP 403": "media_http_403",
    "authorized media request returned HTTP 404": "media_http_404",
    "authorized media request returned HTTP 429": "media_http_429",
    "authorized media request returned HTTP 4xx": "media_http_4xx",
    "authorized media request returned HTTP 5xx": "media_http_5xx",
    "authorized media request returned an unsupported status": "media_http_status_rejected",
    "authorized media response contained HTML": "media_content_html",
    "authorized media response contained JSON": "media_content_json",
    "authorized media signature was rejected": "media_magic_rejected",
    "authorized media is missing a readable MP4 index": "media_index_unreadable",
    "authorized media format was rejected by ffmpeg": "media_format_rejected",
    "authorized media request returned an unsupported redirect": "media_redirect_rejected",
    "authorized media upstream connection failed": "media_connection_failed",
    "authorized media proxy target was rejected": "media_proxy_target_rejected",
    "authorized media proxy request was rejected": "media_proxy_request_rejected",
    "ffmpeg requested an invalid media range": "media_range_invalid",
}


# Checkpoint keys owned by the subtitle stage (ASR chunks and proofread
# windows).  Learning-pack OCR checkpoints re-emit them when a resumed run
# recognizes slides before subtitles, so the reordered stages never discard
# subtitle progress.  The carried counters win the shared completed_chunks /
# total_chunks keys, whose OCR-side copies are informational only:
# process_slides resumes from ocr_completed_items alone.
_SUBTITLE_RESUME_KEYS = (
    "completed_chunks",
    "total_chunks",
    "mode",
    "raw_sensevoice",
    "raw_paraformer",
    "pcm_fingerprint",
    # AS12：rough 源与列车序同属链身份，跨段重排（OCR 先行）时必须随
    # 检查点携带，否则平台链续跑会在守卫处误判为缺键。
    "rough_source",
    "backends",
    "proofread_pairing",
    "proofread_completed_windows",
    "proofread_total_windows",
    "proofread_segments",
    # SUBTITLE-DEEP-1：术语位深校对状态随检查点携带（跨段重排与续跑都
    # 零重复计费；信任门在 term_proofread_segments 内按词级校对完成度判定）。
    "term_proofread_revision",
    "term_proofread_completed_windows",
    "term_proofread_total_windows",
    "term_proofread_terms",
    "term_proofread_segments",
)


# SUBTITLE-DEEP-1：术语位深校对的运行时开关（缺省开启；无术语源时自然跳过）。
_TERM_PROOFREAD_ENV = "COURSELENS_TERM_PROOFREAD"


def _optional_model_dir(name: str) -> Path | None:
    """Optional model directory env (zipformer leg is opt-in)."""
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


def _merged_course_terms(
    payload: dict[str, Any],
    extra: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """payload ``glossary`` ∪ env 术语文件 ∪ 调用方增补（OCR 词表），去重保序。"""
    from .glossary import resolve_course_terms

    return tuple(dict.fromkeys(
        str(term).strip()
        for term in (*resolve_course_terms(payload), *extra)
        if str(term).strip()
    ))


def _payload_memory_terms(payload: dict[str, Any]) -> tuple[str, ...]:
    """payload ``glossary`` 本源术语（课程记忆注入数的事实源；去空去重保序）。"""
    raw = payload.get("glossary")
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(
        str(term).strip() for term in raw if str(term).strip()
    ))


def _env_glossary_terms() -> tuple[str, ...]:
    """env 术语文件本源词（与课程记忆分源计数，绝不冒充记忆注入）。"""
    from .glossary import resolve_course_terms

    return resolve_course_terms({})


def _apply_term_stage(
    value: dict[str, Any],
    *,
    api_key: str,
    payload: dict[str, Any],
    checkpoint_writer: Any,
    warnings: list[str],
    ppt_pages: list[dict[str, Any]] | None = None,
    extra_terms: tuple[str, ...] = (),
    usage_sink: list[dict[str, Any]] | None = None,
) -> None:
    """Term-position deep correction chained after the word-level proofread.

    术语源 = payload 可选 ``glossary``（客户端接线留桩，缺席=旧行为）∪ 环境变量
    术语文件 ∪ OCR 词表（learning_pack）。示例源 = payload 可选 ``examples``
    （课程记忆桩，包B 沉淀后接入）。分歧跨度 = transcribe 返回的
    ``proofread_alternates``（rough 槽位原文）。整层在无术语/无 Key/开关关闭时
    零开销跳过；LLM 失败降级并记闭集警告，词级校对结果绝不因此丢失。检查点
    续跑保留底层 ASR/词级校对状态，术语段零重复计费。``usage_sink``
    （RR-ACCOUNT2-1）非 None 时本段 token 流水并入调用方账本（与词级校对段
    共用一本，任务级 metrics 汇总不重不漏）；缺省维持段内私有账本旧行为。
    """
    if not api_key or os.environ.get(_TERM_PROOFREAD_ENV, "").strip() == "0":
        return
    from .glossary import resolve_course_examples
    from .llm import LLMError, term_proofread_segments

    # v3（总控补充行 2026-09-29）：任意误识位修正+句读标点。无术语表时照跑
    # （标点/常用词路径），terms 缺省为空即可。
    ordered = _merged_course_terms(payload, extra_terms)
    prior = dict(payload.get("checkpoint") or {})
    preserved = {key: item for key, item in prior.items() if key != "stage"}
    audit: list[dict[str, Any]] = []
    usage: list[dict[str, Any]] = usage_sink if usage_sink is not None else []

    def term_checkpoint(term_value: dict[str, Any]) -> None:
        if checkpoint_writer is not None:
            checkpoint_writer({**preserved, **term_value})

    try:
        corrected = term_proofread_segments(
            api_key,
            list(value.get("segments") or []),
            terms=ordered,
            ppt_pages=ppt_pages,
            prior_checkpoint=prior,
            checkpoint=term_checkpoint,
            audit_sink=audit,
            usage_sink=usage,
            alt_segments=value.get("proofread_alternates"),
            course_examples=resolve_course_examples(payload),
        )
    except LLMError:
        warnings.append("term_proofread_degraded")
        return
    value["segments"] = corrected
    if audit:
        # 深校对 diff 审计账（总控补充行 2026-09-29）：位置+改前+改后随结果
        # 透传（result 合同自由键；客户端导入面只读具名键，忽略未知键）。
        value["deep_audit"] = audit
    if usage:
        # SUP3 顺带①：每讲 LLM 成本落账（计数器，零内容）——completion 与
        # reasoning tokens 拆分、缓存命中、总时延。
        value["deep_usage"] = {
            "calls": len(usage),
            "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in usage),
            "completion_tokens": sum(int(item.get("completion_tokens") or 0) for item in usage),
            "reasoning_tokens": sum(int(item.get("reasoning_tokens") or 0) for item in usage),
            "prompt_cache_hit_tokens": sum(
                int(item.get("prompt_cache_hit_tokens") or 0) for item in usage
            ),
            "latency_seconds": round(
                sum(float(item.get("latency_ms") or 0) for item in usage) / 1000, 1
            ),
        }


def _apply_ct_punc_stage(value: dict[str, Any], *, warnings: list[str]) -> None:
    """V4NONTHINK-1 件7：ct-punc 本地标点恢复（env 门控，CTPUNC-DEF-1 起缺省 fill）。

    A 序 fill=仅补无标点 cue 缺口；B 序 full=全部 cue 重标点；off 显式退回。
    内容不等值逐段 fail-closed 保留原文；模型缺席/引擎不可用整讲一次闭集
    回落记账后安静跳过；遥测仅计数。
    """
    if ct_punc_mode() == "off":
        return
    telemetry: list[str] = []
    stats = apply_ct_punc(list(value.get("segments") or []), telemetry=telemetry)
    for line in telemetry:
        print(line, flush=True)
    print(
        f"stage=ct-punc mode={ct_punc_mode()} applied={stats.get('applied', 0)} "
        f"kept={stats.get('kept', 0)}",
        flush=True,
    )


def safe_worker_error_detail(error: BaseException) -> str:
    """Return an optional closed-set reason without importing compute deps."""
    from .source import SourceSecurityError, safe_source_error_code

    if isinstance(error, SourceSecurityError):
        return safe_source_error_code(error)
    if type(error).__name__ == "ASRError":
        return _ASR_ERROR_CODES.get(str(error), "asr_error")
    if type(error).__name__ == "PlatformSessionError":
        value = str(error)
        connection_stage = str(getattr(error, "connection_stage", "") or "")
        if value in {
            "platform_connection_failed", "platform_session_rejected",
        } and connection_stage:
            from .platform_session import _CONNECTION_STAGES

            if connection_stage in _CONNECTION_STAGES:
                return f"{value}_{connection_stage}"
        return value if value in {
            "platform_credentials_missing", "platform_connection_failed",
            "platform_redirect_rejected", "platform_auth_context_missing",
            "platform_auth_method_missing", "platform_key_rejected",
            "platform_auth_failed", "platform_ticket_missing",
            "platform_ticket_rejected", "platform_session_rejected",
            "platform_course_context_missing", "platform_course_request_failed",
            "platform_media_missing", "platform_challenge_required",
        } else "platform_session_failed"
    return ""


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise WorkerError(f"required worker setting is missing: {name}")
    return value


def _progress(stage: str, completed: int, total: int) -> None:
    # Counts and stage identifiers are safe for public logs. No source text,
    # titles, URLs, headers, or exception response bodies are printed.
    print(f"stage={stage} completed={max(0, int(completed))} total={max(0, int(total))}", flush=True)


class SignedProgressPublisher:
    """Publish bounded signed progress without accumulating Issue comments."""

    def __init__(self, publish, *, heartbeat_seconds: float = 15.0):
        self.publish = publish
        self.heartbeat_seconds = max(5.0, float(heartbeat_seconds))
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._latest: dict[str, Any] | None = None
        self._last_sent: dict[str, Any] | None = None
        self._last_sent_at = 0.0
        self._thread = threading.Thread(target=self._heartbeat, name="signed-progress", daemon=True)
        self._thread.start()

    def update(
        self,
        stage: str,
        completed: int | None = None,
        total: int | None = None,
        *,
        status: str = "running",
        error_code: str = "",
        force: bool = False,
    ) -> None:
        completed_value = max(0, int(completed)) if completed is not None else None
        total_value = max(0, int(total)) if total is not None else None
        if completed_value is not None and total_value is not None:
            _progress(stage, completed_value, total_value)
        payload = {
            "stage": str(stage)[:80],
            "status": str(status) if status in {"running", "waiting", "failed", "completed"} else "running",
            "completed": completed_value,
            "total": total_value,
            "error_code": str(error_code)[:80],
        }
        with self._lock:
            self._latest = payload
            now = time.monotonic()
            if force or self._should_send(payload, now):
                self._send_locked(payload, now)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _should_send(self, payload: dict[str, Any], now: float) -> bool:
        prior = self._last_sent
        if prior is None or payload["stage"] != prior.get("stage") or payload["status"] != prior.get("status"):
            return True
        if now - self._last_sent_at >= self.heartbeat_seconds:
            return True
        total = int(payload.get("total") or 0)
        completed = int(payload.get("completed") or 0)
        old_completed = int(prior.get("completed") or 0)
        return bool(total and completed - old_completed >= max(1, int(total * 0.02)))

    def _send_locked(self, payload: dict[str, Any], now: float) -> None:
        try:
            self.publish(dict(payload))
        except Exception as exc:
            print(f"status_update_failed type={type(exc).__name__}", file=sys.stderr, flush=True)
            return
        self._last_sent = dict(payload)
        self._last_sent_at = now

    def _heartbeat(self) -> None:
        while not self._stop.wait(5.0):
            with self._lock:
                if self._latest is None:
                    continue
                now = time.monotonic()
                if now - self._last_sent_at >= self.heartbeat_seconds:
                    self._send_locked(self._latest, now)


# ---- RR-FIX452-1（DIAG-1 主判修复）：检查点评论发布降频/节流/降级 ----
# 每窗检查点照旧逐窗本地落盘（artifact 通道 `if: always()` 上传，续跑粒度
# 不变）；issue 评论才是洪泛面（452282 实测死亡前 70-75 POST/分钟 → 内容
# 创建限流即死）。评论发布节奏：跨阶段/阶段收尾必发 + 阶段内 N 窗（块）一发
# + 90 秒新鲜度兜底；发布线程化（计算循环不等网络），单槽 drop-stale 只保
# 最新待发全量快照。发布失败经 mailbox 有界重试穷尽后降级记账继续跑——
# 丢续跑评论点不丢任务（失败面 checkpoint artifact 逐窗仍在，客户端
# _capture_checkpoint 优先取最新 artifact）。
_CHECKPOINT_PUBLISH_WINDOWS = 12
_CHECKPOINT_PUBLISH_CHUNKS = 4
_CHECKPOINT_PUBLISH_MIN_INTERVAL_SECONDS = 90.0
_CHECKPOINT_PUBLISH_FLUSH_SECONDS = 45.0
_STAGE_PROGRESS_KEYS = {
    "asr": ("completed_chunks", "total_chunks"),
    "summary": ("completed_chunks", "total_chunks"),
    "proofread": ("proofread_completed_windows", "proofread_total_windows"),
    "term_proofread": (
        "term_proofread_completed_windows",
        "term_proofread_total_windows",
    ),
}


def _checkpoint_progress(value: dict[str, Any]) -> tuple[int, int]:
    """(completed, total) of the checkpoint's own stage; (0, 0) when absent."""
    keys = _STAGE_PROGRESS_KEYS.get(str(value.get("stage") or ""))
    if keys is None:
        return 0, 0
    try:
        return int(value.get(keys[0]) or 0), int(value.get(keys[1]) or 0)
    except (TypeError, ValueError):
        return 0, 0


class _CheckpointCadence:
    """Decides which per-stage checkpoints earn an issue-comment publish.

    主线程独占调用；`now` 一律由调用方传 time.monotonic() 便于测试。
    """

    def __init__(self) -> None:
        self.stage: str | None = None
        self.completed = -1
        self.published_at = 0.0

    def _window_size(self, stage: str) -> int:
        return (
            _CHECKPOINT_PUBLISH_CHUNKS
            if stage in {"asr", "summary"}
            else _CHECKPOINT_PUBLISH_WINDOWS
        )

    def should_publish(self, value: dict[str, Any], now: float) -> bool:
        stage = str(value.get("stage") or "work")
        completed, total = _checkpoint_progress(value)
        if self.stage != stage:
            return True
        if total and completed >= total:
            return True
        if completed - self.completed >= self._window_size(stage):
            return True
        if (
            completed > self.completed
            and now - self.published_at >= _CHECKPOINT_PUBLISH_MIN_INTERVAL_SECONDS
        ):
            return True
        return False

    def mark(self, value: dict[str, Any], now: float) -> None:
        self.stage = str(value.get("stage") or "work")
        self.completed, _ = _checkpoint_progress(value)
        self.published_at = now


class _CheckpointPublisher:
    """Background serialized publisher for checkpoint comments.

    单槽 drop-stale：发布落后于计算时只保留最新待发全量快照（被顶掉的计数
    进 ``stats["superseded"]``；本地 artifact 从不顶掉）。发布失败按降级记账
    继续跑（mailbox 已出闭集 retry/failed 行；此处补 checkpoint_publish_
    degraded 行）。``close`` 汇总一行闭集计数后停线程。
    """

    def __init__(
        self,
        publish,
        *,
        flush_timeout: float = _CHECKPOINT_PUBLISH_FLUSH_SECONDS,
    ):
        self._publish = publish
        self._flush_timeout = max(0.0, float(flush_timeout))
        self._cond = threading.Condition()
        self._pending: dict[str, Any] | None = None
        self._closing = False
        self.stats = {"published": 0, "degraded": 0, "superseded": 0, "skipped": 0}
        self._thread = threading.Thread(
            target=self._loop, name="checkpoint-publish", daemon=True
        )
        self._thread.start()

    def submit(self, value: dict[str, Any]) -> None:
        with self._cond:
            if self._pending is not None:
                self.stats["superseded"] += 1
            self._pending = value
            self._cond.notify_all()

    def close(self, *, flush_timeout: float | None = None) -> None:
        timeout = (
            self._flush_timeout if flush_timeout is None else max(0.0, float(flush_timeout))
        )
        deadline = time.monotonic() + timeout
        with self._cond:
            self._closing = True
            if timeout <= 0:
                # 零冲刷（失败路径）：弃待发件——在途一发完成后即停。
                self._pending = None
            self._cond.notify_all()
            while self._pending is not None and time.monotonic() < deadline:
                self._cond.wait(timeout=0.2)
        self._thread.join(timeout=2)
        print(
            f"checkpoint_publish published={self.stats['published']} "
            f"degraded={self.stats['degraded']} "
            f"superseded={self.stats['superseded']} skipped={self.stats['skipped']}",
            flush=True,
        )

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._closing:
                    self._cond.wait(timeout=0.5)
                value, self._pending = self._pending, None
            if value is not None:
                try:
                    self._publish(value)
                    self.stats["published"] += 1
                except Exception as exc:
                    self.stats["degraded"] += 1
                    name = type(exc).__name__
                    if name == "MailboxError":
                        print("checkpoint_publish_degraded stage=checkpoint", flush=True)
                    else:
                        print(
                            f"checkpoint_publish_degraded type={name}", flush=True
                        )
                with self._cond:
                    self._cond.notify_all()
                continue
            with self._cond:
                if self._closing:
                    return


def _process_materialized_job(
    job: dict[str, Any],
    *,
    checkpoint_writer=None,
    progress_callback=None,
) -> dict[str, Any]:
    progress = progress_callback or _progress
    kind = str(job["job_kind"])
    started = time.monotonic()
    warnings: list[str] = []
    if kind == "echo":
        outputs = {"echo": {"ok": True, "protocol_version": PROTOCOL_VERSION}}
        metrics = {"elapsed_seconds": round(time.monotonic() - started, 3)}
    elif kind == "process_canary":
        # This fixture is deliberately runner-owned and constant.  Neither the
        # encrypted request nor any user/account/course value can influence it.
        workflow_profile = _required("COURSELENS_WORKFLOW_PROFILE")
        if workflow_profile != "process-v1":
            raise WorkerError("process canary workflow profile is invalid")
        fixture = (
            b'{"records":[[0,1,0,-1],[1,0,-1,0],[0,-1,0,1]],'
            b'"schema":"fixture.v1"}'
        )
        fixture_sha256 = hashlib.sha256(fixture).hexdigest()
        if (
            len(fixture) != PROCESS_CANARY_FIXTURE_BYTES
            or fixture_sha256 != PROCESS_CANARY_FIXTURE_SHA256
        ):
            raise WorkerError("process canary fixture integrity failed")
        outputs = {
            "process_canary": {
                "schema": PROCESS_CANARY_SCHEMA,
                "fixture_sha256": fixture_sha256,
                "fixture_bytes": len(fixture),
                "fixture_records": PROCESS_CANARY_FIXTURE_RECORDS,
                "worker_commit": _required("GITHUB_SHA").lower(),
                "workflow_profile": workflow_profile,
            }
        }
        metrics = {
            "synthetic_bytes": len(fixture),
            "synthetic_records": PROCESS_CANARY_FIXTURE_RECORDS,
        }
    elif kind == "subtitle":
        from .asr import transcribe
        from .formats import to_srt, to_vtt
        from .llm import proofread_segments

        payload = dict(job.get("payload") or {})
        secrets = dict(job.get("secrets") or {})
        api_key = str(secrets.get("deepseek_api_key") or "")
        # automatic 策略：仅在配置了 DeepSeek Key 时提供校对提供方；
        # 缺 Key 即走非 AI 回退，由 asr.transcribe 依据 proofread 是否为 None 分派。
        # RR-ACCOUNT2-1：字幕链统一账本（词级校对窗 + term 深校对段共用
        # 一本），任务级 metrics 汇总真实 token（纯观测计数）。
        subtitle_usage: list[dict[str, Any]] = []
        value = transcribe(
            job,
            sensevoice_dir=Path(_required("SENSEVOICE_MODEL_DIR")),
            paraformer_dir=Path(_required("PARAFORMER_MODEL_DIR")),
            zipformer_dir=_optional_model_dir("ZIPFORMER_MODEL_DIR"),
            hotwords=_merged_course_terms(payload),
            proofread=(
                (lambda rough, refined, prior, write: proofread_segments(
                    api_key,
                    rough,
                    refined,
                    prior_checkpoint=prior,
                    checkpoint=write,
                    usage_sink=subtitle_usage,
                )) if api_key else None
            ),
            progress=progress,
            checkpoint=checkpoint_writer,
        )
        _apply_term_stage(
            value,
            api_key=api_key,
            payload=payload,
            checkpoint_writer=checkpoint_writer,
            warnings=warnings,
            usage_sink=subtitle_usage,
        )
        _apply_ct_punc_stage(value, warnings=warnings)
        outputs = {
            "subtitle": {
                "mode": value["mode"],
                "segments": value["segments"],
                "srt": to_srt(value["segments"]),
                "vtt": to_vtt(value["segments"]),
                # raw 段键随 SUBTITLE_BACKENDS 序列泛化（raw_<backend>），
                # 默认链仍是 raw_sensevoice + raw_paraformer，客户端合同不变。
                **{key: value[key] for key in value if key.startswith("raw_")},
                **({"deep_audit": value["deep_audit"]} if value.get("deep_audit") else {}),
                **({"deep_usage": value["deep_usage"]} if value.get("deep_usage") else {}),
            }
        }
        metrics = value["metrics"]
        # RR-ACCOUNT2-1：字幕任务的真实 token 随结果上报（prompt+completion
        # 与 deep_usage 同口径），月账不再漏记字幕任务族。
        metrics["deepseek_tokens"] = metrics.get("deepseek_tokens", 0) + sum(
            max(0, int(record.get("prompt_tokens") or 0))
            + max(0, int(record.get("completion_tokens") or 0))
            for record in subtitle_usage
        )
    elif kind in {"summary", "chapters"}:
        from .course_knowledge import normalize_evidence_packet
        from .lecture_ir import build_lecture_ir
        from .llm import LLMError, create_summary
        from .ocr import process_slides

        payload = dict(job.get("payload") or {})
        transcript = list(payload.get("transcript") or [])
        slides = list(payload.get("slides") or [])
        prior = dict(payload.get("checkpoint") or {})
        pages, slides_skipped = process_slides(
            slides,
            progress=progress,
            prior_checkpoint=prior,
            checkpoint=checkpoint_writer,
        ) if slides else (list(prior.get("ppt_pages") or []), dict(prior.get("ppt_skipped") or {}))

        def summary_checkpoint(value: dict[str, Any]) -> None:
            if checkpoint_writer is not None:
                checkpoint_writer({
                    "ocr_completed_items": len(slides),
                    "ppt_pages": pages,
                    "ppt_skipped": slides_skipped,
                    **value,
                })

        # 客户端可选的 evidence packet：校验通过才投喂；整包不可用时如实记警告
        # 并按旧 summary 路径降级，绝不让坏知识落地。
        knowledge = normalize_evidence_packet(
            payload.get("evidence_packet"),
            course_id=str(payload.get("course_id") or ""),
            sub_id=str(payload.get("sub_id") or ""),
            transcript=transcript,
            ppt_pages=pages,
        )
        packet = knowledge if knowledge.get("usable") else None
        if payload.get("evidence_packet") is not None and packet is None:
            warnings.append("evidence_packet_rejected")
        # 只有真的要传多源证据时才带新参数：老 job（无 packet/无 course_context）
        # 的调用保持与历史逐字相同，既有的严格测试替身不会被签名变化打破。
        summary_args: dict[str, Any] = {}
        if packet is not None:
            summary_args["evidence_packet"] = packet
        if payload.get("course_context") is not None:
            summary_args["course_context"] = payload.get("course_context")
        glossary = _merged_course_terms(payload)
        if glossary:
            # V4NONTHINK-1 件6：摘要窗口/合并输入携带课程术语表（笔记写法保险）。
            summary_args["glossary"] = glossary
        # RR-P6MEM-1 + QA-SWEEP-1 P1-6：实报只数课程记忆注入（payload glossary
        # 本源）；env 术语文件词单独计数——并集口径会让 env 词冒充「已应用
        # 课程记忆 N 条」的学生可见标注，宁缺毋滥。
        memory_terms = _payload_memory_terms(payload)
        if memory_terms:
            metrics["course_memory_terms"] = len(memory_terms)
        env_terms = _env_glossary_terms()
        if env_terms:
            metrics["env_glossary_terms"] = len(env_terms)
        # RR-ACCOUNT2-1：摘要链（窗口+合并，含失败尝试）统一账本，任务级
        # metrics 汇总真实 token（纯观测计数）。
        summary_usage: list[dict[str, Any]] = []
        try:
            summary = create_summary(
                str(dict(job.get("secrets") or {}).get("deepseek_api_key") or ""),
                title=str(payload.get("title") or ""),
                transcript=transcript,
                ppt_pages=pages,
                prior_checkpoint=prior,
                checkpoint=summary_checkpoint,
                usage_sink=summary_usage,
                **summary_args,
            )
        except LLMError:
            # 夜10-C 第七波②：远端 LLM 段降级——runner 出口对 DeepSeek 不可达
            # 等闭集失败不再整单失败，改回 llm_pending 回执；客户端凭本地可达
            # 的 key 领回执行同一总结链后按同一导入面落库（UI/消耗计数一致）。
            warnings.append("llm_pending_remote_failed")
            print(f"task={job['task_id']} stage=summary-llm-pending", flush=True)
            return {
                "schema": RESULT_SCHEMA,
                "protocol_version": PROTOCOL_VERSION,
                "task_id": job["task_id"],
                "job_kind": kind,
                "input_hash": job["input_hash"],
                "pipeline_fingerprint": (
                    PROCESS_CANARY_PIPELINE if kind == "process_canary"
                    else str(dict(job.get("pipeline") or {}).get("version") or "v2")
                ),
                "status": "llm_pending",
                "outputs": {"llm_pending": {
                    "title": str(payload.get("title") or ""),
                    "transcript": transcript,
                    "ppt_pages": pages,
                    "evidence_packet": packet,
                    "course_context": (
                        payload.get("course_context")
                        if isinstance(payload.get("course_context"), dict) else None
                    ),
                    "reason_code": "llm_remote_failed",
                }},
                "metrics": {
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "transcript_segments": len(transcript),
                    "ppt_pages": len(pages),
                },
                "warnings": warnings,
            }
        outputs = {"ppt_pages": pages}
        if kind == "chapters":
            outputs["chapters"] = list(summary.get("chapters") or [])
        else:
            outputs["summary"] = summary
        # Additive, evidence-grounded Lecture IR view over the same
        # transcript/chapters/pages; it changes no pre-existing output.
        outputs["lecture_ir"] = build_lecture_ir(
            transcript=transcript,
            chapters=list(summary.get("chapters") or []),
            ppt_pages=pages,
            knowledge_points=list(summary.get("knowledge_points") or []),
            evidence_packet=packet,
        )
        metrics = {
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "transcript_segments": len(transcript),
            "ppt_pages": len(pages),
        }
        # RR-ACCOUNT2-1：总结任务的真实 token 随结果上报（prompt+completion
        # 与 answer 同口径），月账不再漏记总结任务族。llm_pending 降级回执
        # 不落远端账——该面由客户端本地完成后按既有绝对值口径记账。
        metrics["deepseek_tokens"] = metrics.get("deepseek_tokens", 0) + sum(
            max(0, int(record.get("prompt_tokens") or 0))
            + max(0, int(record.get("completion_tokens") or 0))
            for record in summary_usage
        )
        if packet is not None:
            metrics["evidence_items"] = len(packet.get("items") or [])
            metrics["knowledge_points"] = len(list(summary.get("knowledge_points") or []))
            metrics["citations_rejected"] = int(summary.get("citations_rejected") or 0)
        if slides_skipped:
            metrics["slides_skipped"] = slides_skipped
            warnings.append("slides_skipped")
    elif kind == "learning_pack":
        from .asr import transcribe
        from .course_knowledge import normalize_evidence_packet
        from .formats import to_srt, to_vtt
        from .lecture_ir import build_lecture_ir
        from .glossary import build_glossary
        from .llm import LLMError, answer_question, create_summary, drain_call_log, proofread_segments
        from .ocr import process_slides

        payload = dict(job.get("payload") or {})
        requested = set(job.get("requested_outputs") or [])
        secrets = dict(job.get("secrets") or {})
        api_key = str(secrets.get("deepseek_api_key") or "")
        outputs: dict[str, Any] = {}
        metrics = {}
        transcript = list(payload.get("transcript") or [])
        prior = dict(payload.get("checkpoint") or {})
        slides = list(payload.get("slides") or [])
        # Slides are recognized once, before Standard proofreading, so the
        # proofreader can use the active slide text as bounded terminology
        # context; the same pages then serve the requested OCR/summary/chapters
        # outputs without a second OCR pass.
        wants_slides = bool(slides) and bool(requested.intersection({"ocr", "summary", "chapters"}))
        if wants_slides:
            def ocr_checkpoint(value: dict[str, Any]) -> None:
                if checkpoint_writer is not None:
                    subtitle_state = {
                        key: prior[key] for key in _SUBTITLE_RESUME_KEYS if key in prior
                    }
                    checkpoint_writer({**value, **subtitle_state})

            pages, slides_skipped = process_slides(
                slides,
                progress=progress,
                prior_checkpoint=prior,
                checkpoint=ocr_checkpoint,
            )
        else:
            pages, slides_skipped = (
                list(prior.get("ppt_pages") or []), dict(prior.get("ppt_skipped") or {})
            )
        ocr_fields: dict[str, Any] = {
            "ocr_completed_items": len(slides),
            "ppt_pages": pages,
            "ppt_skipped": slides_skipped,
        } if wants_slides else {}
        if "subtitle" in requested:
            # RR-ACCOUNT2-1：字幕链统一账本（词级校对窗 + term 深校对段共
            # 用一本），任务级 metrics 汇总真实 token（纯观测计数）。
            subtitle_usage: list[dict[str, Any]] = []

            def subtitle_checkpoint(value: dict[str, Any]) -> None:
                if checkpoint_writer is not None:
                    checkpoint_writer({**ocr_fields, **value})

            def proofread_with_slides(rough, refined, saved, write):
                def write_with_ocr(proofread_value: dict[str, Any]) -> None:
                    write({**ocr_fields, **proofread_value})

                return proofread_segments(
                    api_key,
                    rough,
                    refined,
                    ppt_pages=pages if wants_slides else None,
                    prior_checkpoint=saved,
                    checkpoint=write_with_ocr,
                    glossary=build_glossary(pages, str(payload.get("title") or "")),
                    usage_sink=subtitle_usage,
                )

            value = transcribe(
                job,
                sensevoice_dir=Path(_required("SENSEVOICE_MODEL_DIR")),
                paraformer_dir=Path(_required("PARAFORMER_MODEL_DIR")),
                zipformer_dir=_optional_model_dir("ZIPFORMER_MODEL_DIR"),
                hotwords=_merged_course_terms(
                    payload, build_glossary(pages, str(payload.get("title") or "")),
                ) if wants_slides else _merged_course_terms(payload),
                proofread=proofread_with_slides if api_key else None,
                progress=progress,
                checkpoint=subtitle_checkpoint,
            )
            # G7：ASR 段产出的闭集警告（如 proofread_degraded）并入结果警告，
            # 随既有 result_notices 通道上屏，绝不静默丢弃。
            for _warning in value.get("warnings") or []:
                if _warning not in warnings:
                    warnings.append(_warning)
            _apply_term_stage(
                value,
                api_key=api_key,
                payload=payload,
                checkpoint_writer=checkpoint_writer,
                warnings=warnings,
                ppt_pages=pages if wants_slides else None,
                extra_terms=build_glossary(pages, str(payload.get("title") or "")),
                usage_sink=subtitle_usage,
            )
            _apply_ct_punc_stage(value, warnings=warnings)
            # RR-ACCOUNT2-1：字幕任务的真实 token 随结果上报（prompt+completion
            # 与 deep_usage/answer 同口径；增量聚合防组合任务分支互相覆盖）。
            metrics["deepseek_tokens"] = metrics.get("deepseek_tokens", 0) + sum(
                max(0, int(record.get("prompt_tokens") or 0))
                + max(0, int(record.get("completion_tokens") or 0))
                for record in subtitle_usage
            )
            transcript = value["segments"]
            outputs["subtitle"] = {
                "mode": value["mode"],
                "segments": transcript,
                "srt": to_srt(transcript),
                "vtt": to_vtt(transcript),
                **{key: value[key] for key in value if key.startswith("raw_")},
                **({"deep_audit": value["deep_audit"]} if value.get("deep_audit") else {}),
                **({"deep_usage": value["deep_usage"]} if value.get("deep_usage") else {}),
            }
            metrics["subtitle"] = value["metrics"]
        if "answer" in requested:
            # RR-P6MEM-1：payload 可选课程记忆术语表进问答链（读侧加性通道，
            # 缺席=旧行为逐字）；实报注入数供客户端标注宁缺毋滥。
            answer_args: dict[str, Any] = {}
            answer_terms = _merged_course_terms(payload)
            if answer_terms:
                answer_args["course_terms"] = answer_terms
            # QA-SWEEP-1 P1-6：同摘要面——实报只数课程记忆本源（payload
            # glossary），env 术语文件词单独计数，不让并集冒充记忆注入数。
            memory_terms = _payload_memory_terms(payload)
            if memory_terms:
                metrics["course_memory_terms"] = len(memory_terms)
            env_terms = _env_glossary_terms()
            if env_terms:
                metrics["env_glossary_terms"] = len(env_terms)
            outputs["answer"] = answer_question(
                api_key,
                query=str(payload.get("query") or ""),
                evidence=list(payload.get("evidence") or []),
                **answer_args,
            )
            metrics["evidence_count"] = len(payload.get("evidence") or [])
            # AS6/RR-PARK-1 P2：解释/解答的真实消耗随结果上报（纯观测计数，
            # total=prompt+completion 与 deep_usage 同口径）。drain 恰在本
            # 分支：字幕深校对已自行收口进 usage_sink，摘要链在其后自
            # drain，互不重叠；RR-ACCOUNT2-1 起各分支增量聚合，组合任务
            # （字幕+解答+摘要同单）不再互相覆盖。
            metrics["deepseek_tokens"] = metrics.get("deepseek_tokens", 0) + sum(
                max(0, int(record.get("prompt_tokens") or 0))
                + max(0, int(record.get("completion_tokens") or 0))
                for record in drain_call_log()
            )
        if "ocr" in requested:
            outputs["ppt_pages"] = pages
        if slides_skipped:
            metrics["slides_skipped"] = slides_skipped
            warnings.append("slides_skipped")
        if requested.intersection({"summary", "chapters"}):
            def summary_checkpoint(value: dict[str, Any]) -> None:
                # Merge the same OCR fields the summary job keeps, so a
                # learning_pack resume never reprocesses completed slides.
                if checkpoint_writer is not None:
                    checkpoint_writer({
                        "ocr_completed_items": len(slides),
                        "ppt_pages": pages,
                        "ppt_skipped": slides_skipped,
                        **value,
                    })

            knowledge = normalize_evidence_packet(
                payload.get("evidence_packet"),
                course_id=str(payload.get("course_id") or ""),
                sub_id=str(payload.get("sub_id") or ""),
                transcript=transcript,
                ppt_pages=pages,
            )
            packet = knowledge if knowledge.get("usable") else None
            if payload.get("evidence_packet") is not None and packet is None:
                warnings.append("evidence_packet_rejected")
            summary_args: dict[str, Any] = {}
            if packet is not None:
                summary_args["evidence_packet"] = packet
            if payload.get("course_context") is not None:
                summary_args["course_context"] = payload.get("course_context")
            pack_terms = (
                _merged_course_terms(
                    payload, build_glossary(pages, str(payload.get("title") or ""))
                )
                if wants_slides else _merged_course_terms(payload)
            )
            if pack_terms:
                # V4NONTHINK-1 件6：learning_pack 摘要同享术语表注入（含 OCR 词表）。
                summary_args["glossary"] = pack_terms
            # RR-ACCOUNT2-1：摘要链（窗口+合并，含失败尝试）统一账本，任务级
            # metrics 汇总真实 token（纯观测计数）。
            summary_usage: list[dict[str, Any]] = []
            try:
                summary = create_summary(
                    api_key,
                    title=str(payload.get("title") or ""),
                    transcript=transcript,
                    ppt_pages=pages,
                    prior_checkpoint=prior,
                    checkpoint=summary_checkpoint,
                    usage_sink=summary_usage,
                    **summary_args,
                )
            except LLMError:
                # 夜10-C 第九波任务2：learning_pack 分支与 summary 分支同享
                # llm_pending 降级——LLM 段失败不再整单失败，OCR/转写产物
                # 随 llm_pending 回执交还客户端领回执行。
                warnings.append("llm_pending_remote_failed")
                print(f"task={job['task_id']} stage=summary-llm-pending", flush=True)
                return {
                    "schema": RESULT_SCHEMA,
                    "protocol_version": PROTOCOL_VERSION,
                    "task_id": job["task_id"],
                    "job_kind": kind,
                    "input_hash": job["input_hash"],
                    "pipeline_fingerprint": str(
                        dict(job.get("pipeline") or {}).get("version") or "v2"
                    ),
                    "status": "llm_pending",
                    "outputs": {"llm_pending": {
                        "title": str(payload.get("title") or ""),
                        "transcript": transcript,
                        "ppt_pages": pages,
                        "evidence_packet": packet,
                        "course_context": (
                            payload.get("course_context")
                            if isinstance(payload.get("course_context"), dict) else None
                        ),
                        "reason_code": "llm_remote_failed",
                        "job_kind": kind,
                    }},
                    "metrics": {
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "transcript_segments": len(transcript),
                        "ppt_pages": len(pages),
                    },
                    "warnings": warnings,
                }
            # RR-ACCOUNT2-1：摘要任务的真实 token 随结果上报（prompt+completion
            # 与 answer 同口径；增量聚合防组合任务分支互相覆盖）。llm_pending
            # 降级回执不落远端账——该面由客户端本地完成后按既有绝对值口径记账。
            metrics["deepseek_tokens"] = metrics.get("deepseek_tokens", 0) + sum(
                max(0, int(record.get("prompt_tokens") or 0))
                + max(0, int(record.get("completion_tokens") or 0))
                for record in summary_usage
            )
            if "summary" in requested:
                outputs["summary"] = summary
            if "chapters" in requested:
                outputs["chapters"] = list(summary.get("chapters") or [])
            # Same additive Lecture IR view as the summary job, over the
            # transcript this pack actually produced.
            outputs["lecture_ir"] = build_lecture_ir(
                transcript=transcript,
                chapters=list(summary.get("chapters") or []),
                ppt_pages=pages,
                knowledge_points=list(summary.get("knowledge_points") or []),
                evidence_packet=packet,
            )
            if packet is not None:
                metrics["evidence_items"] = len(packet.get("items") or [])
                metrics["knowledge_points"] = len(list(summary.get("knowledge_points") or []))
                metrics["citations_rejected"] = int(summary.get("citations_rejected") or 0)
        metrics["elapsed_seconds"] = round(time.monotonic() - started, 3)
    else:
        raise WorkerError("unsupported job kind")
    return {
        "schema": RESULT_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "task_id": job["task_id"],
        "job_kind": kind,
        "input_hash": job["input_hash"],
        "pipeline_fingerprint": (
            PROCESS_CANARY_PIPELINE if kind == "process_canary"
            else str(dict(job.get("pipeline") or {}).get("version") or "v2")
        ),
        "status": "completed",
        "outputs": outputs,
        "metrics": metrics,
        "warnings": warnings,
    }


def process_job(
    job: dict[str, Any],
    *,
    checkpoint_writer=None,
    progress_callback=None,
) -> dict[str, Any]:
    if dict(job.get("payload") or {}).get("source_session"):
        from .platform_session import materialize_job_sources
        job = materialize_job_sources(job)
    payload = dict(job.get("payload") or {})
    close_source_session = payload.pop("_close_source_session", None)
    job = {**job, "payload": payload}
    try:
        return _process_materialized_job(
            job,
            checkpoint_writer=checkpoint_writer,
            progress_callback=progress_callback,
        )
    finally:
        if callable(close_source_session):
            close_source_session()


def run() -> int:
    task_id = validate_task_id(_required("COURSELENS_TASK_ID"))
    if _required("COURSELENS_PROTOCOL_VERSION") != PROTOCOL_VERSION:
        raise WorkerError("workflow requested an unsupported protocol version")
    mailbox = IssueMailbox(_required("PRIVATE_JOB_REPO"), _required("PRIVATE_JOB_REPO_TOKEN"))
    print(f"task={task_id} stage=waiting_for_encrypted_job", flush=True)
    envelope = mailbox.wait(task_id, timeout_seconds=600)
    job = open_job(envelope, _required("WORKER_INPUT_PRIVATE_KEY"))
    if job["task_id"] != task_id:
        raise WorkerError("mailbox task id does not match the workflow")
    print(f"task={task_id} stage=processing kind={job['job_kind']}", flush=True)
    checkpoint_root = Path(".work") / "checkpoints"
    control_sequence = 0
    control_lock = threading.RLock()

    def publish_control(kind: str, payload: dict[str, Any], *, mutable: bool = False) -> None:
        nonlocal control_sequence
        # RR-FIX452-1：序号分配与签名持锁，评论网络 I/O 放锁外——检查点信封
        # 分件 POST 可达分钟级（≤6 POST/分钟节流），持锁会堵死进度心跳 PATCH。
        with control_lock:
            control_sequence += 1
            sequence = control_sequence
            value = {
                "schema": CONTROL_SCHEMA,
                "protocol_version": PROTOCOL_VERSION,
                "task_id": job["task_id"],
                "input_hash": job["input_hash"],
                "sequence": sequence,
                "control_kind": kind,
                "created_at": time.time(),
                "payload": payload,
            }
        sealed_control = seal_control(
            value,
            str(job["result_public_key"]),
            _required("WORKER_SIGNING_PRIVATE_KEY"),
        )
        if mutable:
            mailbox.publish_status(sequence, sealed_control)
        else:
            mailbox.publish_control(sequence, sealed_control)

    checkpoint_publisher = _CheckpointPublisher(
        lambda checkpoint_value: publish_control("checkpoint", {"checkpoint": checkpoint_value})
    )
    checkpoint_cadence = _CheckpointCadence()

    def write_checkpoint(value: dict[str, Any]) -> None:
        checkpoint_result = {
            "schema": RESULT_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "task_id": job["task_id"],
            "job_kind": job["job_kind"],
            "input_hash": job["input_hash"],
            "pipeline_fingerprint": str(dict(job.get("pipeline") or {}).get("version") or "v2"),
            "status": "checkpoint",
            "outputs": {"checkpoint": value},
            "metrics": {},
            "warnings": [],
        }
        sealed_checkpoint = seal_result(
            checkpoint_result,
            str(job["result_public_key"]),
            _required("WORKER_SIGNING_PRIVATE_KEY"),
        )
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        stage = "".join(
            character for character in str(value.get("stage") or "work")
            if character.isascii() and (character.isalnum() or character == "-")
        ) or "work"
        destination = checkpoint_root / (
            f"checkpoint-{time.time_ns():020d}-{stage}-"
            f"{int(value.get('completed_chunks') or 0):04d}.box.json"
        )
        temporary_checkpoint = destination.with_suffix(destination.suffix + ".tmp")
        temporary_checkpoint.write_text(
            json.dumps(sealed_checkpoint, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary_checkpoint, destination)
        now = time.monotonic()
        if checkpoint_cadence.should_publish(value, now):
            checkpoint_cadence.mark(value, now)
            # JSON 往返脱钩调用方的可变结构（窗口循环随后继续扩展转录列表）。
            checkpoint_publisher.submit(json.loads(json.dumps(value)))
        else:
            checkpoint_publisher.stats["skipped"] += 1

    publisher = SignedProgressPublisher(
        lambda payload: publish_control("progress", payload, mutable=True)
    )
    publisher.update("remote_compute", status="waiting", force=True)
    try:
        result = process_job(
            job,
            checkpoint_writer=write_checkpoint,
            progress_callback=lambda stage, completed, total: publisher.update(
                stage, completed, total, status="running"
            ),
        )
    except Exception as exc:
        try:
            error_code = safe_worker_error_detail(exc) or "worker_failed"
        except Exception:
            error_code = "worker_failed"
        # 失败面不等检查点评论收尾（artifact 逐窗在案，客户端失败路径取
        # artifact），立即让 failed 状态 PATCH 上行。
        checkpoint_publisher.close(flush_timeout=0.0)
        publisher.update(
            "remote_compute", status="failed",
            error_code=error_code, force=True,
        )
        raise
    finally:
        publisher.close()
    sealed = seal_result(
        result,
        str(job["result_public_key"]),
        _required("WORKER_SIGNING_PRIVATE_KEY"),
    )
    output = Path(os.environ.get("COURSELENS_RESULT_PATH", "result.box.json"))
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(sealed, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, output)
    publish_control("progress", {
        "stage": "result", "status": "completed", "completed": 1,
        "total": 1, "error_code": "",
    }, mutable=True)
    checkpoint_publisher.close()
    print(f"task={task_id} stage=result_ready", flush=True)
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:
        # Avoid str(exc): networking libraries often embed an authorized URL.
        try:
            reason = safe_worker_error_detail(exc)
            detail = f" reason={reason}" if reason else ""
        except Exception:
            detail = ""
        print(
            f"worker_failed type={type(exc).__name__}{detail}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
