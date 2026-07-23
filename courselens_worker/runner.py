"""Production entrypoint for one encrypted GitHub Actions job."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .mailbox import IssueMailbox
from .protocol import (
    CONTROL_SCHEMA,
    PROTOCOL_VERSION,
    RESULT_SCHEMA,
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
            "platform_media_missing",
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


def process_job(
    job: dict[str, Any],
    *,
    checkpoint_writer=None,
    progress_callback=None,
) -> dict[str, Any]:
    progress = progress_callback or _progress
    if dict(job.get("payload") or {}).get("source_session"):
        from .platform_session import materialize_job_sources
        job = materialize_job_sources(job)
    kind = str(job["job_kind"])
    started = time.monotonic()
    if kind == "echo":
        outputs = {"echo": {"ok": True, "protocol_version": PROTOCOL_VERSION}}
        metrics = {"elapsed_seconds": round(time.monotonic() - started, 3)}
    elif kind == "subtitle":
        from .asr import transcribe
        from .formats import to_srt, to_vtt
        from .llm import proofread_segments

        secrets = dict(job.get("secrets") or {})
        api_key = str(secrets.get("deepseek_api_key") or "")
        value = transcribe(
            job,
            sensevoice_dir=Path(_required("SENSEVOICE_MODEL_DIR")),
            firered_dir=Path(_required("FIRERED_MODEL_DIR")),
            proofread=(lambda sense, fire, prior, write: proofread_segments(
                api_key,
                sense,
                fire,
                prior_checkpoint=prior,
                checkpoint=write,
            )),
            progress=progress,
            checkpoint=checkpoint_writer,
        )
        outputs = {
            "subtitle": {
                "mode": value["mode"],
                "segments": value["segments"],
                "srt": to_srt(value["segments"]),
                "vtt": to_vtt(value["segments"]),
                "raw_sensevoice": value["raw_sensevoice"],
                "raw_firered": value["raw_firered"],
            }
        }
        metrics = value["metrics"]
    elif kind in {"summary", "chapters"}:
        from .llm import create_summary
        from .ocr import process_slides

        payload = dict(job.get("payload") or {})
        transcript = list(payload.get("transcript") or [])
        slides = list(payload.get("slides") or [])
        prior = dict(payload.get("checkpoint") or {})
        pages = process_slides(
            slides,
            progress=progress,
            prior_checkpoint=prior,
            checkpoint=checkpoint_writer,
        ) if slides else list(prior.get("ppt_pages") or [])

        def summary_checkpoint(value: dict[str, Any]) -> None:
            if checkpoint_writer is not None:
                checkpoint_writer({
                    "ocr_completed_items": len(slides),
                    "ppt_pages": pages,
                    **value,
                })

        summary = create_summary(
            str(dict(job.get("secrets") or {}).get("deepseek_api_key") or ""),
            title=str(payload.get("title") or ""),
            transcript=transcript,
            ppt_pages=pages,
            prior_checkpoint=prior,
            checkpoint=summary_checkpoint,
        )
        outputs = {"ppt_pages": pages}
        if kind == "chapters":
            outputs["chapters"] = list(summary.get("chapters") or [])
        else:
            outputs["summary"] = summary
        metrics = {
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "transcript_segments": len(transcript),
            "ppt_pages": len(pages),
        }
    elif kind == "learning_pack":
        from .asr import transcribe
        from .formats import to_srt, to_vtt
        from .llm import answer_question, create_summary, proofread_segments
        from .ocr import process_slides

        payload = dict(job.get("payload") or {})
        requested = set(job.get("requested_outputs") or [])
        secrets = dict(job.get("secrets") or {})
        api_key = str(secrets.get("deepseek_api_key") or "")
        outputs: dict[str, Any] = {}
        metrics = {}
        transcript = list(payload.get("transcript") or [])
        prior = dict(payload.get("checkpoint") or {})
        if "subtitle" in requested:
            value = transcribe(
                job,
                sensevoice_dir=Path(_required("SENSEVOICE_MODEL_DIR")),
                firered_dir=Path(_required("FIRERED_MODEL_DIR")),
                proofread=(lambda sense, fire, saved, write: proofread_segments(
                    api_key, sense, fire, prior_checkpoint=saved, checkpoint=write
                )),
                progress=progress,
                checkpoint=checkpoint_writer,
            )
            transcript = value["segments"]
            outputs["subtitle"] = {
                "mode": value["mode"],
                "segments": transcript,
                "srt": to_srt(transcript),
                "vtt": to_vtt(transcript),
                "raw_sensevoice": value["raw_sensevoice"],
                "raw_firered": value["raw_firered"],
            }
            metrics["subtitle"] = value["metrics"]
        if "answer" in requested:
            outputs["answer"] = answer_question(
                api_key,
                query=str(payload.get("query") or ""),
                evidence=list(payload.get("evidence") or []),
            )
            metrics["evidence_count"] = len(payload.get("evidence") or [])
        slides = list(payload.get("slides") or [])
        pages = process_slides(
            slides,
            progress=progress,
            prior_checkpoint=prior,
            checkpoint=checkpoint_writer,
        ) if slides and requested.intersection({"ocr", "summary", "chapters"}) else list(prior.get("ppt_pages") or [])
        if "ocr" in requested:
            outputs["ppt_pages"] = pages
        if requested.intersection({"summary", "chapters"}):
            summary = create_summary(
                api_key,
                title=str(payload.get("title") or ""),
                transcript=transcript,
                ppt_pages=pages,
                prior_checkpoint=prior,
                checkpoint=checkpoint_writer,
            )
            if "summary" in requested:
                outputs["summary"] = summary
            if "chapters" in requested:
                outputs["chapters"] = list(summary.get("chapters") or [])
        metrics["elapsed_seconds"] = round(time.monotonic() - started, 3)
    else:
        raise WorkerError("unsupported job kind")
    return {
        "schema": RESULT_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "task_id": job["task_id"],
        "job_kind": kind,
        "input_hash": job["input_hash"],
        "pipeline_fingerprint": str(dict(job.get("pipeline") or {}).get("version") or "v2"),
        "status": "completed",
        "outputs": outputs,
        "metrics": metrics,
        "warnings": [],
    }


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
        with control_lock:
            control_sequence += 1
            value = {
                "schema": CONTROL_SCHEMA,
                "protocol_version": PROTOCOL_VERSION,
                "task_id": job["task_id"],
                "input_hash": job["input_hash"],
                "sequence": control_sequence,
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
                mailbox.publish_status(control_sequence, sealed_control)
            else:
                mailbox.publish_control(control_sequence, sealed_control)

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
        publish_control("checkpoint", {"checkpoint": value})

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
