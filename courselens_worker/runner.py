"""Production entrypoint for one encrypted GitHub Actions job."""

from __future__ import annotations

import json
import os
import sys
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


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise WorkerError(f"required worker setting is missing: {name}")
    return value


def _progress(stage: str, completed: int, total: int) -> None:
    # Counts and stage identifiers are safe for public logs. No source text,
    # titles, URLs, headers, or exception response bodies are printed.
    print(f"stage={stage} completed={max(0, int(completed))} total={max(0, int(total))}", flush=True)


def process_job(
    job: dict[str, Any],
    *,
    checkpoint_writer=None,
) -> dict[str, Any]:
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
            progress=_progress,
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
            progress=_progress,
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
        from .llm import create_summary, proofread_segments
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
                progress=_progress,
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
        slides = list(payload.get("slides") or [])
        pages = process_slides(
            slides,
            progress=_progress,
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

    def publish_control(kind: str, payload: dict[str, Any]) -> None:
        nonlocal control_sequence
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
        mailbox.publish_control(
            control_sequence,
            seal_control(
                value,
                str(job["result_public_key"]),
                _required("WORKER_SIGNING_PRIVATE_KEY"),
            ),
        )

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

    publish_control("progress", {"stage": "remote_compute", "percent": 20.0})
    result = process_job(job, checkpoint_writer=write_checkpoint)
    sealed = seal_result(
        result,
        str(job["result_public_key"]),
        _required("WORKER_SIGNING_PRIVATE_KEY"),
    )
    output = Path(os.environ.get("COURSELENS_RESULT_PATH", "result.box.json"))
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(sealed, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, output)
    publish_control("progress", {"stage": "result_ready", "percent": 90.0})
    print(f"task={task_id} stage=result_ready", flush=True)
    return 0


def main() -> int:
    try:
        return run()
    except Exception as exc:
        # Avoid str(exc): networking libraries often embed an authorized URL.
        print(f"worker_failed type={type(exc).__name__}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
