"""Versioned encrypted wire protocol shared by the client and public Worker.

Only opaque ciphertext produced by this module may be placed in GitHub Issues
or Actions artifacts.  Canonical JSON and explicit hashes make retries and
imports deterministic and allow the local app to reject replayed results.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from typing import Any, Iterable

import zstandard
from nacl.exceptions import BadSignatureError, CryptoError
from nacl.public import PrivateKey, PublicKey, SealedBox
from nacl.signing import SigningKey, VerifyKey


PROTOCOL_VERSION = "2"
SUPPORTED_PROTOCOL_VERSIONS = (PROTOCOL_VERSION,)
JOB_SCHEMA = "job.v2"
RESULT_SCHEMA = "result.v2"
CONTROL_SCHEMA = "control.v2"
ENVELOPE_SCHEMA = "sealed.v2"
PROCESS_CANARY_SCHEMA = "process-canary.v1"
PROCESS_CANARY_PIPELINE = "actions-process-canary-v1"
PROCESS_CANARY_FIXTURE_SHA256 = "7793fbb749d3486d26acefe6ac726359b4e2f14078f1e44020638c6140323acb"
PROCESS_CANARY_FIXTURE_BYTES = 68
PROCESS_CANARY_FIXTURE_RECORDS = 3
MAX_CLOCK_SKEW_SECONDS = 120
MAX_JOB_LIFETIME_SECONDS = 30 * 60
_TASK_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class ProtocolError(ValueError):
    """Raised when encrypted data or its schema fails validation."""


def ensure_protocol_supported(version: str) -> str:
    """Return a normalized protocol version or fail with a closed error."""
    value = str(version or "").strip()
    if value not in SUPPORTED_PROTOCOL_VERSIONS:
        raise ProtocolError("unsupported protocol version")
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _b64e(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64d(value: str, *, field: str) -> bytes:
    try:
        return base64.b64decode(str(value).encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise ProtocolError(f"{field} is not valid base64") from exc


def generate_box_keypair() -> tuple[str, str]:
    private_key = PrivateKey.generate()
    return _b64e(bytes(private_key)), _b64e(bytes(private_key.public_key))


def generate_signing_keypair() -> tuple[str, str]:
    signing_key = SigningKey.generate()
    return _b64e(bytes(signing_key)), _b64e(bytes(signing_key.verify_key))


def validate_task_id(task_id: str) -> str:
    value = str(task_id or "").strip().lower()
    if not _TASK_ID_RE.fullmatch(value):
        raise ProtocolError("task_id must be a 32-character lowercase UUID hex value")
    return value


def validate_job(job: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
    if not isinstance(job, dict) or job.get("schema") != JOB_SCHEMA:
        raise ProtocolError(f"job schema must be {JOB_SCHEMA}")
    validate_task_id(str(job.get("task_id") or ""))
    if str(job.get("protocol_version") or "") != PROTOCOL_VERSION:
        raise ProtocolError("unsupported protocol version")
    if job.get("job_kind") not in {
        "echo", "process_canary", "subtitle", "summary", "chapters", "learning_pack",
        "quality_judge",  # P11-CONTRACT-1 PKG-A：LLM-as-judge 离线质检（加性注册）。
    }:
        raise ProtocolError("unsupported job kind")
    created_at = float(job.get("created_at") or 0)
    expires_at = float(job.get("expires_at") or 0)
    current = float(time.time() if now is None else now)
    if created_at <= 0 or expires_at <= created_at:
        raise ProtocolError("invalid job validity window")
    if expires_at - created_at > MAX_JOB_LIFETIME_SECONDS:
        raise ProtocolError("job validity window is too long")
    if created_at > current + MAX_CLOCK_SKEW_SECONDS:
        raise ProtocolError("job was created in the future")
    if expires_at < current - MAX_CLOCK_SKEW_SECONDS:
        raise ProtocolError("job has expired")
    result_key = _b64d(str(job.get("result_public_key") or ""), field="result_public_key")
    if len(result_key) != PublicKey.SIZE:
        raise ProtocolError("result_public_key has the wrong length")
    payload = job.get("payload") or {}
    if not isinstance(payload, dict):
        raise ProtocolError("job payload must be an object")
    if job.get("job_kind") == "process_canary":
        expected_keys = {
            "schema", "protocol_version", "task_id", "job_kind", "created_at",
            "expires_at", "result_public_key", "pipeline", "payload", "secrets",
            "requested_outputs", "input_hash",
        }
        if set(job) != expected_keys:
            raise ProtocolError("process canary job fields are not canonical")
        if job.get("payload") != {}:
            raise ProtocolError("process canary payload fields are not canonical")
        if job.get("pipeline") != {"version": PROCESS_CANARY_PIPELINE}:
            raise ProtocolError("process canary pipeline is invalid")
        if job.get("secrets") != {}:
            raise ProtocolError("process canary secrets must be empty")
        if job.get("requested_outputs") != []:
            raise ProtocolError("process canary requested outputs must be empty")
    requested_outputs = job.get("requested_outputs") or []
    if not isinstance(requested_outputs, list) or any(not isinstance(item, str) for item in requested_outputs):
        raise ProtocolError("requested_outputs must be a string list")
    media = payload.get("media")
    if media is not None:
        if not isinstance(media, dict):
            raise ProtocolError("media must be an object")
        start_seconds = float(media.get("start_seconds") or 0)
        if start_seconds < 0:
            raise ProtocolError("media.start_seconds cannot be negative")
        duration = media.get("duration_seconds")
        if duration is not None and float(duration) <= 0:
            raise ProtocolError("media.duration_seconds must be positive")
    source_session = payload.get("source_session")
    if source_session is not None:
        if not isinstance(source_session, dict) or source_session.get("provider") != "runner-session-v1":
            raise ProtocolError("source_session provider is unsupported")
        if not str(source_session.get("course_id") or "").isdigit():
            raise ProtocolError("source_session course_id is invalid")
        if not str(source_session.get("sub_id") or "").isdigit():
            raise ProtocolError("source_session sub_id is invalid")
        if not (bool(source_session.get("media")) or bool(source_session.get("slides"))):
            raise ProtocolError("source_session requested no resources")
        credentials = dict(dict(job.get("secrets") or {}).get("source_credentials") or {})
        account = str(credentials.get("account") or "")
        password = str(credentials.get("password") or "")
        if not account or len(account) > 64 or not password or len(password) > 512:
            raise ProtocolError("source_session credentials are invalid")
    expected_hash = str(job.get("input_hash") or "")
    hashable = dict(job)
    hashable.pop("input_hash", None)
    actual_hash = sha256_hex(canonical_json(hashable))
    if expected_hash != actual_hash:
        raise ProtocolError("job input_hash does not match its payload")
    return job


def finalize_job(job: dict[str, Any]) -> dict[str, Any]:
    value = dict(job)
    value["schema"] = JOB_SCHEMA
    value["protocol_version"] = PROTOCOL_VERSION
    value["task_id"] = validate_task_id(str(value.get("task_id") or ""))
    hashable = dict(value)
    hashable.pop("input_hash", None)
    value["input_hash"] = sha256_hex(canonical_json(hashable))
    validate_job(value)
    return value


def _seal_payload(payload: dict[str, Any], recipient_public_key: str) -> dict[str, Any]:
    public_bytes = _b64d(recipient_public_key, field="recipient_public_key")
    if len(public_bytes) != PublicKey.SIZE:
        raise ProtocolError("recipient public key has the wrong length")
    compressed = zstandard.ZstdCompressor(level=9).compress(canonical_json(payload))
    ciphertext = SealedBox(PublicKey(public_bytes)).encrypt(compressed)
    return {
        "schema": ENVELOPE_SCHEMA,
        "encoding": "zstd+sealedbox+base64",
        "sha256": sha256_hex(ciphertext),
        "ciphertext": _b64e(ciphertext),
    }


def _open_payload(envelope: dict[str, Any], recipient_private_key: str) -> dict[str, Any]:
    if not isinstance(envelope, dict) or envelope.get("schema") != ENVELOPE_SCHEMA:
        raise ProtocolError("unsupported encrypted envelope")
    ciphertext = _b64d(str(envelope.get("ciphertext") or ""), field="ciphertext")
    if sha256_hex(ciphertext) != str(envelope.get("sha256") or ""):
        raise ProtocolError("ciphertext checksum mismatch")
    private_bytes = _b64d(recipient_private_key, field="recipient_private_key")
    if len(private_bytes) != PrivateKey.SIZE:
        raise ProtocolError("recipient private key has the wrong length")
    try:
        compressed = SealedBox(PrivateKey(private_bytes)).decrypt(ciphertext)
        raw = zstandard.ZstdDecompressor().decompress(compressed, max_output_size=64 * 1024 * 1024)
        value = json.loads(raw.decode("utf-8"))
    except (CryptoError, zstandard.ZstdError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("encrypted payload could not be opened") from exc
    if not isinstance(value, dict):
        raise ProtocolError("encrypted payload must contain a JSON object")
    return value


def seal_job(job: dict[str, Any], worker_public_key: str) -> dict[str, Any]:
    finalized = finalize_job(job)
    envelope = _seal_payload(finalized, worker_public_key)
    envelope["task_id"] = finalized["task_id"]
    envelope["protocol_version"] = PROTOCOL_VERSION
    return envelope


def open_job(envelope: dict[str, Any], worker_private_key: str, *, now: float | None = None) -> dict[str, Any]:
    value = _open_payload(envelope, worker_private_key)
    return validate_job(value, now=now)


def seal_result(
    result: dict[str, Any],
    result_public_key: str,
    worker_signing_private_key: str,
) -> dict[str, Any]:
    if not isinstance(result, dict) or result.get("schema") != RESULT_SCHEMA:
        raise ProtocolError(f"result schema must be {RESULT_SCHEMA}")
    validate_task_id(str(result.get("task_id") or ""))
    if result.get("job_kind") == "process_canary":
        validate_process_canary_result(result)
    envelope = _seal_payload(result, result_public_key)
    ciphertext = _b64d(envelope["ciphertext"], field="ciphertext")
    signing_bytes = _b64d(worker_signing_private_key, field="worker_signing_private_key")
    if len(signing_bytes) != 32:
        raise ProtocolError("worker signing private key has the wrong length")
    envelope["signature"] = _b64e(SigningKey(signing_bytes).sign(ciphertext).signature)
    envelope["task_id"] = result["task_id"]
    envelope["protocol_version"] = PROTOCOL_VERSION
    return envelope


def seal_control(
    control: dict[str, Any],
    result_public_key: str,
    worker_signing_private_key: str,
) -> dict[str, Any]:
    """Encrypt and sign a runner-to-local progress or checkpoint record."""
    if not isinstance(control, dict) or control.get("schema") != CONTROL_SCHEMA:
        raise ProtocolError(f"control schema must be {CONTROL_SCHEMA}")
    validate_task_id(str(control.get("task_id") or ""))
    if control.get("control_kind") not in {"progress", "checkpoint", "refresh_request", "cancel_ack"}:
        raise ProtocolError("unsupported control kind")
    envelope = _seal_payload(control, result_public_key)
    ciphertext = _b64d(envelope["ciphertext"], field="ciphertext")
    signing_bytes = _b64d(worker_signing_private_key, field="worker_signing_private_key")
    if len(signing_bytes) != 32:
        raise ProtocolError("worker signing private key has the wrong length")
    envelope["signature"] = _b64e(SigningKey(signing_bytes).sign(ciphertext).signature)
    envelope["task_id"] = control["task_id"]
    envelope["protocol_version"] = PROTOCOL_VERSION
    envelope["message_kind"] = "control"
    return envelope


def open_result(
    envelope: dict[str, Any],
    result_private_key: str,
    worker_signing_public_key: str,
    *,
    expected_task_id: str,
    expected_input_hash: str,
) -> dict[str, Any]:
    if not isinstance(envelope, dict) or envelope.get("encoding") != "zstd+sealedbox+base64":
        raise ProtocolError("unsupported encrypted envelope encoding")
    if validate_task_id(str(envelope.get("task_id") or "")) != validate_task_id(expected_task_id):
        raise ProtocolError("result envelope task_id mismatch")
    if str(envelope.get("protocol_version") or "") != PROTOCOL_VERSION:
        raise ProtocolError("result envelope protocol is invalid")
    ciphertext = _b64d(str(envelope.get("ciphertext") or ""), field="ciphertext")
    verify_bytes = _b64d(worker_signing_public_key, field="worker_signing_public_key")
    if len(verify_bytes) != 32:
        raise ProtocolError("worker signing public key has the wrong length")
    signature = _b64d(str(envelope.get("signature") or ""), field="signature")
    try:
        VerifyKey(verify_bytes).verify(ciphertext, signature)
    except (BadSignatureError, ValueError) as exc:
        raise ProtocolError("worker result signature is invalid") from exc
    value = _open_payload(envelope, result_private_key)
    if value.get("schema") != RESULT_SCHEMA:
        raise ProtocolError(f"result schema must be {RESULT_SCHEMA}")
    if str(value.get("protocol_version") or "") != PROTOCOL_VERSION:
        raise ProtocolError("result protocol is invalid")
    if validate_task_id(str(value.get("task_id") or "")) != validate_task_id(expected_task_id):
        raise ProtocolError("result task_id mismatch")
    if str(value.get("input_hash") or "") != str(expected_input_hash):
        raise ProtocolError("result input_hash mismatch")
    if value.get("job_kind") not in {
        "echo", "process_canary", "subtitle", "summary", "chapters", "learning_pack",
        "quality_judge",  # P11-CONTRACT-1 PKG-A：与 job intake 闭集同步加性注册。
    }:
        raise ProtocolError("result job kind is invalid")
    if value.get("job_kind") == "process_canary":
        validate_process_canary_result(value)
    return value


def validate_process_canary_result(result: dict[str, Any]) -> dict[str, Any]:
    """Accept only the closed synthetic process-canary result vocabulary."""
    expected_keys = {
        "schema", "protocol_version", "task_id", "job_kind", "input_hash",
        "pipeline_fingerprint", "status", "outputs", "metrics", "warnings",
    }
    if not isinstance(result, dict) or set(result) != expected_keys:
        raise ProtocolError("process canary result fields are not canonical")
    if result.get("schema") != RESULT_SCHEMA or result.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("process canary result protocol is invalid")
    if result.get("job_kind") != "process_canary":
        raise ProtocolError("process canary result kind is invalid")
    validate_task_id(str(result.get("task_id") or ""))
    if not re.fullmatch(r"^[0-9a-f]{64}$", str(result.get("input_hash") or "")):
        raise ProtocolError("process canary input hash is invalid")
    if result.get("pipeline_fingerprint") != PROCESS_CANARY_PIPELINE:
        raise ProtocolError("process canary pipeline fingerprint is invalid")
    if result.get("status") != "completed" or result.get("warnings") != []:
        raise ProtocolError("process canary completion state is invalid")
    outputs = result.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"process_canary"}:
        raise ProtocolError("process canary outputs are not canonical")
    canary = outputs.get("process_canary")
    if not isinstance(canary, dict) or set(canary) != {
        "schema", "fixture_sha256", "fixture_bytes", "fixture_records", "worker_commit",
        "workflow_profile",
    }:
        raise ProtocolError("process canary output fields are not canonical")
    if canary.get("schema") != PROCESS_CANARY_SCHEMA:
        raise ProtocolError("process canary output schema is invalid")
    if canary.get("fixture_sha256") != PROCESS_CANARY_FIXTURE_SHA256:
        raise ProtocolError("process canary fixture digest is invalid")
    if canary.get("fixture_bytes") != PROCESS_CANARY_FIXTURE_BYTES:
        raise ProtocolError("process canary fixture size is invalid")
    if canary.get("fixture_records") != PROCESS_CANARY_FIXTURE_RECORDS:
        raise ProtocolError("process canary fixture record count is invalid")
    if not re.fullmatch(r"^[0-9a-f]{40}$", str(canary.get("worker_commit") or "")):
        raise ProtocolError("process canary worker commit is invalid")
    if canary.get("workflow_profile") != "process-v1":
        raise ProtocolError("process canary workflow profile is invalid")
    if result.get("metrics") != {
        "synthetic_bytes": PROCESS_CANARY_FIXTURE_BYTES,
        "synthetic_records": PROCESS_CANARY_FIXTURE_RECORDS,
    }:
        raise ProtocolError("process canary metrics are not canonical")
    return result


def open_control(
    envelope: dict[str, Any],
    result_private_key: str,
    worker_signing_public_key: str,
    *,
    expected_task_id: str,
    expected_input_hash: str,
) -> dict[str, Any]:
    ciphertext = _b64d(str(envelope.get("ciphertext") or ""), field="ciphertext")
    verify_bytes = _b64d(worker_signing_public_key, field="worker_signing_public_key")
    if len(verify_bytes) != 32:
        raise ProtocolError("worker signing public key has the wrong length")
    signature = _b64d(str(envelope.get("signature") or ""), field="signature")
    try:
        VerifyKey(verify_bytes).verify(ciphertext, signature)
    except (BadSignatureError, ValueError) as exc:
        raise ProtocolError("worker control signature is invalid") from exc
    value = _open_payload(envelope, result_private_key)
    if value.get("schema") != CONTROL_SCHEMA:
        raise ProtocolError(f"control schema must be {CONTROL_SCHEMA}")
    if validate_task_id(str(value.get("task_id") or "")) != validate_task_id(expected_task_id):
        raise ProtocolError("control task_id mismatch")
    if str(value.get("input_hash") or "") != str(expected_input_hash):
        raise ProtocolError("control input_hash mismatch")
    if value.get("control_kind") not in {"progress", "checkpoint", "refresh_request", "cancel_ack"}:
        raise ProtocolError("control kind is invalid")
    return value


def chunk_envelope(envelope: dict[str, Any], *, chunk_chars: int = 48 * 1024) -> list[str]:
    if chunk_chars < 1024:
        raise ValueError("chunk_chars is too small")
    encoded = _b64e(canonical_json(envelope))
    return [encoded[index:index + chunk_chars] for index in range(0, len(encoded), chunk_chars)]


def join_envelope(chunks: Iterable[str]) -> dict[str, Any]:
    raw = _b64d("".join(str(chunk).strip() for chunk in chunks), field="envelope chunks")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("envelope chunks do not contain valid JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError("envelope chunks must contain a JSON object")
    return value
