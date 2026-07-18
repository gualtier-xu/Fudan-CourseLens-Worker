"""Encrypted job.v1/result.v1 protocol; kept byte-compatible with the client."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from typing import Any, Iterable

import zstandard
from nacl.exceptions import CryptoError
from nacl.public import PrivateKey, PublicKey, SealedBox
from nacl.signing import SigningKey

PROTOCOL_VERSION = "1"
JOB_SCHEMA = "job.v1"
RESULT_SCHEMA = "result.v1"
ENVELOPE_SCHEMA = "sealed.v1"
_TASK_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class ProtocolError(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _b64e(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64d(value: str, field: str) -> bytes:
    try:
        return base64.b64decode(str(value).encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise ProtocolError(f"{field} is not valid base64") from exc


def validate_task_id(task_id: str) -> str:
    value = str(task_id or "").strip().lower()
    if not _TASK_ID_RE.fullmatch(value):
        raise ProtocolError("invalid task id")
    return value


def join_envelope(chunks: Iterable[str]) -> dict[str, Any]:
    raw = _b64d("".join(str(value).strip() for value in chunks), "chunks")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid encrypted envelope") from exc
    if not isinstance(value, dict):
        raise ProtocolError("invalid encrypted envelope shape")
    return value


def open_job(envelope: dict[str, Any], worker_private_key: str) -> dict[str, Any]:
    if envelope.get("schema") != ENVELOPE_SCHEMA:
        raise ProtocolError("unsupported envelope")
    ciphertext = _b64d(str(envelope.get("ciphertext") or ""), "ciphertext")
    if sha256_hex(ciphertext) != str(envelope.get("sha256") or ""):
        raise ProtocolError("ciphertext checksum mismatch")
    key = _b64d(worker_private_key, "worker private key")
    if len(key) != PrivateKey.SIZE:
        raise ProtocolError("worker private key has the wrong length")
    try:
        compressed = SealedBox(PrivateKey(key)).decrypt(ciphertext)
        raw = zstandard.ZstdDecompressor().decompress(compressed, max_output_size=64 * 1024 * 1024)
        job = json.loads(raw.decode("utf-8"))
    except (CryptoError, zstandard.ZstdError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("job could not be decrypted") from exc
    validate_job(job)
    return job


def validate_job(job: dict[str, Any]) -> None:
    if not isinstance(job, dict) or job.get("schema") != JOB_SCHEMA:
        raise ProtocolError("unsupported job schema")
    validate_task_id(str(job.get("task_id") or ""))
    if str(job.get("protocol_version") or "") != PROTOCOL_VERSION:
        raise ProtocolError("unsupported protocol version")
    if job.get("job_kind") not in {"echo", "subtitle", "summary"}:
        raise ProtocolError("unsupported job kind")
    created = float(job.get("created_at") or 0)
    expires = float(job.get("expires_at") or 0)
    now = time.time()
    if created <= 0 or expires <= created or expires - created > 1800:
        raise ProtocolError("invalid job validity window")
    if created > now + 120 or expires < now - 120:
        raise ProtocolError("job is outside its validity window")
    result_key = _b64d(str(job.get("result_public_key") or ""), "result public key")
    if len(result_key) != PublicKey.SIZE:
        raise ProtocolError("result public key has the wrong length")
    hashable = dict(job)
    expected = str(hashable.pop("input_hash", ""))
    if expected != sha256_hex(canonical_json(hashable)):
        raise ProtocolError("job input hash mismatch")


def seal_result(result: dict[str, Any], result_public_key: str, signing_private_key: str) -> dict[str, Any]:
    if result.get("schema") != RESULT_SCHEMA:
        raise ProtocolError("unsupported result schema")
    validate_task_id(str(result.get("task_id") or ""))
    public = _b64d(result_public_key, "result public key")
    signing = _b64d(signing_private_key, "signing private key")
    if len(public) != PublicKey.SIZE or len(signing) != 32:
        raise ProtocolError("result key has the wrong length")
    compressed = zstandard.ZstdCompressor(level=9).compress(canonical_json(result))
    ciphertext = SealedBox(PublicKey(public)).encrypt(compressed)
    signature = SigningKey(signing).sign(ciphertext).signature
    return {
        "schema": ENVELOPE_SCHEMA,
        "encoding": "zstd+sealedbox+base64",
        "task_id": result["task_id"],
        "protocol_version": PROTOCOL_VERSION,
        "sha256": sha256_hex(ciphertext),
        "ciphertext": _b64e(ciphertext),
        "signature": _b64e(signature),
    }
