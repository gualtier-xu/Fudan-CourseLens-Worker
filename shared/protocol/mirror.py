"""Signed provenance for deterministic public Worker mirrors.

This module intentionally contains no GitHub authentication or private client
logic.  It is exported with the Worker so both repositories validate the same
manifest and trust formats.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey


MANIFEST_SCHEMA = "courselens.worker-mirror.manifest.v1"
TRUST_SCHEMA = "courselens.worker-mirror.trust.v1"
SIGNATURE_SCHEMA = "courselens.worker-mirror.signature.v1"
ALGORITHM = "ed25519"
METADATA_PATHS = frozenset({
    "worker-mirror.manifest.json",
    "worker-mirror.manifest.sig",
    "worker-mirror-trust.json",
    "worker-mirror-trust.sig",
    "worker-mirror-root.json",
})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_KEY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")


class MirrorVerificationError(ValueError):
    """Closed verification failure safe to map to a user-facing error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def pretty_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def document_sha256(value: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json(dict(value)))


def _b64decode(value: str, *, field: str) -> bytes:
    try:
        return base64.b64decode(str(value).encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise MirrorVerificationError("worker_manifest_invalid", f"{field} is not valid base64") from exc


def _valid_key_id(value: Any, *, field: str = "key_id") -> str:
    key_id = str(value or "").strip().lower()
    if not _KEY_ID_RE.fullmatch(key_id):
        raise MirrorVerificationError("worker_manifest_invalid", f"{field} is invalid")
    return key_id


def _valid_sha256(value: Any, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise MirrorVerificationError("worker_manifest_invalid", f"{field} is invalid")
    return digest


def load_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MirrorVerificationError("worker_manifest_invalid", f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise MirrorVerificationError("worker_manifest_invalid", f"{label} must be a JSON object")
    return value


def sign_document(document: Mapping[str, Any], *, key_id: str, private_key: str) -> dict[str, Any]:
    key_id = _valid_key_id(key_id)
    private_bytes = _b64decode(private_key, field="private_key")
    if len(private_bytes) != 32:
        raise MirrorVerificationError("worker_manifest_invalid", "private_key has the wrong length")
    payload = canonical_json(dict(document))
    signature = SigningKey(private_bytes).sign(payload).signature
    return {
        "schema": SIGNATURE_SCHEMA,
        "algorithm": ALGORITHM,
        "key_id": key_id,
        "payload_sha256": sha256_hex(payload),
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def verify_document_signature(
    document: Mapping[str, Any],
    signature: Mapping[str, Any],
    *,
    public_key: str,
    expected_key_id: str | None = None,
) -> str:
    if str(signature.get("schema") or "") != SIGNATURE_SCHEMA:
        raise MirrorVerificationError("worker_manifest_invalid", "signature schema is unsupported")
    if str(signature.get("algorithm") or "") != ALGORITHM:
        raise MirrorVerificationError("worker_manifest_invalid", "signature algorithm is unsupported")
    key_id = _valid_key_id(signature.get("key_id"))
    if expected_key_id is not None and key_id != _valid_key_id(expected_key_id):
        raise MirrorVerificationError("worker_manifest_invalid", "signature key id does not match")
    payload = canonical_json(dict(document))
    if _valid_sha256(signature.get("payload_sha256"), field="payload_sha256") != sha256_hex(payload):
        raise MirrorVerificationError("worker_manifest_invalid", "signed payload digest does not match")
    public_bytes = _b64decode(public_key, field="public_key")
    signature_bytes = _b64decode(str(signature.get("signature") or ""), field="signature")
    if len(public_bytes) != 32 or len(signature_bytes) != 64:
        raise MirrorVerificationError("worker_manifest_invalid", "signature key or value has the wrong length")
    try:
        VerifyKey(public_bytes).verify(payload, signature_bytes)
    except (BadSignatureError, ValueError) as exc:
        raise MirrorVerificationError("worker_manifest_invalid", "document signature is invalid") from exc
    return key_id


def verify_trust_document(
    trust: Mapping[str, Any],
    signature: Mapping[str, Any],
    *,
    root_keys: Mapping[str, str],
    now: float | None = None,
) -> dict[str, dict[str, Any]]:
    if str(trust.get("schema") or "") != TRUST_SCHEMA:
        raise MirrorVerificationError("worker_manifest_invalid", "trust schema is unsupported")
    epoch = int(trust.get("epoch") or 0)
    expires_at = int(trust.get("expires_at") or 0)
    if epoch <= 0 or expires_at <= int(time.time() if now is None else now):
        raise MirrorVerificationError("worker_release_unavailable", "Worker mirror trust metadata is expired")
    root_key_id = _valid_key_id(signature.get("key_id"))
    root_public_key = str(root_keys.get(root_key_id) or "")
    if not root_public_key:
        raise MirrorVerificationError("worker_manifest_invalid", "trust metadata uses an unknown root key")
    verify_document_signature(trust, signature, public_key=root_public_key, expected_key_id=root_key_id)

    release_keys: dict[str, dict[str, Any]] = {}
    raw_keys = trust.get("release_keys") or []
    if not isinstance(raw_keys, list) or not raw_keys:
        raise MirrorVerificationError("worker_manifest_invalid", "trust metadata has no release keys")
    for raw in raw_keys:
        if not isinstance(raw, dict):
            raise MirrorVerificationError("worker_manifest_invalid", "release key entry is invalid")
        key_id = _valid_key_id(raw.get("key_id"))
        if key_id in release_keys:
            raise MirrorVerificationError("worker_manifest_invalid", "release key ids are not unique")
        status = str(raw.get("status") or "").strip().lower()
        if status not in {"active", "retired", "revoked"}:
            raise MirrorVerificationError("worker_manifest_invalid", "release key status is invalid")
        public_key = str(raw.get("public_key") or "")
        if len(_b64decode(public_key, field=f"release key {key_id}")) != 32:
            raise MirrorVerificationError("worker_manifest_invalid", "release public key has the wrong length")
        release_keys[key_id] = {"key_id": key_id, "status": status, "public_key": public_key}

    revoked = trust.get("revoked_manifests") or []
    if not isinstance(revoked, list):
        raise MirrorVerificationError("worker_manifest_invalid", "revoked manifest list is invalid")
    for digest in revoked:
        _valid_sha256(digest, field="revoked manifest digest")
    return release_keys


def _normalized_manifest_files(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = manifest.get("payload") or {}
    raw_files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(raw_files, list) or not raw_files:
        raise MirrorVerificationError("worker_manifest_invalid", "manifest file list is empty")
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise MirrorVerificationError("worker_manifest_invalid", "manifest file entry is invalid")
        path = str(raw.get("path") or "").replace("\\", "/")
        pure = PurePosixPath(path)
        if not path or pure.is_absolute() or ".." in pure.parts or path in METADATA_PATHS:
            raise MirrorVerificationError("worker_manifest_invalid", "manifest path is invalid")
        if path in seen:
            raise MirrorVerificationError("worker_manifest_invalid", "manifest paths are not unique")
        mode = str(raw.get("mode") or "")
        size = int(raw.get("size") if raw.get("size") is not None else -1)
        if mode not in {"100644", "100755"} or size < 0:
            raise MirrorVerificationError("worker_manifest_invalid", "manifest file mode or size is invalid")
        files.append({
            "path": path,
            "mode": mode,
            "size": size,
            "sha256": _valid_sha256(raw.get("sha256"), field=f"sha256 for {path}"),
        })
        seen.add(path)
    if files != sorted(files, key=lambda item: item["path"].encode("utf-8")):
        raise MirrorVerificationError("worker_manifest_invalid", "manifest files are not sorted")
    return files


def verify_manifest_document(
    manifest: Mapping[str, Any],
    signature: Mapping[str, Any],
    trust: Mapping[str, Any],
    trust_signature: Mapping[str, Any],
    *,
    root_keys: Mapping[str, str],
    supported_protocol_versions: Iterable[str] = ("2",),
    now: float | None = None,
) -> dict[str, Any]:
    if str(manifest.get("schema") or "") != MANIFEST_SCHEMA:
        raise MirrorVerificationError("worker_manifest_invalid", "manifest schema is unsupported")
    release_keys = verify_trust_document(trust, trust_signature, root_keys=root_keys, now=now)
    key_id = _valid_key_id(signature.get("key_id"))
    key = release_keys.get(key_id)
    if not key:
        raise MirrorVerificationError("worker_manifest_invalid", "manifest uses an unknown release key")
    if key["status"] == "revoked":
        raise MirrorVerificationError("worker_signing_key_revoked", "Worker mirror signing key is revoked")
    verify_document_signature(manifest, signature, public_key=key["public_key"], expected_key_id=key_id)
    manifest_digest = document_sha256(manifest)
    revoked = {str(value).lower() for value in trust.get("revoked_manifests") or []}
    if manifest_digest in revoked:
        raise MirrorVerificationError("worker_signing_key_revoked", "Worker mirror manifest is revoked")

    protocols = manifest.get("protocol") or {}
    versions = protocols.get("supported_versions") if isinstance(protocols, dict) else None
    if not isinstance(versions, list) or not versions or any(not isinstance(value, str) for value in versions):
        raise MirrorVerificationError("worker_manifest_invalid", "manifest protocol versions are invalid")
    supported = {str(value) for value in supported_protocol_versions}
    if not supported.intersection(versions):
        raise MirrorVerificationError("worker_protocol_incompatible", "Worker mirror protocol is incompatible")

    files = _normalized_manifest_files(manifest)
    payload = dict(manifest.get("payload") or {})
    expected_tree = _valid_sha256(payload.get("tree_sha256"), field="payload tree_sha256")
    if sha256_hex(canonical_json(files)) != expected_tree:
        raise MirrorVerificationError("worker_manifest_invalid", "manifest payload tree digest does not match")
    git_tree = str(payload.get("git_tree") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", git_tree):
        raise MirrorVerificationError("worker_manifest_invalid", "manifest payload Git tree is invalid")
    return {
        "manifest_sha256": manifest_digest,
        "key_id": key_id,
        "trust_epoch": int(trust.get("epoch") or 0),
        "files": files,
        "protocol_versions": list(versions),
        "payload_git_tree": git_tree,
    }


def verify_snapshot_files(root: Path, files: Iterable[Mapping[str, Any]]) -> None:
    root = Path(root).resolve()
    expected = {str(item["path"]): dict(item) for item in files}
    actual: set[str] = set()
    for path in root.rglob("*"):
        relative_path = path.relative_to(root)
        if relative_path.parts and relative_path.parts[0] == ".git":
            continue
        if path.is_dir():
            continue
        relative = relative_path.as_posix()
        if relative in METADATA_PATHS:
            continue
        if path.is_symlink():
            raise MirrorVerificationError("worker_manifest_invalid", "snapshot contains a symbolic link")
        actual.add(relative)
        item = expected.get(relative)
        if item is None:
            raise MirrorVerificationError("worker_manifest_invalid", f"snapshot contains unknown file {relative}")
        raw = path.read_bytes()
        if len(raw) != int(item["size"]) or sha256_hex(raw) != str(item["sha256"]):
            raise MirrorVerificationError("worker_manifest_invalid", f"snapshot file digest mismatch: {relative}")
    missing = sorted(set(expected) - actual)
    if missing:
        raise MirrorVerificationError("worker_manifest_invalid", f"snapshot is missing {missing[0]}")


__all__ = [
    "ALGORITHM", "MANIFEST_SCHEMA", "METADATA_PATHS", "MirrorVerificationError",
    "SIGNATURE_SCHEMA", "TRUST_SCHEMA", "canonical_json", "document_sha256",
    "load_json_bytes", "pretty_json", "sha256_hex", "sign_document",
    "verify_document_signature", "verify_manifest_document", "verify_snapshot_files",
    "verify_trust_document",
]
