"""Verify signed generated Worker mirror metadata and every payload file."""

from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.protocol.mirror import (
    load_json_bytes,
    verify_manifest_document,
    verify_snapshot_files,
)
from shared.protocol.wire import SUPPORTED_PROTOCOL_VERSIONS


ROOT_SCHEMA = "courselens.worker-mirror.root.v1"


def _load(path: Path, label: str):
    return load_json_bytes(path.read_bytes(), label=label)


def _root_keys(document: dict) -> dict[str, str]:
    if document.get("schema") != ROOT_SCHEMA:
        raise ValueError("trusted root schema is unsupported")
    result = {}
    for item in document.get("root_keys") or []:
        key_id = str(item.get("key_id") or "")
        public_key = str(item.get("public_key") or "")
        if not key_id or key_id in result or len(base64.b64decode(public_key, validate=True)) != 32:
            raise ValueError("trusted root key is invalid")
        result[key_id] = public_key
    if not result:
        raise ValueError("trusted root has no keys")
    return result


def verify(root: Path, trusted_root_path: Path) -> dict:
    root = root.resolve()
    trusted_root = _load(trusted_root_path.resolve(), "trusted root")
    candidate_root = _load(root / "worker-mirror-root.json", "candidate root")
    if candidate_root != trusted_root:
        raise ValueError("candidate attempted to replace the trusted root")
    manifest = _load(root / "worker-mirror.manifest.json", "manifest")
    manifest_signature = _load(root / "worker-mirror.manifest.sig", "manifest signature")
    trust = _load(root / "worker-mirror-trust.json", "trust")
    trust_signature = _load(root / "worker-mirror-trust.sig", "trust signature")
    verified = verify_manifest_document(
        manifest,
        manifest_signature,
        trust,
        trust_signature,
        root_keys=_root_keys(trusted_root),
        supported_protocol_versions=SUPPORTED_PROTOCOL_VERSIONS,
    )
    source = manifest.get("source") or {}
    if not re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit") or "")):
        raise ValueError("manifest source commit is invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", str(source.get("worker_tree") or "")):
        raise ValueError("manifest Worker source tree is invalid")
    verify_snapshot_files(root, verified["files"])
    return verified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--trusted-root", type=Path, required=True)
    args = parser.parse_args()
    verified = verify(args.root, args.trusted_root)
    print(
        "Generated mirror verified "
        f"manifest={verified['manifest_sha256']} key={verified['key_id']} "
        f"files={len(verified['files'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
