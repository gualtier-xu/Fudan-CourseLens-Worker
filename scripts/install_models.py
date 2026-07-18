"""Install pinned upstream ASR models with SHA-256 verification."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import requests


MODELS = {
    "sensevoice": {
        "archive": "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
        "sha256": "7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e",
    },
    "firered": {
        "archive": "sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2",
        "sha256": "1da8b737ecc5e29f36759a4460c754863e7c919a4ba325aea187331fbfc83274",
    },
}
BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"


def _model_directories(root: Path, name: str) -> list[Path]:
    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and (path / "tokens.txt").is_file()
    ] if root.is_dir() else []
    marker = "fire-red" if name == "firered" else "sense-voice"
    return sorted(path for path in candidates if marker in path.name)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:bz2") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError("model archive contains an unsafe path")
        handle.extractall(destination, filter="data")


def _install(name: str, spec: dict[str, str], root: Path) -> Path:
    marker = root / f".{name}-{spec['sha256']}.ready"
    if marker.is_file():
        directories = _model_directories(root, name)
        if directories:
            return directories[0]
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model-install-") as temporary:
        archive = Path(temporary) / spec["archive"]
        digest = hashlib.sha256()
        with requests.get(f"{BASE}/{spec['archive']}", stream=True, timeout=60) as response:
            response.raise_for_status()
            with archive.open("wb") as output:
                for block in response.iter_content(1024 * 1024):
                    digest.update(block)
                    output.write(block)
        if digest.hexdigest() != spec["sha256"]:
            raise RuntimeError(f"{name} model checksum mismatch")
        _safe_extract(archive, root)
    marker.write_text(spec["sha256"], encoding="ascii")
    directories = _model_directories(root, name)
    if not directories:
        raise RuntimeError(f"{name} model directory was not extracted")
    return directories[0]


def main() -> None:
    root = Path(os.environ.get("COURSELENS_MODEL_ROOT", ".models")).resolve()
    installed = {name: _install(name, spec, root) for name, spec in MODELS.items()}
    environment = Path(os.environ.get("GITHUB_ENV", root / "models.env"))
    with environment.open("a", encoding="utf-8") as output:
        output.write(f"SENSEVOICE_MODEL_DIR={installed['sensevoice']}\n")
        output.write(f"FIRERED_MODEL_DIR={installed['firered']}\n")


if __name__ == "__main__":
    main()
