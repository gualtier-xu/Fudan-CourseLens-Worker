"""Run both pinned ASR models on speech generated inside the Actions runner.

The smoke test never downloads or stores a user recording.  It generates a
short English sentence with ``espeak-ng``, converts it to transient mono PCM,
and reports only counts and timings (never the recognized text).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from courselens_worker.asr import RecognizerPool, SAMPLE_RATE


SYNTHETIC_SENTENCE = (
    "This is a synthetic Course Lens speech recognition test for students."
)


def _required_directory(name: str) -> Path:
    value = Path(os.environ.get(name, "")).resolve()
    if not value.is_dir():
        raise RuntimeError(f"required model directory is missing: {name}")
    return value


def _run(command: list[str]) -> None:
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
    )


def main() -> int:
    sensevoice = _required_directory("SENSEVOICE_MODEL_DIR")
    firered = _required_directory("FIRERED_MODEL_DIR")
    started = time.monotonic()
    metrics: dict[str, object] = {
        "schema": "synthetic-asr-smoke.v1",
        "sample_origin": "generated-in-runner",
        "sample_rate": SAMPLE_RATE,
        "models": {},
    }
    with tempfile.TemporaryDirectory(prefix="courselens-synthetic-") as temporary:
        root = Path(temporary)
        wav = root / "speech.wav"
        pcm = root / "speech.f32le"
        _run(["espeak-ng", "-v", "en-us", "-s", "145", "-w", str(wav), SYNTHETIC_SENTENCE])
        _run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(wav), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "f32le", "-y", str(pcm),
        ])
        pool = RecognizerPool(sensevoice, firered, threads=4)
        for backend in ("sensevoice", "firered"):
            model_started = time.monotonic()
            segments = pool.transcribe_pcm(pcm, backend, offset_seconds=0.0)
            character_count = sum(len(str(item.get("text") or "")) for item in segments)
            if not segments or character_count <= 0:
                raise RuntimeError(f"{backend} returned no text for generated speech")
            metrics["models"][backend] = {
                "segments": len(segments),
                "characters": character_count,
                "elapsed_seconds": round(time.monotonic() - model_started, 3),
            }
    metrics["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(metrics, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
