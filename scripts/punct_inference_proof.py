"""Real punctuation-restoration proof for the pinned ct-punc engine.

N10 merge gate (fail-closed): the pinned funasr-onnx engine must load the
real ct-punc model and restore punctuation on a transcript derived from real
audio, not merely import. The install recipe replicates the production split
install declared in process.yml: the core requirements resolve without the
engine wheel (its metadata caps numpy<=1.26.4 against the repo pin
numpy==2.2.6, R1-N17), then the wheel is installed --no-deps; its real
transitive imports are pinned explicitly in requirements.txt.

The audio leg uses a pinned sensevoice test clip (same pinned download face
as production): real speech -> sensevoice ASR transcript -> ct-punc
restoration through the worker's own adapter. Exits non-zero on any missing
model, import failure, empty or non-Chinese transcript, or a restoration
that did not add punctuation with content preserved. No engine reaches the
mirror pin without this evidence.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
WORKER_ROOT = SCRIPTS.parent
ENGINE_PIN = "funasr-onnx==0.4.3"
ENGINE_MODULE = "funasr-onnx"
PUNCT_MARK_RE = re.compile(r"[，。？！、；：,.!?;:]")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
MIN_TRANSCRIPT_CHARS = 8


def _run(command: list[str]) -> None:
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(
            f"proof step failed ({completed.returncode}): {' '.join(command)}"
        )


def install_runtime(requirements_path: Path) -> None:
    """Replicate the process.yml split install from the same requirements file."""
    core_lines = [
        line
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if not line.startswith(f"{ENGINE_PIN.split('==')[0]}")
    ]
    core_handle = tempfile.NamedTemporaryFile(
        "w", suffix="-requirements-core.txt", delete=False, encoding="utf-8"
    )
    with core_handle:
        core_handle.write("\n".join(core_lines) + "\n")
    _run([sys.executable, "-m", "pip", "install", "--requirement", core_handle.name])
    _run([sys.executable, "-m", "pip", "install", "--no-deps", ENGINE_PIN])


def install_models(model_root: Path) -> tuple[Path, Path]:
    """Install the pinned sensevoice + ct-punc models via the production script."""
    environment = os.environ.copy()
    environment["COURSELENS_MODEL_ROOT"] = str(model_root)
    environment["COURSELENS_INSTALL_MODELS"] = "sensevoice,ct-punc"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS / "install_models.py")],
        env=environment, capture_output=True, text=True, encoding="utf-8",
    )
    if completed.returncode != 0:
        raise SystemExit(
            "model install failed:\n"
            f"{(completed.stdout + completed.stderr)[-1200:]}"
        )
    sys.path.insert(0, str(SCRIPTS))
    import install_models

    sensevoice_dirs = install_models._model_directories(model_root, "sensevoice")
    if not sensevoice_dirs:
        raise SystemExit("sensevoice model directory missing after install")
    ct_dir = model_root / install_models.MODELS["ct-punc"]["dir_name"]
    if not (ct_dir / "model_quant.onnx").is_file():
        raise SystemExit("ct-punc model files missing after install")
    return sensevoice_dirs[0], ct_dir


def read_wav(path: Path) -> tuple[int, "list[float]"]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    import numpy as np

    samples = np.frombuffer(frames, dtype=np.int16).astype("float32") / 32768.0
    return sample_rate, samples.tolist()


def transcribe(model_dir: Path, wav_path: Path) -> tuple[str, float]:
    import sherpa_onnx

    sample_rate, samples = read_wav(wav_path)
    model_file = next(
        (name for name in ("model.int8.onnx", "model.onnx")
         if (model_dir / name).is_file()),
        None,
    )
    if model_file is None:
        raise SystemExit("sensevoice model file missing")
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model_dir / model_file), tokens=str(model_dir / "tokens.txt"),
        num_threads=1, use_itn=True, debug=False, provider="cpu",
    )
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    return " ".join(stream.result.text.split()), len(samples) / sample_rate


def engine_restore(ct_dir: Path, text: str) -> tuple[str, float]:
    """Direct engine call: import/construction/inference errors propagate."""
    from funasr_onnx import CT_Transformer

    started = time.monotonic()
    engine = CT_Transformer(str(ct_dir), quantize=True)
    result = engine(text)
    elapsed = time.monotonic() - started
    candidate = ""
    if isinstance(result, (list, tuple)) and result:
        item = result[0]
        if isinstance(item, (list, tuple)) and item:
            if len(item) >= 2 and isinstance(item[0], str) and not isinstance(item[1], str):
                candidate = item[0]
            elif isinstance(item[-1], str):
                candidate = item[-1]
        elif isinstance(item, str):
            candidate = item
    return " ".join(str(candidate).split()), elapsed


def main() -> int:
    model_root = Path(
        os.environ.get("COURSELENS_PROOF_MODEL_ROOT", WORKER_ROOT / ".models-proof")
    ).resolve()
    install_runtime(WORKER_ROOT / "requirements.txt")
    sensevoice_dir, ct_dir = install_models(model_root)

    import importlib.metadata

    sys.path.insert(0, str(WORKER_ROOT))
    from courselens_worker import punct

    wavs = sorted(sensevoice_dir.rglob("*.wav"))
    if not wavs:
        raise SystemExit("no test wav in the sensevoice model archive")
    transcript = ""
    used_wav: Path | None = None
    duration = 0.0
    for wav_path in wavs:
        text, seconds = transcribe(sensevoice_dir, wav_path)
        if len(text) >= MIN_TRANSCRIPT_CHARS and CJK_RE.search(text):
            transcript, used_wav, duration = text, wav_path, seconds
            if "zh" in wav_path.name.lower():
                break
    if used_wav is None or not transcript:
        raise SystemExit(
            "no Chinese transcript from the pinned test clips (ASR leg failed)"
        )

    # sensevoice use_itn 转写可能自带少量句读；剥掉后喂引擎=生产 fill 模式
    # 看到的无标点 ASR 段（内容核心不变），恢复证据因此不含「透传」假阳性。
    raw = "".join(ch for ch in transcript if ch not in punct._PUNCT_SET)

    engine_text, engine_seconds = engine_restore(ct_dir, raw)
    if not engine_text:
        raise SystemExit("engine returned empty output")
    if not PUNCT_MARK_RE.search(engine_text):
        raise SystemExit("engine output carried no punctuation")
    if punct._content_core(engine_text) != punct._content_core(raw):
        raise SystemExit("engine output changed the content characters")

    adapted = punct.restore_text(raw, ct_dir)
    if not PUNCT_MARK_RE.search(adapted):
        raise SystemExit("worker adapter output carried no punctuation")
    if punct._content_core(adapted) != punct._content_core(raw):
        raise SystemExit("worker adapter output changed the content characters")

    evidence = {
        "verdict": "PASS",
        "engine_pin": ENGINE_PIN,
        "funasr_onnx_version": importlib.metadata.version(ENGINE_MODULE),
        "numpy_version": importlib.metadata.version("numpy"),
        "onnxruntime_version": importlib.metadata.version("onnxruntime"),
        "wav": used_wav.name,
        "wav_seconds": round(duration, 3),
        "raw_chars": len(raw),
        "engine_punct_marks": len(PUNCT_MARK_RE.findall(engine_text)),
        "engine_seconds": round(engine_seconds, 3),
        "adapter_punct_marks": len(PUNCT_MARK_RE.findall(adapted)),
        "restored_preview": engine_text[:60],
    }
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - the gate fails closed, never silently
        print(f"punct inference proof FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
