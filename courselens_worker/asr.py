"""CPU-only ASR over bounded transient PCM chunks."""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sherpa_onnx

from .formats import normalize_segments
from .source import (
    MediaResponseProfile,
    pinned_media_proxy,
)

SAMPLE_RATE = 16_000
PCM_CHUNK_SECONDS = 10 * 60
ASR_WINDOW_SECONDS = 30


class ASRError(RuntimeError):
    pass


def _decode_failure(stderr: str) -> ASRError:
    """Classify bounded FFmpeg diagnostics without exposing their text."""
    value = str(stderr or "").casefold()
    for status, message in (
        ("401", "authorized media request returned HTTP 401"),
        ("403", "authorized media request returned HTTP 403"),
        ("404", "authorized media request returned HTTP 404"),
        ("429", "authorized media request returned HTTP 429"),
    ):
        if (
            f"server returned {status}" in value
            or f"http error {status}" in value
            or f"returned error: {status}" in value
        ):
            return ASRError(message)
    if "server returned 5" in value or "http error 5" in value:
        return ASRError("authorized media request returned HTTP 5xx")
    if "moov atom" in value:
        return ASRError("authorized media is missing a readable MP4 index")
    if "invalid data" in value:
        return ASRError("authorized media format was rejected by ffmpeg")
    return ASRError("ffmpeg could not decode the authorized media stream")


def _media_response_error(profile: MediaResponseProfile) -> ASRError | None:
    http_messages = {
        "http_401": "authorized media request returned HTTP 401",
        "http_403": "authorized media request returned HTTP 403",
        "http_404": "authorized media request returned HTTP 404",
        "http_429": "authorized media request returned HTTP 429",
        "http_4xx": "authorized media request returned HTTP 4xx",
        "http_5xx": "authorized media request returned HTTP 5xx",
        "http_3xx": "authorized media request returned an unsupported redirect",
        "http_other": "authorized media request returned an unsupported status",
    }
    if profile.http in http_messages:
        return ASRError(http_messages[profile.http])
    if profile.content == "content_html" or profile.magic == "magic_html":
        return ASRError("authorized media response contained HTML")
    if profile.content == "content_json" or profile.magic == "magic_json":
        return ASRError("authorized media response contained JSON")
    if profile.magic != "magic_iso_bmff":
        return ASRError("authorized media signature was rejected")
    return None


def _drain_bounded(pipe, output: bytearray, *, limit: int = 16 * 1024) -> None:
    while True:
        block = pipe.read(4096)
        if not block:
            return
        remaining = limit - len(output)
        if remaining > 0:
            output.extend(block[:remaining])


def _run_bounded_process(
    command: list[str],
    *,
    timeout: int,
    capture_stdout: bool,
) -> tuple[int, bytes, bytes]:
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    stdout = bytearray()
    stderr = bytearray()
    readers: list[threading.Thread] = []
    if process.stdout is not None:
        readers.append(threading.Thread(
            target=_drain_bounded,
            args=(process.stdout, stdout),
            name="courselens-media-stdout",
            daemon=True,
        ))
    if process.stderr is not None:
        readers.append(threading.Thread(
            target=_drain_bounded,
            args=(process.stderr, stderr),
            name="courselens-media-stderr",
            daemon=True,
        ))
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + max(1, int(timeout))
    try:
        remaining = max(1, int(deadline - time.monotonic()))
        process.wait(timeout=remaining)
    except Exception:
        process.kill()
        process.wait()
        raise
    finally:
        for reader in readers:
            reader.join(timeout=5)
    return int(process.returncode or 0), bytes(stdout), bytes(stderr)


def _run_media_proxy(
    source: dict[str, Any],
    command: Callable[[str], list[str]],
    *,
    timeout: int,
    capture_stdout: bool,
) -> tuple[int, bytes, bytes]:
    with pinned_media_proxy(source) as proxy:
        return _run_bounded_process(
            command(proxy.url),
            timeout=timeout,
            capture_stdout=capture_stdout,
        )


def _probe_duration(source: dict[str, Any]) -> float:
    try:
        returncode, stdout, _ = _run_media_proxy(
            source,
            lambda media_url: [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", "-i", media_url,
            ],
            timeout=120,
            capture_stdout=True,
        )
    except (subprocess.TimeoutExpired, OSError):
        raise ASRError("authorized media duration probe timed out")
    try:
        duration = float(stdout.decode("ascii", errors="ignore").strip())
    except (TypeError, ValueError):
        duration = 0.0
    if returncode != 0 or duration <= 0:
        raise ASRError("authorized media duration could not be determined")
    return duration


class RecognizerPool:
    def __init__(self, sensevoice_dir: Path, firered_dir: Path, *, threads: int = 4):
        self.sensevoice_dir = sensevoice_dir
        self.firered_dir = firered_dir
        self.threads = max(1, min(4, int(threads)))
        self._recognizers: dict[str, Any] = {}

    @staticmethod
    def _model(directory: Path) -> Path:
        for name in ("model.int8.onnx", "model.onnx"):
            path = directory / name
            if path.is_file():
                return path
        raise ASRError("configured ASR model directory is incomplete")

    def get(self, backend: str):
        if backend in self._recognizers:
            return self._recognizers[backend]
        directory = self.sensevoice_dir if backend == "sensevoice" else self.firered_dir
        model = self._model(directory)
        tokens = directory / "tokens.txt"
        if not tokens.is_file():
            raise ASRError("configured ASR token file is missing")
        if backend == "sensevoice":
            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model), tokens=str(tokens), num_threads=self.threads,
                use_itn=True, debug=False, provider="cpu",
            )
        elif backend == "firered":
            recognizer = sherpa_onnx.OfflineRecognizer.from_fire_red_asr_ctc(
                model=str(model), tokens=str(tokens), num_threads=self.threads,
                debug=False, provider="cpu",
            )
        else:
            raise ASRError("unsupported ASR backend")
        self._recognizers[backend] = recognizer
        return recognizer

    def transcribe_pcm(self, path: Path, backend: str, *, offset_seconds: float) -> list[dict[str, Any]]:
        recognizer = self.get(backend)
        samples = np.memmap(path, dtype=np.float32, mode="r")
        window_samples = SAMPLE_RATE * ASR_WINDOW_SECONDS
        streams: list[tuple[Any, int, int]] = []
        for start in range(0, len(samples), window_samples):
            end = min(len(samples), start + window_samples)
            if end - start < SAMPLE_RATE // 2:
                continue
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, np.asarray(samples[start:end]))
            streams.append((stream, start, end))
        if hasattr(recognizer, "decode_streams"):
            recognizer.decode_streams([item[0] for item in streams])
        else:
            for stream, _, _ in streams:
                recognizer.decode_stream(stream)
        segments = []
        base = int(offset_seconds * 1000)
        for stream, start, end in streams:
            text = " ".join(str(stream.result.text or "").replace("<sil>", "").split()).strip()
            if text:
                segments.append({
                    "start_ms": base + int(start / SAMPLE_RATE * 1000),
                    "end_ms": base + int(end / SAMPLE_RATE * 1000),
                    "text": text,
                })
        del samples
        return normalize_segments(segments)


def _ffmpeg_proxy_command(
    target: Path,
    media_url: str,
    *,
    offset: float,
    duration: float,
) -> list[str]:
    return [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{offset:.3f}", "-i", media_url, "-t", f"{duration:.3f}",
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-y", str(target),
    ]


def _decode_chunk_from_url(
    media_url: str,
    target: Path,
    *,
    offset: float,
    duration: float,
) -> None:
    try:
        returncode, _, ffmpeg_stderr = _run_bounded_process(
            _ffmpeg_proxy_command(
                target, media_url, offset=offset, duration=duration,
            ),
            timeout=900,
            capture_stdout=False,
        )
    except subprocess.TimeoutExpired:
        target.unlink(missing_ok=True)
        raise ASRError("authorized media decode timed out")
    except OSError:
        target.unlink(missing_ok=True)
        raise ASRError("authorized media upstream connection failed")
    if returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        raise _decode_failure(ffmpeg_stderr.decode("utf-8", errors="replace"))


def _decode_chunk(source: dict[str, Any], target: Path, *, offset: float, duration: float) -> None:
    """Decode one bounded chunk through a transient pinned media session."""
    with pinned_media_proxy(source) as proxy:
        _decode_chunk_from_url(proxy.url, target, offset=offset, duration=duration)


def transcribe(
    job: dict[str, Any],
    *,
    sensevoice_dir: Path,
    firered_dir: Path,
    proofread: Callable[..., list[dict[str, Any]]] | None,
    progress: Callable[[str, int, int], None],
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    payload = dict(job.get("payload") or {})
    source = dict(payload.get("media") or {})
    mode = str(payload.get("mode") or "standard")
    if mode not in {"fast", "no-proofread", "standard"}:
        raise ASRError("unsupported subtitle mode")
    start_seconds = float(source.get("start_seconds") or 0)
    duration = float(source.get("duration_seconds") or 0)
    if duration <= 0:
        duration = _probe_duration(source) - start_seconds
    if start_seconds < 0:
        raise ASRError("media start is invalid")
    if duration <= 0 or duration > 12 * 60 * 60:
        raise ASRError("media duration is missing or outside the supported range")
    strategy = str(os.environ.get("COURSELENS_STANDARD_STRATEGY") or "sequential").strip().lower()
    if strategy not in {"sequential", "parallel"}:
        strategy = "sequential"
    recognizer_threads = 2 if mode == "standard" and strategy == "parallel" else 4
    pool = RecognizerPool(sensevoice_dir, firered_dir, threads=recognizer_threads)
    if mode == "standard" and strategy == "parallel":
        pool.get("sensevoice")
        pool.get("firered")
    prior = dict(payload.get("checkpoint") or {})
    if prior and str(prior.get("mode") or "") != mode:
        raise ASRError("checkpoint subtitle mode does not match the job")
    sense_segments: list[dict[str, Any]] = list(prior.get("raw_sensevoice") or [])
    firered_segments: list[dict[str, Any]] = list(prior.get("raw_firered") or [])
    total_chunks = max(1, int((duration + PCM_CHUNK_SECONDS - 1) // PCM_CHUNK_SECONDS))
    completed_chunks = max(0, min(total_chunks, int(prior.get("completed_chunks") or 0)))
    started = time.monotonic()
    # Keep one authorized CDN playback session for the complete task.  The
    # runner still launches one bounded FFmpeg process and retains only one
    # transient PCM file per chunk. Rotate the proxy's hidden signed URL before
    # each later chunk so an expired URL is never the first request of a new
    # decode session; a real 401/403 inside a chunk still gets only one bounded
    # refresh retry in the proxy.
    with tempfile.TemporaryDirectory(prefix="courselens-pcm-") as temporary, pinned_media_proxy(source) as proxy:
        root = Path(temporary)
        for index in range(completed_chunks, total_chunks):
            if index > completed_chunks:
                proxy.refresh_source()
            relative_offset = index * PCM_CHUNK_SECONDS
            absolute_offset = start_seconds + relative_offset
            chunk_duration = min(PCM_CHUNK_SECONDS, duration - relative_offset)
            pcm = root / f"chunk-{index:04d}.f32le"
            _decode_chunk_from_url(
                proxy.url,
                pcm,
                offset=absolute_offset,
                duration=chunk_duration,
            )
            if mode == "standard" and strategy == "parallel":
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr") as executor:
                    sense_future = executor.submit(
                        pool.transcribe_pcm, pcm, "sensevoice", offset_seconds=absolute_offset
                    )
                    fire_future = executor.submit(
                        pool.transcribe_pcm, pcm, "firered", offset_seconds=absolute_offset
                    )
                    sense_segments.extend(sense_future.result())
                    firered_segments.extend(fire_future.result())
            else:
                if mode in {"fast", "standard"}:
                    sense_segments.extend(
                        pool.transcribe_pcm(pcm, "sensevoice", offset_seconds=absolute_offset)
                    )
                if mode in {"no-proofread", "standard"}:
                    firered_segments.extend(
                        pool.transcribe_pcm(pcm, "firered", offset_seconds=absolute_offset)
                    )
            pcm.unlink(missing_ok=True)
            progress("asr", index + 1, total_chunks)
            if checkpoint is not None:
                checkpoint({
                    "completed_chunks": index + 1,
                    "total_chunks": total_chunks,
                    "mode": mode,
                    "raw_sensevoice": normalize_segments(sense_segments),
                    "raw_firered": normalize_segments(firered_segments),
                })
    if mode == "fast":
        final = sense_segments
    elif mode == "no-proofread":
        final = firered_segments
    else:
        if proofread is None:
            raise ASRError("standard mode requires a proofreading provider")
        def proofread_checkpoint(value: dict[str, Any]) -> None:
            if checkpoint is not None:
                checkpoint({
                    "stage": "proofread",
                    "completed_chunks": total_chunks,
                    "total_chunks": total_chunks,
                    "mode": mode,
                    "raw_sensevoice": normalize_segments(sense_segments),
                    "raw_firered": normalize_segments(firered_segments),
                    **value,
                })

        final = proofread(
            sense_segments,
            firered_segments,
            prior,
            proofread_checkpoint,
        )
    return {
        "mode": mode,
        "segments": normalize_segments(final),
        "raw_sensevoice": normalize_segments(sense_segments),
        "raw_firered": normalize_segments(firered_segments),
        "metrics": {
            "duration_seconds": duration,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "chunks": total_chunks,
            "threads_per_model": recognizer_threads,
            "strategy": strategy if mode == "standard" else "single-model",
            "start_seconds": start_seconds,
        },
    }
