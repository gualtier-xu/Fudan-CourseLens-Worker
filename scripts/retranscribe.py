"""Optional local re-transcription command (SUBTITLE-DEEP-1 Phase B).

Runs the product ASR chain over LOCAL media with the hotword-capable zipformer
backend, then optionally the term-position deep correction stage. This is the
operator-facing fallback path: it never touches school services and never
reads or prints credentials.

Usage (from the worker root, with the product venv/anaconda env):
  python scripts/retranscribe.py --media lecture.wav --terms terms.txt \
      --out-dir ./retrans-out [--backends sensevoice,zipformer] [--deepseek]

- --media: any file ffmpeg decodes (wav/mp3/mp4/...); internally converted to
  16 kHz mono f32le PCM. Never uploaded anywhere.
- --terms: UTF-8 file, one course term per line; feeds BOTH the zipformer
  hotword file and the term deep-correction closed set.
- --backends: product SUBTITLE_BACKENDS sequence; defaults to
  "sensevoice,zipformer" here because hotwords are the point of this command.
- --deepseek: enable the term deep-correction stage; requires DEEPSEEK_API_KEY
  in the environment (the key itself is never printed or logged).

Output: <out-dir>/retranscribe.{srt,vtt,json} + counters-only stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WORKER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKER_ROOT.parent))

from courselens_worker.asr import (  # noqa: E402
    ASR_HOTWORD_LIMIT,
    PCM_CHUNK_SECONDS,
    SAMPLE_RATE,
    RecognizerPool,
    _slice_pcm_chunk,
    fold_transcript_repetitions,
    subtitle_backend_sequence,
)
from courselens_worker.formats import normalize_segments, to_srt, to_vtt  # noqa: E402


def _decode_to_pcm(media: Path, target: Path) -> float:
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(media),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-y", str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        sys.exit("media decode failed (ffmpeg)")
    return target.stat().st_size / (SAMPLE_RATE * 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", required=True, type=Path)
    parser.add_argument("--terms", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("retrans-out"))
    parser.add_argument("--backends", default="sensevoice,zipformer")
    parser.add_argument("--deepseek", action="store_true")
    parser.add_argument(
        "--cache-file", type=Path, default=None,
        help="optional JSON cache for LLM responses (re-runs zero-billing)",
    )
    args = parser.parse_args()

    os.environ["SUBTITLE_BACKENDS"] = args.backends
    backends = subtitle_backend_sequence()
    hotwords: tuple[str, ...] = ()
    if args.terms is not None:
        hotwords = tuple(
            line.strip() for line in args.terms.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    api_key = ""
    if args.deepseek:
        # 显式环境变量入口；绝不打印、绝不落盘。
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            sys.exit("--deepseek requires DEEPSEEK_API_KEY in the environment")

    directories = {
        "sensevoice": os.environ.get("SENSEVOICE_MODEL_DIR", ""),
        "paraformer": os.environ.get("PARAFORMER_MODEL_DIR", ""),
        "zipformer": os.environ.get("ZIPFORMER_MODEL_DIR", ""),
    }
    pool = RecognizerPool(
        Path(directories["sensevoice"]),
        Path(directories["paraformer"]) if directories["paraformer"] else None,
        zipformer_dir=Path(directories["zipformer"]) if directories["zipformer"] else None,
    )
    effective = list(backends)
    if "zipformer" in effective and not pool.zipformer_ready():
        print("stage=zipformer-fallback reason=model_dir_not_ready", flush=True)
        effective[effective.index("zipformer")] = "paraformer"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="courselens-retrans-") as temporary:
        root = Path(temporary)
        full_pcm = root / "media-full.f32le"
        seconds = _decode_to_pcm(args.media, full_pcm)
        segments: list[dict] = []
        pool.hotwords_file = None
        if "zipformer" in effective and hotwords:
            pool.hotwords_file = root / "hotwords.txt"
            pool.hotwords_file.write_text(
                "\n".join(hotwords[:ASR_HOTWORD_LIMIT]) + "\n", encoding="utf-8",
            )
        backend = effective[1] if len(effective) > 1 else effective[0]
        pool.get(backend)  # 预暖：模型缺席/损坏在解码前按闭集码失败
        offset = 0.0
        while offset < seconds:
            duration = min(PCM_CHUNK_SECONDS, seconds - offset)
            chunk = root / "chunk.f32le"
            _slice_pcm_chunk(full_pcm, chunk, offset=offset, duration=duration)
            segments.extend(pool.transcribe_pcm(chunk, backend, offset_seconds=offset))
            offset += duration
            print(f"stage=decode offset_seconds={round(offset)}", flush=True)
    fold_transcript_repetitions(segments)
    segments = normalize_segments(segments)

    if api_key and hotwords:
        from courselens_worker.llm import term_proofread_segments

        cache: dict[str, str] | None = None
        if args.cache_file is not None:
            if args.cache_file.exists():
                cache = json.loads(args.cache_file.read_text(encoding="utf-8"))
            else:
                cache = {}
        segments = term_proofread_segments(
            api_key, segments, terms=hotwords, cache=cache,
        )
        if cache is not None and args.cache_file is not None:
            args.cache_file.write_text(
                json.dumps(cache, ensure_ascii=False), encoding="utf-8",
            )

    base = args.out_dir / "retranscribe"
    base.with_suffix(".srt").write_text(to_srt(segments), encoding="utf-8")
    base.with_suffix(".vtt").write_text(to_vtt(segments), encoding="utf-8")
    base.with_suffix(".json").write_text(
        json.dumps({"backends": effective, "hotwords": len(hotwords),
                    "segments": segments}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"stage=done backend={backend} segments={len(segments)} "
        f"hotwords={len(hotwords)} seconds={round(time.time() - started, 1)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
