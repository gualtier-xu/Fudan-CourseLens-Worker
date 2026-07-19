# Fudan CourseLens CPU Worker

This public repository contains the generic, CPU-only GitHub Actions worker for
Fudan CourseLens. It transforms a user-authorized, short-lived HTTPS media
stream into derived learning material such as subtitles, OCR text, summaries,
and chapters.

## Important boundary

This repository contains one narrowly scoped session connector. When the user
and platform have explicitly authorized runner-side authentication, it may
establish a one-job in-memory session and authorize only the requested lecture
and slide sources. Credentials arrive inside the encrypted job and are not
stored as a long-lived Actions secret.

The repository **does not** expose course discovery, original-video saving,
resumable media transfer, batch acquisition, or media archiving. The connector
has no original-media persistence API, and the worker still accepts only an
encrypted job created by the user's local private application after a runner
has started.

The media input is decoded directly into bounded mono PCM chunks. Encoded
source containers are never written to disk or uploaded as artifacts. Only an
encrypted derived-content result is retained, for seven days at most; the local
application deletes it immediately after a successful transactional import.

This boundary follows the same responsible-use principle documented by
[Fudan_iCourse_Subscriber](https://github.com/gualtier-xu/Fudan_iCourse_Subscriber#readme).

## Processing modes

- `fast`: SenseVoice INT8 on four CPU threads.
- `no-proofread`: FireRedASR2 CTC INT8 on four CPU threads.
- `standard`: SenseVoice, then FireRedASR2 CTC, then user-authorized DeepSeek
  proofreading. The two ASR models are never run concurrently.
- `summary`: optional RapidOCR, then DeepSeek summary and chapter generation.

All inputs and outputs use the versioned `job.v2` / `result.v2` protocol. Encrypted
`control.v2` records provide progress and resumable checkpoints without exposing
source text or authorized URLs in public logs.
Inputs are encrypted to the worker's X25519 public key. Results are encrypted
to a per-job local public key and signed with the worker's Ed25519 key.
ASR, OCR, proofreading, and summary-map windows emit encrypted checkpoints;
a replacement workflow resumes only after the last fully verified window.

## Repository secrets

Student Worker repositories are created from this repository as a GitHub
template. The desktop client configures the `courselens-worker` environment;
students never create a PAT or manually copy a secret.

The environment contains:

- `COURSELENS_JOB_TOKEN`: an expiring GitHub App user token. It exists only
  while a job is active and can access the student's own Worker and encrypted
  Mailbox repositories.
- `WORKER_INPUT_PRIVATE_KEY`: base64 X25519 private key.
- `WORKER_SIGNING_PRIVATE_KEY`: base64 Ed25519 seed.

`COURSELENS_MAILBOX_REPO` is a repository variable, not a secret. Pull-request
CI uses synthetic fixtures and never receives the production environment.

## Local verification

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py'
python scripts/check_public_boundary.py
```

Maintainers can also dispatch `Synthetic CPU compute smoke`. The workflow
generates speech and a duplicate slide pair inside the runner, loads both
pinned ASR models and RapidOCR, verifies slide deduplication/checkpointing, and
reports only counts and timings. It uses no repository secret, user recording,
course source, transcript artifact, or external media fixture.

## License

Apache License 2.0. Model weights are fetched from their upstream projects and
remain subject to their respective licenses.
