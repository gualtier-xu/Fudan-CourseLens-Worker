"""Fail CI if course-platform acquisition code enters the public worker."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOTS = [
    ROOT / "courselens_worker",
    ROOT / ".github" / "workflows",
    ROOT / "scripts",
]
CODE_PREFIXES = ("courselens_worker/", ".github/workflows/", "scripts/")
SELF = Path(__file__).resolve()
FORBIDDEN = {
    "course platform module": re.compile(r"(?:src\.)?api\.(?:icourse|webvpn)", re.I),
    "course URL acquisition": re.compile(r"\b(?:get_video_url|get_sub_info|sign_video_url)\b", re.I),
    "original media persistence": re.compile(r"\b(?:download_video|resume_download|archive_video)\b", re.I),
    "student credential": re.compile(r"\b(?:student_id|UISPsw|StuId)\b"),
    "platform host": re.compile(r"(?:icourse|webvpn)[-\.][a-z0-9.-]+", re.I),
}
PLATFORM_CONNECTOR = "courselens_worker/platform_session.py"


def _violations(label: str, text: str) -> list[str]:
    normalized = label.replace("\\", "/")
    connector = normalized.endswith(PLATFORM_CONNECTOR)
    failures = []
    for name, pattern in FORBIDDEN.items():
        if connector and name in {
            "course platform module", "course URL acquisition",
            "student credential", "platform host",
        }:
            continue
        if pattern.search(text):
            failures.append(f"{label}: {name}")
    if connector and re.search(r"\b(?:write_bytes|NamedTemporaryFile|mkstemp)\b", text):
        failures.append(f"{label}: connector persistence")
    return failures


def _history_failures() -> list[str]:
    try:
        commits = subprocess.run(
            ["git", "rev-list", "--all"], cwd=ROOT, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return ["Git history could not be scanned"]
    failures: list[str] = []
    for commit in commits:
        names = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout.splitlines()
        for name in names:
            normalized = name.replace("\\", "/")
            if not normalized.startswith(CODE_PREFIXES) or Path(normalized).suffix not in {".py", ".yml", ".yaml", ".sh"}:
                continue
            if normalized == "scripts/check_public_boundary.py":
                # This scanner necessarily contains the forbidden expressions.
                continue
            content = subprocess.run(
                ["git", "show", f"{commit}:{normalized}"], cwd=ROOT, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            ).stdout
            failures.extend(_violations(f"history {commit[:12]}:{normalized}", content))
    return failures


def main() -> int:
    failures: list[str] = []
    for code_root in CODE_ROOTS:
        for path in code_root.rglob("*"):
            if path.suffix not in {".py", ".yml", ".yaml", ".sh"}:
                continue
            if path.resolve() == SELF:
                continue
            text = path.read_text(encoding="utf-8")
            failures.extend(_violations(str(path.relative_to(ROOT)), text))
    failures.extend(_history_failures())
    if failures:
        print("Public boundary violations:")
        print("\n".join(failures))
        return 1
    print("Public boundary check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
