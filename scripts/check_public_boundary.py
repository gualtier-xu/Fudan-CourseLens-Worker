"""Fail CI if course-platform acquisition code enters the public worker."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOTS = [ROOT / "courselens_worker", ROOT / ".github" / "workflows"]
FORBIDDEN = {
    "course platform module": re.compile(r"(?:src\.)?api\.(?:icourse|webvpn)", re.I),
    "course URL acquisition": re.compile(r"\b(?:get_video_url|get_sub_info|sign_video_url)\b", re.I),
    "original media persistence": re.compile(r"\b(?:download_video|resume_download|archive_video)\b", re.I),
    "student credential": re.compile(r"\b(?:student_id|UISPsw|StuId)\b"),
    "platform host": re.compile(r"(?:icourse|webvpn)[-\.][a-z0-9.-]+", re.I),
}


def main() -> int:
    failures = []
    for code_root in CODE_ROOTS:
        for path in code_root.rglob("*"):
            if path.suffix not in {".py", ".yml", ".yaml", ".sh"}:
                continue
            text = path.read_text(encoding="utf-8")
            for label, pattern in FORBIDDEN.items():
                if pattern.search(text):
                    failures.append(f"{path.relative_to(ROOT)}: {label}")
    if failures:
        print("Public boundary violations:")
        print("\n".join(failures))
        return 1
    print("Public boundary check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
