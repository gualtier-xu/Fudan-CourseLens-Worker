"""Reject invalid UTF-8 and replacement characters in the public worker."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TEXT_SUFFIXES = {".bat", ".cmd", ".css", ".html", ".js", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, stdout=subprocess.PIPE)
    errors: list[str] = []
    for name in result.stdout.decode("utf-8").split("\0"):
        if not name:
            continue
        path = root / name
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            errors.append(f"{name}: invalid UTF-8 ({exc})")
            continue
        if "\ufffd" in text:
            errors.append(f"{name}: contains U+FFFD replacement character")
        if "\x00" in text:
            errors.append(f"{name}: contains NUL byte")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Text encoding policy passed: tracked public text is valid UTF-8.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
