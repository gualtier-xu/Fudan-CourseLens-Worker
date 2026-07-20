"""Validate local Markdown links without fetching external resources."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def markdown_files() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*.md") if ".git" not in path.parts)


def failures() -> list[str]:
    errors: list[str] = []
    for document in markdown_files():
        text = document.read_text(encoding="utf-8")
        for raw in LINK.findall(text):
            target = raw.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            relative = unquote(target.split("#", 1)[0])
            resolved = (document.parent / relative).resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                errors.append(f"{document.relative_to(ROOT)}: link escapes repository: {target}")
                continue
            if not resolved.exists():
                errors.append(f"{document.relative_to(ROOT)}: missing link target: {target}")
    return errors


def main() -> int:
    errors = failures()
    if errors:
        print("Markdown link check failed:")
        print("\n".join(errors))
        return 1
    print(f"Markdown link check passed ({len(markdown_files())} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
