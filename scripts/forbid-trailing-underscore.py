#!/usr/bin/env python3
"""Pre-commit hook: forbid identifiers ending with a trailing underscore.

Strips C/C++ comments and string/char literals before scanning, so that
trailing underscores inside comments or strings don't trigger false
positives.

Exit code 0 = clean, 1 = violations found.
"""

import re
import sys


def strip_comments_and_strings(src: str) -> str:
    """Remove // and /* */ comments and all string/char literals."""
    src = re.sub(r'"(?:[^"\\]|\\.)*"', '""', src)
    src = re.sub(r"'(?:[^'\\]|\\.)*'", "''", src)
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    return src


def check_file(path: str) -> list[str]:
    """Return list of 'path:line:msg' strings for each violation."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        src = f.read()

    cleaned = strip_comments_and_strings(src)
    lines = cleaned.split("\n")
    violations: list[str] = []

    pattern = re.compile(r"[_]\b")

    for lineno, line in enumerate(lines, start=1):
        for m in pattern.finditer(line):
            start = m.start()
            if start == 0:
                continue
            if not line[start - 1].isalnum() and line[start - 1] != "_":
                continue
            violations.append(f"{path}:{lineno}: trailing underscore in identifier")

    return violations


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: forbid-trailing-underscore.py <file> [file ...]", file=sys.stderr)
        return 0

    all_violations: list[str] = []
    for path in sys.argv[1:]:
        all_violations.extend(check_file(path))

    if all_violations:
        for v in all_violations:
            print(v)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
