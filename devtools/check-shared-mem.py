#!/usr/bin/env python3
"""Reject dynamically sized shared memory that does not say why it needs to be.

docs/style.md "Shared memory" is the rule: a kernel's shared memory is a
`__shared__` array sized at compile time, from a constexpr or a template
parameter. `extern __shared__` (and HIP's `HIP_DYNAMIC_SHARED`) is allowed only
on a line carrying `dynamic-smem-ok:` and a reason, or the line after a
comment-only line that does -- the reason being one of the two the rule
admits: more than CUDA's 48 KiB static limit, or a runtime size where a
worst-case bound would cost occupancy.

Comments and string literals are blanked before matching, and FelizJueves
kernels are skipped, exactly as devtools/check-int-types.py does (its helpers
are reused, not copied).

Same calling convention as the other commit hooks: pre-commit passes the
staged files, CI's `lint` job passes every tracked C++ file outside deps/.
Exits 1 with one `path:line:col: message` per finding, 0 when clean.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

_INT_TYPES = Path(__file__).resolve().parent / "check-int-types.py"
_spec = importlib.util.spec_from_file_location("check_int_types", _INT_TYPES)
assert _spec is not None and _spec.loader is not None
_int_types = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_int_types)

blank_comments_and_literals = _int_types.blank_comments_and_literals
is_generated = _int_types.is_generated

# The marker must carry a reason: a bare `dynamic-smem-ok:` suppresses nothing.
_SUPPRESS = re.compile(r"dynamic-smem-ok:\s*\S")

# `extern` and `__shared__` in either order, with anything (`__align__(16)`,
# a type) allowed only after both.
_DYNAMIC = re.compile(
    r"\bextern\s+__shared__\b|\b__shared__\s+extern\b|\bHIP_DYNAMIC_SHARED\b"
)


def is_suppressed(raw_lines: list[str], index: int) -> bool:
    """True when line `index` carries the marker, or follows a comment-only
    line that does (so the reason need not share an already-long line)."""
    if _SUPPRESS.search(raw_lines[index]):
        return True
    if index == 0:
        return False
    above = raw_lines[index - 1].strip()
    return above.startswith("//") and _SUPPRESS.search(above) is not None


def find_violations(text: str) -> list[tuple[int, int]]:
    """Return (line, column) for each unexplained dynamic declaration, 1-based."""
    raw_lines = text.split("\n")
    code_lines = blank_comments_and_literals(text).split("\n")
    found = []
    for lineno, code in enumerate(code_lines, 1):
        if is_suppressed(raw_lines, lineno - 1):
            continue
        for m in _DYNAMIC.finditer(code):
            found.append((lineno, m.start() + 1))
    return found


def main(argv: list[str]) -> int:
    status = 0
    for name in argv:
        text = Path(name).read_text(encoding="utf-8")
        if is_generated(text):
            continue
        for lineno, col in find_violations(text):
            print(
                f"{name}:{lineno}:{col}: dynamic shared memory -- size a "
                "`__shared__` array at compile time, or mark the line "
                '`// dynamic-smem-ok: <reason>`; see docs/style.md "Shared memory"'
            )
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
