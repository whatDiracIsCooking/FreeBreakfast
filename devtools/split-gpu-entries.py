#!/usr/bin/env python3
"""Split a ctest listing into the GPU entries one gtest process can batch.

cpp-tier.sh runs every ctest entry as its own process, and each GPU entry pays
the runtime's context creation, about 0.45s on the RTX 3080, before its case
starts: most of the tier's wall clock once the cases are cheap. So it runs the
batchable GPU entries as a few sharded processes of the gpu binary instead, as
cpp-smoke.sh does, and ctest runs the rest.

An entry is batchable when its command is exactly the gpu binary with one
`--gtest_filter=<case>` (and gtest's own flags): a TEST_LAUNCHER such as
compute-sanitizer puts itself first, and running the binary bare would drop
it. RUN_SERIAL entries stay with ctest, which is what keeps them alone on the
card (test/gtest/CMakeLists.txt, `_fb_serial_cases`).

Reads `ctest --show-only=json-v1` on stdin, so the preset's filters and any
`-R`/`-E`/`-L` the caller passed are already applied. Writes the batched
entries' ctest names, one per line, to EXCLUDE (for `ctest
--exclude-from-file`), and their gtest names joined by `:` to FILTER (a
GTEST_FILTER). Writes both empty, so the caller runs ctest alone, when there is
nothing to batch or the filter would be too long for one environment string.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Linux caps one argument or environment string at 128 KiB (MAX_ARG_STRLEN).
MAX_FILTER = 100_000
FILTER_FLAG = "--gtest_filter="


def batchable(test: dict, gpu_bin: str) -> str | None:
    """The gtest name of a batchable entry, else None."""
    command = test.get("command") or []
    if not command or Path(command[0]).resolve() != Path(gpu_bin).resolve():
        return None
    properties = {p["name"]: p["value"] for p in test.get("properties", [])}
    if "gpu" not in properties.get("LABELS", []) or properties.get("RUN_SERIAL"):
        return None
    filters = [a for a in command[1:] if a.startswith(FILTER_FLAG)]
    rest = [a for a in command[1:] if not a.startswith(FILTER_FLAG)]
    if len(filters) != 1 or any(a != "--gtest_also_run_disabled_tests" for a in rest):
        return None
    return filters[0][len(FILTER_FLAG) :]


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: split-gpu-entries.py GPU_BIN EXCLUDE FILTER < listing.json",
            file=sys.stderr,
        )
        return 2
    gpu_bin, exclude, filter_out = sys.argv[1:]
    text = sys.stdin.read()
    # ctest's own chatter (-V, say) can precede the document.
    listing = json.loads(text[text.index("{") :])
    names, cases = [], []
    for test in listing.get("tests", []):
        case = batchable(test, gpu_bin)
        if case is not None:
            names.append(test["name"])
            cases.append(case)
    joined = ":".join(cases)
    if len(joined) > MAX_FILTER:
        names, joined = [], ""
    Path(exclude).write_text("".join(f"{n}\n" for n in names))
    Path(filter_out).write_text(joined)
    return 0


if __name__ == "__main__":
    sys.exit(main())
