#!/usr/bin/env python3
"""Split a ctest listing into the GPU entries one gtest process can batch.

cpp-tier.sh runs every ctest entry as its own process, and each GPU entry pays
the runtime's context creation, about 0.45s on the RTX 3080, before its case
starts: most of the tier's wall clock once the cases are cheap. So it runs the
batchable GPU entries of each GPU binary as one process of that binary instead,
as cpp-smoke.sh does, and ctest runs the rest. There are several GPU binaries
(test/core's and test/gtest's), so one batch per binary; cpp-tier.sh runs them
one after another, so only one process is ever on the card.

An entry is batchable when its command is exactly one of the GPU binaries with
one `--gtest_filter=<case>` (and gtest's own flags): a TEST_LAUNCHER such as
compute-sanitizer puts itself first, and running the binary bare would drop
it. RUN_SERIAL entries stay with ctest, which is what keeps them alone on the
card (test/gtest/CMakeLists.txt, `_fb_serial_cases`).

Reads `ctest --show-only=json-v1` on stdin, so the preset's filters and any
`-R`/`-E`/`-L` the caller passed are already applied. Writes every batched
entry's ctest name, one per line, to EXCLUDE (one list for all the binaries,
for `ctest --exclude-from-file`), and for each GPU_BIN the gtest names of its
own batched entries, joined by `:`, to the FILTER after it (a GTEST_FILTER).
A binary's FILTER is written empty, and its entries left to ctest, when it has
nothing to batch or its filter would be too long for one environment string.
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
    args = sys.argv[1:]
    if len(args) < 3 or len(args) % 2 == 0:
        print(
            "usage: split-gpu-entries.py EXCLUDE GPU_BIN FILTER [GPU_BIN FILTER ...]"
            " < listing.json",
            file=sys.stderr,
        )
        return 2
    exclude, pairs = args[0], list(zip(args[1::2], args[2::2], strict=True))
    text = sys.stdin.read()
    # ctest's own chatter (-V, say) can precede the document.
    listing = json.loads(text[text.index("{") :])
    excluded = []
    for gpu_bin, filter_out in pairs:
        names, cases = [], []
        for test in listing.get("tests", []):
            case = batchable(test, gpu_bin)
            if case is not None:
                names.append(test["name"])
                cases.append(case)
        joined = ":".join(cases)
        if len(joined) > MAX_FILTER:
            names, joined = [], ""
        excluded += names
        Path(filter_out).write_text(joined)
    Path(exclude).write_text("".join(f"{n}\n" for n in excluded))
    return 0


if __name__ == "__main__":
    sys.exit(main())
