"""Per-class device time of the packed three-center kernels in an nsys export.

    python3 kernels.py run.sqlite out.json

Sums every `peri3<la, lb, lP>` (Cartesian aux) and `speri3<la, lb, lP>`
(spherical aux) launch in the last full run of the file (from the last
Cholesky kernel on, as timeline.py's window), keyed "la,lb,lP".
"""

import json
import re
import sqlite3
import sys

CLASS = re.compile(
    r"\b(s?peri3)<\(unsigned char\)(\d), \(unsigned char\)(\d), \(unsigned char\)(\d)>"
)


def main(db: str, out: str) -> None:
    con = sqlite3.connect(db)
    names = dict(con.execute("SELECT id, value FROM StringIds"))
    rows = sorted(
        (s, e, names.get(n, ""))
        for s, e, n in con.execute(
            "SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
    )
    chol = [i for i, r in enumerate(rows) if "cholesky" in r[2].lower()]
    rows = rows[chol[-1] if chol else 0 :]
    per: dict = {}
    kind = None
    for s, e, name in rows:
        m = CLASS.search(name)
        if not m:
            continue
        kind = m.group(1)
        key = ",".join(m.group(i) for i in (2, 3, 4))
        c = per.setdefault(key, {"launches": 0, "ms": 0.0})
        c["launches"] += 1
        c["ms"] += (e - s) / 1e6
    with open(out, "w") as f:
        json.dump({"kernel": kind, "classes": per}, f, indent=1, sort_keys=True)
        f.write("\n")
    total = sum(c["ms"] for c in per.values())
    print(f"{kind}: {len(per)} classes, {total:.1f} ms")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
