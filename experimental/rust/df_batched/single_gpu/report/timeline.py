"""Extract the B build's device timeline from an nsys SQLite export.

    nsys profile -t cuda -o run df-batched --xyz gly10.xyz --room-mib 1200 --reps 1
    nsys export -t sqlite -o run.sqlite run.nsys-rep
    python3 timeline.py run.sqlite timeline.json

The plan carries no NVTX ranges, so each event is classed by kernel name and
by its place in the trip. A trip is: the packed three-center kernel(s)
(`peri3`), the spherical transform's GEMMs, the trsm against L (its own
kernels and the GEMMs it issues), one `geam`, then the device-to-host copy.
GEMMs after a `peri3` and before the trip's first trsm kernel are the
transform. Anything from that first trsm kernel up to the `geam` is the fit.

The window is the last full run in the file: from the first `peri3` after
the last Cholesky kernel.
"""

import json
import sqlite3
import sys


def classify(name: str) -> str:
    n = name.lower()
    if "peri3" in n or "three_center" in n:
        return "integrals"
    if "trsm" in n:
        return "trsm"
    if "geam" in n or "transpose" in n:
        return "transpose"
    if "gemm" in n or "xmma" in n or "cutlass" in n or "ampere" in n or "sm80" in n:
        return "gemm"
    if "cholesky" in n or "potrf" in n:
        return "cholesky"
    return "other"


def main(db: str, out: str) -> None:
    con = sqlite3.connect(db)
    names = dict(con.execute("SELECT id, value FROM StringIds"))
    events = [
        (s, e, stream, classify(names.get(n, "")), names.get(n, ""))
        for s, e, stream, n in con.execute(
            "SELECT start, end, streamId, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
    ]
    kinds = {1: "h2d", 2: "d2h", 8: "d2d"}
    events += [
        (s, e, stream, kinds.get(k, "copy"), f"{b} bytes")
        for s, e, stream, k, b in con.execute(
            "SELECT start, end, streamId, copyKind, bytes "
            "FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        )
    ]
    events.sort()
    chol = [i for i, ev in enumerate(events) if ev[3] == "cholesky"]
    start_i = chol[-1] if chol else 0
    first = next(i for i in range(start_i, len(events)) if events[i][3] == "integrals")
    run = events[first:]
    # Attribute the GEMMs. After a trip's integrals, the first GEMM (with any
    # split-K reduction it issues) is the dense aux GEMM W = U_aux^T X; the
    # GEMMs after it, up to the first trsm kernel, are the ket and bra
    # transforms; from that trsm kernel on, the fit.
    phase, aux_seen = "aux", False
    for i, (s, e, stream, kind, name) in enumerate(run):
        if kind == "integrals":
            phase, aux_seen = "aux", False
        elif kind == "trsm":
            phase = "fit"
        elif kind == "transpose":
            phase = "after"
        elif kind == "gemm" and phase == "aux":
            if aux_seen and "splitk" not in name.lower():
                phase = "orbital"
            aux_seen = True
        if kind == "gemm":
            kind = {"aux": "transform aux", "orbital": "transform bra/ket"}.get(
                phase, "fit"
            )
        elif kind == "trsm":
            kind = "fit"
        run[i] = (s, e, stream, kind, name)
    t0 = run[0][0]
    streams = sorted({ev[2] for ev in run})
    doc = {
        "t0_ns": t0,
        "streams": streams,
        "events": [
            {
                "start_ms": (s - t0) / 1e6,
                "end_ms": (e - t0) / 1e6,
                "stream": stream,
                "kind": kind,
                "name": name[:80],
            }
            for s, e, stream, kind, name in run
        ],
    }
    with open(out, "w") as f:
        json.dump(doc, f)
        f.write("\n")
    total = {}
    for s, e, _, kind, _ in run:
        total[kind] = total.get(kind, 0) + (e - s) / 1e6
    span = (run[-1][1] - t0) / 1e6
    print(f"{len(run)} events over {span:.1f} ms on streams {streams}")
    for k, v in sorted(total.items(), key=lambda x: -x[1]):
        print(f"  {k:10s} {v:9.1f} ms")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
