"""Read results/run/ (run.sh's output) and print three markdown tables.

- Time: the median seconds of each routine per matrix, eigenvectors or not,
  at each thread count, and faer's time over the faster LAPACK routine's.
- Accuracy: each routine's worst residual max_j |A v_j - l_j v_j| / max|l|
  and orthogonality max |V^T V - I| over thread counts.
- Spectrum: every eigenvalue set against LAPACK dsyevd's (eigenvectors, one
  thread), as max |dl| / max|l|; and how many eigenvalues fall below
  tau * max|l| for each tau, which must agree across all of them: that
  count is what a canonical orthogonalization or a metric pseudo-inverse
  keeps or drops.

    python3 experimental/faer_blas_lapack/compare.py [RUN_DIR]

Standard library only: the eigenvalue files are 1-D <f8 .npy, which
read_npy parses, so no dependency group is needed.
"""

import ast
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

TAUS = (1e-6, 1e-8, 1e-10)


def read_npy(path: Path) -> list[float]:
    """The values of eigh-bench's 1-D <f8 .npy (version 1.0, C order)."""
    data = path.read_bytes()
    header_len = struct.unpack("<H", data[8:10])[0]
    header = ast.literal_eval(data[10 : 10 + header_len].decode())
    assert header["descr"] == "<f8" and len(header["shape"]) == 1, header
    (n,) = header["shape"]
    return list(
        struct.unpack(f"<{n}d", data[10 + header_len : 10 + header_len + 8 * n])
    )


def load(run: Path) -> list[dict]:
    rows = []
    for path in sorted(run.glob("*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line]
    return rows


def matrices(rows: list[dict]) -> list[tuple[str, int]]:
    return sorted(
        {(r["matrix"], r["n"]) for r in rows}, key=lambda m: (m[0].split("_")[0], m[1])
    )


def time_table(rows: list[dict]) -> None:
    threads = sorted({r["threads"] for r in rows})
    by = {
        (r["matrix"], r["vectors"], r["threads"], r["routine"]): r["median_s"]
        for r in rows
    }
    print("## Time (median s; faer / best LAPACK)\n")
    head = "| matrix | n | vectors |"
    for t in threads:
        head += f" faer t{t} | dsyevd t{t} | dsyevr t{t} | ratio t{t} |"
    print(head)
    print("|" + "---|" * (3 + 4 * len(threads)))
    for name, n in matrices(rows):
        for vectors in (False, True):
            line = f"| {name} | {n} | {'yes' if vectors else 'no'} |"
            for t in threads:
                f = by.get((name, vectors, t, "self_adjoint_evd"))
                d = by.get((name, vectors, t, "dsyevd"))
                r = by.get((name, vectors, t, "dsyevr"))
                best = min(x for x in (d, r) if x is not None) if (d or r) else None
                cell = lambda x: f"{x:.3g}" if x is not None else "-"  # noqa: E731
                ratio = f"{f / best:.2f}" if f and best else "-"
                line += f" {cell(f)} | {cell(d)} | {cell(r)} | {ratio} |"
            print(line)
    print()


def accuracy_table(rows: list[dict]) -> None:
    worst = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        if r["vectors"]:
            w = worst[(r["matrix"], r["n"], r["routine"])]
            w[0] = max(w[0], r["residual"])
            w[1] = max(w[1], r["orthogonality"])
    print("## Accuracy (worst over thread counts; n eps is about n * 2.2e-16)\n")
    print("| matrix | n | routine | residual | orthogonality | n eps |")
    print("|---|---|---|---|---|---|")
    for (name, n, routine), (res, orth) in sorted(
        worst.items(), key=lambda kv: (kv[0][0].split("_")[0], kv[0][1], kv[0][2])
    ):
        print(
            f"| {name} | {n} | {routine} | {res:.2e} | {orth:.2e} | {n * 2.2e-16:.1e} |"
        )
    print()


def spectrum_table(rows: list[dict], run: Path) -> bool:
    print("## Spectrum against dsyevd (vectors, t1)\n")
    head = "| matrix | n | lambda_min | lambda_max | max dl / max l |"
    head += "".join(f" #< {tau:g} max l |" for tau in TAUS)
    print(head)
    print("|" + "---|" * (5 + len(TAUS)))
    agree = True
    for name, n in matrices(rows):
        sets = {
            (r["routine"], r["vectors"], r["threads"]): read_npy(
                run / "eigvals" / r["eigvals"]
            )
            for r in rows
            if r["matrix"] == name
        }
        ref = sets.get(("dsyevd", True, 1))
        if ref is None:
            continue
        scale = max(abs(x) for x in ref)
        apart = (
            max(
                max(abs(a - b) for a, b in zip(w, ref, strict=True))
                for w in sets.values()
            )
            / scale
        )
        counts = []
        for tau in TAUS:
            seen = {sum(x < tau * scale for x in w) for w in sets.values()}
            agree &= len(seen) == 1
            counts.append("/".join(str(c) for c in sorted(seen)))
        print(
            f"| {name} | {n} | {ref[0]:.3e} | {ref[-1]:.3e} | {apart:.1e} | "
            + " | ".join(counts)
            + " |"
        )
    print()
    return agree


def main() -> int:
    run = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent / "results" / "run"
    )
    rows = load(run)
    if not rows:
        print(f"compare.py: no results in {run}", file=sys.stderr)
        return 2
    time_table(rows)
    accuracy_table(rows)
    agree = spectrum_table(rows, run)
    if not agree:
        print("**The below-threshold counts disagree somewhere (shown as a/b).**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
