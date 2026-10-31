#!/usr/bin/env python3
"""pt2_scan.csv -> pt2_scan.svg: CASSCF, PC- and SC-NEVPT2 E(R) on one
figure, each relative to its own lowest point on the scan (not yet to the
reactant: step 2 has no reactant calculation). Stdlib only.

    python3 experimental/cope_rearrangement/step_2/plot.py
"""

from __future__ import annotations

import csv
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SERIES = [
    ("dE_casscf", "CASSCF(6,6)", "#1f77b4"),
    ("dE_pc", "PC-NEVPT2", "#d62728"),
    ("dE_sc", "SC-NEVPT2", "#2ca02c"),
]
W, H, L, R, T, B = 640, 420, 70, 150, 30, 55


def main():
    rows = list(csv.DictReader((HERE / "pt2_scan.csv").open()))
    xs = [float(r["R"]) for r in rows]
    ys = {k: [float(r[k]) for r in rows] for k, _, _ in SERIES}
    x0, x1 = min(xs), max(xs)
    y1 = max(max(v) for v in ys.values())
    y1 = max(1.0, float(int(y1) + 1))

    def px(x):
        return L + (x - x0) / (x1 - x0) * (W - L - R)

    def py(y):
        return T + (1 - y / y1) * (H - T - B)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        'font-family="sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<line x1="{L}" y1="{py(0)}" x2="{W - R}" y2="{py(0)}" stroke="black"/>',
        f'<line x1="{L}" y1="{T}" x2="{L}" y2="{py(0)}" stroke="black"/>',
    ]
    step = 1 if y1 <= 10 else 2
    for yt in range(0, int(y1) + 1, step):
        out.append(f'<text x="{L - 8}" y="{py(yt) + 4}" text-anchor="end">{yt}</text>')
        out.append(
            f'<line x1="{L}" y1="{py(yt)}" x2="{W - R}" y2="{py(yt)}" '
            'stroke="#ddd" stroke-width="0.5"/>'
        )
    for i in range(int(round((x1 - x0) / 0.1)) + 1):
        xt = x0 + 0.1 * i
        out.append(
            f'<text x="{px(xt)}" y="{py(0) + 18}" text-anchor="middle">{xt:.1f}</text>'
        )
    out.append(
        f'<text x="{(L + W - R) / 2}" y="{H - 12}" text-anchor="middle">'
        "R16 = R34 / Å (step-1 CASSCF C2h geometries)</text>"
    )
    out.append(
        f'<text x="18" y="{(T + H - B) / 2}" text-anchor="middle" '
        f'transform="rotate(-90 18 {(T + H - B) / 2})">'
        "ΔE / kcal mol⁻¹ (each curve to its own minimum)</text>"
    )
    for n, (k, label, color) in enumerate(SERIES):
        pts = " ".join(
            f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys[k], strict=True)
        )
        out.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        for x, y in zip(xs, ys[k], strict=True):
            out.append(
                f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.5" fill="{color}"/>'
            )
        ly = T + 10 + 20 * n
        out.append(
            f'<line x1="{W - R + 12}" y1="{ly}" x2="{W - R + 36}" y2="{ly}" '
            f'stroke="{color}" stroke-width="2"/>'
        )
        out.append(f'<text x="{W - R + 42}" y="{ly + 4}">{label}</text>')
    out.append("</svg>")
    (HERE / "pt2_scan.svg").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
