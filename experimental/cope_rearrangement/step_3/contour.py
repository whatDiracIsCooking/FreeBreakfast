"""Contour plots of the relaxed (R16, R34) grids as stdlib SVG (no matplotlib
in uv.lock; step 2's plot.py is the precedent): filled cells coloured by
E - E_ref, contour lines by marching squares at a fixed interval, the grid's
stationary points marked, and an optional path (the IRC) overlaid.

plot() draws one method's surface (surface_<method>.svg); comparison() draws
both side by side on one colour scale, with the relaxed diagonal cut E(R, R)
beside them (comparison.svg). Both are built from panel()."""

from __future__ import annotations

import csv
from pathlib import Path

KCAL = 627.5094740631
SIZE, LEFT, TOP, PAD = 560, 70, 40, 150
STEP_KCAL = 2.0  # contour interval
CAP_KCAL = 40.0  # the colour scale saturates here
TITLE = {"pc": "DF-PC-NEVPT2", "casscf": "DF-CASSCF(6,6)"}
IRC_RED = ({"stroke": "#d62728", "stroke_width": 2},)


def el(tag: str, text: str | None = None, **attrs) -> str:
    """One SVG element; attribute names take '-' for '_'."""
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f"<{tag} {a}/>" if text is None else f"<{tag} {a}>{text}</{tag}>"


def colour(t: float) -> str:
    """t in [0, 1] -> a dark-blue-to-yellow ramp (viridis-like, five stops,
    linear in between)."""
    stops = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]
    t = min(max(t, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    f = t - i
    c = [round(a + (b - a) * f) for a, b in zip(stops[i], stops[i + 1], strict=True)]
    return f"rgb({c[0]},{c[1]},{c[2]})"


def segments(g: list[float], de: dict, level: float) -> list:
    """Marching squares over the cell centres: the level's line segments."""
    out = []
    for i in range(len(g) - 1):
        for j in range(len(g) - 1):
            corners = [
                (g[i], g[j]),
                (g[i + 1], g[j]),
                (g[i + 1], g[j + 1]),
                (g[i], g[j + 1]),
            ]
            if not all(c in de for c in corners):
                continue
            vals = [de[c] for c in corners]
            pts = []
            for k in range(4):
                (a0, b0), (a1, b1) = corners[k], corners[(k + 1) % 4]
                v0, v1 = vals[k], vals[(k + 1) % 4]
                if (v0 - level) * (v1 - level) < 0:
                    t = (level - v0) / (v1 - v0)
                    pts.append((a0 + t * (a1 - a0), b0 + t * (b1 - b0)))
            out += list(zip(pts[0::2], pts[1::2], strict=False))
    return out


def marker(kind: str, x: float, y: float, r: float = 5) -> str:
    """A minimum a circle, a saddle a diamond, a maximum a square."""
    if kind == "minimum":
        return el("circle", cx=f"{x:.2f}", cy=f"{y:.2f}", r=r, fill="white",
                  stroke="black")  # fmt: skip
    turn = f"rotate(45 {x:.2f} {y:.2f})" if kind == "saddle" else "rotate(0)"
    return el("rect", x=f"{x - r:.2f}", y=f"{y - r:.2f}", width=2 * r,
              height=2 * r, fill="white", stroke="black",
              transform=turn)  # fmt: skip


def irc_path(path: Path) -> list[tuple[float, float]] | None:
    """Step 4d's IRC as (R16, R34) pairs, or None before it exists."""
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    return [(float(r["r16"]), float(r["r34"])) for r in rows]


def panel(x0: float, y0: float, size: float, g: list[float], de: dict,
          cap: float, step: float, lohi: tuple[float, float], irc=None,
          points=(), irc_style=IRC_RED, r: float = 5,
          unit: str = "Angstrom") -> list[str]:  # fmt: skip
    """One contour panel, its frame at (x0, y0), size x size px over R16,
    R34 in lohi: cells of de (kcal/mol) coloured on [0, cap], contours every
    `step` up to max(de), the diagonal, the path `irc` (one polyline per
    style in irc_style, drawn in order), each stationary point in `points`
    and its mirror, then ticks and axis labels."""
    lo, hi = lohi
    span = hi - lo
    h = g[1] - g[0]
    top = max(de.values())

    def px(v):
        return x0 + (v - lo) / span * size

    def py(v):
        return y0 + size - (v - lo) / span * size

    out = []
    cell = h / span * size
    for (a, b), v in de.items():
        left, right = max(px(a) - cell / 2, x0), min(px(a) + cell / 2, x0 + size)
        upper, lower = max(py(b) - cell / 2, y0), min(py(b) + cell / 2, y0 + size)
        out.append(
            el("rect", x=f"{left:.2f}", y=f"{upper:.2f}",
               width=f"{right - left:.2f}", height=f"{lower - upper:.2f}",
               fill=colour(v / cap))
        )  # fmt: skip
    level = step
    while level < top:
        for p, q in segments(g, de, level):
            out.append(
                el("line", x1=f"{px(p[0]):.2f}", y1=f"{py(p[1]):.2f}",
                   x2=f"{px(q[0]):.2f}", y2=f"{py(q[1]):.2f}", stroke="white",
                   stroke_width=0.8, stroke_opacity=0.8)
            )  # fmt: skip
        level += step
    out.append(
        el("line", x1=px(lo), y1=py(lo), x2=px(hi), y2=py(hi), stroke="black",
           stroke_dasharray="3 3", stroke_width=0.6)
    )  # fmt: skip
    if irc:
        inside = [(a, b) for a, b in irc if lo <= a <= hi and lo <= b <= hi]
        pts = " ".join(f"{px(a):.2f},{py(b):.2f}" for a, b in inside)
        for style in irc_style:
            out.append(el("polyline", points=pts, fill="none", **style))
    for s in points:
        for a, b in {(s["r16"], s["r34"]), (s["r34"], s["r16"])}:
            out.append(marker(s["kind"], px(a), py(b), r))
    out.append(
        el("rect", x=x0, y=y0, width=size, height=size, fill="none", stroke="black")
    )
    for v in g:
        if round(v * 100) % 20:
            continue  # label every 0.2 A
        out.append(
            el("text", f"{v:.1f}", x=f"{px(v):.1f}", y=y0 + size + 16,
               text_anchor="middle")
        )  # fmt: skip
        out.append(
            el("text", f"{v:.1f}", x=x0 - 6, y=f"{py(v) + 4:.1f}", text_anchor="end")
        )
    out.append(
        el("text", f"R16 / {unit}", x=x0 + size / 2, y=y0 + size + 38,
           text_anchor="middle")
    )  # fmt: skip
    mid, xl = y0 + size / 2, x0 - 50
    out.append(
        el("text", f"R34 / {unit}", x=xl, y=mid, transform=f"rotate(-90 {xl} {mid})",
           text_anchor="middle")
    )  # fmt: skip
    return out


def colour_bar(x0: float, bottom: float, height: float, cap: float,
               tick: int) -> list[str]:  # fmt: skip
    """The ramp from 0 (at `bottom`) to cap (`height` above it), labelled every
    `tick` kcal/mol."""
    out = []
    for k in range(100):
        y = bottom - (k + 1) * height / 100
        out.append(
            el("rect", x=x0, y=f"{y:.2f}", width=16, height=f"{height / 100 + 0.5:.2f}",
               fill=colour(k / 99))
        )  # fmt: skip
    for v in range(0, int(cap) + 1, tick):
        y = bottom - v / cap * height
        out.append(el("text", str(v), x=x0 + 22, y=f"{y + 4:.1f}"))
    out.append(el("text", "kcal/mol", x=x0, y=bottom - height - 12))
    return out


def plot(path: Path, method: str, surface: dict, irc, summary: dict) -> None:
    g = surface["grid"]
    e = surface["energy"]
    emin = min(e.values())
    de = {k: (v - emin) * KCAL for k, v in e.items()}
    cap = min(max(de.values()), CAP_KCAL)
    h = g[1] - g[0]
    lohi = (g[0] - h / 2, g[-1] + h / 2)
    width, height = LEFT + SIZE + PAD, TOP + SIZE + 60
    title = (
        f"{TITLE[method]}/cc-pVDZ relaxed surface, E - E_min (kcal/mol), "
        f"contours every {STEP_KCAL:g}"
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" font-family="sans-serif" font-size="12">',
        el("rect", width=width, height=height, fill="white"),
        el(
            "text", title, x=LEFT + SIZE / 2, y=22, text_anchor="middle", font_size=14
        ),  # fmt: skip
    ]
    points = summary.get("grid_stationary_points", [])
    out += panel(LEFT, TOP, SIZE, g, de, cap, STEP_KCAL, lohi, irc, points)
    # the colour bar (its upper half) and the legend
    x0 = LEFT + SIZE + 30
    out += colour_bar(x0, TOP + SIZE, SIZE / 2, cap, 10)
    legend = [
        ("minimum", "minimum"),
        ("saddle", "saddle"),
        ("maximum", "maximum"),
        ("irc", "IRC (step 4d)"),
    ]
    for i, (kind, text) in enumerate(legend):
        y = TOP + 20 + 20 * i
        if kind == "irc":
            out.append(
                el("line", x1=x0, x2=x0 + 14, y1=y, y2=y, stroke="#d62728",
                   stroke_width=2)
            )  # fmt: skip
        else:
            out.append(marker(kind, x0 + 6, y))
        out.append(el("text", text, x=x0 + 20, y=y + 4))
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n")


# comparison.svg: three panels of C_SIZE px in a row, the two surfaces on one
# colour scale (0 to C_CAP, each method's E from its own lowest grid point)
# and contour interval, then the diagonal cut.
C_SIZE, C_TOP = 380, 104
C_CAP, C_STEP = 45.0, 1.0
C_CUT_TOP = 20.0  # the cut's kcal/mol axis runs 0 .. C_CUT_TOP
C_X = (80, 550, 1095)  # the panels' left edges
C_BAR = C_X[1] + C_SIZE + 22
C_WIDTH, C_HEIGHT = C_X[2] + C_SIZE + 25, C_TOP + C_SIZE + 125
# Okabe-Ito blue and vermillion, solid and dashed: told apart without colour.
C_LINE = {
    "pc": {"stroke": "#0072B2", "stroke_width": 2.5},
    "casscf": {"stroke": "#D55E00", "stroke_width": 2.5, "stroke_dasharray": "8 5"},
}
C_IRC = ({"stroke": "white", "stroke_width": 5}, {"stroke": "black", "stroke_width": 2})
# the cut's annotations: (method, kind of grid stationary point on the
# diagonal, its label), the R taken from surface.json
C_NOTES = [
    ("casscf", "minimum", "CASSCF diyl minimum"),
    ("pc", "saddle", "PC-NEVPT2 TS"),
    ("casscf", "saddle", "CASSCF aromatic saddle"),
]


def diagonal_point(summary: dict, kind: str) -> dict:
    """The one grid stationary point of `kind` on the diagonal."""
    (s,) = [
        s
        for s in summary["grid_stationary_points"]
        if s["kind"] == kind and s["r16"] == s["r34"]
    ]
    return s


def diagonal(surface: dict) -> list[tuple[float, float]]:
    """The grid's (R, E(R, R)) points, in R order."""
    e = surface["energy"]
    return [(a, e[(a, a)]) for a in surface["grid"] if (a, a) in e]


def diagonal_minima(surfaces: dict) -> dict:
    """Each method's R of its lowest diagonal grid point: the cut's zero."""
    return {m: min(diagonal(s), key=lambda p: p[1])[0] for m, s in surfaces.items()}


def cut(x0: float, y0: float, size: float, lohi: tuple[float, float],
        surfaces: dict, summaries: dict) -> list[str]:  # fmt: skip
    """The relaxed diagonal E(R, R) of each method, from its own lowest
    diagonal grid point, the grid's points joined by straight lines (no
    interpolation beyond that), and the diagonal stationary points of
    C_NOTES marked by a dotted line at their R."""
    lo, hi = lohi

    def px(v):
        return x0 + (v - lo) / (hi - lo) * size

    def py(v):
        return y0 + size - v / C_CUT_TOP * size

    out = []
    for v in range(0, int(C_CUT_TOP) + 1, 2):
        out.append(
            el("line", x1=x0, x2=x0 + size, y1=f"{py(v):.2f}", y2=f"{py(v):.2f}",
               stroke="#dddddd", stroke_width=0.8)
        )  # fmt: skip
        out.append(
            el("text", str(v), x=x0 - 6, y=f"{py(v) + 5:.1f}", text_anchor="end")
        )
    for method, kind, text in C_NOTES:
        s = diagonal_point(summaries[method], kind)
        x = px(s["r16"])
        out.append(
            el("line", x1=f"{x:.2f}", x2=f"{x:.2f}", y1=y0, y2=y0 + size,
               stroke="#555555", stroke_width=1, stroke_dasharray="2 3")
        )  # fmt: skip
        tx = x + 15  # rotated, the glyphs sit left of the baseline
        out.append(
            el("text", f"{text}, {s['r16']:.3f} Å", x=f"{tx:.2f}", y=y0 + 8,
               transform=f"rotate(-90 {tx:.2f} {y0 + 8})", text_anchor="end",
               font_size=14, fill=C_LINE[method]["stroke"])
        )  # fmt: skip
    for method, surface in surfaces.items():
        diag = diagonal(surface)
        ref = min(v for _, v in diag)
        pts = [(px(a), py((v - ref) * KCAL)) for a, v in diag]
        style = C_LINE[method]
        out.append(
            el("polyline", points=" ".join(f"{x:.2f},{y:.2f}" for x, y in pts),
               fill="none", **style)
        )  # fmt: skip
        for x, y in pts:
            out.append(
                el("circle", cx=f"{x:.2f}", cy=f"{y:.2f}", r=2.6,
                   fill=style["stroke"])
            )  # fmt: skip
    out.append(
        el("rect", x=x0, y=y0, width=size, height=size, fill="none", stroke="black")
    )
    v = lo + 0.025
    while v <= hi:
        if round(v * 100) % 20 == 0:
            out.append(
                el("text", f"{v:.1f}", x=f"{px(v):.1f}", y=y0 + size + 16,
                   text_anchor="middle")
            )  # fmt: skip
        v = round(v + 0.05, 2)
    out.append(
        el("text", "R = R16 = R34 / Å", x=x0 + size / 2, y=y0 + size + 38,
           text_anchor="middle")
    )  # fmt: skip
    mid, xl = y0 + size / 2, x0 - 42
    out.append(
        el("text", "E − own diagonal min. / kcal/mol", x=xl, y=mid,
           transform=f"rotate(-90 {xl} {mid})", text_anchor="middle")
    )  # fmt: skip
    return out


def comparison(path: Path, surfaces: dict, summaries: dict, irc) -> None:
    """comparison.svg: the PC-NEVPT2 surface (with the IRC), the CASSCF
    surface, on one colour scale and contour interval, each from its own
    lowest grid point; then the diagonal cut of both."""
    pc_g = surfaces["pc"]["grid"]
    h = pc_g[1] - pc_g[0]
    lohi = (pc_g[0] - h / 2, pc_g[-1] + h / 2)  # the PC grid's extent, both
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{C_WIDTH}" '
        f'height="{C_HEIGHT}" viewBox="0 0 {C_WIDTH} {C_HEIGHT}" '
        f'font-family="sans-serif" font-size="18">',
        el("rect", width=C_WIDTH, height=C_HEIGHT, fill="white"),
        el("text", "Chair Cope rearrangement, (6,6)/cc-pVDZ: DF-PC-NEVPT2 is "
           "concerted, DF-CASSCF stepwise", x=C_WIDTH / 2, y=30,
           text_anchor="middle", font_size=24, font_weight="bold"),
        el("text", "Relaxed (R16, R34) surfaces: E from each method's own lowest "
           f"grid point, one colour scale, contours every {C_STEP:g} kcal/mol",
           x=C_WIDTH / 2, y=58, text_anchor="middle", font_size=17),
    ]  # fmt: skip
    heads = [
        "DF-PC-NEVPT2, with the IRC (step 4d)",
        "DF-CASSCF(6,6)",
        "Relaxed diagonal cut, R16 = R34",
    ]
    for x, text in zip(C_X, heads, strict=True):
        out.append(
            el("text", text, x=x + C_SIZE / 2, y=C_TOP - 12, text_anchor="middle",
               font_size=19, font_weight="bold")
        )  # fmt: skip
    for x, method in zip(C_X[:2], ("pc", "casscf"), strict=True):
        s = surfaces[method]
        emin = min(s["energy"].values())
        de = {k: (v - emin) * KCAL for k, v in s["energy"].items()}
        out += panel(
            x, C_TOP, C_SIZE, s["grid"], de, C_CAP, C_STEP, lohi,
            irc if method == "pc" else None,
            summaries[method]["grid_stationary_points"], C_IRC, 7, "Å",
        )  # fmt: skip
    out += colour_bar(C_BAR, C_TOP + C_SIZE, C_SIZE, C_CAP, 5)
    out += cut(C_X[2], C_TOP, C_SIZE, lohi, surfaces, summaries)
    # the legends, under the panels they belong to
    y = C_TOP + C_SIZE + 78
    x = C_X[0] - 20
    for kind in ("minimum", "saddle", "maximum", "irc"):
        if kind == "irc":
            for style in C_IRC:
                out.append(el("line", x1=x, x2=x + 24, y1=y, y2=y, **style))
            out.append(el("text", "IRC", x=x + 32, y=y + 6))
            x += 90
        else:
            out.append(marker(kind, x + 7, y, 7))
            out.append(el("text", kind, x=x + 22, y=y + 6))
            x += 125
    out.append(
        el("text", "dashed: R16 = R34", x=x + 10, y=y + 6,
           fill="#333333")
    )  # fmt: skip
    x = C_X[2] - 20
    for method, text in (("pc", "DF-PC-NEVPT2"), ("casscf", "DF-CASSCF(6,6)")):
        out.append(el("line", x1=x, x2=x + 40, y1=y, y2=y, **C_LINE[method]))
        out.append(el("text", text, x=x + 48, y=y + 6))
        x += 190
    refs = ", ".join(
        f"{'PC' if m == 'pc' else 'CASSCF'} {r:.2f} Å"
        for m, r in diagonal_minima(surfaces).items()
    )
    out.append(
        el("text", f"zero: lowest diagonal point, {refs}", x=C_X[2] - 20,
           y=y + 30, font_size=14, fill="#333333")
    )  # fmt: skip
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n")
