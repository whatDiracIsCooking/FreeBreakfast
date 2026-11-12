"""A contour plot of a relaxed (R16, R34) grid as stdlib SVG (no matplotlib in
uv.lock; step 2's plot.py is the precedent): filled cells coloured by
E - E_min, contour lines by marching squares at a fixed interval, the grid's
stationary points marked, and an optional path (the IRC) overlaid."""

from __future__ import annotations

from pathlib import Path

KCAL = 627.5094740631
SIZE, LEFT, TOP, PAD = 560, 70, 40, 150
STEP_KCAL = 2.0  # contour interval
CAP_KCAL = 40.0  # the colour scale saturates here
TITLE = {"pc": "DF-PC-NEVPT2", "casscf": "DF-CASSCF(6,6)"}


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


def marker(kind: str, x: float, y: float) -> str:
    """A minimum a circle, a saddle a diamond, a maximum a square."""
    if kind == "minimum":
        return el("circle", cx=f"{x:.2f}", cy=f"{y:.2f}", r=5, fill="white",
                  stroke="black")  # fmt: skip
    turn = f"rotate(45 {x:.2f} {y:.2f})" if kind == "saddle" else "rotate(0)"
    return el("rect", x=f"{x - 5:.2f}", y=f"{y - 5:.2f}", width=10, height=10,
              fill="white", stroke="black", transform=turn)  # fmt: skip


def plot(path: Path, method: str, surface: dict, irc, summary: dict) -> None:
    g = surface["grid"]
    e = surface["energy"]
    emin = min(e.values())
    de = {k: (v - emin) * KCAL for k, v in e.items()}
    top = max(de.values())
    cap = min(top, CAP_KCAL)
    h = g[1] - g[0]
    lo, hi = g[0] - h / 2, g[-1] + h / 2
    span = hi - lo
    width, height = LEFT + SIZE + PAD, TOP + SIZE + 60

    def px(r):
        return LEFT + (r - lo) / span * SIZE

    def py(r):
        return TOP + SIZE - (r - lo) / span * SIZE

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
    cell = SIZE / len(g)
    for (a, b), v in de.items():
        out.append(
            el("rect", x=f"{px(a) - cell / 2:.2f}", y=f"{py(b) - cell / 2:.2f}",
               width=f"{cell:.2f}", height=f"{cell:.2f}", fill=colour(v / cap))
        )  # fmt: skip
    level = STEP_KCAL
    while level < top:
        for p, q in segments(g, de, level):
            out.append(
                el("line", x1=f"{px(p[0]):.2f}", y1=f"{py(p[1]):.2f}",
                   x2=f"{px(q[0]):.2f}", y2=f"{py(q[1]):.2f}", stroke="white",
                   stroke_width=0.8, stroke_opacity=0.8)
            )  # fmt: skip
        level += STEP_KCAL
    out.append(
        el("line", x1=px(lo), y1=py(lo), x2=px(hi), y2=py(hi), stroke="black",
           stroke_dasharray="3 3", stroke_width=0.6)
    )  # fmt: skip
    if irc:
        inside = [(a, b) for a, b in irc if lo <= a <= hi and lo <= b <= hi]
        pts = " ".join(f"{px(a):.2f},{py(b):.2f}" for a, b in inside)
        out.append(
            el("polyline", points=pts, fill="none", stroke="#d62728", stroke_width=2)
        )
    for s in summary.get("grid_stationary_points", []):
        for a, b in {(s["r16"], s["r34"]), (s["r34"], s["r16"])}:
            out.append(marker(s["kind"], px(a), py(b)))
    out.append(
        el("rect", x=LEFT, y=TOP, width=SIZE, height=SIZE, fill="none", stroke="black")
    )
    for r in g:
        if round(r * 100) % 20:
            continue  # label every 0.2 A
        out.append(
            el("text", f"{r:.1f}", x=f"{px(r):.1f}", y=TOP + SIZE + 16,
               text_anchor="middle")
        )  # fmt: skip
        out.append(
            el("text", f"{r:.1f}", x=LEFT - 6, y=f"{py(r) + 4:.1f}", text_anchor="end")
        )
    out.append(
        el("text", "R16 / Angstrom", x=LEFT + SIZE / 2, y=TOP + SIZE + 38,
           text_anchor="middle")
    )  # fmt: skip
    mid = TOP + SIZE / 2
    out.append(
        el("text", "R34 / Angstrom", x=20, y=mid, transform=f"rotate(-90 20 {mid})",
           text_anchor="middle")
    )  # fmt: skip
    # the colour bar (its upper half) and the legend
    x0 = LEFT + SIZE + 30
    for k in range(100):
        y = TOP + SIZE - (k + 1) * SIZE / 200
        out.append(
            el("rect", x=x0, y=f"{y:.2f}", width=16, height=f"{SIZE / 200 + 0.5:.2f}",
               fill=colour(k / 99))
        )  # fmt: skip
    for v in range(0, int(cap) + 1, 10):
        y = TOP + SIZE - v / cap * SIZE / 2
        out.append(el("text", str(v), x=x0 + 22, y=f"{y + 4:.1f}"))
    out.append(el("text", "kcal/mol", x=x0, y=TOP + SIZE / 2 - 12))
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
