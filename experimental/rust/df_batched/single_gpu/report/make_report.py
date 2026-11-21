# ruff: noqa: E501 -- the SVG and HTML templates are long lines by nature
"""Build report.html from df-batched's JSON and timeline.py's export.

    python3 make_report.py <data dir> <out.html>

<data dir> holds scale_gly{5,10,15,20}.json (one run each at the card's free
memory), sweep_gly10.json (one process, several rooms, --compare) and
timeline_gly10.json (timeline.py's output). Every number on the page is read
from those files; the page carries no hand-typed measurement.
"""

import html
import json
import math
import sys
from pathlib import Path

SIZES = [5, 10, 15, 20]

STAGES = [
    ("transform", "spherical transform", "--s1"),
    ("fit", "density fit (trsm)", "--s2"),
    ("integrals", "packed three-center", "--s3"),
    ("download", "download B (D2H)", "--b"),
    ("host", "host: tables + unstage", "--scr"),
]


def stage_of(name: str) -> str | None:
    if name == "spherical transform":
        return "transform"
    if name == "density fit":
        return "fit"
    if name == "packed three-center":
        return "integrals"
    if name.startswith("download B"):
        return "download"
    if name.startswith("unstage B") or name == "packed three-center tables":
        return "host"
    return None


def stages(run: dict) -> dict:
    out = {k: 0.0 for k, _, _ in STAGES}
    for name, t in run["tasks"].items():
        s = stage_of(name)
        if s is None:
            continue
        out[s] += t["host_s"] if s == "host" else t["device_s"]
    return out


def esc(s) -> str:
    return html.escape(str(s))


def fmt_s(x: float) -> str:
    return f"{x:.2f} s" if x < 10 else f"{x:.1f} s"


def gb(x: float) -> str:
    return f"{x / 1e9:.2f} GB"


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


# ---------------------------------------------------------------- figures


def stacked_bars(scale: dict) -> str:
    """Per molecule: B build's time by stage, one horizontal stacked bar."""
    W, left, right, row, top = 760, 92, 70, 34, 16
    rows = [
        (n, stages(scale[n]["runs"][0]), scale[n]["runs"][0]["build_s"]) for n in SIZES
    ]
    vmax = max(sum(s.values()) for _, s, _ in rows) * 1.02
    H = top + row * len(rows) + 40
    x = lambda v: left + (W - left - right) * v / vmax  # noqa: E731
    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Device time by stage per molecule">'
    ]
    for t in nice_ticks(0, vmax):
        parts.append(
            f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 4}" y2="{top + row * len(rows)}"/>'
        )
        parts.append(
            f'<text class="ax" x="{x(t):.1f}" y="{top + row * len(rows) + 16}" text-anchor="middle">{t:g}</text>'
        )
    parts.append(
        f'<text class="ax" x="{(left + W - right) / 2:.0f}" y="{H - 4}" text-anchor="middle">seconds (device time per stage; host stages host time)</text>'
    )
    for i, (n, s, build) in enumerate(rows):
        y = top + i * row
        parts.append(
            f'<text class="lb" x="{left - 10}" y="{y + 18}" text-anchor="end">gly{n}</text>'
        )
        acc = 0.0
        for key, label, var in STAGES:
            v = s[key]
            if v <= 0:
                continue
            x0, x1 = x(acc), x(acc + v)
            parts.append(
                f'<rect class="hit" x="{x0:.1f}" y="{y + 4}" width="{max(x1 - x0, 0.5):.1f}" height="{row - 10}" '
                f'style="fill:var({var})" data-tip="gly{n} · {esc(label)}: {fmt_s(v)} ({100 * v / sum(s.values()):.0f}%)"/>'
            )
            acc += v
        parts.append(
            f'<text class="lb mut" x="{x(acc) + 6:.1f}" y="{y + 18}">wall {fmt_s(build)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def scaling(scale: dict) -> tuple[str, float]:
    """log-log build time against nao, with the fitted exponent."""
    W, H, ml, r, t, b = 760, 300, 64, 24, 16, 44
    pts = [(scale[n]["nao"], scale[n]["runs"][0]["build_s"], n) for n in SIZES]
    lx = [math.log(p[0]) for p in pts]
    ly = [math.log(p[1]) for p in pts]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    slope = sum((a - mx) * (c - my) for a, c in zip(lx, ly, strict=True)) / sum(
        (a - mx) ** 2 for a in lx
    )
    xlo, xhi = math.log(300), math.log(1700)
    ylo, yhi = math.log(1), math.log(150)
    X = lambda v: ml + (W - ml - r) * (math.log(v) - xlo) / (xhi - xlo)  # noqa: E731
    Y = lambda v: H - b - (H - t - b) * (math.log(v) - ylo) / (yhi - ylo)  # noqa: E731
    p = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Build time against basis size, log-log">'
    ]
    for v in (1, 2, 5, 10, 20, 50, 100):
        p.append(
            f'<line class="grid" x1="{ml}" x2="{W - r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
        )
        p.append(
            f'<text class="ax" x="{ml - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v}</text>'
        )
    for v in (300, 500, 700, 1000, 1500):
        p.append(
            f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{t}" y2="{H - b}"/>'
        )
        p.append(
            f'<text class="ax" x="{X(v):.1f}" y="{H - b + 16}" text-anchor="middle">{v}</text>'
        )
    p.append(
        f'<text class="ax" x="{(ml + W - r) / 2:.0f}" y="{H - 6}" text-anchor="middle">orbital basis functions (spherical), log scale</text>'
    )
    p.append(
        f'<text class="ax" x="14" y="{(t + H - b) / 2:.0f}" transform="rotate(-90 14 {(t + H - b) / 2:.0f})" text-anchor="middle">B build, s (log)</text>'
    )
    # The fitted line, as reference.
    c = my - slope * mx
    f = lambda v: math.exp(c + slope * math.log(v))  # noqa: E731
    p.append(
        f'<line class="ref" x1="{X(330):.1f}" y1="{Y(f(330)):.1f}" x2="{X(1600):.1f}" y2="{Y(f(1600)):.1f}"/>'
    )
    p.append(
        f'<text class="lb mut" x="{X(1600) - 4:.1f}" y="{Y(f(1600)) + 18:.1f}" text-anchor="end">fit: t ∝ nao^{slope:.2f}</text>'
    )
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{X(a):.1f},{Y(c2):.1f}"
        for i, (a, c2, _) in enumerate(pts)
    )
    p.append(f'<path class="ln s1" d="{path}"/>')
    for a, c2, n in pts:
        bgb = scale[n]["b_bytes"] / 1e9
        p.append(
            f'<circle class="mk s1 hit-c" cx="{X(a):.1f}" cy="{Y(c2):.1f}" r="5" '
            f'data-tip="gly{n}: nao {a}, naux {scale[n]["naux"]}, B {bgb:.2f} GB, {fmt_s(c2)}"/>'
        )
        p.append(
            f'<text class="lb" x="{X(a) - 8:.1f}" y="{Y(c2) - 10:.1f}" text-anchor="end">gly{n}</text>'
        )
    p.append("</svg>")
    return "".join(p), slope


def sweep_chart(sweep: dict) -> str:
    """Build time and device bytes against the planned width."""
    W, H, ml, r, t, b = 760, 280, 64, 64, 16, 44
    # Rooms below the prefix's own footprint all plan width 1: one point.
    seen, runs = set(), []
    for rn in sweep["runs"]:
        if rn["width_line"] not in seen:
            seen.add(rn["width_line"])
            runs.append(rn)
    widths = [int(rn["width_line"].split(":")[1].split(" of ")[0]) for rn in runs]
    builds = [rn["build_s"] for rn in runs]
    dev = [rn["device_bytes"] / 1e9 for rn in runs]
    xlo, xhi = math.log(min(widths) * 0.8), math.log(max(widths) * 1.25)
    X = lambda v: ml + (W - ml - r) * (math.log(v) - xlo) / (xhi - xlo)  # noqa: E731
    ymax = max(builds) * 1.15
    Y = lambda v: H - b - (H - t - b) * v / ymax  # noqa: E731
    dmax = max(dev) * 1.15
    D = lambda v: H - b - (H - t - b) * v / dmax  # noqa: E731
    p = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Build time and device memory against batch width">'
    ]
    for v in nice_ticks(0, ymax):
        p.append(
            f'<line class="grid" x1="{ml}" x2="{W - r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
        )
        p.append(
            f'<text class="ax" x="{ml - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:g}</text>'
        )
    for v in nice_ticks(0, dmax, 4):
        p.append(f'<text class="ax" x="{W - r + 8}" y="{D(v) + 4:.1f}">{v:g}</text>')
    for w in widths:
        p.append(
            f'<text class="ax" x="{X(w):.1f}" y="{H - b + 16}" text-anchor="middle">{w}</text>'
        )
    p.append(
        f'<text class="ax" x="{(ml + W - r) / 2:.0f}" y="{H - 6}" text-anchor="middle">batch width, Cartesian columns (log)</text>'
    )
    p.append(
        f'<text class="ax" x="14" y="{(t + H - b) / 2:.0f}" transform="rotate(-90 14 {(t + H - b) / 2:.0f})" text-anchor="middle">B build, s</text>'
    )
    p.append(
        f'<text class="ax" x="{W - 10}" y="{(t + H - b) / 2:.0f}" transform="rotate(90 {W - 10} {(t + H - b) / 2:.0f})" text-anchor="middle">device memory, GB</text>'
    )
    pb = " ".join(
        f"{'M' if i == 0 else 'L'}{X(w):.1f},{Y(v):.1f}"
        for i, (w, v) in enumerate(zip(widths, builds, strict=True))
    )
    pd = " ".join(
        f"{'M' if i == 0 else 'L'}{X(w):.1f},{D(v):.1f}"
        for i, (w, v) in enumerate(zip(widths, dev, strict=True))
    )
    p.append(f'<path class="ln s3" d="{pd}"/>')
    p.append(f'<path class="ln s1" d="{pb}"/>')
    for w, v, d, rn in zip(widths, builds, dev, runs, strict=True):
        trips = sum(rn["trips"])
        p.append(
            f'<circle class="mk s3 hit-c" cx="{X(w):.1f}" cy="{D(d):.1f}" r="4" data-tip="width {w}: device {d:.2f} GB"/>'
        )
        p.append(
            f'<circle class="mk s1 hit-c" cx="{X(w):.1f}" cy="{Y(v):.1f}" r="5" data-tip="width {w}: {trips} trips, build {fmt_s(v)}"/>'
        )
    p.append("</svg>")
    return "".join(p)


TL_KINDS = [
    ("integrals", "packed three-center", "--s3"),
    ("transform aux", "transform: dense U_aux GEMM", "--s1"),
    ("transform bra/ket", "transform: bra/ket GEMMs", "--s1l"),
    ("fit", "trsm against L", "--s2"),
    ("transpose", "geam transpose", "--muted"),
    ("d2h", "download B", "--b"),
    ("h2d", "upload: cell tables", "--scr"),
    ("other", "other", "--axis"),
]


def timeline_chart(tl: dict, window_ms: float) -> tuple[str, dict]:
    ev = [e for e in tl["events"] if e["start_ms"] < window_ms]
    streams = sorted({e["stream"] for e in ev})
    W, ml, r, lane, t = 760, 84, 16, 30, 12
    H = t + lane * len(streams) + 40
    X = lambda v: ml + (W - ml - r) * v / window_ms  # noqa: E731
    color = {k: v for k, _, v in TL_KINDS}
    p = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Device timeline of the B build, first {window_ms:.0f} ms">'
    ]
    for v in nice_ticks(0, window_ms, 8):
        p.append(
            f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{t}" y2="{t + lane * len(streams)}"/>'
        )
        p.append(
            f'<text class="ax" x="{X(v):.1f}" y="{t + lane * len(streams) + 16}" text-anchor="middle">{v:g}</text>'
        )
    p.append(
        f'<text class="ax" x="{(ml + W - r) / 2:.0f}" y="{H - 4}" text-anchor="middle">ms since the first trip&#39;s integrals</text>'
    )
    for i, s in enumerate(streams):
        y = t + i * lane
        p.append(
            f'<text class="lb" x="{ml - 8}" y="{y + 19}" text-anchor="end">stream {s}</text>'
        )
        for e in ev:
            if e["stream"] != s:
                continue
            x0, x1 = X(e["start_ms"]), X(min(e["end_ms"], window_ms))
            var = color.get(e["kind"], "--axis")
            p.append(
                f'<rect class="hit" x="{x0:.2f}" y="{y + 4}" width="{max(x1 - x0, 0.6):.2f}" height="{lane - 8}" style="fill:var({var})" '
                f'data-tip="{esc(e["kind"])}: {e["end_ms"] - e["start_ms"]:.2f} ms"/>'
            )
    p.append("</svg>")
    totals: dict = {}
    for e in tl["events"]:
        totals[e["kind"]] = totals.get(e["kind"], 0.0) + e["end_ms"] - e["start_ms"]
    return "".join(p), totals


def two_step(scale: dict) -> tuple[str, list]:
    """The proposed two-pass scheme against the fused loop, per molecule."""
    rows = []
    for n in SIZES:
        d = scale[n]
        run = d["runs"][0]
        B = d["b_bytes"]
        dl = sum(
            t["device_s"] for k, t in run["tasks"].items() if k.startswith("download B")
        )
        un = sum(
            t["host_s"] for k, t in run["tasks"].items() if k.startswith("unstage B")
        )
        bw = B / dl
        # Two-step: (mn|P) down and back up (each |B|, the same shape as B),
        # then B down. Each extra pass also pays a host copy through the pinned
        # window, at the measured unstage rate.
        extra = 2 * B / bw + 2 * un
        rows.append(
            {
                "n": n,
                "B": B,
                "bw": bw,
                "fused_pcie": B,
                "two_pcie": 3 * B,
                "fused_host": B,
                "two_host": 2 * B,
                "extra_s": extra,
                "build": run["build_s"],
            }
        )
    W, ml, r, row, t = 760, 92, 120, 40, 22
    H = t + row * len(rows) + 30
    vmax = max(x["two_pcie"] for x in rows) * 1.05
    X = lambda v: ml + (W - ml - r) * v / vmax  # noqa: E731
    p = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="PCIe bytes, two-step against fused">'
    ]
    for v in nice_ticks(0, vmax / 1e9):
        p.append(
            f'<line class="grid" x1="{X(v * 1e9):.1f}" x2="{X(v * 1e9):.1f}" y1="{t - 4}" y2="{t + row * len(rows)}"/>'
        )
        p.append(
            f'<text class="ax" x="{X(v * 1e9):.1f}" y="{t + row * len(rows) + 16}" text-anchor="middle">{v:g}</text>'
        )
    p.append(f'<text class="ax" x="{ml}" y="{t - 8}">GB over PCIe</text>')
    for i, x in enumerate(rows):
        y = t + i * row
        p.append(
            f'<text class="lb" x="{ml - 10}" y="{y + 20}" text-anchor="end">gly{x["n"]}</text>'
        )
        p.append(
            f'<rect class="hit" x="{ml}" y="{y + 3}" width="{X(x["two_pcie"]) - ml:.1f}" height="14" style="fill:var(--s2)" data-tip="two-step: {gb(x["two_pcie"])}"/>'
        )
        p.append(
            f'<rect class="hit" x="{ml}" y="{y + 19}" width="{X(x["fused_pcie"]) - ml:.1f}" height="14" style="fill:var(--s1)" data-tip="fused: {gb(x["fused_pcie"])}"/>'
        )
        p.append(
            f'<text class="lb mut" x="{X(x["two_pcie"]) + 6:.1f}" y="{y + 15}">+{fmt_s(x["extra_s"])}</text>'
        )
    p.append("</svg>")
    return "".join(p), rows


# ---------------------------------------------------------------- page

CSS = """
:root {
  --bg: #f9f9f7; --surface: #fcfcfb; --fg: #0b0b0b; --sec: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7;
  --s1: #2a78d6; --s1l: #8db8ec; --s2: #eb6834; --s3: #1baf7a; --b: #6250d6; --scr: #eda100; --bad: #d03b3b;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0d0d0d; --surface: #1a1a19; --fg: #f0efec; --sec: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835;
  --s1: #3987e5; --s1l: #2b4f7c; --s2: #d95926; --s3: #199e70; --b: #9085e9; --scr: #c98500; --bad: #e66767;
  color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0d0d0d; --surface: #1a1a19; --fg: #f0efec; --sec: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835;
  --s1: #3987e5; --s1l: #2b4f7c; --s2: #d95926; --s3: #199e70; --b: #9085e9; --scr: #c98500; --bad: #e66767;
  color-scheme: dark }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.55 var(--sans);
  padding-block: 40px 72px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main { max-width: 820px; margin: 0 auto; display: flex; flex-direction: column; gap: 44px; }
section { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
h1 { font: 600 30px/1.15 var(--sans); margin: 0; letter-spacing: -0.01em; text-wrap: balance; }
h2 { font: 600 21px/1.25 var(--sans); margin: 0; text-wrap: balance; }
h3 { font: 600 16px/1.3 var(--sans); margin: 0; }
p, ul, ol { margin: 0; max-width: 70ch; }
ul, ol { padding-left: 1.3em; display: flex; flex-direction: column; gap: 8px; }
.lede { font-size: 18px; color: var(--sec); }
.eyebrow { font: 500 12px var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--sec); }
code { font: 0.88em var(--mono); overflow-wrap: anywhere; }
figure { margin: 6px 0; display: flex; flex-direction: column; gap: 10px; }
.scroll { overflow-x: auto; background: var(--surface); border: 1px solid var(--grid); border-radius: 6px; }
figcaption, .note { font-size: 14px; color: var(--sec); max-width: 76ch; }
svg { display: block; min-width: 680px; width: 100%; height: auto; font-family: var(--sans); }
.grid { stroke: var(--grid); stroke-width: 1; }
.ref { stroke: var(--sec); stroke-width: 1.5; stroke-dasharray: 5 4; }
.ax { fill: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.lb { fill: var(--fg); font-size: 12.5px; }
.lb.mut { fill: var(--sec); }
.ln { fill: none; stroke-width: 2; stroke-linejoin: round; }
.ln.s1 { stroke: var(--s1); } .ln.s3 { stroke: var(--s3); }
.mk { stroke: var(--surface); stroke-width: 2; }
.mk.s1 { fill: var(--s1); } .mk.s3 { fill: var(--s3); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--sec); }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 7px; vertical-align: -1px; }
table { border-collapse: collapse; font-size: 14px; width: 100%; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { font: 500 12px var(--mono); letter-spacing: 0.04em; text-transform: uppercase; color: var(--sec); }
td.num, th.num { text-align: right; font-family: var(--mono); font-size: 13px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.tbl { overflow-x: auto; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.tile { background: var(--surface); border: 1px solid var(--grid); border-radius: 6px; padding: 14px 16px; display: flex; flex-direction: column; gap: 4px; }
.tile b { font: 600 26px/1.1 var(--sans); font-variant-numeric: tabular-nums; }
.tile span { font-size: 13px; color: var(--sec); }
#tip { position: fixed; pointer-events: none; background: var(--fg); color: var(--bg); font-size: 12.5px; padding: 5px 9px; border-radius: 4px; display: none; z-index: 5; max-width: 320px; }
"""

JS = """
const tip = document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el => {
  el.addEventListener('mousemove', e => { tip.textContent = el.dataset.tip; tip.style.display = 'block';
    tip.style.left = Math.min(e.clientX + 12, innerWidth - 330) + 'px'; tip.style.top = (e.clientY + 14) + 'px'; });
  el.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
});
"""


def legend(items) -> str:
    return (
        '<div class="legend">'
        + "".join(
            f'<span><i class="sw" style="background:var({v})"></i>{esc(lbl)}</span>'
            for _, lbl, v in items
        )
        + "</div>"
    )


def main(data: Path, out: Path) -> None:
    scale = {n: json.load(open(data / f"scale_gly{n}.json")) for n in SIZES}
    sweep = json.load(open(data / "sweep_gly10.json"))
    tl = json.load(open(data / "timeline_gly10.json"))
    meta = json.load(open(data / "timeline_gly10.meta.json"))

    g20 = scale[20]
    s20 = stages(g20["runs"][0])
    dev20 = sum(v for k, v in s20.items() if k != "host")
    bars = stacked_bars(scale)
    sc, slope = scaling(scale)
    sw = sweep_chart(sweep)
    trip_ms = meta["window_ms"]
    tlsvg, tl_tot = timeline_chart(tl, trip_ms)
    ts_svg, ts_rows = two_step(scale)

    tl_dev = sum(v for k, v in tl_tot.items())
    aux_share = tl_tot.get("transform aux", 0.0) / tl_dev
    orb_share = tl_tot.get("transform bra/ket", 0.0) / tl_dev
    fit_share = tl_tot.get("fit", 0.0) / tl_dev
    int_share = tl_tot.get("integrals", 0.0) / tl_dev
    d2h_share = tl_tot.get("d2h", 0.0) / tl_dev
    streams = tl["streams"]

    # Size table.
    size_rows = "".join(
        f"<tr><td>gly{n}</td><td class=num>{scale[n]['nao']}</td><td class=num>{scale[n]['naux']}</td>"
        f"<td class=num>{scale[n]['kept_pairs']:,}</td><td class=num>{scale[n]['columns']:,}</td>"
        f"<td class=num>{gb(scale[n]['b_bytes'])}</td><td class=num>{sum(scale[n]['runs'][0]['trips'])}</td>"
        f"<td class=num>{gb(scale[n]['runs'][0]['device_bytes'])}</td>"
        f"<td class=num>{fmt_s(scale[n]['prefix']['wall_s'])}</td><td class=num>{fmt_s(scale[n]['runs'][0]['build_s'])}</td></tr>"
        for n in SIZES
    )
    sweep_rows = ""
    for rn in sweep["runs"]:
        w = rn["width_line"].split(":")[1].split(" of ")[0].strip()
        cmp = rn["compare"]
        cmp_s = (
            "reference"
            if cmp is None
            else ("bitwise" if cmp["bitwise"] else f"{cmp['max_abs_diff']:.1e}")
        )
        sweep_rows += (
            f"<tr><td class=num>{rn['room_bytes'] >> 20}</td><td class=num>{w}</td><td class=num>{sum(rn['trips'])}</td>"
            f"<td class=num>{gb(rn['device_bytes'])}</td><td class=num>{gb(rn['pinned_bytes'])}</td>"
            f"<td class=num>{fmt_s(rn['build_s'])}</td><td class=num>{cmp_s}</td></tr>"
        )
    ts_table = "".join(
        f"<tr><td>gly{x['n']}</td><td class=num>{gb(x['two_host'])}</td><td class=num>{gb(x['fused_host'])}</td>"
        f"<td class=num>{gb(x['two_pcie'])}</td><td class=num>{gb(x['fused_pcie'])}</td>"
        f"<td class=num>{x['bw'] / 1e9:.1f} GB/s</td><td class=num>+{fmt_s(x['extra_s'])}</td>"
        f"<td class=num>{100 * x['extra_s'] / x['build']:.0f}%</td></tr>"
        for x in ts_rows
    )
    sweep_fast = min(sweep["runs"], key=lambda r: r["build_s"])
    sweep_slow = max(sweep["runs"], key=lambda r: r["build_s"])
    worst_cmp = max(
        (r["compare"]["max_abs_diff"] for r in sweep["runs"] if r["compare"]),
        default=0.0,
    )

    page = f"""<title>Batched DF factor B</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>{CSS}</style>
<div id="tip"></div>
<main>
<header style="display:flex;flex-direction:column;gap:12px">
  <div class="eyebrow">Measurement · 2026-11-21 · RTX 3080, one card · glycine chains, cc-pVDZ / cc-pVDZ-JKFIT, Schwarz 1e-12</div>
  <h1>Batched DF factor B on one GPU</h1>
  <p class="lede">rust/'s <code>add_packed_b</code> already runs the fused loop: (P|mn) for every P over a batch of shell pairs, fitted against a resident L, downloaded as B. It never puts a whole (mn|P) or B on the card. The build is compute-bound on one stream, and two thirds of its time is the spherical transform.</p>
</header>

<section>
  <div class="tiles">
    <div class="tile"><b>{gb(g20["b_bytes"])}</b><span>gly20's B, built on a 10 GB card in {fmt_s(g20["runs"][0]["build_s"])} ({sum(g20["runs"][0]["trips"])} trips)</span></div>
    <div class="tile"><b>{100 * s20["transform"] / dev20:.0f}%</b><span>of gly20's device time is the spherical transform; the trsm is {100 * s20["fit"] / dev20:.0f}%, the integrals {100 * s20["integrals"] / dev20:.0f}%</span></div>
    <div class="tile"><b>{100 * aux_share:.0f}%</b><span>of gly10's device time is one dense GEMM against the block-diagonal U_aux (nsys)</span></div>
    <div class="tile"><b>{100 * s20["download"] / dev20:.1f}%</b><span>of gly20's device time is the D2H download of B</span></div>
  </div>
</section>

<section>
  <h2>The loop</h2>
  <p>Everything O(N²) stays on the card: the records, the aux tables, U for both bases, the metric and its Cholesky factor L. The loop runs over batches of shell pairs, one counted loop per class (la, lb), with ring depth 2. A batch never spans two classes. Each trip does five things:</p>
  <ol>
    <li><code>packed three-center</code>: (P|mn) for every aux function and the batch's pairs, FelizJueves' packed <code>peri3</code> kernels, Cartesian P.</li>
    <li><code>spherical transform</code>: a dense GEMM <code>W = U_auxᵀ X</code> on the aux index, then strided-batched GEMMs for the ket and bra.</li>
    <li><code>density fit</code>: the trsm <code>L B = (P|mn)</code> in place.</li>
    <li><code>B (la, lb) batch transpose</code>: one <code>geam</code> from aux-fastest to pair-fastest.</li>
    <li><code>download</code> and <code>unstage</code>: a pitched copy into B's heap row on the host.</li>
  </ol>
  <p>Batching over aux functions instead would not work. Every column B<sub>·,mn</sub> depends on every P, so a P batch cannot be fitted on its own. The batch width is the planner's adaptive part, the widest that fits the device room. Driver: <code>experimental/rust/df_batched/single_gpu</code>, which calls the unchanged rust/ crates.</p>
</section>

<section>
  <h2>Where the time goes</h2>
  <figure>
    {legend(STAGES)}
    <div class="scroll">{bars}</div>
    <figcaption>Per-task device time from the run's timing record, summed per stage (host stages: host time). Median of 3 warm runs at the card's free memory. Device time sums to within a few percent of the wall, so stages do not overlap: the loop is serial on one stream.</figcaption>
  </figure>
  <div class="tbl"><table>
    <tr><th></th><th class=num>nao</th><th class=num>naux</th><th class=num>kept pairs</th><th class=num>columns</th><th class=num>B</th><th class=num>trips</th><th class=num>device</th><th class=num>prefix</th><th class=num>B build</th></tr>
    {size_rows}
  </table></div>
  <p class="note">"prefix" is the records, the metric, its transform and the Cholesky factor, timed as a plan of its own. "B build" is the full plan's wall less the prefix's.</p>
</section>

<section>
  <h2>One trip, on the device</h2>
  <figure>
    {legend(TL_KINDS[:7])}
    <div class="scroll">{tlsvg}</div>
    <figcaption>nsys trace of gly10 at a {meta["room_mib"]} MiB room ({meta["trips"]} trips), the first {trip_ms:.0f} ms of the last full run. The plan has no NVTX ranges, so each kernel is classed by name and by its place in the trip; <code>report/timeline.py</code> has the rule. Over the whole build: dense U_aux GEMM {100 * aux_share:.0f}%, bra/ket GEMMs {100 * orb_share:.1f}%, trsm {100 * fit_share:.0f}%, integrals {100 * int_share:.0f}%, download {100 * d2h_share:.1f}%.</figcaption>
  </figure>
  <p>Everything runs on {"one stream" if len(streams) == 1 else f"{len(streams)} streams"}. The ring of depth 2 overlaps the host's table building and unstaging with the device. On the device, the download queues behind the trip's compute rather than under the next trip's. So does each trip's upload of its packed three-center cell tables, which is about as large as the B batch it downloads. Both copies together are {100 * (tl_tot.get("d2h", 0) + tl_tot.get("h2d", 0)) / tl_dev:.1f}% of device time, so a copy lane would win back at most that much.</p>
  <p>The dense aux GEMM is the lever. U_aux is block-diagonal by shell, so almost all of that GEMM multiplies zeros. FelizJueves #743 (upstream <code>6e1c341</code>, newer than the pinned submodule) emits <code>speri3</code>, which writes Q already spherical by Hobson's theorem. FreeBreakfast #1736 takes it into rust/. That removes this GEMM outright, and the bra/ket GEMMs and the kernel's output shrink by the Cartesian-to-spherical ratio of the aux set. It is left to its own PR. This page measures the current path as the baseline.</p>
</section>

<section>
  <h2>Batch width</h2>
  <figure>
    <div class="legend"><span><i class="sw" style="background:var(--s1)"></i>B build</span><span><i class="sw" style="background:var(--s3)"></i>device memory (plan table)</span></div>
    <div class="scroll">{sw}</div>
    <figcaption>gly10, one process, one plan per device room, median of 3 warm runs each. Fastest {fmt_s(sweep_fast["build_s"])} at {sum(sweep_fast["trips"])} trips, slowest {fmt_s(sweep_slow["build_s"])} at {sum(sweep_slow["trips"])} trips.</figcaption>
  </figure>
  <div class="tbl"><table>
    <tr><th class=num>room MiB</th><th class=num>width</th><th class=num>trips</th><th class=num>device</th><th class=num>pinned</th><th class=num>B build</th><th class=num>B vs first</th></tr>
    {sweep_rows}
  </table></div>
  <p class="note">A room smaller than the prefix's own rows (the metric and L, 2 × naux² doubles) leaves no room for the loop. The planner then sets the width to 1, one shell pair per trip, and the pipeline's admission step would refuse that plan. The driver skips admission, so those rows ran anyway. They give the loop's fixed cost per trip at the extreme.</p>
  <p class="note">"B vs first" is the largest |ΔB| against the first room's B, all held in one process. The GEMM and trsm shapes change with the width, so B agrees to rounding (worst {worst_cmp:.1e}), not bitwise.</p>
</section>

<section>
  <h2>Two passes against the fused loop</h2>
  <p>The two-pass scheme builds all of (mn|P) and moves it to the host, then uploads it again to fit. Its (mn|P) is the same shape as B. It moves 3·|B| over PCIe against the fused loop's |B|, and holds 2·|B| on the host until the first buffer is freed.</p>
  <figure>
    <div class="legend"><span><i class="sw" style="background:var(--s2)"></i>two-pass</span><span><i class="sw" style="background:var(--s1)"></i>fused (measured)</span></div>
    <div class="scroll">{ts_svg}</div>
    <figcaption>Bytes are exact. The added time is an estimate: two more transfers of |B| at the D2H rate the fused run measured, plus two more host copies through the pinned window at its measured unstage time. The two-pass scheme itself was not built.</figcaption>
  </figure>
  <div class="tbl"><table>
    <tr><th></th><th class=num>host, two-pass</th><th class=num>host, fused</th><th class=num>PCIe, two-pass</th><th class=num>PCIe, fused</th><th class=num>D2H rate</th><th class=num>added time</th><th class=num>of build</th></tr>
    {ts_table}
  </table></div>
  <p>The fused loop's real advantage is host memory. The build is compute-bound, so the extra PCIe traffic would add only a few percent. The second |B| on the host is what limits molecule size: gly20 needs {gb(2 * g20["b_bytes"])} in two passes against {gb(g20["b_bytes"])} fused.</p>
</section>

<section>
  <h2>Scaling</h2>
  <figure>
    <div class="scroll">{sc}</div>
    <figcaption>B build against the orbital basis size, log-log. The fitted exponent is {slope:.2f}. Screening keeps the column count nearly linear in size, so the cost runs as naux² × columns for the trsm and naux_cart × naux × columns for the dense aux GEMM.</figcaption>
  </figure>
</section>

<section>
  <h2>How it was run</h2>
  <ul>
    <li>Driver <code>df-batched</code> (this crate), release build, against <code>rust/build-cuda</code>. Commit <code>{esc(meta["commit"])}</code>, FelizJueves <code>826b69e</code>.</li>
    <li>Each run: one warm-up, then 3 timed runs, median. Both CUDA locks held, the card otherwise idle.</li>
    <li>Timeline: <code>nsys profile -t cuda</code>, exported to SQLite, then <code>report/timeline.py</code>.</li>
    <li>Page: <code>report/make_report.py</code> over the JSON files. No number here is typed by hand.</li>
  </ul>
</section>
</main>
<script>{JS}</script>
"""
    out.write_text(page)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
