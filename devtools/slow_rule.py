#!/usr/bin/env python3
"""The slow-test rule, enforced (#987; docs/testing.md "The slow-test rule,
enforced").

Reads the latest nightly or weekly record of this host from the `metrics`
branch, and the series of the ones before it, and reports two things:

- **Over the line:** a fast-set gtest case that took more than LINE_SECONDS
  (test/support/slow_test.h's 1.5 s on an idle RTX 3080) and is not its
  family's representative, nor in an exempt family. The families are
  devtools/test_families.py's, read from a built build dir. A case is in the
  fast set when its record says `slow: false`: FBF_SKIP_IF_FAST_TESTS records
  the gtest property `fbf_slow` in every mode, and metrics.gtest_cases reads
  it. A case run by ctest (a host binary, a RUN_SERIAL case) has no gtest
  report, so its membership is not recorded; one over the line is listed
  apart, as unknown.
- **A step:** a case whose time stepped up against its own recent history,
  past a bar set by the series' spread: the median of its passes over the
  last WINDOW records, plus K robust standard deviations. The deviation is
  the larger of the case's own (1.4826 x its median absolute deviation), the
  box's (the median over the cases of that, relative to their medians, times
  this case's median), and the reports' resolution. A case with fewer than
  MIN_HISTORY passes in the window is not judged.

docs/testing.md has where each constant comes from. Neither finding fails the
night: they are a report, written into the nightly's log and summary, and as
JSON beside its per-case reports, `<log>.results/slow-rule.json`, which is
what the nightly issue (#986) takes.

  devtools/slow_rule.py BUILD_DIR [--metrics FILE] [--host H] [--json]

Standard library only: it runs under a bare python3, inside test.py nightly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCHEMA = 1
LINE_SECONDS = 1.5  # test/support/slow_test.h
WINDOW = 14  # records: two weeks of nights, one weekly among them
MIN_HISTORY = 7  # passes in the window; half of it, the median's breakdown point
K = 5.0  # robust standard deviations above the median
MAD_TO_SIGMA = 1.4826  # 1 / Phi^-1(3/4): the MAD of a normal sample, as its sigma
RESOLUTION = 0.001  # s: gtest's JSON times are whole milliseconds
# A case's relative spread joins the box's only where the resolution is at most
# 1% of its time, so quantization does not pass for noise.
POOL_MIN_SECONDS = 100 * RESOLUTION
SERIES_KINDS = ("nightly", "weekly")
_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    """A sibling tool, by path: under `python3 -I` this directory is not on
    sys.path. Registered, so its dataclasses resolve their module."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _families_tool():
    return _load("fbf_test_families", _ROOT / "devtools" / "test_families.py")


# --- the series --------------------------------------------------------------


def series(text: str, host: str) -> list[dict]:
    """This host's nightly and weekly records, schema 1, oldest first."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            rec.get("schema") == 1
            and rec.get("kind") in SERIES_KINDS
            and rec.get("host") == host
        ):
            out.append(rec)
    out.sort(key=lambda r: r.get("started", ""))
    return out


def _passes(record: dict) -> dict[str, float]:
    return {
        c["name"]: float(c["seconds"])
        for c in record.get("cases", [])
        if c.get("status") == "pass" and isinstance(c.get("seconds"), int | float)
    }


# --- the two checks ----------------------------------------------------------


def over_line(record: dict, fams: dict) -> tuple[list[dict], list[dict]]:
    """(fast-set cases over the line, cases over it whose membership the record
    does not carry). Only gtest cases a family names count: pytest and the
    examples are outside slow_test.h's rule."""
    family_of = {m: f for f in fams.values() for m in f.members}
    over, unknown = [], []
    for c in record.get("cases", []):
        fam = family_of.get(c.get("name"))
        seconds = c.get("seconds")
        if fam is None or c.get("status") != "pass":
            continue
        if not isinstance(seconds, int | float) or seconds <= LINE_SECONDS:
            continue
        if fam.by == "exempt" or fam.representative == c["name"]:
            continue
        row = {"name": c["name"], "family": fam.name, "seconds": float(seconds)}
        if c.get("slow") is False:
            over.append(row)
        elif c.get("slow") is None:
            unknown.append(row)
    key = lambda r: (-r["seconds"], r["name"])  # noqa: E731
    return sorted(over, key=key), sorted(unknown, key=key)


def _centre(window: list[float]) -> tuple[float, float]:
    median = statistics.median(window)
    return median, statistics.median(abs(x - median) for x in window)


def pooled_spread(histories: dict[str, list[float]]) -> float:
    """The box's relative spread: the median, over the cases with a full
    enough window and a time well above the resolution, of each one's robust
    sigma over its median. 0 where no case qualifies."""
    ratios = []
    for history in histories.values():
        window = history[-WINDOW:]
        if len(window) < MIN_HISTORY:
            continue
        median, mad = _centre(window)
        if median >= POOL_MIN_SECONDS:
            ratios.append(MAD_TO_SIGMA * mad / median)
    return statistics.median(ratios) if ratios else 0.0


def step(history: list[float], seconds: float, pooled: float = 0.0) -> dict | None:
    """The bar a time is judged against, and whether it is past it; None when
    the history is too short to judge. `pooled` is pooled_spread's."""
    window = history[-WINDOW:]
    if len(window) < MIN_HISTORY:
        return None
    median, mad = _centre(window)
    sigma = max(MAD_TO_SIGMA * mad, pooled * median, RESOLUTION)
    bar = median + K * sigma
    return {
        "seconds": seconds,
        "median": median,
        "sigma": sigma,
        "bar": bar,
        "nights": len(window),
        "stepped": seconds > bar,
    }


def steps(records: list[dict]) -> tuple[list[dict], int, int, float]:
    """(the latest record's steps, cases judged, cases with too short a
    history, the pooled relative spread). The window is the WINDOW records
    before the latest."""
    if not records:
        return [], 0, 0, 0.0
    current, prior = records[-1], records[:-1][-WINDOW:]
    history: dict[str, list[float]] = defaultdict(list)
    for rec in prior:
        for name, seconds in _passes(rec).items():
            history[name].append(seconds)
    pooled = pooled_spread(history)
    flagged, judged, short = [], 0, 0
    for name, seconds in sorted(_passes(current).items()):
        verdict = step(history.get(name, []), seconds, pooled)
        if verdict is None:
            short += 1
            continue
        judged += 1
        if verdict.pop("stepped"):
            flagged.append({"name": name, **verdict})
    flagged.sort(key=lambda r: (-(r["seconds"] - r["median"]), r["name"]))
    return flagged, judged, short, pooled


# --- the report --------------------------------------------------------------


def check(text: str, host: str, fams: dict) -> dict | None:
    """The findings for the latest record of `host` in `text` (the metrics
    branch's records), or None when it has no nightly or weekly record."""
    records = series(text, host)
    if not records:
        return None
    current = records[-1]
    over, unknown = over_line(current, fams)
    stepped, judged, short, pooled = steps(records)
    return {
        "schema": SCHEMA,
        "host": host,
        "commit": current.get("commit"),
        "kind": current.get("kind"),
        "started": current.get("started"),
        "line_seconds": LINE_SECONDS,
        "window": WINDOW,
        "min_history": MIN_HISTORY,
        "k": K,
        "pooled_relative_sigma": pooled,
        "over_line": over,
        "over_line_unrecorded": unknown,
        "steps": stepped,
        "judged": judged,
        "too_short": short,
    }


def render(findings: dict) -> list[str]:
    f = findings
    commit = (f.get("commit") or "?")[:12]
    out = [
        f"slow-rule: {commit} ({f['kind']}, {f['started']}), host {f['host']!r},"
        f" line {f['line_seconds']} s"
    ]
    out.append(
        f"slow-rule: {len(f['over_line'])} fast-set case(s) over the line,"
        " not their family's representative" + (":" if f["over_line"] else "")
    )
    out += [
        f"  {r['seconds']:9.3f} s  {r['name']}  (family {r['family']})"
        for r in f["over_line"]
    ]
    if f["over_line_unrecorded"]:
        out.append(
            f"slow-rule: {len(f['over_line_unrecorded'])} case(s) over the line"
            " whose fast-set membership the record does not carry (run by ctest,"
            " or recorded before #987):"
        )
        out += [
            f"  {r['seconds']:9.3f} s  {r['name']}  (family {r['family']})"
            for r in f["over_line_unrecorded"]
        ]
    out.append(
        f"slow-rule: {len(f['steps'])} step(s) past median + {f['k']:g} sigma"
        f" over the last {f['window']} nights (the box's relative sigma"
        f" {100 * f['pooled_relative_sigma']:.2f}%; {f['judged']} case(s) judged,"
        f" {f['too_short']} with under {f['min_history']} passes in the window)"
        + (":" if f["steps"] else "")
    )
    out += [
        f"  {r['seconds']:9.3f} s  {r['name']}  (median {r['median']:.3f} s over"
        f" {r['nights']}, bar {r['bar']:.3f} s)"
        for r in f["steps"]
    ]
    return out


def summary_field(findings: dict | None) -> str:
    """The summary line's field: `slow-rule=<over>/<steps>`."""
    if findings is None:
        return "slow-rule=none"
    return f"slow-rule={len(findings['over_line'])}/{len(findings['steps'])}"


def families_of(build: Path, source: Path, text: str, host: str) -> dict:
    """test_families' families of a built build dir, their representatives
    by the same records' times."""
    tf = _families_tool()
    cases, _ = tf.build_cases(build, source)
    markers = tf.markers_in(source, {c.file for c in cases})
    fams, _, _ = tf.families(cases, markers, tf.parse_metrics(text, host))
    return fams


def nightly_report(
    build: Path, source: Path, host: str, out: Path | None = None
) -> dict | None:
    """What test.py nightly calls once its record is on the branch: the
    findings, written as JSON to `out` when given."""
    text = _families_tool().metrics_branch_text(source) or ""
    findings = check(text, host, families_of(build, source, text, host))
    if out is not None and findings is not None:
        out.write_text(json.dumps(findings, indent=1) + "\n")
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="devtools/slow_rule.py",
        description="Report the latest nightly's fast-set cases over the line,"
        " and its steps.",
    )
    ap.add_argument("build_dir", type=Path, help="a built build dir, for families")
    ap.add_argument(
        "--source",
        type=Path,
        default=_ROOT,
        help="the source tree (default: this checkout)",
    )
    ap.add_argument(
        "--metrics",
        action="append",
        type=Path,
        default=[],
        help="a metrics .jsonl file (repeatable; default: the metrics branch)",
    )
    ap.add_argument("--host", help="whose records (default: this machine's)")
    ap.add_argument("--json", action="store_true", help="print the findings as JSON")
    args = ap.parse_args(argv)
    tf = _families_tool()
    source = args.source.resolve()
    host = args.host or tf.this_host()
    if args.metrics:
        text = "".join(p.read_text() for p in args.metrics)
    else:
        text = tf.metrics_branch_text(source) or ""
    try:
        fams = families_of(args.build_dir.resolve(), source, text, host)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2
    findings = check(text, host, fams)
    if findings is None:
        print(f"slow_rule: no nightly or weekly record of {host!r}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(findings, indent=1))
    else:
        print("\n".join(render(findings)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
