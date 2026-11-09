#!/usr/bin/env python3
"""Has the nightly run? A dead man's switch (#1127; docs/testing.md "How to
tell it ran").

A passing night makes no noise, and neither does a night that never ran: the
box off at 02:30, the combined devcontainer down, a broken unit. This reads
`<reports>/summary.log` and judges its newest `test.py nightly` line from a
full run of origin/main, what the timer starts. A `--rev` run (`ref=` on its
line) and a reduced one (`run=reduced`) say nothing about the timer and are
passed over.

- **ok**: that line is a PASS, and newer than NIGHTLY_STALE_HOURS.
- **warn**: it is older, or it is a FAIL, or there is none and the timer unit
  is installed ("no nightly has run here").
- **skip**: there is none and no timer unit is installed (or systemctl cannot
  say: the container). A VM or a box that is not meant to run the nightly is
  never nagged.

A night's time is its log's mtime (`<mode>-<stamp>-<rev>-<preset>.log`, beside
summary.log) where the log is there, else its `stamp=` read as local time. The
mtime is the end of the run and means the same on the host and in the
container, whose clock zones differ; the stamp is the writer's local time.

`devtools/test.py push` and `pr` print `check()`'s line when it warns, and
never fail on it. `devtools/doctor.sh` runs this file, which prints the same
verdict and, with `--systemd`, the host's timer and service state:

  devtools/nightly_stale.py --reports DIR [--hours H] [--primary PATH] [--systemd]

Each output line is `<ok|warn|note>\\t<text>`. Standard library only: it runs
under a bare python3, inside the push hook.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# devtools/systemd/<UNIT>@.service and .timer; the instance is the primary
# checkout's path, `systemd-escape --path`ed.
UNIT = "free-breakfast-nightly"
DEFAULT_HOURS = 36.0
_PREFIX = "test.py nightly "
_FIELD = re.compile(r"(\w[\w-]*)=(\S+)")


class Night:
    """One `test.py nightly` line of summary.log. A plain class, not a
    dataclass: test.py loads this file by path, unregistered in sys.modules,
    where @dataclass fails."""

    def __init__(self, line: str, verdict: str, fields: dict[str, str]) -> None:
        self.line = line
        self.verdict = verdict  # PASS, or FAIL(rc=N)
        self.fields = fields

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    @property
    def timer(self) -> bool:
        """A full run of origin/main: neither --rev nor reduced."""
        return "ref" not in self.fields and self.fields.get("run") != "reduced"

    def when(self, reports: Path) -> float | None:
        """Seconds since the epoch: the log's mtime, else the stamp as local
        time, else None (a line from before stamps were written)."""
        f = self.fields
        stamp = f.get("stamp")
        if stamp is None:
            return None
        if "mode" in f and "rev" in f and "preset" in f:
            log = reports / f"{f['mode']}-{stamp}-{f['rev']}-{f['preset']}.log"
            try:
                return log.stat().st_mtime
            except OSError:
                pass
        try:
            return time.mktime(time.strptime(stamp, "%Y%m%d-%H%M%S"))
        except ValueError:
            return None


def parse(line: str) -> Night | None:
    """A summary.log line as a Night, or None for any other gate's line."""
    if not line.startswith(_PREFIX):
        return None
    rest = line[len(_PREFIX) :].split(None, 1)
    if not rest:
        return None
    tail = rest[1] if len(rest) > 1 else ""
    return Night(line.rstrip("\n"), rest[0], dict(_FIELD.findall(tail)))


def newest(text: str) -> Night | None:
    """The last timer night in summary.log's text (it is append-only, so the
    last is the newest)."""
    nights = [n for n in map(parse, text.splitlines()) if n is not None]
    timer = [n for n in nights if n.timer]
    return timer[-1] if timer else None


def _systemctl(*args: str) -> str | None:
    """`systemctl --user <args>`'s stdout, or None where it cannot run (no
    systemctl, or no user manager: the container)."""
    if shutil.which("systemctl") is None:
        return None
    try:
        r = subprocess.run(
            ["systemctl", "--user", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def unit_name(primary: Path) -> str | None:
    """`<UNIT>@<escaped path>`, the instance for this primary checkout."""
    if shutil.which("systemd-escape") is None:
        return None
    try:
        r = subprocess.run(
            ["systemd-escape", "--path", str(primary)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{UNIT}@{r.stdout.strip()}" if r.returncode == 0 else None


def _show(unit: str, *props: str) -> dict[str, str] | None:
    out = _systemctl("show", unit, *(f"--property={p}" for p in props))
    if out is None:
        return None
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def timer_installed(primary: Path) -> bool:
    """Whether this checkout's timer unit is loaded in the user manager.
    False where that cannot be asked (the container)."""
    unit = unit_name(primary)
    if unit is None:
        return False
    shown = _show(f"{unit}.timer", "LoadState")
    return shown is not None and shown.get("LoadState") == "loaded"


def _age(seconds: float) -> str:
    hours = seconds / 3600
    return f"{hours:.0f}h" if hours < 72 else f"{hours / 24:.1f}d"


def check(
    reports: Path,
    hours: float,
    primary: Path,
    now: float | None = None,
    installed=timer_installed,
) -> tuple[str, str]:
    """The verdict, `(level, text)`, level one of ok, warn and skip.
    `installed(primary)` is asked only when no timer night is on record."""
    now = time.time() if now is None else now
    summary = reports / "summary.log"
    try:
        text = summary.read_text(errors="replace")
    except OSError:
        text = ""
    night = newest(text)
    if night is None:
        if installed(primary):
            return (
                "warn",
                f"no nightly has run here (the timer is installed; {summary})",
            )
        return "skip", f"no nightly expected here (no timer unit, none in {summary})"
    when = night.when(reports)
    if when is None:
        return "warn", f"the newest nightly line has no stamp: {night.line}"
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(when))
    age = now - when
    if not night.passed:
        return (
            "warn",
            f"the last nightly FAILED, {_age(age)} ago ({stamp}): {night.line}",
        )
    if age > hours * 3600:
        return "warn", (
            f"the last nightly ran {_age(age)} ago ({stamp}, PASS), over"
            f" NIGHTLY_STALE_HOURS={hours:g}: has the timer stopped?"
        )
    return "ok", f"the last nightly passed {_age(age)} ago ({stamp})"


def systemd_report(primary: Path) -> list[tuple[str, str]]:
    """The host's timer and service state, when the unit is installed: the
    last trigger, the next, and the last result when it is not success."""
    unit = unit_name(primary)
    timer = None
    if unit is not None:
        timer = _show(
            f"{unit}.timer",
            "LoadState",
            "ActiveState",
            "LastTriggerUSec",
            "NextElapseUSecRealtime",
        )
    if timer is None:
        return [
            ("note", "no systemd user manager here (the container?): timer unchecked")
        ]
    if timer.get("LoadState") != "loaded":
        return [("note", f"timer {unit}.timer not installed")]
    out: list[tuple[str, str]] = []
    state = timer.get("ActiveState", "?")
    last = timer.get("LastTriggerUSec") or "never"
    nxt = timer.get("NextElapseUSecRealtime") or "none"
    level = "ok" if state == "active" else "warn"
    out.append((level, f"timer {unit}.timer {state}: last trigger {last}, next {nxt}"))
    service = _show(
        f"{unit}.service", "Result", "ExecMainStatus", "ExecMainExitTimestamp"
    )
    if service is not None:
        result = service.get("Result", "?")
        if result != "success":
            out.append(
                (
                    "warn",
                    f"service {unit}.service last result {result} (exit status"
                    f" {service.get('ExecMainStatus', '?')}, at"
                    f" {service.get('ExecMainExitTimestamp') or '?'})",
                )
            )
            out.append(("note", f'read: journalctl --user -u "{unit}.service" -n 50'))
    return out


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="devtools/nightly_stale.py", description=__doc__)
    p.add_argument("--reports", required=True, help="the SLOW_TIER_REPORTS directory")
    p.add_argument("--hours", type=float, default=DEFAULT_HOURS)
    p.add_argument("--primary", default=".", help="the primary checkout")
    p.add_argument("--systemd", action="store_true", help="also the timer's state")
    args = p.parse_args(argv)
    primary = Path(args.primary).resolve()
    level, text = check(Path(args.reports), args.hours, primary)
    print(f"{'note' if level == 'skip' else level}\t{text}")
    if args.systemd:
        for level, text in systemd_report(primary):
            print(f"{level}\t{text}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
