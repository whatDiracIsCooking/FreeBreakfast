"""devtools/nightly_stale.py, the nightly's dead man's switch (#1127), over
stubbed summary.log files, and with stub systemctl/systemd-escape for the
host's timer report. Nothing here reads this box's own reports or systemd."""

from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stale = _load("devtools_nightly_stale", _ROOT / "devtools" / "nightly_stale.py")
gates = _load("devtools_test_py_for_stale", _ROOT / "devtools" / "test.py")

NOW = time.mktime(time.strptime("20261109-120000", "%Y%m%d-%H%M%S"))
FRESH = "20261109-023000"  # 9.5 h before NOW
STALE = "20261106-023000"  # 81.5 h before NOW


def _night(verdict: str, stamp: str, extra: str = "") -> str:
    return (
        f"test.py nightly {verdict}  rev=abc1234  preset=default  mode=nightly"
        f"  stamp={stamp}{extra}  metrics=0123456789ab"
    )


PR = "test.py pr PASS  rev=abc1234  preset=default  stamp=20261109-110000"


def _reports(tmp_path: Path, *lines: str) -> Path:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "summary.log").write_text("".join(f"{x}\n" for x in lines))
    return reports


def _never(primary: Path) -> bool:
    raise AssertionError("the timer is asked only when no night is on record")


def _check(reports: Path, installed=_never, hours: float = 36):
    return stale.check(reports, hours, reports.parent, now=NOW, installed=installed)


def test_fresh_pass_is_ok(tmp_path: Path) -> None:
    reports = _reports(tmp_path, _night("PASS", STALE), _night("PASS", FRESH), PR)
    level, text = _check(reports)
    assert level == "ok"
    assert "10h ago" in text


def test_stale_pass_warns(tmp_path: Path) -> None:
    reports = _reports(tmp_path, _night("PASS", STALE), PR)
    level, text = _check(reports)
    assert level == "warn"
    assert "NIGHTLY_STALE_HOURS=36" in text
    assert "3.4d ago" in text


def test_zero_hours_warns_on_any_night(tmp_path: Path) -> None:
    reports = _reports(tmp_path, _night("PASS", FRESH))
    assert _check(reports, hours=0)[0] == "warn"


def test_fresh_fail_warns(tmp_path: Path) -> None:
    reports = _reports(tmp_path, _night("PASS", STALE), _night("FAIL(rc=1)", FRESH))
    level, text = _check(reports)
    assert level == "warn"
    assert "FAILED" in text
    assert "FAIL(rc=1)" in text


def test_a_failed_checkout_line_counts(tmp_path: Path) -> None:
    # The outer half's line: no mode, no preset, its error in parentheses.
    line = f"test.py nightly FAIL(rc=1)  rev=origin/main  stamp={FRESH}  (fetch failed)"
    reports = _reports(tmp_path, _night("PASS", STALE), line)
    level, text = _check(reports)
    assert (level, "FAILED" in text) == ("warn", True)


def test_none_at_all_without_a_timer_skips(tmp_path: Path) -> None:
    reports = tmp_path / "reports"  # no summary.log at all
    level, text = _check(reports, installed=lambda p: False)
    assert level == "skip"
    assert "no nightly expected here" in text


def test_none_at_all_with_a_timer_warns(tmp_path: Path) -> None:
    reports = _reports(tmp_path, PR)
    level, text = _check(reports, installed=lambda p: True)
    assert level == "warn"
    assert "no nightly has run here" in text


def test_only_rev_and_reduced_lines_count_as_none(tmp_path: Path) -> None:
    reports = _reports(
        tmp_path,
        _night("PASS", FRESH, "  ref=HEAD~3"),
        _night("PASS", FRESH, "  run=reduced"),
        _night("FAIL(rc=1)", FRESH, "  ref=abc  run=reduced"),
    )
    assert _check(reports, installed=lambda p: False)[0] == "skip"
    assert _check(reports, installed=lambda p: True)[0] == "warn"


def test_rev_and_reduced_lines_are_passed_over(tmp_path: Path) -> None:
    # A by-hand reduced run after a stale timer night does not freshen it.
    reports = _reports(
        tmp_path, _night("PASS", STALE), _night("FAIL(rc=1)", FRESH, "  run=reduced")
    )
    level, text = _check(reports)
    assert level == "warn"
    assert "PASS" in text and "3.4d ago" in text


def test_the_logs_mtime_beats_the_stamp(tmp_path: Path) -> None:
    reports = _reports(tmp_path, _night("PASS", STALE))
    log = reports / f"nightly-{STALE}-abc1234-default.log"
    log.write_text("")
    os.utime(log, (NOW - 3600, NOW - 3600))
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(NOW - 3600))
    assert _check(reports) == ("ok", f"the last nightly passed 1h ago ({when})")


def test_a_line_without_a_stamp_warns(tmp_path: Path) -> None:
    reports = _reports(tmp_path, "test.py nightly FAIL(rc=1)  rev=origin/main  (old)")
    assert _check(reports)[0] == "warn"


def test_the_default_matches_config() -> None:
    cfg = gates.read_config(
        env={k: v for k, v in os.environ.items() if k != "NIGHTLY_STALE_HOURS"}
    )
    assert float(cfg["NIGHTLY_STALE_HOURS"]) == stale.DEFAULT_HOURS


def test_the_unit_is_the_shipped_one() -> None:
    for kind in ("service", "timer"):
        assert (_ROOT / "devtools" / "systemd" / f"{stale.UNIT}@.{kind}").is_file()


# --- the summary line's markers (test.py) ------------------------------------


@pytest.mark.parametrize(
    ("rev", "passthrough", "fields"),
    [
        ("origin/main", [], ""),
        ("HEAD~1", [], "  ref=HEAD~1"),
        ("origin/main", ["-R", "Foo"], "  run=reduced"),
        ("abc", ["-R", "Foo"], "  ref=abc  run=reduced"),
    ],
)
def test_run_fields_mark_what_the_timer_did_not_start(
    rev: str, passthrough: list[str], fields: str
) -> None:
    assert gates.run_fields(argparse.Namespace(rev=rev), passthrough) == fields
    night = stale.parse(_night("PASS", FRESH, fields))
    assert night is not None and night.timer == (not fields)


# --- test.py's warning line ----------------------------------------------------


def _cfg(tmp_path: Path, hours: str) -> dict[str, str]:
    return {
        "PRIMARY_ROOT": str(tmp_path),
        "SLOW_TIER_REPORTS": "reports",
        "NIGHTLY_STALE_HOURS": hours,
    }


def test_the_gates_warn_in_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    _reports(tmp_path, _night("PASS", STALE))
    gates.nightly_warning(_cfg(tmp_path, "36"), "push")
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err.startswith("push: WARNING nightly: the last nightly ran")
    assert out.err.count("\n") == 1


def test_the_gates_say_nothing_when_fresh(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    _reports(tmp_path, _night("PASS", time.strftime("%Y%m%d-%H%M%S")))
    gates.nightly_warning(_cfg(tmp_path, "36"), "pr")
    assert capsys.readouterr() == ("", "")


def test_the_gates_never_fail_on_the_check(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    gates.nightly_warning(_cfg(tmp_path, "not-a-number"), "pr")
    assert capsys.readouterr() == ("", "")


# --- the host's systemd half, stubbed ------------------------------------------


def _stub_systemd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timer: str, service: str
) -> None:
    """systemd-escape and a systemctl that answers `show` from fixed text."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "systemd-escape").write_text('#!/bin/sh\necho "x-y"\n')
    (bin_ / "systemctl").write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        f"  *.timer*) printf '%s' '{timer}' ;;\n"
        f"  *.service*) printf '%s' '{service}' ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n"
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")


def test_systemd_report_a_healthy_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_systemd(
        tmp_path,
        monkeypatch,
        "LoadState=loaded\nActiveState=active\n"
        "LastTriggerUSec=Mon 2026-11-09 02:30:00 PST\n"
        "NextElapseUSecRealtime=Tue 2026-11-10 02:30:00 PST\n",
        "Result=success\nExecMainStatus=0\n",
    )
    report = stale.systemd_report(tmp_path)
    assert report == [
        (
            "ok",
            f"timer {stale.UNIT}@x-y.timer active: last trigger Mon 2026-11-09"
            " 02:30:00 PST, next Tue 2026-11-10 02:30:00 PST",
        )
    ]
    assert stale.timer_installed(tmp_path)


def test_systemd_report_a_failed_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The #1120 night: the timer fired, the service exited 127 at once.
    _stub_systemd(
        tmp_path,
        monkeypatch,
        "LoadState=loaded\nActiveState=active\n"
        "LastTriggerUSec=Sun 2026-11-08 02:30:00 PST\n",
        "Result=exit-code\nExecMainStatus=127\n"
        "ExecMainExitTimestamp=Sun 2026-11-08 02:30:00 PST\n",
    )
    levels = [level for level, _ in stale.systemd_report(tmp_path)]
    assert levels == ["ok", "warn", "note"]
    assert "exit status 127" in stale.systemd_report(tmp_path)[1][1]


def test_systemd_report_no_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_systemd(tmp_path, monkeypatch, "LoadState=not-found\n", "")
    assert stale.systemd_report(tmp_path) == [
        ("note", f"timer {stale.UNIT}@x-y.timer not installed")
    ]
    assert not stale.timer_installed(tmp_path)


def test_systemd_report_no_user_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The container: systemctl is there, its user bus is not.
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "systemd-escape").write_text('#!/bin/sh\necho "x-y"\n')
    (bin_ / "systemctl").write_text(
        "#!/bin/sh\necho 'Failed to connect to bus' >&2\nexit 1\n"
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
    [(level, text)] = stale.systemd_report(tmp_path)
    assert level == "note" and "no systemd user manager" in text
    assert not stale.timer_installed(tmp_path)


def test_the_command_line_prints_tab_separated_levels(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    reports = _reports(tmp_path, _night("PASS", STALE))
    assert stale.main(["--reports", str(reports), "--hours", "0"]) == 0
    level, _, text = capsys.readouterr().out.partition("\t")
    assert level == "warn" and text.startswith("the last nightly ran")
