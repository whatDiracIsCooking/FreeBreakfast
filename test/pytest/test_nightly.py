"""devtools/test.py nightly: the mode of a night, the clean checkout (in
throwaway repositories under tmp_path), the launcher switch, the record's host
identity, the card wait, and the systemd units. Nothing here builds, needs a
GPU, or reads this checkout's history."""

from __future__ import annotations

import fcntl
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DATA = Path(__file__).resolve().parent / "data" / "metrics"
_UNITS = _ROOT / "devtools" / "systemd"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gates = _load("devtools_test_py_for_nightly", _ROOT / "devtools" / "test.py")
metrics = _load("devtools_metrics_for_nightly", _ROOT / "devtools" / "metrics.py")


def _clean_env(**extra: str) -> dict[str, str]:
    # The pre-push hook exports GIT_DIR and PRE_COMMIT_*; a test that runs git
    # under them acts on this checkout's own repository (#694).
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "PRE_COMMIT"))
    }
    env.update(extra)
    return env


@pytest.fixture(autouse=True)
def _no_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(("GIT_", "PRE_COMMIT")):
            monkeypatch.delenv(name)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# --- arguments and the mode of a night --------------------------------------


def test_nightly_defaults() -> None:
    args, rest = gates.parse(["nightly"])
    assert args.gate == "nightly" and args.mode == "auto"
    assert args.rev == "origin/main" and args.preset is None
    assert not args.in_checkout and rest == []


def test_nightly_passes_ctest_args_through() -> None:
    # A reduced run (#986's acceptance): they go to ctest.
    args, rest = gates.parse(["nightly", "--", "-R", "X"])
    assert args.gate == "nightly" and rest == ["-R", "X"]


@pytest.mark.parametrize(
    ("requested", "day", "today", "mode"),
    [
        ("auto", "Sun", 6, "weekly"),
        ("auto", "Sun", 0, "nightly"),
        ("auto", "wednesday", 2, "weekly"),
        ("weekly", "Sun", 0, "weekly"),
        ("nightly", "Sun", 6, "nightly"),
    ],
)
def test_nightly_mode(requested: str, day: str, today: int, mode: str) -> None:
    assert gates.nightly_mode(requested, day, today) == mode


def test_nightly_mode_refuses_a_bad_day() -> None:
    with pytest.raises(gates.NightlyError, match="not a weekday"):
        gates.nightly_mode("auto", "Someday", 0)


def test_config_carries_the_nightly_settings() -> None:
    cfg = gates.read_config(_clean_env())
    assert cfg["NIGHTLY_WORKTREE"] == ".claude/worktrees/nightly"
    assert cfg["NIGHTLY_WEEKLY_DAY"] in gates.WEEKDAYS
    assert int(cfg["NIGHTLY_GPU_BATCH_BUDGET_SECONDS"]) > int(
        cfg["GPU_BATCH_BUDGET_SECONDS"]
    )


def test_full_env_drops_the_fast_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FREE_BREAKFAST_FAST_TESTS", "1")
    assert "FREE_BREAKFAST_FAST_TESTS" not in gates.full_env()
    assert gates.fast_env()["FREE_BREAKFAST_FAST_TESTS"] == "1"


# --- the clean checkout ------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A primary checkout with two commits on main, ignoring build/ and
    .claude/worktrees/ as this tree's .gitignore does."""
    r = tmp_path / "main"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    _git(r, "config", "commit.gpgsign", "false")
    (r / ".gitignore").write_text("build/\n.claude/worktrees/\n")
    (r / "src.txt").write_text("one\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "one")
    (r / "src.txt").write_text("two\n")
    _git(r, "commit", "-q", "-am", "two")
    return r


def test_first_checkout_is_made_detached_at_the_rev(repo: Path) -> None:
    wt = repo / ".claude" / "worktrees" / "nightly"
    before = (_git(repo, "status", "--porcelain"), _git(repo, "rev-parse", "HEAD"))
    sha = gates.nightly_checkout(repo, wt, "main~1")
    assert sha == _git(repo, "rev-parse", "main~1")
    assert _git(wt, "rev-parse", "HEAD") == sha
    assert (wt / "src.txt").read_text() == "one\n"
    # Detached: no branch of the primary checkout is held.
    assert subprocess.run(
        ["git", "-C", str(wt), "symbolic-ref", "-q", "HEAD"],
        env=_clean_env(),
        check=False,
    ).returncode
    assert (
        _git(repo, "status", "--porcelain"),
        _git(repo, "rev-parse", "HEAD"),
    ) == before


def test_a_later_night_resets_everything_but_the_ignored(repo: Path) -> None:
    wt = repo / ".claude" / "worktrees" / "nightly"
    gates.nightly_checkout(repo, wt, "main~1")
    (wt / "src.txt").write_text("edited\n")
    (wt / "stray.txt").write_text("untracked\n")
    (wt / "build").mkdir()
    (wt / "build" / "warm.o").write_text("object\n")
    sha = gates.nightly_checkout(repo, wt, "main")
    assert sha == _git(repo, "rev-parse", "main")
    assert (wt / "src.txt").read_text() == "two\n"
    assert not (wt / "stray.txt").exists()
    assert (wt / "build" / "warm.o").is_file()
    assert _git(wt, "status", "--porcelain") == ""


def test_a_deleted_checkout_is_remade(repo: Path) -> None:
    wt = repo / ".claude" / "worktrees" / "nightly"
    gates.nightly_checkout(repo, wt, "main")
    shutil.rmtree(wt)
    assert gates.nightly_checkout(repo, wt, "main") == _git(repo, "rev-parse", "main")


def test_an_unknown_rev_is_refused(repo: Path) -> None:
    with pytest.raises(gates.NightlyError, match="rev-parse"):
        gates.nightly_checkout(repo, repo / "wt", "no-such-ref")


def test_an_unrelated_directory_is_refused(repo: Path) -> None:
    wt = repo / ".claude" / "worktrees" / "nightly"
    wt.mkdir(parents=True)
    (wt / "mine.txt").write_text("keep\n")
    with pytest.raises(gates.NightlyError, match="not a worktree"):
        gates.nightly_checkout(repo, wt, "main")
    assert (wt / "mine.txt").read_text() == "keep\n"


def test_another_repositorys_worktree_is_refused(repo: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    shutil.copytree(repo, other)
    wt = repo / ".claude" / "worktrees" / "nightly"
    _git(other, "worktree", "add", "-q", "--detach", str(wt), "main")
    with pytest.raises(gates.NightlyError, match="another repository"):
        gates.nightly_checkout(repo, wt, "main")


# --- the launcher, on and off ------------------------------------------------


@pytest.mark.parametrize(
    ("value", "on"),
    [
        ("/usr/local/bin/sccache", True),
        ("OFF", False),
        ("off", False),
        ("FREE_BREAKFAST_SCCACHE-NOTFOUND", False),
        ("", False),
    ],
)
def test_launcher_reads_the_weekly_off(tmp_path: Path, value: str, on: bool) -> None:
    d = tmp_path / "build"
    shutil.copytree(_DATA / "build", d)
    cache = d / "CMakeCache.txt"
    cache.write_text(
        cache.read_text().replace(
            "FREE_BREAKFAST_SCCACHE:FILEPATH=/usr/local/bin/sccache",
            f"FREE_BREAKFAST_SCCACHE:FILEPATH={value}",
        )
    )
    assert metrics.launcher_on(d) is on


def test_the_modes_configure_the_launcher() -> None:
    assert gates.LAUNCHER_OFF == ("-DFREE_BREAKFAST_SCCACHE=OFF",)
    assert gates.LAUNCHER_ON == ("-UFREE_BREAKFAST_SCCACHE",)


def test_cpp_build_passes_configure_args(tmp_path: Path) -> None:
    """build.sh's cpp_build hands what follows the jobs to the configure,
    and nothing to the build. A stub cmake on PATH records each call."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    (bin_dir / "cmake").write_text(f'#!/bin/sh\necho "$*" >> {calls}\n')
    (bin_dir / "cmake").chmod(0o755)
    env = _clean_env(PATH=f"{bin_dir}:{os.environ['PATH']}")
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f'. "{_ROOT}/devtools/lib.sh"; . "{_ROOT}/devtools/build.sh";'
            " init_submodules() { :; }; BUILD_LOG=/dev/null;"
            ' cpp_build "$@"',
            "_",
            "default",
            "0",
            "1",
            "8",
            "-DFREE_BREAKFAST_SCCACHE=OFF",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert calls.read_text().splitlines() == [
        "--preset default -DFREE_BREAKFAST_SCCACHE=OFF",
        "--build --preset default -j 8 --clean-first",
    ]


# --- the record's host -------------------------------------------------------


def _dmi(tmp_path: Path, **fields: str) -> Path:
    d = tmp_path / "dmi"
    d.mkdir()
    for name, value in fields.items():
        (d / name).write_text(value + "\n")
    return d


def test_host_is_the_board(tmp_path: Path) -> None:
    dmi = _dmi(
        tmp_path,
        board_vendor="ASUSTeK COMPUTER INC.",
        board_name="ROG CROSSHAIR VIII DARK HERO",
        sys_vendor="ASUS",
        product_name="System Product Name",
    )
    assert metrics.host_name(None, dmi) == (
        "ASUSTeK COMPUTER INC. ROG CROSSHAIR VIII DARK HERO"
    )
    # The same on every call: nothing in it is the container's.
    assert metrics.host_name("", dmi) == metrics.host_name(None, dmi)


def test_host_without_a_board_is_the_system(tmp_path: Path) -> None:
    dmi = _dmi(tmp_path, sys_vendor="QEMU", product_name="Standard PC (Q35)")
    assert metrics.host_name(None, dmi) == "QEMU Standard PC (Q35)"


def test_host_override_wins(tmp_path: Path) -> None:
    dmi = _dmi(tmp_path, board_vendor="V", board_name="B")
    assert metrics.host_name("  lab-box ", dmi) == "lab-box"


def test_host_without_dmi_is_the_hostname(tmp_path: Path) -> None:
    import socket

    assert metrics.host_name(None, tmp_path / "absent") == socket.gethostname()


def test_config_has_metrics_host_empty_by_default() -> None:
    env = _clean_env()
    env.pop("METRICS_HOST", None)
    assert gates.read_config(env)["METRICS_HOST"] == ""
    assert gates.read_config(_clean_env(METRICS_HOST="x"))["METRICS_HOST"] == "x"


def test_a_record_defaults_to_host_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "host_name", lambda: "the-board")
    record = metrics.make_record(
        kind="nightly",
        started="2026-11-07T02:30:00Z",
        finished="2026-11-07T04:00:00Z",
        commit="0" * 40,
        backend="cuda",
        preset="default",
        gpu={"name": None, "driver": None, "vram_mib": None},
        toolchain={},
        build=None,
        cases=[],
        result="pass",
    )
    assert record["host"] == "the-board"


# --- the card: a held one is waited for, never shared ------------------------


def test_nightly_waits_on_a_held_card(tmp_path: Path) -> None:
    """A Card is not entered while another process holds either lock file,
    and is entered once that holder exits."""
    cfg = dict(
        gates.read_config(_clean_env()),
        PRIMARY_ROOT=str(tmp_path),
        GPU_LOCK_DIR="locks",
        GPU_POLL_SECONDS="0.1",
        GPU_FOREIGN_VRAM_PCT="101",
    )
    (tmp_path / "locks").mkdir()
    holder = subprocess.Popen(
        [
            "flock",
            str(tmp_path / "locks" / "cuda.1.lock"),
            sys.executable,
            "-c",
            "import time; time.sleep(1.5)",
        ]
    )
    # Until flock(1) holds it, the test would race the holder.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with (tmp_path / "locks" / "cuda.1.lock").open("w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
            except OSError:
                break
        time.sleep(0.02)
    start = time.monotonic()
    card = gates.Card("cuda", cfg)
    card.vram = lambda: "?"
    with card:
        assert holder.poll() is not None, "entered while the holder held the card"
    holder.wait()
    assert time.monotonic() - start > 0.5


# --- the systemd units --------------------------------------------------------


def test_the_service_runs_the_nightly_in_the_container() -> None:
    text = (_UNITS / "free-breakfast-nightly@.service").read_text()
    start = [line for line in text.splitlines() if line.startswith("ExecStart=")]
    assert len(start) == 1
    assert "devtools/devcontainer.sh" in start[0] and "--combined shell -c" in start[0]
    assert "devtools/test.py nightly" in start[0]
    assert "Environment=FREE_BREAKFAST_ROOT=%I" in text
    assert "Type=oneshot" in text


def test_the_timer_is_installable_and_nightly() -> None:
    text = (_UNITS / "free-breakfast-nightly@.timer").read_text()
    assert "OnCalendar=*-*-* " in text
    assert "WantedBy=timers.target" in text


@pytest.mark.skipif(
    shutil.which("systemd-analyze") is None or shutil.which("systemd-escape") is None,
    reason="no systemd-analyze/systemd-escape to verify the units with",
)
def test_the_units_verify(tmp_path: Path) -> None:
    instance = subprocess.run(
        ["systemd-escape", "--path", str(tmp_path / "main")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    paths = []
    for kind in ("service", "timer"):
        dst = tmp_path / f"free-breakfast-nightly@{instance}.{kind}"
        shutil.copy(_UNITS / f"free-breakfast-nightly@.{kind}", dst)
        paths.append(str(dst))
    proc = subprocess.run(
        ["systemd-analyze", "verify", "--man=no", *paths],
        capture_output=True,
        text=True,
        check=False,
    )
    # verify prints every parse problem, and exits 0 for some of them. It also
    # loads the host's own units and warns about theirs (a newer systemd on a
    # CI image, #1026), so only the lines naming ours count.
    ours = [
        line for line in proc.stderr.splitlines() if "free-breakfast-nightly" in line
    ]
    assert proc.returncode == 0 and not ours, proc.stderr
