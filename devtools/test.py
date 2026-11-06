#!/usr/bin/env python3
"""The test gates, one subcommand per gate (docs/testing.md "Gates").

  devtools/test.py push [--preset NAME] [-j N] [--dry-run]
  devtools/test.py pr   [--preset NAME] [--fresh] [--clean] [--no-test]
                        [--tidy] [--only cpp|python] [-j N] [--report DIR]
                        [-- <ctest args>]

push: what `git push` runs, by pre-commit's pre-push stage. It reads the
  pushed range (PRE_COMMIT_FROM_REF...PRE_COMMIT_TO_REF, which pre-commit
  exports; by hand, origin/main...HEAD) and runs the Python half when a .py
  changed and the C++ half when C++ or a build file did. Python: the fast
  pytest set (FAST_TEST_ARGS over PREPUSH_PATHS). C++: an incremental build,
  the host binaries, then each GPU binary as one process under the card lock.
  The C++ half skips itself, exit 0, where the toolchain or a GPU is absent.
  --dry-run prints the changed files and which halves would run.

pr: the gate before opening a PR (milestone-run requires it). Configure,
  build, ctest, then the fast pytest set. The GPU entries that can share a
  process run as one gtest process per GPU binary; ctest runs the rest. Logged
  to <report>/pr-<stamp>-<rev>-<preset>.log, and its result line appended to
  <report>/summary.log.
    --preset NAME  configure/build preset; the test preset follows it
                   (default: CMAKE_PRESET, `auto`, the least-loaded live card)
    --fresh        wipe the CMake cache first
    --clean        rebuild every object, keeping the cache
    --no-test      configure and build only
    --tidy         advisory clang-tidy over src/*.cpp after the build
    --only LANG    run only the cpp or the python half
    -j N           compile jobs (default: BUILD_JOBS)
    --report DIR   log directory (default: SLOW_TIER_REPORTS)
    -- <args>      passed to ctest verbatim; they select the GPU batches too

Both gates run with FREE_BREAKFAST_FAST_TESTS=1, so every case under
FBF_SKIP_IF_FAST_TESTS() skips; cpp-slow-tier.sh is the unskipped run. Each
GPU batch is watched: a batch silent for GPU_BATCH_STALL_SECONDS, or running
past GPU_BATCH_BUDGET_SECONDS, is killed, its case named on a
`[gpu] hung in <case>` line, and its state written to <log>.gpu-wedge.txt
first (issue #477). Every setting is read from devtools/config.sh, and an
environment variable still overrides it.

Standard library only: this is the push hook on the host and in the container,
under a bare python3.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIB = REPO_ROOT / "devtools" / "lib.sh"
BUILD_SH = REPO_ROOT / "devtools" / "build.sh"

# The values test.py takes from devtools/config.sh (and PRIMARY_ROOT from
# lib.sh). config.sh is the one place they are written down.
CONFIG_VARS = (
    "CMAKE_PRESET",
    "CTEST_PRESET",
    "GPU_HIP_PRESETS",
    "BUILD_JOBS",
    "CTEST_JOBS",
    "GPU_LOCK_DIR",
    "GPU_FOREIGN_VRAM_PCT",
    "GPU_POLL_SECONDS",
    "GPU_BATCH_STALL_SECONDS",
    "GPU_BATCH_BUDGET_SECONDS",
    "GPU_WEDGE_TOOL_SECONDS",
    "FAST_TEST_ARGS",
    "PREPUSH_PATHS",
    "SLOW_TIER_REPORTS",
    "VENV_PATHS",
    "JOBS",
    "PRIMARY_ROOT",
)

# What fires each half of the push gate. The C++ set is the build's inputs:
# sources, headers and the CMake files. A submodule bump changes no path under
# these patterns, and its own hooks are its repository's business; the exclude
# is .pre-commit-config.yaml's.
CPP_FILES = re.compile(
    r"\.(cpp|cppm|cu|cuh|h|hpp)$|(^|/)CMakeLists\.txt$"
    r"|(^|/)CMakePresets\.json$|\.cmake$"
)
PY_FILES = re.compile(r"\.py$")
EXCLUDED = re.compile(r"^(vendor/|third_party/|deps/WarpWraps/|deps/FelizJueves/)")

# The test binaries, relative to a build directory. The GPU ones, one batch
# each, in this order.
HOST_BINS = (
    "test/core/free_breakfast_core_tests_host",
    "test/atom/free_breakfast_atom_tests_host",
    "test/basis/free_breakfast_basis_tests_host",
    "test/io/free_breakfast_io_tests_host",
    "test/sad/free_breakfast_sad_tests_host",
    "test/pipeline/free_breakfast_pipeline_tests_host",
    "test/gtest/free_breakfast_tests_host",
)
GPU_BINS = (
    "test/core/free_breakfast_core_tests_gpu",
    "test/recipes/free_breakfast_recipes_tests_gpu",
    "test/gtest/free_breakfast_tests_gpu",
)

# The two lock files of a card, held together. Branches from before
# 2026-10-28 let two runs share a card, each holding one; holding both keeps
# such a run out. lib.sh's GPU_CARD_LOCKS, which the shell tools take.
CARD_LOCKS = ("0", "1")

SUMMARY_LINE = re.compile(r"ran\.|PASSED|SKIPPED|FAILED")
RUN_LINE = re.compile(r"^\[ RUN      \]")
VERDICT_LINE = re.compile(r"^\[ *(OK|FAILED|SKIPPED) *\]")
PASSED_LINE = re.compile(r"^\[ *(OK|SKIPPED) *\]")
FAILED_CASE_LINE = re.compile(r"^\[  FAILED  \] .*\)$")

# Linux caps one argument or environment string at 128 KiB (MAX_ARG_STRLEN),
# and a batch's filter goes in GTEST_FILTER.
MAX_FILTER = 100_000
FILTER_FLAG = "--gtest_filter="

PR_SET_PTRACER = 0x59616D61


def say(*parts: object, err: bool = False, log: Path | None = None) -> None:
    """Print one line, and append it to `log` when given."""
    line = " ".join(str(p) for p in parts)
    print(line, file=sys.stderr if err else sys.stdout, flush=True)
    if log is not None:
        with log.open("a") as f:
            f.write(line + "\n")


# --- config.sh and lib.sh --------------------------------------------------


def read_config(env: dict[str, str] | None = None) -> dict[str, str]:
    """devtools/config.sh's values, evaluated by bash through lib.sh.

    Sourcing it in bash, rather than parsing it, is what keeps one copy of
    every default, and lets an environment variable win as it does for the
    shell tools."""
    script = '. "$1"; shift; for v; do printf "%s\\0%s\\0" "$v" "${!v}"; done'
    out = subprocess.run(
        ["bash", "-c", script, "_", str(LIB), *CONFIG_VARS],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    fields = out.split("\0")
    return dict(zip(fields[0:-1:2], fields[1::2], strict=True))


def lib(
    script: str, *args: str, capture: bool = True, build: bool = False, **kw
) -> subprocess.CompletedProcess[str]:
    """Run a bash snippet after sourcing lib.sh (and, with `build`, build.sh):
    the shell tools' own functions, so test.py keeps no second copy of them.
    `args` are the snippet's $1, $2, ..."""
    sources = f'. "{LIB}"' + (f'; . "{BUILD_SH}"' if build else "")
    return subprocess.run(
        ["bash", "-c", f"{sources}; {script}", "_", *args],
        capture_output=capture,
        text=True,
        check=False,
        **kw,
    )


def backend_of_preset(preset: str, cfg: dict[str, str]) -> str:
    """`hip` for a preset named in GPU_HIP_PRESETS, else `cuda`."""
    return "hip" if preset in cfg["GPU_HIP_PRESETS"].split() else "cuda"


def auto_preset() -> str:
    """lib.sh's gpu_auto_preset: the least-loaded live card's preset. Its
    `gpu: auto -> <preset>` line goes to stderr."""
    proc = lib("gpu_auto_preset", capture=False, stdout=subprocess.PIPE)
    return proc.stdout.strip()


def build_dir_of(preset: str) -> Path:
    return Path(lib('cpp_build_dir "$1"', preset, build=True).stdout.strip())


def tree_rev() -> tuple[str, str]:
    """lib.sh's tree_rev: (HEAD's short sha, +dirty with tracked changes; the
    hash of `git diff HEAD`, empty on a clean tree) (#694)."""
    out = lib(
        'tree_rev "$REPO_ROOT"; printf "%s\\n%s\\n" "$TREE_REV" "$TREE_DIFF_HASH"'
    )
    rev, diff_hash = (out.stdout.split("\n") + ["", ""])[:2]
    return rev or "unknown", diff_hash


def tree_rev_header(tag: str, log: Path) -> None:
    """lib.sh's tree_rev_header: what made the tree dirty, to stdout and the
    head of `log`; nothing on a clean tree."""
    lib(
        'tree_rev "$REPO_ROOT"; tree_rev_header "$1" "$2"', tag, str(log), capture=False
    )


def report_dir_of(value: str) -> Path:
    d = Path(value)
    d = d if d.is_absolute() else REPO_ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_pytest(cfg: dict[str, str]) -> str | None:
    """lib.sh's find_tool: each VENV_PATHS entry's bin/ under this checkout,
    then under the primary one, then PATH. A .claude/worktrees/ checkout has no
    virtualenv of its own."""
    roots = [REPO_ROOT] + ([Path(cfg["PRIMARY_ROOT"])] if cfg["PRIMARY_ROOT"] else [])
    for root in roots:
        for v in cfg["VENV_PATHS"].split():
            p = Path(v) if v.startswith("/") else root / v
            if os.access(p / "bin" / "pytest", os.X_OK):
                return str(p / "bin" / "pytest")
    return shutil.which("pytest")


def strip_git_env() -> None:
    """pre-commit's pre-push stage exports GIT_DIR, GIT_WORK_TREE and
    GIT_INDEX_FILE for its own dispatch. Every child would inherit them --
    FetchContent's nested git clone resolving refs against this repo, a test's
    `git init` marking this one bare (#694) -- and nothing here needs them."""
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        os.environ.pop(name, None)


# --- the card lock ---------------------------------------------------------


class Card:
    """One whole card, held: every lock file of it under GPU_LOCK_DIR.

    fcntl.flock is flock(2), the lock flock(1) takes in lib.sh's
    with_gpu_card, so a test.py run and a shell tool's run exclude each other.
    The lock belongs to the open file, so it is released on exit, SIGKILL
    included."""

    def __init__(self, backend: str, cfg: dict[str, str]) -> None:
        self.backend = backend
        self.cfg = cfg
        # lib.sh's gpu_lock_dir, spelled the same way: under the MAIN
        # checkout, so every worktree sees the same files.
        root = cfg["PRIMARY_ROOT"] or str(REPO_ROOT)
        self.dir = Path(f"{root}/{cfg['GPU_LOCK_DIR']}")
        self.files: list = []

    def _take(self) -> bool:
        self.dir.mkdir(parents=True, exist_ok=True)
        for i in CARD_LOCKS:
            f = (self.dir / f"{self.backend}.{i}.lock").open("w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                self.release()
                return False
            self.files.append(f)
        return True

    def release(self) -> None:
        for f in self.files:
            f.close()
        self.files = []

    def vram(self) -> str:
        return lib('gpu_card_mem "$1"', self.backend).stdout.strip() or "?"

    def __enter__(self) -> Card:
        poll = float(self.cfg["GPU_POLL_SECONDS"])
        limit = int(self.cfg["GPU_FOREIGN_VRAM_PCT"])
        said = False
        while not self._take():
            if not said:
                say(
                    f"devtools: waiting for the {self.backend} card --"
                    f" another run holds it ({self.dir})",
                    err=True,
                )
                said = True
            time.sleep(poll)
        # Holding the lock, memory in use belongs to something that takes
        # none. Starting into a full card ends in an abort (CreatePolicy), so
        # it is waited out with no timeout; compute load never makes it wait.
        said = False
        while (mem := self.vram()) != "?" and int(mem) >= limit:
            if not said:
                say(
                    f"devtools: waiting -- {self.backend} VRAM {mem}% >= {limit}%"
                    " held by a job outside these locks",
                    err=True,
                )
                said = True
            time.sleep(poll)
        say(f"devtools: {self.backend} card held (VRAM {mem}%)", err=True)
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


# --- one gtest process, watched --------------------------------------------


def running_case(lines: list[str]) -> str:
    """The case a gtest log was in the middle of, or `<unknown>`."""
    case = ""
    for line in lines:
        if RUN_LINE.match(line):
            case = line.split()[-1]
        if VERDICT_LINE.match(line):
            case = ""
    return case or "<unknown>"


def failed_output(lines: list[str]) -> list[str]:
    """Every failed case's lines, [ RUN ] to verdict, and the case a crash or
    a kill stopped in: what the summary lines leave out (#870)."""
    out: list[str] = []
    buf: list[str] = []
    in_case = False
    for line in lines:
        if RUN_LINE.match(line):
            buf = []
            in_case = True
        if in_case:
            buf.append(line)
            if VERDICT_LINE.match(line):
                in_case = False
        if PASSED_LINE.match(line):
            buf = []
        if FAILED_CASE_LINE.match(line) and buf:
            out += buf
            buf = []
    return out + buf


def _allow_tracer(tracer: int):
    """A preexec_fn naming `tracer` as allowed to ptrace the child. Yama's
    ptrace_scope=1 lets a process trace only its descendants; the debugger the
    watchdog runs is the batch's sibling. PR_SET_PTRACER survives the exec."""
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None

    def allow() -> None:
        prctl(PR_SET_PTRACER, ctypes.c_ulong(tracer), 0, 0, 0)

    return allow


def wedge_snapshot(
    pid: int,
    log_lines: list[str],
    case: str,
    reason: str,
    build_log: Path | None,
    secs: int,
) -> Path:
    """What a hung batch was doing, written before the kill frees the card:
    the vendor tools, every thread's state and wait channel, a debugger's
    backtrace of every thread, and the end of its log. Beside `build_log`, as
    <log>.gpu-wedge.txt. The kernel's side is in the host's dmesg, which a
    container cannot read, so it ends with that command."""
    if build_log is None or str(build_log) == os.devnull:
        fd, name = tempfile.mkstemp(prefix="gpu-wedge.")
        os.close(fd)
        out = Path(name)
    else:
        name = str(build_log)
        out = Path((name[:-4] if name.endswith(".log") else name) + ".gpu-wedge.txt")
    parts = [
        f"=== GPU batch, pid {pid}, {time.strftime('%F %T', time.gmtime())} UTC",
        f"watchdog: {reason}; in {case}",
    ]

    def run(cmd: list[str]) -> str:
        try:
            p = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=secs,
                check=False,
            )
            return (
                p.stdout
                + p.stderr
                + (
                    ""
                    if p.returncode == 0
                    else f"({' '.join(cmd)}: exit {p.returncode})\n"
                )
            )
        except subprocess.TimeoutExpired:
            return f"({' '.join(cmd)}: timed out after {secs}s)\n"

    for tool in (
        ["rocm-smi", "--showuse", "--showmemuse", "--showmeminfo", "vram"],
        ["nvidia-smi"],
    ):
        if shutil.which(tool[0]):
            parts += ["", f"=== {' '.join(tool)}", run(tool).rstrip("\n")]
    parts += ["", "=== threads: tid state utime stime (clock ticks) wchan"]
    for task in sorted(Path(f"/proc/{pid}/task").glob("*")):
        try:
            # comm is parenthesised and may hold spaces: read after the `)`.
            stat = task.joinpath("stat").read_text().rsplit(")", 1)[1].split()
            wchan = task.joinpath("wchan").read_text()
            parts.append(f"{task.name} {stat[0]} {stat[11]} {stat[12]} {wchan}")
        except (OSError, IndexError):
            continue
    debugger = (
        shutil.which("rocgdb")
        or shutil.which("/opt/rocm/bin/rocgdb")
        or shutil.which("gdb")
    )
    parts += ["", f"=== backtrace ({debugger or 'no debugger found'})"]
    if debugger:
        parts.append(
            run(
                [
                    debugger,
                    "-p",
                    str(pid),
                    "-batch",
                    "-ex",
                    "info threads",
                    "-ex",
                    "thread apply all bt 30",
                ]
            ).rstrip("\n")
        )
    parts += ["", "=== the last 40 lines of the batch's log", *log_lines[-40:], ""]
    parts += [
        "=== the kernel's side is in the HOST's dmesg; there, before a reboot:",
        "  sudo dmesg -T | grep -iE 'amdgpu|kfd|mes|ring|timeout|reset|NVRM|Xid'",
    ]
    out.write_text("\n".join(parts) + "\n")
    return out


def run_gpu_batch(
    tag: str,
    binary: Path,
    cfg: dict[str, str],
    build_log: Path | None,
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    echo_log: bool = False,
) -> int:
    """Run a gtest binary as one process inside the card the caller holds.

    One process, so the runtime's context creation (about 0.45s on the 3080)
    is paid once, and not several: processes on one card time-slice it
    (config.sh, by CTEST_JOBS). The output goes to `build_log`, and its
    summary lines are printed, tagged `<tag>:   [gpu]` (to the log too with
    `echo_log`). A failed case's whole output, and the case a crash stopped
    in, are printed on stderr, tagged `<tag>:   [gpu] |`.

    A batch that prints nothing for GPU_BATCH_STALL_SECONDS, or is still
    running GPU_BATCH_BUDGET_SECONDS after the start (0 turns either off), is
    killed and its case named: a wedged gfx1200 spins in a HIP sync printing
    nothing (issue #477). Its state is written down first (wedge_snapshot).
    Returns 0 on a pass, 1 otherwise."""
    if not os.access(binary, os.X_OK):
        say(f"{tag}: missing binary {binary} -- see {build_log}", err=True)
        return 1
    stall = int(cfg["GPU_BATCH_STALL_SECONDS"])
    budget = int(cfg["GPU_BATCH_BUDGET_SECONDS"])
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "log"
        with log.open("w") as out:
            proc = subprocess.Popen(
                [str(binary)],
                stdout=out,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                cwd=cwd,
                preexec_fn=_allow_tracer(os.getpid()),
            )
        killed = wedge = ""
        start = quiet = time.monotonic()
        last = -1
        while proc.poll() is None:
            size = log.stat().st_size
            now = time.monotonic()
            if size != last:
                last, quiet = size, now
            if stall > 0 and now - quiet >= stall:
                killed = f"no output for {stall}s (GPU_BATCH_STALL_SECONDS)"
            elif budget > 0 and now - start >= budget:
                killed = f"still running after {budget}s (GPU_BATCH_BUDGET_SECONDS)"
            if killed:
                lines = log.read_text(errors="replace").splitlines()
                case = running_case(lines)
                say(
                    f"{tag}:   [gpu] watchdog: {killed} -- killing pid {proc.pid},"
                    f" in {case} (see issue #477)",
                    err=True,
                )
                wedge = str(
                    wedge_snapshot(
                        proc.pid,
                        lines,
                        case,
                        killed,
                        build_log,
                        int(cfg["GPU_WEDGE_TOOL_SECONDS"]),
                    )
                )
                proc.terminate()
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            time.sleep(1)
        status = proc.wait()
        text = log.read_text(errors="replace")
    lines = text.splitlines()
    if build_log is not None:
        with build_log.open("a") as f:
            f.write(f"--- gpu batch: {binary} ---\n{text}")
    for line in lines:
        if SUMMARY_LINE.search(line):
            say(f"{tag}:   [gpu] {line}", log=build_log if echo_log else None)
    for line in failed_output(lines):
        say(f"{tag}:   [gpu] | {line}", err=True)
    if killed:
        say(f"{tag}:   [gpu] KILLED by the watchdog: {killed}", err=True)
        say(f"{tag}:   [gpu] hung in {running_case(lines)}", err=True)
        say(f"{tag}:   [gpu] state before the kill: {wedge}", err=True)
        say(
            f"{tag}: the GPU batch hung -- possibly a card wedge (issue #477: host"
            " spinning, GPU 100% busy, no memory activity). Check the card"
            " (devtools/doctor.sh; dmesg on the HOST), and rerun the gate on the"
            " other backend if it is the card.",
            err=True,
        )
        return 1
    if status != 0:
        # A signal reads as 128+N, as the shell reported it.
        code = 128 - status if status < 0 else status
        say(f"{tag}:   [gpu] exited {code}", err=True)
        say(f"{tag}:   [gpu] died in {running_case(lines)}", err=True)
        return 1
    return 0


# --- the GPU batches of a ctest listing ------------------------------------


def batchable(test: dict, gpu_bin: Path) -> str | None:
    """The gtest name of an entry one process of `gpu_bin` can run, else None.

    Its command must be exactly the binary with one --gtest_filter (and
    gtest's own flag): a TEST_LAUNCHER such as compute-sanitizer puts itself
    first, and running the binary bare would drop it. RUN_SERIAL entries stay
    with ctest, which keeps them alone on the card."""
    command = test.get("command") or []
    if not command or Path(command[0]).resolve() != gpu_bin.resolve():
        return None
    properties = {p["name"]: p["value"] for p in test.get("properties", [])}
    if "gpu" not in properties.get("LABELS", []) or properties.get("RUN_SERIAL"):
        return None
    filters = [a for a in command[1:] if a.startswith(FILTER_FLAG)]
    rest = [a for a in command[1:] if not a.startswith(FILTER_FLAG)]
    if len(filters) != 1 or any(a != "--gtest_also_run_disabled_tests" for a in rest):
        return None
    return filters[0][len(FILTER_FLAG) :]


def split_gpu_entries(
    listing_text: str, gpu_bins: list[Path]
) -> tuple[list[str], list[str]]:
    """From `ctest --show-only=json-v1`, the ctest names to exclude (every
    batched entry) and, per GPU binary, its batch as a GTEST_FILTER. A binary
    with nothing to batch, or whose filter would not fit one environment
    string, gets an empty filter and leaves its entries to ctest. The listing
    already has the preset's filters and any -R/-E/-L applied."""
    # ctest's own chatter (-V, say) can precede the document.
    listing = json.loads(listing_text[listing_text.index("{") :])
    excluded: list[str] = []
    filters: list[str] = []
    for gpu_bin in gpu_bins:
        names, cases = [], []
        for test in listing.get("tests", []):
            case = batchable(test, gpu_bin)
            if case is not None:
                names.append(test["name"])
                cases.append(case)
        joined = ":".join(cases)
        if len(joined) > MAX_FILTER:
            names, joined = [], ""
        excluded += names
        filters.append(joined)
    return excluded, filters


# --- phases ----------------------------------------------------------------


def run_phase(
    name: str, cmd: list[str], log: Path, env: dict[str, str] | None = None
) -> int:
    """build.sh's run_phase: a header, then the command's output, to stdout
    and the log; the command's own status."""
    say(f"--- {name}: {' '.join(cmd)} ---", log=log)
    with log.open("a") as f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
    return proc.wait()


def cpp_build(
    preset: str, fresh: bool, clean: bool, jobs: str, log: Path, quiet: bool
) -> int:
    """build.sh's cpp_build, the one copy of the configure and build phases."""
    out = subprocess.DEVNULL if quiet else None
    return lib(
        'BUILD_LOG=$1; cpp_build "$2" "$3" "$4" "$5"',
        str(log),
        preset,
        "1" if fresh else "0",
        "1" if clean else "0",
        jobs,
        capture=False,
        build=True,
        stdout=out,
        stderr=out,
    ).returncode


def fast_env() -> dict[str, str]:
    # The heavy reference cases skip (test/support/slow_test.h).
    return dict(os.environ, FREE_BREAKFAST_FAST_TESTS="1")


def python_half(tag: str, cfg: dict[str, str], log: Path | None) -> int:
    """The fast pytest set: FAST_TEST_ARGS over PREPUSH_PATHS, JOBS workers,
    -rs so the skips are listed, and no .pytest_cache written."""
    pytest = find_pytest(cfg)
    if pytest is None:
        say(f"{tag}: no pytest found in {cfg['VENV_PATHS']} or on PATH", err=True)
        say(
            f"{' ' * len(tag)}  run 'uv sync'"
            " (or set VENV_PATHS in devtools/config.sh)",
            err=True,
        )
        return 1
    cmd = [pytest]
    if cfg["JOBS"]:
        cmd += ["-n", cfg["JOBS"]]
    cmd += ["-q", "-rs", "-p", "no:cacheprovider"]
    cmd += shlex.split(cfg["FAST_TEST_ARGS"]) + shlex.split(cfg["PREPUSH_PATHS"])
    if log is None:
        say(f"--- pytest: {' '.join(cmd)} ---")
        return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode
    return run_phase("pytest", cmd, log)


# --- push ------------------------------------------------------------------


class DiffError(Exception):
    """The range a push gate was asked to read cannot be diffed."""


def changed_files(env: dict[str, str], root: Path = REPO_ROOT) -> list[str] | None:
    """The files a push changes: pre-commit's PRE_COMMIT_FROM_REF...TO_REF,
    the same three-dot diff its own `files:` filters read (two dots when the
    two have no merge base); by hand, origin/main...HEAD.

    None means both halves run: pre-commit sets no refs for a push of a whole
    history, and under pre-commit a range git cannot read (a shallow clone
    missing the remote's commit) is reported and tested in full, never
    skipped. By hand, an unreadable range raises DiffError instead, naming
    the fix."""
    old, new = env.get("PRE_COMMIT_FROM_REF"), env.get("PRE_COMMIT_TO_REF")
    hooked = env.get("PRE_COMMIT") == "1"
    if not (old and new):
        if hooked:
            return None
        old, new = "origin/main", "HEAD"
    git_env = {k: v for k, v in env.items() if not k.startswith("GIT_")}
    for spec in (f"{old}...{new}", f"{old}..{new}"):
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "diff",
                "--name-only",
                "--no-ext-diff",
                "-z",
                spec,
            ],
            env=git_env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return [f for f in proc.stdout.split("\0") if f]
    why = proc.stderr.strip().splitlines()[-1:] or ["no output"]
    if hooked:
        say(
            f"push: cannot diff {old}...{new} ({why[0]}) -- testing both halves",
            err=True,
        )
        return None
    raise DiffError(
        f"cannot diff {old}...{new} ({why[0]}). By hand, `push` diffs against"
        " origin/main: fetch it (`git fetch origin main`, or `git fetch"
        " --unshallow` in a shallow clone), or set PRE_COMMIT_FROM_REF and"
        " PRE_COMMIT_TO_REF to the range to test."
    )


def halves(files: list[str] | None) -> tuple[bool, bool]:
    """(C++, Python): which halves of the push gate a change fires."""
    if files is None:
        return True, True
    files = [f for f in files if not EXCLUDED.match(f)]
    return any(CPP_FILES.search(f) for f in files), any(
        PY_FILES.search(f) for f in files
    )


def cmake_floor() -> str | None:
    text = (REPO_ROOT / "CMakeLists.txt").read_text()
    m = re.search(r"^cmake_minimum_required\(VERSION ([0-9.]+)\)", text, re.M)
    return m.group(1) if m else None


def version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v))


def cpp_unavailable(backend: str) -> str | None:
    """Why this machine cannot run the C++ half, or None. The toolchain lives
    in the GPU container, so a host-side push skips rather than fails."""
    for tool in ("cmake", "ninja"):
        if shutil.which(tool) is None:
            return f"{tool} not found -- the C++ toolchain lives in the GPU container"
    # Present is not usable: a distro cmake below the tree's floor cannot even
    # parse CMakePresets.json, and that read as a build failure.
    floor = cmake_floor()
    out = subprocess.run(
        ["cmake", "--version"], capture_output=True, text=True, check=False
    ).stdout
    m = re.search(r"([0-9][0-9.]*)", out.splitlines()[0] if out else "")
    if floor and m and version_tuple(m.group(1)) < version_tuple(floor):
        return (
            f"cmake {m.group(1)} is below this tree's {floor} floor"
            " -- the C++ toolchain lives in the GPU container"
        )
    # Device nodes, not nvidia-smi: NVML fails inside a container whose CUDA
    # runtime works. `default`'s native arch reads a live device at configure.
    if backend == "hip":
        if not Path("/dev/kfd").exists():
            return "no HIP device node (/dev/kfd) -- the GPU smoke needs one"
    elif not Path("/dev/nvidiactl").exists() or not list(
        Path("/dev").glob("nvidia[0-9]*")
    ):
        return (
            "no CUDA device node (/dev/nvidia*)"
            " -- native configure and the GPU smoke need one"
        )
    return None


def cpp_push(cfg: dict[str, str], preset: str, jobs: str) -> tuple[int, str]:
    """The C++ half of the push gate: an incremental build of every test
    binary, the host binaries unlocked, then the GPU binaries in turn, each
    one process, under one hold of the card. (status, preset)."""
    tag = "push"
    if preset == "auto":
        preset = auto_preset()
    backend = backend_of_preset(preset, cfg)
    why = cpp_unavailable(backend)
    if why:
        say(f"{tag}: {why}")
        say(f"{tag}: skipping the C++ half (devtools/test.py pr remains the backstop)")
        return 0, preset
    report = report_dir_of(cfg["SLOW_TIER_REPORTS"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rev, _ = tree_rev()
    log = report / f"push-{stamp}-{rev}-{preset}.log"
    say(
        f"{tag}: {REPO_ROOT} @ {rev}  preset={preset}"
        "  (incremental build + host + GPU smoke)"
    )
    tree_rev_header(tag, log)
    say(f"{tag}: full log -> {log}")
    build_dir = build_dir_of(preset)

    say(f"{tag}: building (incremental, preset={preset}, -j {jobs or 'auto'}) ...")
    rc = cpp_build(preset, False, False, jobs, log, quiet=True)
    if rc != 0:
        say(f"{tag}: BUILD FAILED (rc={rc}) -- see {log}", err=True)
        for line in log.read_text(errors="replace").splitlines()[-25:]:
            say(line, err=True)
        return rc, preset

    env = fast_env()
    say(f"{tag}: host tests ...")
    for rel in HOST_BINS:
        binary = build_dir / rel
        if not os.access(binary, os.X_OK):
            say(f"{tag}: missing binary {binary} -- see {log}", err=True)
            rc = rc or 1
            continue
        say(f"{tag}:   --- {binary.name} ---")
        proc = subprocess.run(
            [str(binary)],
            env=env,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        with log.open("a") as f:
            f.write(f"--- {binary.name}: {binary} ---\n{proc.stdout}")
        for line in proc.stdout.splitlines():
            if SUMMARY_LINE.search(line):
                say(f"{tag}:   {line}")
        rc = rc or proc.returncode

    say(
        f"{tag}: GPU smoke"
        " (heavy reference cases skipped via FREE_BREAKFAST_FAST_TESTS) ..."
    )
    with Card(backend, cfg):
        for rel in GPU_BINS:
            say(f"{tag}:   --- {Path(rel).name} ---")
            if (
                run_gpu_batch(tag, build_dir / rel, cfg, log, env=env, cwd=REPO_ROOT)
                != 0
            ):
                rc = rc or 1
    return rc, preset


def cmd_push(args: argparse.Namespace, cfg: dict[str, str]) -> int:
    try:
        files = changed_files(dict(os.environ))
    except DiffError as e:
        say(f"push: {e}", err=True)
        return 2
    cpp, py = halves(files)
    what = "the whole history" if files is None else f"{len(files)} changed file(s)"
    say(
        f"push: {what} -> C++ half: {'yes' if cpp else 'no'},"
        f" Python half: {'yes' if py else 'no'}"
    )
    if args.dry_run:
        for f in files or []:
            say(f"push:   {f}")
        return 0
    rc = 0
    if py:
        say("push: Python half (fast pytest set) ...")
        rc = python_half("push", cfg, None)
    preset = None
    if cpp:
        crc, preset = cpp_push(
            cfg, args.preset or cfg["CMAKE_PRESET"], args.jobs or cfg["BUILD_JOBS"]
        )
        rc = rc or crc
    rev, _ = tree_rev()
    ran = ",".join(h for h, on in (("cpp", cpp), ("python", py)) if on) or "none"
    fields = f"rev={rev}  ran={ran}" + (f"  preset={preset}" if preset else "")
    if rc != 0:
        say(
            f"test.py push FAIL(rc={rc})  {fields}  (bypass one push with --no-verify)",
            err=True,
        )
        return rc
    say(f"test.py push PASS  {fields}")
    return 0


# --- pr --------------------------------------------------------------------


def tier_tests(
    cfg: dict[str, str],
    test_preset: str,
    build_dir: Path,
    passthrough: list[str],
    log: Path,
) -> int:
    """The C++ test phase, inside the held card. ctest runs the host entries,
    the RUN_SERIAL ones and anything a launcher wraps, CTEST_JOBS at once;
    the GPU entries that can share a process then run as one process per GPU
    binary, one after another. The split reads ctest's own listing, so the
    preset's filters and a `-- -R ...` pick the batches too; when the listing
    cannot be read, ctest runs every entry."""
    tag = "pr"
    env = fast_env()
    gpu_bins = [build_dir / rel for rel in GPU_BINS]
    listing = subprocess.run(
        ["ctest", "--preset", test_preset, "--show-only=json-v1", *passthrough],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        if listing.returncode != 0:
            raise ValueError(listing.returncode)
        excluded, filters = split_gpu_entries(listing.stdout, gpu_bins)
    except (ValueError, KeyError, TypeError):
        say(f"{tag}: could not split the GPU entries -- ctest runs them all", log=log)
        excluded, filters = [], [""] * len(gpu_bins)
    rc = 0
    with tempfile.NamedTemporaryFile("w", suffix=".exclude") as exclude:
        exclude.write("".join(f"{n}\n" for n in excluded))
        exclude.flush()
        # Before the passthrough, so a `-- -j 1` still wins.
        cmd = [
            "ctest",
            "--preset",
            test_preset,
            "--parallel",
            cfg["CTEST_JOBS"],
            "--exclude-from-file",
            exclude.name,
        ]
        rc = run_phase("ctest", cmd + passthrough, log, env=env)
    for gpu_bin, gtest_filter in zip(gpu_bins, filters, strict=True):
        if not gtest_filter:
            continue
        n = gtest_filter.count(":") + 1
        say(f"--- gpu batch: {n} entries in one process of {gpu_bin} ---", log=log)
        # ctest's WORKING_DIRECTORY, the binary's own build directory, and its
        # --gtest_also_run_disabled_tests.
        batch_env = dict(
            env, GTEST_FILTER=gtest_filter, GTEST_ALSO_RUN_DISABLED_TESTS="1"
        )
        if (
            run_gpu_batch(
                tag, gpu_bin, cfg, log, env=batch_env, cwd=gpu_bin.parent, echo_log=True
            )
            != 0
        ):
            rc = 1
    return rc


def cmd_pr(
    args: argparse.Namespace, cfg: dict[str, str], passthrough: list[str]
) -> int:
    tag = "pr"
    preset = args.preset or cfg["CMAKE_PRESET"]
    # --preset moves the test preset with it, or the run tests the wrong build
    # directory.
    test_preset = args.preset or cfg["CTEST_PRESET"]
    if preset == "auto":
        preset = auto_preset()
    if test_preset == "auto":
        test_preset = preset
    jobs = args.jobs or cfg["BUILD_JOBS"]
    do_cpp = args.only in (None, "cpp")
    do_py = args.only in (None, "python") and not args.no_test

    if (
        do_cpp
        and lib("cpp_require_build_tools", build=True, capture=False).returncode != 0
    ):
        return 1
    report = report_dir_of(args.report or cfg["SLOW_TIER_REPORTS"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rev, diff_hash = tree_rev()
    log = report / f"pr-{stamp}-{rev}-{preset}.log"
    say(f"{tag}: {REPO_ROOT} @ {rev}  preset={preset} -> {log}")
    tree_rev_header(tag, log)

    rc = 0
    if do_cpp:
        build_dir = build_dir_of(preset)
        rc = cpp_build(preset, args.fresh, args.clean, jobs, log, quiet=False)
        if rc == 0 and not args.no_test:
            has_test_preset = (
                bool(test_preset)
                and subprocess.run(
                    ["ctest", "--preset", test_preset, "--show-only"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                ).returncode
                == 0
            )
            if has_test_preset:
                say(
                    f"{tag}: skipping reference-grade validation cases"
                    " (FREE_BREAKFAST_FAST_TESTS=1)",
                    log=log,
                )
                with Card(backend_of_preset(test_preset, cfg), cfg):
                    rc = tier_tests(cfg, test_preset, build_dir, passthrough, log)
            else:
                say(
                    f"{tag}: no test preset '{test_preset or '<empty>'}' -- built only",
                    log=log,
                )
                say(
                    "    (CMakePresets.json defines test presets for default,"
                    " workstation, debug and asan)",
                    log=log,
                )
        # Advisory: the .cppm interfaces are not clang-tidy clean, and the
        # findings there are the design (see .clang-tidy). Never folded into rc.
        if rc == 0 and args.tidy:
            if shutil.which("clang-tidy") is None:
                say(f"{tag}: clang-tidy not found -- skipping --tidy", log=log)
            else:
                files = subprocess.run(
                    ["git", "-C", str(REPO_ROOT), "ls-files", "src/*.cpp"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.split()
                if not files:
                    say(f"{tag}: no src/*.cpp to tidy", log=log)
                elif (
                    run_phase(
                        "clang-tidy",
                        ["clang-tidy", "-p", str(build_dir), "--quiet", *files],
                        log,
                    )
                    != 0
                ):
                    say(f"{tag}: clang-tidy reported findings (advisory)", log=log)
    if do_py:
        prc = python_half(tag, cfg, log)
        rc = rc or prc

    verdict = "PASS" if rc == 0 else f"FAIL(rc={rc})"
    summary = f"test.py pr {verdict}  rev={rev}  preset={preset}  stamp={stamp}"
    if args.only:
        summary += f"  only={args.only}"
    if diff_hash:
        summary += f"  diff={diff_hash}"
    say(summary, log=log)
    with (report / "summary.log").open("a") as f:
        f.write(summary + "\n")
    return rc


# --- arguments -------------------------------------------------------------


def parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """The gate and its options, and everything after `--` (ctest's, for
    `pr`)."""
    passthrough: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1 :]
    parser = argparse.ArgumentParser(
        prog="devtools/test.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="More in devtools/test.py's docstring and docs/testing.md.",
    )
    gates = parser.add_subparsers(dest="gate", required=True, metavar="{push,pr}")
    push = gates.add_parser("push", help="what the pre-push hook runs")
    pr = gates.add_parser("pr", help="the gate before opening a PR")
    for p in (push, pr):
        p.add_argument(
            "--preset", help="configure/build preset (default: CMAKE_PRESET)"
        )
        p.add_argument("-j", "--jobs", help="compile jobs (default: BUILD_JOBS)")
    push.add_argument(
        "--dry-run",
        action="store_true",
        help="say which halves would run, and run neither",
    )
    pr.add_argument("--fresh", action="store_true", help="wipe the CMake cache first")
    pr.add_argument(
        "--clean", action="store_true", help="rebuild every object, keeping the cache"
    )
    pr.add_argument("--no-test", action="store_true", help="configure and build only")
    pr.add_argument(
        "--tidy", action="store_true", help="advisory clang-tidy after the build"
    )
    pr.add_argument("--only", choices=("cpp", "python"), help="run one half")
    pr.add_argument("--report", help="log directory (default: SLOW_TIER_REPORTS)")
    args = parser.parse_args(argv)
    if passthrough and args.gate != "pr":
        parser.error("only `pr` takes arguments after `--` (they go to ctest)")
    return args, passthrough


def main(argv: list[str]) -> int:
    args, passthrough = parse(argv)
    strip_git_env()
    os.chdir(REPO_ROOT)
    cfg = read_config()
    if args.gate == "push":
        return cmd_push(args, cfg)
    return cmd_pr(args, cfg, passthrough)


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    sys.exit(main(sys.argv[1:]))
