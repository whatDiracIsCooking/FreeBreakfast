#!/usr/bin/env python3
"""The metrics recorder (docs/testing.md "Metrics").

One run becomes one JSON record, on one line, appended to
runs/YYYY-MM.jsonl on the orphan branch `metrics`, which is written through
its own worktree (METRICS_WORKTREE in devtools/config.sh, under the primary
checkout) and never through main's tree or index. The branch and the
worktree are created on first use.

The record is read from what the run already wrote:

  gtest_cases   a GPU batch's --gtest_output=json
  junit_cases   ctest's --output-junit, and pytest's --junitxml
                (junit_family=xunit1, which carries each case's file)
  ninja_files   the build's .ninja_log, for a clean build's per-file times
  probe         the card and the toolchain the build directory was made with

and make_record puts them together under the documented field names;
append writes it. `devtools/test.py pr --record` is one caller; the nightly
(#985) is the other.

Standard library only, like test.py: it runs under a bare python3.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import socket
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

SCHEMA = 1
BRANCH = "metrics"
KINDS = ("nightly", "weekly", "manual", "shadow")
STATUSES = ("pass", "fail", "skip")

# The record's fields, in docs/testing.md's order. A test holds the table
# there to this tuple.
FIELDS = (
    "schema",
    "kind",
    "started",
    "finished",
    "host",
    "commit",
    "backend",
    "preset",
    "gpu",
    "toolchain",
    "build",
    "cases",
    "selection",
    "result",
)


class MetricsError(Exception):
    """A record that cannot be written as asked."""


# --- the run's own outputs --------------------------------------------------


def _seconds(value: object) -> float:
    """gtest writes "0.012s"; JUnit writes "0.012"."""
    text = str(value or "0").strip().removesuffix("s")
    try:
        return round(float(text), 6)
    except ValueError:
        return 0.0


def gtest_cases(path: Path, binary: str) -> list[dict]:
    """Every case of one gtest process's --gtest_output=json report.

    A case GTEST_SKIP'd (result SKIPPED) or never run (status NOTRUN, a
    disabled case the filter left out) is `skip`; one with any failure is
    `fail`. A process that crashed wrote no report, and its cases are absent:
    the record's `result` carries that run's failure."""
    doc = json.loads(Path(path).read_text())
    cases = []
    for suite in doc.get("testsuites", []):
        for case in suite.get("testsuite", []):
            if case.get("failures"):
                status = "fail"
            elif case.get("status") == "NOTRUN" or case.get("result") in (
                "SKIPPED",
                "SUPPRESSED",
            ):
                status = "skip"
            else:
                status = "pass"
            cases.append(
                {
                    "name": f"{suite['name']}.{case['name']}",
                    "binary": binary,
                    "status": status,
                    "seconds": _seconds(case.get("time")),
                }
            )
    return cases


def _pytest_node(file: str, classname: str, name: str) -> str:
    """A pytest node id from xunit1's file, classname and name: the classname
    is the module's dotted path, then any class."""
    module = file.removesuffix(".py").replace("/", ".")
    rest = classname[len(module) :].lstrip(".") if classname.startswith(module) else ""
    return "::".join([file, *([rest.replace(".", "::")] if rest else []), name])


def junit_cases(path: Path, binaries: dict[str, str] | None = None) -> list[dict]:
    """Every case of a JUnit XML report: ctest's --output-junit or pytest's
    --junitxml.

    A ctest case's binary is looked up by its name in `binaries` (from
    ctest's --show-only=json-v1 listing), `ctest` when absent. A pytest case,
    known by its `file` attribute (junit_family=xunit1), is named by its node
    id, with binary `pytest`. <failure> or <error> is `fail`; <skipped>, and
    ctest's notrun or disabled, is `skip`."""
    root = ET.parse(path).getroot()
    cases = []
    for case in root.iter("testcase"):
        name = case.get("name", "")
        file = case.get("file")
        if file:
            name = _pytest_node(file, case.get("classname", ""), name)
            binary = "pytest"
        else:
            binary = (binaries or {}).get(name, "ctest")
        tags = {child.tag for child in case}
        if tags & {"failure", "error"} or case.get("status") == "fail":
            status = "fail"
        elif "skipped" in tags or case.get("status") in ("notrun", "disabled"):
            status = "skip"
        else:
            status = "pass"
        cases.append(
            {
                "name": name,
                "binary": binary,
                "status": status,
                "seconds": _seconds(case.get("time")),
            }
        )
    return cases


def ctest_binaries(listing_text: str) -> dict[str, str]:
    """ctest's --show-only=json-v1 listing as {test name: binary name}."""
    listing = json.loads(listing_text[listing_text.index("{") :])
    out = {}
    for test in listing.get("tests", []):
        command = test.get("command") or []
        if command:
            out[test["name"]] = Path(command[0]).name
    return out


def ninja_offset(path: Path) -> tuple[int, int] | None:
    """Where a .ninja_log ends now, as (inode, size): what ninja_files reads
    from once the build has appended to it. None when there is no log yet."""
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return st.st_ino, st.st_size


def ninja_files(path: Path, since: tuple[int, int] | None = None) -> list[dict]:
    """Each output's build time, from a .ninja_log (v5 to v7: start, end, mtime,
    output, hash; milliseconds from the start of that ninja run).

    With `since` (ninja_offset before the build), the entries appended after
    it: everything this build ran, a restart after CMake regenerated included.
    Without it, or when ninja rewrote the log meanwhile (a recompaction is a
    new file, so a new inode), the log's last run, which starts where an
    entry's end time goes backwards. An output built twice keeps its last
    time."""
    path = Path(path)
    data = path.read_bytes()
    offset = 0
    if since is not None:
        ino, size = since
        st = path.stat()
        if st.st_ino == ino and st.st_size >= size:
            offset = size
        else:
            since = None
    entries = []
    for line in data[offset:].decode(errors="replace").splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        try:
            start, end = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        entries.append((start, end, fields[3]))
    if since is None:
        last = 0
        for i in range(1, len(entries)):
            if entries[i][1] < entries[i - 1][1]:
                last = i
        entries = entries[last:]
    times: dict[str, float] = {}
    for start, end, output in entries:
        times.pop(output, None)
        times[output] = round((end - start) / 1000, 3)
    return [{"file": f, "seconds": s} for f, s in times.items()]


# --- the machine ------------------------------------------------------------


def _run(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        p = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout if p.returncode == 0 else None


def _cache(build_dir: Path) -> dict[str, str]:
    """CMakeCache.txt's NAME:TYPE=VALUE entries."""
    out = {}
    try:
        text = (Path(build_dir) / "CMakeCache.txt").read_text(errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_.+-]+):[A-Z]+=(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _compiler_version(build_dir: Path, lang: str) -> str | None:
    """CMAKE_<LANG>_COMPILER_VERSION as the configure recorded it: the
    compiler that built this directory, not whichever is first on PATH."""
    for f in sorted(Path(build_dir).glob(f"CMakeFiles/*/CMake{lang}Compiler.cmake")):
        m = re.search(rf'set\(CMAKE_{lang}_COMPILER_VERSION "([^"]+)"\)', f.read_text())
        if m:
            return m.group(1)
    return None


def gpu_info(backend: str) -> dict:
    """The card's name, driver version and VRAM (MiB); None where the vendor
    tool cannot say. The first card: a gate runs on one."""
    info: dict = {"name": None, "driver": None, "vram_mib": None}
    if backend == "cuda":
        out = _run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        if out and out.strip():
            name, driver, vram = (
                [f.strip() for f in out.splitlines()[0].split(",")] + ["", "", ""]
            )[:3]
            info.update(
                name=name or None,
                driver=driver or None,
                vram_mib=int(vram) if vram.isdigit() else None,
            )
        return info
    out = _run(
        [
            "rocm-smi",
            "--showproductname",
            "--showdriverversion",
            "--showmeminfo",
            "vram",
            "--json",
        ]
    )
    try:
        doc = json.loads(out or "")
    except ValueError:
        return info
    cards = [v for k, v in sorted(doc.items()) if k.startswith("card")]
    if cards:
        card = cards[0]
        info["name"] = card.get("Card Series") or card.get("Card series")
        total = card.get("VRAM Total Memory (B)")
        if str(total).isdigit():
            info["vram_mib"] = int(total) // (1024 * 1024)
    info["driver"] = (doc.get("system") or {}).get("Driver version")
    return info


def toolchain(build_dir: Path, backend: str) -> dict:
    """clang, CMake and nvcc (CUDA) or hipcc (HIP) versions. The compilers'
    are the build directory's own, read from its configure."""
    cache = _cache(build_dir)
    major = cache.get("CMAKE_CACHE_MAJOR_VERSION")
    cmake = (
        ".".join(
            cache.get(f"CMAKE_CACHE_{p}_VERSION", "0")
            for p in ("MAJOR", "MINOR", "PATCH")
        )
        if major
        else None
    )
    out = {"clang": _compiler_version(build_dir, "CXX"), "cmake": cmake}
    if backend == "cuda":
        out["nvcc"] = _compiler_version(build_dir, "CUDA")
    else:
        hip = _run(["hipconfig", "--version"]) or _run(
            ["/opt/rocm/bin/hipconfig", "--version"]
        )
        out["hipcc"] = hip.strip() if hip else None
    return out


# CMake's false constants, as if() reads them: what the root CMakeLists.txt's
# `if(FREE_BREAKFAST_SCCACHE)` sees. The weekly nightly configures with
# -DFREE_BREAKFAST_SCCACHE=OFF, which find_program then leaves alone.
_CMAKE_FALSE = ("", "0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND")


def launcher_on(build_dir: Path) -> bool:
    """Whether the build compiled through sccache: the root CMakeLists.txt's
    find_program(FREE_BREAKFAST_SCCACHE), which is what installs the
    launchers. A false constant or a *-NOTFOUND is off."""
    value = _cache(build_dir).get("FREE_BREAKFAST_SCCACHE", "")
    return value.upper() not in _CMAKE_FALSE and not value.endswith("-NOTFOUND")


DMI = Path("/sys/class/dmi/id")


def host_name(override: str | None = None, dmi: Path = DMI) -> str:
    """The record's `host`: the machine, named the same way on the host and
    in every container on it, and across a devcontainer rebuild.

    METRICS_HOST (devtools/config.sh, or the environment) wins when set.
    Otherwise the board's vendor and name from DMI (else the system's), which
    a container reads as the host does: /sys/class/dmi/id is world-readable
    there, where the hostname is the container's id and /etc/machine-id the
    image's. Only where DMI says nothing (no /sys, a non-x86 board) is it the
    hostname."""
    if override and override.strip():
        return override.strip()

    def field(name: str) -> str:
        try:
            return (dmi / name).read_text(errors="replace").strip()
        except OSError:
            return ""

    for vendor, name in (
        ("board_vendor", "board_name"),
        ("sys_vendor", "product_name"),
    ):
        if field(name):
            return " ".join(p for p in (field(vendor), field(name)) if p)
    return socket.gethostname()


def probe(build_dir: Path, backend: str) -> tuple[dict, dict]:
    """(gpu, toolchain) for this machine and this build directory."""
    return gpu_info(backend), toolchain(build_dir, backend)


# --- the record -------------------------------------------------------------


def build_info(
    *,
    mode: str,
    launcher: bool,
    jobs: int,
    seconds: float,
    files: list[dict] | None = None,
) -> dict:
    """The record's `build`. Per-file times only for a clean build: a warm
    build's are cache hits, and its own times are not a trend."""
    if mode not in ("warm", "clean"):
        raise MetricsError(f"build mode {mode!r} is neither warm nor clean")
    out = {
        "mode": mode,
        "launcher": "on" if launcher else "off",
        "jobs": int(jobs),
        "seconds": round(seconds, 3),
    }
    if mode == "clean":
        out["files"] = files or []
    return out


def make_record(
    *,
    kind: str,
    started: str,
    finished: str,
    commit: str,
    backend: str,
    preset: str,
    gpu: dict,
    toolchain: dict,
    build: dict | None,
    cases: list[dict],
    result: str,
    selection: dict | None = None,
    host: str | None = None,
) -> dict:
    """One record, schema 1, with docs/testing.md's fields in its order.
    `build` is None for a run that built nothing; `selection` belongs to
    shadow records only; `host` defaults to host_name()."""
    if kind not in KINDS:
        raise MetricsError(f"kind {kind!r} is not one of {', '.join(KINDS)}")
    if result not in ("pass", "fail"):
        raise MetricsError(f"result {result!r} is neither pass nor fail")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise MetricsError(f"commit {commit!r} is not a full SHA")
    if (selection is not None) != (kind == "shadow"):
        raise MetricsError("`selection` is for shadow records, and only those")
    for case in cases:
        if case.get("status") not in STATUSES:
            raise MetricsError(f"case {case.get('name')!r} has no valid status")
    record = {
        "schema": SCHEMA,
        "kind": kind,
        "started": started,
        "finished": finished,
        "host": host or host_name(),
        "commit": commit,
        "backend": backend,
        "preset": preset,
        "gpu": gpu,
        "toolchain": toolchain,
        "build": build,
        "cases": cases,
        "selection": selection,
        "result": result,
    }
    if selection is None:
        del record["selection"]
    return record


# --- the metrics branch -----------------------------------------------------


def _git_env() -> dict[str, str]:
    """No GIT_* (pre-commit's pre-push stage exports GIT_DIR, which would aim
    every command here at the hook's repository, #694) and no PRE_COMMIT_*."""
    return {
        k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "PRE_COMMIT"))
    }


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git", "-C", str(cwd), *args],
        env=_git_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise MetricsError(
            f"git {' '.join(args)}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc


def ensure_worktree(repo: Path, worktree: Path) -> None:
    """The `metrics` worktree at `worktree`, made if absent: on the local
    branch if there is one, else tracking origin's, else on a new orphan
    branch. git worktree add writes under the common git directory and into
    `worktree`, never into the checkout it is run from."""
    worktree = Path(worktree)
    if (worktree / ".git").exists():
        head = _git(worktree, "symbolic-ref", "--short", "HEAD", check=False)
        on = head.stdout.strip() or "a detached HEAD"
        if on != BRANCH:
            raise MetricsError(f"{worktree} is a worktree of {on}, not {BRANCH}")
        return
    if worktree.exists() and any(worktree.iterdir()):
        raise MetricsError(f"{worktree} exists and is not the {BRANCH} worktree")
    # A worktree directory deleted by hand is still registered; forget it.
    _git(repo, "worktree", "prune")
    worktree.parent.mkdir(parents=True, exist_ok=True)

    def has(ref: str) -> bool:
        return (
            _git(repo, "rev-parse", "--verify", "--quiet", ref, check=False).returncode
            == 0
        )

    if has(f"refs/heads/{BRANCH}"):
        _git(repo, "worktree", "add", str(worktree), BRANCH)
    elif has(f"refs/remotes/origin/{BRANCH}"):
        _git(
            repo,
            "worktree",
            "add",
            "--track",
            "-b",
            BRANCH,
            str(worktree),
            f"origin/{BRANCH}",
        )
    else:
        _git(repo, "worktree", "add", "--orphan", "-b", BRANCH, str(worktree))


def append(record: dict, repo: Path, worktree: Path) -> tuple[str, Path]:
    """Append `record` to runs/<started's YYYY-MM>.jsonl on `metrics`, and
    commit it there. (the commit's SHA, the file). Appends from two runs at
    once are serialised by a lock in the common git directory."""
    worktree = Path(worktree)
    common = Path(
        _git(
            repo, "rev-parse", "--path-format=absolute", "--git-common-dir"
        ).stdout.strip()
    )
    line = json.dumps(record, separators=(",", ":"), sort_keys=False)
    if "\n" in line:
        raise MetricsError("a record must be one line")
    with (common / "metrics.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ensure_worktree(repo, worktree)
        rel = Path("runs") / f"{record['started'][:7]}.jsonl"
        target = worktree / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as f:
            f.write(line + "\n")
        _git(worktree, "add", "--", str(rel))
        message = (
            f"metrics: {record['kind']} {record['commit'][:12]}"
            f" {record['preset']} {record['result']}"
        )
        # commit-tree and update-ref, not `git commit`: the hooks installed
        # in the common git directory are main's pre-commit hooks, which
        # have no config on a data branch, and no pre-commit at all to run
        # on the host. The worktree's HEAD names refs/heads/metrics, so the
        # update leaves it clean.
        parent = _git(worktree, "rev-parse", "--verify", "--quiet", "HEAD", check=False)
        old = parent.stdout.strip() if parent.returncode == 0 else "0" * 40
        tree = _git(worktree, "write-tree").stdout.strip()
        sha = _git(
            worktree,
            "commit-tree",
            tree,
            *(["-p", old] if parent.returncode == 0 else []),
            "-m",
            message,
        ).stdout.strip()
        _git(worktree, "update-ref", "-m", message, f"refs/heads/{BRANCH}", sha, old)
    return sha, target
