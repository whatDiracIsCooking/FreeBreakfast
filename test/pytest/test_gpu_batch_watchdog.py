"""devtools/lib.sh run_gpu_batch: a hung batch is killed and named (#477).

A wedged gfx1200 card leaves the process on it spinning in a HIP sync with
nothing printed, and before the watchdog the gate waited on it forever. These
cases drive the launcher with a fake batch binary -- a shell script that prints
gtest-shaped lines and then hangs, finishes, or keeps talking -- and short
limits, so no GPU and no real wedge is needed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

_LIB = Path(__file__).resolve().parents[2] / "devtools" / "lib.sh"


def run_batch(
    tmp_path: Path, body: str, stall: int = 2, budget: int = 0
) -> tuple[subprocess.CompletedProcess[str], float]:
    batch = tmp_path / "fake_batch"
    batch.write_text("#!/usr/bin/env bash\n" + body)
    batch.chmod(0o755)
    env = dict(
        os.environ,
        GPU_BATCH_STALL_SECONDS=str(stall),
        GPU_BATCH_BUDGET_SECONDS=str(budget),
        GPU_WEDGE_TOOL_SECONDS="20",
        BUILD_LOG=str(tmp_path / "build.log"),
    )
    script = f'. "{_LIB}"; run_gpu_batch fake "{batch}"'
    start = time.monotonic()
    proc = subprocess.run(
        ["bash", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return proc, time.monotonic() - start


def test_passing_batch_passes(tmp_path: Path) -> None:
    proc, _ = run_batch(
        tmp_path,
        'echo "[ RUN      ] Suite.Case"\necho "[       OK ] Suite.Case"\n'
        'echo "[  PASSED  ] 1 test."\n',
    )
    assert proc.returncode == 0, proc.stderr
    assert "watchdog" not in proc.stderr
    assert "fake:   [gpu] [  PASSED  ] 1 test." in proc.stdout


def test_silent_batch_is_killed_and_its_case_named(tmp_path: Path) -> None:
    # One case finishes; the next starts and goes silent, as a wedge does.
    # `exec` so the pid the launcher holds is the hanging process itself.
    proc, elapsed = run_batch(
        tmp_path,
        'echo "[ RUN      ] Suite.Fine"\necho "[       OK ] Suite.Fine"\n'
        'echo "[ RUN      ] Wedge.Hangs/h2o"\nexec sleep 300\n',
    )
    assert proc.returncode == 1
    assert elapsed < 30
    assert "[gpu] watchdog: no output for 2s" in proc.stderr
    assert "[gpu] hung in Wedge.Hangs/h2o" in proc.stderr
    assert "#477" in proc.stderr
    # The killed batch's output still reaches the build log.
    assert "Wedge.Hangs/h2o" in (tmp_path / "build.log").read_text()
    # Its state was written down before the kill, beside the build log.
    snapshot = tmp_path / "build.gpu-wedge.txt"
    assert f"state before the kill: {snapshot}" in proc.stderr
    text = snapshot.read_text()
    assert "in Wedge.Hangs/h2o" in text
    assert "=== threads:" in text
    assert "sudo dmesg" in text


@pytest.mark.skipif(
    not (
        shutil.which("rocgdb")
        or Path("/opt/rocm/bin/rocgdb").exists()
        or shutil.which("gdb")
    ),
    reason="no rocgdb or gdb to attach",
)
def test_snapshot_backtraces_the_hung_batch(tmp_path: Path) -> None:
    # Yama's ptrace_scope=1 forbids a sibling attaching; ptraceable-exec.py is
    # what lets the watchdog's debugger in, so a refusal here is a regression.
    proc, _ = run_batch(tmp_path, 'echo "[ RUN      ] Wedge.Spins"\nexec sleep 300\n')
    assert proc.returncode == 1
    text = (tmp_path / "build.gpu-wedge.txt").read_text()
    assert "Operation not permitted" not in text, text
    assert "Thread 1" in text, text


def test_chatty_batch_is_killed_at_its_budget(tmp_path: Path) -> None:
    # Never silent, so only the wall-clock budget can catch it.
    proc, elapsed = run_batch(
        tmp_path,
        'echo "[ RUN      ] Loop.Forever"\n'
        "while :; do echo iteration; sleep 0.2; done\n",
        stall=0,
        budget=2,
    )
    assert proc.returncode == 1
    assert elapsed < 30
    assert "still running after 2s" in proc.stderr
    assert "hung in Loop.Forever" in proc.stderr


def test_term_ignoring_batch_is_killed(tmp_path: Path) -> None:
    proc, elapsed = run_batch(
        tmp_path,
        "trap '' TERM\necho \"[ RUN      ] Stuck.InDriver\"\n"
        "while :; do sleep 1; done\n",
        stall=1,
    )
    assert proc.returncode == 1
    assert elapsed < 40
    assert "hung in Stuck.InDriver" in proc.stderr


@pytest.mark.parametrize("stall", [0, 2])
def test_crash_is_still_a_crash(tmp_path: Path, stall: int) -> None:
    proc, _ = run_batch(
        tmp_path,
        'echo "[ RUN      ] Boom.Segv"\nexit 139\n',
        stall=stall,
    )
    assert proc.returncode == 1
    assert "exited 139" in proc.stderr
    assert "died in Boom.Segv" in proc.stderr
    assert "watchdog" not in proc.stderr
