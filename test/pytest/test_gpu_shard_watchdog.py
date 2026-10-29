"""devtools/lib.sh run_gpu_sharded: a hung shard is killed and named (#477).

A wedged gfx1200 card leaves every shard on it spinning in a HIP sync with
nothing printed, and before the watchdog the gate waited on them forever. These
cases drive the launcher with fake shard binaries -- shell scripts that print
gtest-shaped lines and then hang, finish, or keep talking -- and short limits,
so no GPU and no real wedge is needed.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

_LIB = Path(__file__).resolve().parents[2] / "devtools" / "lib.sh"


def run_sharded(
    tmp_path: Path, shard_body: str, n: int = 2, stall: int = 2, budget: int = 0
) -> tuple[subprocess.CompletedProcess[str], float]:
    shard = tmp_path / "fake_shard"
    shard.write_text("#!/usr/bin/env bash\n" + shard_body)
    shard.chmod(0o755)
    env = dict(
        os.environ,
        GPU_SHARD_STALL_SECONDS=str(stall),
        GPU_SHARD_BUDGET_SECONDS=str(budget),
        BUILD_LOG=str(tmp_path / "build.log"),
    )
    script = f'. "{_LIB}"; run_gpu_sharded fake "{shard}" {n}'
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


def test_passing_shards_pass(tmp_path: Path) -> None:
    proc, _ = run_sharded(
        tmp_path,
        'echo "[ RUN      ] Suite.Case"\necho "[       OK ] Suite.Case"\n'
        'echo "[  PASSED  ] 1 test."\n',
    )
    assert proc.returncode == 0, proc.stderr
    assert "watchdog" not in proc.stderr
    assert "PASSED" in proc.stdout


def test_silent_shard_is_killed_and_its_case_named(tmp_path: Path) -> None:
    # Shard 0 finishes; shard 1 starts a case and goes silent, as a wedge does.
    # `exec` so the pid the launcher holds is the hanging process itself.
    proc, elapsed = run_sharded(
        tmp_path,
        'echo "[ RUN      ] Suite.Fine"\necho "[       OK ] Suite.Fine"\n'
        '[ "$GTEST_SHARD_INDEX" = 0 ] && exit 0\n'
        'echo "[ RUN      ] Wedge.Hangs/h2o"\nexec sleep 300\n',
    )
    assert proc.returncode == 1
    assert elapsed < 30
    assert "[shard 1/2] watchdog: no output for 2s" in proc.stderr
    assert "[shard 1/2] hung in Wedge.Hangs/h2o" in proc.stderr
    assert "[shard 0/2]" not in proc.stderr
    assert "#477" in proc.stderr
    # The killed shard's output still reaches the build log.
    assert "Wedge.Hangs/h2o" in (tmp_path / "build.log").read_text()


def test_chatty_shard_is_killed_at_its_budget(tmp_path: Path) -> None:
    # Never silent, so only the wall-clock budget can catch it.
    proc, elapsed = run_sharded(
        tmp_path,
        'echo "[ RUN      ] Loop.Forever"\n'
        "while :; do echo iteration; sleep 0.2; done\n",
        n=1,
        stall=0,
        budget=2,
    )
    assert proc.returncode == 1
    assert elapsed < 30
    assert "still running after 2s" in proc.stderr
    assert "hung in Loop.Forever" in proc.stderr


def test_term_ignoring_shard_is_killed(tmp_path: Path) -> None:
    proc, elapsed = run_sharded(
        tmp_path,
        "trap '' TERM\necho \"[ RUN      ] Stuck.InDriver\"\n"
        "while :; do sleep 1; done\n",
        n=1,
        stall=1,
    )
    assert proc.returncode == 1
    assert elapsed < 40
    assert "hung in Stuck.InDriver" in proc.stderr


@pytest.mark.parametrize("stall", [0, 2])
def test_crash_is_still_a_crash(tmp_path: Path, stall: int) -> None:
    proc, _ = run_sharded(
        tmp_path,
        'echo "[ RUN      ] Boom.Segv"\nexit 139\n',
        n=1,
        stall=stall,
    )
    assert proc.returncode == 1
    assert "exited 139" in proc.stderr
    assert "died in Boom.Segv" in proc.stderr
    assert "watchdog" not in proc.stderr
