"""devtools/cross-backend-check.sh: the backend follows CROSS_CHECK_PRESET, so
rust/'s half is configured for what the root tree builds (#1528).

`--plan` prints what a run would use and exits before any cmake, so no
toolchain, docker or GPU is needed. A pair that disagrees exits 2, with a
message, rather than one quietly winning.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "devtools" / "cross-backend-check.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("python3") is None, reason="the script reads presets with python3"
)


def _plan(**env: str) -> subprocess.CompletedProcess[str]:
    # Without the caller's CROSS_CHECK_* (they are the input under test), or
    # its GIT_*: lib.sh runs git at source time, and a pre-push hook exports
    # GIT_DIR (#694).
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("GIT_", "PRE_COMMIT", "CROSS_CHECK_"))
    }
    clean.update(env)
    return subprocess.run(
        [str(_SCRIPT), "--plan"],
        cwd=_ROOT,
        env=clean,
        capture_output=True,
        text=True,
        check=False,
    )


def _fields(proc: subprocess.CompletedProcess[str]) -> dict[str, str]:
    assert proc.returncode == 0, proc.stderr
    return dict(line.split("=", 1) for line in proc.stdout.splitlines())


@pytest.mark.parametrize(
    ("env", "backend", "preset", "probe"),
    [
        # config.sh's defaults: HIP, hip.
        ({}, "HIP", "hip", "hipconfig"),
        # The preset alone decides, through `inherits` (ci-cuda -> base: CUDA,
        # ci-hip -> hip: HIP).
        ({"CROSS_CHECK_PRESET": "ci-cuda"}, "CUDA", "ci-cuda", "nvcc"),
        ({"CROSS_CHECK_PRESET": "ci-hip"}, "HIP", "ci-hip", "hipconfig"),
        ({"CROSS_CHECK_PRESET": "default"}, "CUDA", "default", "nvcc"),
        # The backend alone picks its preset.
        ({"CROSS_CHECK_BACKEND": "CUDA"}, "CUDA", "ci-cuda", "nvcc"),
        ({"CROSS_CHECK_BACKEND": "hip"}, "HIP", "hip", "hipconfig"),
        # Both, agreeing.
        (
            {"CROSS_CHECK_BACKEND": "cuda", "CROSS_CHECK_PRESET": "ci-cuda"},
            "CUDA",
            "ci-cuda",
            "nvcc",
        ),
    ],
)
def test_backend_follows_preset(
    env: dict[str, str], backend: str, preset: str, probe: str
) -> None:
    plan = _fields(_plan(**env))
    assert plan["backend"] == backend
    assert plan["preset"] == preset
    assert plan["rust_preset"] == backend.lower()
    assert plan["probe"] == probe
    assert plan["image"].endswith(f":{backend.lower()}")
    # The two backends never share a cache.
    assert plan["build_dir"].endswith("-cuda") == (backend == "CUDA")


@pytest.mark.parametrize(
    "env",
    [
        {"CROSS_CHECK_BACKEND": "HIP", "CROSS_CHECK_PRESET": "ci-cuda"},
        {"CROSS_CHECK_BACKEND": "CUDA", "CROSS_CHECK_PRESET": "ci-hip"},
    ],
)
def test_contradictory_pair_is_refused(env: dict[str, str]) -> None:
    proc = _plan(**env)
    assert proc.returncode == 2
    assert proc.stdout == ""
    assert f"CROSS_CHECK_PRESET={env['CROSS_CHECK_PRESET']}" in proc.stderr


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"CROSS_CHECK_PRESET": "no-such-preset"}, "no configure preset"),
        ({"CROSS_CHECK_BACKEND": "SYCL"}, "neither CUDA nor HIP"),
    ],
)
def test_unresolvable_is_refused(env: dict[str, str], message: str) -> None:
    proc = _plan(**env)
    assert proc.returncode == 2
    assert message in proc.stderr


def test_explicit_overrides_are_kept() -> None:
    plan = _fields(
        _plan(
            CROSS_CHECK_PRESET="ci-cuda",
            CROSS_CHECK_TOOL="my-nvcc",
            CROSS_CHECK_IMAGE="img:tag",
            CROSS_CHECK_BUILD_DIR="build-mine",
        )
    )
    assert (plan["probe"], plan["image"], plan["build_dir"]) == (
        "my-nvcc",
        "img:tag",
        "build-mine",
    )
