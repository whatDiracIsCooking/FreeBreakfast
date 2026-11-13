"""GPU_LOCK_DIR: a relative path sits under the main checkout, an absolute one
is used as given, in lib.sh's gpu_lock_dir and test.py's Card alike (#1332).

An absolute value used to be joined under the checkout too, so a run quietly
locked a stray main/home/... directory and held no shared card lock. No GPU:
the locks are plain flock(2) files in a temporary directory.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_LIB = _ROOT / "devtools" / "lib.sh"
_spec = importlib.util.spec_from_file_location(
    "devtools_test_py", _ROOT / "devtools" / "test.py"
)
gates = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gates)

pytestmark = pytest.mark.skipif(shutil.which("flock") is None, reason="needs flock(1)")


def _env(**extra: str) -> dict[str, str]:
    # Without the caller's GIT_*: lib.sh runs git at source time, and a
    # pre-push hook exports GIT_DIR (#694).
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "PRE_COMMIT"))
    }
    env.update(extra)
    return env


def _lib(snippet: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'. "{_LIB}" && {snippet}'],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _primary_root() -> Path:
    return Path(gates.read_config(_env())["PRIMARY_ROOT"] or str(_ROOT))


def test_absolute_lock_dir_is_used_as_given(tmp_path: Path) -> None:
    locks = tmp_path / "gpu-locks"
    env = _env(GPU_LOCK_DIR=str(locks))
    proc = _lib("gpu_lock_dir", env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(locks)

    # The shell tools take the card there, and nowhere under the checkout.
    proc = _lib('gpu_take_card hip && ls "$(gpu_lock_dir)"', env)
    assert proc.returncode == 0, proc.stderr
    assert sorted(proc.stdout.split()) == ["hip.0.lock", "hip.1.lock"]
    stray = _primary_root() / str(locks).lstrip("/")
    assert not stray.exists(), f"stray lock dir {stray}"

    # test.py's Card resolves the same directory, and lib.sh sees its hold.
    card = gates.Card("cuda", gates.read_config(env))
    assert card.dir == locks
    assert card._take()
    try:
        assert _lib("gpu_card_held cuda", env).returncode == 0
    finally:
        card.release()
    assert _lib("gpu_card_held cuda", env).returncode == 1
    assert not stray.exists()


def test_relative_lock_dir_resolves_under_the_main_checkout(tmp_path: Path) -> None:
    root = _primary_root()
    rel = os.path.relpath(tmp_path / "gpu-locks", root)
    assert not rel.startswith("/")
    env = _env(GPU_LOCK_DIR=rel)
    proc = _lib("gpu_lock_dir", env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"{root}/{rel}"
    assert gates.Card("cuda", gates.read_config(env)).dir == root / rel
