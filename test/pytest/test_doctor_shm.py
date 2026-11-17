"""doctor's /dev/shm check: with more than one GPU, a /dev/shm under 512 MiB
warns, naming the devcontainers' --shm-size and the rebuild, because the
collectives library's shared-memory transport between cards with no peer
path does not fit docker's default 64 MiB. A stub nvidia-smi lists the cards
and a stub df reports the size, so no GPU and no container are needed.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _env(**extra: str) -> dict[str, str]:
    # Without the caller's GIT_*: lib.sh runs git at source time, and a
    # pre-push hook exports GIT_DIR (#694).
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "PRE_COMMIT"))
    }
    return {**env, **extra}


def _stub(bin_dir: Path, name: str, body: str) -> None:
    (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n")
    (bin_dir / name).chmod(0o755)


def _gpu_section(tmp_path: Path, cards: int, shm_kib: int | None) -> str:
    """doctor's `gpu devices` section over a throwaway repository, with
    `cards` NVIDIA cards and a /dev/shm of `shm_kib` KiB (None: unreadable)."""
    repo = tmp_path / "repo"
    (repo / "devtools").mkdir(parents=True)
    for name in ("doctor.sh", "lib.sh", "config.sh"):
        (repo / "devtools" / name).write_bytes((_ROOT / "devtools" / name).read_bytes())
    env = _env(
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.invalid",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.invalid",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_NOSYSTEM="1",
    )
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "init", "-q", "-b", "main"], env=env, check=True)
    subprocess.run([*git, "add", "-A"], env=env, check=True)
    subprocess.run([*git, "commit", "-qm", "init"], env=env, check=True)

    bin_dir = tmp_path / "stub"
    bin_dir.mkdir()
    listing = "\n".join(["NVIDIA RTX A6000, 8.6"] * cards)
    _stub(
        bin_dir,
        "nvidia-smi",
        f'[ "$1" = --query-gpu=name,compute_cap ] || exit 9\n'
        f"cat <<'EOF'\n{listing}\nEOF",
    )
    if shm_kib is None:
        _stub(bin_dir, "df", "exit 1")
    else:
        _stub(
            bin_dir,
            "df",
            "echo 'Filesystem 1024-blocks Used Available Capacity Mounted on'\n"
            f"echo 'shm {shm_kib} 0 {shm_kib} 0% /dev/shm'",
        )
    proc = subprocess.run(
        ["bash", str(repo / "devtools" / "doctor.sh")],
        cwd=repo,
        env={
            **env,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "DOCTOR_GPU_VENDORS": "nvidia",
            # No card nodes: the pin check (test_gpu_device_pins.py) stays out.
            "NVIDIA_DEV_ROOT": str(tmp_path / "no-dev"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.split("\ngpu devices\n", 1)[1].split("\nrequired paths\n", 1)[0]


def test_doctor_warns_on_a_small_shm_with_several_gpus(tmp_path: Path) -> None:
    section = _gpu_section(tmp_path, cards=4, shm_kib=64 * 1024)
    warns = [ln for ln in section.splitlines() if "[warn]" in ln]
    assert len(warns) == 1, section
    assert "/dev/shm is 64 MiB with 4 GPUs" in warns[0], section
    assert "--shm-size=1g: devtools/devcontainer.sh rebuild" in section, section
    assert "NCCL_P2P_LEVEL=SYS" in section, section


@pytest.mark.parametrize(
    ("cards", "shm_kib", "seen"),
    [
        (4, 1024 * 1024, "/dev/shm is 1024 MiB for 4 GPUs"),
        (2, 512 * 1024, "/dev/shm is 512 MiB for 2 GPUs"),
        (1, 64 * 1024, None),  # one card: no collective crosses a bridge
        (4, None, "/dev/shm size unreadable"),
    ],
)
def test_doctor_says_nothing_wrong_otherwise(
    tmp_path: Path, cards: int, shm_kib: int | None, seen: str | None
) -> None:
    section = _gpu_section(tmp_path, cards=cards, shm_kib=shm_kib)
    assert "[warn]" not in section, section
    if seen is None:
        assert "/dev/shm" not in section, section
    else:
        assert seen in section, section
