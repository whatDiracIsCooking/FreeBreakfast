"""Every NVIDIA card pinned with --device= (#1702).

devtools/devcontainer.sh adds one `--device=/dev/nvidiaN` per host card the
config does not already pass, through a generated --override-config, and
doctor.sh warns about any card that is not pinned: inside the container, a
card node that does not open read-write; on the host, one missing from a
container's HostConfig.Devices. `NVIDIA_DEV_ROOT` stands in for /dev and a
stub `docker` answers `ps`/`inspect`, so no GPU, no docker and no container
are needed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_CUDA = _ROOT / ".devcontainer" / "cuda" / "devcontainer.json"
_COMBINED = _ROOT / ".devcontainer" / "combined" / "devcontainer.json"
_HIP = _ROOT / ".devcontainer" / "hip" / "devcontainer.json"


def _env(**extra: str) -> dict[str, str]:
    # Without the caller's GIT_*: lib.sh runs git at source time, and a
    # pre-push hook exports GIT_DIR (#694).
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "PRE_COMMIT"))
    }
    return {**env, **extra}


def _dev(tmp_path: Path, cards: int, extra: tuple[str, ...] = ()) -> Path:
    """A stand-in /dev: nvidia0..cards-1, plus the nodes that are NOT cards."""
    dev = tmp_path / "dev"
    dev.mkdir()
    for name in (
        *(f"nvidia{i}" for i in range(cards)),
        "nvidiactl",
        "nvidia-uvm",
        "nvidia-uvm-tools",
        "nvidia-modeset",
        *extra,
    ):
        (dev / name).write_text("")
    (dev / "nvidia-caps").mkdir()
    return dev


def _lib(script: str, *args: str, **env: str) -> str:
    proc = subprocess.run(
        ["bash", "-c", f'. "{_ROOT}/devtools/lib.sh"; {script}', "lib", *args],
        env=_env(**env),
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def _jsonc(text: str) -> dict:
    """Parse devcontainer JSONC: comments outside strings, trailing commas."""
    out, i, in_str = [], 0, False
    while i < len(text):
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\":
                out.append(text[i + 1])
                i += 2
                continue
            in_str = c != '"'
            i += 1
        elif c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif text.startswith("//", i):
            while i < len(text) and text[i] != "\n":
                i += 1
        elif text.startswith("/*", i):
            i = text.index("*/", i) + 2
        else:
            out.append(c)
            i += 1
    return json.loads(re.sub(r",(\s*[\]}])", r"\1", "".join(out)))


# --- the device list --------------------------------------------------------


def test_card_nodes_are_the_cards_alone_in_card_order(tmp_path: Path) -> None:
    dev = _dev(tmp_path, 3, extra=("nvidia10", "nvidia1x"))
    nodes = _lib("nvidia_card_nodes", NVIDIA_DEV_ROOT=str(dev)).split()
    assert [Path(n).name for n in nodes] == [
        "nvidia0",
        "nvidia1",
        "nvidia2",
        "nvidia10",
    ]


@pytest.mark.parametrize(("cards", "want"), [(4, [1, 2, 3]), (1, []), (0, [])])
def test_pin_args_add_the_cards_the_config_does_not_pass(
    tmp_path: Path, cards: int, want: list[int]
) -> None:
    dev = _dev(tmp_path, cards)
    for cfg in (_CUDA, _COMBINED):
        args = _lib('nvidia_pin_args "$1"', str(cfg), NVIDIA_DEV_ROOT=str(dev)).split()
        assert args == [f"--device=/dev/nvidia{i}" for i in want], cfg


def test_pin_args_are_empty_for_the_hip_variant(tmp_path: Path) -> None:
    dev = _dev(tmp_path, 4)
    assert _lib('nvidia_pin_args "$1"', str(_HIP), NVIDIA_DEV_ROOT=str(dev)) == ""


@pytest.mark.parametrize("cfg", [_CUDA, _COMBINED], ids=["cuda", "combined"])
def test_override_is_the_config_plus_the_pins(tmp_path: Path, cfg: Path) -> None:
    out = tmp_path / "override.json"
    pins = ["--device=/dev/nvidia1", "--device=/dev/nvidia2", "--device=/dev/nvidia3"]
    _lib('nvidia_pin_override "$@"', str(cfg), str(out), *pins)
    text = out.read_text()
    # Only the pin lines are new: every comment and line of the config stays.
    added = [ln for ln in text.splitlines() if ln not in cfg.read_text().splitlines()]
    assert [ln.strip() for ln in added] == [f'"{p}",' for p in pins]
    run_args = _jsonc(text)["runArgs"]
    want = _jsonc(cfg.read_text())["runArgs"]
    at = want.index("--device=/dev/nvidia0") + 1
    assert run_args == [*want[:at], *pins, *want[at:]]


def test_override_fails_without_an_nvidia_line(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f'. "{_ROOT}/devtools/lib.sh"; '
            'nvidia_pin_override "$1" "$2" --device=/dev/nvidia1',
            "lib",
            str(_HIP),
            str(tmp_path / "o.json"),
        ],
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0


# --- devcontainer.sh's dry run ----------------------------------------------


def _repo(tmp_path: Path, files: tuple[str, ...]) -> Path:
    """A throwaway git repository holding `files` copied from this tree."""
    repo = tmp_path / "repo"
    for rel in files:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes((_ROOT / rel).read_bytes())
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
    return repo


def _dry_run(tmp_path: Path, cards: int, variant: str) -> str:
    repo = _repo(
        tmp_path,
        (
            "devtools/devcontainer.sh",
            "devtools/lib.sh",
            "devtools/config.sh",
            ".devcontainer/cuda/devcontainer.json",
            ".devcontainer/hip/devcontainer.json",
        ),
    )
    dev = _dev(tmp_path, cards)
    home = tmp_path / "home"
    home.mkdir()
    proc = subprocess.run(
        ["bash", str(repo / "devtools/devcontainer.sh"), f"--{variant}", "rebuild"],
        env=_env(
            DEVCONTAINER_DRY_RUN="1",
            NVIDIA_DEV_ROOT=str(dev),
            HOME=str(home),
            GH_TOKEN="ghp_secret123",
            HOST_ENV_PASSTHROUGH="GH_TOKEN",
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout + proc.stderr


def test_dry_run_pins_every_card(tmp_path: Path) -> None:
    out = _dry_run(tmp_path, 4, "cuda")
    cmd = next(ln for ln in out.splitlines() if ln.startswith("would run:"))
    assert " up " in cmd
    assert "--remove-existing-container" in cmd
    assert "--config " in cmd
    override = re.search(r"--override-config (\S+)", cmd)
    assert override, cmd
    # A temp file, removed when the script exits.
    assert not Path(override.group(1)).exists()
    # The redaction holds on the printed command, too.
    assert "ghp_secret123" not in out
    assert "GH_TOKEN=***REDACTED***" in cmd
    pins = re.findall(r'"--device=/dev/nvidia(\d+)"', out)
    assert pins == ["0", "1", "2", "3"], out
    assert "pinning /dev/nvidia1 /dev/nvidia2 /dev/nvidia3" in out


@pytest.mark.parametrize(("cards", "variant"), [(1, "cuda"), (0, "cuda"), (4, "hip")])
def test_dry_run_adds_nothing_when_nothing_is_missing(
    tmp_path: Path, cards: int, variant: str
) -> None:
    out = _dry_run(tmp_path, cards, variant)
    assert "would run:" in out
    assert "--override-config" not in out
    assert "pinning" not in out


# --- doctor -----------------------------------------------------------------


def _stub(bin_dir: Path, name: str, body: str) -> None:
    (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n")
    (bin_dir / name).chmod(0o755)


def _doctor_gpu(
    tmp_path: Path, dev: Path, inspect: dict[str, tuple[int, str, list[str]]] | None
) -> str:
    """doctor's `gpu devices` section; `inspect` maps a container id to its
    (DeviceRequests count, config_file label, HostConfig.Devices) for a stub
    docker; None, a docker with no container for this worktree."""
    repo = _repo(
        tmp_path, ("devtools/doctor.sh", "devtools/lib.sh", "devtools/config.sh")
    )
    bin_dir = tmp_path / "stub"
    bin_dir.mkdir()
    listing = "\n".join(["NVIDIA RTX A6000, 8.6"] * 4)
    _stub(bin_dir, "nvidia-smi", f"cat <<'EOF'\n{listing}\nEOF")
    # A big /dev/shm, so the shm check stays out of the way.
    _stub(
        bin_dir,
        "df",
        "echo 'Filesystem 1024-blocks Used Available Capacity Mounted on'\n"
        "echo 'shm 1048576 0 1048576 0% /dev/shm'",
    )
    # Always a stub docker, so the machine's own (the host's) is never asked.
    cases = []
    for cid, (requests, label, devices) in (inspect or {}).items():
        lines = "\n".join([str(requests), label, *devices])
        cases.append(f"  {cid}) cat <<'EOF'\n{lines}\nEOF\n  ;;")
    ids = "\n".join(inspect or {})
    _stub(
        bin_dir,
        "docker",
        f'case "$1" in\n'
        f'  ps) case "$*" in *"label=devcontainer.local_folder={repo}"*) '
        f"cat <<'EOF'\n{ids}\nEOF\n ;; esac ;;\n"
        f'  inspect) case "$2" in\n'
        + "\n".join(cases)
        + "\n  *) exit 1 ;;\n  esac ;;\nesac",
    )
    proc = subprocess.run(
        ["bash", str(repo / "devtools" / "doctor.sh")],
        cwd=repo,
        env=_env(
            PATH=f"{bin_dir}:{os.environ['PATH']}",
            DOCTOR_GPU_VENDORS="nvidia",
            NVIDIA_DEV_ROOT=str(dev),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.split("\ngpu devices\n", 1)[1].split("\nrequired paths\n", 1)[0]


def _warns(section: str) -> list[str]:
    return [ln for ln in section.splitlines() if "[warn]" in ln]


def test_doctor_ok_when_every_card_opens(tmp_path: Path) -> None:
    section = _doctor_gpu(tmp_path, _dev(tmp_path, 4), None)
    assert _warns(section) == [], section
    assert "nvidia: all 4 card node(s) open read-write" in section, section


@pytest.mark.skipif(os.geteuid() == 0, reason="root opens a mode-000 file")
def test_doctor_names_a_card_that_does_not_open(tmp_path: Path) -> None:
    dev = _dev(tmp_path, 4)
    (dev / "nvidia2").chmod(0)
    section = _doctor_gpu(tmp_path, dev, None)
    warns = _warns(section)
    assert len(warns) == 1, section
    assert "/dev/nvidia2 present but cannot be opened read-write" in warns[0]
    assert "devtools/devcontainer.sh rebuild" in section


_CUDA_LABEL = "/x/.devcontainer/cuda/devcontainer.json"
_PINNED_ONE = ["/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm"]


def test_doctor_names_cards_a_container_does_not_pin(tmp_path: Path) -> None:
    section = _doctor_gpu(
        tmp_path, _dev(tmp_path, 4), {"0123456789abcdef": (1, _CUDA_LABEL, _PINNED_ONE)}
    )
    warns = _warns(section)
    assert len(warns) == 1, section
    assert (
        "container cuda (0123456789ab) does not pin /dev/nvidia1 /dev/nvidia2 "
        "/dev/nvidia3 -- dropped on the next cgroup regeneration" in warns[0]
    ), section
    assert "devtools/devcontainer.sh --cuda rebuild" in section, section


def test_doctor_ok_for_a_container_that_pins_every_card(tmp_path: Path) -> None:
    devices = [*_PINNED_ONE, "/dev/nvidia1", "/dev/nvidia2", "/dev/nvidia3"]
    hip = (0, "/x/.devcontainer/hip/devcontainer.json", ["/dev/kfd", "/dev/dri"])
    section = _doctor_gpu(
        tmp_path,
        _dev(tmp_path, 4),
        {"aaaaaaaaaaaaaaaa": (1, _CUDA_LABEL, devices), "bbbbbbbbbbbbbbbb": hip},
    )
    assert _warns(section) == [], section
    assert "container cuda (aaaaaaaaaaaa) pins all 4 card(s)" in section, section
    # The hip container has no NVIDIA plumbing, so nothing is said about it.
    assert "bbbbbbbbbbbb" not in section, section


def test_doctor_catches_a_container_made_with_gpus_all_alone(tmp_path: Path) -> None:
    # Before #190: --gpus all, no --device= at all. Still an NVIDIA container.
    section = _doctor_gpu(
        tmp_path, _dev(tmp_path, 2), {"cccccccccccccccc": (1, "", [])}
    )
    warns = _warns(section)
    assert len(warns) == 1, section
    assert (
        "container ? (cccccccccccc) does not pin /dev/nvidia0 /dev/nvidia1" in warns[0]
    )
    assert "fix: devtools/devcontainer.sh rebuild" in section, section
