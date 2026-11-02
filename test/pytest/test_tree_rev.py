"""devtools/lib.sh tree_rev: a tier's rev= says when the tree was dirty (#694).

A tier run before the commit used to stamp the base commit's sha on code that
was not in it. These cases build a throwaway repository with a submodule and
check the stamp each kind of change produces -- no build, no GPU.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_LIB = Path(__file__).resolve().parents[2] / "devtools" / "lib.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("sha256sum") is None,
    reason="needs git and sha256sum",
)

_ENV = dict(
    os.environ,
    GIT_AUTHOR_NAME="t",
    GIT_AUTHOR_EMAIL="t@example.invalid",
    GIT_COMMITTER_NAME="t",
    GIT_COMMITTER_EMAIL="t@example.invalid",
    GIT_CONFIG_GLOBAL=os.devnull,
    GIT_CONFIG_NOSYSTEM="1",
)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "protocol.file.allow=always", *args],
        cwd=cwd,
        env=_ENV,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    sub = tmp_path / "sub"
    sub.mkdir()
    git(sub, "init", "-q")
    (sub / "s.txt").write_text("sub\n")
    git(sub, "add", ".")
    git(sub, "commit", "-qm", "sub")

    top = tmp_path / "top"
    top.mkdir()
    git(top, "init", "-q")
    (top / "a.txt").write_text("a\n")
    git(top, "submodule", "add", "-q", str(sub), "deps/sub")
    git(top, "add", ".")
    git(top, "commit", "-qm", "top")
    return top


def stamp(root: Path, log: Path | None = None) -> tuple[str, str, str]:
    """(TREE_REV, TREE_DIFF_HASH, tree_rev_header's stdout)."""
    log = log or root.parent / "run.log"
    script = (
        f'. "{_LIB}"; tree_rev "{root}"; '
        f'printf "%s\\n%s\\n" "$TREE_REV" "$TREE_DIFF_HASH"; '
        f'tree_rev_header tier "{log}"'
    )
    out = subprocess.run(
        ["bash", "-c", f"set -euo pipefail; {script}"],
        env=_ENV,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rev, diff, header = out.split("\n", 2)
    return rev, diff, header


def test_clean_tree_is_the_bare_sha(repo: Path) -> None:
    log = repo.parent / "run.log"
    rev, diff, header = stamp(repo, log)
    assert rev == git(repo, "rev-parse", "--short", "HEAD")
    assert diff == ""
    assert header == ""
    assert not log.exists()


def test_tracked_edit_marks_and_hashes(repo: Path) -> None:
    sha = git(repo, "rev-parse", "--short", "HEAD")
    (repo / "a.txt").write_text("a changed\n")
    log = repo.parent / "run.log"
    rev, diff, header = stamp(repo, log)
    assert rev == f"{sha}+dirty"
    assert len(diff) == 12
    assert f"rev={sha}+dirty  diff={diff}" in header
    assert "a.txt" in header
    assert log.read_text() == header

    # A different edit is a different hash; the same edit, the same one.
    _, again, _ = stamp(repo)
    assert again == diff
    (repo / "a.txt").write_text("a changed twice\n")
    _, other, _ = stamp(repo)
    assert other != diff


def test_staged_edit_counts(repo: Path) -> None:
    (repo / "a.txt").write_text("staged\n")
    git(repo, "add", "a.txt")
    rev, diff, _ = stamp(repo)
    assert rev.endswith("+dirty")
    assert diff


def test_untracked_file_does_not_mark_but_is_listed(repo: Path) -> None:
    (repo / "scratch.txt").write_text("x\n")
    rev, diff, _ = stamp(repo)
    assert not rev.endswith("+dirty")
    assert diff == ""

    # Beside a tracked change, it is listed and left out of the hash.
    (repo / "a.txt").write_text("a changed\n")
    _, with_scratch, header = stamp(repo)
    assert "1 untracked file(s), not in the hash:" in header
    assert "scratch.txt" in header
    (repo / "scratch.txt").unlink()
    _, without, _ = stamp(repo)
    assert with_scratch == without


def test_submodule_content_edit_counts(repo: Path) -> None:
    (repo / "deps/sub/s.txt").write_text("sub changed\n")
    rev, first, header = stamp(repo)
    assert rev.endswith("+dirty")
    assert "deps/sub" in header
    # The hash covers the submodule's content, not only that it is modified.
    (repo / "deps/sub/s.txt").write_text("sub changed again\n")
    _, second, _ = stamp(repo)
    assert first != second


def test_submodule_pointer_move_counts(repo: Path) -> None:
    sub = repo / "deps/sub"
    (sub / "s.txt").write_text("new commit\n")
    git(sub, "commit", "-qam", "move")
    rev, diff, _ = stamp(repo)
    assert rev.endswith("+dirty")
    assert diff


def test_submodule_untracked_file_does_not_mark(repo: Path) -> None:
    (repo / "deps/sub/scratch.txt").write_text("x\n")
    rev, diff, _ = stamp(repo)
    assert not rev.endswith("+dirty")
    assert diff == ""


def test_not_a_repository_is_unknown(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    env = dict(_ENV, GIT_CEILING_DIRECTORIES=str(tmp_path))
    out = subprocess.run(
        ["bash", "-c", f'. "{_LIB}"; tree_rev "{plain}"; echo "$TREE_REV"'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "unknown"
