"""The one Python test the template ships: it proves the harness is wired up.

There are two suites in this repo and they answer different questions:

  test/gtest/   C++ unit tests, linked against the whole module graph, run by
                ctest. They fail at BUILD time when the graph breaks, which is
                most of their value.
  test/pytest/  this suite -- Python-level checks over the built binaries:
                validation against a reference implementation, parameter
                sweeps, anything easier to express in numpy than in C++.

Only this one is on the push gate, because it is the one that costs seconds.
Delete this file once the project has real tests -- but keep one case in each
tier, or the fast/slow split is asserted rather than exercised.
"""

from __future__ import annotations

import shutil

import pytest


def test_suite_is_discovered():
    """A green run on an empty suite proves nothing; this is the floor."""
    assert True


@pytest.mark.slow
def test_slow_tier_is_wired():
    """Marked slow, so the fast/slow split is exercised rather than assumed.

    devtools/prepush-tests.sh and `devcontainer.sh test` deselect this one;
    devtools/slow-tier.sh runs exactly this one and nothing else. If both hold,
    the marker and FAST_TEST_ARGS/SLOW_TEST_ARGS agree. A bare `pytest` runs
    everything -- the marker filter lives in devtools/config.sh, not in
    pyproject's addopts, so an explicit `pytest -m slow` still works.
    """
    assert True


@pytest.mark.gpu
@pytest.mark.skipif(
    shutil.which("nvidia-smi") is None and shutil.which("rocm-smi") is None,
    reason="no GPU driver (neither nvidia-smi nor rocm-smi)",
)
def test_gpu_marker_is_registered():
    """Skips loudly rather than silently on a box with no device.

    The `-rs` that devtools puts on every invocation is what makes the skip
    visible; without it this file looks the same on a GPU box and a laptop.
    `gpu` is a sub-category of the tier split, not a replacement for it: mark a
    GPU test `slow` as well if it is also expensive.
    """
    assert True
