"""tools/emit/nevpt2_golden.py: the committed SC-NEVPT2 goldens round-trip.

On the smallest case, the dm1, dm2, dm3, f3ca and f3ac PySCF builds from the
committed CI vector and active block are the committed ones, and the Sr and
Si classes rebuilt from them and the committed integrals are the demo's
answer -- the check the script makes before it writes.

PySCF lives in the opt-in ``tools`` dependency group, not ``dev``, so this
module skips (with a reason) wherever that group is not installed -- the push
gate among them. Run it with the group::

    uv run --group tools pytest -rs test/pytest/test_nevpt2_golden.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# before ANY third-party import: numpy is in the `tools` group too, not `dev`
pytest.importorskip("pyscf", reason="needs the `tools` dependency group")
pytest.importorskip("numpy", reason="needs the `tools` dependency group")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tools.emit import nevpt2_golden  # noqa: E402

CASE = ROOT / "test" / "gtest" / "data" / "nevpt2" / "n2_ccpvdz_cas44_df"
# N2 CAS(4,4): two alpha and two beta electrons in four orbitals.
NELEC = (2, 2)


def test_rdms_and_sr_si_round_trip():
    a = {p.stem: np.load(p) for p in CASE.glob("*.npy")}
    assert set(a) == set(nevpt2_golden.COPIED) | {
        "dm3", "f3ca", "f3ac", "e_nevpt2_total"
    }  # fmt: skip

    rdms = nevpt2_golden.pyscf_rdms(a["ci"], a["h2e"], NELEC)
    for name, got in rdms.items():
        assert np.abs(got - a[name]).max() <= 1e-14, name

    assert nevpt2_golden.check(a, rdms, NELEC) <= nevpt2_golden.CHECK_TOL
    # The total is the eight classes' sum.
    assert abs(a["class_energies"].sum() - a["e_nevpt2_total"]) <= 1e-12
