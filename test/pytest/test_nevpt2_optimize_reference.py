"""The committed SC-NEVPT2 optimization reference,
test/gtest/data/nevpt2_gradient/nevpt2_optimize.json.

Its invariants are checked with the standard library alone, so they run on
the push gate: the case is the start nevpt2_gradient.json's skew water
optimizes from, the total is the CASSCF energy plus SC's correction, every
central difference and the Richardson estimate are the ones the recorded
displaced energies give, and that gradient is under the optimizer threshold
the file states. Re-running the single point through
tools/emit/nevpt2_optimize_reference.py needs PySCF and block2, in the
opt-in ``tools`` dependency group, so that test is slow and skips (with a
reason) wherever the group is not installed::

    uv run --group tools pytest -m slow -rs \\
        test/pytest/test_nevpt2_optimize_reference.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "test" / "gtest" / "data" / "nevpt2_gradient"
MANIFEST = DATA / "nevpt2_optimize.json"
# test/gtest/pipeline_optimize_gpu.cpp's kMaxForce, the [optimize] max_force.
MAX_FORCE = 1.5e-5


def document() -> dict:
    return json.loads(MANIFEST.read_text())


def test_the_case_starts_at_the_gradient_reference():
    doc = document()
    (start,) = [
        c
        for c in json.loads((DATA / "nevpt2_gradient.json").read_text())["cases"]
        if c["name"] == "h2o_skew_ccpvdz"
    ]
    assert doc["name"] == start["name"]
    assert doc["start"] == start["atoms"]
    for key in ("basis", "ncas", "nelecas", "active", "ncore", "charge"):
        assert doc[key] == start[key], key
    assert doc["aux_basis"] == start["aux_basis"]
    assert [a["symbol"] for a in doc["atoms"]] == [a["symbol"] for a in doc["start"]]
    # The optimum is not the start.
    assert doc["atoms"] != doc["start"]


def test_the_total_is_the_casscf_energy_plus_sc():
    e = document()["energies"]
    assert abs(e["sc"] - e["casscf"] - e["e_corr"]) < 1e-12


def test_the_gradient_is_the_recorded_energies_and_under_the_threshold():
    doc = document()
    steps = doc["steps"]
    assert len(steps) == 2 and steps[1] == steps[0] / 2
    natm = len(doc["atoms"])
    displaced = doc["displaced"]
    assert [(d["atom"], d["axis"]) for d in displaced] == [
        (a, k) for a in range(natm) for k in range(3)
    ]
    central = doc["gradient"]["central"]
    richardson = doc["gradient"]["richardson"]
    for d in displaced:
        a, k = d["atom"], d["axis"]
        for i, h in enumerate(steps):
            minus, plus = d["sc"][i]
            assert central[i][a][k] == (plus - minus) / (2 * h), (a, k, i)
        assert richardson[a][k] == (4 * central[1][a][k] - central[0][a][k]) / 3
    worst = max(abs(g) for row in richardson for g in row)
    assert worst == doc["max_gradient"]
    assert doc["force_threshold"] == MAX_FORCE
    assert worst < MAX_FORCE


@pytest.mark.slow
def test_the_single_point_reruns():
    # before ANY third-party import: numpy is in the `tools` group too
    pytest.importorskip("pyscf", reason="needs the `tools` dependency group")
    pytest.importorskip("pyblock2", reason="needs the `tools` dependency group")
    pytest.importorskip("numpy", reason="needs the `tools` dependency group")
    sys.path.insert(0, str(ROOT / "tools" / "emit"))
    import nevpt2_optimize_reference as emit
    from casci_reference import aux_basis
    from casscf_reference import casscf, molecule
    from pyscf import lib

    lib.num_threads(1)
    want = document()
    assert [list(r) for _, r in emit.OPTIMUM] == [a["coord"] for a in want["atoms"]]
    case = emit.CASE
    nelecas, ncas = case["cas"]
    mol = molecule(case)
    auxbasis, _ = aux_basis(case, mol)
    mf = emit.run_rhf(case["name"], mol)
    start = casscf(mf, ncas, nelecas, [1.0]).sort_mo(case["active"], base=0)
    _, mo, e_casscf, _ = emit.polished(case["name"], mf, start, ncas, nelecas, auxbasis)
    e_corr = emit.sc_energy(case["name"], mf, auxbasis, mo, ncas, nelecas, e_casscf)
    assert abs(e_casscf - want["energies"]["casscf"]) < 1e-11
    assert abs(e_corr - want["energies"]["e_corr"]) < 1e-11
