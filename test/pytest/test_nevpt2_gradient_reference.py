"""The committed DF-NEVPT2 numerical-gradient references,
test/gtest/data/nevpt2_gradient/.

Their invariants are checked with the standard library alone, so they run on
the push gate: the cases the issue asks for are there, every central
difference and Richardson estimate is the one its recorded displaced energies
give, the Richardson estimate and the finest central difference agree well
inside the consumers' 1e-6 hartree/bohr, every gradient sums to ~0 over
atoms, and every displacement's guard values clear the bounds the file
states. Re-running points through tools/emit/nevpt2_gradient_reference.py
needs PySCF and block2, in the opt-in ``tools`` dependency group, so that test
is slow and skips (with a reason) wherever the group is not installed::

    uv run --group tools pytest -m slow -rs \\
        test/pytest/test_nevpt2_gradient_reference.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "test" / "gtest" / "data" / "nevpt2_gradient" / "nevpt2_gradient.json"
QUANTITIES = ("casscf", "sc", "pc")
# The gate the consumers (#563) hold the analytic gradient to.
CONSUMER_GATE = 1e-6


def document() -> dict:
    return json.loads(MANIFEST.read_text())


def cases() -> list[dict]:
    return document()["cases"]


def test_the_issue_cases_are_covered():
    by_name = {c["name"]: c for c in cases()}
    assert set(by_name) == {"h2o_skew_ccpvdz", "h2co_ccpvdz"}
    for case in cases():
        assert case["weights"] == [1.0]
        assert case["aux_basis"]["file"] == "data/aux_basis/cc-pVDZ-JKFIT.json"
        assert (ROOT / case["aux_basis"]["file"]).is_file()
        assert 2 * case["ncore"] + case["nelecas"] == case["electrons"]
        assert len(case["active"]) == case["ncas"]
    # One case is larger than water, and none is N2 (degenerate pi*).
    assert len(by_name["h2co_ccpvdz"]["atoms"]) == 4
    assert all(a["symbol"] in {"H", "C", "O"} for c in cases() for a in c["atoms"])


def test_the_differences_are_the_recorded_energies():
    for case in cases():
        steps = case["steps"]
        assert len(steps) == 3
        assert steps[1] == steps[0] / 2 and steps[2] == steps[1] / 2
        natm = len(case["atoms"])
        displaced = case["displaced"]
        assert [(d["atom"], d["axis"]) for d in displaced] == [
            (a, k) for a in range(natm) for k in range(3)
        ]
        for q in QUANTITIES:
            central = case["gradients"][q]["central"]
            richardson = case["gradients"][q]["richardson"]
            assert len(central) == len(steps)
            for d in displaced:
                a, k = d["atom"], d["axis"]
                for i, h in enumerate(steps):
                    minus, plus = d[q][i]
                    assert central[i][a][k] == (plus - minus) / (2 * h), (q, a, k, i)
                fine, coarse = central[-1][a][k], central[-2][a][k]
                assert richardson[a][k] == (4 * fine - coarse) / 3, (q, a, k)
                coarser = (4 * coarse - central[-3][a][k]) / 3
                error_bar = case["gradients"][q]["error_bar"][a][k]
                assert error_bar == abs(richardson[a][k] - coarser), (q, a, k)


def test_richardson_agrees_and_translation_vanishes():
    tolerances = document()["tolerances"]
    assert tolerances["richardson"] <= CONSUMER_GATE / 5
    for case in cases():
        for q in QUANTITIES:
            what = (case["name"], q)
            richardson = case["gradients"][q]["richardson"]
            finest = case["gradients"][q]["central"][-1]
            miss = max(
                abs(r - f)
                for rr, fr in zip(richardson, finest, strict=True)
                for r, f in zip(rr, fr, strict=True)
            )
            assert miss < tolerances["richardson"], what
            assert miss == case["checks"][q]["richardson_minus_finest"], what
            for k in range(3):
                total = sum(row[k] for row in richardson)
                assert abs(total) < tolerances["translation"], what
        assert case["checks"]["casscf"]["analytic"] < tolerances["analytic"]


def test_every_displacement_passed_the_guard():
    guard = document()["guard"]
    for case in cases():
        worst = case["guard"]
        assert worst["prediction"] < guard["prediction_bound"], case["name"]
        assert worst["subspace"] < guard["subspace_tol"], case["name"]
        assert worst["ci"] < guard["ci_tol"], case["name"]
        assert worst["inactive_gap"] > guard["degenerate"], case["name"]
        # The reference point: nevpt2_reference.py's checks.
        point = case["checks"]["reference_point"]
        assert point["sc_invariance"] < 1e-10
        assert point["block2_sc"] < 1e-10
        assert point["pc_invariance"] < 1e-10
        # The totals are the CASSCF energy plus each correction.
        e = case["energies"]
        assert abs(e["sc"] - e["casscf"] - case["e_corr"]["sc"]) < 1e-12
        assert abs(e["pc"] - e["casscf"] - case["e_corr"]["pc"]) < 1e-12
        assert (
            abs(sum(c["energy"] for c in case["classes"]) - case["e_corr"]["sc"])
            < 1e-12
        )


@pytest.mark.slow
def test_points_rerun():
    # before ANY third-party import: numpy is in the `tools` group too
    pytest.importorskip("pyscf", reason="needs the `tools` dependency group")
    pytest.importorskip("pyblock2", reason="needs the `tools` dependency group")
    pytest.importorskip("numpy", reason="needs the `tools` dependency group")
    sys.path.insert(0, str(ROOT / "tools" / "emit"))
    import nevpt2_gradient_reference as emit
    import numpy

    (spec,) = [c for c in emit.CASES if c["name"] == "h2o_skew_ccpvdz"]
    (want,) = [c for c in cases() if c["name"] == "h2o_skew_ccpvdz"]
    ref = emit.Case(spec)
    assert abs(ref.energy - want["energies"]["casscf"]) < 1e-11
    assert abs(ref.nevpt2["sc"] - want["e_corr"]["sc"]) < 1e-11
    assert abs(ref.nevpt2["pc"] - want["e_corr"]["pc"]) < 1e-11
    # One displaced point, the finest step's +h along H1's x.
    entry = want["displaced"][3]
    assert (entry["atom"], entry["axis"]) == (1, 0)
    displacement = numpy.zeros((3, 3))
    displacement[1, 0] = want["steps"][-1]
    got = ref.point(displacement)
    for q in emit.QUANTITIES:
        assert abs(got[q] - entry[q][-1][1]) < 1e-11, q
