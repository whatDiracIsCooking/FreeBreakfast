"""The committed DF-CASSCF gradient references, test/gtest/data/casscf_gradient/.

Their invariants are checked with the standard library alone, so they run on
the push gate: the cases the issue asks for are there (one with a core, one
two-root state average), every gradient is n_atom x 3 and sums to ~0 over
atoms, each root's parts (and an SA root's responses) add up to its total,
and the multipliers are laid out as the README says. Re-running a case
through tools/emit/casscf_gradient_reference.py needs PySCF, in the opt-in
``tools`` dependency group, and minutes of CASSCFs, so that test is slow and
skips (with a reason) wherever the group is not installed::

    uv run --group tools pytest -m slow -rs \\
        test/pytest/test_casscf_gradient_reference.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "test" / "gtest" / "data" / "casscf_gradient" / "casscf_gradient.json"
PARTS = {"kinetic", "potential", "overlap", "two_electron", "nuclear_repulsion"}


def cases() -> list[dict]:
    return json.loads(MANIFEST.read_text())["cases"]


def add(*gs: list) -> list:
    return [[sum(c) for c in zip(*rows, strict=True)] for rows in zip(*gs, strict=True)]


def assert_close(got: list, want: list, tol: float, what: str) -> None:
    for g, w in zip(got, want, strict=True):
        for x, y in zip(g, w, strict=True):
            assert abs(x - y) < tol, (what, x, y)


def strings(n: int, k: int) -> int:
    return math.comb(n, k)


def test_the_issue_cases_are_covered():
    by_name = {c["name"]: c for c in cases()}
    assert set(by_name) == {"h2o_ccpvdz", "h2o_skew_ccpvdz", "h2co_ccpvdz_sa2"}
    assert all(c["ncore"] > 0 for c in cases())
    assert by_name["h2co_ccpvdz_sa2"]["weights"] == [0.5, 0.5]
    for case in cases():
        assert case["aux_basis"]["file"] == "data/aux_basis/cc-pVDZ-JKFIT.json"
        assert (ROOT / case["aux_basis"]["file"]).is_file()
        assert len(case["roots"]) == len(case["weights"]) == len(case["energies"])


def test_gradients_are_translation_invariant_and_add_up():
    for case in cases():
        natm = len(case["atoms"])
        for root in case["roots"]:
            what = f"{case['name']}[{root['root']}]"
            total = root["total"]
            assert len(total) == natm and all(len(r) == 3 for r in total), what
            for k in range(3):
                assert abs(sum(row[k] for row in total)) < 1e-8, what
            assert set(root["parts"]) == PARTS, what
            assert_close(add(*root["parts"].values()), total, 1e-10, what)
            if len(case["weights"]) > 1:
                assert set(root["responses"]) == {"hamiltonian", "orbital", "ci"}
                assert_close(add(*root["responses"].values()), total, 1e-10, what)
        # the nuclear repulsion does not depend on the root
        first = case["roots"][0]["parts"]["nuclear_repulsion"]
        for root in case["roots"]:
            assert root["parts"]["nuclear_repulsion"] == first


def test_the_layout_matches_the_readme():
    for case in cases():
        ncore, ncas, nmo = case["ncore"], case["ncas"], case["nmo"]
        n = case["nelecas"]
        shape = (strings(ncas, n - n // 2), strings(ncas, n // 2))
        assert len(case["orbitals"]["core_energies"]) == ncore
        assert len(case["orbitals"]["natural_occupations"]) == ncas
        assert len(case["orbitals"]["virtual_energies"]) == nmo - ncore - ncas
        assert 2 * ncore + n == case["electrons"]
        for vector in case["ci"]:
            assert (len(vector), len(vector[0])) == shape
            norm = sum(x * x for row in vector for x in row)
            assert abs(norm - 1) < 1e-10
        for key, value in case["checks"].items():
            assert value < 1e-7, (case["name"], key)
        if len(case["weights"]) == 1:
            assert "pairs" not in case
            assert all("z" not in root for root in case["roots"])
            continue
        nocc = ncore + ncas
        expected = [(p, q) for p in range(ncore, nocc) for q in range(ncore)]
        expected += [(p, q) for p in range(nocc, nmo) for q in range(nocc)]
        assert [tuple(pq) for pq in case["pairs"]] == expected
        for root in case["roots"]:
            z = root["z"]
            assert len(z["orbital"]) == len(case["pairs"])
            assert len(z["ci"]) == len(case["ci"])
            assert z["residual"] < 1e-10
            for zc in z["ci"]:
                assert (len(zc), len(zc[0])) == shape
                for c in case["ci"]:
                    dot = sum(
                        a * b
                        for zr, cr in zip(zc, c, strict=True)
                        for a, b in zip(zr, cr, strict=True)
                    )
                    assert abs(dot) < 1e-10


@pytest.mark.slow
def test_a_case_reruns():
    # before ANY third-party import: numpy is in the `tools` group too
    pytest.importorskip("pyscf", reason="needs the `tools` dependency group")
    pytest.importorskip("numpy", reason="needs the `tools` dependency group")
    sys.path.insert(0, str(ROOT / "tools" / "emit"))
    import casscf_gradient_reference as emit

    (case,) = [c for c in emit.CASES if c["name"] == "h2o_ccpvdz"]
    got = emit.case_reference(case)
    (want,) = [c for c in cases() if c["name"] == "h2o_ccpvdz"]
    for g, w in zip(got["roots"], want["roots"], strict=True):
        assert_close(g["total"], w["total"], 1e-10, "total")
        for key in PARTS:
            assert_close(g["parts"][key], w["parts"][key], 1e-10, key)
