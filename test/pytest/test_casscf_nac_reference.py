"""The committed DF SA-CASSCF coupling references, test/gtest/data/casscf_nac/.

Their invariants are checked with the standard library alone, so they run on
the push gate: the cases the issue asks for are there (the two-root twin of
casscf_gradient/'s and a three-root one), every ordered pair is written once,
d_IJ = -d_JI and h_IJ = h_JI, the pieces add up (h = Hellmann-Feynman +
Lagrange, d = etf + csf, d = h / (E_J - E_I) + csf), the CSF-free coupling
sums to ~0 over atoms, and the RDMs and multipliers are laid out as the README
says. Re-running a case through tools/emit/casscf_nac_reference.py needs
PySCF, in the opt-in ``tools`` dependency group, and minutes of CASSCFs, so
that test is slow and skips (with a reason) wherever the group is not
installed::

    uv run --group tools pytest -m slow -rs \\
        test/pytest/test_casscf_nac_reference.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "test" / "gtest" / "data" / "casscf_nac" / "casscf_nac.json"
VECTORS = ("total", "etf", "csf", "h", "hellmann_feynman", "lagrange")


def cases() -> list[dict]:
    return json.loads(MANIFEST.read_text())["cases"]


def combine(a: list, b: list, scale: float = 1.0) -> list:
    return [
        [x + scale * y for x, y in zip(ra, rb, strict=True)]
        for ra, rb in zip(a, b, strict=True)
    ]


def assert_close(got: list, want: list, tol: float, what: str) -> None:
    for g, w in zip(got, want, strict=True):
        for x, y in zip(g, w, strict=True):
            assert abs(x - y) < tol, (what, x, y)


def by_pair(case: dict) -> dict:
    return {(c["bra"], c["ket"]): c for c in case["couplings"]}


def test_the_issue_cases_are_covered():
    by_name = {c["name"]: c for c in cases()}
    assert set(by_name) == {"h2co_ccpvdz_sa2", "h2co_skew_ccpvdz_sa3"}
    assert by_name["h2co_ccpvdz_sa2"]["from"] == "h2co_ccpvdz_sa2"
    assert len(by_name["h2co_skew_ccpvdz_sa3"]["weights"]) == 3
    for case in cases():
        assert case["aux_basis"]["file"] == "data/aux_basis/cc-pVDZ-JKFIT.json"
        assert (ROOT / case["aux_basis"]["file"]).is_file()
        n = len(case["weights"])
        assert len(set(case["weights"])) == 1
        assert len(case["energies"]) == len(case["ci"]) == n
        pairs = sorted(by_pair(case))
        assert pairs == [(i, j) for i in range(n) for j in range(n) if i != j]


def test_couplings_are_antisymmetric_and_add_up():
    for case in cases():
        natm = len(case["atoms"])
        pairs = by_pair(case)
        for (i, j), c in pairs.items():
            what = f"{case['name']}({i},{j})"
            for key in VECTORS:
                assert len(c[key]) == natm, what
                assert all(len(r) == 3 for r in c[key]), what
            gap = case["energies"][j] - case["energies"][i]
            assert abs(c["energy_gap"] - gap) < 1e-12, what
            t = pairs[(j, i)]
            assert_close(c["total"], [[-x for x in r] for r in t["total"]], 1e-8, what)
            assert_close(c["h"], t["h"], 1e-8, what)
            hf_plus_l = combine(c["hellmann_feynman"], c["lagrange"])
            assert_close(hf_plus_l, c["h"], 1e-10, what)
            assert_close(combine(c["etf"], c["csf"]), c["total"], 1e-10, what)
            ratio = [[x / gap for x in r] for r in c["h"]]
            assert_close(combine(ratio, c["csf"]), c["total"], 1e-8, what)
            for k in range(3):
                assert abs(sum(r[k] for r in c["etf"])) < 1e-8, what


def test_the_layout_matches_the_readme():
    for case in cases():
        ncore, ncas, nmo = case["ncore"], case["ncas"], case["nmo"]
        n = case["nelecas"]
        shape = (math.comb(ncas, n - n // 2), math.comb(ncas, n // 2))
        assert 2 * ncore + n == case["electrons"]
        for vector in case["ci"]:
            assert (len(vector), len(vector[0])) == shape
            assert abs(sum(x * x for row in vector for x in row) - 1) < 1e-10
            flat = [x for row in vector for x in row]
            top = max(abs(x) for x in flat)
            assert next(x for x in flat if abs(x) >= top - 1e-8) > 0
        nocc = ncore + ncas
        expected = [(p, q) for p in range(ncore, nocc) for q in range(ncore)]
        expected += [(p, q) for p in range(nocc, nmo) for q in range(nocc)]
        assert [tuple(pq) for pq in case["pairs"]] == expected
        for c in case["couplings"]:
            rdm = c["rdm"]
            assert len(rdm["trans_1"]) == ncas and len(rdm["sym_1"]) == ncas
            assert len(rdm["trans_2"]) == len(rdm["sym_2"]) == ncas**4
            for p in range(ncas):
                for q in range(ncas):
                    sym = 0.5 * (rdm["trans_1"][p][q] + rdm["trans_1"][q][p])
                    assert abs(rdm["sym_1"][p][q] - sym) < 1e-14
            assert len(c["b"]["orbital"]) == len(case["pairs"])
            assert len(c["z"]["orbital"]) == len(case["pairs"])
            assert len(c["z"]["ci"]) == len(case["ci"])
            assert c["z"]["residual"] < 1e-9
            assert c["b"]["ci_norm"] < 1e-8
            for zc in c["z"]["ci"]:
                assert (len(zc), len(zc[0])) == shape
        checks = case["checks"]
        assert checks["finite_difference"] < 1e-6
        ladder = checks["aux_ladder"]
        assert all(b < a for a, b in zip(ladder, ladder[1:], strict=False))


@pytest.mark.slow
def test_a_case_reruns():
    # before ANY third-party import: numpy is in the `tools` group too
    pytest.importorskip("pyscf", reason="needs the `tools` dependency group")
    pytest.importorskip("numpy", reason="needs the `tools` dependency group")
    sys.path.insert(0, str(ROOT / "tools" / "emit"))
    import casscf_nac_reference as emit

    (case,) = [c for c in emit.CASES if c["name"] == "h2co_ccpvdz_sa2"]
    got = emit.case_reference(case)
    (want,) = [c for c in cases() if c["name"] == "h2co_ccpvdz_sa2"]
    for g, w in zip(got["couplings"], want["couplings"], strict=True):
        for key in VECTORS:
            assert_close(g[key], w[key], 1e-9, key)
