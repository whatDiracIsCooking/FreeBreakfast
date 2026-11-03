"""The brute-force QD-NEVPT2 coupling references,
test/gtest/data/qd_nevpt2_nac/brute_force/ (#744).

Their invariants are checked with the standard library alone, so they run on
the push gate: the three brute-force cases are there with the directions
qd_nevpt2_gradient/ draws, every central difference and Richardson value is
the one its recorded overlaps give, total = zeroth + pt2, every error bar is
under the stated bar, the self-checks and the guard passed, every separately
coded term is visible (#744's acceptance), and each PT2 transition density is
a '<f8' Fortran-order (n, n, n_mo, n_mo) array. Re-running the fixture part
and one displaced point through tools/emit/qd_nevpt2_nac_reference.py needs
PySCF, in the opt-in ``tools`` dependency group, so that test is slow and
skips (with a reason) wherever the group is not installed::

    uv run --group tools pytest -m slow -rs \\
        test/pytest/test_qd_nevpt2_nac_reference.py
"""

from __future__ import annotations

import ast
import functools
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "test" / "gtest" / "data"
OUT = DATA / "qd_nevpt2_nac" / "brute_force"
CASES = ("h6_631g_sa2", "h8_sto3g_sa3", "h8_ring_sto3g_sa3")
SCHEMES = ("sc", "pc")
TERMS = ("zeroth", "pt2", "total")
# The design's gate for a numerical coupling (1/bohr): the error bars must
# sit well inside it.
CONSUMER_GATE = 1e-6


@functools.cache
def case(name: str) -> dict:
    return json.loads((OUT / f"{name}.json").read_text())


def families(record: dict) -> list[tuple[str, dict, list]]:
    out = [("directional", record["directional"], record["displaced"]["directional"])]
    if record["full"] is not None:
        out.append(("full", record["full"], record["displaced"]["full"]))
    return out


def test_the_issue_cases_are_covered():
    assert sorted(p.stem for p in OUT.glob("*.json")) == sorted(CASES)
    for name in CASES:
        record = case(name)
        gradient = json.loads(
            (DATA / "qd_nevpt2_gradient" / f"{name}.json").read_text()
        )
        assert record["directions"] == gradient["directions"], name
        assert record["seed"] == gradient["seed"] == 696
        n = len(record["weights"])
        assert record["pairs"] == [[p, q] for p in range(n) for q in range(p + 1, n)]
        assert len(record["directional"]["sc"]["total"]) == 3
        natm = len(record["atoms"])
        if name == "h8_ring_sto3g_sa3":
            assert len(record["full"]["pc"]["pt2"]) == 3 * natm
        else:
            assert record["full"] is None


def test_the_differences_are_the_recorded_overlaps():
    for name in CASES:
        record = case(name)
        steps = record["steps"]
        assert steps[1] == steps[0] / 2 and steps[2] == steps[1] / 2
        index = record["pairs"]
        for _, family, raw in families(record):
            for s in SCHEMES:
                noise = record["noise"][s]
                for row, point in enumerate(raw):
                    z, y = point[s]["z"], point[s]["y"]
                    for k, (p, q) in enumerate(index):
                        zeroth = [[m[p][q] for m in pair] for pair in z]
                        pt2 = [[0.5 * (m[p][q] - m[q][p]) for m in pair] for pair in y]
                        for term, values in (("zeroth", zeroth), ("pt2", pt2)):
                            got = family[s][term][row]
                            central = [
                                (plus - minus) / (2 * h)
                                for (minus, plus), h in zip(values, steps, strict=True)
                            ]
                            for i in range(3):
                                assert abs(
                                    got["central"][i][k] - central[i]
                                ) <= 1e-12 * (1 + abs(central[i]))
                            fine = (4 * central[2] - central[1]) / 3
                            coarse = (4 * central[1] - central[0]) / 3
                            assert abs(got["richardson"][k] - fine) <= 1e-12 * (
                                1 + abs(fine)
                            )
                            bar = abs(fine - coarse) + noise[term][k] / (2 * steps[2])
                            assert abs(got["error_bar"][k] - bar) <= 1e-12 * bar
                        total = family[s]["total"][row]["richardson"][k]
                        parts = (
                            family[s]["zeroth"][row]["richardson"][k]
                            + family[s]["pt2"][row]["richardson"][k]
                        )
                        assert abs(total - parts) <= 1e-12 * (1 + abs(total))


def test_error_bars_and_ladders():
    for name in CASES:
        record = case(name)
        tol = record["tolerances"]
        assert tol["error_bar"] <= CONSUMER_GATE / 5
        lo, hi = tol["ladder_window"]
        for s in SCHEMES:
            for term in TERMS:
                measured = record["noise_measured"][s].get(term)
                for k, noise in enumerate(record["noise"][s][term]):
                    if measured is not None:
                        assert noise >= tol["noise_margin"] * measured[k]
                        assert noise >= tol["noise_floor"]
        for label, family, _ in families(record):
            for s in SCHEMES:
                for term in TERMS:
                    rows = family[s][term]
                    bar = max(max(r["error_bar"]) for r in rows)
                    assert bar < tol["error_bar"], (name, label, s, term)
                    assert record["checks"][label][f"{s}_{term}"]["error_bar"] == bar
                    for r in rows:
                        assert all(lo < x < hi for x in r["ratios"])


def test_self_checks_and_guard():
    for name in CASES:
        record = case(name)
        tol = record["tolerances"]
        fixture = record["fixture"]
        assert fixture["ci_vs_committed"] == 0.0
        assert fixture["heff_vs_committed"] < tol["brute_force"]
        reference = record["reference"]
        assert reference["gradient_norm"] < tol["polish"]
        for s in SCHEMES:
            for checks in (fixture[s], reference["psi1"][s]):
                assert checks["coupling_vs_golden"] < tol["brute_force"]
                assert checks["heff_vs_golden"] < tol["brute_force"]
                assert checks["overlap_with_references"] < tol["brute_force"]
            assert fixture[s]["density_off_blocks"] < tol["off_block"]
        guard = record["guard"]
        assert guard["gradient_norm"] < tol["polish"]
        for key in ("ci", "sc_vectors", "pc_vectors"):
            assert guard[key] < tol["guard"], (name, key)


def test_every_term_is_visible():
    terms = ("zeroth", "pt2", "D_ti", "D_at", "D_ai")
    for s in SCHEMES:
        for term in terms:
            best = max(
                row[s][term]["max_over_bar"]
                for name in CASES
                for row in case(name)["visibility"]
            )
            assert best >= 10, (s, term, best)


def npy_header(path: Path) -> dict:
    raw = path.read_bytes()
    assert raw[:6] == b"\x93NUMPY"
    length = int.from_bytes(raw[8:10], "little")
    header = ast.literal_eval(raw[10 : 10 + length].decode("latin1"))
    header["data"] = len(raw) - 10 - length
    return header


def test_densities_are_laid_out():
    for name in CASES:
        record = case(name)
        n, nmo = len(record["weights"]), record["nmo"]
        for s in SCHEMES:
            header = npy_header(OUT / name / f"{s}_pt2_density.npy")
            assert header["descr"] == "<f8"
            assert header["fortran_order"] is True
            assert header["shape"] == (n, n, nmo, nmo)
            assert header["data"] == 8 * n * n * nmo * nmo


@pytest.mark.slow
def test_case_reruns():
    # before ANY third-party import: numpy is in the `tools` group too
    pytest.importorskip("pyscf", reason="needs the `tools` dependency group")
    pytest.importorskip("pyblock2", reason="needs the `tools` dependency group")
    pytest.importorskip("numpy", reason="needs the `tools` dependency group")
    sys.path.insert(0, str(ROOT / "tools" / "emit"))
    import numpy
    import qd_nevpt2_nac_reference as emit

    name = "h8_sto3g_sa3"
    record = case(name)
    (entry,) = [
        e for e in emit.CASES if e.get("brute_force") and e["case"]["name"] == name
    ]
    fixture = emit.build_state(entry)
    space = emit.full_space(fixture)
    density, _ = emit.pt2_density(fixture, space, emit.sc_psi1(fixture, space))
    want = numpy.load(OUT / name / "sc_pt2_density.npy")
    assert numpy.abs(density - want).max() < 1e-14
    # The finest step's +h along the first direction.
    ref = emit.Reference(name, entry["case"], fixture.mo)
    coords = numpy.array([c for _, c in entry["case"]["atoms"]], dtype=float)
    u = numpy.array(record["directions"][0]).reshape(coords.shape)
    got = ref.point(coords + record["steps"][2] * u, ref.state.mo)
    raw = record["displaced"]["directional"][0]
    for s in SCHEMES:
        for k in ("z", "y"):
            assert numpy.abs(got[s][k] - numpy.array(raw[s][k][2][1])).max() < 1e-12
