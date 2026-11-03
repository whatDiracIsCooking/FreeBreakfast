"""The committed DF-SA-CASSCF MECI reference, test/gtest/data/casscf_meci/.

Its claims are re-derived with the standard library alone, so they run on the
push gate: at the recorded MECI the gap is under the loop's tolerance, the
projected force -P gbar rebuilt from the recorded geometry, gradients and
h_01 (#771's projector: rigid modes off, then x = g_1 - g_0 and y = h_01
orthonormalized in that order) is under its tolerance and is the one written,
x_norm, h_norm and gbar_plane are those vectors', k_min is positive and the
least seam curvature, the seam's own curvature (the plane's turn included)
is positive too, and the S2 margin clears its floor. An independent
PySCF single point at the recorded geometry needs the opt-in ``tools``
dependency group and a minute of CASSCFs, so that test is slow and skips
(with a reason) wherever the group is not installed::

    uv run --group tools pytest -m slow -rs test/pytest/test_meci_reference.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "test" / "gtest" / "data" / "casscf_meci" / "casscf_meci.json"


def doc() -> dict:
    return json.loads(MANIFEST.read_text())


def flat(rows: list) -> list[float]:
    return [x for row in rows for x in row]


def dot(a: list[float], b: list[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def removed_basis(coords: list, x: list[float], y: list[float]) -> list[list[float]]:
    """Gram-Schmidt over the translations, the rotations about the centroid,
    x and y, in that order: the directions #771's projector removes."""
    n = len(coords)
    centroid = [math.fsum(r[k] for r in coords) / n for k in range(3)]
    centred = [[r[k] - centroid[k] for k in range(3)] for r in coords]
    vectors = [
        [1.0 if k == j else 0.0 for _ in range(n) for k in range(3)] for j in range(3)
    ]
    for j in range(3):
        axis = [1.0 if k == j else 0.0 for k in range(3)]
        vectors.append(
            [
                c
                for r in centred
                for c in (
                    r[1] * axis[2] - r[2] * axis[1],
                    r[2] * axis[0] - r[0] * axis[2],
                    r[0] * axis[1] - r[1] * axis[0],
                )
            ]
        )
    vectors += [x, y]
    basis: list[list[float]] = []
    for v in vectors:
        for _ in range(2):  # twice is enough for orthogonality to rounding
            for u in basis:
                c = dot(u, v)
                v = [a - c * b for a, b in zip(v, u, strict=True)]
        norm = math.sqrt(dot(v, v))
        assert norm > 1e-8
        basis.append([a / norm for a in v])
    return basis


def test_the_case_is_the_issues():
    d = doc()
    assert d["basis"] == "cc-pvdz"
    assert d["aux_basis"]["file"] == "data/aux_basis/cc-pVDZ-JKFIT.json"
    assert (ROOT / d["aux_basis"]["file"]).is_file()
    assert [a["symbol"] for a in d["meci"]["atoms"]] == list("CCHHHH")
    assert (d["ncas"], d["nelecas"]) == (2, 2)
    assert d["weights"] == [0.5, 0.5]
    start = d["start"]
    for key in ("gradients",):
        assert len(start[key]) == 2
    for vector in start["ci"]:
        values = flat(vector)
        assert abs(dot(values, values) - 1) < 1e-10
        top = max(abs(x) for x in values)
        assert next(x for x in values if abs(x) >= top - 1e-8) > 0


def test_the_meci_meets_the_loop_tolerances():
    d = doc()
    meci, tolerances = d["meci"], d["optimizer"]
    coords = [a["coord"] for a in meci["atoms"]]
    g0, g1 = (flat(g) for g in meci["gradients"])
    h = flat(meci["h"])
    x = [b - a for a, b in zip(g0, g1, strict=True)]
    gbar = [0.5 * (a + b) for a, b in zip(g0, g1, strict=True)]
    gap = meci["energies"][1] - meci["energies"][0]
    assert abs(gap - meci["gap"]) < 1e-14
    assert 0 <= gap < tolerances["gap_tolerance"] == 1e-8

    basis = removed_basis(coords, x, h)
    force = [-v for v in gbar]
    for u in basis:
        c = dot(u, gbar)
        force = [f + c * b for f, b in zip(force, u, strict=True)]
    for got, want in zip(force, flat(meci["projected_force"]), strict=True):
        assert abs(got - want) < 1e-12
    worst = max(abs(f) for f in force)
    assert abs(worst - meci["max_projected_force"]) < 1e-12
    assert worst < tolerances["force_tolerance"] == 1e-6

    assert abs(meci["x_norm"] - math.sqrt(dot(x, x))) < 1e-12
    assert abs(meci["h_norm"] - math.sqrt(dot(h, h))) < 1e-12
    plane = math.hypot(dot(basis[6], gbar), dot(basis[7], gbar))
    assert abs(meci["gbar_plane"] - plane) < 1e-12


def test_the_cone_slopes_bound_the_norms():
    # The gap's least and greatest slope in the plane, twice the singular
    # values of [x/2; h], are invariant where x_norm and h_norm are not;
    # each of x_norm and 2 h_norm is at least the least (README).
    meci = doc()["meci"]
    g0, g1 = (flat(g) for g in meci["gradients"])
    half = [0.5 * (b - a) for a, b in zip(g0, g1, strict=True)]
    h = flat(meci["h"])
    gram = (dot(half, half), dot(half, h), dot(h, h))
    trace, det = gram[0] + gram[2], gram[0] * gram[2] - gram[1] ** 2
    root = math.sqrt(max(trace * trace / 4 - det, 0.0))
    least = 2 * math.sqrt(trace / 2 - root)
    greatest = 2 * math.sqrt(trace / 2 + root)
    assert abs(least - 0.1413) < 1e-4 and abs(greatest - 0.2290) < 1e-4
    assert min(meci["x_norm"], 2 * meci["h_norm"]) >= least * (1 - 1e-12)


def test_the_seam_minimum_and_the_margin():
    d = doc()
    meci = d["meci"]
    curvatures = meci["seam_curvatures"]
    assert len(curvatures) == 3 * len(meci["atoms"]) - 8
    assert meci["k_min"] == min(curvatures) > 0
    reduced = meci["reduced_curvatures"]
    assert len(reduced) == len(curvatures) and min(reduced) > 0
    assert d["s2_margin"] > d["optimizer"]["s2_margin_floor"] == 1e-2
    assert min(p["s2_margin"] for p in d["path"]) >= d["s2_margin"]
    last = d["path"][-1]
    assert last["gap"] == meci["gap"]
    assert last["max_projected_force"] == meci["max_projected_force"]
    fresh = d["fresh"]
    assert fresh["gap"] < 1e-8 and fresh["max_projected_force"] < 1e-6


@pytest.mark.slow
def test_an_independent_single_point_reproduces_the_meci():
    # before ANY third-party import: numpy is in the `tools` group too
    pytest.importorskip("pyscf", reason="needs the `tools` dependency group")
    numpy = pytest.importorskip("numpy", reason="needs the `tools` dependency group")
    sys.path.insert(0, str(ROOT / "tools" / "emit"))
    import meci_reference as emit
    from pyscf import lib

    lib.num_threads(emit.THREADS)
    d = doc()
    coords = numpy.array([a["coord"] for a in d["meci"]["atoms"]])
    at = emit.analysis(coords, emit.evaluate(coords, None, full=True))
    assert at["gap"] < 1e-8
    assert at["max_force"] < 1e-6
