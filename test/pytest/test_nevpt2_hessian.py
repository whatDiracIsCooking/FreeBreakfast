"""tools/nevpt2_hessian: the stencil helpers, and the committed SC- and
PC-NEVPT2 Hessian references in test/gtest/data/nevpt2_hessian/ (#803).

The helpers (tools/nevpt2_hessian/stencil.py) and the fixtures' invariants
are checked with the standard library alone, so they run on the push gate:
the stencil's order and its ratio of 16, the ladder's verdicts, the symmetry
and bilinearity checks; then every fixture's ladder recomputed from its
recorded gradients, its bars and ratios against the rule, the skew and
near-degenerate cases symmetric to their bar and bilinear on every pair, and
the symmetric case's failing pairs. Re-running the reference point through
the generator needs PySCF and block2, in the opt-in ``tools`` dependency
group, so that test is slow and skips (with a reason) wherever the group is
not installed::

    uv run --group tools pytest -m slow -rs test/pytest/test_nevpt2_hessian.py
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "nevpt2_hessian"
DATA = ROOT / "test" / "gtest" / "data" / "nevpt2_hessian"
CASES = {
    "h2co_skew_sto3g": "skew",
    "nh3_cs_sto3g": "symmetric",
    "h2co_neardeg_sto3g": "near_degenerate",
}


def load_stencil():
    spec = importlib.util.spec_from_file_location(
        "nevpt2_hessian_stencil", TOOL / "stencil.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stencil = load_stencil()


def close(a: float, b: float) -> bool:
    # Recomputed from fixture inputs, not read back: the stencil's sum()
    # reorders (3.12 compensates float sums), so it may move the last ulp of
    # each gradient, which the 1/s of a step leaves near 1e-16 / s absolute
    # (1e-12 is far below any bar, 2.6e-8 at least).
    return math.isclose(a, b, rel_tol=1e-13, abs_tol=1e-12)


@functools.cache
def fixture(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


# ------------------------------------------------------------------ helpers


def test_the_stencil_is_exact_through_quartics():
    # Fourth order: exact on every polynomial of degree four or less.
    for s in (0.3, 0.05):
        for coefficients in ((1.0,), (0.0, 2.0), (0.5, -1.0, 3.0, -2.0, 0.7)):
            values = {
                o * s: sum(c * (o * s) ** k for k, c in enumerate(coefficients))
                for o in stencil.OFFSETS
            }
            want = coefficients[1] if len(coefficients) > 1 else 0.0
            assert stencil.derivative(values, s) == pytest.approx(want, abs=1e-12)


def test_the_ladder_ratio_is_sixteen_on_a_smooth_function():
    h = 0.08
    values = {
        d: math.exp(0.7 * d) + math.sin(1.3 * d) for d in stencil.displacements(h)
    }
    coarse, middle, fine = (stencil.derivative(values, s) for s in stencil.steps(h))
    ratio = (coarse - middle) / (middle - fine)
    assert abs(ratio - stencil.TARGET) < 0.1
    assert stencil.TARGET == 16.0


def test_the_ladder_reads_eight_points():
    h = 0.04
    assert stencil.steps(h) == [0.04, 0.02, 0.01]
    points = stencil.displacements(h)
    assert len(points) == 8
    assert points == sorted(points)
    assert points == [-x for x in reversed(points)]
    assert max(points) == 2 * h and min(abs(x) for x in points) == h / 4


def test_noise_gain():
    norm = math.sqrt(1 / 144 + 64 / 144 + 64 / 144 + 1 / 144)
    assert stencil.noise_gain(0.5) == pytest.approx(norm / 0.5)


def test_column_ladder_differentiates_each_component():
    h = 0.1
    points = {d: [d**2, 3.0 * d, math.sin(d)] for d in stencil.displacements(h)}
    for column in stencil.column_ladder(points, h):
        assert column[0] == pytest.approx(0.0, abs=1e-12)
        assert column[1] == pytest.approx(3.0, abs=1e-12)
        assert column[2] == pytest.approx(1.0, abs=1e-5)


def test_judge():
    window = (12.8, 19.2)
    settled = stencil.judge([1.0 + 17e-6, 1.0 + 1e-6, 1.0], 1e-8, window)
    assert settled["settled"] is True and settled["bar"] == pytest.approx(1e-6)
    assert settled["ratio"] == pytest.approx(16.0)
    # Not fourth order: a ratio of 4 fails.
    assert (
        stencil.judge([1.0 + 5e-6, 1.0 + 1e-6, 1.0], 1e-8, window)["settled"] is False
    )
    # Below the floor: not judged, the bar is the floor.
    quiet = stencil.judge([1.0 + 1e-9, 1.0 + 2e-10, 1.0], 1e-8, window)
    assert quiet["ratio"] is None and quiet["settled"] is None
    assert quiet["bar"] == 1e-8
    # Below the floor at the fine rung but far above it at the coarse one:
    # a kink the fine rung steps over.
    kink = stencil.judge([1.0 + 1e-6, 1.0 + 2e-10, 1.0], 1e-8, window)
    assert kink["ratio"] is None and kink["settled"] is False


def test_symmetry_helpers():
    m = [[1.0, 2.0], [2.0 + 3e-8, 5.0]]
    bar = [[1e-8, 1e-8], [5e-8, 1e-8]]
    assert stencil.symmetry_misses(m, bar) == []
    assert stencil.symmetry_misses(m, [[1e-8] * 2] * 2) == [
        (0, 1, pytest.approx(3e-8), 1e-8)
    ]
    assert stencil.symmetrized(m)[0][1] == stencil.symmetrized(m)[1][0]
    assert stencil.symmetric_bar(bar) == [[1e-8, 5e-8], [5e-8, 1e-8]]
    assert stencil.asymmetry(m)[1][0] == pytest.approx(3e-8)


def test_bilinearity_passes_a_quadratic_form_and_fails_a_ratio_of_them():
    # E = x^T A x / 2: its second derivative along the unit
    # (e_0 + e_1) / sqrt 2 is (A00 + A11 + 2 A01) / 2.
    a = [[2.0, 0.5], [0.5, 1.0]]
    bar = [[1e-9] * 2] * 2
    along = 0.5 * (a[0][0] + a[1][1] + 2 * a[0][1])
    assert not stencil.bilinearity(a, bar, 0, 1, along, 1e-9)["fails"]

    # E = t^2 Q(q) with Q a ratio of quadratic forms (an empty perturber's
    # T / (Delta + H / N)): each direction is quadratic, but not bilinear.
    def q(x, y):
        return (x * x + y * y) * (x * x + 2 * y * y) / (x * x + 4 * y * y)

    m = [[2 * q(1, 0), 0.0], [0.0, 2 * q(0, 1)]]
    # Along the unit (1, 1) / sqrt 2: 2 Q(1, 1) / 2, Q being homogeneous.
    check = stencil.bilinearity(m, bar, 0, 1, q(1, 1), 1e-9)
    assert check["fails"] and check["pair"] == [0, 1]


# ----------------------------------------------------------------- fixtures


def test_the_issue_cases_are_there():
    assert sorted(p.stem for p in DATA.glob("*.json")) == sorted(CASES)
    for name, kind in CASES.items():
        record = fixture(name)
        assert record["name"] == name and record["kind"] == kind
        assert record["weights"] == [1.0] and record["frozen_core"] == 0
        assert 2 * record["ncore"] + record["nelecas"] == record["electrons"]
        assert len(record["active"]) == record["ncas"]
        for key in ("basis", "aux_basis"):
            path = ROOT / record[key]["file"]
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest() == record[key]["sha256"]
            )
        assert record["aux_basis"]["file"] == "data/aux_basis/cc-pVDZ-JKFIT.json"
        assert record["provenance"]["threads"] == 1
    # The near-degenerate case's pair: its smallest inactive gap is the
    # smallest of the three.
    gaps = {name: min(fixture(name)["inactive_gaps"].values()) for name in CASES}
    assert min(gaps, key=gaps.get) == "h2co_neardeg_sto3g"
    assert gaps["h2co_neardeg_sto3g"] < 0.01
    # The symmetric case has empty perturbers.
    symmetric = fixture("nh3_cs_sto3g")
    assert sum(c["sc_empty_perturbers"] for c in symmetric["classes"]) > 0


def gated(record: dict, q: str, i: int, j: int) -> bool:
    """The generator's rule: a quantity the case expects to fail is held
    only on the totally symmetric block."""
    irreps = record["coordinates"]["irreps"]
    total = record["coordinates"]["irreps_order"][0]
    return q not in record["may_fail"] or irreps[i] == irreps[j] == total


def schemes(record: dict) -> list[str]:
    return [q for q in ("casscf", "sc", "pc") if q in record]


def test_every_record_holds_what_the_issue_asks():
    for name in CASES:
        record = fixture(name)
        n = 3 * len(record["atoms"])
        assert "pc" in schemes(record) and "casscf" in schemes(record)
        for q in schemes(record):
            part = record[q]
            assert len(part["gradient"]) == len(record["atoms"])
            for key in ("hessian", "bar", "ratios", "column_bar", "asymmetry"):
                assert len(part[key]) == n and all(len(r) == n for r in part[key]), key
            assert len(part["ladder"]) == 3
            assert part["noise"]["floor"] > 0
            assert q == "casscf" or q in record["energies"]
    assert "sc" in fixture("h2co_skew_sto3g") and "sc" in fixture("nh3_cs_sto3g")


def test_the_ladders_are_the_recorded_gradients():
    for name in CASES:
        record = fixture(name)
        h = record["steps"]["h"]
        assert record["steps"]["ladder"] == stencil.steps(h)
        assert record["steps"]["displacements"] == stencil.displacements(h)
        n = 3 * len(record["atoms"])
        for q in schemes(record):
            part = record[q]
            for entry in part["points"]:
                j = entry["column"]
                points = dict(
                    zip(
                        record["steps"]["displacements"],
                        entry["gradients"],
                        strict=True,
                    )
                )
                ladder = stencil.column_ladder(points, h)
                for k in range(3):
                    for i in range(n):
                        assert close(ladder[k][i], part["ladder"][k][i][j]), (
                            name,
                            q,
                            k,
                            i,
                            j,
                        )
            finest = part["ladder"][-1]
            assert part["hessian"] == stencil.symmetrized(finest)
            assert part["asymmetry"] == stencil.asymmetry(finest)


def test_bars_and_ratios_follow_the_rule():
    for name in CASES:
        record = fixture(name)
        window = record["tolerances"]["ratio_window"]
        assert window == [0.8 * 16, 1.2 * 16]
        for q in schemes(record):
            part = record[q]
            floor = part["noise"]["floor"]
            n = len(part["hessian"])
            judged = []
            for i in range(n):
                for j in range(n):
                    verdict = stencil.judge(
                        [part["ladder"][k][i][j] for k in range(3)], floor, window
                    )
                    assert verdict["bar"] == part["column_bar"][i][j]
                    assert verdict["ratio"] == part["ratios"][i][j]
                    if not gated(record, q, i, j):
                        continue
                    assert verdict["settled"] is not False, (name, q, i, j)
                    if verdict["ratio"] is not None:
                        judged.append(verdict["ratio"])
            assert part["bar"] == stencil.symmetric_bar(part["column_bar"])
            # Every case shows its ladder in the asymptotic range somewhere.
            assert judged, (name, q)
            assert all(window[0] < r < window[1] for r in judged)


def test_every_held_block_is_symmetric_and_bilinear():
    for name in CASES:
        record = fixture(name)
        for q in schemes(record):
            part = record[q]
            n = len(part["hessian"])
            assert len(part["bilinearity"]) == n * (n - 1) // 2
            failing = [p["pair"] for p in part["bilinearity"] if p["fails"]]
            assert failing == part["failing_pairs"]
            misses = stencil.symmetry_misses(part["ladder"][-1], part["column_bar"])
            for i, j, _, _ in misses:
                assert not gated(record, q, i, j), (name, q, i, j)
            for i, j in failing:
                assert not gated(record, q, i, j), (name, q, i, j)
            for p in part["bilinearity"]:
                i, j = p["pair"]
                if gated(record, q, i, j):
                    assert p["settled"] is not False, (name, q, i, j)
            if q not in record["may_fail"]:
                assert part["failed_blocks"] == [], (name, q)


def test_the_symmetric_case_records_its_failing_block():
    record = fixture("nh3_cs_sto3g")
    assert record["may_fail"] == ["sc"]
    irreps = record["coordinates"]["irreps"]
    assert record["coordinates"]["irreps_order"] == ["A'", "A''"]
    assert irreps.count("A'") == 7 and irreps.count("A''") == 5
    # SC's non-totally-symmetric block has no Hessian: recorded, and never
    # in the totally symmetric block.
    blocks = record["sc"]["failed_blocks"]
    assert blocks and ["A'", "A'"] not in blocks
    failing = record["sc"]["failing_pairs"]
    assert failing
    for i, j in failing:
        assert (irreps[i], irreps[j]) != ("A'", "A'")
    # PC is defined on every block there.
    assert record["pc"]["failing_pairs"] == [] and record["pc"]["failed_blocks"] == []
    h = record["steps"]["h"]
    for p in record["sc"]["bilinearity"]:
        derivatives = {
            d: [x]
            for d, x in zip(
                record["steps"]["displacements"], p["derivatives"], strict=True
            )
        }
        recomputed = [v[0] for v in stencil.column_ladder(derivatives, h)]
        assert len(recomputed) == len(p["ladder"])
        assert all(map(close, recomputed, p["ladder"])), p["pair"]


def test_the_guard_and_the_checks():
    for name in CASES:
        record = fixture(name)
        tol, guard = record["tolerances"], record["guard"]
        assert guard["prediction"] < tol["prediction_bound"]
        assert guard["subspace"] < tol["subspace_bound"]
        assert guard["ci"] < tol["ci_bound"]
        assert guard["inactive_gap"] > tol["degenerate"]
        checks = record["checks"]
        assert checks["sc_invariance"] < 1e-10
        assert checks["block2_sc"] < 1e-10
        assert checks["block2_pc"] < tol["block2_pc"]
        assert checks["casscf_gradient_vs_analytic"] < tol["analytic"]
        e = record["energies"]
        assert abs(e["sc"] - e["casscf"] - record["e_corr"]["sc"]) < 1e-12
        assert abs(e["pc"] - e["casscf"] - record["e_corr"]["pc"]) < 1e-12
        assert record["polish_grad"] < tol["polish_grad"]


@pytest.mark.slow
def test_the_reference_point_reruns():
    # before ANY third-party import: numpy is in the `tools` group too
    pytest.importorskip("pyscf", reason="needs the `tools` dependency group")
    pytest.importorskip("pyblock2", reason="needs the `tools` dependency group")
    pytest.importorskip("numpy", reason="needs the `tools` dependency group")
    sys.path.insert(0, str(TOOL))
    import nevpt2_hessian as tool

    (spec,) = [c for c in tool.CASES if c["name"] == "h2co_skew_sto3g"]
    want = fixture("h2co_skew_sto3g")
    ref = tool.Reference(spec)
    assert abs(ref.energy - want["energies"]["casscf"]) < 1e-11
    assert abs(ref.nevpt2["sc"] - want["e_corr"]["sc"]) < 1e-11
    assert abs(ref.nevpt2["pc"] - want["e_corr"]["pc"]) < 1e-11
