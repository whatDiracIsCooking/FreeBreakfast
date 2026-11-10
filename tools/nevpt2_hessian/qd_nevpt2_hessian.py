#!/usr/bin/env python3
"""Write the SC- and PC-QD-NEVPT2 nuclear Hessian references the analytic QD
Hessian (#959, #965-#967) is gated on: test/gtest/data/qd_nevpt2_hessian/.

No program has a QD-NEVPT2 Hessian, so the oracle is docs/nevpt2/nevpt2.tex's
semi-numerical one (Table qdhesschecks): a fourth-order central stencil over
the analytic QD gradient, here this tree's own `nevpt2::qd_sc_gradient` and
`qd_pc_gradient` (milestone 20), which test/gtest/data/qd_nevpt2_gradient/
ladder-validates against the tree's own energy. No other program has a QD
gradient either, and Python cannot call it, so the C++ driver
`free-breakfast-qd-nevpt2-hessian` (tools/qd_nevpt2_hessian/main.cpp) runs
the points: at every displaced geometry the chain and continuity guard of
tools/qd_nevpt2_gradient (`ChainSampler`, every point polished; the CASSCF
roots matched by their overlaps, H_eff's eigenstates by u_K's, never by
rank), then the analytic G(W) at each W this tool asks for, in the
reference's root basis:

- **the state**, W^K = u_K u_K^T, u_K the eigenvector matched to the
  reference's state K: its ladder is the whole state Hessian of E_K,
  Eq. qd-hess, roots tracked by R_P's overlap;
- **the pair sum**, W^P + W^Q, both tracked: Eq. qd-pair's left side.

**The SA3 crossing cases** (`CROSSING_CASES`, #1059): ethylene's
DF-SA3-CASSCF(2,2) at the MECI of QD states 0 and 1, one per scheme, each at
its own scheme's MECI, which qd_meci.py converges with the tree's
`[optimize]` MECI mode and records first. At a cone E_0 and E_1 have no
Hessian, and u_0, u_1 are not defined one by one, so the driver tracks the
pair by its two-state eigenspace (its `crossing`; src/pipeline/hessian's
`tracked_qd_pair`, a principal-angle match) and differences only what is
smooth there, while the third state stays gapped:

- **the pair sum** G(W^0 + W^1), W^0 + W^1 the projector onto the tracked
  eigenspace: Eq. qd-pair's left side at a crossing;
- **the seam** G(W) at the reference's fixed W = 1/2 s (V_0 V_1^T + V_1
  V_0^T) (a non-projector W: Eq. qd-lowdin's first term is its Hessian at
  fixed W). V is the pair's eigenspace's own basis, not the eigensolver's
  R_0, R_1, which a rerun returns turned and signed differently (#1098): the
  Loewdin-orthonormalized projections of CASSCF roots r_0 < r_1 onto it (the
  roots whose 2 x 2 block of [R_0 R_1] has the largest |det|), as
  hessian::df_qd_sc_nevpt2_crossing's c basis; s the sign of H_eff[r_0][r_1],
  which carries those roots' phases. The driver builds it
  (`seam_weight` here is the same convention), and each seam entry records
  the W it was laddered at (`w`, `roots`, `sign`);
- **the third state**, W^2 = u_2 u_2^T, tracked by u_2 as away from a
  crossing.

The third state's gap from the pair, at the reference and at every point,
must stay above mcscf::kRootGapFloor (`ROOT_GAP_FLOOR`), and is recorded.

This tool writes the request, runs the driver (or reads its output back with
--points), and judges every ladder; standard library only, over stencil.py.

**The C2v coupling case** (`COUPLING_CASES`, #1058; nevpt2.tex's symmetric
geometries): a C2v SA2 whose roots are A1 and B2. The driver runs with
`gradients` false, and the tool ladders C_MN = H_eff[M][N] (the reference's
root basis) along each coupled b2 coordinate and along their sum. The
coordinates are C2v-adapted (`symmetry_coordinates`). Each first-derivative
ladder is judged as a Hessian element is, over sigma from re-solves of every
positive displacement started from its negative image's orbitals. The pair's
bilinearity is D(u) against (D(q_a) + D(q_b)) / sqrt 2. SC's expected
refusals are the pairs that fail it. PC's come from qd_coupling_analysis.py
(its lifted metric modes), whose output must exist first. `pc_probe` records
the sampler's verdict on PC at one displaced point.

**The C2v whole-Hessian case** (`SYMMETRIC_CASES`, #1093): the coupling
case's system, its state Hessians laddered as the C1 cases' are, in its
C2v-adapted coordinates. Each scheme skips the coordinates, states and pairs
its `not_laddered` names (recorded): PC the two b2 vibrations, along which
its sampler refuses a point and its analytic Hessian every block; SC the
three that lower the symmetry, along which its surface crosses over on a
scale of 1e-5 to 1e-4 bohr, and state 1, which its analytic Hessian refuses.

**The Hessian.** Column j is the fourth-order central stencil over the
gradients at R0 +- s q_j, R0 +- 2 s q_j for s = h, h/2 and h/4 (eight points,
`stencil.displacements`); element (i, j) is q_i^T dG/dq_j. Per element the
ratio of successive differences must lie in RATIO_WINDOW about 16 wherever
the finer difference is above the noise floor, and below it the coarser
difference must be one a settled ladder could have (`stencil.judge`): an
element that does not settle fails the generator (STUCK, with the numbers),
never a looser bar. The bar is the finer difference, or the floor below it.
The recorded Hessian is the finest step's, symmetrized; before that it must
be symmetric to its bar, the oracle's own check that the displaced solves lie
on one surface and that the states were tracked.

**The noise floor.** sigma_g per quantity is measured, not assumed: the
reference geometry re-converged from the orbitals of NOISE_COLUMNS' displaced
points (the +-h points of the first columns), every gradient against the
reference's own, the RMS over components and starts. A stencil at s
multiplies independent noise by stencil.noise_gain(s), so the finer
difference carries at most sigma_g (gain(h/2) + gain(h/4)); the floor is
FLOOR_FACTOR = 20 times that, as #803's, so a one-sigma noise moves a judged
ratio by 5% and the 20% window holds four.

On a symmetry-adapted case sigma_g is the larger of the restarts' and the one
the point group's zeros imply (`symmetry_sigma`): an element H[i][j] whose
coordinates are of different irreps is zero by symmetry, so its value on each
rung is that stencil's noise alone, and the RMS of value / noise_gain(s) over
those elements and rungs is a second measurement of sigma_g. The restarts
miss noise that the displaced points carry and a restart at R0 does not (on
#1093's case SC's read 4.2e-15 against the zeros' 4.6e-14). The restarts
start from the first NOISE_COLUMNS columns that are not a rigid motion
(`noise_columns`; on a Cartesian case, the first ones, unchanged).

**The guard** is the driver's (its docstring); a refused point fails the run
and nothing is written. Its worst values are recorded.

    tools/nevpt2_hessian/qd_nevpt2_hessian.py [--only CASE ...] \\
        [--binary build/tools/qd_nevpt2_hessian/free-breakfast-qd-nevpt2-hessian]

on a GPU, from a build of the tree, writes
test/gtest/data/qd_nevpt2_hessian/<case>.json. Never hand-edited. The tree's
GPU sums reassociate run to run, so a rerun reproduces every number to its
noise, not byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

# isort: split
import stencil  # noqa: E402

OUT = ROOT / "test" / "gtest" / "data" / "qd_nevpt2_hessian"
SOURCE = ROOT / "test" / "gtest" / "data" / "qd_nevpt2" / "qd_nevpt2.json"
# qd_weight_search.py's screening: the reference point of a case with its
# own `system`.
SEARCH = OUT / "qd_weight_search.json"
# qd_meci.py's MECIs: the geometry of each crossing case, and the [optimize]
# end point it is held to.
MECI = OUT / "ethylene_meci.json"
# mcscf::kRootGapFloor (src/mcscf/root_rotation): the third state's gap from
# a crossing pair must stay above it.
ROOT_GAP_FLOOR = 1e-8
BINARY = (
    ROOT / "build" / "tools" / "qd_nevpt2_hessian" / "free-breakfast-qd-nevpt2-hessian"
)
BASIS_FILES = {
    "cc-pvdz": "data/basis/cc-pVDZ.json",
    "sto-3g": "data/basis/sto-3g.json",
    "6-31g": "data/basis/6-31g.json",
}
AUX = ("cc-pVDZ-JKFIT", "data/aux_basis/cc-pVDZ-JKFIT.json")

# The ladder's verdicts, #803's: the window about 16 a judged ratio must lie
# in, and the noise floor's multiple of the finer difference's noise.
RATIO_WINDOW = (0.8 * stencil.TARGET, 1.2 * stencil.TARGET)
FLOOR_FACTOR = 20.0
# Every gradient's Z-vector tolerance (nevpt2::qd_*_gradient's
# zvector_tolerance; its true residual is held to it).
ZVECTOR_TOL = 3e-12
# The noise measurement: the reference geometry from the orbitals of the
# +h and -h points of this many columns, and those points and the reference
# again with the Z-vectors solved to ZVECTOR_LOOSE. The columns are the first
# that are not a rigid translation or rotation (`noise_columns`): a restart
# from a rigidly moved start re-converges to the reference's own orbitals.
NOISE_COLUMNS = 2
ZVECTOR_LOOSE = 10 * ZVECTOR_TOL

SKEW = [
    ["C", [0.0, 0.0, 0.0]],
    ["O", [0.12, -0.08, 2.29]],
    ["H", [0.45, 1.74, -1.13]],
    ["H", [-0.31, -1.81, -1.02]],
]

# Every case: qd_nevpt2.json's system (`source`) or its own (`system`), the
# ladder's h (bohr), the schemes, and the pairs whose sum is differenced.
# The coordinates are Cartesian (C1).
#
# SC does not settle on 0.4/0.4/0.2 (README), a different SA-CASSCF
# solution from the equal-weight one: its closest core pair is 2.6e-3 Eh
# apart. `h2co_skew_ccpvdz_sa3_w424` is SC's unequal-weight case (#1060):
# the same geometry at 0.4/0.2/0.4, which qd_weight_search.py's screening
# puts on the equal-weight solution's branch (core pair 0.084 Eh apart) and
# whose reference point it records. Its SC ladder settles at the equal-weight
# case's step.
CASES = [
    {
        "name": "h2co_skew_ccpvdz_sa3",
        "source": "h2co_skew_ccpvdz_sa3",
        "h": {"sc": 0.0003125, "pc": 0.0025},
        "pairs": [[0, 1], [0, 2], [1, 2]],
    },
    {
        "name": "h2co_skew_ccpvdz_sa3_weighted",
        "source": "h2co_skew_ccpvdz_sa3_weighted",
        "h": {"pc": 0.0025},
        "pairs": [[0, 1], [0, 2], [1, 2]],
    },
    {
        "name": "h2co_skew_ccpvdz_sa3_w424",
        "system": {
            "atoms": SKEW,
            "basis": "cc-pvdz",
            "ncas": 3,
            "nelecas": 4,
            "active": None,
            "weights": [0.4, 0.2, 0.4],
        },
        "h": {"sc": 0.0003125},
        "pairs": [[0, 1], [0, 2], [1, 2]],
    },
    # The fast tier's case (#1243): the smallest C1 SA2 whose analytic QD
    # Hessian has every part (#1207 timed the cc-pVDZ and H2N-B cases at 6 to
    # 50 s, their cost the steps run over every Cartesian coordinate). Two
    # stretched, twisted H2 units, no symmetry (bohr), 6-31G: a core of one,
    # CAS(2,2) over the frontier sigma and sigma*, five virtuals, so the
    # virtual rotations give kappa_oo its rows (STO-3G's one virtual has
    # none). PC at twice the C1 cases' step: at 2.5e-3 its pair's column
    # along H1 y missed on the finest rung alone, noise past the restarts'
    # (README). Its reference point is held to PySCF's, qd_pyscf_point.py's
    # `<name>.pyscf.json`, written first.
    {
        "name": "h4_631g_sa2",
        "system": {
            "atoms": [
                ["H", [0.0, 0.0, -0.85]],
                ["H", [0.05, 0.1, 0.85]],
                ["H", [2.9, 0.4, -0.3]],
                ["H", [3.1, 1.6, 0.6]],
            ],
            "basis": "6-31g",
            "ncas": 2,
            "nelecas": 2,
            "active": None,
            "weights": [0.5, 0.5],
        },
        "pyscf": True,
        "h": {"sc": 0.0003125, "pc": 0.005},
        "pairs": [[0, 1]],
    },
]

# The C2v SA2 case with A1/B2 roots (#1058, nevpt2.tex's symmetric
# geometries, Table qdhesschecks): aminoborylene H2N-B, planar in yz, the C2
# axis z, B-N 1.36 A, N-H 1.00 A, angle BNH 115 deg (bohr below). STO-3G
# keeps every inactive pair apart (README: 0.07 Eh at least); in cc-pVDZ two
# virtuals of A1 and B2 sit 8.7e-4 Eh apart and SC's coupling moves on that
# scale. CAS(4,3) over the frontier pi (b1), sigma (a1) and the empty
# in-plane p (b2): roots 1A1 and 1B2, which the two b2 vibrations couple.
# Its analysis (guarded tuples, PC's lifted metric modes) is
# qd_coupling_analysis.py's, written first.
#
# `h` per scheme: SC alone. PC's lifted modes pass its cut at every
# displaced point (`pc_probe`), so the sampler refuses PC there. SC's C_01 / s
# drifts by about 1e-7 Eh/bohr between s = 1e-3 and 2e-5 and is flat to its
# s^2 term below 1e-4 (README), so h = 4e-5 (points 1e-5 to 8e-5); at
# h = 2.5e-3 the CAS(4,3) ladder along B2 1 reads a ratio of -1.6.
H2NB = [
    ["B", [0.0, 0.0, 0.0]],
    ["N", [0.0, 0.0, -2.570028]],
    ["H", [0.0, 1.712674, -3.36866]],
    ["H", [0.0, -1.712674, -3.36866]],
]
COUPLING_CASES = [
    {
        "name": "h2nb_sto3g_sa2",
        "system": {
            "atoms": H2NB,
            "basis": "sto-3g",
            "ncas": 3,
            "nelecas": 4,
            "active": None,
            "weights": [0.5, 0.5],
        },
        "irreps": ["A1", "B2"],
        "h": {"sc": 4e-5},
        "state_pairs": [[0, 1]],
        # The two b2 vibrations: C_01's ladder along each and along their sum.
        "coupled": ["B2 1", "B2 2"],
    },
]

# The A1/B2 case's whole state Hessians (#1093): COUPLING_CASES' system in
# its C2v-adapted coordinates (`symmetry_coordinates`), the reference point
# held to the coupling record's (`coupling`). Each scheme ladders every
# coordinate but its `not_laddered` ones, and every state and pair but its
# `not_laddered` ones (README):
#
# - PC, at the C1 cases' 2.5e-3, both states and the pair, skips the two b2
#   vibrations: its sampler refuses a point along either once the lifted
#   modes pass its cut (from 1.5e-5 bohr along B2 2 and 2e-5 along B2 1),
#   and the analytic PC Hessian refuses every block they make (the coupling
#   record's `expected_refusals`).
# - SC, at the C1 cases' 3.125e-4, skips the three vibrations that lower the
#   symmetry, B1 1, B2 1 and B2 2, which populate perturbers empty by
#   symmetry at second order (nevpt2.tex Sec. hess-refusals): along them its
#   surface crosses over on a scale of 1e-5 to 1e-4 bohr and settles at
#   neither 4e-5 nor 3.125e-4; the b2 ones' only check is the
#   coupling record's C_01 ladders. State 0 alone: the analytic SC Hessian
#   refuses state 1 by its conditioning rule (root 1's Si, #1102), so state
#   1 and the pair are not laddered.
SYMMETRIC_CASES = [
    {
        "name": "h2nb_sto3g_sa2_hessian",
        "coupling": "h2nb_sto3g_sa2",
        "system": COUPLING_CASES[0]["system"],
        "h": {"sc": 0.0003125, "pc": 0.0025},
        "pairs": [[0, 1]],
        "not_laddered": {
            "sc": {
                "coordinates": ["B1 1", "B2 1", "B2 2"],
                "states": [1],
                "pairs": [[0, 1]],
            },
            "pc": {"coordinates": ["B2 1", "B2 2"], "states": [], "pairs": []},
        },
    },
]


# The SA3 crossing (#1059): ethylene's DF-SA3-CASSCF(2,2) at the MECI of QD
# states 0 and 1, one case per scheme at that scheme's MECI (qd_meci.py's
# `variant`), Cartesian coordinates. `crossing` is the degenerate pair the
# driver tracks by its eigenspace; `pairs` and `seams` what is differenced
# over it; the third state's own W^2 beside them.
CROSSING_CASES = [
    {
        "name": "ethylene_ccpvdz_sa3_sc_meci",
        "meci": "sc",
        "h": {"sc": 0.0003125},
        "crossing": [0, 1],
        "pairs": [[0, 1]],
        "seams": [[0, 1]],
    },
    {
        "name": "ethylene_ccpvdz_sa3_pc_meci",
        "meci": "pc",
        "h": {"pc": 0.0025},
        "crossing": [0, 1],
        "pairs": [[0, 1]],
        "seams": [[0, 1]],
    },
]


def sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def system(case: dict) -> dict:
    """qd_nevpt2.json's entry for the case, its own `system`, or its MECI's
    (qd_meci.py's record)."""
    if "system" in case:
        return case["system"]
    if "meci" in case:
        return meci_entry(case)["system"]
    doc = json.loads(SOURCE.read_text())
    (entry,) = [c for c in doc["cases"] if c["name"] == case["source"]]
    return {
        "atoms": entry["atoms"],
        "basis": entry["basis"],
        "ncas": entry["ncas"],
        "nelecas": entry["nelecas"],
        "active": None,
        "weights": entry["weights"],
    }


def meci_entry(case: dict) -> dict:
    """qd_meci.py's record of the case's MECI."""
    doc = json.loads(MECI.read_text())
    (entry,) = [m for m in doc["meci"] if m["variant"] == case["meci"]]
    return entry


def coordinates(natom: int, symbols: list[str]) -> dict:
    """The Cartesian coordinates: q_j = e_j."""
    n = 3 * natom
    return {
        "labels": [
            f"{symbols[a]}{a} {'xyz'[k]}" for a in range(natom) for k in range(3)
        ],
        "vectors": [[1.0 if i == j else 0.0 for i in range(n)] for j in range(n)],
    }


def scheme_labels(case: dict, scheme: str, coords: dict) -> list[str]:
    """The coordinates @p scheme ladders: every one but the case's
    `not_laddered` ones, in @p coords' order."""
    skipped = case.get("not_laddered", {}).get(scheme, {}).get("coordinates", [])
    return [label for label in coords["labels"] if label not in skipped]


def laddered(coords: dict, labels) -> dict:
    """@p coords restricted to the coordinates @p labels names, in its order:
    the columns a scheme ladders."""
    keep = [i for i, label in enumerate(coords["labels"]) if label in labels]
    assert len(keep) == len(labels), (labels, coords["labels"])
    return {key: [value[i] for i in keep] for key, value in coords.items()}


def noise_columns(coords: dict) -> list[int]:
    """The first NOISE_COLUMNS columns that are not labelled a rigid motion
    (`symmetry_coordinates`' T_ and R_; every Cartesian one counts)."""
    vibrations = [
        j
        for j, label in enumerate(coords["labels"])
        if not label.startswith(("T_", "R_"))
    ]
    return vibrations[:NOISE_COLUMNS]


def plan(h: float, coords: dict) -> tuple[list[dict], dict]:
    """The request's points at the ladder's @p h, and where each lands: index
    0 the reference; then per column and displacement; then the noise
    restarts and the looser solves."""
    disps = stencil.displacements(h)
    vectors = coords["vectors"]
    points = [{"displacement": [0.0] * len(vectors[0]), "from": None}]
    # noise: (restart, original) pairs of the same geometry.
    where = {"reference": 0, "columns": [], "noise": []}
    for q in vectors:
        row = {}
        for d in disps:
            row[d] = len(points)
            points.append({"displacement": [d * x for x in q], "from": None})
        where["columns"].append(row)
    zero = [0.0] * len(vectors[0])
    for j in noise_columns(coords):
        for d in (-h, h):
            where["noise"].append((len(points), 0))
            points.append({"displacement": zero, "from": where["columns"][j][d]})
    loose = [0] + [
        where["columns"][j][d] for j in noise_columns(coords) for d in (-h, h)
    ]
    for original in loose:
        where["noise"].append((len(points), original))
        points.append(
            {
                "displacement": points[original]["displacement"],
                "from": None,
                "zvector_tolerance": ZVECTOR_LOOSE,
            }
        )
    return points, where


def run_driver(binary: pathlib.Path, request: dict, out: pathlib.Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(request, f)
        path = f.name
    subprocess.run([str(binary), path, str(out)], check=True)
    pathlib.Path(path).unlink()


def dot(a: list[float], b: list[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def quantities(case: dict, n: int) -> list[tuple[str, int]]:
    """Each differenced G(W): ("state", K) per state but a crossing pair's,
    then ("pair", i) per requested pair, then ("seam", i) per requested seam,
    in the driver's order."""
    crossing = case.get("crossing") or []
    return (
        [("state", k) for k in range(n) if k not in crossing]
        + [("pair", i) for i in range(len(case["pairs"]))]
        + [("seam", i) for i in range(len(case.get("seams", [])))]
    )


def floor_of(sigma: float, h: float) -> float:
    """FLOOR_FACTOR times the finer difference's noise (see the docstring)."""
    _, half, quarter = stencil.steps(h)
    return (
        FLOOR_FACTOR * sigma * (stencil.noise_gain(half) + stencil.noise_gain(quarter))
    )


def symmetry_sigma(ladder: list, irreps: list[str], h: float) -> float:
    """sigma_g implied by the elements a point group forbids: H[i][j] with
    q_i and q_j of different irreps is zero by symmetry, so each rung's value
    there is the stencil's noise, sigma_g times stencil.noise_gain(s). The RMS
    of value / gain(s) over those elements and the three rungs."""
    n = len(irreps)
    scaled = [
        ladder[k][i][j] / stencil.noise_gain(s)
        for k, s in enumerate(stencil.steps(h))
        for i in range(n)
        for j in range(n)
        if irreps[i] != irreps[j]
    ]
    return math.sqrt(math.fsum(x * x for x in scaled) / len(scaled))


def analyse(
    case: dict, coords: dict, result: dict, where: dict, schemes: list[str], h: float
) -> dict:
    """Every quantity of @p schemes over one driver run at the ladder's h."""
    name = case["name"]
    disps = stencil.displacements(h)
    vectors = coords["vectors"]
    n = len(vectors)
    nstates = len(result["reference"]["casscf_energies"])
    pts = result["points"]
    failures = []
    out = {}
    for scheme in schemes:
        part = {"states": [], "pairs": []}
        if case.get("seams"):
            part["seams"] = []
        skipped = case.get("not_laddered", {}).get(scheme, {})
        for slot, (kind, index) in enumerate(quantities(case, nstates)):
            if (kind == "state" and index in skipped.get("states", [])) or (
                kind == "pair" and case["pairs"][index] in skipped.get("pairs", [])
            ):
                continue

            def grad(p: int, slot: int = slot, scheme: str = scheme) -> list[float]:
                return pts[p][scheme]["gradients"][slot]

            def projected(g: list[float]) -> list[float]:
                return [dot(q, g) for q in vectors]

            g0 = grad(where["reference"])
            spread = [
                [x - y for x, y in zip(grad(a), grad(b), strict=True)]
                for a, b in where["noise"]
            ]
            restarts = math.sqrt(
                math.fsum(x * x for d in spread for x in d) / (len(spread) * len(g0))
            )
            points = [
                {d: projected(grad(where["columns"][j][d])) for d in disps}
                for j in range(n)
            ]
            columns = [stencil.column_ladder(points[j], h) for j in range(n)]
            ladder = [
                [[columns[j][k][i] for j in range(n)] for i in range(n)]
                for k in range(3)
            ]
            # A symmetry-adapted case's sigma_g is also read off the elements
            # its point group forbids (the docstring's noise floor).
            forbidden = (
                symmetry_sigma(ladder, coords["irreps"], h)
                if "irreps" in coords
                else None
            )
            sigma = max(restarts, forbidden or 0.0)
            floor = floor_of(sigma, h)
            verdicts = [
                [
                    stencil.judge(
                        [ladder[k][i][j] for k in range(3)], floor, RATIO_WINDOW
                    )
                    for j in range(n)
                ]
                for i in range(n)
            ]
            bar = [[v["bar"] for v in row] for row in verdicts]
            finest = ladder[-1]
            asym = stencil.asymmetry(finest)
            label = f"{name} {scheme} {kind} {index}"
            judged = []
            for i, row in enumerate(verdicts):
                for j, v in enumerate(row):
                    if v["settled"] is False:
                        shown = (
                            "unjudged" if v["ratio"] is None else f"{v['ratio']:.3f}"
                        )
                        failures.append(
                            f"{label}: H[{i}][{j}] ratio {shown} (ladder "
                            f"{ladder[0][i][j]:.9e} {ladder[1][i][j]:.9e} "
                            f"{ladder[2][i][j]:.9e}; D(h)-D(h/2) {v['upper']:.3e}, "
                            f"D(h/2)-D(h/4) {v['lower']:.3e}, floor {floor:.1e})"
                        )
                    elif v["ratio"] is not None:
                        judged.append(v["ratio"])
            misses = stencil.symmetry_misses(finest, bar)
            for i, j, value, allowed in misses:
                failures.append(
                    f"{label}: |H[{i}][{j}] - H[{j}][{i}]| {value:.3e} "
                    f"> bar {allowed:.3e}"
                )
            print(
                f"{label}: sigma_g {sigma:.1e} (restarts {restarts:.1e}, "
                f"forbidden {forbidden or 0.0:.1e}), floor {floor:.1e}, "
                f"{len(judged)} ratios judged {min(judged, default=0):.3f}.."
                f"{max(judged, default=0):.3f}, max bar {max(map(max, bar)):.2e}, "
                f"max asymmetry {max(map(max, asym)):.2e}, "
                f"{len(misses)} asymmetric pairs",
                flush=True,
            )
            entry = {
                "gradient": g0,
                "hessian": stencil.symmetrized(finest),
                "bar": stencil.symmetric_bar(bar),
                "ladder": ladder,
                "ratios": [[v["ratio"] for v in row] for row in verdicts],
                "column_bar": bar,
                "asymmetry": asym,
                "noise": {
                    "sigma_g": sigma,
                    "floor": floor,
                    "pairs": where["noise"],
                    "differences": spread,
                    **(
                        {"sigma_restarts": restarts, "sigma_forbidden": forbidden}
                        if forbidden is not None
                        else {}
                    ),
                },
                "ratio_range": [min(judged), max(judged)] if judged else None,
                "ratios_judged": len(judged),
                "worst_ratio_deviation": max(
                    (abs(r - stencil.TARGET) for r in judged), default=None
                ),
                "points": [
                    {"column": j, "gradients": [points[j][d] for d in disps]}
                    for j in range(n)
                ],
            }
            if kind == "pair":
                entry["pair"] = case["pairs"][index]
                part["pairs"].append(entry)
            elif kind == "seam":
                entry["seam"] = case["seams"][index]
                fixed = result["reference"][scheme]["seams"][index]
                assert fixed["seam"] == entry["seam"], (fixed["seam"], entry["seam"])
                entry |= {k: fixed[k] for k in ("w", "roots", "sign")}
                part["seams"].append(entry)
            else:
                entry["state"] = index
                part["states"].append(entry)
        out[scheme] = part
    return {"quantities": out, "failures": failures}


def guard(runs: list[tuple[dict, list[str]]]) -> dict:
    """The worst guard values over every point of every run (a run is the
    driver's output and the schemes taken from it); over a crossing, also
    the pair's eigenspace miss and the third state's gap."""
    pts = [p for result, _ in runs for p in result["points"]]
    worst = {
        "prediction": max(p["prediction"] for p in pts),
        "subspace": max(p["subspace"] for p in pts),
        "ci": max(p["ci"] for p in pts),
        "gradient_norm": max(p["gradient_norm"] for p in pts),
        "rerun_gradient_norm": max(p["rerun"]["gradient_norm"] for p in pts),
        "rerun_energy": max(p["rerun"]["energy"] for p in pts),
        # None on a core of one (h4_631g_sa2): no pair to be apart.
        "core_gap": min(
            (p["core_gap"] for p in pts if p["core_gap"] is not None), default=None
        ),
        "virtual_gap": min(p["virtual_gap"] for p in pts),
        "points": len(pts),
    }
    for result, schemes in runs:
        for scheme in schemes:
            run = result["points"]
            worst[f"{scheme}_vectors"] = max(p[scheme]["vector_miss"] for p in run)
            if result.get("crossing"):
                worst[f"{scheme}_pair_miss"] = max(p[scheme]["pair_miss"] for p in run)
                # What a u_K match would have seen: the worst, over points
                # and the pair's K, of 1 - the larger |<R_K|u>| over the
                # pair's two displaced states.
                worst[f"{scheme}_pair_rotation"] = max(
                    pair_rotation(
                        result["reference"][scheme]["vectors"],
                        p[scheme]["vectors"],
                        result["crossing"],
                    )
                    for p in run
                )
                worst[f"{scheme}_third_state_gap"] = min(
                    third_state_gap(p[scheme]["energies"], result["crossing"])
                    for p in run
                )
            # Each at its own tolerance: the noise's looser solves apart.
            for key, tol in (
                ("zvector_residual", ZVECTOR_TOL),
                ("loose_zvector_residual", ZVECTOR_LOOSE),
            ):
                worst[f"{scheme}_{key}"] = max(
                    max(p[scheme]["zvector_residuals"])
                    for p in run
                    if p["zvector_tolerance"] == tol
                )
    return worst


def pair_rotation(reference: list[float], vectors: list[float], crossing: list[int]):
    """1 - min over the crossing's K of max over its two displaced states u
    of |<R_K|u>| (n x n, state k at [k n, (k + 1) n)): 0 for a pair that
    did not turn inside its eigenspace, up to 1 - 1/sqrt 2."""
    n = math.isqrt(len(reference))

    def column(m: list[float], k: int) -> list[float]:
        return m[k * n : (k + 1) * n]

    return 1.0 - min(
        max(abs(dot(column(reference, big), column(vectors, k))) for k in crossing)
        for big in crossing
    )


def seam_weight(vectors: list[float], heff: list[float], x: int, y: int) -> dict:
    """The driver's seam W for (x, y) (its seam_weight): from the reference's
    eigenvectors (state k at [k n, (k + 1) n)) and its symmetrized H_eff
    (n x n row-major), W = 1/2 s (V_0 V_1^T + V_1 V_0^T), n x n row-major,
    V the Loewdin-orthonormalized projections of CASSCF roots r_0 < r_1 onto
    span{R_x, R_y} and s the sign of H_eff[r_0][r_1]."""
    n = math.isqrt(len(heff))
    u0, u1 = vectors[x * n : (x + 1) * n], vectors[y * n : (y + 1) * n]
    dets = {
        (i, j): abs(u0[i] * u1[j] - u0[j] * u1[i])
        for i in range(n)
        for j in range(i + 1, n)
    }
    r0, r1 = max(dets, key=dets.get)
    a = [[u0[r0], u0[r1]], [u1[r0], u1[r1]]]
    s = [
        [math.fsum(a[k][i] * a[k][j] for k in range(2)) for j in range(2)]
        for i in range(2)
    ]
    root = math.sqrt(s[0][0] * s[1][1] - s[0][1] ** 2)
    norm = math.sqrt(s[0][0] + s[1][1] + 2 * root)
    t00, t11 = s[0][0] + root, s[1][1] + root
    inv = norm / (t00 * t11 - s[0][1] ** 2)
    half = [[t11 * inv, -s[0][1] * inv], [-s[0][1] * inv, t00 * inv]]
    m = [
        [math.fsum(a[x_][u] * half[u][b] for u in range(2)) for b in range(2)]
        for x_ in range(2)
    ]
    v = [[u0[i] * m[0][b] + u1[i] * m[1][b] for i in range(n)] for b in range(2)]
    sign = -1.0 if heff[r0 * n + r1] < 0 else 1.0
    w = [
        0.5 * sign * (v[0][i] * v[1][j] + v[1][i] * v[0][j])
        for i in range(n)
        for j in range(n)
    ]
    return {"seam": [x, y], "roots": [r0, r1], "sign": sign, "w": w}


def third_state_gap(energies: list[float], crossing: list[int]) -> float:
    """The smallest |E_S - E_X| over the crossing pair's X and every other
    state S."""
    return min(
        abs(energies[s] - energies[x])
        for x in crossing
        for s in range(len(energies))
        if s not in crossing
    )


def gaps(values: list[float]) -> list[float]:
    return [b - a for a, b in zip(values, values[1:], strict=False)]


def agreement(case: dict, runs: list[tuple[dict, list[str]]]) -> dict:
    """The reference point's energies against qd_nevpt2.json's goldens (the
    CASSCF roots, SC's and PC's H_eff eigenvalues), or, for a case with its
    own `system`, against qd_weight_search.py's PySCF point at its weights
    (the CASSCF roots and SC's), or, for one marked `pyscf`, against
    qd_pyscf_point.py's (the CASSCF roots, SC's and PC's): a different CASSCF
    solution misses by millihartree. A crossing case's is the pair's QD
    energies against its MECI's last evaluation, qd_meci.py's record of the
    [optimize] run (which records no CASSCF energies). A symmetric case's is
    its coupling record's reference point, the same driver's (the CASSCF
    roots, SC's and PC's)."""
    if "meci" in case:
        final = meci_entry(case)["final"]
        out = {}
        for result, schemes in runs:
            ref = result["reference"]
            for scheme in schemes:
                lower, upper = sorted(
                    ref[scheme]["energies"][x] for x in case["crossing"]
                )
                out[scheme] = max(
                    abs(lower - final["lower"]), abs(upper - final["upper"])
                )
        return out
    if "coupling" in case:
        # The coupling record's reference point: the same system, run by the
        # same driver (its CASSCF roots and both schemes' H_eff eigenvalues).
        doc = json.loads((OUT / f"{case['coupling']}.json").read_text())
        entry = doc["reference"]
        golden = {scheme: sorted(entry[scheme]["energies"]) for scheme in ("sc", "pc")}
    elif case.get("pyscf"):
        # qd_pyscf_point.py's PySCF point: the CASSCF roots, SC's and PC's.
        entry = json.loads((OUT / f"{case['name']}.pyscf.json").read_text())
        golden = {scheme: entry[f"{scheme}_energies"] for scheme in ("sc", "pc")}
    elif "system" in case:
        doc = json.loads(SEARCH.read_text())
        (entry,) = [
            c for c in doc["candidates"] if c["weights"] == case["system"]["weights"]
        ]
        golden = {"sc": entry["sc_energies"]}
    else:
        doc = json.loads(SOURCE.read_text())
        (entry,) = [c for c in doc["cases"] if c["name"] == case["source"]]
        (pc,) = [c for c in doc["pc"]["cases"] if c["name"] == case["source"]]
        golden = {"sc": entry["eigenvalues"], "pc": pc["eigenvalues"]}
    out = {
        "casscf": max(
            abs(a - b)
            for result, _ in runs
            for a, b in zip(
                result["reference"]["casscf_energies"],
                entry["casscf_energies"],
                strict=True,
            )
        )
    }
    for result, schemes in runs:
        for scheme in schemes:
            out[scheme] = max(
                abs(a - b)
                for a, b in zip(
                    result["reference"][scheme]["energies"], golden[scheme], strict=True
                )
            )
    return out


def record(
    case: dict,
    sys_: dict,
    coords: dict,
    runs: list[tuple[dict, list[str]]],
    quantities_: dict,
) -> dict:
    ref = runs[0][0]["reference"]
    # Each scheme's H_eff from the run that differenced it.
    heff = {
        scheme: result["reference"][scheme]
        for result, schemes in runs
        for scheme in schemes
    }
    basis_file = BASIS_FILES[sys_["basis"]]
    return {
        "description": "Semi-numerical SC- and PC-QD-NEVPT2 nuclear Hessians "
        "(DF-SA-CASSCF reference): a fourth-order central stencil at steps h, "
        "h/2, h/4 over this tree's analytic QD gradient G(W) at the tracked "
        "W^K = u_K u_K^T (each state) and W^P + W^Q (each pair), every point "
        "re-converged through the continuity guard. Hartree, bohr; "
        "H[i][j] = q_i . dG/dq_j. tools/nevpt2_hessian/qd_nevpt2_hessian.py "
        "over free-breakfast-qd-nevpt2-hessian",
        "provenance": {
            "command": "tools/nevpt2_hessian/qd_nevpt2_hessian.py",
            "driver": "tools/qd_nevpt2_hessian/main.cpp",
            "source": (
                f"{case['coupling']}.json (the coupling case's system)"
                if "coupling" in case
                else f"{case['name']}.pyscf.json (tools/nevpt2_hessian/"
                "qd_pyscf_point.py)"
                if case.get("pyscf")
                else "tools/nevpt2_hessian/qd_weight_search.py"
                if "system" in case
                else f"{MECI.name} {case['meci']} (tools/nevpt2_hessian/qd_meci.py)"
                if "meci" in case
                else f"test/gtest/data/qd_nevpt2/qd_nevpt2.json {case['source']}"
            ),
        },
        "name": case["name"],
        "basis": {
            "name": sys_["basis"],
            "file": basis_file,
            "sha256": sha256(basis_file),
        },
        "aux_basis": {"name": AUX[0], "file": AUX[1], "sha256": sha256(AUX[1])},
        "atoms": sys_["atoms"],
        "ncas": sys_["ncas"],
        "nelecas": sys_["nelecas"],
        "active": sys_["active"],
        "weights": sys_["weights"],
        "schemes": list(case["h"]),
        "pairs": case["pairs"],
        **(
            {
                "crossing": case["crossing"],
                "seams": case["seams"],
                "meci": meci_summary(case),
            }
            if "crossing" in case
            else {}
        ),
        "reference": {
            "casscf_energies": ref["casscf_energies"],
            "casscf_gaps": gaps(ref["casscf_energies"]),
            "gradient_norm": ref["gradient_norm"],
            "inactive_gaps": {"core": ref["core_gap"], "virtual": ref["virtual_gap"]},
            "pc_dropped": ref["pc_dropped"],
            "agreement": agreement(case, runs),
            **{
                scheme: {
                    "energies": heff[scheme]["energies"],
                    "heff_gaps": gaps(heff[scheme]["energies"]),
                    "heff": heff[scheme]["heff"],
                    "vectors": heff[scheme]["vectors"],
                    **(
                        {
                            "pair_gap": abs(
                                heff[scheme]["energies"][case["crossing"][1]]
                                - heff[scheme]["energies"][case["crossing"][0]]
                            ),
                            "third_state_gap": third_state_gap(
                                heff[scheme]["energies"], case["crossing"]
                            ),
                        }
                        if "crossing" in case
                        else {}
                    ),
                }
                for scheme in case["h"]
            },
        },
        "steps": {
            scheme: {
                "h": h,
                "ladder": stencil.steps(h),
                "displacements": stencil.displacements(h),
                # The coordinates the scheme ladders, row and column.
                "coordinates": scheme_labels(case, scheme, coords),
            }
            for scheme, h in case["h"].items()
        },
        "stencil": {"offsets": list(stencil.OFFSETS), "weights": list(stencil.WEIGHTS)},
        "coordinates": coords,
        "tolerances": {
            "ratio_window": list(RATIO_WINDOW),
            "floor_factor": FLOOR_FACTOR,
            "zvector": ZVECTOR_TOL,
            "zvector_loose": ZVECTOR_LOOSE,
            "noise_columns": NOISE_COLUMNS,
        },
        "guard": guard(runs),
        **({"not_laddered": case["not_laddered"]} if "not_laddered" in case else {}),
        **quantities_,
    }


def meci_summary(case: dict) -> dict:
    """What the crossing case's geometry is: qd_meci.py's record of the
    [optimize] MECI, its end point and the criteria it met."""
    entry = meci_entry(case)
    return {
        "variant": entry["variant"],
        "record": MECI.name,
        "steps": entry["steps"],
        "criteria": entry["criteria"],
        "final": entry["final"],
    }


def crossing_failures(case: dict, runs: list[tuple[dict, list[str]]]) -> list[str]:
    """A crossing case's own refusals: the third state within ROOT_GAP_FLOOR
    of the pair at any point."""
    out = []
    for result, schemes in runs:
        for scheme in schemes:
            for index, p in enumerate(result["points"]):
                gap = third_state_gap(p[scheme]["energies"], case["crossing"])
                if not gap > ROOT_GAP_FLOOR:
                    out.append(
                        f"{case['name']} {scheme}: point {index}'s third state is "
                        f"{gap:.3e} from the crossing pair, not above {ROOT_GAP_FLOOR}"
                    )
    return out


def dumps(value, depth: int = 0) -> str:
    """JSON with objects laid out key by key down to depth 3 and every array
    on one line: a file of 30 000 numbers stays readable and a third of
    json.dumps(indent=1)'s size."""
    if isinstance(value, dict) and depth < 3 and value:
        pad = " " * (depth + 1)
        body = ",\n".join(
            f"{pad}{json.dumps(k)}: {dumps(v, depth + 1)}" for k, v in value.items()
        )
        return "{\n" + body + "\n" + " " * depth + "}"
    if isinstance(value, list) and depth < 3 and value and isinstance(value[0], dict):
        pad = " " * (depth + 1)
        body = ",\n".join(f"{pad}{dumps(v, depth + 1)}" for v in value)
        return "[\n" + body + "\n" + " " * depth + "]"
    return json.dumps(value, separators=(",", ":"))


# ---------------------------------------------------- the C2v coupling case

# C2v on Cartesian displacements, the C2 axis z: each operation's action on
# (x, y, z), and each irrep's characters over (E, C2, sigma_xz, sigma_yz).
C2V_OPERATIONS = ((1, 1, 1), (-1, -1, 1), (1, -1, 1), (-1, 1, 1))
C2V_CHARACTERS = {
    "A1": (1, 1, 1, 1),
    "A2": (1, 1, -1, -1),
    "B1": (1, -1, 1, -1),
    "B2": (1, -1, -1, 1),
}
# A vector shorter than this after projection is not in the irrep.
SALC_TOL = 1e-8


def normalized(v: list[float]) -> list[float]:
    norm = math.sqrt(dot(v, v))
    return [x / norm for x in v]


def symmetry_coordinates(atoms: list) -> dict:
    """An orthonormal basis of the 3N Cartesian displacements adapted to C2v
    (the C2 axis z through the origin): per irrep A1, A2, B1, B2, first the
    translations and rotations (about the centroid) it holds, labelled T_x ..
    R_z, then its vibrations, "<irrep> k", the projections of the Cartesian
    unit vectors in order, each orthogonalized against those before it."""
    pos = [a[1] for a in atoms]
    n = 3 * len(atoms)

    def image(op: tuple, v: list[float]) -> list[float]:
        out = [0.0] * n
        for i, p in enumerate(pos):
            q = [op[k] * p[k] for k in range(3)]
            (j,) = [
                j
                for j, r in enumerate(pos)
                if atoms[j][0] == atoms[i][0]
                and max(abs(q[k] - r[k]) for k in range(3)) < 1e-9
            ]
            for k in range(3):
                out[3 * j + k] = op[k] * v[3 * i + k]
        return out

    def project(irrep: str, v: list[float]) -> list[float]:
        images = [image(op, v) for op in C2V_OPERATIONS]
        chars = C2V_CHARACTERS[irrep]
        return [
            sum(c * w[m] for c, w in zip(chars, images, strict=True)) / 4.0
            for m in range(n)
        ]

    centre = [sum(p[k] for p in pos) / len(pos) for k in range(3)]
    rigid = []
    for k, axis in enumerate("xyz"):
        rigid.append((f"T_{axis}", [1.0 if m % 3 == k else 0.0 for m in range(n)]))
    for k, axis in enumerate("xyz"):
        e = [1.0 if m == k else 0.0 for m in range(3)]
        v = []
        for p in pos:
            r = [p[m] - centre[m] for m in range(3)]
            v += [
                e[1] * r[2] - e[2] * r[1],
                e[2] * r[0] - e[0] * r[2],
                e[0] * r[1] - e[1] * r[0],
            ]
        rigid.append((f"R_{axis}", v))
    labels, irreps, vectors = [], [], []
    for irrep in C2V_CHARACTERS:
        basis = []
        candidates = [(label, v) for label, v in rigid] + [
            (None, [1.0 if m == j else 0.0 for m in range(n)]) for j in range(n)
        ]
        count = 0
        for label, v in candidates:
            w = project(irrep, v)
            for b in basis:
                c = dot(b, w)
                w = [x - c * y for x, y in zip(w, b, strict=True)]
            if math.sqrt(dot(w, w)) < SALC_TOL * math.sqrt(dot(v, v)):
                continue
            if label is None:
                count += 1
                label = f"{irrep} {count}"
            basis.append(normalized(w))
            labels.append(label)
            irreps.append(irrep)
        vectors += basis
    assert len(vectors) == n, (len(vectors), n)
    return {"labels": labels, "irreps": irreps, "vectors": vectors}


def coupling_directions(case: dict, coords: dict) -> list[tuple[str, list[float]]]:
    """The directions C_MN is laddered along: each coupled coordinate, then
    each pair's (q_a + q_b) / sqrt 2."""
    by_label = dict(zip(coords["labels"], coords["vectors"], strict=True))
    coupled = case["coupled"]
    out = [(label, by_label[label]) for label in coupled]
    for a in range(len(coupled)):
        for b in range(a + 1, len(coupled)):
            qa, qb = by_label[coupled[a]], by_label[coupled[b]]
            out.append(
                (
                    f"({coupled[a]} + {coupled[b]})/sqrt2",
                    [(x + y) / math.sqrt(2.0) for x, y in zip(qa, qb, strict=True)],
                )
            )
    return out


def plan_coupling(h: float, directions: list) -> tuple[list[dict], dict]:
    """The request's points: index 0 the reference, then per direction its
    eight displacements; then the noise, each positive displacement solved
    again from the orbitals of its negative image (a different start)."""
    disps = stencil.displacements(h)
    dim = len(directions[0][1])
    points = [{"displacement": [0.0] * dim, "from": None}]
    where = {"reference": 0, "directions": [], "noise": []}
    for _, q in directions:
        row = {}
        for d in disps:
            row[d] = len(points)
            points.append({"displacement": [d * x for x in q], "from": None})
        where["directions"].append(row)
    for row in where["directions"]:
        for d in disps:
            if d > 0.0:
                where["noise"].append((len(points), row[d]))
                points.append(
                    {"displacement": points[row[d]]["displacement"], "from": row[-d]}
                )
    return points, where


def coupling_request(case: dict, schemes: list[str], points: list[dict]) -> dict:
    return {
        "system": case["system"],
        "schemes": schemes,
        "gradients": False,
        "zvector_tolerance": ZVECTOR_TOL,
        "pairs": [],
        "seams": [],
        "points": points,
    }


def element(point: dict, scheme: str, pair: list[int]) -> float:
    """H_eff[M][N] at a point, in the reference's root basis (the driver's
    `heff`, symmetrized)."""
    heff = point[scheme]["heff"]
    n = math.isqrt(len(heff))
    return heff[pair[0] * n + pair[1]]


def analyse_coupling(
    case: dict, result: dict, where: dict, directions: list, scheme: str, h: float
) -> tuple[list[dict], list[str]]:
    """Per state pair (M, N): C_MN's ladder of first derivatives along each
    direction, judged as the Hessians' elements are, and the bilinearity of
    each coupled pair: D(u) against (D(q_a) + D(q_b)) / sqrt 2 at the finest
    rung, u = (q_a + q_b) / sqrt 2. Returns the entries and the failures."""
    disps = stencil.displacements(h)
    pts = result["points"]
    failures = []
    out = []
    coupled = case["coupled"]
    for pair in case["state_pairs"]:
        label = f"{case['name']} {scheme} C_{pair[0]}{pair[1]}"
        spread = [
            element(pts[a], scheme, pair) - element(pts[b], scheme, pair)
            for a, b in where["noise"]
        ]
        sigma = math.sqrt(math.fsum(x * x for x in spread) / len(spread))
        floor = floor_of(sigma, h)
        lines = []
        for (name, q), row in zip(directions, where["directions"], strict=True):
            values = {d: element(pts[row[d]], scheme, pair) for d in disps}
            ladder = [stencil.derivative(values, s) for s in stencil.steps(h)]
            verdict = stencil.judge(ladder, floor, RATIO_WINDOW)
            if verdict["settled"] is False:
                shown = (
                    "unjudged"
                    if verdict["ratio"] is None
                    else f"{verdict['ratio']:.3f}"
                )
                failures.append(
                    f"{label} along {name}: ratio {shown} (ladder {ladder[0]:.9e} "
                    f"{ladder[1]:.9e} {ladder[2]:.9e}; D(h)-D(h/2) "
                    f"{verdict['upper']:.3e}, D(h/2)-D(h/4) {verdict['lower']:.3e}, "
                    f"floor {floor:.1e})"
                )
            lines.append(
                {
                    "direction": name,
                    "vector": q,
                    "values": [values[d] for d in disps],
                    "ladder": ladder,
                    "ratio": verdict["ratio"],
                    "settled": verdict["settled"],
                    "bar": verdict["bar"],
                }
            )
        by_name = {line["direction"]: line for line in lines}
        checks = []
        for a in range(len(coupled)):
            for b in range(a + 1, len(coupled)):
                la, lb = by_name[coupled[a]], by_name[coupled[b]]
                lu = by_name[f"({coupled[a]} + {coupled[b]})/sqrt2"]
                residuals = [
                    lu["ladder"][k]
                    - (la["ladder"][k] + lb["ladder"][k]) / math.sqrt(2.0)
                    for k in range(3)
                ]
                allowed = lu["bar"] + (la["bar"] + lb["bar"]) / math.sqrt(2.0)
                checks.append(
                    {
                        "coordinates": [coupled[a], coupled[b]],
                        "residuals": residuals,
                        "residual": residuals[-1],
                        "allowed": allowed,
                        "fails": abs(residuals[-1]) > allowed,
                    }
                )
        print(
            f"{label}: sigma {sigma:.1e}, floor {floor:.1e}, "
            + ", ".join(
                f"{line['direction']} {line['ladder'][-1]:.6e} (bar {line['bar']:.1e})"
                for line in lines
            )
            + ", "
            + ", ".join(
                f"bilinearity {c['coordinates']} {c['residual']:.2e} of "
                f"{c['allowed']:.2e}"
                for c in checks
            ),
            flush=True,
        )
        out.append(
            {
                "pair": pair,
                "directions": lines,
                "bilinearity": checks,
                "noise": {
                    "sigma": sigma,
                    "floor": floor,
                    "pairs": where["noise"],
                    "differences": spread,
                },
            }
        )
    return out, failures


def pc_probe(binary: pathlib.Path, case: dict, h: float, coords: dict) -> dict:
    """Both schemes at the reference and at +h along the first coupled
    coordinate: the reference's PC H_eff, and the sampler's verdict on PC
    at the displaced point (a refused point fails the run; its message is
    recorded)."""
    by_label = dict(zip(coords["labels"], coords["vectors"], strict=True))
    q = by_label[case["coupled"][0]]
    zero = [0.0] * len(q)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for key, points in (
            ("reference", [{"displacement": zero, "from": None}]),
            (
                "displaced",
                [
                    {"displacement": zero, "from": None},
                    {"displacement": [h * x for x in q], "from": None},
                ],
            ),
        ):
            request = pathlib.Path(tmp) / f"{key}.request.json"
            answer = pathlib.Path(tmp) / f"{key}.points.json"
            request.write_text(json.dumps(coupling_request(case, ["sc", "pc"], points)))
            run = subprocess.run(
                [str(binary), str(request), str(answer)],
                capture_output=True,
                text=True,
                check=False,
            )
            lines = [line for line in run.stderr.splitlines() if line.strip()]
            out[key] = {
                "displacement": h if key == "displaced" else 0.0,
                "returncode": run.returncode,
                "message": lines[-1] if lines else "",
                "result": json.loads(answer.read_text())
                if run.returncode == 0
                else None,
            }
    return out


def coupling_guard(result: dict, schemes: list[str]) -> dict:
    pts = result["points"]
    worst = {
        "prediction": max(p["prediction"] for p in pts),
        "subspace": max(p["subspace"] for p in pts),
        "ci": max(p["ci"] for p in pts),
        "gradient_norm": max(p["gradient_norm"] for p in pts),
        # None on a core of one (h4_631g_sa2): no pair to be apart.
        "core_gap": min(
            (p["core_gap"] for p in pts if p["core_gap"] is not None), default=None
        ),
        "virtual_gap": min(p["virtual_gap"] for p in pts),
        "points": len(pts),
    }
    for scheme in schemes:
        worst[f"{scheme}_vectors"] = max(p[scheme]["vector_miss"] for p in pts)
    return worst


def analysis_path(out: pathlib.Path, name: str) -> pathlib.Path:
    return out / f"{name}.analysis.json"


def pc_expected(analysis: dict) -> list[dict]:
    """PC's expected-refused blocks (nevpt2.tex's symmetric geometries): a
    state pair and two coordinates along both of which a null mode of some
    ket's metric lifts at second order with a first-order overlap
    <phi|V|M> to the other state (qd_coupling_analysis.py's verdicts)."""
    lifted = analysis["pc"]["lifted_coordinates"]
    out = []
    for a in range(len(lifted)):
        for b in range(a, len(lifted)):
            out.append(
                {
                    "states": analysis["pc"]["state_pair"],
                    "coordinates": [lifted[a], lifted[b]],
                }
            )
    return out


def write_coupling(case: dict, args) -> bool:
    """One coupling case: the ladders of C_MN over one driver run per scheme
    and step, the PC probe, and the record. False (STUCK, nothing written)
    when a ladder does not settle."""
    name = case["name"]
    started = time.perf_counter()
    coords = symmetry_coordinates(case["system"]["atoms"])
    directions = coupling_directions(case, coords)
    analysis_file = analysis_path(args.out, name)
    if not analysis_file.exists():
        print(
            f"STUCK: {name}: {analysis_file} is missing; run "
            "qd_coupling_analysis.py first",
            flush=True,
        )
        return False
    analysis = json.loads(analysis_file.read_text())
    assert analysis["coordinates"] == coords, "the analysis's coordinates differ"
    runs = {}
    parts = {}
    failures = []
    for scheme, h in case["h"].items():
        points, where = plan_coupling(h, directions)
        file = f"{name}.{scheme}.h{h:g}.points.json"
        if args.points and not args.keep:
            path = args.points / file
        else:
            directory = args.points or pathlib.Path(tempfile.mkdtemp())
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / file
            run_driver(args.binary, coupling_request(case, [scheme], points), path)
        result = json.loads(path.read_text())
        assert len(result["points"]) == len(points), path
        entries, missed = analyse_coupling(case, result, where, directions, scheme, h)
        runs[scheme] = result
        parts[scheme] = entries
        failures += missed
    print(f"{name}: {time.perf_counter() - started:.0f} s", flush=True)
    if failures:
        print(f"STUCK: {name}", flush=True)
        for line in failures:
            print(f"  {line}", flush=True)
        return False
    probe_file = (args.points / f"{name}.pc_probe.json") if args.points else None
    if probe_file is not None and not args.keep and probe_file.exists():
        probe = json.loads(probe_file.read_text())
    else:
        probe = pc_probe(args.binary, case, min(case["h"].values()), coords)
        if probe_file is not None:
            probe_file.write_text(json.dumps(probe))
    ref = next(iter(runs.values()))["reference"]
    pc_ref = probe["reference"]["result"]["reference"]
    tree = {
        "casscf": ref["casscf_energies"],
        "sc": ref["sc"]["heff"],
        "pc": pc_ref["pc"]["heff"],
    }
    python = analysis["reference"]
    agreement = {
        key: max(abs(a - b) for a, b in zip(tree[key], python[key], strict=True))
        for key in tree
    }
    expected = {
        "sc": [
            {"states": entry["pair"], "coordinates": check["coordinates"]}
            for entry in parts["sc"]
            for check in entry["bilinearity"]
            if check["fails"]
        ],
        "pc": pc_expected(analysis),
    }
    basis_file = BASIS_FILES[case["system"]["basis"]]
    record = {
        "description": "C_MN = H_eff[M][N] of SC-QD-NEVPT2 (DF-SA-CASSCF "
        "reference) at a C2v geometry whose two roots are 1A1 and 1B2: per "
        "state pair the fourth-order central stencil at h, h/2, h/4 of C_MN's "
        "first derivative along each coupled b2 coordinate and along their "
        "sum, every point re-converged through the continuity guard, the "
        "bilinearity of each pair, and the blocks the analytic Hessians are "
        "expected to refuse, per variant. Hartree, bohr. "
        "tools/nevpt2_hessian/qd_nevpt2_hessian.py over "
        "free-breakfast-qd-nevpt2-hessian (gradients false)",
        "provenance": {
            "command": "tools/nevpt2_hessian/qd_nevpt2_hessian.py",
            "driver": "tools/qd_nevpt2_hessian/main.cpp",
            "analysis": f"{analysis_file.name} "
            "(tools/nevpt2_hessian/qd_coupling_analysis.py)",
        },
        "name": name,
        "basis": {
            "name": case["system"]["basis"],
            "file": basis_file,
            "sha256": sha256(basis_file),
        },
        "aux_basis": {"name": AUX[0], "file": AUX[1], "sha256": sha256(AUX[1])},
        **{k: case["system"][k] for k in ("atoms", "ncas", "nelecas", "active")},
        "weights": case["system"]["weights"],
        "irreps": case["irreps"],
        "schemes": list(case["h"]),
        "state_pairs": case["state_pairs"],
        "coupled": case["coupled"],
        "coordinates": coords,
        "reference": {
            "casscf_energies": ref["casscf_energies"],
            "casscf_gaps": gaps(ref["casscf_energies"]),
            "gradient_norm": ref["gradient_norm"],
            "inactive_gaps": {"core": ref["core_gap"], "virtual": ref["virtual_gap"]},
            "sc": {
                "energies": ref["sc"]["energies"],
                "heff_gaps": gaps(ref["sc"]["energies"]),
                "heff": ref["sc"]["heff"],
                "vectors": ref["sc"]["vectors"],
            },
            "pc": {
                "energies": pc_ref["pc"]["energies"],
                "heff": pc_ref["pc"]["heff"],
                "dropped": pc_ref["pc_dropped"],
            },
            "agreement": agreement,
        },
        "steps": {
            scheme: {
                "h": h,
                "ladder": stencil.steps(h),
                "displacements": stencil.displacements(h),
            }
            for scheme, h in case["h"].items()
        },
        "stencil": {"offsets": list(stencil.OFFSETS), "weights": list(stencil.WEIGHTS)},
        "tolerances": {
            "ratio_window": list(RATIO_WINDOW),
            "floor_factor": FLOOR_FACTOR,
        },
        "guard": {
            scheme: coupling_guard(result, [scheme]) for scheme, result in runs.items()
        },
        "pc_probe": {
            key: {k: v for k, v in value.items() if k != "result"}
            for key, value in probe.items()
        },
        **parts,
        "expected_refusals": expected,
        "guarded_tuple": analysis["sc"]["chosen"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{name}.json"
    out.write_text(dumps(record) + "\n")
    print(f"wrote {out}", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=OUT, help="directory")
    parser.add_argument("--only", nargs="+", help="these cases alone")
    parser.add_argument("--binary", type=pathlib.Path, default=BINARY)
    parser.add_argument(
        "--points",
        type=pathlib.Path,
        help="a directory of the driver's <case>.h<h>.points.json: read them "
        "back instead of running it, or, with --keep, where the run leaves them",
    )
    parser.add_argument(
        "--keep", action="store_true", help="run, and keep the points in --points"
    )
    args = parser.parse_args()
    status = 0
    for case in CASES + CROSSING_CASES + SYMMETRIC_CASES:
        if args.only and case["name"] not in args.only:
            continue
        started = time.perf_counter()
        sys_ = system(case)
        symbols = [a[0] for a in sys_["atoms"]]
        coords = (
            symmetry_coordinates(sys_["atoms"])
            if "coupling" in case
            else coordinates(len(sys_["atoms"]), symbols)
        )
        # One driver run per distinct h and columns, over the schemes that
        # take them.
        by_h: dict[tuple, list[str]] = {}
        for scheme, h in case["h"].items():
            labels = tuple(scheme_labels(case, scheme, coords))
            by_h.setdefault((h, labels), []).append(scheme)
        runs = []
        quantities_ = {}
        failures = []
        for (h, labels), schemes in by_h.items():
            columns = laddered(coords, labels)
            points, where = plan(h, columns)
            file = f"{case['name']}.h{h:g}.points.json"
            if args.points and not args.keep:
                path = args.points / file
            else:
                directory = args.points or pathlib.Path(tempfile.mkdtemp())
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / file
                request = {
                    "system": sys_,
                    "schemes": schemes,
                    "gradients": True,
                    "zvector_tolerance": ZVECTOR_TOL,
                    "pairs": case["pairs"],
                    "seams": case.get("seams", []),
                    "crossing": case.get("crossing"),
                    "points": points,
                }
                run_driver(args.binary, request, path)
            result = json.loads(path.read_text())
            assert len(result["points"]) == len(points), path
            analysis = analyse(case, columns, result, where, schemes, h)
            runs.append((result, schemes))
            quantities_.update(analysis["quantities"])
            failures += analysis["failures"]
            if "crossing" in case:
                failures += crossing_failures(case, [(result, schemes)])
        print(f"{case['name']}: {time.perf_counter() - started:.0f} s", flush=True)
        if failures:
            print(f"STUCK: {case['name']}", flush=True)
            for line in failures:
                print(f"  {line}", flush=True)
            status = 1
            continue
        args.out.mkdir(parents=True, exist_ok=True)
        out = args.out / f"{case['name']}.json"
        out.write_text(dumps(record(case, sys_, coords, runs, quantities_)) + "\n")
        print(f"wrote {out}", flush=True)
    for case in COUPLING_CASES:
        if args.only and case["name"] not in args.only:
            continue
        if not write_coupling(case, args):
            status = 1
    sys.exit(status)


if __name__ == "__main__":
    main()
