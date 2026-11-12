"""Step 6: the comparison with experiment. Hessians at the hexadiene minima,
a conformer survey, the boat TS, and from them ΔH‡, ΔS‡, ΔG‡ and the
boat-chair gap.

Every run is one free-breakfast config under inputs/<name>/, run into
runs/<name>/, as step 4b's stationary.py. Phases, each resumable:

  prepare <name>   (PySCF)  writes inputs/<name>/: the geometry, the starting
                            orbitals over this tree's AOs (always a
                            free-breakfast CASSCF's, carried with the atoms
                            where they moved; PySCF only evaluates AOs), the
                            config. A run
                            that starts where another ended reads that run's
                            files, so prepare it after that run finished.
  run <name> ...   (GPU)    the binary on each prepared run, its log to
                            runs/<name>/log.txt, its wall clock to wall.txt
                            and the card's memory, sampled with nvidia-smi
                            every 0.5 s, to memory.json. Skips a run done.
  analyse          (PySCF)  every finished run -> results.json,
                            frequencies.csv and geometries/*.xyz; the
                            thermochemistry is thermo.py's.

Same method and thresholds as step 4b throughout: DF-CASSCF(6,6)/cc-pVDZ
(cc-pVDZ-JKFIT) and PC-NEVPT2 on it, Opt=Tight walks at conv_tol_grad 1e-9,
analytic Hessians at 1e-10. No [memory] cap anywhere.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
STEP4 = STUDY / "step_4"
ROOT = STUDY.parent.parent
sys.path.insert(0, str(STEP4))
sys.path.insert(0, str(STUDY / "step_1"))

import energies  # noqa: E402  step 4a's projection and xyz writer
import stationary as s4  # noqa: E402  step 4b's geometry and symmetry helpers

INPUTS = HERE / "inputs"
RUNS = HERE / "runs"
GEOMETRIES = HERE / "geometries"
BOHR = s4.BOHR

# The study's numbering (step_1/cope.py): C1..C6 are atoms 0-5, then H on
# C1 (6, 7), C2 (8), C3 (9, 10), C4 (11, 12), C5 (13), C6 (14, 15).
# Checked by connectivity in analyse (check_connectivity).
H_ON = {0: (6, 7), 1: (8,), 2: (9, 10), 3: (11, 12), 4: (13,), 5: (14, 15)}

# The conformer starts: (C2-C3-C4-C5, C1=C2-C3-C4, C3-C4-C5=C6) in degrees,
# anti/gauche about C3-C4 times skew+-/cis about each C2-C3. Under the
# anti's inversion (phi1, phi2) -> (-phi2, -phi1) and under C2 (phi1, phi2)
# -> (phi2, phi1); a mirror flips every sign. These ten are the classes
# left, each once.
SKEW, CIS = 120.0, 0.0
CONFORMERS = {
    "a_pp": (180.0, SKEW, SKEW),  # anti, C2
    "a_pm": (180.0, SKEW, -SKEW),  # anti, Ci
    "a_pc": (180.0, SKEW, CIS),
    "a_cc": (180.0, CIS, CIS),  # anti, C2h
    "g_pp": (60.0, SKEW, SKEW),  # gauche, C2
    "g_mm": (60.0, -SKEW, -SKEW),  # gauche, C2
    "g_pm": (60.0, SKEW, -SKEW),
    "g_pc": (60.0, SKEW, CIS),
    "g_mc": (60.0, -SKEW, CIS),
    "g_cc": (60.0, CIS, CIS),  # gauche, C2
}

# The numerical Hessian: the tree's NEVPT2-chain defaults (the ladder at
# h = 1e-3 bohr) with no bilinearity pairs: every pair of 48 coordinates is
# past the 12-coordinate default, which runs none and says so; [] says it.
HESSIAN_NUMERICAL = '[hessian]\nmethod = "numerical"\npairs = []\n'
HESSIAN_DIFFERENCE = '[hessian]\nmethod = "numerical"\nladder = false\n'

# The boat start: R16 = R34 (Angstrom), the allyls step 4b's TS ones (boat_c2v).
BOAT_R = 2.0

HEAD = """\
# Step 6, {name}: {what}. Written by experiment.py prepare.
[molecule]
xyz = "geometry.xyz"
[basis_set]
basis = "cc-pvdz"
[eri]
mode = "density_fitting"
aux_basis = "cc-pvdz-jkfit"
[casscf]
ncas = 6
nelecas = 6
orbitals = "orbitals.npy"
conv_tol = {conv_tol}
conv_tol_grad = {conv_tol_grad}
write_npy = true              # casscf_orbitals.npy and casscf_ci.npy
[nevpt2_gradient]
variant = "pc"
"""

# name -> (kind, start). kind:
#   "walk"       an [optimize] over the PC-NEVPT2 gradient
#   "hessian"    the analytic [hessian]
#   "numerical"  the numerical [hessian]: central differences of the
#                analytic PC gradient, on the tree's default ladder (h, h/2,
#                h/4 at h = 1e-3 bohr, fourth-order stencils), no pairs
#   "difference" the numerical [hessian] without the ladder: one central
#                difference of the analytic gradient per coordinate, h = 1e-3
#   "energy"     a single point: CASSCF and [nevpt2], no derivative
# start:
#   float R                     step 1's C2h geometry at R, its PySCF CASSCF
#                               orbitals projected (ts_walk only, as step 4b)
#   ("xyz", file, run)          a committed geometry, `run`'s CASSCF orbitals
#   ("final", run)              a walk's final geometry and orbitals
#   ("mode", hessian, +1 | -1)  step 4b's IRC stand-in start
#   ("conformer", label)        pc_plus with its dihedrals set, orbitals
#                               rotated with the atoms
#   ("boat", R)                 a C2v boat, ts_walk's orbitals carried over
#   ("fd", irrep, delta)        the TS displaced along ts_hessian's lowest
#                               mode of that C2h irrep, largest Cartesian
#                               component delta bohr (signed); ts_hessian's
#                               CASSCF orbitals
KINDS = ("walk", "hessian", "numerical", "difference", "energy")
RUNS_SPEC: dict[str, tuple] = {
    # Step 4b's chain, rerun for the orbitals its outputs held: the TS walk,
    # the TS Hessian (uncapped) at step 4b's committed TS geometry, and the
    # two displaced walks to the hexadiene ends.
    "ts_walk": ("walk", 1.95),
    "ts_hessian": ("hessian", ("xyz", STEP4 / "geometries" / "pc_ts.xyz", "ts_walk")),
    "plus_walk": ("walk", ("mode", "ts_hessian", +1)),
    "minus_walk": ("walk", ("mode", "ts_hessian", -1)),
    # Part 1: the end points step 4b committed, each from its rerun's
    # orbitals. The analytic Hessian refuses them (a degenerate inactive
    # pair, README "The end points"); the numerical one is the result.
    "plus_hessian_analytic": (
        "hessian",
        ("xyz", STEP4 / "geometries" / "pc_plus.xyz", "plus_walk"),
    ),
    "minus_hessian_analytic": (
        "hessian",
        ("xyz", STEP4 / "geometries" / "pc_minus.xyz", "minus_walk"),
    ),
    "plus_hessian": (
        "numerical",
        ("xyz", STEP4 / "geometries" / "pc_plus.xyz", "plus_walk"),
    ),
    "minus_hessian": (
        "numerical",
        ("xyz", STEP4 / "geometries" / "pc_minus.xyz", "minus_walk"),
    ),
    # Part 4: the boat.
    "boat_walk": ("walk", ("boat", BOAT_R)),
    "boat_hessian": ("hessian", ("final", "boat_walk")),
    "boat_hessian_difference": ("difference", ("final", "boat_walk")),
    # Part 5: the CASSCF stationary points of step 4b, each a single point
    # (its free-breakfast CASSCF orbitals, from ts_walk's) and then a PC walk.
    "diyl_sp": ("energy", ("xyz", STEP4 / "geometries" / "casscf_1.65.xyz", "ts_walk")),
    "diyl_walk": ("walk", ("xyz", STEP4 / "geometries" / "casscf_1.65.xyz", "diyl_sp")),
    "aromatic_sp": (
        "energy",
        ("xyz", STEP4 / "geometries" / "casscf_2.30_restart.xyz", "ts_walk"),
    ),
    "aromatic_walk": (
        "walk",
        ("xyz", STEP4 / "geometries" / "casscf_2.30_restart.xyz", "aromatic_sp"),
    ),
}
# Part 2: one walk per conformer start, a Hessian for those analyse needs.
for _label in CONFORMERS:
    RUNS_SPEC[f"conf_{_label}"] = ("walk", ("conformer", _label))
    RUNS_SPEC[f"conf_{_label}_hessian"] = ("difference", ("final", f"conf_{_label}"))
# The lowest conformer (C1, its closest inactive pair 2.7e-4 Eh apart) by the
# analytic Hessian too: the difference Hessians' check.
RUNS_SPEC["conf_g_pm_hessian_analytic"] = ("hessian", ("final", "conf_g_pm"))
# Part 6: the energy-only check of the TS, along its lowest mode of each
# C2h irrep at +-0.01 and +-0.02 bohr, and the TS itself.
FD_IRREPS = ("au", "ag", "bg", "bu")
FD_DELTAS = (0.01, 0.02)
RUNS_SPEC["fd_ts"] = (
    "energy",
    ("xyz", STEP4 / "geometries" / "pc_ts.xyz", "ts_hessian"),
)
for _irrep in FD_IRREPS:
    for _d in FD_DELTAS:
        for _sign, _tag in ((1, "p"), (-1, "m")):
            RUNS_SPEC[f"fd_{_irrep}_{_tag}{round(_d * 100):02d}"] = (
                "energy",
                ("fd", _irrep, _sign * _d),
            )


# --- geometry ------------------------------------------------------------


def dihedral(x, a, b, c, d) -> float:
    """The a-b-c-d dihedral, degrees, (-180, 180]."""
    import numpy

    b0, b1, b2 = x[a] - x[b], x[c] - x[b], x[d] - x[c]
    b1 = b1 / numpy.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    return float(numpy.degrees(numpy.arctan2(numpy.cross(b1, v) @ w, v @ w)))


def dihedrals(x) -> tuple[float, float, float]:
    """(C2-C3-C4-C5, C1=C2-C3-C4, C3-C4-C5=C6) for a hexadiene with C3-C4
    bonded (pc_plus's frame)."""
    return dihedral(x, 1, 2, 3, 4), dihedral(x, 0, 1, 2, 3), dihedral(x, 2, 3, 4, 5)


def rotation(axis, angle_deg):
    """The rotation matrix by angle about the unit axis (right hand)."""
    import numpy

    k = axis / numpy.linalg.norm(axis)
    kx = numpy.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    t = numpy.radians(angle_deg)
    return numpy.eye(3) + numpy.sin(t) * kx + (1 - numpy.cos(t)) * kx @ kx


def side(atoms_bonded: tuple[int, int], keep: int) -> list[int]:
    """The atoms (C and their H) on `atoms_bonded[1]`'s side of the bond,
    the chain C1..C6 being linear: carbons past it and their hydrogens."""
    a, b = atoms_bonded
    carbons = list(range(b, 6)) if b > a else list(range(0, b + 1))
    del keep
    return carbons + [h for c in carbons for h in H_ON[c]]


def set_dihedrals(x, target: tuple[float, float, float]):
    """x (bohr, pc_plus's frame) with its three dihedrals set to `target`
    by rigid rotations about C3-C4 (moving C4's side), C2-C3 (moving C2's)
    and C4-C5 (moving C5's). Returns the new x and each atom's accumulated
    rotation matrix, which the orbitals then follow."""
    import numpy

    x = x.copy()
    rot = [numpy.eye(3) for _ in x]
    # (bond, the moving atoms, which dihedral)
    moves = [
        ((2, 3), side((2, 3), 3), 0),  # C3-C4: C4..C6
        ((2, 1), side((2, 1), 1), 1),  # C2-C3: C1, C2
        ((3, 4), side((3, 4), 4), 2),  # C4-C5: C5, C6
    ]
    for (a, b), moving, k in moves:
        delta = target[k] - dihedrals(x)[k]
        # The sense that raises the dihedral depends on which end moves:
        # take whichever sense lands on the target.
        for sign in (1.0, -1.0):
            r = rotation(x[b] - x[a], sign * delta)
            moved = x.copy()
            for i in moving:
                moved[i] = x[b] + r @ (x[i] - x[b])
            got = dihedrals(moved)[k]
            if abs((got - target[k] + 180) % 360 - 180) < 1e-6:
                break
        else:
            raise AssertionError((k, got, target))
        x = moved
        for i in moving:
            rot[i] = r @ rot[i]
    return x, rot


def ao_rotation(symbol: str, r):
    """U with C_new = U C_old for one atom's AOs (this tree's order) when the
    atom's functions rotate rigidly by r: phi_mu(r^T x) = sum_nu phi_nu(x)
    U_nu,mu, fitted on random points (exact: each shell is closed under
    rotation, and D, the contraction norm, is the same for a shell's m)."""
    import numpy
    from pyscf import gto
    from scf_reference import pyscf_basis

    basis = pyscf_basis("cc-pVDZ.json", {symbol})
    mol = gto.M(
        atom=[(symbol, (0, 0, 0))],
        basis=basis,
        unit="Bohr",
        spin=None,
        verbose=0,
        charge=0 if symbol != "H" else -1,
    )
    rng = numpy.random.default_rng(7)
    pts = rng.normal(scale=1.2, size=(400, 3))
    a = mol.eval_gto("GTOval_sph", pts)
    b = mol.eval_gto("GTOval_sph", pts @ r)  # rows (r^T p)^T = p^T r
    u, *_ = numpy.linalg.lstsq(a, b, rcond=None)
    assert numpy.abs(a @ u - b).max() < 1e-9 * numpy.abs(b).max()
    return u


def rotate_orbitals(c, symbols: list[str], rot: list, source=None):
    """C (nao x nmo, this tree's AOs) with each atom's rows rotated by its
    own matrix: the orbitals follow the atoms that moved. With `source`,
    atom i's rows are taken from atom source[i]'s (of the same element)
    before rotating: the orbitals follow atoms that were also renumbered."""
    import numpy

    sys.path.insert(0, str(ROOT / "tools" / "emit"))
    source = list(range(len(symbols))) if source is None else source
    sizes = [ao_rotation(sym, numpy.eye(3)).shape[0] for sym in symbols]
    offsets = numpy.concatenate([[0], numpy.cumsum(sizes)])
    assert offsets[-1] == c.shape[0]
    out = numpy.empty_like(c)
    for i, (sym, r) in enumerate(zip(symbols, rot, strict=True)):
        j = source[i]
        assert symbols[j] == sym, (i, j)
        block = c[offsets[j] : offsets[j + 1]]
        out[offsets[i] : offsets[i + 1]] = ao_rotation(sym, r) @ block
    return out


def boat_c2v(r_inter: float):
    """An exactly C2v boat (Angstrom) built from step 4b's PC-NEVPT2 chair TS:
    its allyl C1 C2 C3 (with H 6-10) in a frame with z along the mean of
    C1->C6 and C3->C4, symmetrised under its own mirror x -> -x (C1<->C3),
    its terminal carbons put at z = -r_inter/2, and the second allyl its
    mirror image through z = 0 (C1<->C6, C2<->C5, C3<->C4) instead of the
    chair's inversion image. Both central carbons then sit on the same side,
    each lifted away from the other allyl: the boat. Ops: x -> -x, z -> -z
    and their product, the C2 axis y. (Step 1's chair builder, mirrored the
    same way, puts the inner terminal hydrogens 1.07 A apart.)"""
    import numpy

    path = STEP4 / "geometries" / "pc_ts.xyz"
    (_, ts), *_ = s4.read_xyz_frames(path)
    x = numpy.array([v for _, v in ts])
    z = (x[5] - x[0]) + (x[3] - x[2])
    z /= numpy.linalg.norm(z)
    ex = x[0] - x[2]
    ex -= (ex @ z) * z
    ex /= numpy.linalg.norm(ex)
    frame = numpy.array([ex, numpy.cross(z, ex), z])
    allyl = [0, 1, 2, 6, 7, 8, 9, 10]
    y = (x[allyl] - x[:6].mean(axis=0)) @ frame.T
    mx = y * numpy.array([-1.0, 1.0, 1.0])
    perm = numpy.linalg.norm(y[:, None] - mx[None], axis=-1).argmin(axis=1)
    assert sorted(perm) == list(range(len(allyl))), perm
    y = (y + mx[perm]) / 2.0
    y[:, 2] -= (y[0, 2] + y[2, 2]) / 2.0 + r_inter / 2.0
    m = numpy.array([1.0, 1.0, -1.0])
    c1, c2, c3, h1a, h1b, h2, h3a, h3b = y
    carbons = [c1, c2, c3, c3 * m, c2 * m, c1 * m]
    hydrogens = [h1a, h1b, h2, h3a, h3b, h3a * m, h3b * m, h2 * m, h1a * m, h1b * m]
    atoms = [("C", c) for c in carbons] + [("H", h) for h in hydrogens]
    # Where each boat atom came from, as a proper rigid motion of a TS
    # atom, for its orbitals: allyl A is the TS's allyl A rotated into the
    # frame; allyl B is the mirror image of A, which is C2(z) applied to the
    # TS's allyl B (the TS's B being A's inversion image, and mirror = C2 x
    # inversion). Matched by position, so the renumbering is found, not
    # assumed.
    c2z = numpy.diag([-1.0, -1.0, 1.0])
    ts_frame = (x - x[:6].mean(axis=0)) @ frame.T
    source, rot = [], []
    for i, (sym, pos) in enumerate(atoms):
        in_a = pos[2] < 0.0
        r = frame if in_a else c2z @ frame
        guess = ts_frame @ (numpy.eye(3) if in_a else c2z).T
        d = numpy.linalg.norm(guess[:, :2] - pos[None, :2], axis=1)
        d += 10.0 * numpy.array([s != sym for s, _ in ts])
        d += 10.0 * numpy.array([(g[2] < 0.0) != in_a for g in guess])
        j = int(d.argmin())
        assert d[j] < 1e-3, (i, d[j])
        source.append(j)
        rot.append(r)
    assert sorted(source) == list(range(len(atoms))), source
    return atoms, source, rot


def boat_start(r_inter: float):
    """(atoms in Angstrom, orbitals over this tree's AOs, what): the C2v boat
    and, as its start, the chair TS walk's converged CASSCF orbitals
    (ts_walk's casscf_orbitals.npy), each atom's AO rows carried with the
    rigid motion that took its TS atom to the boat (boat_c2v): the allyl pi
    orbitals land on the boat's allyls. free-breakfast's CASSCF converges
    from there (its frontier default, columns 20-25)."""
    import numpy

    built, source, rot = boat_c2v(r_inter)
    atoms = [(s, tuple(float(v) for v in xyz)) for s, xyz in built]
    c = numpy.load(RUNS / "ts_walk" / "casscf_orbitals.npy")
    c = rotate_orbitals(c, [s for s, _ in atoms], rot, source)
    what = (
        f"C2v boat at R = {r_inter} A from step 4b's TS allyls; ts_walk's "
        f"CASSCF orbitals carried with the atoms (TS atom for each boat "
        f"atom: {source})"
    )
    return atoms, c, what


# --- prepare -------------------------------------------------------------


def start_of(name: str):
    """(atoms in Angstrom, orbitals over this tree's AOs, a description)."""
    import numpy

    _, start = RUNS_SPEC[name]
    if isinstance(start, float):
        atoms, c, _ = energies.projected(start)
        return atoms, c, f"step 1's R = {start:.2f} A, its orbitals projected"
    kind = start[0]
    if kind == "xyz":
        _, path, run = start
        (_, atoms), *_ = s4.read_xyz_frames(Path(path))
        c = numpy.load(RUNS / run / "casscf_orbitals.npy")
        rel = Path(path).relative_to(STUDY)
        return atoms, c, f"{rel}, {run}'s CASSCF orbitals"
    if kind == "final":
        (_, atoms), *_ = s4.read_xyz_frames(RUNS / start[1] / "final.xyz")
        c = numpy.load(RUNS / start[1] / "casscf_orbitals.npy")
        return atoms, c, f"{start[1]}'s final geometry and CASSCF orbitals"
    if kind == "mode":
        _, hessian, sign = start
        h = json.loads((RUNS / hessian / "hessian.json").read_text())
        (_, atoms), *_ = s4.read_xyz_frames(INPUTS / hessian / "geometry.xyz")
        k = int(numpy.argmin(h["frequencies"]))
        assert h["frequencies"][k] < 0, f"{hessian} has no imaginary frequency"
        mode = numpy.array(h["modes"][k]).reshape(-1, 3)
        x = s4.coords_bohr(atoms)
        if s4.s_direction(x) @ mode.ravel() < 0:
            mode = -mode
        mode *= sign * s4.DISPLACEMENT / abs(mode).max()
        moved = (x + mode) * BOHR
        atoms = [
            (s, tuple(map(float, v))) for (s, _), v in zip(atoms, moved, strict=True)
        ]
        c = numpy.load(RUNS / hessian / "casscf_orbitals.npy")
        what = (
            f"{hessian}'s geometry {'+' if sign > 0 else '-'} its imaginary mode "
            f"({h['frequencies'][k]:.1f} cm-1), largest component "
            f"{s4.DISPLACEMENT} bohr"
        )
        return atoms, c, what
    if kind == "conformer":
        label = start[1]
        base = RUNS_SPEC["plus_hessian"][1]
        (_, atoms), *_ = s4.read_xyz_frames(Path(base[1]))
        c = numpy.load(RUNS / base[2] / "casscf_orbitals.npy")
        x, rot = set_dihedrals(s4.coords_bohr(atoms), CONFORMERS[label])
        symbols = [s for s, _ in atoms]
        c = rotate_orbitals(c, symbols, rot)
        atoms = [
            (s, tuple(map(float, v * BOHR))) for s, v in zip(symbols, x, strict=True)
        ]
        what = (
            f"pc_plus with dihedrals (C2C3C4C5, C1C2C3C4, C3C4C5C6) set to "
            f"{CONFORMERS[label]}, plus_walk's orbitals rotated with the atoms"
        )
        return atoms, c, what
    if kind == "boat":
        return boat_start(start[1])
    if kind == "fd":
        _, irrep, delta = start
        (_, atoms), *_ = s4.read_xyz_frames(INPUTS / "ts_hessian" / "geometry.xyz")
        k, freq, q = fd_direction(irrep)
        x = s4.coords_bohr(atoms) + delta * q
        atoms = [
            (s, tuple(map(float, v * BOHR))) for (s, _), v in zip(atoms, x, strict=True)
        ]
        c = numpy.load(RUNS / "ts_hessian" / "casscf_orbitals.npy")
        what = (
            f"ts_hessian's geometry {delta:+.2f} bohr x its mode {k} ({freq:.1f} cm-1, "
            f"{irrep}) scaled to largest component 1; ts_hessian's CASSCF orbitals"
        )
        return atoms, c, what
    raise ValueError(start)


def c2h_irreps(hessian_run: str) -> list[tuple[int, float, str, float]]:
    """(mode, frequency, C2h irrep, worst |character| defect) for each of
    hessian.json's modes at that run's geometry, from the characters under
    its own four operations (analyse.mode_characters)."""
    import analyse

    entry = analyse.hessian_entry(hessian_run)
    h = json.loads((RUNS / hessian_run / "hessian.json").read_text())
    import numpy

    table = {
        (1, 1, 1): "ag",
        (-1, 1, -1): "bg",
        (1, -1, -1): "au",
        (-1, -1, 1): "bu",
    }
    out = []
    for k, f in enumerate(h["frequencies"]):
        entry["_modes"] = numpy.array(h["modes"])
        chars = analyse.mode_characters(entry, k)
        assert len(chars) == 4, f"{hessian_run} is not C2h: {len(chars)} operations"
        c2 = next(c for c in chars if c["det"] > 0 and c["op"].count(-1.0) == 2)
        inv = next(c for c in chars if c["op"] == [-1.0, -1.0, -1.0])
        sig = next(c for c in chars if c["det"] < 0 and c is not inv)
        key = tuple(int(round(c["character"])) for c in (c2, inv, sig))
        defect = max(abs(abs(c["character"]) - 1.0) for c in (c2, inv, sig))
        out.append((k, float(f), table[key], defect))
    return out


def fd_direction(irrep: str):
    """(mode, frequency, q): ts_hessian's lowest mode of `irrep`, from
    hessian.json's Cartesian modes, scaled to largest component 1 (bohr)."""
    import numpy

    h = json.loads((RUNS / "ts_hessian" / "hessian.json").read_text())
    k, f, _, _ = next(m for m in c2h_irreps("ts_hessian") if m[2] == irrep)
    q = numpy.array(h["modes"][k]).reshape(-1, 3)
    q /= numpy.abs(q).max()
    # A fixed phase: the largest component positive.
    if q.ravel()[numpy.abs(q).argmax()] < 0:
        q = -q
    return k, f, q


def prepare(name: str) -> None:
    import numpy

    kind, _ = RUNS_SPEC[name]
    atoms, c, what = start_of(name)
    out = INPUTS / name
    out.mkdir(parents=True, exist_ok=True)
    energies.write_xyz(out / "geometry.xyz", atoms, f"{name}: {what}")
    numpy.save(out / "orbitals.npy", numpy.asfortranarray(c), allow_pickle=False)
    grad = s4.CONV_TOL_GRAD_WALK if kind == "walk" else s4.CONV_TOL_GRAD_HESSIAN
    head = (
        HEAD
        if kind != "energy"
        else HEAD.replace(
            '[nevpt2_gradient]\nvariant = "pc"\n',
            "[nevpt2]\n",
        )
    )
    text = head.format(
        name=name,
        what=f"PC-NEVPT2 {kind} from {what}",
        conv_tol=s4.CONV_TOL,
        conv_tol_grad=grad,
    )
    if kind == "walk":
        text += s4.OPTIMIZE.format(**s4.OPT_TIGHT, max_steps=s4.MAX_STEPS)
    elif kind == "hessian":
        text += s4.HESSIAN
    elif kind == "numerical":
        text += HESSIAN_NUMERICAL
    elif kind == "difference":
        text += HESSIAN_DIFFERENCE
    (out / "config.toml").write_text(text)
    r16, r34 = s4.distances(atoms)
    print(f"{name}: prepared, R16 {r16:.4f} R34 {r34:.4f} A ({what})")


# --- run -----------------------------------------------------------------


def done(name: str) -> bool:
    kind, _ = RUNS_SPEC[name]
    out = RUNS / name
    if kind == "energy":
        return pc_energy(out) is not None
    if kind in ("hessian", "numerical", "difference"):
        return (out / "hessian.json").exists()
    return (out / "nevpt2_gradient.json").exists()


def pc_energy(run_dir: Path) -> float | None:
    """The PC-NEVPT2 total energy an energy run printed (10 decimals)."""
    log = run_dir / "log.txt"
    if not log.exists():
        return None
    for line in log.read_text().splitlines():
        if line.startswith("PC-NEVPT2 root 0:"):
            return float(line.split()[3])
    return None


def card_memory_mib() -> int | None:
    """The card's memory.used (MiB). The container sees no per-process list
    (nvidia-smi's compute-apps query comes back empty), so the process's
    own use is read as this minus the idle value before the run."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return int(out.stdout.split()[0])
    except (subprocess.SubprocessError, ValueError, IndexError, OSError):
        return None


def run(names: list[str], binary: Path) -> None:
    for name in names:
        if done(name):
            print(f"{name}: already run")
            continue
        out = RUNS / name
        out.mkdir(parents=True, exist_ok=True)
        inp = INPUTS / name
        cmd = [str(binary), "-i", str(inp), "-o", str(out), str(inp / "config.toml")]
        idle = card_memory_mib()
        samples: list[int] = []
        stop = threading.Event()

        def sample(samples=samples, stop=stop) -> None:
            while not stop.is_set():
                m = card_memory_mib()
                if m is not None:
                    samples.append(m)
                stop.wait(0.5)

        sampler = threading.Thread(target=sample, daemon=True)
        t0 = time.monotonic()
        with (out / "log.txt").open("w") as f:
            sampler.start()
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        wall = time.monotonic() - t0
        stop.set()
        sampler.join()
        (out / "wall.txt").write_text(f"{wall:.1f}\n")
        peak = max(samples) if samples else None
        mem = {
            "idle_mib": idle,
            "peak_card_mib": peak,
            "peak_process_mib": (peak - idle) if peak is not None and idle else None,
            "samples": len(samples),
            "exit": proc.returncode,
        }
        (out / "memory.json").write_text(json.dumps(mem, indent=1) + "\n")
        status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        print(
            f"{name}: {status} in {wall:.0f} s, card peak {peak} MiB (idle {idle})",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=["prepare", "run", "analyse"])
    parser.add_argument("names", nargs="*", help="runs (RUNS_SPEC's names)")
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "build" / "src" / "free-breakfast"
    )
    args = parser.parse_args()
    unknown = [n for n in args.names if n not in RUNS_SPEC]
    if unknown:
        parser.error(f"unknown runs {unknown}; known: {list(RUNS_SPEC)}")
    if args.phase == "prepare":
        for name in args.names:
            prepare(name)
    elif args.phase == "run":
        run(args.names, args.binary)
    else:
        import analyse

        analyse.main()


if __name__ == "__main__":
    main()
