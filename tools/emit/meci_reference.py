#!/usr/bin/env python3
"""Write the DF-SA-CASSCF minimum-energy conical intersection reference.

One case: ethylene, SA2-CASSCF(2,2) (pi, pi*), cc-pVDZ over cc-pVDZ-JKFIT,
equal weights, its S0/S1 minimum-energy conical intersection (MECI), found
from a twisted and pyramidalized start. The DF-SA-CASSCF at every geometry is
built, converged and polished as casscf_nac_reference.py builds it (over a
DF-RHF _scf sharing its fit, mc1step polished by Newton steps to |g| <
1e-10, put in canonical orbitals, its CI re-solved densely), and
differentiated by PySCF: the roots' gradients g_0, g_1 by `df.grad.sacasscf`
and the coupling h_01 by casscf_nac_reference.py's DFNonAdiabaticCouplings,
the `mult_ediff=True`, `use_etfs=True` numerator (casscf_nac/'s `h`).

**No program finds a DF-SA-CASSCF MECI,** so the reference is this tool's
own loop over PySCF's quantities. Each step is the Bearpark-Robb-Schlegel
composite gradient's two parts, each taken as a Newton step:

* in the branching plane, spanned by x = g_1 - g_0 and y = h_01, the
  first-order step onto the seam: the least displacement d in span{x, y}
  with x.d = -gap and y.d = 0 (exact for a linear cone, elliptic or tilted:
  the gap is homogeneous of degree one in the distance from the seam, and
  y is tangential to it);
* in the seam, Newton on the projected force -P gbar, gbar = (g_0 + g_1)/2:
  its value at the geometry the plane step reaches, over its Jacobian along
  the seam, both from central differences (HESSIAN_STEP bohr, recomputed at
  every iterate) of the projected force along the seam's and the plane's
  orthonormal directions, each displaced point projected over its own
  plane. That Jacobian is P (dgbar/dR) P plus the plane's turn along the
  seam against gbar's in-plane part; with P (dgbar/dR) P alone the loop
  contracts by only ~0.4 per step here and stalls near 2e-6 Eh/bohr.

The projector is #771's (geometry::optimize_meci's): the three translations
and three rotations about the centroid are removed, {x, y} orthonormalized
in that order after them, and P projects off all eight; the projected force
is -P gbar. The loop stops when the gap is under GAP_TOLERANCE and the
largest projected force under FORCE_TOLERANCE. At the MECI the tool records
what the consumer's bars are derived from:

* k_min, the smallest eigenvalue of the seam-projected Hessian P H P on the
  3N - 8 seam directions, H gbar's central difference along the seam's
  orthonormal directions (the same differences), symmetrized;
* x_norm = |x| and h_norm = |h_01|;
* gbar_plane, the norm of gbar's component in the branching plane (the
  cone's tilt);

and refuses to write a k_min that is not positive, or a point whose seam
curvature with the plane's turn (the symmetrized Jacobian of P gbar along
the seam, `reduced_curvatures`) is not positive definite: either is a
saddle on the seam. From a Cs start the loop keeps Cs and finds one (k_min
0.012 but a reduced curvature of -0.028), so the start twists off 90.

**Root flipping.** At every geometry the tool evaluates (the iterates and
every difference's displaced point), a CASCI at that geometry's SA2 orbitals with
three singlet roots gives E_2; the tool refuses to write the fixture if
min(E_2 - E_1) falls under S2_MARGIN.

**An independent single point** at the recorded MECI (a fresh exact RHF,
the SA-CASSCF from it by mc1step and by mc2step, which must agree) must
reproduce the gap under GAP_TOLERANCE and the projected force under
FORCE_TOLERANCE.

**At the start** the tool records E_0, E_1, g_0, g_1 and h_01, the CI signed
by casscf_nac/'s rule (each vector's first coefficient, row-major, within
1e-8 of its largest magnitude, positive), so h_01's sign is fixed.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/meci_reference.py

writes test/gtest/data/casscf_meci/casscf_meci.json (`--out` writes
elsewhere, for comparing a rerun). Re-run it to regenerate; the file is never
hand-edited. PySCF's OpenMP is pinned to THREADS and BLAS to one thread, so
a run does not depend on the machine's core count; threaded reductions make
a rerun agree to the precision the fixture's README states, not bit for bit.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: PySCF's OpenMP at THREADS, BLAS at one. PySCF
# calls BLAS from inside its OpenMP loops, so a threaded BLAS there nests and
# runs past THREADS cores; at one it costs this tool no wall time.
THREADS = 8
os.environ["OMP_NUM_THREADS"] = str(THREADS)
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
import scipy.sparse.linalg  # noqa: E402
from casci_reference import aux_basis  # noqa: E402
from casscf_gradient_reference import (  # noqa: E402
    LAGRANGE_ATOL,
    LAGRANGE_RTOL,
    build,
    canonical,
    matrix,
    minimum,
    polish,
    rows,
)
from casscf_nac_reference import DFNonAdiabaticCouplings, signed  # noqa: E402
from casscf_reference import (  # noqa: E402
    POLISH_GRAD,
    POLISH_STEPS,
    check_aux,
    molecule,
    optimize,
    spin_roots,
)
from pyscf import fci, gto, lib, scf  # noqa: E402
from pyscf.df.grad import sacasscf as dfsacasscf_grad  # noqa: E402
from pyscf.mcscf import mc1step  # noqa: E402

# casscf_nac_reference and casscf_reference set the variables again on
# import; numpy's BLAS is loaded by then, and PySCF's OpenMP is pinned here.
lib.num_threads(THREADS)

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "casscf_meci" / "casscf_meci.json"

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/meci_reference.py"
)

BOHR_PER_ANGSTROM = 1 / 0.52917721092


def twisted_pyramidalized() -> list[tuple[str, tuple[float, float, float]]]:
    """Ethylene with its second CH2 twisted TWIST degrees about C-C and
    pyramidalized: C-C 1.40 A, C-H 1.08 A, H-C-H 117 degrees; C2's H-C-H
    bisector bent PYRAMIDALIZATION degrees off the C-C axis, toward its own
    p orbital. A twist off 90 degrees leaves no symmetry: from a Cs start
    the loop keeps Cs, and the Cs point on this seam is a saddle."""
    half = numpy.radians(58.5)
    tilt = numpy.radians(PYRAMIDALIZATION)
    c1, c2 = numpy.zeros(3), numpy.array([0.0, 0.0, 1.40])
    flat = numpy.radians(60.0)
    h1 = [
        c1 + 1.08 * numpy.array([s * numpy.sin(flat), 0, -numpy.cos(flat)])
        for s in (1, -1)
    ]
    twist = numpy.radians(TWIST)
    axis = numpy.array([0.0, 0.0, 1.0])
    side = numpy.array([-numpy.cos(twist), numpy.sin(twist), 0.0])
    bisector = numpy.cos(tilt) * axis + numpy.sin(tilt) * numpy.cross(axis, side)
    h2 = [
        c2 + 1.08 * (numpy.cos(half) * bisector + s * numpy.sin(half) * side)
        for s in (1, -1)
    ]
    points = [c1, c2, *h1, *h2]
    return [
        (symbol, tuple(float(x) for x in BOHR_PER_ANGSTROM * p))
        for symbol, p in zip("CCHHHH", points, strict=True)
    ]


TWIST = 80.0
PYRAMIDALIZATION = 100.0

CASE = {
    "name": "c2h4_ccpvdz_sa2",
    "atoms": twisted_pyramidalized(),
    "basis": "cc-pvdz",
    "cas": (2, 2),
    "weights": [0.5, 0.5],
}

# The loop's stop: the gap (hartree) and the largest projected force
# (hartree/bohr).
GAP_TOLERANCE = 1e-8
FORCE_TOLERANCE = 1e-6
# The central difference of gbar (bohr).
HESSIAN_STEP = 1e-3
# The cap on one step (bohr per coordinate), and the loop's length.
MAX_STEP = 0.2
MAX_ITERATIONS = 40
# The least curvature a seam step divides by (hartree/bohr^2).
MIN_CURVATURE = 5e-3
# The least E_2 - E_1 over the path (hartree): a margin, not a bound.
S2_MARGIN = 1e-2
# The fresh single point's mc1step and mc2step must agree to this (hartree).
FRESH_AGREEMENT = 1e-9


def placed(coords: numpy.ndarray) -> dict:
    return dict(
        CASE,
        atoms=[(s, tuple(r)) for (s, _), r in zip(CASE["atoms"], coords, strict=True)],
    )


def carried(mol: gto.Mole, mo: numpy.ndarray) -> numpy.ndarray:
    """mo Lowdin-orthonormalized in mol's overlap."""
    overlap = mo.T @ mol.intor("int1e_ovlp") @ mo
    values, vectors = numpy.linalg.eigh(overlap)
    return mo @ vectors @ numpy.diag(values**-0.5) @ vectors.T


def exact_rhf(mol: gto.Mole) -> numpy.ndarray:
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        # At the twisted geometries DIIS reaches the energy but stalls
        # above conv_tol_grad. The RHF is only the CASSCF's start, so
        # second-order SCF from there, to PySCF's default gradient bar.
        mf = mf.newton()
        mf.conv_tol, mf.conv_tol_grad = 1e-12, None
        mf.kernel(mf.mo_coeff, mf.mo_occ)
    assert mf.converged, "RHF did not converge"
    return mf.mo_coeff


def krylov_polish(mc, mo: numpy.ndarray, weights: list[float]) -> numpy.ndarray:
    """polish's Newton steps to |g| < POLISH_GRAD, each solved by MINRES over
    gen_g_hop's Hessian-vector product rather than the dense Hessian: the
    Hessian displacements', started a Hessian step from a polished point."""
    ncas, nelec = mc.ncas, mc.nelecas
    for _ in range(POLISH_STEPS):
        _, vectors, _ = spin_roots(mc, mo, len(weights))
        dm1 = numpy.zeros((ncas, ncas))
        dm2 = numpy.zeros((ncas,) * 4)
        for vector, weight in zip(vectors, weights, strict=True):
            d1, d2 = fci.direct_spin1.make_rdm12(vector, ncas, nelec)
            dm1 += weight * d1
            dm2 += weight * d2
        mc.mo_coeff = mo
        half_g, _, h_op, _ = mc1step.gen_g_hop(mc, mo, 1, dm1, dm2, mc.ao2mo(mo))
        if 2 * numpy.linalg.norm(half_g) < POLISH_GRAD:
            return mo
        operator = scipy.sparse.linalg.LinearOperator((half_g.size,) * 2, h_op)
        x, info = scipy.sparse.linalg.minres(operator, -half_g, rtol=1e-12)
        assert info == 0, f"MINRES stopped with {info}"
        mo = mo @ mc1step.expmat(mc.unpack_uniq_var(x))
    raise AssertionError(f"the Krylov polish did not reach |g| {POLISH_GRAD}")


def evaluate(coords: numpy.ndarray, mo: numpy.ndarray | None, full: bool) -> dict:
    """The polished DF-SA-CASSCF at coords from mo carried over (a fresh RHF
    start when None): both roots' energies and gradients, h_01, and E_2 at
    its orbitals. With `full` (the dense polish's curvature gate, canonical
    orbitals, signed CI) it is a point the tool records; without, a
    difference's displaced point, polished by MINRES."""
    case = placed(coords)
    weights = case["weights"]
    mol = molecule(case)
    auxbasis, _ = aux_basis(case, mol)
    start = exact_rhf(mol) if mo is None else carried(mol, mo)
    mc = build(case, mol, auxbasis, start)
    optimize(mc, mc.kernel, start)
    assert mc.converged, "mc1step did not converge"
    if full:
        mo = polish(case["name"], mc, mc.mo_coeff, weights)
        state = canonical(case["name"], mc, mo, weights)
        state["ci"] = signed(case["name"], state["ci"])
    else:
        mo = krylov_polish(mc, mc.mo_coeff, weights)
        energies, vectors, _ = spin_roots(mc, mo, len(weights))
        state = {"mo": mo, "ci": vectors, "energies": energies}
    three, _, _ = spin_roots(mc, state["mo"], 3)
    assert numpy.allclose(three[:2], state["energies"], rtol=0, atol=1e-10)
    mc.mo_coeff, mc.ci = state["mo"], list(state["ci"])
    mc.fcisolver.e_states = list(state["energies"])
    mc.e_tot = float(numpy.dot(weights, state["energies"]))
    gradients = []
    for root in range(len(weights)):
        grad = mc.nuc_grad_method()
        assert isinstance(grad, dfsacasscf_grad.Gradients), type(grad)
        grad.verbose = 0
        grad.conv_rtol, grad.conv_atol = LAGRANGE_RTOL, LAGRANGE_ATOL
        gradients.append(grad.kernel(state=root))
        assert grad.converged, f"root {root}'s Lagrange solve did not converge"
    out = {
        "mol": mol,
        "mc": mc,
        "auxbasis": auxbasis,
        "state": state,
        "energies": [float(e) for e in state["energies"]],
        "e2": float(three[2]),
        "gradients": gradients,
    }
    method = DFNonAdiabaticCouplings(mc, state=(0, 1))
    method.verbose = 0
    method.conv_rtol, method.conv_atol = LAGRANGE_RTOL, LAGRANGE_ATOL
    out["h"] = method.kernel(state=(0, 1), use_etfs=True, mult_ediff=True)
    assert method.converged, "h_01's Lagrange solve did not converge"
    return out


def rigid_basis(coords: numpy.ndarray) -> numpy.ndarray:
    """The orthonormalized translations and rotations about the centroid."""
    centred = coords - coords.mean(axis=0)
    rigid = [numpy.tile(numpy.eye(3)[k], len(coords)) for k in range(3)]
    rigid += [numpy.cross(centred, numpy.eye(3)[k]).ravel() for k in range(3)]
    q, _ = numpy.linalg.qr(numpy.array(rigid).T)
    return q


def branching(coords: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray):
    """#771's projector: rigid modes off, then x and y orthonormalized in
    that order. Returns P, the plane's orthonormal pair and the seam's
    orthonormal basis (P's range)."""
    q = rigid_basis(coords)
    plane = []
    for v in (x.ravel(), y.ravel()):
        v = v - q @ (q.T @ v)
        for u in plane:
            v = v - u * (u @ v)
        norm = numpy.linalg.norm(v)
        assert norm > 1e-8, "the branching plane is degenerate"
        plane.append(v / norm)
    removed = numpy.column_stack([q, *plane])
    projector = numpy.eye(coords.size) - removed @ removed.T
    values, vectors = numpy.linalg.eigh(projector)
    seam = vectors[:, values > 0.5]
    assert seam.shape[1] == coords.size - 8
    return projector, plane, seam


def analysis(coords: numpy.ndarray, point: dict) -> dict:
    """The gap, x, y = h_01, gbar and the projected force at a point."""
    g0, g1 = point["gradients"]
    x = g1 - g0
    gbar = 0.5 * (g0 + g1)
    projector, plane, seam = branching(coords, x, point["h"])
    force = -(projector @ gbar.ravel())
    return {
        "gap": point["energies"][1] - point["energies"][0],
        "x": x,
        "gbar": gbar,
        "projector": projector,
        "plane": plane,
        "seam": seam,
        "force": force,
        "max_force": float(numpy.abs(force).max()),
    }


def differences(coords: numpy.ndarray, point: dict, at: dict, margins: list):
    """Central differences (HESSIAN_STEP bohr) along the seam's and then the
    plane's orthonormal directions, each displaced point's SA-CASSCF from
    the point's orbitals carried: of gbar, and of the projected force, each
    displaced point's over its own plane. Each one's E_2 - E_1 into margins.

    Returns the directions and the two n x m matrices of differences."""
    directions = numpy.column_stack([at["seam"], *at["plane"]])
    mo = point["state"]["mo"]
    gbar = numpy.zeros((coords.size, directions.shape[1]))
    force = numpy.zeros_like(gbar)
    for j, direction in enumerate(directions.T):
        for s in (1, -1):
            moved = coords + s * HESSIAN_STEP * direction.reshape(coords.shape)
            displaced = evaluate(moved, mo, full=False)
            margins.append(displaced["e2"] - displaced["energies"][1])
            there = analysis(moved, displaced)
            gbar[:, j] += s * there["gbar"].ravel()
            force[:, j] += s * there["force"]
    return directions, gbar / (2 * HESSIAN_STEP), force / (2 * HESSIAN_STEP)


def seam_curvatures(at: dict, gbar: numpy.ndarray, force: numpy.ndarray):
    """The seam block of gbar's difference (P dgbar/dR P on the seam,
    symmetrized: k_min's), and of the projected gradient P gbar's (which
    also carries the plane's turn along the seam), each's eigenvalues."""
    m = at["seam"].shape[1]
    block = at["seam"].T @ gbar[:, :m]
    reduced = -(at["seam"].T @ force[:, :m])
    return (
        numpy.linalg.eigvalsh(0.5 * (block + block.T)),
        numpy.linalg.eigvalsh(0.5 * (reduced + reduced.T)),
    )


def step(coords: numpy.ndarray, point: dict, at: dict, force: numpy.ndarray):
    """The plane's first-order step onto the seam, then Newton on the
    projected force in the seam at the point that step reaches."""
    q = rigid_basis(coords)
    x = at["x"].ravel() - q @ (q.T @ at["x"].ravel())
    y = point["h"].ravel() - q @ (q.T @ point["h"].ravel())
    a = numpy.array([x, y])
    plane_step = -a.T @ numpy.linalg.solve(a @ a.T, numpy.array([at["gap"], 0.0]))
    seam = at["seam"]
    m = seam.shape[1]
    plane = numpy.column_stack(at["plane"])
    reached = at["force"] + force[:, m:] @ (plane.T @ plane_step)
    # The symmetrized Jacobian's modes, each at its |curvature|: Newton on a
    # minimum, downhill along a mode that is not one yet.
    jacobian = -(seam.T @ force[:, :m])
    curvatures, modes = numpy.linalg.eigh(0.5 * (jacobian + jacobian.T))
    rhs = modes.T @ (seam.T @ -reached)
    scale = numpy.maximum(numpy.abs(curvatures), MIN_CURVATURE)
    seam_step = -seam @ (modes @ (rhs / scale))
    total = plane_step + seam_step
    total *= min(1.0, MAX_STEP / numpy.abs(total).max())
    return coords + total.reshape(coords.shape)


def atoms_out(coords: numpy.ndarray) -> list[dict]:
    return [
        {"symbol": symbol, "z": gto.charge(symbol), "coord": [float(x) for x in r]}
        for (symbol, _), r in zip(CASE["atoms"], coords, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    start = numpy.array([coord for _, coord in CASE["atoms"]], dtype=float)
    coords = start.copy()
    point = evaluate(coords, None, full=True)
    first = point
    margins = [point["e2"] - point["energies"][1]]
    path = []
    while True:
        at = analysis(coords, point)
        record = {
            "energies": point["energies"],
            "gap": at["gap"],
            "max_projected_force": at["max_force"],
            "s2_margin": point["e2"] - point["energies"][1],
            "coords": rows(coords),
        }
        path.append(record)
        print(f"iteration {len(path) - 1}: {json.dumps(record)}", flush=True)
        if at["gap"] < GAP_TOLERANCE and at["max_force"] < FORCE_TOLERANCE:
            break
        assert len(path) <= MAX_ITERATIONS, "the MECI loop did not converge"
        _, gbar_d, force_d = differences(coords, point, at, margins)
        curvatures, reduced = seam_curvatures(at, gbar_d, force_d)
        print(f"  seam curvatures {curvatures}\n  reduced {reduced}", flush=True)
        mo = point["state"]["mo"]
        coords = step(coords, point, at, force_d)
        point = evaluate(coords, mo, full=True)
        margins.append(point["e2"] - point["energies"][1])

    # What the consumer's bars are derived from, at the MECI.
    _, gbar_d, force_d = differences(coords, point, at, margins)
    curvatures, reduced = seam_curvatures(at, gbar_d, force_d)
    k_min = float(curvatures[0])
    assert k_min > 0.0, f"a saddle on the seam: k_min {k_min}"
    assert reduced[0] > 0.0, f"a saddle on the seam: curvature {reduced[0]}"
    s2_margin = float(min(margins))
    assert s2_margin > S2_MARGIN, f"E_2 comes within {s2_margin} of E_1"
    gbar_plane = float(numpy.linalg.norm([u @ at["gbar"].ravel() for u in at["plane"]]))

    # An independent single point: a fresh RHF, mc1step and mc2step.
    case = placed(coords)
    mol = molecule(case)
    auxbasis, aux_provenance = aux_basis(case, mol)
    (mc, state), _ = minimum(case, mol, auxbasis, exact_rhf(mol))
    check_aux(case["name"], mc, mol, auxbasis)
    fresh = evaluate(coords, None, full=True)
    agreement = float(
        numpy.abs(numpy.subtract(fresh["energies"], state["energies"])).max()
    )
    assert agreement < FRESH_AGREEMENT, f"the fresh points differ by {agreement}"
    fresh_at = analysis(coords, fresh)
    assert fresh_at["gap"] < GAP_TOLERANCE, f"fresh gap {fresh_at['gap']}"
    assert fresh_at["max_force"] < FORCE_TOLERANCE, (
        f"fresh force {fresh_at['max_force']}"
    )

    ncas, nelecas = CASE["cas"][1], CASE["cas"][0]
    g0, g1 = point["gradients"]
    doc = {
        "description": "The S0/S1 minimum-energy conical intersection of "
        "ethylene's density-fitted SA2-CASSCF(2,2), this tool's own loop over "
        "PySCF's df.grad.sacasscf gradients and DF SA-CASSCF coupling h_01, "
        "over this tree's data/basis and data/aux_basis files. Hartree, bohr; "
        "row A, column k is d/dR_Ak",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "command": COMMAND,
            "threads": THREADS,
        },
        "name": CASE["name"],
        "basis": CASE["basis"],
        "aux_basis": aux_provenance,
        "charge": 0,
        "electrons": int(mol.nelectron),
        "nao": int(mol.nao),
        "nmo": int(point["state"]["mo"].shape[1]),
        "ncore": int(point["mc"].ncore),
        "ncas": ncas,
        "nelecas": nelecas,
        "weights": CASE["weights"],
        "start": {
            "atoms": atoms_out(start),
            "energies": first["energies"],
            "gradients": [rows(g) for g in first["gradients"]],
            "h": rows(first["h"]),
            "ci": [matrix(c) for c in first["state"]["ci"]],
            "natural_occupations": [
                float(n) for n in first["state"]["natural_occupations"]
            ],
        },
        "meci": {
            "atoms": atoms_out(coords),
            "energies": point["energies"],
            "gap": at["gap"],
            "mean_energy": 0.5 * (point["energies"][0] + point["energies"][1]),
            "gradients": [rows(g0), rows(g1)],
            "h": rows(point["h"]),
            "projected_force": rows(at["force"].reshape(coords.shape)),
            "max_projected_force": at["max_force"],
            "k_min": k_min,
            "seam_curvatures": [float(c) for c in curvatures],
            "reduced_curvatures": [float(c) for c in reduced],
            "x_norm": float(numpy.linalg.norm(at["x"])),
            "h_norm": float(numpy.linalg.norm(point["h"])),
            "gbar_plane": gbar_plane,
        },
        "s2_margin": s2_margin,
        "fresh": {
            "energies": fresh["energies"],
            "gap": fresh_at["gap"],
            "max_projected_force": fresh_at["max_force"],
            "mc2step_agreement": agreement,
        },
        "path": path,
        "optimizer": {
            "gap_tolerance": GAP_TOLERANCE,
            "force_tolerance": FORCE_TOLERANCE,
            "hessian_step": HESSIAN_STEP,
            "max_step": MAX_STEP,
            "s2_margin_floor": S2_MARGIN,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
