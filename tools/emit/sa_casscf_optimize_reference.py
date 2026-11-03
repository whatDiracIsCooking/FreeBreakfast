#!/usr/bin/env python3
"""Write the reference minimum the DF-SA-CASSCF root-optimization test reads.

One case: formaldehyde in cc-pVDZ over cc-pVDZ-JKFIT, the DF-SA2-CASSCF(4,3)
over its frontier orbitals (equal weights), **root 1** (the n -> pi* S1),
optimized from casscf_nac_reference.py's FORMALDEHYDE_SKEW (pyramidalized
and twisted, C1) to PySCF's minimum of that root.

**Why the skew start.** S1's minimum is pyramidal. From a C2v start the
gradient keeps C2v, and any optimizer, this one or the tree's, stops at the
planar saddle point; the two would agree there and the test would not see
it. So the start is non-planar, and the tool refuses a minimum that is not:
the smallest eigenvalue of the final rigid-projected Hessian, `k_min`, must be
positive (a minimum, not the planar saddle), and the out-of-plane angle at
the carbon nonzero. The minimum comes as two mirror images; the test aligns
allowing a reflection.

**The optimizer.** Newton's method over PySCF's analytic DF-SA-CASSCF
gradient of root 1 (`df.grad.sacasscf`, its Lagrange solve at conv_rtol
1e-12), its Hessian a central difference of that gradient (HESSIAN_STEP
bohr) recomputed every step, as casscf_optimize_reference.py does. Rigid
translations and rotations are projected out, and each internal direction
is stepped by 1/|lambda|, so a direction of negative curvature (the start is
far from the minimum) is a step downhill, not toward a saddle; steps are
capped at MAX_STEP bohr per coordinate. Converged at max |P dE/dR| <
FORCE_TOLERANCE, P the rigid projector; the Hessian is then taken once more
there for `k_min`.

**Each energy and gradient.** The DF-SA-CASSCF is built as
casscf_gradient_reference.py builds it (over a DF-RHF _scf sharing its fit),
converged by mc1step and polished by Newton steps to |g| < 1e-10, the roots
re-solved densely there, and differentiated at those orbitals and CI. At the
start its orbitals are the exact RHF's; at every later geometry the previous
one's converged orbitals, Lowdin-orthonormalized in the new overlap.

**The root is followed.** Each Newton step's roots are overlapped with the
last step's (casscf_nac_reference.py's `ci_overlap`, over the two
geometries' orbitals and their mixed AO overlap), and root 1 must match root
1, its row of overlaps normalized, at least ROOT_FOLLOW_FLOOR, the tree's own
floor: Newton stays on the S1 surface. The row is normalized because the
whole determinant's overlap decays with the step common to every root (0.93
at 0.05 bohr, 0.76 at 0.1 bohr from the start), which is not the root
turning. The smallest is written as `min_root_overlap`. At the minimum
the SA-CASSCF is rerun from a fresh RHF and must reach the same root-1
energy (to FRESH_AGREEMENT).

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/sa_casscf_optimize_reference.py

writes test/gtest/data/casscf_gradient/sa_casscf_optimize.json. Re-run it to
regenerate; the file is never hand-edited. OpenMP and MKL run at THREADS
threads and OpenBLAS at one (its own pool otherwise oversubscribes OpenMP's),
so a rerun agrees to threaded rounding, not bit for bit; nothing
run-dependent (date, host) is written.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS.
THREADS = 8
os.environ["OMP_NUM_THREADS"] = str(THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = str(THREADS)

import json  # noqa: E402
import pathlib  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from casci_reference import aux_basis  # noqa: E402
from casscf_gradient_reference import build, polish  # noqa: E402
from casscf_nac_reference import FORMALDEHYDE_SKEW, ci_overlap  # noqa: E402
from casscf_reference import molecule, optimize, spin_roots  # noqa: E402
from gradient_reference import internal_projector  # noqa: E402
from pyscf import gto, lib, scf  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "casscf_gradient" / "sa_casscf_optimize.json"

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/sa_casscf_optimize_reference.py"
)

CASE = {
    "name": "h2co_skew_ccpvdz_sa2",
    "atoms": FORMALDEHYDE_SKEW,
    "basis": "cc-pvdz",
    "cas": (4, 3),
    "weights": [0.5, 0.5],
}
STATE = 1

FORCE_TOLERANCE = 1e-6
HESSIAN_STEP = 1e-3
# Newton steps before the tool gives up, and the cap on one step (bohr per
# coordinate).
MAX_NEWTON_STEPS = 30
MAX_STEP = 0.2
# The Lagrange solve's tolerances (PySCF's defaults: rtol 1e-7, atol 1e-12).
LAGRANGE_RTOL = 1e-12
LAGRANGE_ATOL = 1e-14
# Root 1 must match root 1 across each Newton step at least this well:
# src/pipeline/optimize's kStateOverlapFloor.
ROOT_FOLLOW_FLOOR = 0.9
# The SA-CASSCF rerun from a fresh start at the minimum must reach the
# carried one's root-1 energy to this (hartree).
FRESH_AGREEMENT = 1e-9
# An eigenvalue of the projected Hessian below this in magnitude is one of
# its six rigid zeros.
RIGID_ZERO = 1e-6
# The out-of-plane angle (degrees) a pyramidal minimum must exceed.
PYRAMIDAL_ANGLE = 1.0


def placed(coords: numpy.ndarray) -> dict:
    return dict(
        CASE,
        atoms=[(s, tuple(r)) for (s, _), r in zip(CASE["atoms"], coords, strict=True)],
    )


def fresh_start(mol: gto.Mole) -> numpy.ndarray:
    """The exact RHF's orbitals: the frontier active space."""
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, "RHF did not converge"
    return mf.mo_coeff


def carried(mol: gto.Mole, mo: numpy.ndarray) -> numpy.ndarray:
    """mo Lowdin-orthonormalized in mol's overlap."""
    overlap = mo.T @ mol.intor("int1e_ovlp") @ mo
    values, vectors = numpy.linalg.eigh(overlap)
    return mo @ vectors @ numpy.diag(values**-0.5) @ vectors.T


def evaluate(coords: numpy.ndarray, mo: numpy.ndarray | None) -> dict:
    """The polished DF-SA-CASSCF at coords, from mo carried over (a fresh
    start when None): its roots' energies and CI, root STATE's gradient, its
    orbitals and its molecule."""
    case = placed(coords)
    weights = case["weights"]
    mol = molecule(case)
    auxbasis, _ = aux_basis(case, mol)
    start = fresh_start(mol) if mo is None else carried(mol, mo)
    mc = build(case, mol, auxbasis, start)
    optimize(mc, mc.kernel, start)
    assert mc.converged, "mc1step did not converge"
    mo = polish(case["name"], mc, mc.mo_coeff, weights)
    # The gradient at the polished orbitals and their dense CI, not at
    # mc1step's own stop (|g| near conv_tol_grad).
    energies, vectors, _ = spin_roots(mc, mo, len(weights))
    mc.mo_coeff, mc.ci = mo, list(vectors)
    mc.fcisolver.e_states = list(energies)
    grad = mc.nuc_grad_method()
    grad.verbose = 0
    grad.conv_rtol, grad.conv_atol = LAGRANGE_RTOL, LAGRANGE_ATOL
    gradient = grad.kernel(state=STATE)
    assert grad.converged, "the Lagrange solve did not converge"
    return {
        "energies": [float(e) for e in energies],
        "ci": list(vectors),
        "gradient": gradient,
        "mo": mo,
        "mol": mol,
        "ncore": mc.ncore,
        "nelecas": mc.nelecas,
    }


def hessian_at(coords: numpy.ndarray, mo: numpy.ndarray) -> numpy.ndarray:
    """The central difference of root STATE's gradient, symmetrized."""
    n = coords.size
    hessian = numpy.zeros((n, n))
    for i in range(n):
        for s in (1, -1):
            moved = coords.ravel().copy()
            moved[i] += s * HESSIAN_STEP
            moved_gradient = evaluate(moved.reshape(coords.shape), mo)["gradient"]
            hessian[i] += s * moved_gradient.ravel()
    return 0.5 * (hessian + hessian.T) / (2.0 * HESSIAN_STEP)


def internal_spectrum(coords: numpy.ndarray, hessian: numpy.ndarray):
    """The projected Hessian's internal eigenvalues and eigenvectors."""
    projector = internal_projector(coords)
    values, vectors = numpy.linalg.eigh(projector @ hessian @ projector)
    internal = numpy.abs(values) > RIGID_ZERO
    assert internal.sum() == coords.size - 6, values
    return values[internal], vectors[:, internal]


def projected_force(coords: numpy.ndarray, gradient: numpy.ndarray) -> float:
    return float(numpy.abs(internal_projector(coords) @ gradient.ravel()).max())


def followed_overlap(last: dict, now: dict) -> float:
    """|<STATE(last)|STATE(now)>| over the norm of <STATE(last)|J(now)>
    over every root J, refused unless STATE matches STATE: the matched share
    of the root within the averaged space, as src/pipeline/optimize's
    match_roots takes it."""
    s = ci_overlap(
        last["mol"],
        last["mo"],
        last["ci"],
        now["mol"],
        now["mo"],
        now["ci"],
        now["ncore"],
        CASE["cas"][1],
        now["nelecas"],
    )
    row = s[STATE] / numpy.linalg.norm(s[STATE])
    best = int(numpy.argmax(numpy.abs(row)))
    overlap = float(abs(row[best]))
    assert best == STATE, f"root {STATE} matched root {best}: {s}"
    assert overlap >= ROOT_FOLLOW_FLOOR, f"root {STATE} overlap {overlap}"
    return overlap


def out_of_plane(coords: numpy.ndarray) -> float:
    """The angle (degrees) between C=O and the H-C-H plane, at the carbon."""
    c, o, h1, h2 = coords
    normal = numpy.cross(h1 - c, h2 - c)
    normal /= numpy.linalg.norm(normal)
    bond = (o - c) / numpy.linalg.norm(o - c)
    return float(numpy.degrees(numpy.arcsin(abs(normal @ bond))))


def atoms_out(coords: numpy.ndarray) -> list[dict]:
    return [
        {"symbol": symbol, "z": gto.charge(symbol), "coord": [float(x) for x in r]}
        for (symbol, _), r in zip(CASE["atoms"], coords, strict=True)
    ]


def main() -> None:
    lib.num_threads(THREADS)
    start = numpy.array([coord for _, coord in CASE["atoms"]], dtype=float)
    coords = start.copy()
    here = evaluate(coords, None)
    start_energies = here["energies"]
    steps = 0
    min_overlap = 1.0
    print(
        f"start: E {here['energies'][STATE]:.12f} "
        f"max |Pg| {projected_force(coords, here['gradient']):.2e}",
        flush=True,
    )
    while projected_force(coords, here["gradient"]) >= FORCE_TOLERANCE:
        assert steps < MAX_NEWTON_STEPS, "Newton did not converge"
        values, vectors = internal_spectrum(coords, hessian_at(coords, here["mo"]))
        projector = internal_projector(coords)
        step = -(vectors / numpy.abs(values)) @ (
            vectors.T @ (projector @ here["gradient"].ravel())
        )
        step *= min(1.0, MAX_STEP / numpy.abs(step).max())
        coords = coords + step.reshape(coords.shape)
        last, here = here, evaluate(coords, here["mo"])
        min_overlap = min(min_overlap, followed_overlap(last, here))
        steps += 1
        print(
            f"newton {steps}: E {here['energies'][STATE]:.12f} "
            f"max |Pg| {projected_force(coords, here['gradient']):.2e} "
            f"least curvature {values.min():.4e} oop {out_of_plane(coords):.3f}",
            flush=True,
        )
    values, _ = internal_spectrum(coords, hessian_at(coords, here["mo"]))
    k_min = float(values.min())
    assert k_min > 0.0, f"not a minimum: the least curvature is {k_min}"
    angle = out_of_plane(coords)
    assert angle > PYRAMIDAL_ANGLE, f"planar: out of plane by {angle} degrees"
    fresh = evaluate(coords, None)
    miss = abs(fresh["energies"][STATE] - here["energies"][STATE])
    assert miss < FRESH_AGREEMENT, f"a fresh start is off by {miss}"
    max_force = projected_force(coords, here["gradient"])
    print(f"k_min {k_min:.6e}, out of plane {angle:.4f} deg, fresh by {miss:.1e}")

    mol = molecule(placed(coords))
    _, aux_provenance = aux_basis(CASE, mol)
    nelecas, ncas = CASE["cas"]
    doc = {
        "description": "PySCF's DF-SA-CASSCF minimum of one root "
        "(df.grad.sacasscf by Newton's method) over this tree's data/basis and "
        "data/aux_basis files, from a non-planar start. Hartree, bohr",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "command": COMMAND,
            "threads": THREADS,
        },
        "cases": [
            {
                "name": CASE["name"],
                "from": "casscf_nac/casscf_nac.json h2co_skew_ccpvdz_sa3's atoms",
                "basis": CASE["basis"],
                "aux_basis": aux_provenance,
                "charge": 0,
                "ncas": ncas,
                "nelecas": nelecas,
                "weights": CASE["weights"],
                "state": STATE,
                "start": atoms_out(start),
                "start_energies": start_energies,
                "atoms": atoms_out(coords),
                "energies": here["energies"],
                "energy": here["energies"][STATE],
                "max_force": max_force,
                "k_min": k_min,
                "out_of_plane_degrees": angle,
                "min_root_overlap": min_overlap,
                "fresh_start_energy": fresh["energies"][STATE],
                "newton_steps": steps,
                "optimizer": "Newton's method over PySCF's analytic DF-SA-CASSCF "
                f"gradient of root {STATE} (df.grad.sacasscf), the Hessian a "
                f"central difference of it (step {HESSIAN_STEP} bohr) recomputed "
                "every step, rigid translations and rotations projected out, each "
                "internal direction stepped by 1/|curvature|, steps capped at "
                f"{MAX_STEP} bohr per coordinate; converged to max |P dE/dR| < "
                f"{FORCE_TOLERANCE}, P the rigid projector",
            }
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
