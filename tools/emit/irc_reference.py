#!/usr/bin/env python3
"""Write the references the `[irc]` tests read (#1269).

Two files under test/gtest/data/irc/, both over this tree's own data/basis
(and data/aux_basis) files handed to PySCF as explicit shells:

**hcn_rhf.json: an independent IRC.** HCN <-> HNC at RHF/STO-3G, exact
(integral-direct) ERIs. The transition state is found by Newton's method on
PySCF's analytic gradient with its analytic Hessian (`hessian.rhf`), rigid
motions projected out, to max |dE/dR| < TS_FORCE; its Hessian there, and
`pyscf.hessian.thermo.harmonic_analysis`'s frequencies and normal modes (the
tree's masses and mode convention, `norm_mode`), are written at the top level
under the keys a `[hessian]` writes, `hessian`, `frequencies` and `modes`, so
the file is itself a `[hessian] from = ...` start. The imaginary mode's phase
is fixed so its largest component is positive: forward is that phase.

The IRC is Gonzalez and Schlegel's (GS2) in mass-weighted Cartesians,
q = sqrt(m) x, the masses `elements.MASSES` (harmonic_analysis's), at
STEP_SIZE amu^1/2 bohr, each side from the TS along +-the mode: point k + 1
is the minimum of E on the hypersphere of radius s/2 about the pivot
q_k + (s/2) u, u the mode at the TS and the steepest descent -g/|g| after.
**The sphere is solved independently of the tree's method**: not by a model
Hessian and the secular equation, but as an unconstrained minimization over
the direction, q = q* + (s/2) B v / |B v| (B the internal coordinates at the
pivot), by SciPy's BFGS on PySCF's exact energy and gradient, until the
gradient's part tangent to the sphere has 2-norm under REF_TANGENTIAL
(checked afresh at the point), the BFGS polished by a root of the tangential
gradient itself. A side stops by the tree's own rules: where the walk's
own mass-weighted gradient norm halved over the last step,
2 |g_k| <= |g_(k-1)| (the path's minimum within a step, #1278), at a point
whose energy rose or that lies behind its pivot (not kept), or past
MAX_POINTS.

**What a test derives its bars from**, per point: the energy, the
coordinates, the tangential gradient, lambda = g.p / |p|^2 (the sphere's
multiplier, p = q - q*), and kappa, the least eigenvalue of H - lambda I
(mass-weighted, exact) over the sphere's tangent space within the internal
coordinates: the curvature of E on the sphere at its minimum, so a point
whose tangential gradient is t lies within t / kappa of it. At each side's
end the RHF minimum it descends to (Newton, analytic Hessian, to
max |dE/dR| < MIN_FORCE), its energy and the largest mass-weighted Hessian
eigenvalue there.

**nh3_casscf.json: a correlated chain's start.** NH3's inversion at
DF-CASSCF(8,7)/STO-3G over cc-pVDZ-JKFIT, the full valence space (every
orbital but N 1s, so no virtual is left to rotate in): the planar D3h
transition state, found by
solving dE/dr = 0 for the one bond length the symmetry leaves (brentq on
PySCF's analytic DF-CASSCF gradient, `df.grad.casscf`, its CASSCF over a
DF-RHF sharing its fit), with its gradient, the frequencies of a
central-difference Hessian (HESSIAN_STEP bohr; exactly one imaginary, the
umbrella, or refused), each displaced CASSCF started from the TS's
orbitals. The full valence space is the point: a CAS(2,2) of the frontier
pair has a second, lower solution off the plane, which both PySCF's fresh
starts and the tree's carried orbitals fell into mid-path. No minimum is
written; the test holds the ends to being minima by the tree's own analytic
Hessian there.

**Contraction norms (#1220).** PySCF renormalizes each contraction, the tree
does not; nothing in the AO basis is written here, and energies, gradients,
Hessians and geometries are invariant to the AO scaling.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/irc_reference.py

writes both files. Re-run it to regenerate; they are never hand-edited. BLAS
and OpenMP run at one thread; nothing run-dependent (date, host) is written.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json  # noqa: E402
import pathlib  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
import pyscf.hessian.rhf  # noqa: E402
import scipy  # noqa: E402
from gradient_reference import internal_projector  # noqa: E402
from pyscf import gto, lib, mcscf, scf  # noqa: E402
from pyscf.df.grad import casscf as df_casscf_grad  # noqa: E402
from pyscf.hessian import thermo  # noqa: E402
from scf_reference import pyscf_basis  # noqa: E402
from scipy import optimize  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "test" / "gtest" / "data" / "irc"

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/emit/irc_reference.py"
)

ANGSTROM = 1.0 / lib.param.BOHR

# HCN's TS search starts here (Angstrom): H bridging, nearer C.
HCN_GUESS = [
    ("C", (0.0, 0.0, 0.0)),
    ("N", (1.18, 0.0, 0.0)),
    ("H", (0.55, 1.1, 0.0)),
]
TS_FORCE = 1e-8
MIN_FORCE = 1e-7
STEP_SIZE = 0.1
REF_TANGENTIAL = 1e-7
MAX_POINTS = 60
HESSIAN_STEP = 1e-3
NH3_BOND_BRACKET = (1.80, 2.00)


def molecule(symbols, coords, basis_file: str) -> gto.Mole:
    return gto.M(
        atom=[[s, tuple(r)] for s, r in zip(symbols, coords, strict=True)],
        unit="Bohr",
        basis=pyscf_basis(basis_file, set(symbols)),
        cart=False,
        verbose=0,
    )


def converged_rhf(mol: gto.Mole, dm0=None) -> scf.hf.SCF:
    def rhf() -> scf.hf.SCF:
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-13
        mf.conv_tol_grad = 1e-10
        mf.max_cycle = 200
        return mf

    mf = rhf()
    try:
        mf.kernel(dm0)
    except numpy.linalg.LinAlgError:
        # Started from a density converged at a geometry next door, DIIS's
        # error vectors are at rounding, and its matrix can be singular enough
        # for LAPACK's eigh to fail (HCN's forward sphere 22, #1278): the SCF
        # again from the atomic guess.
        mf = rhf()
        mf.kernel()
    assert mf.converged
    return mf


def masses_of(symbols) -> numpy.ndarray:
    return numpy.array([pyscf.data.elements.MASSES[gto.charge(s)] for s in symbols])


def rigid_mw(coords: numpy.ndarray, masses: numpy.ndarray) -> numpy.ndarray:
    """The orthonormal rigid motions in mass-weighted coordinates (columns)."""
    centred = coords - coords.mean(axis=0)
    sqrt_m = numpy.repeat(numpy.sqrt(masses), 3)
    rigid = [numpy.tile(numpy.eye(3)[k], len(coords)) * sqrt_m for k in range(3)]
    rigid += [numpy.cross(numpy.eye(3)[k], centred).ravel() * sqrt_m for k in range(3)]
    q, r = numpy.linalg.qr(numpy.array(rigid).T)
    keep = numpy.abs(numpy.diag(r)) > 1e-8 * numpy.abs(numpy.diag(r)).max()
    return q[:, keep]


def internal_mw(coords: numpy.ndarray, masses: numpy.ndarray) -> numpy.ndarray:
    """An orthonormal basis of the complement of the rigid motions (columns)."""
    q = rigid_mw(coords, masses)
    w, v = numpy.linalg.eigh(numpy.eye(coords.size) - q @ q.T)
    return v[:, w > 0.5]


def harmonic(mol: gto.Mole, hessian: numpy.ndarray):
    """harmonic_analysis's frequencies (cm^-1, imaginary negative) and modes."""
    n = mol.natm
    out = thermo.harmonic_analysis(
        mol, hessian.reshape(n, 3, n, 3).transpose(0, 2, 1, 3), imaginary_freq=False
    )
    return numpy.array(out["freq_wavenumber"]), out["norm_mode"].reshape(-1, 3 * n)


def rows(matrix: numpy.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in matrix]


def atoms_json(symbols, coords) -> list[dict]:
    return [
        {"symbol": s, "coord": [float(x) for x in r]}
        for s, r in zip(symbols, coords, strict=True)
    ]


class Rhf:
    """RHF/STO-3G energies, gradients and Hessians of one molecule's atoms,
    each SCF started from the last density."""

    def __init__(self, symbols):
        self.symbols = symbols
        self.dm = None
        self.evaluations = 0

    def scf(self, x: numpy.ndarray) -> scf.hf.SCF:
        mf = converged_rhf(
            molecule(self.symbols, x.reshape(-1, 3), "sto-3g.json"), self.dm
        )
        self.dm = mf.make_rdm1()
        return mf

    def energy_gradient(self, x: numpy.ndarray):
        self.evaluations += 1
        mf = self.scf(x)
        return float(mf.e_tot), mf.nuc_grad_method().kernel().ravel()

    def hessian(self, x: numpy.ndarray) -> numpy.ndarray:
        mf = self.scf(x)
        n = len(self.symbols)
        h = pyscf.hessian.rhf.Hessian(mf).kernel()
        return h.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)


def newton(model: Rhf, x: numpy.ndarray, force: float, label: str) -> numpy.ndarray:
    """Newton's method on the gradient with the analytic Hessian, rigid motions
    projected out, every internal eigenvalue taken as it is: it converges to
    the stationary point whose quadratic region it starts in."""
    for _ in range(50):
        _, g = model.energy_gradient(x)
        if numpy.abs(g).max() < force:
            return x
        p = internal_projector(x.reshape(-1, 3))
        w, v = numpy.linalg.eigh(p @ model.hessian(x) @ p)
        step = numpy.zeros_like(x)
        for i in range(len(w)):
            if abs(w[i]) > 1e-6:
                step -= v[:, i] * (v[:, i] @ g) / w[i]
        norm = numpy.linalg.norm(step)
        x = x + (step if norm < 0.3 else step * 0.3 / norm)
    raise AssertionError(f"{label}: Newton did not converge")


def irc_side(model: Rhf, x_ts, hessian_ts, mode, sign: float, masses) -> dict:
    """One side of GS2, each sphere solved by BFGS over its direction."""
    sqrt_m = numpy.repeat(numpy.sqrt(masses), 3)
    radius = 0.5 * STEP_SIZE
    q = x_ts * sqrt_m
    e, g = model.energy_gradient(x_ts)
    points = [
        {
            "arc_length": 0.0,
            "energy": e,
            "coords": [float(v) for v in x_ts],
            "tangential_gradient": 0.0,
            "lam": 0.0,
            "kappa": None,
        }
    ]
    leave = mode * sqrt_m
    leave = internal_mw(x_ts.reshape(-1, 3), masses) @ (
        internal_mw(x_ts.reshape(-1, 3), masses).T @ leave
    )
    leave = sign * leave / numpy.linalg.norm(leave)
    stop = "max_points"
    g_previous = 0.0
    for k in range(MAX_POINTS):
        basis = internal_mw((q / sqrt_m).reshape(-1, 3), masses)
        g_mw = basis @ (basis.T @ (g / sqrt_m))
        g_norm = numpy.linalg.norm(g_mw)
        if k == 0:
            down = leave
        else:
            down = -g_mw / g_norm
            # The path's minimum within a step: |g| halved over the last one.
            if 2.0 * g_norm <= g_previous:
                stop = "minimum_within_step"
                break
        g_previous = g_norm
        pivot = q + radius * down
        basis = internal_mw((pivot / sqrt_m).reshape(-1, 3), masses)

        def on_sphere(v, pivot=pivot, basis=basis):
            w = basis @ v
            return pivot + radius * w / numpy.linalg.norm(w)

        def objective(v, pivot=pivot, basis=basis):
            w = basis @ v
            norm = numpy.linalg.norm(w)
            w_hat = w / norm
            energy, gradient = model.energy_gradient((pivot + radius * w_hat) / sqrt_m)
            gm = gradient / sqrt_m
            tangent = gm - (w_hat @ gm) * w_hat
            return energy, basis.T @ ((radius / norm) * tangent)

        # BFGS on the energy, then a root of the tangential gradient itself
        # (MINPACK's hybrid method, its Jacobian by differences): near the
        # sphere's minimum the energy changes by t^2 / 2 kappa, under the
        # energy's own rounding once t is about 1e-7, where BFGS stalls; the
        # gradient keeps resolving the point below that.
        result = optimize.minimize(
            objective,
            basis.T @ down,
            jac=True,
            method="BFGS",
            options={"gtol": 0.05 * radius * REF_TANGENTIAL, "maxiter": 500},
        )
        evaluations = result.nfev
        v = result.x / numpy.linalg.norm(basis @ result.x)
        polish = optimize.root(
            lambda u: objective(u)[1], v, method="hybr", options={"xtol": 1e-13}
        )
        evaluations += polish.nfev
        v = polish.x
        q_new = on_sphere(v)
        x_new = q_new / sqrt_m
        e_new, g_new = model.energy_gradient(x_new)
        inner = internal_mw(x_new.reshape(-1, 3), masses)
        gm = inner @ (inner.T @ (g_new / sqrt_m))
        p = q_new - pivot
        lam = (gm @ p) / (p @ p)
        tangential = numpy.linalg.norm(gm - lam * p)
        assert tangential < REF_TANGENTIAL, f"point {k + 1}: tangential {tangential}"
        # Past the minimum, as the tree stops: a rise, or a point behind its
        # pivot.
        if e_new > e or (q_new - pivot) @ down < 0.0:
            stop = "energy_rose"
            break
        # The curvature of E on the sphere: H - lambda I over the tangent
        # space inside the internal coordinates.
        h_mw = model.hessian(x_new) / numpy.outer(sqrt_m, sqrt_m)
        p_hat = p / numpy.linalg.norm(p)
        tangent = inner - numpy.outer(p_hat, p_hat @ inner)
        u, s, _ = numpy.linalg.svd(tangent, full_matrices=False)
        tangent = u[:, s > 0.5]
        kappa = numpy.linalg.eigvalsh(
            tangent.T @ (h_mw - lam * numpy.eye(len(q))) @ tangent
        ).min()
        assert kappa > 0.0, f"point {k + 1}: the sphere's point is not a minimum"
        points.append(
            {
                "arc_length": sign * STEP_SIZE * (k + 1),
                "energy": e_new,
                "coords": [float(v) for v in x_new],
                "tangential_gradient": float(tangential),
                "lam": float(lam),
                "kappa": float(kappa),
            }
        )
        print(
            f"  {'forward' if sign > 0 else 'backward'} {k + 1:3d} E {e_new:.10f} "
            f"|g_t| {tangential:.1e} lambda {lam:+.4f} kappa {kappa:.4f} "
            f"({evaluations} evaluations)"
        )
        q, e, g = q_new, e_new, g_new
    return {"stop": stop, "points": points}


def hcn() -> dict:
    symbols = [s for s, _ in HCN_GUESS]
    masses = masses_of(symbols)
    model = Rhf(symbols)
    x = numpy.array([r for _, r in HCN_GUESS]).ravel() * ANGSTROM
    x = newton(model, x, TS_FORCE, "hcn TS")
    energy, gradient = model.energy_gradient(x)
    hessian = model.hessian(x)
    hessian = 0.5 * (hessian + hessian.T)
    mol = molecule(symbols, x.reshape(-1, 3), "sto-3g.json")
    frequencies, modes = harmonic(mol, hessian)
    assert (frequencies < 0).sum() == 1, frequencies
    if modes[0][numpy.argmax(numpy.abs(modes[0]))] < 0:
        modes[0] = -modes[0]
    print(f"hcn TS E {energy:.12f}, frequencies {frequencies}")
    sides = {}
    minima = {}
    sqrt_m = numpy.repeat(numpy.sqrt(masses), 3)
    for name, sign in (("forward", 1.0), ("backward", -1.0)):
        sides[name] = irc_side(model, x, hessian, modes[0], sign, masses)
        end = numpy.array(sides[name]["points"][-1]["coords"])
        xm = newton(model, end, MIN_FORCE, f"hcn {name} minimum")
        em, gm = model.energy_gradient(xm)
        h_mw = model.hessian(xm) / numpy.outer(sqrt_m, sqrt_m)
        inner = internal_mw(xm.reshape(-1, 3), masses)
        curvatures = numpy.linalg.eigvalsh(inner.T @ h_mw @ inner)
        assert curvatures.min() > 0.0, f"{name}: not a minimum"
        minima[name] = {
            "energy": em,
            "coords": [float(v) for v in xm],
            "max_force": float(numpy.abs(gm).max()),
            "max_curvature": float(curvatures.max()),
            "min_curvature": float(curvatures.min()),
        }
        print(
            f"hcn {name}: {len(sides[name]['points']) - 1} points, minimum E {em:.12f}"
        )
    return {
        "description": "An RHF/STO-3G IRC of HCN <-> HNC by GS2, each hypersphere "
        "minimized by BFGS over PySCF's exact energy and gradient; the TS's "
        "Hessian, frequencies and modes under [hessian]'s keys; coordinates in "
        "bohr, energies in hartree, arc lengths in amu^1/2 bohr, lambda and "
        "kappa (mass-weighted) in hartree / (amu bohr^2)",
        "provenance": {
            "pyscf": pyscf.__version__,
            "scipy": scipy.__version__,
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "basis": "sto-3g",
        "charge": 0,
        "atoms": atoms_json(symbols, x.reshape(-1, 3)),
        "energy": energy,
        "max_force": float(numpy.abs(gradient).max()),
        "hessian": rows(hessian),
        "frequencies": [float(f) for f in frequencies],
        "modes": rows(modes),
        "step_size": STEP_SIZE,
        "reference_tangential": REF_TANGENTIAL,
        "forward": sides["forward"],
        "backward": sides["backward"],
        "minima": minima,
    }


# NH3's full valence space in STO-3G: every orbital but N 1s.
NH3_CAS = (7, 8)


class DfCasscf:
    """DF-CASSCF(8,7) energies and gradients over cc-pVDZ-JKFIT, each CASSCF
    started from the orbitals of `start` (the last call's by default),
    projected onto the new geometry's basis, as the tree's chains carry
    them."""

    def __init__(self, symbols):
        self.symbols = symbols
        self.aux = pyscf_basis("../aux_basis/cc-pVDZ-JKFIT.json", set(symbols))
        self.last = None

    def energy_gradient(self, x: numpy.ndarray, start=None):
        mol = molecule(self.symbols, x.reshape(-1, 3), "sto-3g.json")
        mf = scf.RHF(mol).density_fit(auxbasis=self.aux)
        mf.conv_tol = 1e-13
        mf.kernel()
        assert mf.converged
        mc = mcscf.CASSCF(mf, *NH3_CAS)
        assert mc.with_df is mf.with_df, "the CASSCF and its _scf fit apart"
        # 1e-12 stalls PySCF's CASSCF at some pyramidal geometries; 1e-10
        # converges them all, far below anything a test reads off this file.
        mc.conv_tol = 1e-10
        mc.conv_tol_grad = 1e-6
        mc.max_cycle_macro = 200
        mc.verbose = 0
        start = start if start is not None else self.last
        mo0 = (
            None if start is None else mcscf.project_init_guess(mc, start[0], start[1])
        )
        mc.kernel(mo0)
        assert mc.converged
        self.last = (mc.mo_coeff, mol)
        return float(mc.e_tot), df_casscf_grad.Gradients(mc).kernel().ravel()

    def hessian(self, x: numpy.ndarray) -> numpy.ndarray:
        """Central differences of the gradient, every displaced CASSCF started
        from the reference's orbitals."""
        self.energy_gradient(x)
        reference = self.last
        n = len(x)
        out = numpy.zeros((n, n))
        for i in range(n):
            d = numpy.zeros(n)
            d[i] = HESSIAN_STEP
            plus = self.energy_gradient(x + d, reference)[1]
            minus = self.energy_gradient(x - d, reference)[1]
            out[i] = (plus - minus) / (2 * HESSIAN_STEP)
        self.last = reference
        return 0.5 * (out + out.T)


def planar_nh3(r: float) -> numpy.ndarray:
    angles = numpy.radians([90.0, 210.0, 330.0])
    h = [(r * numpy.cos(a), r * numpy.sin(a), 0.0) for a in angles]
    return numpy.array([(0.0, 0.0, 0.0), *h]).ravel()


def nh3() -> dict:
    symbols = ["N", "H", "H", "H"]
    model = DfCasscf(symbols)

    def slope(r: float) -> float:
        h = 1e-6
        dx = (planar_nh3(r + h) - planar_nh3(r - h)) / (2 * h)
        return float(model.energy_gradient(planar_nh3(r))[1] @ dx)

    r = optimize.brentq(slope, *NH3_BOND_BRACKET, xtol=1e-12)
    x = planar_nh3(r)
    energy, gradient = model.energy_gradient(x)
    hessian = model.hessian(x)
    frequencies, _ = harmonic(
        molecule(symbols, x.reshape(-1, 3), "sto-3g.json"), hessian
    )
    assert (frequencies < 0).sum() == 1, frequencies
    print(f"nh3 TS r {r:.8f}, E {energy:.12f}, frequencies {frequencies}")

    return {
        "description": "NH3's inversion at DF-CASSCF(8,7)/STO-3G over cc-pVDZ-JKFIT: "
        "the planar TS and its frequencies (central-difference Hessian); "
        "coordinates in bohr, energies in hartree, frequencies in cm^-1",
        "provenance": {
            "pyscf": pyscf.__version__,
            "scipy": scipy.__version__,
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "basis": "sto-3g",
        "aux_basis": "cc-pvdz-jkfit",
        "cas": [NH3_CAS[1], NH3_CAS[0]],
        "charge": 0,
        "atoms": atoms_json(symbols, x.reshape(-1, 3)),
        "energy": energy,
        "max_force": float(numpy.abs(gradient).max()),
        "frequencies": [float(f) for f in frequencies],
    }


def main() -> None:
    lib.num_threads(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, make in (("hcn_rhf.json", hcn), ("nh3_casscf.json", nh3)):
        doc = make()
        (OUT_DIR / name).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
