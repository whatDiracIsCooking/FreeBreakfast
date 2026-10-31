#!/usr/bin/env python3
"""Write the numerical-gradient references the DF-NEVPT2 gradient tests read.

No reference program has a NEVPT2 nuclear gradient this tree can import, so
the oracle is the numerical gradient of re-converged energies, at the same
settings and over the same fitting set the tree uses (this tree's
data/aux_basis/cc-pVDZ-JKFIT.json, parsed as casci_reference.py parses it):
`.temp/nevpt2/nuclear_gradient.tex`, "Checks", "The oracle".

At each geometry, for a single-state closed-shell DF-CASSCF:

- **CASSCF.** casscf_reference.py's DF mc1step from the reference's polished
  orbitals carried over (Lowdin-orthonormalized in the displaced overlap),
  then polished to |g| < POLISH_GRAD as nevpt2_reference.py polishes it,
  but against one orbital Hessian built at mc1step's orbitals (`polished`).
- **SC-NEVPT2.** nevpt2_reference.py's: PySCF's `mrpt.NEVPT` over a DFCASCI on
  the polished orbitals (the exact RHF's orbitals wrapped as a DF-RHF over
  the same fitting set), its CI vector swapped for the dense singlet, core and
  virtuals canonicalized over the root's Fock and the active orbitals left
  unrotated (`cas_natorb=False`), as the C++ step leaves them. Every point
  must take PySCF's DF branch, and refuses two canonical core or virtual
  energies within DEGENERATE.
- **PC-NEVPT2.** block2's `WickICNEVPT2` on that same state over PySCF's own
  fitted integrals, through nevpt2_reference.py's `block2`, with its gap gate
  (PC_MIN_GAP at PC_TAU) at every point.

At the reference geometry alone, every one of nevpt2_reference.py's checks
runs as well: the SC active-rotation invariance (cas_natorb=True against
False), block2's SC against PySCF's per class, PC Sijrs against SC Sijrs, and
PC on the natural orbitals against the unrotated one.

**The differences.** For every Cartesian component (atom A, axis k), the
energies at R0 +- h e_Ak for each h of the case's steps (bohr, halving), so 6 x 3N
displaced points per case; the central difference D(h) = (E(+h) - E(-h)) / 2h
at each step, and one Richardson step over the two finest,
(4 D(h/2) - D(h)) / 3, with nevpt2-gpu-demo's error bar beside it: its
distance from the same step over the two coarser. For the CASSCF, SC and PC
total energies alike.

**The continuity guard.** A displaced optimisation that lands on another
stationary point gives a number from another reference, so every displaced
CASSCF must, before it is differenced, (1) have converged and polished, (2)
sit within PREDICTION_BOUND h^2 of the first-order prediction E0 + g . dR,
g being PySCF's analytic DF-CASSCF gradient at the reference
(`df.grad.casscf`, over a DF _scf sharing the CASSCF's fit, as
casscf_gradient_reference.py builds it), (3) have an active space whose
smallest singular value of overlap with the reference's (the AO cross
overlap between the two geometries) is above 1 - SUBSPACE_TOL, and (4) a CI
vector whose overlap with the reference's, through that orbital overlap, is
above 1 - CI_TOL in magnitude. And every displaced PC must keep exactly the
reference's number of modes over all its solves, so no displacement crosses a
rank change. A miss refuses emission.

**The checks.** Per component, the Richardson estimate and the finest
central difference agree to RICHARDSON_TOL (the gate the consumers use is
1e-6 hartree/bohr, so this is "well inside" it); the step ladder's ratio
(D(4h) - D(2h)) / (D(2h) - D(h)) lies in (3.2, 4.8) wherever its denominator is
above LADDER_FLOOR; the translational sum of each Richardson gradient is below
TRANSLATION_TOL; and the CASSCF's Richardson gradient matches PySCF's
analytic one to ANALYTIC_TOL, which checks the differencing machinery itself.

BLAS and OpenMP run on one thread and nothing run-dependent (date, host) is
written, so a rerun on the same PySCF, block2 and numpy is byte-identical:

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/nevpt2_gradient_reference.py

writes test/gtest/data/nevpt2_gradient/nevpt2_gradient.json. Never
hand-edited.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402
from importlib.metadata import version  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402

# isort: split
# casscf_reference pins its own thread count on import; numpy is loaded by
# then, so this module's single thread stands (and main() pins PySCF's).
import casscf_reference  # noqa: E402

# nevpt2_reference imports casscf_reference's dense-singlet helper under the
# name it had before casci, casscf: "over ROHF orbitals" renamed it
# spin_roots (same signature and, for a closed shell, the same roots).
if not hasattr(casscf_reference, "singlets"):
    casscf_reference.singlets = casscf_reference.spin_roots

# isort: split
from casci_reference import aux_basis  # noqa: E402
from casscf_gradient_reference import fitted_scf  # noqa: E402
from casscf_reference import (  # noqa: E402
    FORMALDEHYDE,
    POLISH_CURVATURE,
    POLISH_GRAD,
    POLISH_STEPS,
    ROOT_GAP,
    SPIN_S2,
    casscf,
    check_aux,
    derivatives,
    molecule,
    optimize,
)
from df_reference import WATER_SKEW  # noqa: E402
from fci_reference import dense_hamiltonian  # noqa: E402
from nevpt2_reference import (  # noqa: E402
    BLOCK2_SC,
    CASCI_AGREEMENT,
    CLASSES,
    DEGENERATE,
    INVARIANCE,
    PC_MIN_GAP,
    PC_TAU,
    SAME_STATE,
    block2,
    canonical,
    nevpt,
)
from pyblock2.icmr import icnevpt2_full  # noqa: E402
from pyscf import ao2mo, fci, gto, lib, mcscf, scf  # noqa: E402
from pyscf.df.grad import casscf as dfcasscf_grad  # noqa: E402
from pyscf.mcscf import mc1step  # noqa: E402
from pyscf.mcscf.df import _DFCAS  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "nevpt2_gradient" / "nevpt2_gradient.json"
COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/nevpt2_gradient_reference.py"
)

# Single-state closed-shell singlets, cc-pVDZ over cc-pVDZ-JKFIT, neither
# with any symmetry a displacement could break: water off its symmetry
# (casscf_gradient.json's h2o_skew_ccpvdz; smallest canonical virtual gap
# 3.0e-3), and formaldehyde, the larger one and the one far from any inactive
# degeneracy (smallest core gap 0.24, virtual 0.016), over the RHF's six
# orbitals 5-10 about the gap (casscf_df.json's single-state frontier
# CAS(4,3) polishes to a saddle, H's least eigenvalue -2.6e-3, and so do
# CAS(4,4) over 6-9 and 6-8 + 10 and CAS(2,2)). `active` is the CASSCF's
# start over the RHF orbitals, as the [casscf] table takes it.
#
# Not symmetric water (casscf_df.json's h2o_ccpvdz): its two lowest
# canonical virtuals are 1.4e-3 apart and of different C2v symmetry, a
# symmetry-breaking displacement mixes them, and SC, which is not invariant
# to that rotation, curves ever more sharply as the step shrinks (along an H's
# in-plane y: d2E/dy2 1.09, 1.75, 2.19, 2.34 at h = 2e-3 .. 2.5e-4, PC's flat at
# 0.3866), so no step ladder converges and the ladder check refuses it. That
# is the SC hazard of nuclear_gradient.tex (near-degenerate inactive
# orbitals), and the analytic gradient there is right but this oracle cannot
# see it.
CASES = [
    {
        "name": "h2o_skew_ccpvdz",
        "atoms": WATER_SKEW,
        "basis": "cc-pvdz",
        "cas": (4, 4),
        "active": [2, 3, 5, 6],
        "steps": (5e-4, 2.5e-4, 1.25e-4),
    },
    {
        "name": "h2co_ccpvdz",
        "atoms": FORMALDEHYDE,
        "basis": "cc-pvdz",
        "cas": (6, 6),
        "active": [5, 6, 7, 8, 9, 10],
        "steps": (2e-3, 1e-3, 5e-4),
    },
]

# The central-difference steps (bohr), halving; Richardson over the last two.
# Each case's central-difference steps (bohr, "steps" above), halving;
# Richardson over the last two. A step is a balance: the Richardson residual
# falls as h^2 and the energies' noise (about 1e-11 hartree in SC and PC,
# under 1e-12 in the CASSCF) grows as 1/h. Water's SC has a third derivative
# near 25 hartree/bohr^3 (its 3.0e-3 virtual gap): with the finest step at
# 2.5e-4 the residual is 2.7e-7, at 1.25e-4 6.6e-8, so it takes the smaller
# steps. Formaldehyde's residual is 1.4e-7 at 5e-4, while at 1.25e-4 its SC
# and PC noise reaches 4e-8 in the translational sum (7.8e-9 at 5e-4), so it
# takes the larger ones.
# The guard: a displaced CASSCF energy within PREDICTION_BOUND h^2 of the
# first-order prediction (hartree/bohr^2: half the diagonal force constant,
# at most 0.52 here, formaldehyde's O along C=O, so a factor near four above
# the largest measured); its active space's smallest singular value of
# overlap with the reference's above 1 - SUBSPACE_TOL, and its CI vector's
# overlap above 1 - CI_TOL. A different stationary point misses by orders of
# magnitude: a swapped orbital's overlap is ~0, another state's energy is
# millihartree away.
PREDICTION_BOUND = 2.0
SUBSPACE_TOL = 1e-3
CI_TOL = 1e-3
# The checks (hartree/bohr): Richardson against the finest central
# difference, the ladder's ratio wherever its denominator is above
# LADDER_FLOOR, the translational sum, and the CASSCF's Richardson gradient
# against PySCF's analytic one.
RICHARDSON_TOL = 2e-7
LADDER = (3.2, 4.8)
LADDER_FLOOR = 1e-7
TRANSLATION_TOL = 1e-8
ANALYTIC_TOL = 1e-8

QUANTITIES = ("casscf", "sc", "pc")


def rows(gradient: numpy.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in gradient]


def run_rhf(name: str, mol: gto.Mole) -> scf.hf.SCF:
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, f"{name}: RHF did not converge"
    return mf


def singlet(mc, mo: numpy.ndarray) -> tuple[float, numpy.ndarray]:
    """casscf_reference.spin_roots for one root of a closed shell: the
    lowest singlet over mo by a dense diagonalisation, refused unless the
    next singlet is ROOT_GAP above it. The spectrum is walked upward and
    stops at the second singlet, since <S^2> of all of it dominates a point."""
    ncas, nelec = mc.ncas, mc.nelecas
    h1, e_core = mc.get_h1eff(mo)
    eri = ao2mo.restore(1, mc.get_h2eff(mo), ncas)
    h2e = fci.direct_spin1.absorb_h1e(h1, eri, ncas, nelec, 0.5)
    na = fci.cistring.num_strings(ncas, nelec[0])
    values, vectors = numpy.linalg.eigh(dense_hamiltonian(h2e, ncas, nelec))
    found = []
    for value, vector in zip(values, vectors.T, strict=True):
        vector = vector.reshape(na, -1)
        if abs(fci.spin_op.spin_square0(vector, ncas, nelec)[0]) < SPIN_S2:
            found.append((float(value + e_core), vector))
            if len(found) == 2:
                break
    assert found, "no singlet in the sector"
    if len(found) == 2:
        assert found[1][0] - found[0][0] > ROOT_GAP, "the root is degenerate"
    return found[0]


def orbital_gradient(mc, mo: numpy.ndarray):
    """The lowest singlet's energy and CI vector over mo, and the orbital
    gradient there (casscf_reference.derivatives' g, without H)."""
    energy, ci = singlet(mc, mo)
    dm1, dm2 = fci.direct_spin1.make_rdm12(ci, mc.ncas, mc.nelecas)
    mc.mo_coeff = mo
    half_g, *_ = mc1step.gen_g_hop(mc, mo, 1, dm1, dm2, mc.ao2mo(mo))
    return energy, ci, 2 * half_g


def polished(name, mf, mo, ncas, nelecas, auxbasis):
    """casscf_reference's DF mc1step from mo, polished to |g| < POLISH_GRAD.
    nevpt2_reference's polish rebuilds the explicit orbital Hessian at every
    step; here it is built once, at mc1step's orbitals, and every step solves
    against it (a chord method: the same fixed point, |g| < POLISH_GRAD, at
    a fraction of the cost, since gen_g_hop's Hessian leaves out the CI
    response either way and neither iteration is quadratic). Its least
    eigenvalue is gated as there. Returns the CASSCF, the polished orbitals,
    the energy there and the dense singlet's CI vector."""
    mc = casscf(mf, ncas, nelecas, [1.0], auxbasis)
    optimize(mc, mc.kernel, mo)
    assert mc.converged, f"{name}: mc1step did not converge"
    orbitals = mc.mo_coeff
    _, _, _, _, hessian, _ = derivatives(mc, orbitals, [1.0])
    curvature = numpy.linalg.eigvalsh(hessian)[0]
    assert curvature > POLISH_CURVATURE, f"{name}: H's least eigenvalue {curvature}"
    energy, ci, g = orbital_gradient(mc, orbitals)
    for _ in range(POLISH_STEPS):
        if numpy.linalg.norm(g) < POLISH_GRAD:
            break
        x = -numpy.linalg.solve(hessian, g)
        orbitals = orbitals @ mc1step.expmat(mc.unpack_uniq_var(x))
        energy, ci, g = orbital_gradient(mc, orbitals)
    else:
        raise AssertionError(f"{name}: the polish did not reach |g| {POLISH_GRAD}")
    return mc, orbitals, energy, ci


class KeptModes:
    """Counts the modes PC's per-tuple solves keep (relative singular value
    above PC_TAU), over all of a run's solves. nevpt2_reference.block2 wraps
    icnevpt2_full._linear_solve as it finds it, so this wrapper runs inside
    its gap gate."""

    def __init__(self):
        self.kept = 0
        self.saved = icnevpt2_full._linear_solve

    def __enter__(self):
        def counted(a, b):
            if len(a):
                sv = numpy.linalg.svd(a, compute_uv=False)
                live = sv[:, 0] > 0
                self.kept += int((sv[live] / sv[live, :1] > PC_TAU).sum())
            return self.saved(a, b)

        icnevpt2_full._linear_solve = counted
        return self

    def __exit__(self, *exc):
        icnevpt2_full._linear_solve = self.saved


def nevpt2_at(name, mf, auxbasis, orbitals, ncas, nelecas, e_casscf, full):
    """SC and PC at the polished orbitals, over a DFCASCI, as
    nevpt2_reference.case_reference computes them for one root. `full` adds
    that tool's invariance and block2 gates. Returns the SC and PC
    correlation energies, PC's kept-mode count, the root's smallest
    canonical core and virtual gaps, and (full) the SC classes."""
    casci = mcscf.CASCI(mf.density_fit(auxbasis=auxbasis), ncas, nelecas)
    assert isinstance(casci, _DFCAS), f"{name}: the CASCI is not fitted"
    assert casci.with_df is casci._scf.with_df, f"{name}: two fitting objects"
    casci.fcisolver.conv_tol = 1e-14
    casci.fix_spin_(ss=0)
    casci.verbose = 0
    casci.kernel(orbitals)
    check_aux(name, casci, mf.mol, auxbasis)
    e_casci, ci = singlet(casci, casci.mo_coeff)
    overlap = abs(numpy.vdot(casci.ci, ci))
    assert overlap > 1 - SAME_STATE, f"{name}: Davidson overlap {overlap}"
    assert abs(e_casci - casci.e_tot) < CASCI_AGREEMENT, f"{name}: Davidson"
    assert abs(e_casci - e_casscf) < CASCI_AGREEMENT, f"{name}: CASCI"

    sc, classes = nevpt(casci, ci, 0, natorb=False)
    state = canonical(casci, ci, natorb=False)
    with KeptModes() as kept:
        pc = block2(casci, state, "pc")
    e_pc = sum(pc[key] for key in CLASSES)
    _, _, mo_energy = state
    ncore, nocc = casci.ncore, casci.ncore + casci.ncas
    gaps = {
        space: float(numpy.diff(numpy.sort(values)).min())
        for space, values in (
            ("core", mo_energy[:ncore]),
            ("virtual", mo_energy[nocc:]),
        )
        if len(values) > 1
    }
    record = {"sc": sc, "pc": e_pc, "kept": kept.kept, "gaps": gaps}
    if not full:
        return record

    natural, natural_classes = nevpt(casci, ci, 0, natorb=True)
    misses = [abs(natural - sc)]
    for key, (norm, energy) in classes.items():
        misses += [
            abs(natural_classes[key][0] - norm),
            abs(natural_classes[key][1] - energy),
        ]
    assert max(misses) < INVARIANCE, f"{name}: cas_natorb differs by {max(misses)}"
    b2_sc = block2(casci, state, "sc")
    b2_miss = max(abs(b2_sc[key] - classes[key][1]) for key in CLASSES)
    assert b2_miss < BLOCK2_SC, f"{name}: block2 SC off PySCF's by {b2_miss}"
    assert abs(pc["Sijrs"] - b2_sc["Sijrs"]) < BLOCK2_SC, f"{name}: PC Sijrs"
    pc_natural = block2(casci, canonical(casci, ci, natorb=True), "pc")
    pc_miss = max(abs(pc_natural[key] - pc[key]) for key in CLASSES)
    assert pc_miss < INVARIANCE, f"{name}: PC on natural orbitals by {pc_miss}"
    record["classes"] = classes
    record["pc_classes"] = pc
    record["checks"] = {
        "sc_invariance": float(max(misses)),
        "block2_sc": float(b2_miss),
        "pc_invariance": float(pc_miss),
    }
    return record


def moved(case: dict, coords: numpy.ndarray) -> dict:
    atoms = [(s, tuple(r)) for (s, _), r in zip(case["atoms"], coords, strict=True)]
    return dict(case, atoms=atoms)


def carried(mo: numpy.ndarray, mol: gto.Mole) -> numpy.ndarray:
    """mo Lowdin-orthonormalized in mol's overlap."""
    values, vectors = numpy.linalg.eigh(mo.T @ mol.intor("int1e_ovlp") @ mo)
    return mo @ vectors @ numpy.diag(values**-0.5) @ vectors.T


class Case:
    """One case's reference point and the guard every displacement passes."""

    def __init__(self, case: dict):
        self.case = case
        self.name = case["name"]
        self.nelecas, self.ncas = case["cas"]
        self.coords = numpy.array([r for _, r in case["atoms"]], dtype=float)
        self.mol = molecule(case)
        self.auxbasis, self.aux_provenance = aux_basis(case, self.mol)
        mf = run_rhf(self.name, self.mol)
        start = mf.mo_coeff
        if "active" in case:
            start = casscf(mf, self.ncas, self.nelecas, [1.0]).sort_mo(
                case["active"], base=0
            )
        mc, self.mo, self.energy, self.ci = polished(
            self.name, mf, start, self.ncas, self.nelecas, self.auxbasis
        )
        self.ncore = mc.ncore
        check_aux(self.name, mc, self.mol, self.auxbasis)
        self.nevpt2 = nevpt2_at(
            self.name,
            mf,
            self.auxbasis,
            self.mo,
            self.ncas,
            self.nelecas,
            self.energy,
            full=True,
        )
        self.gradient = self.analytic()
        self.worst = {
            "prediction": 0.0,
            "subspace": 0.0,
            "ci": 0.0,
            "inactive_gap": min(self.nevpt2["gaps"].values()),
        }

    def analytic(self) -> numpy.ndarray:
        """PySCF's DF-CASSCF gradient at the reference, over a DF _scf
        sharing the CASSCF's fit, as casscf_gradient_reference builds it."""
        mf = fitted_scf(self.mol, self.auxbasis, self.mo)
        mc = casscf(mf, self.ncas, self.nelecas, [1.0], self.auxbasis)
        assert mc.with_df is mc._scf.with_df, f"{self.name}: two fits"
        mc.mo_coeff, mc.ci, mc.e_tot = self.mo, self.ci, self.energy
        grad = mc.nuc_grad_method()
        assert isinstance(grad, dfcasscf_grad.Gradients), type(grad)
        assert grad.auxbasis_response
        grad.verbose = 0
        return grad.kernel(mo_coeff=self.mo, ci=self.ci)

    def point(self, displacement: numpy.ndarray) -> dict:
        """The three energies at R0 + displacement, guarded."""
        name = self.name
        case = moved(self.case, self.coords + displacement)
        mol = molecule(case)
        auxbasis, _ = aux_basis(case, mol)
        mf = run_rhf(name, mol)
        _, mo, energy, ci = polished(
            name, mf, carried(self.mo, mol), self.ncas, self.nelecas, auxbasis
        )

        step2 = float(numpy.sum(displacement**2))
        predicted = self.energy + float(numpy.sum(self.gradient * displacement))
        prediction = abs(energy - predicted) / step2
        assert prediction < PREDICTION_BOUND, (
            f"{name}: {prediction:.3f} h^2 off the first-order prediction"
        )
        act = slice(self.ncore, self.ncore + self.ncas)
        cross = gto.intor_cross("int1e_ovlp", self.mol, mol)
        s = self.mo[:, act].T @ cross @ mo[:, act]
        subspace = 1 - numpy.linalg.svd(s, compute_uv=False).min()
        assert subspace < SUBSPACE_TOL, f"{name}: active subspace by {subspace}"
        nelec = (self.nelecas // 2, self.nelecas // 2)
        ci_miss = 1 - abs(float(fci.addons.overlap(self.ci, ci, self.ncas, nelec, s)))
        assert ci_miss < CI_TOL, f"{name}: CI overlap off 1 by {ci_miss}"
        for key, value in (
            ("prediction", prediction),
            ("subspace", subspace),
            ("ci", ci_miss),
        ):
            self.worst[key] = max(self.worst[key], float(value))

        nevpt2 = nevpt2_at(
            name, mf, auxbasis, mo, self.ncas, self.nelecas, energy, full=False
        )
        gap = min(nevpt2["gaps"].values())
        self.worst["inactive_gap"] = min(self.worst["inactive_gap"], gap)
        assert nevpt2["kept"] == self.nevpt2["kept"], (
            f"{name}: PC keeps {nevpt2['kept']} modes, the reference "
            f"{self.nevpt2['kept']}: a rank change"
        )
        return {
            "casscf": energy,
            "sc": energy + nevpt2["sc"],
            "pc": energy + nevpt2["pc"],
        }


def ladder_ratios(central: list[numpy.ndarray]) -> list[float]:
    """(D(4h) - D(2h)) / (D(2h) - D(h)) where the denominator is above
    LADDER_FLOOR."""
    coarse, middle, fine = central
    upper, lower = (coarse - middle).ravel(), (middle - fine).ravel()
    return [
        float(u / v) for u, v in zip(upper, lower, strict=True) if abs(v) > LADDER_FLOOR
    ]


def case_reference(case: dict) -> dict:
    started = time.perf_counter()
    ref = Case(case)
    name = ref.name
    natom = len(case["atoms"])
    steps = case["steps"]
    e0 = {"casscf": ref.energy}
    e0["sc"] = ref.energy + ref.nevpt2["sc"]
    e0["pc"] = ref.energy + ref.nevpt2["pc"]

    displaced = []
    central = {q: [numpy.zeros((natom, 3)) for _ in steps] for q in QUANTITIES}
    for atom in range(natom):
        for axis in range(3):
            entry = {"atom": atom, "axis": axis}
            for q in QUANTITIES:
                entry[q] = []
            for i, h in enumerate(steps):
                pair = {}
                for sign in (-1, 1):
                    displacement = numpy.zeros((natom, 3))
                    displacement[atom, axis] = sign * h
                    pair[sign] = ref.point(displacement)
                for q in QUANTITIES:
                    entry[q].append([pair[-1][q], pair[1][q]])
                    central[q][i][atom, axis] = (pair[1][q] - pair[-1][q]) / (2 * h)
            displaced.append(entry)
            print(f"{name}: atom {atom} axis {axis} done", flush=True)

    gradients, checks = {}, {}
    for q in QUANTITIES:
        coarse, fine = central[q][-2], central[q][-1]
        richardson = (4 * fine - coarse) / 3
        miss = float(numpy.abs(richardson - fine).max())
        ratios = ladder_ratios(central[q])
        translation = float(numpy.abs(richardson.sum(axis=0)).max())
        print(
            f"{name} {q}: Richardson - finest {miss:.1e}, translation "
            f"{translation:.1e}, ladder ratios {min(ratios, default=0):.3f}.."
            f"{max(ratios, default=0):.3f} over {len(ratios)}",
            flush=True,
        )
        assert miss < RICHARDSON_TOL, f"{name} {q}: Richardson off the finest by {miss}"
        for ratio in ratios:
            assert LADDER[0] < ratio < LADDER[1], f"{name} {q}: ladder ratio {ratio}"
        assert translation < TRANSLATION_TOL, f"{name} {q}: translation {translation}"
        # nevpt2-gpu-demo's error bar: the Richardson step over the coarser
        # pair against the one over the finer.
        coarser = (4 * coarse - central[q][-3]) / 3
        gradients[q] = {
            "central": [rows(d) for d in central[q]],
            "richardson": rows(richardson),
            "error_bar": rows(numpy.abs(richardson - coarser)),
        }
        checks[q] = {
            "richardson_minus_finest": miss,
            "ladder_ratios": [min(ratios), max(ratios)] if ratios else None,
            "ladder_components": len(ratios),
            "translation": translation,
        }
    analytic = float(
        numpy.abs(numpy.array(gradients["casscf"]["richardson"]) - ref.gradient).max()
    )
    print(f"{name}: CASSCF Richardson - analytic {analytic:.1e}", flush=True)
    assert analytic < ANALYTIC_TOL, (
        f"{name}: CASSCF Richardson off analytic by {analytic}"
    )
    checks["casscf"]["analytic"] = analytic

    print(f"{name}: {time.perf_counter() - started:.0f} s", flush=True)
    return {
        "name": name,
        "basis": case["basis"],
        "aux_basis": ref.aux_provenance,
        "charge": 0,
        "atoms": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in case["atoms"]
        ],
        "electrons": int(ref.mol.nelectron),
        "nao": int(ref.mol.nao),
        "ncore": int(ref.ncore),
        "ncas": ref.ncas,
        "nelecas": ref.nelecas,
        "active": case.get("active"),
        "weights": [1.0],
        "energies": e0,
        "e_corr": {"sc": ref.nevpt2["sc"], "pc": ref.nevpt2["pc"]},
        "classes": [
            {
                "name": key,
                "norm": ref.nevpt2["classes"][key][0],
                "energy": ref.nevpt2["classes"][key][1],
                "pc_energy": ref.nevpt2["pc_classes"][key],
            }
            for key in CLASSES
        ],
        "inactive_gaps": ref.nevpt2["gaps"],
        "pc_kept_modes": ref.nevpt2["kept"],
        "casscf_gradient": rows(ref.gradient),
        "steps": list(steps),
        "gradients": gradients,
        "displaced": displaced,
        "guard": dict(ref.worst),
        "checks": {
            **checks,
            "reference_point": ref.nevpt2["checks"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=OUT,
        help="where to write (default: the tree's)",
    )
    parser.add_argument("--only", nargs="+", help="these cases alone (development)")
    args = parser.parse_args()
    lib.num_threads(1)
    assert lib.num_threads() == 1
    cases = [c for c in CASES if not args.only or c["name"] in args.only]
    doc = {
        "description": "Numerical nuclear gradients of the single-state "
        "density-fitted CASSCF, SC-NEVPT2 (PySCF mrpt.NEVPT, DF throughout) "
        "and PC-NEVPT2 (block2 WickICNEVPT2 over PySCF's fitted integrals) "
        "total energies, each re-converged at every displaced geometry over "
        "this tree's data/basis and data/aux_basis files: central "
        "differences at each of 'steps' and one Richardson step over the "
        "two finest, every displacement through the continuity guard. "
        "Hartree, bohr; G[A][k] = dE/dR_Ak",
        "provenance": {
            "pyscf": pyscf.__version__,
            "block2": version("block2"),
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "guard": {
            "prediction_bound": PREDICTION_BOUND,
            "subspace_tol": SUBSPACE_TOL,
            "ci_tol": CI_TOL,
            "degenerate": DEGENERATE,
            "pc_tau": PC_TAU,
            "pc_min_gap": PC_MIN_GAP,
        },
        "tolerances": {
            "richardson": RICHARDSON_TOL,
            "ladder": list(LADDER),
            "ladder_floor": LADDER_FLOOR,
            "translation": TRANSLATION_TOL,
            "analytic": ANALYTIC_TOL,
        },
        "cases": [case_reference(case) for case in cases],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
