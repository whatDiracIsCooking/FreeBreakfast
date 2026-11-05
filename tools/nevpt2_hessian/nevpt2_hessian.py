#!/usr/bin/env python3
"""Write the SC- and PC-NEVPT2 nuclear Hessian references the analytic
Hessian (milestone 24) is gated on: test/gtest/data/nevpt2_hessian/.

No program has an SC- or PC-NEVPT2 Hessian, so the oracle is a stencil over
the reference gradient: `.temp/nevpt2/nuclear_hessian.tex`, "The
semi-numerical Hessian". Nor is there an analytic NEVPT2 gradient outside
this tree's C++ (nevpt2-gpu-demo's nevpt2_grad.py takes its source terms
from the GPU's dumps or from difference quotients), and the oracle must not
be the code it gates, so the reference gradient is the one
tools/emit/nevpt2_gradient_reference.py already is: a central difference of
re-converged PySCF (and, for PC, the C++ step's cut) energies, here at one
fourth-order step. Nothing of this tree's C++ runs.

**The energy at a geometry** (`Reference.point`), single-state closed-shell
DF-CASSCF over this tree's data/basis file and data/aux_basis's
cc-pVDZ-JKFIT (parsed as the emit tools parse them, the same fitting set the
tree's DF uses):

- **CASSCF.** casscf_reference.py's DF mc1step, warm-started from the
  reference geometry's polished orbitals Lowdin-orthonormalized in the
  displaced overlap (nevpt2_gradient_reference's `carried`), then polished
  by Newton steps over gen_g_hop's orbital Hessian, the CI re-solved densely
  at each, to |g| < POLISH_GRAD.
- **SC-NEVPT2.** nevpt2_reference.py's: PySCF's `mrpt.NEVPT` over a
  DFCASCI on those orbitals, core and virtuals canonicalized over the root's
  Fock, the active orbitals unrotated, DF branch asserted.
- **PC-NEVPT2.** The C++ step's rule (src/nevpt2/energy/README.md,
  "PC-NEVPT2"), as qd_nevpt2_reference.py's one-root PC computes it: each
  class's metric S cut at nevpt2_reference.pc_cut(d) of its largest (no gap
  test, #853), K' = X^T K X over the kept modes, refused when S has no
  positive eigenvalue or a mode's amplitude |y / (lambda + Delta)| exceeds
  1 (#874); Sijrs is SC's. At the reference geometry it is checked against
  block2's `WickICNEVPT2` (its own lstsq cut) and recorded beside it.

**The guard.** Every displaced point must (1) converge and polish, (2) sit
within PREDICTION_BOUND |dR|^2 of E0 + g . dR, g PySCF's analytic DF-CASSCF
gradient at the reference, (3) keep its active space, 1 - the smallest
singular value of the AO cross overlap with the reference's below
SUBSPACE_BOUND |dR|^2 (a rotation by an angle ~ |dR| misses by its square;
a swapped orbital misses by ~1), (4) keep its CI vector, 1 - |overlap|
through that orbital overlap below CI_BOUND |dR|^2, (5) drop exactly the
reference's number of PC modes in every class, so no step crosses a rank
change of the cut, and (6) keep its canonical core and virtual energies
DEGENERATE apart. A miss refuses emission.

**The gradient** at a geometry R is the fourth-order central stencil over
the energies at R +- delta e_i, R +- 2 delta e_i, for every Cartesian
component i (`stencil.derivative`): the reference gradient's own
construction at one step, with the reference's orbitals as every point's
start. Its truncation, delta^4 E^(5) / 30, is a smooth function of R, so it
moves the Hessian by its derivative (delta^4 E^(6) / 30, below 1e-10 at the
cases' delta) and keeps it symmetric to that.

**The coordinates.** Cartesian, or for the symmetric case its
symmetry-adapted ones (an orthonormal Q, `coordinates`), so a failing block
is named by its irreps rather than smeared over a Cartesian frame
(nuclear_hessian.tex, "Which coordinates"). Element (i, j) is
q_i^T dg/dq_j; a Cartesian Hessian is Q H Q^T.

**The Hessian.** Column j is the fourth-order central stencil over those
gradients at R0 +- s q_j, R0 +- 2 s q_j, for s = h, h/2 and h/4 (eight
points, `stencil.displacements`). Per element, the ratio of successive
differences (D(h) - D(h/2)) / (D(h/2) - D(h/4)) must lie in RATIO_WINDOW
about 16 wherever the finer difference is above the noise floor, and below
it the coarser difference must be one a settled ladder could have
(`stencil.judge`); one that does not fails the generator (STUCK, with the
numbers), never a looser bar. The bar is the finer difference, or the floor
where it is below it. The recorded Hessian is the finest step's,
symmetrized; before that it must be symmetric to its bar.

**The noise floor.** sigma_E per quantity is measured, not assumed: at
NOISE_POINTS seeded geometries within +-h of R0, the energy from the
reference's start and from NOISE_STARTS starts rotated off it by
NOISE_KICK, the RMS of the per-point standard deviations (the reference
point alone underestimated it threefold: its four restarts spread 4e-14,
displaced ones 1e-13). A stencil at s over one at delta multiplies
independent noise by stencil.noise_gain(delta) noise_gain(s), so the finer
difference carries at most sigma_E gain(delta) (gain(h/2) + gain(h/4)); the
floor is FLOOR_FACTOR = 20 times that, so a one-sigma noise moves a judged
ratio by 5% and the 20% window holds four.

**Bilinearity**, Eq. (bilinear), on every coordinate pair (i, j): the
second derivative along the unit q = (q_i + q_j) / sqrt 2, by the same
ladder over the directional derivative q . g (four energies on the line at
each point, the same reach as a column's), against
(H_ii + H_jj + H_ij + H_ji) / 2 from the columns. A residual above qq's bar
plus half the four elements' is a failing pair, and that directional ladder
is judged as an element is.

**What fails where.** Every quantity of the skew and near-degenerate cases
is held to all of the above on every element and pair. The symmetric case
names the quantities expected to fail (`may_fail`, SC: its empty
perturbers, #894): those are held on the totally symmetric block only, and
every miss elsewhere is recorded as its block's failure (`failed_blocks`,
`failing_pairs`), not reported as a Hessian; the case is refused unless one
such block fails. Its other quantities are held everywhere.

BLAS and OpenMP run on one thread in every worker and every point starts
from the reference's orbitals, so a point's energy does not depend on the
order the pool runs them in, and nothing run-dependent (date, host,
timings) is written: a rerun on the same PySCF, block2 and numpy is
byte-identical.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/nevpt2_hessian/nevpt2_hessian.py [--jobs N]

(N worker processes of one thread each, at most and by default JOBS, 8)
writes test/gtest/data/nevpt2_hessian/<case>.json. Never hand-edited.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from importlib.metadata import version  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "emit"))
sys.path.insert(0, str(HERE))

# isort: split
# The emit tools pin their own thread counts on import; numpy is loaded by
# then, so this module's single thread stands (and main() pins PySCF's).
import nevpt2_gradient_reference as gradient_reference  # noqa: E402
import nevpt2_reference  # noqa: E402
import qd_nevpt2_reference as qd  # noqa: E402
import stencil  # noqa: E402
from casscf_gradient_reference import fitted_scf  # noqa: E402
from casscf_reference import (  # noqa: E402
    POLISH_CURVATURE,
    casscf,
    check_aux,
    derivatives,
    optimize,
)
from nevpt2_reference import CLASSES, DEGENERATE, pc_cut  # noqa: E402
from pyscf import df, fci, gto, lib, mcscf, scf  # noqa: E402
from pyscf.df.grad import casscf as dfcasscf_grad  # noqa: E402
from pyscf.mcscf import mc1step  # noqa: E402
from pyscf.mrpt import dfnevpt2, nevpt2  # noqa: E402
from scf_reference import pyscf_basis  # noqa: E402

OUT = ROOT / "test" / "gtest" / "data" / "nevpt2_hessian"
COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/nevpt2_hessian/nevpt2_hessian.py"
)
AUX = ("cc-pVDZ-JKFIT", "../aux_basis/cc-pVDZ-JKFIT.json")
BASES = {"sto-3g": "sto-3g.json"}

# The mirror plane of the Cs ammonia below is xz (y -> -y),
# H1 in it and H3 H2's image. Its symmetry-adapted coordinates: A' (the
# totally symmetric, listed first) and A''.
NH3_CS_COORDINATES = [
    ("N x", "A'", {0: 1.0}),
    ("N z", "A'", {2: 1.0}),
    ("H1 x", "A'", {3: 1.0}),
    ("H1 z", "A'", {5: 1.0}),
    ("H2x + H3x", "A'", {6: 1.0, 9: 1.0}),
    ("H2y - H3y", "A'", {7: 1.0, 10: -1.0}),
    ("H2z + H3z", "A'", {8: 1.0, 11: 1.0}),
    ("N y", "A''", {1: 1.0}),
    ("H1 y", "A''", {4: 1.0}),
    ("H2y + H3y", "A''", {7: 1.0, 10: 1.0}),
    ("H2x - H3x", "A''", {6: 1.0, 9: -1.0}),
    ("H2z - H3z", "A''", {8: 1.0, 11: -1.0}),
]

CASES = [
    {
        "name": "h2co_skew_sto3g",
        "kind": "skew",
        "atoms": [
            ("C", (-0.2072, -0.1316, 0.1506)),
            ("O", (0.1611, -0.2829, 2.2566)),
            ("H", (0.4395, 1.5699, -1.0127)),
            ("H", (-0.5032, -1.8644, -1.0116)),
        ],
        "basis": "sto-3g",
        "cas": (4, 4),
        "active": [6, 7, 8, 9],
        "quantities": ("casscf", "sc", "pc"),
        "irreps": ("A",),
        "h": 0.03,
        "delta": 5e-3,
    },
    {
        "name": "nh3_cs_sto3g",
        "kind": "symmetric",
        "atoms": [
            ("N", (0.0, 0.0, 0.0)),
            ("H", (1.804, 0.0, -0.984)),
            ("H", (-0.54, 1.782, -0.416)),
            ("H", (-0.54, -1.782, -0.416)),
        ],
        "basis": "sto-3g",
        "cas": (2, 2),
        # HOMO-1 and LUMO, both A': the symmetric minimum, the lower of the
        # two CAS(2,2) minima here (-55.437459 against a mirror-broken
        # -55.432707). From the frontier pair [4, 5] PySCF reaches it, but
        # this tree's solver leaves the mirror along the start's A''
        # negative curvature and lands in the broken one (#927); from
        # [3, 5] both reach the symmetric minimum.
        "active": [3, 5],
        "quantities": ("casscf", "sc", "pc"),
        "irreps": ("A'", "A''"),
        "coordinates": NH3_CS_COORDINATES,
        "may_fail": ("sc",),
        "h": 0.03,
        "delta": 5e-3,
    },
    {
        "name": "h2co_neardeg_sto3g",
        "kind": "near_degenerate",
        "atoms": [
            ("C", (-0.4776, 0.4103, -0.0952)),
            ("O", (0.4002, -0.2509, 1.8994)),
            ("H", (0.2584, 1.6398, -1.5664)),
            ("H", (-0.5763, -1.3959, -1.1052)),
        ],
        "basis": "sto-3g",
        "cas": (4, 4),
        "active": [6, 7, 8, 9],
        "quantities": ("casscf", "pc"),
        "irreps": ("A",),
        "h": 0.04,
        "delta": 5e-3,
    },
]

# The polish's target, |g| of the orbital gradient. SC and PC are not
# stationary in the orbitals, so an orbital error of |g| / curvature moves
# them linearly: at 1e-12 that is below the energies' round-off.
POLISH_GRAD = 1e-12
POLISH_STEPS = 200
# The guard (see the docstring), each per bohr^2 of displacement: the energy
# off its first-order prediction (hartree; half a diagonal force constant,
# worst measured 1.72, the skew formaldehyde), and the active space's and CI
# vector's overlap misses (worst 1.68 and 2.12, both the near-degenerate
# case). A different stationary point misses by orders of magnitude: a
# swapped orbital's overlap is ~0, another state's energy is millihartree off.
PREDICTION_BOUND = 2.0
SUBSPACE_BOUND = 5.0
CI_BOUND = 5.0
# The noise measurement: the geometries, the restarts at each, and the size
# of the random rotation each restart starts from (radians, every
# non-redundant pair).
NOISE_POINTS = 8
NOISE_STARTS = 4
NOISE_KICK = 1e-3
NOISE_SEED = 803
# The ladder's verdicts: the window about 16 a judged ratio must lie in (the
# reference gradient's (3.2, 4.8) about 4, the same 20%), and the noise
# floor's multiple of the finer difference's noise.
RATIO_WINDOW = (0.8 * stencil.TARGET, 1.2 * stencil.TARGET)
FLOOR_FACTOR = 20.0
# The CASSCF's stencil gradient at the reference against PySCF's analytic
# DF-CASSCF gradient (hartree/bohr): the differencing machinery's own check.
ANALYTIC_TOL = 1e-8
# block2's PC against this tool's at the reference, per class, where the
# metric's cut drops nothing block2's lstsq keeps (hartree).
BLOCK2_PC = 1e-8

# Worker processes, one BLAS/OpenMP thread each: the box's per-agent budget
# is eight threads, so eight is the default and the cap.
JOBS = 8


def rows(m) -> list[list[float]]:
    return [[float(x) for x in row] for row in m]


def provenance(file: str) -> dict:
    path = (ROOT / "data" / "basis" / file).resolve()
    return {
        "file": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def molecule(case: dict, coords: numpy.ndarray) -> tuple[gto.Mole, dict]:
    """The case's molecule at coords (bohr) and its fitting set."""
    symbols = {symbol for symbol, _ in case["atoms"]}
    mol = gto.M(
        atom=[
            [symbol, tuple(float(x) for x in r)]
            for (symbol, _), r in zip(case["atoms"], coords, strict=True)
        ],
        unit="Bohr",
        basis=pyscf_basis(BASES[case["basis"]], symbols),
        charge=0,
        spin=0,
        cart=False,
        verbose=0,
    )
    return mol, pyscf_basis(AUX[1], symbols)


def polished(name: str, mf, mo, ncas: int, nelecas: int, auxbasis: dict):
    """The DF mc1step from mo, polished to |g| < POLISH_GRAD against one
    orbital Hessian built at mc1step's orbitals (nevpt2_gradient_reference's
    chord iteration, run tighter). Returns the CASSCF, the orbitals, the
    energy, the dense singlet's CI vector and the final |g|."""
    mc = casscf(mf, ncas, nelecas, [1.0], auxbasis)
    optimize(mc, mc.kernel, mo)
    assert mc.converged, f"{name}: mc1step did not converge"
    orbitals = mc.mo_coeff
    _, _, _, _, hessian, _ = derivatives(mc, orbitals, [1.0])
    curvature = numpy.linalg.eigvalsh(hessian)[0]
    assert curvature > POLISH_CURVATURE, f"{name}: H's least eigenvalue {curvature}"
    energy, ci, g = gradient_reference.orbital_gradient(mc, orbitals)
    for _ in range(POLISH_STEPS):
        if numpy.linalg.norm(g) < POLISH_GRAD:
            break
        x = -numpy.linalg.solve(hessian, g)
        orbitals = orbitals @ mc1step.expmat(mc.unpack_uniq_var(x))
        energy, ci, g = gradient_reference.orbital_gradient(mc, orbitals)
    else:
        raise AssertionError(f"{name}: the polish did not reach |g| {POLISH_GRAD}")
    return mc, orbitals, energy, ci, float(numpy.linalg.norm(g))


def dfcasci(name: str, mf, auxbasis: dict, orbitals, ci, energy, ncas, nelecas):
    """A DFCASCI over the polished orbitals, the dense singlet as its CI
    vector (as nevpt2_gradient_reference's, without re-running Davidson:
    the vector is the CASSCF's own)."""
    casci = mcscf.CASCI(mf.density_fit(auxbasis=auxbasis), ncas, nelecas)
    assert casci.with_df is casci._scf.with_df, f"{name}: two fitting objects"
    casci.fcisolver.conv_tol = 1e-14
    casci.verbose = 0
    casci.mo_coeff = orbitals
    casci.ci = ci
    casci.e_tot = energy
    return casci


def pc_classes(name: str, casci, ci) -> tuple[dict, dict, dict, dict]:
    """One root's PC-NEVPT2 by the C++ step's rule (see the docstring) and
    its SC class norms, on the root's canonical state: per class the PC
    energy (Sijrs left to SC) and the modes the cut drops, the SC
    per-perturber norms, and the state's canonical core and virtual gaps."""
    mo, root_ci, mo_energy = nevpt2_reference.canonical(casci, ci, natorb=False)
    state = copy.copy(casci)
    state.mo_coeff, state.mo_energy = mo, mo_energy
    eris = dfnevpt2._ERIS(state, mo, casci.with_df)
    blocks = qd.integral_blocks(name, state, mo, eris)
    ncas, nelec = casci.ncas, casci.nelecas
    root_ci = numpy.ascontiguousarray(root_ci)
    dm1t, dm2, dm3 = fci.rdm.make_dm123(
        "FCI3pdm_kern_sf", root_ci, root_ci, ncas, nelec
    )
    link = tuple(fci.cistring.gen_linkstr_index(range(ncas), n) for n in nelec)
    aaaa = numpy.ascontiguousarray(blocks["h2e"].transpose(0, 2, 1, 3))
    rdms = {
        "dm1": dm1t.T,
        "dm2": dm2,
        "dm3": dm3,
        "f3ca": nevpt2._contract4pdm(
            "NEVPTkern_cedf_aedf", aaaa, root_ci, ncas, nelec, link
        ),
        "f3ac": nevpt2._contract4pdm(
            "NEVPTkern_aedf_ecdf", aaaa, root_ci, ncas, nelec, link
        ),
    }
    metrics = qd.pc_metrics(blocks, rdms, 1.0)
    dyall = qd.pc_dyall(blocks, rdms)
    energies, dropped = {}, {}
    for key, (x, h1, gap, pairing) in qd.pc_tuples(blocks).items():
        s, s1 = metrics[key]
        solve = qd.pc_solve(s, dyall[key])
        assert solve is not None, f"{name} {key}: no positive eigenvalue (refused)"
        y = qd.pc_y(x, h1, solve["s"], s1, solve["u"])
        reason = qd.pc_intruder(y, solve["lambda"], gap, s.shape[0])
        assert reason is None, f"{name} {key}: refused, {reason}"
        energies[key] = qd.pc_pair(y, y, solve["lambda"], gap, pairing)
        dropped[key] = int(solve["dropped"])
    norms = qd.class_norms(blocks, rdms, 1.0)
    ncore, nocc = casci.ncore, casci.ncore + ncas
    gaps = {
        space: float(numpy.diff(numpy.sort(values)).min())
        for space, values in (
            ("core", mo_energy[:ncore]),
            ("virtual", mo_energy[nocc:]),
        )
        if len(values) > 1
    }
    return energies, dropped, norms, gaps


def nevpt2_at(name: str, casci, ci) -> dict:
    """SC and PC at one geometry: the correlation energies, SC's and PC's
    classes, PC's dropped modes, SC's per-perturber norms, the gaps."""
    sc, sc_classes = nevpt2_reference.nevpt(casci, ci, 0, natorb=False)
    pc, dropped, norms, gaps = pc_classes(name, casci, ci)
    pc["Sijrs"] = sc_classes["Sijrs"][1]
    return {
        "sc": sc,
        "pc": float(sum(pc[key] for key in CLASSES)),
        "sc_classes": sc_classes,
        "pc_classes": pc,
        "dropped": dropped,
        "norms": norms,
        "gaps": gaps,
    }


def kicked(mc, mo: numpy.ndarray, rng: numpy.random.Generator) -> numpy.ndarray:
    """mo rotated by a random NOISE_KICK in every non-redundant pair."""
    x = rng.uniform(
        -NOISE_KICK, NOISE_KICK, mc.pack_uniq_var(numpy.zeros_like(mo)).size
    )
    return mo @ mc1step.expmat(mc.unpack_uniq_var(x))


class Reference:
    """One case's reference geometry, and the guarded energy at any other."""

    def __init__(self, case: dict):
        self.case = case
        self.name = name = case["name"]
        self.nelecas, self.ncas = case["cas"]
        self.coords = numpy.array([r for _, r in case["atoms"]], dtype=float)
        self.mol, self.auxbasis = molecule(case, self.coords)
        mf = scf.RHF(self.mol)
        mf.conv_tol = 1e-14
        mf.conv_tol_grad = 1e-11
        mf.max_cycle = 200
        mf.kernel()
        assert mf.converged, f"{name}: RHF did not converge"
        start = casscf(mf, self.ncas, self.nelecas, [1.0]).sort_mo(
            case["active"], base=0
        )
        mc, self.mo, self.energy, self.ci, self.polish = polished(
            name, mf, start, self.ncas, self.nelecas, self.auxbasis
        )
        check_aux(name, mc, self.mol, self.auxbasis)
        self.ncore = mc.ncore
        self.mc = mc
        casci = dfcasci(
            name,
            mf,
            self.auxbasis,
            self.mo,
            self.ci,
            self.energy,
            self.ncas,
            self.nelecas,
        )
        self.nevpt2 = nevpt2_at(name, casci, self.ci)
        check_aux(name, casci, self.mol, self.auxbasis)
        self.checks = self.reference_checks(casci)
        self.analytic = self.casscf_gradient()

    def reference_checks(self, casci) -> dict:
        """nevpt2_gradient_reference's checks at the reference point: SC's
        active-rotation invariance, block2's SC per class against PySCF's,
        and block2's PC (its own lstsq) beside this tool's."""
        name, ci = self.name, self.ci
        sc = self.nevpt2
        natural, natural_classes = nevpt2_reference.nevpt(casci, ci, 0, natorb=True)
        misses = [abs(natural - sc["sc"])]
        for key, (norm, energy) in sc["sc_classes"].items():
            misses += [
                abs(natural_classes[key][0] - norm),
                abs(natural_classes[key][1] - energy),
            ]
        invariance = max(misses)
        assert invariance < nevpt2_reference.INVARIANCE, f"{name}: SC invariance"
        state = nevpt2_reference.canonical(casci, ci, natorb=False)
        b2_sc = nevpt2_reference.block2(casci, state, "sc")
        b2_miss = max(abs(b2_sc[key] - sc["sc_classes"][key][1]) for key in CLASSES)
        assert b2_miss < nevpt2_reference.BLOCK2_SC, f"{name}: block2 SC {b2_miss}"
        b2_pc = nevpt2_reference.block2(casci, state, "pc")
        pc_miss = {key: abs(b2_pc[key] - sc["pc_classes"][key]) for key in CLASSES}
        assert max(pc_miss.values()) < BLOCK2_PC, f"{name}: block2 PC {pc_miss}"
        casci.ci = ci
        return {
            "sc_invariance": float(invariance),
            "block2_sc": float(b2_miss),
            "block2_pc": float(max(pc_miss.values())),
        }

    def casscf_gradient(self) -> numpy.ndarray:
        """PySCF's analytic DF-CASSCF gradient at the reference, over a DF
        _scf sharing the CASSCF's fit (casscf_gradient_reference's)."""
        mf = fitted_scf(self.mol, self.auxbasis, self.mo)
        mc = casscf(mf, self.ncas, self.nelecas, [1.0], self.auxbasis)
        assert mc.with_df is mc._scf.with_df, f"{self.name}: two fits"
        mc.mo_coeff, mc.ci, mc.e_tot = self.mo, self.ci, self.energy
        grad = mc.nuc_grad_method()
        assert isinstance(grad, dfcasscf_grad.Gradients), type(grad)
        assert grad.auxbasis_response
        grad.verbose = 0
        return grad.kernel(mo_coeff=self.mo, ci=self.ci)

    def point(self, displacement: numpy.ndarray, start=None) -> dict:
        """The three energies at R0 + displacement, guarded; the guard's
        numbers beside them."""
        name = self.name
        mol, auxbasis = molecule(self.case, self.coords + displacement)
        mf = scf.RHF(mol)
        mo = gradient_reference.carried(self.mo if start is None else start, mol)
        _, mo, energy, ci, _ = polished(name, mf, mo, self.ncas, self.nelecas, auxbasis)

        step2 = float(numpy.sum(displacement**2))
        predicted = self.energy + float(numpy.sum(self.analytic * displacement))
        prediction = abs(energy - predicted) / step2 if step2 > 0 else 0.0
        assert prediction < PREDICTION_BOUND, (
            f"{name}: {prediction:.3f} |dR|^2 off the first-order prediction"
        )
        act = slice(self.ncore, self.ncore + self.ncas)
        cross = gto.intor_cross("int1e_ovlp", self.mol, mol)
        s = self.mo[:, act].T @ cross @ mo[:, act]
        subspace = float(1 - numpy.linalg.svd(s, compute_uv=False).min())
        subspace /= step2 if step2 > 0 else 1.0
        assert subspace < SUBSPACE_BOUND, f"{name}: active subspace by {subspace}"
        nelec = (self.nelecas // 2, self.nelecas // 2)
        overlap = float(fci.addons.overlap(self.ci, ci, self.ncas, nelec, s))
        ci_miss = (1 - abs(overlap)) / (step2 if step2 > 0 else 1.0)
        assert ci_miss < CI_BOUND, f"{name}: CI overlap off 1 by {ci_miss}"

        casci = dfcasci(name, mf, auxbasis, mo, ci, energy, self.ncas, self.nelecas)
        at = nevpt2_at(name, casci, ci)
        assert at["dropped"] == self.nevpt2["dropped"], (
            f"{name}: PC drops {at['dropped']}, the reference "
            f"{self.nevpt2['dropped']}: a rank change"
        )
        gap = min(at["gaps"].values())
        assert gap > DEGENERATE, f"{name}: inactive gap {gap}"
        return {
            "casscf": energy,
            "sc": energy + at["sc"],
            "pc": energy + at["pc"],
            "guard": {
                "prediction": prediction,
                "subspace": subspace,
                "ci": ci_miss,
                "inactive_gap": gap,
            },
        }

    def noise(self) -> dict:
        """sigma_E per quantity: at each of NOISE_POINTS displaced geometries
        (uniform within +-h of R0 in every coordinate, seeded), the energy
        from the reference's start and from NOISE_STARTS kicked ones; the
        sample standard deviation at each, and their root mean square."""
        rng = numpy.random.default_rng(NOISE_SEED)
        h = self.case["h"]
        variance = dict.fromkeys(self.case["quantities"], 0.0)
        for _ in range(NOISE_POINTS):
            displacement = rng.uniform(-h, h, self.coords.shape)
            starts = [None] + [
                kicked(self.mc, self.mo, rng) for _ in range(NOISE_STARTS)
            ]
            got = [self.point(displacement, start=start) for start in starts]
            for q in self.case["quantities"]:
                variance[q] += numpy.var([g[q] for g in got], ddof=1) / NOISE_POINTS
        return {q: float(numpy.sqrt(v)) for q, v in variance.items()}


# ------------------------------------------------------------------ the pool

REFERENCE: Reference | None = None


def init_worker() -> None:
    lib.num_threads(1)


def run_point(task: tuple) -> tuple:
    key, displacement = task
    assert REFERENCE is not None
    return key, REFERENCE.point(numpy.asarray(displacement))


def unit(natom: int, *coords: int) -> numpy.ndarray:
    """The sum of the unit Cartesian displacements at the given indices."""
    out = numpy.zeros(3 * natom)
    for k in coords:
        out[k] += 1.0
    return out.reshape(natom, 3)


AXES = "xyz"


def coordinates(case: dict) -> tuple[numpy.ndarray, list[str], list[str]]:
    """The case's displacement coordinates: an orthonormal 3N x 3N basis Q
    (column j is coordinate j as a Cartesian vector), their labels, and
    their irreducible representations, the totally symmetric one first in
    `case["irreps"]`. Cartesian (Q = 1, every coordinate in one irrep "A")
    unless the case lists symmetry-adapted ones: (label, irrep, {Cartesian
    index: weight}), normalized here."""
    natom = len(case["atoms"])
    n = 3 * natom
    if "coordinates" not in case:
        labels = [
            f"{symbol}{a}{AXES[k]}"
            for a, (symbol, _) in enumerate(case["atoms"])
            for k in range(3)
        ]
        return numpy.eye(n), labels, ["A"] * n
    q = numpy.zeros((n, n))
    labels, irreps = [], []
    for j, (label, irrep, weights) in enumerate(case["coordinates"]):
        for k, w in weights.items():
            q[k, j] = w
        q[:, j] /= numpy.linalg.norm(q[:, j])
        labels.append(label)
        irreps.append(irrep)
    assert numpy.allclose(q.T @ q, numpy.eye(n), atol=1e-14), "Q is not orthonormal"
    assert set(irreps) <= set(case["irreps"])
    return q, labels, irreps


def tasks_for(case: dict, natom: int) -> list[tuple]:
    """Every energy the case reads, keyed by what it is for:
    ("hessian", j, d, i, o): along coordinate j by d, Cartesian gradient
    component i at offset o delta; ("bilinear", i, j, d, o): along
    (q_i + q_j) / sqrt 2, at d + o delta; ("gradient", i, o): the reference
    gradient."""
    h, delta = case["h"], case["delta"]
    n = 3 * natom
    basis, _, _ = coordinates(case)
    out = []
    for i in range(n):
        for o in stencil.OFFSETS:
            out.append((("gradient", i, o), o * delta * unit(natom, i)))
    for j in range(n):
        along = basis[:, j].reshape(natom, 3)
        for d in stencil.displacements(h):
            for i in range(n):
                for o in stencil.OFFSETS:
                    dr = d * along + o * delta * unit(natom, i)
                    out.append((("hessian", j, d, i, o), dr))
    for i in range(n):
        for j in range(i + 1, n):
            q = ((basis[:, i] + basis[:, j]) / numpy.sqrt(2.0)).reshape(natom, 3)
            for d in stencil.displacements(h):
                for o in stencil.OFFSETS:
                    out.append((("bilinear", i, j, d, o), (d + o * delta) * q))
    return out


def evaluate(case: dict, ref: Reference, jobs: int) -> dict:
    natom = len(case["atoms"])
    tasks = tasks_for(case, natom)
    print(f"{case['name']}: {len(tasks)} energies over {jobs} workers", flush=True)
    results = {}
    started = time.perf_counter()
    # Forked workers inherit the reference as it stands.
    global REFERENCE
    REFERENCE = ref
    context = multiprocessing.get_context("fork")
    with context.Pool(jobs, initializer=init_worker) as pool:
        for count, (key, value) in enumerate(
            pool.imap_unordered(run_point, tasks, chunksize=4), 1
        ):
            results[key] = value
            if count % 500 == 0:
                elapsed = time.perf_counter() - started
                print(f"  {count}/{len(tasks)} in {elapsed:.0f} s", flush=True)
    return results


# --------------------------------------------------------------- the analysis


def floor_of(sigma: float, case: dict) -> float:
    """FLOOR_FACTOR times the finer difference's noise (see the docstring)."""
    _, half, quarter = stencil.steps(case["h"])
    gain = stencil.noise_gain(case["delta"])
    return (
        FLOOR_FACTOR
        * sigma
        * gain
        * (stencil.noise_gain(half) + stencil.noise_gain(quarter))
    )


def analyse(case: dict, ref: Reference, results: dict, sigma: dict) -> dict:
    name = case["name"]
    natom = len(case["atoms"])
    n = 3 * natom
    h, delta = case["h"], case["delta"]
    disps = stencil.displacements(h)

    def grad_at(q: str, energy) -> list[float]:
        return [
            stencil.derivative(
                {o * delta: energy(i, o) for o in stencil.OFFSETS}, delta
            )
            for i in range(n)
        ]

    basis, labels, irreps = coordinates(case)
    out = {
        "quantities": {},
        "coordinates": {"labels": labels, "irreps": irreps, "vectors": rows(basis.T)},
    }
    failures = []
    worst = {"prediction": 0.0, "subspace": 0.0, "ci": 0.0, "inactive_gap": numpy.inf}
    for value in results.values():
        for key, number in value["guard"].items():
            if key == "inactive_gap":
                worst[key] = min(worst[key], number)
            else:
                worst[key] = max(worst[key], number)
    out["guard"] = {k: float(v) for k, v in worst.items()}

    for q in case["quantities"]:
        gradient = grad_at(q, lambda i, o, q=q: results["gradient", i, o][q])
        floor = floor_of(sigma[q], case)
        # points[j][d] = Q^T g at R0 + d q_j: the gradient in the
        # coordinates.
        points = [
            {
                d: [
                    float(x)
                    for x in basis.T
                    @ numpy.array(
                        grad_at(
                            q,
                            lambda i, o, j=j, d=d, q=q: results["hessian", j, d, i, o][
                                q
                            ],
                        )
                    )
                ]
                for d in disps
            }
            for j in range(n)
        ]
        # ladder[k][i][j] = D_k(i, j) at the k-th step.
        columns = [stencil.column_ladder(points[j], h) for j in range(n)]
        ladder = [
            [[columns[j][k][i] for j in range(n)] for i in range(n)] for k in range(3)
        ]
        verdicts = [
            [
                stencil.judge([ladder[k][i][j] for k in range(3)], floor, RATIO_WINDOW)
                for j in range(n)
            ]
            for i in range(n)
        ]
        bar = [[v["bar"] for v in row] for row in verdicts]
        finest = ladder[-1]
        asym = stencil.asymmetry(finest)
        may_fail = q in case.get("may_fail", ())

        def gated(i: int, j: int, may_fail: bool = may_fail) -> bool:
            """Whether a miss at (i, j) refuses the case, or is recorded as
            its block's failure: a quantity the case expects to fail is
            held only on the totally symmetric block."""
            return not may_fail or irreps[i] == irreps[j] == case["irreps"][0]

        failed_blocks = set()

        def miss(
            i: int, j: int, text: str, q: str = q, failed: set = failed_blocks
        ) -> None:
            if gated(i, j):
                failures.append(f"{name} {q}: {text}")
            else:
                failed.add(tuple(sorted((irreps[i], irreps[j]))))

        judged = []
        for i, row in enumerate(verdicts):
            for j, v in enumerate(row):
                if v["settled"] is False:
                    shown = "unjudged" if v["ratio"] is None else f"{v['ratio']:.3f}"
                    miss(
                        i,
                        j,
                        f"H[{i}][{j}] ratio {shown} (D(h)-D(h/2) {v['upper']:.3e}, "
                        f"D(h/2)-D(h/4) {v['lower']:.3e}, floor {floor:.1e})",
                    )
                elif v["ratio"] is not None and gated(i, j):
                    judged.append(v["ratio"])
        misses = stencil.symmetry_misses(finest, bar)
        for i, j, value, allowed in misses:
            miss(i, j, f"|H[{i}][{j}] - H[{j}][{i}]| {value:.3e} > bar {allowed:.3e}")

        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                along = {
                    d: [
                        stencil.derivative(
                            {
                                o * delta: results["bilinear", i, j, d, o][q]
                                for o in stencil.OFFSETS
                            },
                            delta,
                        )
                    ]
                    for d in disps
                }
                qq_ladder = [v[0] for v in stencil.column_ladder(along, h)]
                verdict = stencil.judge(qq_ladder, floor, RATIO_WINDOW)
                if verdict["settled"] is False:
                    shown = verdict["ratio"]
                    shown = "unjudged" if shown is None else f"{shown:.3f}"
                    miss(
                        i,
                        j,
                        f"(q_{i} + q_{j}) / sqrt 2 ratio {shown} "
                        f"({verdict['upper']:.3e} / {verdict['lower']:.3e})",
                    )
                elif verdict["ratio"] is not None and gated(i, j):
                    judged.append(verdict["ratio"])
                check = stencil.bilinearity(
                    finest, bar, i, j, qq_ladder[-1], verdict["bar"]
                )
                check["ladder"] = qq_ladder
                check["ratio"] = verdict["ratio"]
                check["settled"] = verdict["settled"]
                check["derivatives"] = [along[d][0] for d in disps]
                if check["fails"]:
                    miss(
                        i,
                        j,
                        f"pair {[i, j]} not bilinear, residual "
                        f"{check['residual']:.3e} > {check['allowed']:.3e}",
                    )
                pairs.append(check)
        failing = [p["pair"] for p in pairs if p["fails"]]
        worst_pair = max(pairs, key=lambda p: abs(p["residual"]) / p["allowed"])
        print(
            f"{name} {q}: sigma_E {sigma[q]:.1e}, floor {floor:.1e}, "
            f"{len(judged)} ratios judged {min(judged, default=0):.3f}.."
            f"{max(judged, default=0):.3f}, max bar {max(map(max, bar)):.2e}, "
            f"max asymmetry {max(map(max, asym)):.2e}, "
            f"{len(misses)} asymmetric pairs, {len(failing)} failing pairs "
            f"(worst residual {worst_pair['residual']:.2e} against "
            f"{worst_pair['allowed']:.2e} at {worst_pair['pair']}), "
            f"failed blocks {sorted(failed_blocks)}",
            flush=True,
        )
        out["quantities"][q] = {
            "gradient": [gradient[3 * a : 3 * a + 3] for a in range(natom)],
            "hessian": rows(stencil.symmetrized(finest)),
            "bar": rows(stencil.symmetric_bar(bar)),
            "ladder": [rows(m) for m in ladder],
            "ratios": [[v["ratio"] for v in row] for row in verdicts],
            "column_bar": rows(bar),
            "asymmetry": rows(asym),
            "noise": {"sigma_e": sigma[q], "floor": floor},
            "ratio_range": [min(judged), max(judged)] if judged else None,
            "ratios_judged": len(judged),
            "worst_ratio_deviation": max(
                (abs(r - stencil.TARGET) for r in judged), default=None
            ),
            "points": [
                {"column": j, "gradients": [points[j][d] for d in disps]}
                for j in range(n)
            ],
            "bilinearity": pairs,
            "failing_pairs": failing,
            "asymmetric_pairs": [[i, j] for i, j, _, _ in misses],
            "failed_blocks": [list(b) for b in sorted(failed_blocks)],
        }
        if not judged:
            failures.append(f"{name} {q}: no ratio above the floor")
    if case["kind"] == "symmetric":
        failed = [
            q for q in case.get("may_fail", ()) if out["quantities"][q]["failed_blocks"]
        ]
        if not failed:
            failures.append(f"{name}: the symmetric case has no failing block")
    analytic = float(
        numpy.abs(
            numpy.array(out["quantities"]["casscf"]["gradient"]) - ref.analytic
        ).max()
    )
    print(f"{name}: CASSCF stencil gradient - analytic {analytic:.1e}", flush=True)
    if analytic > ANALYTIC_TOL:
        failures.append(f"{name}: CASSCF gradient off analytic by {analytic:.1e}")
    out["analytic_casscf_gradient_miss"] = analytic
    out["failures"] = failures
    return out


def record(case: dict, ref: Reference, sigma: dict, analysis: dict) -> dict:
    nev = ref.nevpt2
    zero = nevpt2.NUMERICAL_ZERO
    return {
        "description": "Semi-numerical single-state SC- and PC-NEVPT2 (and "
        "CASSCF) nuclear Hessian: a fourth-order central stencil at steps h, "
        "h/2, h/4 over the fourth-order central gradient of re-converged DF "
        "energies (PySCF; PC by the C++ step's cut), every point through the "
        "continuity guard. Hartree, bohr; H[i][j] = dg_i/dR_j, i = 3 atom + "
        "axis. tools/nevpt2_hessian/nevpt2_hessian.py",
        "provenance": {
            "pyscf": pyscf.__version__,
            "block2": version("block2"),
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "name": case["name"],
        "kind": case["kind"],
        "basis": {"name": case["basis"], **provenance(BASES[case["basis"]])},
        "aux_basis": {
            "name": AUX[0],
            **provenance(AUX[1]),
            "naux": int(df.addons.make_auxmol(ref.mol, ref.auxbasis).nao),
        },
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
        "active": case["active"],
        "weights": [1.0],
        "frozen_core": 0,
        "polish_grad": ref.polish,
        "energies": {
            "casscf": ref.energy,
            "sc": ref.energy + nev["sc"],
            "pc": ref.energy + nev["pc"],
        },
        "e_corr": {"sc": nev["sc"], "pc": nev["pc"]},
        "classes": [
            {
                "name": key,
                "norm": nev["sc_classes"][key][0],
                "energy": nev["sc_classes"][key][1],
                "pc_energy": nev["pc_classes"][key],
                "sc_empty_perturbers": int((numpy.abs(nev["norms"][key]) <= zero).sum())
                if key != "Sijrs"
                else 0,
                "sc_perturbers": int(nev["norms"][key].size),
                "pc_dropped_modes": nev["dropped"].get(key, 0),
            }
            for key in CLASSES
        ],
        "inactive_gaps": nev["gaps"],
        "casscf_analytic_gradient": rows(ref.analytic),
        "steps": {
            "h": case["h"],
            "ladder": stencil.steps(case["h"]),
            "displacements": stencil.displacements(case["h"]),
            "delta": case["delta"],
            "offsets": list(stencil.OFFSETS),
            "weights": list(stencil.WEIGHTS),
        },
        "tolerances": {
            "ratio_window": list(RATIO_WINDOW),
            "floor_factor": FLOOR_FACTOR,
            "polish_grad": POLISH_GRAD,
            "prediction_bound": PREDICTION_BOUND,
            "subspace_bound": SUBSPACE_BOUND,
            "ci_bound": CI_BOUND,
            "degenerate": DEGENERATE,
            "pc_cut": pc_cut(1),
            "analytic": ANALYTIC_TOL,
            "block2_pc": BLOCK2_PC,
        },
        "noise": {"starts": NOISE_STARTS, "kick": NOISE_KICK, "sigma_e": sigma},
        "checks": {
            **ref.checks,
            "casscf_gradient_vs_analytic": analysis["analytic_casscf_gradient_miss"],
        },
        "guard": analysis["guard"],
        "coordinates": {
            **analysis["coordinates"],
            "irreps_order": list(case["irreps"]),
        },
        "may_fail": list(case.get("may_fail", ())),
        **analysis["quantities"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=OUT, help="directory")
    parser.add_argument("--only", nargs="+", help="these cases alone")
    parser.add_argument(
        "--jobs",
        type=int,
        default=JOBS,
        help=f"worker processes, one thread each (default and cap {JOBS})",
    )
    args = parser.parse_args()
    assert 1 <= args.jobs <= JOBS, f"--jobs {args.jobs}: 1 to {JOBS}"
    lib.num_threads(1)
    assert lib.num_threads() == 1
    status = 0
    for case in CASES:
        if args.only and case["name"] not in args.only:
            continue
        started = time.perf_counter()
        ref = Reference(case)
        sigma = ref.noise()
        print(
            f"{case['name']}: reference in {time.perf_counter() - started:.0f} s, "
            f"sigma_E {sigma}",
            flush=True,
        )
        results = evaluate(case, ref, args.jobs)
        analysis = analyse(case, ref, results, sigma)
        print(f"{case['name']}: {time.perf_counter() - started:.0f} s wall", flush=True)
        if analysis["failures"]:
            print(f"STUCK: {case['name']}", flush=True)
            for line in analysis["failures"]:
                print(f"  {line}", flush=True)
            status = 1
            continue
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"{case['name']}.json"
        path.write_text(json.dumps(record(case, ref, sigma, analysis), indent=1) + "\n")
        print(f"wrote {path}", flush=True)
    sys.exit(status)


if __name__ == "__main__":
    main()
