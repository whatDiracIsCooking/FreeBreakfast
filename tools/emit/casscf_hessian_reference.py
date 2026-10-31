#!/usr/bin/env python3
"""Write the reference data the DF-CASSCF analytic nuclear Hessian's tests read.

The design is `casscf_hessian.tex` (with `check_df.py` and `check_sa.py`
beside it), ported here onto density fitting throughout: a Coulomb-metric fit
over this tree's own data/aux_basis file, the metric applied through its
Cholesky factor everywhere (never an explicit M^-1: at cond(M) ~ 2e6 an
inverse puts ~1e-10 on every energy and 1e-4 on a second difference).

**The energy.** E(x, lambda) is the CASSCF energy over the fitted integrals
at geometry x along the symmetric (Lowdin) connection, orbitals
C(x, kappa) = C T(x) e^K, T = (C^T S(x) C)^{-1/2} (the connection first, the
rotation second: the design's eq. (order)), and the roots
(V e^A)[:, :nroots] over an orthonormal basis V of the singlet CI space, the
roots first. The parameters lambda are the non-redundant orbital rotations
(core-active, core-virtual, active-virtual), the root--complement CI
rotations, and at unequal weights the rotations among the roots (at equal
weights E_SA is invariant to them, and they are not parameters). PySCF's
DF-(SA-)CASSCF over the same fitting set is the starting point only: the
solution is taken to a minimum of this script's own energy and polished by
Newton steps to |dE_SA/dlambda| < POLISH_GRADIENT, and every reference is
that energy. **Three of PySCF's solutions here are saddle points**, each
along an active--virtual rotation: the two-root averages (the SA Hessian's
least eigenvalue -3.8e-3 at (0.5, 0.5), -1.4e-2 at (0.7, 0.3); minima 1.9e-3
and 6.7e-3 hartree lower in E_SA), and the cc-pVDZ single state (-2.4e-2:
PySCF's default active space settles on an O lone pair and a d function,
natural occupations 1.997 and 0.002; the minimum, 5.2e-2 lower, is the two
sigma/sigma* pairs, 1.89, 1.85, 0.16, 0.11). Saddle-free Newton steps take
each to the minimum, which is what a second-order optimizer converges to and
what a conjugate-gradient CP solve needs. The
manifest's `descent` records both.

**Analytic, not differenced.** PySCF supplies the derivative AO integrals
(first and second, analytic: int1e_ip*/ipip*, int3c2e_ip*/ipip*/ipvip1/
ip1ip2, int2c2e_ip*, its RHF Hessian's hcore generator and hess_nuc), each
checked against a finite difference and the translational sum rule. Every
fixed-parameter quantity (F^[x], the CP right-hand sides, E_xy and its
pieces, L_x lambda, the couplings) is a closed formula over them. Derivatives
in lambda come from the exact gradient of E(lambda) (the exponentials'
Frechet derivatives, scipy's expm_frechet), differenced to fourth order:
the parameter Hessians, Hessian-vector products and the third-derivative
action. So the written quantities are good to ~1e-10, far past what the
finite-difference checks below can see.

**The checks** (the script refuses to write if one misses; the numbers go to
the manifest's `checks`):

- the derivative integrals against finite differences and the
  translational sum rule;
- the design's check_df table: (df2) against a fourth-order difference of
  1/2 Gamma.(T N T^T) at fixed Gamma^AO, and without its cross term; the J,
  K and active cross terms against the generic Gamma_ab one; E_xy (exy)
  against the fixed-parameter difference of the energy; the CP right-hand
  sides against differences of the exact gradient in x; the Hessian against
  second differences of relaxed energies (diagonal and three random
  directions);
- the design's check_sa table: root I's gradient with the root rotations
  projected out of the Z-vector and with them in A, against differences of
  relaxed energies (projected is wrong at three roots, unequal weights); the
  Hessian with (sa) + (rot) at equal weights, and without (rot) and with one
  copy of it (both must miss);
- E_xy at general densities (a random rotation) against differences, the z
  part's densities against a difference in z, L_x lambda's z part against a
  difference of the Hessian-vector product in x, the third-derivative
  action's symmetry and its step-halving;
- every analytic Hessian against `relaxed_fd`, a fourth-order central
  difference of the analytic gradient of the relaxed energy (re-solved at
  each displaced geometry, Newton to RELAX_GRADIENT, with its own Z-vector at
  every displaced geometry); its symmetry and the translational sum rule.

Cases, all over the design's stretched C1 water (O-H 1.40 and 1.51 A),
CAS(4,4) from PySCF's default active space, with cc-pVDZ-JKFIT:

- `h2o_631g`: 6-31G, state-specific (the design's case);
- `h2o_631g_sa2`, `h2o_631g_sa2_unequal`: two singlet roots at weights
  (0.5, 0.5) and (0.7, 0.3);
- `h2o_631g_sa3`, `h2o_631g_sa3_unequal`: three singlet roots at
  (1/3, 1/3, 1/3) and (0.5, 0.3, 0.2);
- `h2o_ccpvdz`: cc-pVDZ (d shells on O), state-specific.

`test/gtest/data/README.md` has every array's shape, index order and sign
convention. Arrays are '<f8' Fortran-order .npy files (first axis fastest);
the manifest is casscf_hessian.json. BLAS and OpenMP are pinned to one
thread and nothing run-dependent is written, so a rerun regenerates every
file byte for byte.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/casscf_hessian_reference.py

writes test/gtest/data/casscf_hessian/. Re-run it to regenerate; the files
are never hand-edited.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
import scipy  # noqa: E402
from df_reference import libcint_version  # noqa: E402
from pyscf import df, fci, gto, lib, mcscf, scf  # noqa: E402
from pyscf.grad import rhf as rhf_grad  # noqa: E402
from pyscf.hessian import rhf as rhf_hessian  # noqa: E402
from scf_reference import pyscf_basis  # noqa: E402
from scipy.linalg import expm, expm_frechet, solve_triangular  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "casscf_hessian"
MANIFEST = OUT / "casscf_hessian.json"

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/casscf_hessian_reference.py"
)

ANGSTROM = 1.0 / lib.param.BOHR
# The design's water (check_casscf.py): both O-H bonds stretched, to 1.40 and
# 1.51 A, in no symmetry plane, so the active space is the two sigma/sigma*
# pairs (natural occupations near 1.88, 1.84, 0.16, 0.12). Bohr.
WATER_STRETCHED = [
    ("O", (0.0, 0.0, 0.1 * ANGSTROM)),
    ("H", (0.0, 1.25 * ANGSTROM, -0.75 * ANGSTROM)),
    ("H", (0.1 * ANGSTROM, -1.15 * ANGSTROM, -0.7 * ANGSTROM)),
]

# This tree's basis files (pyscf_basis reads from data/basis, hence the
# relative path for the fitting set).
BASES = {"6-31g": "6-31g.json", "cc-pvdz": "cc-pVDZ.json"}
AUX_FILES = {"cc-pvdz-jkfit": "../aux_basis/cc-pVDZ-JKFIT.json"}

CASES = [
    {"name": "h2o_631g", "basis": "6-31g", "weights": [1.0]},
    {"name": "h2o_631g_sa2", "basis": "6-31g", "weights": [0.5, 0.5]},
    {"name": "h2o_631g_sa2_unequal", "basis": "6-31g", "weights": [0.7, 0.3]},
    {"name": "h2o_631g_sa3", "basis": "6-31g", "weights": [1 / 3, 1 / 3, 1 / 3]},
    {"name": "h2o_631g_sa3_unequal", "basis": "6-31g", "weights": [0.5, 0.3, 0.2]},
    {"name": "h2o_ccpvdz", "basis": "cc-pvdz", "weights": [1.0]},
]
AUX_BASIS = "cc-pvdz-jkfit"
NCAS, NELECAS = 4, (2, 2)

# The polish and every re-relaxation: Newton until no component of dE_SA /
# dlambda is above these.
POLISH_GRADIENT = 1e-12
RELAX_GRADIENT = 1e-11
# Fourth-order differences of the exact gradient in lambda (unit-norm
# directions), and of the third-derivative action (the second step checks the
# first).
LAMBDA_STEP = 1e-3
THIRD_STEP = 1e-2
# The descent from a saddle: the largest step, and the step off it.
MAX_STEP = 0.2
SADDLE_STEP = 0.3
# Differences in x (bohr): the relaxed-gradient reference (fourth order), the
# fixed-parameter checks (fourth order), the relaxed-energy checks (second
# order, as the design), and the derivative integrals (central).
X_STEP = 1e-3
INTEGRAL_STEP = 1e-4

# Tolerances, each a little above what the run measures (the manifest's
# `checks` has the measured values).
TOL_INTEGRAL = 1e-6  # derivative integrals vs a central difference, relative
TOL_SUM_RULE = 1e-9  # translational sum rules, relative to the largest element
TOL_EXACT = 1e-11  # two analytic forms of the same quantity
TOL_FIXED = 1e-7  # fixed-parameter analytic vs fourth-order differences
TOL_RELAXED_FD = 1e-7  # every Hessian vs relaxed_fd
TOL_RELAXED_ENERGY = 5e-6  # the design's second differences of energies
TOL_THIRD = 1e-8  # the third-derivative action, step against half step
# A term the design says is not optional must miss by at least this.
MUST_MISS = 1e-3

FOURTH = ((-2, 1 / 12), (-1, -8 / 12), (1, 8 / 12), (2, -1 / 12))


# --- integrals ------------------------------------------------------------


def molecules(case: dict, coords: numpy.ndarray):
    symbols = [symbol for symbol, _ in WATER_STRETCHED]
    mol = gto.M(
        atom=[[s, tuple(r)] for s, r in zip(symbols, coords, strict=True)],
        unit="Bohr",
        basis=pyscf_basis(BASES[case["basis"]], set(symbols)),
        cart=False,
        verbose=0,
    )
    aux = pyscf_basis(AUX_FILES[AUX_BASIS], set(symbols))
    return mol, df.addons.make_auxmol(mol, aux)


def lower_solve(lower: numpy.ndarray, rhs: numpy.ndarray) -> numpy.ndarray:
    return solve_triangular(lower, rhs, lower=True)


def slices(mol: gto.Mole) -> list[slice]:
    return [slice(p0, p1) for _, _, p0, p1 in mol.aoslice_by_atom()]


def two_center_first(ip: numpy.ndarray, atoms: list[slice]) -> numpy.ndarray:
    """d/dR (a|b) from ip = (nabla a|b), each nabla = -d/dR of its centre."""
    n = ip.shape[1]
    out = numpy.zeros((3 * len(atoms), n, n))
    for a, sa in enumerate(atoms):
        for k in range(3):
            m = numpy.zeros((n, n))
            m[sa] = -ip[k][sa]
            out[3 * a + k] = m + m.T
    return out


def two_center_second(ipip, ipip_mixed, atoms: list[slice]) -> numpy.ndarray:
    """d^2/dR dR (a|b) from (nabla nabla a|b) and (nabla a|nabla b)."""
    n = ipip.shape[1]
    na = len(atoms)
    out = numpy.zeros((3 * na, 3 * na, n, n))
    for a, sa in enumerate(atoms):
        for b, sb in enumerate(atoms):
            for k in range(3):
                for j in range(3):
                    m = out[3 * a + k, 3 * b + j]
                    if a == b:
                        m[sa] += ipip[3 * k + j][sa]
                        m[:, sa] += ipip[3 * k + j][sa].T
                    m[sa, sb] += ipip_mixed[3 * k + j][sa, sb]
                    m[sb, sa] += ipip_mixed[3 * j + k][sb, sa]
    return out


class Geometry:
    """The AO and fitted integrals at one geometry, and their derivatives to
    the given order. Pair-space arrays are (nao^2, naux) with pair m n at
    m nao + n; the fitted factor B = T L^-T, L the metric's Cholesky factor."""

    def __init__(self, case: dict, coords: numpy.ndarray, order: int = 0):
        self.case, self.coords = case, coords
        mol, aux = molecules(case, coords)
        self.mol, self.aux = mol, aux
        nao, naux = mol.nao, aux.nao
        self.nao, self.naux, self.n = nao, naux, 3 * mol.natm
        self.h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        self.s = mol.intor("int1e_ovlp")
        self.t = df.incore.aux_e2(mol, aux, "int3c2e", aosym="s1").reshape(
            nao * nao, naux
        )
        self.metric = aux.intor("int2c2e")
        self.enuc = mol.energy_nuc()
        self.chol = numpy.linalg.cholesky(self.metric)
        self.b = lower_solve(self.chol, self.t.T).T
        if order >= 1:
            self.first()
        if order >= 2:
            self.second()

    def first(self):
        mol, aux, nao, naux, n = self.mol, self.aux, self.nao, self.naux, self.n
        ao, ax = slices(mol), slices(aux)
        grad = scf.RHF(mol).nuc_grad_method()
        hcore = grad.hcore_generator(mol)
        self.h1 = numpy.array([hcore(a)[k] for a in range(mol.natm) for k in range(3)])
        self.s1 = two_center_first(mol.intor("int1e_ipovlp"), ao)
        self.m1 = two_center_first(aux.intor("int2c2e_ip1"), ax)
        ip1 = df.incore.aux_e2(mol, aux, "int3c2e_ip1", aosym="s1", comp=3)
        ip2 = df.incore.aux_e2(mol, aux, "int3c2e_ip2", aosym="s1", comp=3)
        t1 = numpy.zeros((n, nao, nao, naux))
        for a, (sa, pa) in enumerate(zip(ao, ax, strict=True)):
            for k in range(3):
                t1[3 * a + k, sa] -= ip1[k][sa]
                t1[3 * a + k, :, sa] -= ip1[k].transpose(1, 0, 2)[:, sa]
                t1[3 * a + k, :, :, pa] -= ip2[k][:, :, pa]
        self.t1 = t1.reshape(n, nao * nao, naux)
        self.enuc1 = rhf_grad.grad_nuc(mol).reshape(-1)
        # R^x L^-T = (T^x - c M^x) L^-T = Y^x - B Z^x, Y^x = T^x L^-T,
        # Z^x = L^-1 M^x L^-T: the fit's first derivative, never through N.
        self.y1 = numpy.array([lower_solve(self.chol, t.T).T for t in self.t1])
        self.z1 = numpy.array(
            [lower_solve(self.chol, lower_solve(self.chol, m).T) for m in self.m1]
        )
        self.r1 = numpy.array(
            [y - self.b @ z for y, z in zip(self.y1, self.z1, strict=True)]
        )

    def second(self):
        mol, aux, nao, naux, n = self.mol, self.aux, self.nao, self.naux, self.n
        ao, ax = slices(mol), slices(aux)
        hobj = scf.RHF(mol).Hessian()
        hcore = hobj.hcore_generator(mol)
        h2 = numpy.zeros((n, n, nao, nao))
        for a in range(mol.natm):
            for b in range(mol.natm):
                h2[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] = hcore(a, b)
        self.h2 = h2
        self.s2 = two_center_second(
            mol.intor("int1e_ipipovlp"), mol.intor("int1e_ipovlpip"), ao
        )
        self.m2 = two_center_second(
            aux.intor("int2c2e_ipip1"), aux.intor("int2c2e_ip1ip2"), ax
        )

        def three(name):
            return df.incore.aux_e2(mol, aux, name, aosym="s1", comp=9)

        ipip1, ipvip1 = three("int3c2e_ipip1"), three("int3c2e_ipvip1")
        ip1ip2, ipip2 = three("int3c2e_ip1ip2"), three("int3c2e_ipip2")
        t2 = numpy.zeros((n, n, nao, nao, naux))
        for a, (sa, pa) in enumerate(zip(ao, ax, strict=True)):
            for b, (sb, pb) in enumerate(zip(ao, ax, strict=True)):
                for k in range(3):
                    for j in range(3):
                        kl, lk = 3 * k + j, 3 * j + k
                        m = t2[3 * a + k, 3 * b + j]
                        if a == b:
                            m[sa] += ipip1[kl][sa]
                            m[:, sa] += ipip1[kl].transpose(1, 0, 2)[:, sa]
                            m[:, :, pa] += ipip2[kl][:, :, pa]
                        m[sa, sb] += ipvip1[kl][sa, sb]
                        m[sb, sa] += ipvip1[lk][sb, sa]
                        m[sa, :, pb] += ip1ip2[kl][sa, :, pb]
                        m[sb, :, pa] += ip1ip2[lk][sb, :, pa]
                        m[:, sa, pb] += ip1ip2[kl].transpose(1, 0, 2)[:, sa, pb]
                        m[:, sb, pa] += ip1ip2[lk].transpose(1, 0, 2)[:, sb, pa]
        self.t2 = t2.reshape(n, n, nao * nao, naux)
        self.enuc2 = rhf_hessian.hess_nuc(mol).transpose(0, 2, 1, 3).reshape(n, n)

    def eri(self) -> numpy.ndarray:
        """The fitted (mn|ls), pair space."""
        return self.b @ self.b.T


def displaced(coords: numpy.ndarray, *steps) -> numpy.ndarray:
    out = coords.copy()
    for k, s in steps:
        out.flat[k] += s
    return out


def check_integrals(case: dict, geom: Geometry, checks: dict):
    """The analytic derivative integrals against central differences (the
    first of the integrals, the second of the analytic first) and the
    translational sum rule."""
    h = INTEGRAL_STEP
    worst, worst_rule = 0.0, 0.0
    for k in range(geom.n):
        plus = Geometry(case, displaced(geom.coords, (k, h)), 1)
        minus = Geometry(case, displaced(geom.coords, (k, -h)), 1)
        first = [
            (geom.h1[k], plus.h, minus.h),
            (geom.s1[k], plus.s, minus.s),
            (geom.t1[k], plus.t, minus.t),
            (geom.m1[k], plus.metric, minus.metric),
            (geom.enuc1[k], plus.enuc, minus.enuc),
        ]
        second = [
            (geom.h2[k], plus.h1, minus.h1),
            (geom.s2[k], plus.s1, minus.s1),
            (geom.t2[k], plus.t1, minus.t1),
            (geom.m2[k], plus.m1, minus.m1),
            (geom.enuc2[k], plus.enuc1, minus.enuc1),
        ]
        for analytic, p, m in first + second:
            fd = (numpy.asarray(p) - numpy.asarray(m)) / (2 * h)
            scale = max(numpy.abs(analytic).max(), 1.0)
            worst = max(worst, numpy.abs(analytic - fd).max() / scale)
    natm = geom.n // 3
    for arr in (geom.h1, geom.s1, geom.t1, geom.m1, geom.enuc1):
        rule = numpy.abs(arr.reshape(natm, 3, -1).sum(axis=0)).max()
        worst_rule = max(worst_rule, rule / numpy.abs(arr).max())
    for arr in (geom.h2, geom.s2, geom.t2, geom.m2, geom.enuc2):
        rule = numpy.abs(arr.reshape(geom.n, natm, 3, -1).sum(axis=1)).max()
        worst_rule = max(worst_rule, rule / numpy.abs(arr).max())
    print(
        f"  derivative integrals vs central difference {worst:.1e} (relative), "
        f"sum rules {worst_rule:.1e}"
    )
    assert worst < TOL_INTEGRAL, (case["name"], "integrals", worst)
    assert worst_rule < TOL_SUM_RULE, (case["name"], "integral sum rule", worst_rule)
    checks["derivative_integrals_vs_fd"] = worst
    checks["derivative_integrals_sum_rule"] = worst_rule


# --- the CASSCF energy and its exact parameter gradient ---------------------


def inv_sqrt(a: numpy.ndarray) -> numpy.ndarray:
    w, v = numpy.linalg.eigh(a)
    return (v / numpy.sqrt(w)) @ v.T


def symmetrize_rdm2(g: numpy.ndarray) -> numpy.ndarray:
    g = (g + g.transpose(1, 0, 2, 3)) / 2
    g = (g + g.transpose(0, 1, 3, 2)) / 2
    return (g + g.transpose(2, 3, 0, 1)) / 2


def one_index(hm: numpy.ndarray, gm: numpy.ndarray, a: numpy.ndarray):
    """The integrals one-index transformed by a on every index."""
    h1 = a.T @ hm + hm @ a
    g = numpy.einsum("ap,aqrs->pqrs", a, gm)
    g = g + g.transpose(1, 0, 2, 3)
    g = g + g.transpose(2, 3, 0, 1)
    return h1, g


def singlet_count(norb: int, nelec: tuple[int, int]) -> int:
    """The number of singlet CSFs of n = nelec[0] + nelec[1] electrons in
    norb orbitals (Weyl's formula at S = 0)."""
    n = nelec[0] + nelec[1]
    return math.comb(norb + 1, n // 2) * math.comb(norb + 1, n // 2 + 1) // (norb + 1)


class Model:
    """One case's spaces and parameters: ncore, the active space, the CI
    strings, the roots and their weights, and the parameter layout."""

    def __init__(self, case: dict, nmo: int, nelectron: int):
        self.weights = numpy.array(case["weights"])
        self.nroots = len(self.weights)
        self.equal = bool(numpy.ptp(self.weights) < 1e-12)
        if not self.equal:
            gaps = [
                abs(a - b)
                for i, a in enumerate(self.weights)
                for b in self.weights[i + 1 :]
            ]
            assert min(gaps) > 1e-12, "partly equal weights are not handled"
        self.nmo = nmo
        self.ncas, self.nelecas = NCAS, NELECAS
        self.ncore = (nelectron - sum(NELECAS)) // 2
        self.nocc = self.ncore + NCAS
        self.core = slice(0, self.ncore)
        self.act = slice(self.ncore, self.nocc)
        self.na = fci.cistring.num_strings(NCAS, NELECAS[0])
        self.nb = fci.cistring.num_strings(NCAS, NELECAS[1])
        assert self.na == self.nb
        self.ndet = self.na * self.nb
        spaces = [
            range(self.ncore),
            range(self.ncore, self.nocc),
            range(self.nocc, nmo),
        ]
        self.pairs = [
            (p, q)
            for a in range(3)
            for b in range(a)
            for p in spaces[a]
            for q in spaces[b]
        ]
        self.nk = len(self.pairs)
        # The basis V's columns: the roots, then their complement in the
        # singlet CI space (set by basis()).
        self.nv = singlet_count(NCAS, NELECAS)
        self.cirot = [
            (r, q) for r in range(self.nroots) for q in range(self.nroots, self.nv)
        ]
        self.ntheta = self.nk + len(self.cirot)
        self.thetas = [
            (r, s) for r in range(self.nroots) for s in range(r + 1, self.nroots)
        ]
        self.npar = self.ntheta + len(self.thetas)
        # The rotations among the roots are parameters only at unequal
        # weights; at equal weights E_SA is invariant to them.
        self.p = numpy.arange(self.npar if not self.equal else self.ntheta)

    def basis(self, roots: numpy.ndarray) -> numpy.ndarray:
        """An orthonormal basis of the singlet CI space, the roots first."""
        shape = (self.na, self.nb)
        roots = numpy.array([(0.5 * (c + c.T)).ravel() for c in roots]).T
        roots = numpy.linalg.qr(roots)[0]
        dets = numpy.eye(self.ndet)
        s2 = numpy.array(
            [
                fci.spin_op.contract_ss(
                    d.reshape(shape), self.ncas, self.nelecas
                ).ravel()
                for d in dets
            ]
        )
        w, u = numpy.linalg.eigh(0.5 * (s2 + s2.T))
        singlets = u[:, numpy.abs(w) < 1e-8]
        assert singlets.shape[1] == self.nv, (singlets.shape, self.nv)
        assert numpy.abs(roots - singlets @ (singlets.T @ roots)).max() < 1e-4
        rest = singlets - roots @ (roots.T @ singlets)
        u = numpy.linalg.svd(rest, full_matrices=False)[0]
        return numpy.hstack([roots, u[:, : self.nv - self.nroots]])

    def kmat(self, lam: numpy.ndarray) -> numpy.ndarray:
        k = numpy.zeros((self.nmo, self.nmo))
        for (p, q), x in zip(self.pairs, lam[: self.nk], strict=True):
            k[p, q], k[q, p] = x, -x
        return k

    def amat(self, lam: numpy.ndarray) -> numpy.ndarray:
        a = numpy.zeros((self.nv, self.nv))
        for (r, q), x in zip(self.cirot, lam[self.nk : self.ntheta], strict=True):
            a[q, r], a[r, q] = x, -x
        for (r, s), x in zip(self.thetas, lam[self.ntheta :], strict=True):
            a[s, r], a[r, s] = x, -x
        return a

    def full_rdms(self, g1, g2, n0) -> tuple[numpy.ndarray, numpy.ndarray]:
        """D and Gamma over every MO (E = sum h D + 1/2 sum (pq|rs) Gamma)
        from active (transition) RDMs, the core weighted by n0 = <bra|ket>."""
        nmo, act = self.nmo, self.act
        d = numpy.zeros((nmo, nmo))
        g = numpy.zeros((nmo,) * 4)
        for i in range(self.ncore):
            d[i, i] = 2 * n0
            for j in range(self.ncore):
                g[i, i, j, j] += 4 * n0
                g[i, j, j, i] -= 2 * n0
            g[i, i, act, act] += 2 * g1
            g[act, act, i, i] += 2 * g1
            g[act, i, i, act] -= g1
            g[i, act, act, i] -= g1.T
        d[act, act] = g1
        g[act, act, act, act] += g2
        return d, symmetrize_rdm2(g)

    def weighted_rdms(self, cs: numpy.ndarray, w: numpy.ndarray):
        """sum_rs W_rs of the roots' (transition) RDMs, over every MO."""
        g1 = numpy.zeros((self.ncas, self.ncas))
        g2 = numpy.zeros((self.ncas,) * 4)
        n0 = 0.0
        shape = (self.na, self.nb)
        for r in range(cs.shape[1]):
            for s in range(cs.shape[1]):
                if w[r, s] == 0:
                    continue
                t1, t2 = fci.direct_spin1.trans_rdm12(
                    cs[:, r].reshape(shape),
                    cs[:, s].reshape(shape),
                    self.ncas,
                    self.nelecas,
                )
                g1 += w[r, s] * t1
                g2 += w[r, s] * t2
                n0 += w[r, s] * (cs[:, r] @ cs[:, s])
        return self.full_rdms(0.5 * (g1 + g1.T), g2, n0)

    def sigma(self, ho: numpy.ndarray, go: numpy.ndarray, cs: numpy.ndarray):
        """H c for each column of cs, H over occupied-MO integrals, the core
        energy included (no V_nn)."""
        i, a = slice(0, self.ncore), slice(self.ncore, self.nocc)
        ecore = (
            2 * numpy.trace(ho[i, i])
            + 2 * numpy.einsum("iijj->", go[i, i, i, i])
            - numpy.einsum("ijji->", go[i, i, i, i])
        )
        heff = (
            ho[a, a]
            + 2 * numpy.einsum("tuii->tu", go[a, a, i, i])
            - numpy.einsum("tiiu->tu", go[a, i, i, a])
        )
        h2e = fci.direct_spin1.absorb_h1e(
            heff, go[a, a, a, a], self.ncas, self.nelecas, 0.5
        )
        shape = (self.na, self.nb)
        out = numpy.empty_like(cs)
        for r in range(cs.shape[1]):
            out[:, r] = (
                fci.direct_spin1.contract_2e(
                    h2e, cs[:, r].reshape(shape), self.ncas, self.nelecas
                ).ravel()
                + ecore * cs[:, r]
            )
        return out


class Point:
    """A reference (orbitals C, CI basis V, roots its first columns) at one
    geometry: the MO integrals in C's basis, the energy matrix and the exact
    gradient of any weighted sum of it at any lambda."""

    def __init__(self, model: Model, geom: Geometry, c: numpy.ndarray, v):
        self.model, self.geom, self.c, self.v = model, geom, c, v
        nmo, naux = c.shape[1], geom.naux
        self.hm = c.T @ geom.h @ c
        bmo = numpy.einsum(
            "mp,nq,mnP->pqP", c, c, geom.b.reshape(geom.nao, geom.nao, naux)
        ).reshape(nmo * nmo, naux)
        self.bmo = bmo
        self.gm = (bmo @ bmo.T).reshape((nmo,) * 4)
        self.ints = (self.hm, self.gm)
        self.cs = v[:, : model.nroots]

    def rotated(self, lam: numpy.ndarray):
        m = self.model
        return expm(m.kmat(lam)), self.v @ expm(m.amat(lam))

    def hmat(self, lam=None, ints=None, enuc=True) -> numpy.ndarray:
        """<c_r|H|c_s> over the roots at lam, + V_nn unless ints are given."""
        m = self.model
        hm, gm = self.ints if ints is None else ints
        if lam is None:
            u, y = numpy.eye(m.nmo), self.v
        else:
            u, y = self.rotated(lam)
        uo = u[:, : m.nocc]
        ho = uo.T @ hm @ uo
        go = numpy.einsum("pqrs,pa,qb,rc,sd->abcd", gm, uo, uo, uo, uo, optimize=True)
        cs = y[:, : m.nroots]
        h = cs.T @ m.sigma(ho, go, cs)
        h = 0.5 * (h + h.T)
        if ints is None and enuc:
            h = h + self.geom.enuc * numpy.eye(m.nroots)
        return h

    def energy(self, w: numpy.ndarray, lam=None, ints=None) -> float:
        return float(numpy.sum(w * self.hmat(lam, ints)))

    def gradient(self, w: numpy.ndarray, lam=None, ints=None) -> numpy.ndarray:
        """The exact d/dlambda of sum_rs W_rs H_rs at lam, every parameter:
        dE/dU and dE/dY pulled back through the exponentials' Frechet
        derivatives (the adjoint of L(K, .) is L(K^T, .))."""
        m = self.model
        hm, gm = self.ints if ints is None else ints
        lam = numpy.zeros(m.npar) if lam is None else lam
        kk, aa = m.kmat(lam), m.amat(lam)
        u, y = expm(kk), self.v @ expm(aa)
        uo = u[:, : m.nocc]
        g3 = numpy.einsum("pqrs,qb,rc,sd->pbcd", gm, uo, uo, uo, optimize=True)
        go = numpy.einsum("pa,pbcd->abcd", uo, g3)
        ho = uo.T @ hm @ uo
        cs = y[:, : m.nroots]
        d, g = m.weighted_rdms(cs, w)
        do, gocc = d[: m.nocc, : m.nocc], g[: m.nocc, : m.nocc, : m.nocc, : m.nocc]
        gu = numpy.zeros((m.nmo, m.nmo))
        gu[:, : m.nocc] = 2 * (
            hm @ uo @ do + numpy.einsum("pqrs,tqrs->pt", g3, gocc, optimize=True)
        )
        gk = expm_frechet(kk.T, gu, compute_expm=False)
        dy = numpy.zeros((m.ndet, m.nv))
        dy[:, : m.nroots] = 2 * m.sigma(ho, go, cs) @ w
        ga = expm_frechet(aa.T, self.v.T @ dy, compute_expm=False)
        out = numpy.empty(m.npar)
        out[: m.nk] = [gk[p, q] - gk[q, p] for p, q in m.pairs]
        out[m.nk : m.ntheta] = [ga[q, r] - ga[r, q] for r, q in m.cirot]
        out[m.ntheta :] = [ga[s, r] - ga[r, s] for r, s in m.thetas]
        return out

    def hvp(self, w, vec, ints=None, at=None, step=LAMBDA_STEP) -> numpy.ndarray:
        """d/de of the exact gradient at at + e vec, fourth order."""
        norm = numpy.linalg.norm(vec)
        if norm == 0:
            return numpy.zeros(self.model.npar)
        at = numpy.zeros(self.model.npar) if at is None else at
        h = step / norm
        return sum(c * self.gradient(w, at + s * h * vec, ints) for s, c in FOURTH) / h

    def hessian(self, w, idx, ints=None) -> numpy.ndarray:
        """The exact parameter Hessian over idx, by hvp, symmetrized (its
        asymmetry is round-off, asserted)."""
        out = numpy.zeros((len(idx), len(idx)))
        for j, k in enumerate(idx):
            e = numpy.zeros(self.model.npar)
            e[k] = 1.0
            out[:, j] = self.hvp(w, e, ints)[idx]
        asym = numpy.abs(out - out.T).max()
        assert asym < 1e-8 * max(1.0, numpy.abs(out).max()), ("hessian", asym)
        return 0.5 * (out + out.T)

    # --- derivative integrals in this basis ---------------------------------

    def derivatives(self):
        """Per coordinate: S^x and h^x in the MO basis, the fitted
        (pq|rs)^x, and I^x, the same plus the tau^x = -S^x/2 one-index
        transform (the x-derivative, at fixed parameters, of every MO
        integral the energy sees)."""
        geom, c = self.geom, self.c
        nmo, nao = c.shape[1], geom.nao

        def pair_mo(a):
            return numpy.einsum(
                "mp,nq,mnP->pqP", c, c, a.reshape(nao, nao, -1), optimize=True
            ).reshape(nmo * nmo, -1)

        self.sx = numpy.array([c.T @ s @ c for s in geom.s1])
        self.hx = numpy.array([c.T @ h @ c for h in geom.h1])
        self.tau = -0.5 * self.sx
        self.gx, self.ix = [], []
        for x in range(geom.n):
            ymo = pair_mo(geom.y1[x])
            g = ymo @ self.bmo.T
            g = g + g.T - self.bmo @ geom.z1[x] @ self.bmo.T
            g = g.reshape((nmo,) * 4)
            self.gx.append(g)
            ht, gt = one_index(self.hm, self.gm, self.tau[x])
            self.ix.append((self.hx[x] + ht, g + gt))

    def root_gradient(self, root: int, w_sa, idx, a_sa=None):
        """The relaxed energy's nuclear gradient for one root, the Lagrangian
        E_I + z.g_SA with A z = -dE_I/dlambda over idx; and z."""
        m = self.model
        e_root = numpy.zeros((m.nroots, m.nroots))
        e_root[root, root] = 1.0
        b = self.gradient(e_root)[idx]
        z = numpy.zeros(m.npar)
        if m.nroots > 1:
            if a_sa is None:
                a_sa = self.hessian(w_sa, idx)
            z[idx] = -numpy.linalg.solve(a_sa, b)
        out = numpy.empty(self.geom.n)
        for x in range(self.geom.n):
            out[x] = self.hmat(ints=self.ix[x])[root, root] + self.geom.enuc1[x]
            if m.nroots > 1:
                out[x] += z @ self.gradient(w_sa, ints=self.ix[x])
        return out, z


# --- the solution, and re-solving it at displaced geometries ----------------


def start(case: dict, model: Model, geom: Geometry):
    """PySCF's DF-(SA-)CASSCF over the same fitting set: the starting point,
    and its energy."""
    mol = geom.mol
    aux = pyscf_basis(AUX_FILES[AUX_BASIS], {s for s, _ in WATER_STRETCHED})
    mf = scf.RHF(mol).density_fit(auxbasis=aux).run(conv_tol=1e-12)
    mc = mcscf.DFCASSCF(mf, NCAS, NELECAS, auxbasis=aux)
    if model.nroots > 1:
        mc.fix_spin_(ss=0)
        mc = mc.state_average_(list(model.weights))
    mc.conv_tol, mc.conv_tol_grad, mc.max_cycle_macro = 1e-11, 1e-6, 300
    mc.kernel()
    assert mc.converged, case["name"]
    cis = mc.ci if model.nroots > 1 else [mc.ci]
    for ci in cis:
        assert numpy.abs(ci - ci.T).max() < 1e-4, "roots must be M_s = 0 singlets"
    e = mc.e_average if model.nroots > 1 else mc.e_tot
    return mc.mo_coeff, model.basis(cis), float(e)


def newton(model: Model, geom, c, v, hessian, tolerance):
    """Newton steps on E_SA over the parameters, `hessian` (over model.p)
    fixed; then the roots diagonalized in their span (at equal weights the
    rotations among them are not parameters) and each kept in sign. The
    re-centred Point, at lambda = 0 a solution."""
    w = numpy.diag(model.weights)
    c = c @ inv_sqrt(c.T @ geom.s @ c)
    for _ in range(60):
        point = Point(model, geom, c, v)
        g = point.gradient(w)[model.p]
        if numpy.abs(g).max() < tolerance:
            break
        lam = numpy.zeros(model.npar)
        lam[model.p] = -numpy.linalg.solve(hessian, g)
        u, v = point.rotated(lam)
        c = c @ u
    else:
        raise RuntimeError(f"Newton stalled at |g| {numpy.abs(g).max():.1e}")
    h = point.hmat()
    if model.nroots > 1:
        _, rot = numpy.linalg.eigh(h)
        rot = rot * numpy.sign(numpy.diag(rot))
        v = v.copy()
        v[:, : model.nroots] = v[:, : model.nroots] @ rot
        point = Point(model, geom, c, v)
        h = point.hmat()
    off = numpy.abs(h - numpy.diag(numpy.diag(h))).max()
    assert off < 1e-9, ("roots not diagonal", off)
    return point


def minimum(model: Model, geom, c, v) -> tuple[Point, dict]:
    """From PySCF's solution to a minimum of E_SA: saddle-free Newton steps
    (the Hessian's eigenvalues by magnitude, a step at most MAX_STEP), and at
    a saddle a step of SADDLE_STEP down its negative mode (the module
    docstring has which of PySCF's solutions are saddles)."""
    w = numpy.diag(model.weights)
    c = c @ inv_sqrt(c.T @ geom.s @ c)
    info: dict = {}
    for _ in range(200):
        point = Point(model, geom, c, v)
        g = point.gradient(w)[model.p]
        e, u = numpy.linalg.eigh(point.hessian(w, model.p))
        if not info:
            info = {
                "pyscf_least_eigenvalue": float(e[0]),
                "pyscf_e_sa": point.energy(w),
            }
        if e[0] > 0 and numpy.abs(g).max() < 1e-8:
            return point, info
        if e[0] <= 0 and numpy.abs(g).max() < 1e-6:
            step = SADDLE_STEP * u[:, 0]
        else:
            step = -u @ ((u.T @ g) / numpy.abs(e))
            step *= min(1.0, MAX_STEP / numpy.linalg.norm(step))
        lam = numpy.zeros(model.npar)
        lam[model.p] = step
        rot, v = point.rotated(lam)
        c = c @ rot
    raise RuntimeError("no minimum of E_SA")


def relaxed(model: Model, ref: Point, coords, hessian, order=1) -> Point:
    """The solution at coords, from ref's orbitals (orthonormalized in the
    displaced metric) and CI; each root checked to be the same root."""
    geom = Geometry(ref.geom.case, coords, order)
    point = newton(model, geom, ref.c, ref.v, hessian, RELAX_GRADIENT)
    overlap = numpy.abs(numpy.diag(point.cs.T @ ref.cs))
    assert overlap.min() > 0.9, ("a root changed", overlap)
    for r in range(model.nroots):
        if point.cs[:, r] @ ref.cs[:, r] < 0:
            point.v[:, r] *= -1
    point.cs = point.v[:, : model.nroots]
    return point


# --- E_xy, eq. (exy), at any densities -------------------------------------


def fock(hm, gm, d, g) -> numpy.ndarray:
    """eq. (fock): F_pq = sum_r h_pr D_rq + sum_rst (pr|st) Gamma_qrst."""
    return hm @ d + numpy.einsum("prst,qrst->pq", gm, g, optimize=True)


def e2_matrix(point: Point, d, g) -> numpy.ndarray:
    """E^(2)[tau^x, tau^y] for every pair: the energy at the integrals
    one-index transformed by tau^x on one index and tau^y on another, every
    ordered pair of positions. With Gamma and (pq|rs) both 8-fold
    symmetric the twelve orderings are four of one shape and eight of
    another."""
    hm, gm, tau = point.hm, point.gm, point.tau
    n = len(tau)
    ga = [numpy.einsum("ap,pqrs->aqrs", t, g, optimize=True) for t in tau]
    gb = [numpy.einsum("bq,abrs->aqrs", t, gm, optimize=True) for t in tau]
    gc = [numpy.einsum("cr,aqcs->aqrs", t, gm, optimize=True) for t in tau]
    out = numpy.empty((n, n))
    for x in range(n):
        for y in range(n):
            out[x, y] = 2 * numpy.sum(d * (tau[x].T @ hm @ tau[y])) + 0.5 * (
                4 * numpy.sum(ga[x] * gb[y]) + 8 * numpy.sum(ga[x] * gc[y])
            )
    return out


def two_index(hm, gm, a, b) -> tuple[numpy.ndarray, numpy.ndarray]:
    """check_df.py's doubly one-index transform, every ordered pair of
    positions spelled out: the brute-force form e2_matrix is checked
    against."""
    h2 = a.T @ hm @ b + b.T @ hm @ a
    g = numpy.zeros_like(gm)
    for pa in range(4):
        for pb in range(4):
            if pa == pb:
                continue
            t = gm
            for pos, mat in ((pa, a), (pb, b)):
                idx = "pqrs"
                src = idx[:pos] + "z" + idx[pos + 1 :]
                t = numpy.einsum(f"z{idx[pos]},{src}->{idx}", mat, t)
            g = g + t
    return h2, g


def exy_terms(point: Point, d, g, split: bool) -> dict:
    """eq. (exy) at densities (d, g) over every MO, by piece (no V_nn):
    the explicit one-electron sum D^AO h^xy; the explicit two-electron eq.
    (df2), its three terms, and at a root's densities its cross term by the
    J, K and active pieces of section 6; and the three connection terms."""
    geom, c, m = point.geom, point.c, point.model
    n, nao = geom.n, geom.nao
    out = {}
    dao = c @ d @ c.T
    gao = numpy.einsum("pqrs,mp,nq,lr,ts->mnlt", g, c, c, c, c, optimize=True).reshape(
        nao * nao, nao * nao
    )
    out["one_electron"] = numpy.einsum("xymn,mn->xy", geom.h2, dao)
    # c = T N, through the Cholesky factor.
    cfit = scipy.linalg.cho_solve((geom.chol, True), geom.t.T).T
    gt = gao @ cfit
    what = cfit.T @ gt
    out["df2_three_center"] = numpy.einsum("xyaP,aP->xy", geom.t2, gt)
    out["df2_metric"] = -0.5 * numpy.einsum("xyPQ,PQ->xy", geom.m2, what)
    gr = numpy.array([gao @ r for r in geom.r1])
    cross = numpy.einsum("xaP,yaP->xy", geom.r1, gr)
    out["df2_cross"] = cross
    if split:
        core, act = m.core, m.act
        co, cc, ca = c[:, : m.nocc], c[:, core], c[:, act]
        g1, g2 = d[act, act], g[act, act, act, act]
        dao_a = ca @ g1 @ ca.T
        gk = d[: m.nocc, : m.nocc].copy()
        gk[core, core] = numpy.eye(m.ncore)
        u = numpy.array([dao.ravel() @ r for r in geom.r1])
        va = numpy.array([dao_a.ravel() @ r for r in geom.r1])
        r3 = [r.reshape(nao, nao, -1) for r in geom.r1]
        rk = [numpy.einsum("mi,np,mnP->ipP", cc, co, r, optimize=True) for r in r3]
        ra = [numpy.einsum("mt,nu,mnP->tuP", ca, ca, r, optimize=True) for r in r3]
        out["df2_cross_j"] = u @ u.T - va @ va.T
        out["df2_cross_k"] = numpy.array(
            [
                [-2 * numpy.einsum("pq,ipP,iqP->", gk, rk[x], rk[y]) for y in range(n)]
                for x in range(n)
            ]
        )
        out["df2_cross_active"] = numpy.array(
            [
                [
                    numpy.einsum("tuvw,tuP,vwP->", g2, ra[x], ra[y], optimize=True)
                    for y in range(n)
                ]
                for x in range(n)
            ]
        )
        split_sum = out["df2_cross_j"] + out["df2_cross_k"] + out["df2_cross_active"]
        out["check_jk_split"] = float(numpy.abs(split_sum - cross).max())
        del out["df2_cross"]
    out["two_electron"] = (
        out["df2_three_center"]
        + out["df2_metric"]
        + (cross if not split else split_sum)
    )
    sx = point.sx
    s2 = numpy.einsum("mp,xymn,nq->xypq", c, geom.s2, c, optimize=True)
    f0 = fock(point.hm, point.gm, d, g)
    fx = [fock(point.hx[x], point.gx[x], d, g) for x in range(n)]
    out["connection_second"] = numpy.array(
        [
            [
                2
                * numpy.sum(
                    f0 * (-0.5 * s2[x, y] + 3 / 8 * (sx[x] @ sx[y] + sx[y] @ sx[x]))
                )
                for y in range(n)
            ]
            for x in range(n)
        ]
    )
    tau = point.tau
    ft = numpy.array([[numpy.sum(fx[x] * tau[y]) for y in range(n)] for x in range(n)])
    out["connection_fock"] = 2 * (ft + ft.T)
    out["connection_twice"] = e2_matrix(point, d, g)
    out["exy"] = (
        out["one_electron"]
        + out["two_electron"]
        + out["connection_second"]
        + out["connection_fock"]
        + out["connection_twice"]
    )
    out["fock"] = f0
    out["fock_x"] = numpy.array(fx)
    return out


def rotated_densities(point: Point, w, lam) -> tuple[numpy.ndarray, numpy.ndarray]:
    """sum W_rs of the (transition) RDMs at lam, over every MO in the
    unrotated basis: the rotation moved onto the densities, U D U^T."""
    m = point.model
    u, y = point.rotated(lam)
    d, g = m.weighted_rdms(y[:, : m.nroots], w)
    d = u @ d @ u.T
    g = numpy.einsum("pqrs,ap,bq,cr,ds->abcd", g, u, u, u, u, optimize=True)
    return d, g


def z_densities(point: Point, w, z) -> tuple[numpy.ndarray, numpy.ndarray]:
    """d/de of rotated_densities at e z: the orbital part K D - D K (and K
    on every index of Gamma), the CI part the transition RDMs of the roots'
    first-order change with the roots."""
    m = point.model
    k = m.kmat(z)
    d0, g0 = m.weighted_rdms(point.cs, w)
    d = k @ d0 - d0 @ k
    g = numpy.einsum("ap,pqrs->aqrs", k, g0)
    g = g + g.transpose(1, 0, 2, 3)
    g = g + g.transpose(2, 3, 0, 1)
    dc = point.v @ m.amat(z)
    shape = (m.na, m.nb)
    g1 = numpy.zeros((m.ncas, m.ncas))
    g2 = numpy.zeros((m.ncas,) * 4)
    n0 = 0.0
    for r in range(m.nroots):
        if w[r, r] == 0:
            continue
        for bra, ket in ((dc[:, r], point.cs[:, r]), (point.cs[:, r], dc[:, r])):
            t1, t2 = fci.direct_spin1.trans_rdm12(
                bra.reshape(shape), ket.reshape(shape), m.ncas, m.nelecas
            )
            g1 += w[r, r] * t1
            g2 += w[r, r] * t2
            n0 += w[r, r] * (bra @ ket)
    dci, gci = m.full_rdms(0.5 * (g1 + g1.T), g2, n0)
    return d + dci, symmetrize_rdm2(g) + gci


# --- output layouts -----------------------------------------------------------


def vector_out(model: Model, point: Point, lam) -> dict:
    """A parameter-space vector (lambda^x, z): K, with C -> C e^K; each
    root's first-order change projected off every root; and at unequal
    weights Theta, root r's change along root s (dc_r = sum_s Theta_sr c_s)."""
    a = model.amat(lam)
    out = {
        "orb": model.kmat(lam),
        "ci": point.v[:, model.nroots :] @ a[model.nroots :, : model.nroots],
    }
    if not model.equal:
        out["theta"] = a[: model.nroots, : model.nroots]
    return out


def covector_out(model: Model, point: Point, g) -> dict:
    """A parameter-space covector (a gradient, a right-hand side): G_pq =
    dE/dkappa_pq for p > q, G_qp = -G_pq; per root the CI-space gradient
    projected off every root; and at unequal weights gamma_sr = dE/dTheta_sr,
    antisymmetric. Its pairing with a vector is 1/2 G.K + sum_r ci_r.dc_r +
    1/2 gamma.Theta."""
    gk = numpy.zeros((model.nmo, model.nmo))
    for (p, q), x in zip(model.pairs, g[: model.nk], strict=True):
        gk[p, q], gk[q, p] = x, -x
    ga = numpy.zeros((model.nv, model.nroots))
    for (r, q), x in zip(model.cirot, g[model.nk : model.ntheta], strict=True):
        ga[q, r] = x
    out = {"orb": gk, "ci": point.v @ ga}
    if not model.equal:
        th = numpy.zeros((model.nroots, model.nroots))
        for (r, s), x in zip(model.thetas, g[model.ntheta :], strict=True):
            th[s, r], th[r, s] = x, -x
        out["theta"] = th
    return out


def pair(cov: dict, vec: dict) -> float:
    total = 0.5 * numpy.sum(cov["orb"] * vec["orb"]) + numpy.sum(cov["ci"] * vec["ci"])
    if "theta" in cov:
        total += 0.5 * numpy.sum(cov["theta"] * vec["theta"])
    return float(total)


def stack(items: list[dict]) -> dict:
    """Per-coordinate outputs stacked on a last axis."""
    return {k: numpy.stack([it[k] for it in items], axis=-1) for k in items[0]}


# --- one case --------------------------------------------------------------


def worst(a, b) -> float:
    return float(numpy.abs(numpy.asarray(a) - numpy.asarray(b)).max())


def fixed_parameter_fd(model, ref: Point, ws: list, lam_r, z_list) -> dict:
    """Fourth-order differences in x at fixed parameters (lambda = 0, and
    lam_r), of: each weighted energy in ws, E_SA at lam_r, 1/2 Gamma^AO.(T N
    T^T) at root 0's fixed Gamma^AO; and first differences of the exact
    gradient (each ws) and of the Hessian-vector products along z_list."""
    case, x0, n = ref.geom.case, ref.geom.coords, ref.geom.n
    h = 5 * X_STEP
    d0, g0 = model.weighted_rdms(ref.cs, ws[0])
    c = ref.c
    gao = numpy.einsum("pqrs,mp,nq,lr,ts->mnlt", g0, c, c, c, c, optimize=True).reshape(
        ref.geom.nao**2, -1
    )
    w_sa = numpy.diag(model.weights)
    cache = {}

    def values(steps):
        key = tuple(sorted(steps))
        if key not in cache:
            geom = Geometry(case, displaced(x0, *steps))
            pt = Point(model, geom, c @ inv_sqrt(c.T @ geom.s @ c), ref.v)
            e = [pt.energy(w) for w in ws]
            cache[key] = numpy.array(
                [*e, pt.energy(w_sa, lam_r), 0.5 * numpy.sum(gao * geom.eri())]
            )
        return cache[key]

    nv = len(ws) + 2
    second = numpy.zeros((n, n, nv))
    for k in range(n):
        for j in range(k + 1):
            if k == j:
                acc = -30 * values([])
                for s, cf in ((1, 16), (2, -1)):
                    acc = acc + cf * (values([(k, s * h)]) + values([(k, -s * h)]))
                second[k, k] = acc / (12 * h * h)
            else:
                acc = 0
                for sk, ck in FOURTH:
                    for sj, cj in FOURTH:
                        acc = acc + ck * cj * values([(k, sk * h), (j, sj * h)])
                second[k, j] = second[j, k] = acc / (h * h)
    h = X_STEP
    grads = numpy.zeros((n, len(ws), model.npar))
    hvps = numpy.zeros((n, len(z_list), model.npar))
    for k in range(n):
        for s, cf in FOURTH:
            geom = Geometry(case, displaced(x0, (k, s * h)))
            pt = Point(model, geom, c @ inv_sqrt(c.T @ geom.s @ c), ref.v)
            for j, w in enumerate(ws):
                grads[k, j] += cf * pt.gradient(w) / h
            for j, z in enumerate(z_list):
                hvps[k, j] += cf * pt.hvp(w_sa, z) / h
    return {"second": second, "grads": grads, "hvps": hvps}


def third_action(point: Point, w, z, cols: numpy.ndarray, step) -> numpy.ndarray:
    """cols^T (z . d2 g_SA / dlambda dlambda) cols: the mixed fourth-order
    difference, in (eta, e), of cols^T g(eta col_y + e z)."""
    n = cols.shape[1]
    out = numpy.zeros((n, n))
    hz = step / numpy.linalg.norm(z)
    for y in range(n):
        hv = step / numpy.linalg.norm(cols[:, y])
        acc = numpy.zeros(point.model.npar)
        for s1, c1 in FOURTH:
            for s2, c2 in FOURTH:
                acc += c1 * c2 * point.gradient(w, s1 * hv * cols[:, y] + s2 * hz * z)
        out[:, y] = cols.T @ acc / (hv * hz)
    return out


def sum_rule(h: numpy.ndarray) -> float:
    n = h.shape[0]
    return float(numpy.abs(h.reshape(n, n // 3, 3).sum(axis=1)).max())


def case_reference(case: dict) -> tuple[dict, dict, list]:
    t0 = time.time()
    name = case["name"]
    print(f"{name}: weights {case['weights']}")
    x0 = numpy.array([coord for _, coord in WATER_STRETCHED])
    geom = Geometry(case, x0, 2)
    n = geom.n
    checks: dict = {}
    check_integrals(case, geom, checks)
    model = Model(case, geom.nao, geom.mol.nelectron)
    nroots, idx = model.nroots, model.p
    w_sa = numpy.diag(model.weights)
    e_root = []
    for r in range(nroots):
        e = numpy.zeros((nroots, nroots))
        e[r, r] = 1.0
        e_root.append(e)

    # The solution: PySCF's, polished on this energy.
    c, v, e_pyscf = start(case, model, geom)
    p0 = Point(model, geom, c @ inv_sqrt(c.T @ geom.s @ c), v)
    checks["energy_vs_pyscf"] = abs(p0.energy(w_sa) - e_pyscf)
    print(f"  E_SA here vs PySCF's DF-CASSCF {checks['energy_vs_pyscf']:.1e}")
    assert checks["energy_vs_pyscf"] < 1e-8
    low, descent = minimum(model, geom, c, v)
    ref = newton(model, geom, low.c, low.v, low.hessian(w_sa, idx), POLISH_GRADIENT)
    a_sa = ref.hessian(w_sa, idx)
    ref = newton(model, geom, ref.c, ref.v, a_sa, POLISH_GRADIENT)
    checks["polish_gradient"] = float(numpy.abs(ref.gradient(w_sa)[idx]).max())
    curvature = float(numpy.linalg.eigvalsh(a_sa)[0])
    assert curvature > 1e-3, (name, "the SA Hessian's least eigenvalue", curvature)
    energies = numpy.diag(ref.hmat())
    descent["e_sa_drop"] = descent["pyscf_e_sa"] - ref.energy(w_sa)
    descent["least_eigenvalue"] = curvature
    if descent["pyscf_least_eigenvalue"] < 0:
        print(
            f"  PySCF's solution is a saddle (A's least eigenvalue "
            f"{descent['pyscf_least_eigenvalue']:.1e}); the minimum is "
            f"{descent['e_sa_drop']:.1e} below it"
        )
    energies = numpy.diag(ref.hmat())
    print(f"  roots {energies}, |g_SA| {checks['polish_gradient']:.1e}")
    ref.derivatives()

    # lambda^x, the SA parameters' response: A lambda^x = -d_x g_SA.
    g_sa_x = numpy.array([ref.gradient(w_sa, ints=ref.ix[x]) for x in range(n)])
    lam = numpy.zeros((model.npar, n))
    lam[idx] = -numpy.linalg.solve(a_sa, g_sa_x[:, idx].T)

    rng = numpy.random.default_rng(7)
    lam_r = numpy.zeros(model.npar)
    lam_r[idx] = 0.05 * rng.standard_normal(len(idx))

    roots, root_arrays = [], []
    z_list = []
    for r in range(nroots):
        _, z = ref.root_gradient(r, w_sa, idx, a_sa)
        z_list.append(z)
    fixed = fixed_parameter_fd(model, ref, e_root, lam_r, z_list)

    # The generic densities: E_xy at a random rotation, and the z densities'
    # linearization.
    d_r, g_r = rotated_densities(ref, w_sa, lam_r)
    generic = exy_terms(ref, d_r, g_r, split=False)["exy"] + geom.enuc2
    checks["exy_generic_vs_fd"] = worst(generic, fixed["second"][:, :, nroots])
    print(
        "  E_xy at a random rotation vs fixed-parameter FD "
        f"{checks['exy_generic_vs_fd']:.1e}"
    )
    assert checks["exy_generic_vs_fd"] < TOL_FIXED

    for r in range(nroots):
        t1 = time.time()
        arrays: dict = {}
        info: dict = {"root": r, "energy": float(energies[r])}
        d, g = model.weighted_rdms(ref.cs, e_root[r])
        terms = exy_terms(ref, d, g, split=True)
        checks.setdefault("jk_split", 0.0)
        checks["jk_split"] = max(checks["jk_split"], terms.pop("check_jk_split"))
        assert checks["jk_split"] < TOL_EXACT, (
            name,
            "J + K + active",
            checks["jk_split"],
        )
        if r == 0:
            # The fast E^(2) against the brute-force transform, one pair.
            h2, g2 = two_index(ref.hm, ref.gm, ref.tau[0], ref.tau[4])
            brute = numpy.sum(d * h2) + 0.5 * numpy.sum(g * g2)
            checks["e2_vs_brute_force"] = abs(brute - terms["connection_twice"][0, 4])
            assert checks["e2_vs_brute_force"] < TOL_EXACT
            # check_df's A: (df2) vs the difference of 1/2 Gamma.(T N T^T).
            ref_e2 = fixed["second"][:, :, nroots + 1]
            checks["df2_vs_fd"] = worst(terms["two_electron"], ref_e2)
            checks["df2_without_cross_vs_fd"] = worst(
                terms["df2_three_center"] + terms["df2_metric"], ref_e2
            )
            print(
                f"  A (df2) vs FD {checks['df2_vs_fd']:.1e}, without the cross "
                f"term {checks['df2_without_cross_vs_fd']:.1e}; B J + K + active "
                f"vs generic {checks['jk_split']:.1e}"
            )
            assert checks["df2_vs_fd"] < TOL_FIXED
            assert checks["df2_without_cross_vs_fd"] > MUST_MISS
        exy = terms["exy"] + geom.enuc2
        terms["nuclear_repulsion"] = geom.enuc2
        terms["exy"] = exy
        err = worst(exy, fixed["second"][:, :, r])
        key = "exy_vs_fd"
        checks[key] = max(checks.get(key, 0.0), err)
        assert err < TOL_FIXED, (name, r, key, err)
        for k in (
            "one_electron",
            "nuclear_repulsion",
            "df2_three_center",
            "df2_metric",
            "df2_cross_j",
            "df2_cross_k",
            "df2_cross_active",
            "two_electron",
            "connection_second",
            "connection_fock",
            "connection_twice",
            "exy",
            "fock",
            "fock_x",
        ):
            arrays[k] = terms[k]

        # Root I's CP right-hand side, and its check against the gradient's
        # difference in x.
        rhs_root = numpy.array(
            [ref.gradient(e_root[r], ints=ref.ix[x]) for x in range(n)]
        )
        err = worst(rhs_root, fixed["grads"][:, r])
        checks["rhs_vs_fd"] = max(checks.get("rhs_vs_fd", 0.0), err)
        assert err < TOL_FIXED, (name, r, "rhs", err)
        z = z_list[r]
        lxl = rhs_root.copy()
        arrays["root_hessian_lambda"] = numpy.array(
            [
                lam[:, x] @ ref.hvp(e_root[r], lam[:, y])
                for x in range(n)
                for y in range(n)
            ]
        ).reshape(n, n)
        hessian = exy.copy()
        if nroots > 1:
            # L_x lambda's z part: the SA Hessian at I^x along z.
            hz = numpy.array([ref.hvp(w_sa, z, ints=ref.ix[x]) for x in range(n)])
            err = worst(hz[:, idx], fixed["hvps"][:, r][:, idx])
            checks["lxl_z_vs_fd"] = max(checks.get("lxl_z_vs_fd", 0.0), err)
            assert err < TOL_FIXED, (name, r, "lxl z", err)
            lxl = lxl + hz
            # L_xy's z part: eq. (exy) at z's densities.
            dz, gz = z_densities(ref, w_sa, z)
            fd_d = [
                sum(
                    cf * rotated_densities(ref, w_sa, s * 1e-3 * z)[j]
                    for s, cf in FOURTH
                )
                / 1e-3
                for j in range(2)
            ]
            err = max(worst(dz, fd_d[0]), worst(gz, fd_d[1]))
            checks["z_densities_vs_fd"] = max(checks.get("z_densities_vs_fd", 0.0), err)
            assert err < TOL_FIXED, (name, r, "z densities", err)
            zterms = exy_terms(ref, dz, gz, split=False)
            for k in (
                "one_electron",
                "two_electron",
                "connection_second",
                "connection_fock",
                "connection_twice",
            ):
                arrays[f"z_{k}"] = zterms[k]
            arrays["exy_z"] = zterms["exy"]
            hessian = hessian + zterms["exy"]
            # The third-derivative action, and its step against half the step.
            third = third_action(ref, w_sa, z, lam, THIRD_STEP)
            half = third_action(ref, w_sa, z, lam, THIRD_STEP / 2)
            err = max(worst(third, half), worst(third, third.T))
            checks["third_step_and_symmetry"] = max(
                checks.get("third_step_and_symmetry", 0.0), err
            )
            assert err < TOL_THIRD, (name, r, "third", err)
            arrays["third"] = third
            u = numpy.zeros((model.npar, 2))
            u[idx] = rng.standard_normal((len(idx), 2))
            info["third_random"] = float(
                third_action(ref, w_sa, z, u, THIRD_STEP)[0, 1]
            )
            for k, val in vector_out(model, ref, u[:, 0]).items():
                arrays[f"random_{k}"] = numpy.stack(
                    [val, vector_out(model, ref, u[:, 1])[k]], axis=-1
                )
            hessian = hessian + third
            for k, val in vector_out(model, ref, z).items():
                arrays[f"z_{k}"] = val
            for k, val in stack(
                [covector_out(model, ref, g) for g in rhs_root]
            ).items():
                arrays[f"root_rhs_{k}"] = val
            for k, val in stack([covector_out(model, ref, g) for g in hz]).items():
                arrays[f"lxl_z_{k}"] = val
        arrays["lxl_lambda"] = lxl @ lam
        # The output layouts pair as the parameters do.
        cov = covector_out(model, ref, lxl[0])
        vec = vector_out(model, ref, lam[:, 1])
        assert abs(pair(cov, vec) - arrays["lxl_lambda"][0, 1]) < TOL_EXACT
        hessian = (
            hessian
            + arrays["lxl_lambda"]
            + arrays["lxl_lambda"].T
            + arrays["root_hessian_lambda"]
        )
        if nroots == 1:
            arrays["response"] = hessian - exy
        if nroots > 1 and model.equal:
            # eq. (rot): the orbital-relaxed couplings to the other roots.
            coupling = numpy.zeros((nroots, n))
            for s in range(nroots):
                if s == r:
                    continue
                w_rs = numpy.zeros((nroots, nroots))
                w_rs[r, s] = w_rs[s, r] = 0.5
                for x in range(n):
                    coupling[s, x] = (
                        ref.hmat(ints=ref.ix[x])[r, s] + ref.gradient(w_rs) @ lam[:, x]
                    )
            rot = sum(
                2 * numpy.outer(coupling[s], coupling[s]) / (energies[r] - energies[s])
                for s in range(nroots)
                if s != r
            )
            arrays["coupling"] = coupling
            arrays["rot"] = rot
            hessian = hessian + rot
        arrays["hessian"] = hessian
        info["hessian_asymmetry"] = worst(hessian, hessian.T)
        info["hessian_sum_rule"] = sum_rule(hessian)
        assert info["hessian_asymmetry"] < 1e-8, (name, r, info["hessian_asymmetry"])
        assert info["hessian_sum_rule"] < 1e-7, (name, r, info["hessian_sum_rule"])
        root_arrays.append(arrays)
        roots.append(info)
        print(f"  root {r}: analytic pieces in {time.time() - t1:.0f} s")

    # The end-to-end reference: the relaxed gradient differenced.
    t1 = time.time()
    h = X_STEP
    grads = {}
    energies_at = {}
    for k in range(n):
        for s in (-2, -1, 1, 2):
            pt = relaxed(model, ref, displaced(x0, (k, s * h)), a_sa, order=1)
            pt.derivatives()
            a_pt = pt.hessian(w_sa, idx) if nroots > 1 else None
            grads[k, s] = [
                pt.root_gradient(r, w_sa, idx, a_pt)[0] for r in range(nroots)
            ]
            energies_at[k, s] = numpy.diag(pt.hmat())
    dirs = [d / numpy.linalg.norm(d) for d in rng.standard_normal((3, n))]
    dir_e = [
        [
            numpy.diag(
                relaxed(model, ref, x0 + s * h * d.reshape(x0.shape), a_sa).hmat()
            )
            for s in (1, -1)
        ]
        for d in dirs
    ]
    print(f"  {4 * n + 6} re-relaxations in {time.time() - t1:.0f} s")
    for r in range(nroots):
        arrays, info = root_arrays[r], roots[r]
        gradient, z = ref.root_gradient(r, w_sa, idx, a_sa)
        fd_grad = numpy.array(
            [sum(cf * energies_at[k, s][r] for s, cf in FOURTH) / h for k in range(n)]
        )
        info["gradient"] = [float(x) for x in gradient]
        info["gradient_vs_fd"] = worst(gradient, fd_grad)
        assert info["gradient_vs_fd"] < 1e-8, (name, r, info["gradient_vs_fd"])
        if not model.equal and nroots > 1:
            # check_sa's gradient row: the rotations among the roots projected
            # out of the Z-vector, as the tree and PySCF do.
            proj = numpy.arange(model.ntheta)
            a_proj = ref.hessian(w_sa, proj)
            proj_grad, _ = ref.root_gradient(r, w_sa, proj, a_proj)
            info["gradient_projected_vs_fd"] = worst(proj_grad, fd_grad)
        fd = numpy.zeros((n, n))
        for k in range(n):
            fd[:, k] = sum(cf * grads[k, s][r] for s, cf in FOURTH) / h
        arrays["relaxed_fd"] = fd
        info["hessian_vs_relaxed_fd"] = worst(arrays["hessian"], fd)
        diag = numpy.array(
            [
                (energies_at[k, 1][r] - 2 * energies[r] + energies_at[k, -1][r]) / h**2
                for k in range(n)
            ]
        )
        along = numpy.array(
            [(ep[r] - 2 * energies[r] + em[r]) / h**2 for ep, em in dir_e]
        )

        def energy_misses(hm, diag=diag, along=along):
            return max(
                worst(numpy.diag(hm), diag),
                max(abs(d @ hm @ d - a) for d, a in zip(dirs, along, strict=True)),
            )

        info["hessian_vs_relaxed_energies"] = float(energy_misses(arrays["hessian"]))
        if "rot" in arrays:
            info["without_rot_vs_relaxed_energies"] = float(
                energy_misses(arrays["hessian"] - arrays["rot"])
            )
            info["one_copy_of_rot_vs_relaxed_energies"] = float(
                energy_misses(arrays["hessian"] - arrays["rot"] / 2)
            )
            assert info["without_rot_vs_relaxed_energies"] > MUST_MISS
            assert info["one_copy_of_rot_vs_relaxed_energies"] > MUST_MISS
        print(
            f"  root {r}: max |H| {numpy.abs(fd).max():.3f}; hessian vs relaxed_fd "
            f"{info['hessian_vs_relaxed_fd']:.1e}, vs relaxed energies "
            f"{info['hessian_vs_relaxed_energies']:.1e}; gradient vs FD "
            f"{info['gradient_vs_fd']:.1e}"
            + (
                f", projected {info['gradient_projected_vs_fd']:.1e}"
                if "gradient_projected_vs_fd" in info
                else ""
            )
            + (
                f"; without (rot) {info['without_rot_vs_relaxed_energies']:.1e}, "
                f"one copy {info['one_copy_of_rot_vs_relaxed_energies']:.1e}"
                if "rot" in arrays
                else ""
            )
        )
        assert info["hessian_vs_relaxed_fd"] < TOL_RELAXED_FD, (name, r)
        assert info["hessian_vs_relaxed_energies"] < TOL_RELAXED_ENERGY, (name, r)

    case_arrays = {"c": ref.c, "ci": ref.cs}
    for k, val in stack([covector_out(model, ref, g) for g in g_sa_x]).items():
        case_arrays[f"rhs_{k}"] = val
    for k, val in stack([vector_out(model, ref, lam[:, x]) for x in range(n)]).items():
        case_arrays[f"lambda_{k}"] = val
    print("  checks: " + ", ".join(f"{k} {v:.1e}" for k, v in checks.items()))
    print(f"  {name} done in {time.time() - t0:.0f} s")
    info = {
        "name": name,
        "basis": case["basis"],
        "aux_basis": AUX_BASIS,
        "atoms": [
            {
                "symbol": s,
                "charge": int(geom.mol.atom_charge(a)),
                "coord": list(map(float, x0[a])),
            }
            for a, (s, _) in enumerate(WATER_STRETCHED)
        ],
        "nao": geom.nao,
        "naux": geom.naux,
        "nmo": model.nmo,
        "ncore": model.ncore,
        "ncas": model.ncas,
        "nelecas": list(model.nelecas),
        "na": model.na,
        "nb": model.nb,
        "weights": [float(w) for w in model.weights],
        "equal_weights": model.equal,
        "energies": [float(e) for e in energies],
        "e_sa": float(model.weights @ energies),
        "descent": descent,
        "checks": checks,
        "roots": roots,
    }
    return info, case_arrays, root_arrays


def save(name: str, array: numpy.ndarray) -> str:
    """Write one '<f8' Fortran-order .npy; its file name."""
    path = OUT / f"{name}.npy"
    numpy.save(path, numpy.asfortranarray(array, dtype="<f8"), allow_pickle=False)
    return path.name


def main() -> None:
    lib.num_threads(1)
    names = sys.argv[1:]
    cases = [c for c in CASES if not names or c["name"] in names]
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    for case in cases:
        info, case_arrays, root_arrays = case_reference(case)
        out.append((info, case_arrays, root_arrays))
    if names:
        print("a subset of the cases: nothing written")
        return
    for stale in OUT.glob("*.npy"):
        stale.unlink()
    written = []
    for info, case_arrays, root_arrays in out:
        name = info["name"]
        info["arrays"] = {k: save(f"{name}_{k}", v) for k, v in case_arrays.items()}
        for root, arrays in zip(info["roots"], root_arrays, strict=True):
            root["arrays"] = {
                k: save(f"{name}_root{root['root']}_{k}", v) for k, v in arrays.items()
            }
        written.append(info)
    doc = {
        "description": "The DF-CASSCF analytic nuclear Hessian's reference data "
        "(casscf_hessian.tex over a Coulomb-metric fit): per case the "
        "orbitals and roots, the state-averaged CP right-hand sides and "
        "responses lambda^x; per root E_xy and its pieces, F and F^[x], "
        "the response terms, at a state average the multipliers z and the z "
        "terms, the third-derivative action and the root-rotation term, the "
        "analytic Hessian and the relaxed-gradient difference it is checked "
        "against. Coordinates in bohr, energies in hartree; arrays are '<f8' "
        "Fortran-order .npy files beside this one, test/gtest/data/README.md "
        "has every layout",
        "provenance": {
            "pyscf": pyscf.__version__,
            "libcint": libcint_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "cases": written,
    }
    MANIFEST.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
