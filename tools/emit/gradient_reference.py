#!/usr/bin/env python3
"""Write the reference data the nuclear-gradient tests read.

Everything runs over this tree's own basis files, `data/basis/*.json` for the
orbitals and `data/aux_basis/*.json` for the fitting sets, handed to PySCF as
explicit shells (not its library's copies), spherical throughout. Every
gradient is n_atom x 3, G[A][k] = dE/dR_Ak in hartree/bohr, atoms in the
order written:

- `integrals`: libcint's derivative integrals, each contracted with one fixed
  symmetric AO matrix M into a per-atom gradient, sum_mn M_mn dX_mn/dR, for
  H2O in cc-pVDZ and cc-pVTZ (oxygen's f shells): the overlap (int1e_ipovlp),
  the kinetic (int1e_ipkin) and the nuclear attraction, split into the
  functions' term (int1e_ipnuc) and the operator's (int1e_iprinv at each
  nucleus); and int2e_ip1 contracted into dE_J, E_J = 1/2 sum D_ab D_cd
  (ab|cd), and dE_K, E_K = 1/2 sum D_ab D_cd (ac|bd), with D = M.
- `density_fitting`: int2c2e_ip1 contracted with a fixed W, sum_PQ W_PQ
  d(P|Q)/dR, and int3c2e_ip1 / int3c2e_ip2 with a fixed Gamma, sum_mnP
  Gamma_mnP d(mn|P)/dR, for H2O in cc-pVDZ / cc-pVDZ-JKFIT, cc-pVTZ /
  cc-pVTZ-JKFIT (aux to g) and cc-pVTZ / cc-pVTZ-acCD-6-6 (aux to i, so (ff|i)
  and (i|i)).
- `df_rhf`: PySCF's DF-RHF gradient (`df.grad.rhf`, auxbasis response on),
  total and each term, for H2O and NH4+ in cc-pVDZ / cc-pVDZ-JKFIT, a
  distorted H2O in the same, H2O in cc-pVTZ / cc-pVTZ-JKFIT and H2O in
  cc-pVDZ / cc-pVDZ-acCD.
- `rhf`: PySCF's integral-direct RHF gradient (`grad.rhf`), total and each
  term, for H2O in cc-pVDZ, the distorted H2O and H2O in cc-pVTZ.
- `optimized`: DF-RHF / cc-pVDZ / cc-pVDZ-JKFIT minima of H2O and anti
  ethanol, max |dE/dR| below 1e-7, with their energies.

The fixed matrices are a recipe, not a file: M_mn = 1 / (m + n + 1) (the
Hilbert matrix the coulomb/ fixtures already use), W_PQ = 1 / (P + Q + 1),
Gamma_mnP = 1 / (m + n + P + 1), each index 0-based in PySCF's spherical AO
order. One IEEE division each, so a C++ test rebuilds them bit for bit.

Every analytic number written is checked here against a fourth-order central
difference of the matching energy at fixed matrices (step 1e-3 bohr), and each
full gradient's terms against PySCF's own total. BLAS and OpenMP are pinned to
one thread, and nothing run-dependent (date, host) is written, so a rerun
regenerates the file byte for byte.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/gradient_reference.py

writes test/gtest/data/gradient/gradient.json. Re-run it to regenerate; the
file is never hand-edited.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json  # noqa: E402
import pathlib  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from df_reference import WATER_SKEW, libcint_version, shells  # noqa: E402
from pyscf import df, gto, lib, scf  # noqa: E402
from scf_reference import AMMONIUM, BASES, WATER, pyscf_basis  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "gradient" / "gradient.json"

# Each fitting set's file under data/aux_basis (pyscf_basis reads from
# data/basis, hence the relative path), by this tree's builtin name.
AUX_FILES = {
    "cc-pvdz-jkfit": "../aux_basis/cc-pVDZ-JKFIT.json",
    "cc-pvtz-jkfit": "../aux_basis/cc-pVTZ-JKFIT.json",
    "cc-pvdz-accd": "../aux_basis/cc-pVDZ-acCD.json",
    "cc-pvtz-accd-6-6": "../aux_basis/cc-pVTZ-acCD-6-6.json",
}

# Water far from its minimum: one bond stretched to 2.25 bohr, the other
# compressed to 1.67, the angle opened to 115 degrees, in no symmetry plane,
# so no gradient component is near zero. Bohr.
WATER_DISTORTED = [
    ("O", (0.1, -0.05, 0.02)),
    ("H", (2.2, 0.45, -0.6)),
    ("H", (-1.1, 0.5, -1.0)),
]

# Anti ethanol in its mirror plane z = 0, a rough start (angstrom, converted
# to bohr): the optimizer keeps the plane, so it reaches the Cs minimum.
ETHANOL_START = [
    ("C", (1.1879, -0.3829, 0.0)),
    ("C", (0.0, 0.5526, 0.0)),
    ("O", (-1.1867, -0.2472, 0.0)),
    ("H", (-1.9237, 0.385, 0.0)),
    ("H", (2.0985, 0.2306, 0.0)),
    ("H", (1.1184, -1.0093, 0.8869)),
    ("H", (1.1184, -1.0093, -0.8869)),
    ("H", (-0.0227, 1.1812, 0.8852)),
    ("H", (-0.0227, 1.1812, -0.8852)),
]
ANGSTROM = 1.0 / lib.param.BOHR

INTEGRAL_CASES = [
    {"name": "h2o_ccpvdz", "atoms": WATER_SKEW, "basis": "cc-pvdz", "max_l": 2},
    {"name": "h2o_ccpvtz", "atoms": WATER_SKEW, "basis": "cc-pvtz", "max_l": 3},
]
DF_INTEGRAL_CASES = [
    {
        "name": "h2o_ccpvdz",
        "atoms": WATER_SKEW,
        "basis": "cc-pvdz",
        "aux_basis": "cc-pvdz-jkfit",
        "max_l": (2, 3),
    },
    {
        "name": "h2o_ccpvtz",
        "atoms": WATER_SKEW,
        "basis": "cc-pvtz",
        "aux_basis": "cc-pvtz-jkfit",
        "max_l": (3, 4),
    },
    {
        "name": "h2o_ccpvtz_accd66",
        "atoms": WATER_SKEW,
        "basis": "cc-pvtz",
        "aux_basis": "cc-pvtz-accd-6-6",
        "max_l": (3, 6),
    },
]
DF_RHF_CASES = [
    {
        "name": "h2o_ccpvdz",
        "atoms": WATER,
        "basis": "cc-pvdz",
        "aux_basis": "cc-pvdz-jkfit",
        "charge": 0,
    },
    {
        "name": "nh4_ccpvdz",
        "atoms": AMMONIUM,
        "basis": "cc-pvdz",
        "aux_basis": "cc-pvdz-jkfit",
        "charge": 1,
    },
    {
        "name": "h2o_distorted_ccpvdz",
        "atoms": WATER_DISTORTED,
        "basis": "cc-pvdz",
        "aux_basis": "cc-pvdz-jkfit",
        "charge": 0,
    },
    {
        "name": "h2o_ccpvtz",
        "atoms": WATER,
        "basis": "cc-pvtz",
        "aux_basis": "cc-pvtz-jkfit",
        "charge": 0,
    },
    {
        "name": "h2o_ccpvdz_accd",
        "atoms": WATER,
        "basis": "cc-pvdz",
        "aux_basis": "cc-pvdz-accd",
        "charge": 0,
    },
]
RHF_CASES = [
    {"name": "h2o_ccpvdz", "atoms": WATER, "basis": "cc-pvdz", "charge": 0},
    {
        "name": "h2o_distorted_ccpvdz",
        "atoms": WATER_DISTORTED,
        "basis": "cc-pvdz",
        "charge": 0,
    },
    {"name": "h2o_ccpvtz", "atoms": WATER, "basis": "cc-pvtz", "charge": 0},
]
OPTIMIZED_CASES = [
    {
        "name": "h2o_ccpvdz",
        "atoms": WATER,
        "basis": "cc-pvdz",
        "aux_basis": "cc-pvdz-jkfit",
        "charge": 0,
    },
    {
        "name": "ethanol_ccpvdz",
        "atoms": [(s, tuple(x * ANGSTROM for x in r)) for s, r in ETHANOL_START],
        "basis": "cc-pvdz",
        "aux_basis": "cc-pvdz-jkfit",
        "charge": 0,
    },
]

# The finite-difference check: fourth-order central difference at this step
# (bohr), and the agreement it must reach, relative to max(1, max |G|).
FD_STEP = 1e-3
FD_TOLERANCE = 1e-7
# Optimization: max |dE/dR| at the minimum, and the Hessian's step (bohr).
FORCE_TOLERANCE = 1e-7
HESSIAN_STEP = 1e-3

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/gradient_reference.py"
)


def molecules(case: dict, coords: numpy.ndarray | None = None):
    """The orbital molecule of one case and its fitting molecule (None for
    an orbital-only case), at coords (bohr, n_atom x 3) if given."""
    symbols = [symbol for symbol, _ in case["atoms"]]
    if coords is None:
        coords = numpy.array([coord for _, coord in case["atoms"]])
    mol = gto.M(
        atom=[[s, tuple(r)] for s, r in zip(symbols, coords, strict=True)],
        unit="Bohr",
        basis=pyscf_basis(BASES[case["basis"]], set(symbols)),
        charge=case.get("charge", 0),
        cart=False,
        verbose=0,
    )
    if "aux_basis" not in case:
        return mol, None
    aux = pyscf_basis(AUX_FILES[case["aux_basis"]], set(symbols))
    return mol, df.addons.make_auxmol(mol, aux)


def recipe(*sizes: int) -> numpy.ndarray:
    """The fixed contraction tensor: 1 / (i + j + ... + 1) over 0-based indices."""
    return 1.0 / (sum(numpy.ix_(*(numpy.arange(n) for n in sizes))) + 1.0)


def atoms_out(mol: gto.Mole) -> list[dict]:
    return [
        {
            "symbol": mol.atom_pure_symbol(a),
            "z": int(mol.atom_charge(a)),
            "coord": [float(x) for x in mol.atom_coord(a)],
        }
        for a in range(mol.natm)
    ]


def rows(gradient: numpy.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in gradient]


def bra(m: gto.Mole, ip: numpy.ndarray, matrix: numpy.ndarray) -> numpy.ndarray:
    """sum matrix_m... d/dR of the integral, the derivative on the first index
    only: ip is libcint's ip1 (nabla on that index, -d/dR of its centre), and
    matrix is indexed alike."""
    out = numpy.zeros((m.natm, 3))
    for a, (_, _, p0, p1) in enumerate(m.aoslice_by_atom()):
        out[a] = -numpy.tensordot(ip[:, p0:p1], matrix[p0:p1], axes=matrix.ndim)
    return out


def one_electron(mol: gto.Mole, matrix: numpy.ndarray) -> dict:
    """sum_mn M_mn dX_mn/dR for the overlap, kinetic and nuclear attraction,
    M symmetric, so each derivative on n equals the one on m."""
    potential_operator = numpy.zeros((mol.natm, 3))
    for a in range(mol.natm):
        with mol.with_rinv_at_nucleus(a):
            iprinv = mol.intor("int1e_iprinv")
        # d/dR_A of <m| -Z_A / |r - A| |n> = -Z_A (<nabla m|..|n> + <m|..|nabla n>)
        potential_operator[a] = (
            -2.0 * mol.atom_charge(a) * numpy.einsum("xmn,mn->x", iprinv, matrix)
        )
    potential_functions = 2.0 * bra(mol, mol.intor("int1e_ipnuc"), matrix)
    return {
        "overlap": 2.0 * bra(mol, mol.intor("int1e_ipovlp"), matrix),
        "kinetic": 2.0 * bra(mol, mol.intor("int1e_ipkin"), matrix),
        "potential": {
            "functions": potential_functions,
            "operator": potential_operator,
            "total": potential_functions + potential_operator,
        },
    }


def one_electron_energies(mol: gto.Mole, matrix: numpy.ndarray) -> dict:
    return {
        "overlap": numpy.sum(matrix * mol.intor("int1e_ovlp")),
        "kinetic": numpy.sum(matrix * mol.intor("int1e_kin")),
        "potential": numpy.sum(matrix * mol.intor("int1e_nuc")),
    }


def two_electron(mol: gto.Mole, density: numpy.ndarray) -> dict:
    """dE_J and dE_K, E_J = 1/2 sum D_ab D_cd (ab|cd) and E_K = 1/2 sum D_ab
    D_cd (ac|bd): each derivative on the four functions alike, so 4 x the
    first's, unscreened."""
    ip = mol.intor("int2e_ip1", aosym="s1")
    vj = numpy.einsum("xabcd,cd->xab", ip, density)
    vk = numpy.einsum("xacbd,cd->xab", ip, density)
    return {
        "coulomb": 2.0 * bra(mol, vj, density),
        "exchange": 2.0 * bra(mol, vk, density),
    }


def two_electron_energies(mol: gto.Mole, density: numpy.ndarray) -> dict:
    eri = mol.intor("int2e", aosym="s1")
    return {
        "coulomb": 0.5 * numpy.einsum("abcd,ab,cd->", eri, density, density),
        "exchange": 0.5 * numpy.einsum("acbd,ab,cd->", eri, density, density),
    }


def metric(auxmol: gto.Mole, w: numpy.ndarray) -> numpy.ndarray:
    """sum_PQ W_PQ d(P|Q)/dR, W symmetric."""
    return 2.0 * bra(auxmol, auxmol.intor("int2c2e_ip1"), w)


def three_center(mol: gto.Mole, auxmol: gto.Mole, gamma: numpy.ndarray) -> dict:
    """sum_mnP Gamma_mnP d(mn|P)/dR, Gamma symmetric in m and n: the orbital
    functions' term (int3c2e_ip1, both of them) and the aux one's (ip2)."""
    assert numpy.array_equal(gamma, gamma.transpose(1, 0, 2))
    ip1 = df.incore.aux_e2(mol, auxmol, "int3c2e_ip1_sph", aosym="s1", comp=3)
    ip2 = df.incore.aux_e2(mol, auxmol, "int3c2e_ip2_sph", aosym="s1", comp=3)
    orbital = 2.0 * bra(mol, ip1, gamma)
    aux = bra(auxmol, ip2.transpose(0, 3, 1, 2), gamma.transpose(2, 0, 1))
    return {"orbital": orbital, "aux": aux, "total": orbital + aux}


def df_two_electron(mol: gto.Mole, auxmol: gto.Mole, density: numpy.ndarray):
    """The density-fitted dE_J and dE_K of two_electron's E_J and E_K, at
    fixed density, and those energies.

    With B = (mn|P), M = (P|Q) and X = M^-1 B (P fitted):
      E_J = 1/2 sum D_mn B_mnP X_lsP D_ls: Gamma = D (x) d, W = -1/2 d d^T,
        d = M^-1 B.D;
      E_K = 1/2 sum D_ml D_ns B_mnP X_lsP: Gamma_mnP = sum D_ml X_lsP D_sn,
        W = -1/2 Y, Y_PQ = sum_mn X_mnP Gamma_mnQ;
    each dE = sum Gamma dB + sum W dM.
    """
    b = df.incore.aux_e2(mol, auxmol, "int3c2e_sph", aosym="s1")
    m = auxmol.intor("int2c2e")
    x = numpy.linalg.solve(m, b.reshape(-1, auxmol.nao).T).T.reshape(b.shape)
    d = numpy.einsum("mnP,mn->P", x, density)
    gamma_j = numpy.einsum("mn,P->mnP", density, d)
    gamma_k = numpy.einsum("ml,lsP,sn->mnP", density, x, density, optimize=True)
    gamma_k = 0.5 * (gamma_k + gamma_k.transpose(1, 0, 2))
    y = numpy.einsum("mnP,mnQ->PQ", x, gamma_k, optimize=True)
    gradients = {
        "coulomb": three_center(mol, auxmol, gamma_j)["total"]
        + metric(auxmol, -0.5 * numpy.outer(d, d)),
        "exchange": three_center(mol, auxmol, gamma_k)["total"]
        + metric(auxmol, -0.5 * 0.5 * (y + y.T)),
    }
    energies = {
        "coulomb": 0.5 * numpy.einsum("mnP,mn,P->", b, density, d),
        "exchange": 0.5 * numpy.einsum("mnP,mnP->", b, gamma_k),
    }
    return gradients, energies


def finite_difference(case: dict, energy) -> numpy.ndarray:
    """Fourth-order central difference of energy(mol, auxmol) over every
    nuclear coordinate, the matrices it closes over held fixed."""
    coords = numpy.array([coord for _, coord in case["atoms"]])
    out = numpy.zeros(coords.shape)
    for a in range(coords.shape[0]):
        for k in range(3):
            values = {}
            for s in (-2, -1, 1, 2):
                moved = coords.copy()
                moved[a, k] += s * FD_STEP
                values[s] = energy(*molecules(case, moved))
            out[a, k] = (8.0 * (values[1] - values[-1]) - (values[2] - values[-2])) / (
                12.0 * FD_STEP
            )
    return out


def check_fd(name: str, analytic: numpy.ndarray, case: dict, energy) -> None:
    numeric = finite_difference(case, energy)
    error = numpy.abs(analytic - numeric).max()
    scale = max(1.0, numpy.abs(analytic).max())
    assert error < FD_TOLERANCE * scale, (case["name"], name, error)


def check_translation(name: str, gradient: numpy.ndarray) -> None:
    drift = numpy.abs(gradient.sum(axis=0)).max()
    assert drift < 1e-9 * max(1.0, numpy.abs(gradient).max()), (name, drift)


def integral_case(case: dict) -> dict:
    mol, _ = molecules(case)
    assert max(mol.bas_angular(i) for i in range(mol.nbas)) == case["max_l"]
    matrix = recipe(mol.nao, mol.nao)
    one = one_electron(mol, matrix)
    two = two_electron(mol, matrix)
    for key in ("overlap", "kinetic"):
        check_fd(
            key, one[key], case, lambda m, _, k=key: one_electron_energies(m, matrix)[k]
        )
    check_fd(
        "potential",
        one["potential"]["total"],
        case,
        lambda m, _: one_electron_energies(m, matrix)["potential"],
    )
    for key in ("coulomb", "exchange"):
        check_fd(
            key, two[key], case, lambda m, _, k=key: two_electron_energies(m, matrix)[k]
        )
    for key in ("overlap", "kinetic", "coulomb", "exchange"):
        check_translation(key, (one | two)[key])
    check_translation("potential", one["potential"]["total"])
    return {
        "name": case["name"],
        "basis": case["basis"],
        "atoms": atoms_out(mol),
        "nao": int(mol.nao),
        "shells": shells(mol),
        "overlap": rows(one["overlap"]),
        "kinetic": rows(one["kinetic"]),
        "potential": {key: rows(value) for key, value in one["potential"].items()},
        "coulomb": rows(two["coulomb"]),
        "exchange": rows(two["exchange"]),
    }


def df_integral_case(case: dict) -> dict:
    mol, auxmol = molecules(case)
    lmax = max(mol.bas_angular(i) for i in range(mol.nbas))
    aux_lmax = max(auxmol.bas_angular(i) for i in range(auxmol.nbas))
    assert (lmax, aux_lmax) == case["max_l"], (lmax, aux_lmax)
    w = recipe(auxmol.nao, auxmol.nao)
    gamma = recipe(mol.nao, mol.nao, auxmol.nao)
    got_metric = metric(auxmol, w)
    got_three = three_center(mol, auxmol, gamma)
    check_fd("metric", got_metric, case, lambda _, a: numpy.sum(w * a.intor("int2c2e")))
    check_fd(
        "three_center",
        got_three["total"],
        case,
        lambda m, a: numpy.sum(
            gamma * df.incore.aux_e2(m, a, "int3c2e_sph", aosym="s1")
        ),
    )
    check_translation("metric", got_metric)
    check_translation("three_center", got_three["total"])
    return {
        "name": case["name"],
        "basis": case["basis"],
        "aux_basis": case["aux_basis"],
        "atoms": atoms_out(mol),
        "nao": int(mol.nao),
        "naux": int(auxmol.nao),
        "shells": shells(mol),
        "aux_shells": shells(auxmol),
        "metric": rows(got_metric),
        "three_center": {key: rows(value) for key, value in got_three.items()},
    }


def converged(mf: scf.hf.SCF, dm0: numpy.ndarray | None = None) -> scf.hf.SCF:
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.kernel(dm0)
    assert mf.converged
    return mf


def df_rhf(mol: gto.Mole, auxmol: gto.Mole, dm0=None) -> scf.hf.SCF:
    """PySCF's converged DF-RHF over exactly auxmol's fitting functions."""
    mf = converged(scf.RHF(mol).density_fit(auxbasis=auxmol._basis), dm0)
    used = mf.with_df.auxmol
    assert numpy.array_equal(used._bas, auxmol._bas)
    assert numpy.array_equal(used._env, auxmol._env)
    return mf


def gradient_case(case: dict, density_fitted: bool) -> dict:
    """One full RHF gradient: PySCF's total, and each term at its density,
    each checked by finite difference at fixed matrices, their sum against the
    total."""
    mol, auxmol = molecules(case)
    mf = df_rhf(mol, auxmol) if density_fitted else converged(scf.RHF(mol))
    mf_grad = mf.nuc_grad_method()
    if density_fitted:
        assert mf_grad.auxbasis_response
    total = mf_grad.kernel()
    density = mf.make_rdm1()
    weighted = mf_grad.make_rdm1e()
    one = one_electron(mol, density)
    if density_fitted:
        two, energies = df_two_electron(mol, auxmol, density)
        check_fd(
            "coulomb",
            two["coulomb"],
            case,
            lambda m, a: df_two_electron(m, a, density)[1]["coulomb"],
        )
        check_fd(
            "exchange",
            two["exchange"],
            case,
            lambda m, a: df_two_electron(m, a, density)[1]["exchange"],
        )
        vj, vk = mf.get_jk(mol, density)
        e_j = 0.5 * numpy.sum(density * vj)
        e_k = 0.5 * numpy.sum(density * vk)
        assert abs(e_j - energies["coulomb"]) < 1e-10 * abs(e_j)
        assert abs(e_k - energies["exchange"]) < 1e-10 * abs(e_k)
    else:
        two = two_electron(mol, density)
    terms = {
        "kinetic": one["kinetic"],
        "potential": one["potential"]["total"],
        "overlap": -one_electron(mol, weighted)["overlap"],
        "coulomb": two["coulomb"],
        "exchange": -0.5 * two["exchange"],
        "nuclear_repulsion": mf_grad.grad_nuc(),
    }
    summed = sum(terms.values())
    assert numpy.abs(summed - total).max() < 1e-9, numpy.abs(summed - total).max()
    for key, value in terms.items():
        check_translation(key, value)
    check_translation("total", total)
    out = {
        "name": case["name"],
        "basis": case["basis"],
        "aux_basis": case.get("aux_basis"),
        "charge": case["charge"],
        "atoms": atoms_out(mol),
        "nao": int(mol.nao),
        "electrons": int(mol.nelectron),
        "energy": float(mf.e_tot),
        "total": rows(total),
        "terms": {key: rows(value) for key, value in terms.items()},
    }
    if density_fitted:
        out["naux"] = int(auxmol.nao)
    return out


def internal_projector(coords: numpy.ndarray) -> numpy.ndarray:
    """The projector off rigid translations and rotations of coords."""
    centred = coords - coords.mean(axis=0)
    rigid = [numpy.tile(numpy.eye(3)[k], len(coords)) for k in range(3)]
    rigid += [numpy.cross(centred, numpy.eye(3)[k]).ravel() for k in range(3)]
    q, _ = numpy.linalg.qr(numpy.array(rigid).T)
    return numpy.eye(coords.size) - q @ q.T


def optimized_case(case: dict) -> dict:
    """Newton's method on PySCF's analytic DF-RHF gradient, its Hessian a
    central difference of that gradient, translations and rotations
    projected out, to max |dE/dR| below FORCE_TOLERANCE."""
    start = numpy.array([coord for _, coord in case["atoms"]])
    coords = start.copy()
    state = {"dm": None}

    def energy_gradient(at: numpy.ndarray):
        mf = df_rhf(*molecules(case, at), state["dm"])
        state["dm"] = mf.make_rdm1()
        return mf.e_tot, mf.nuc_grad_method().kernel()

    steps = 0
    energy, gradient = energy_gradient(coords)
    while numpy.abs(gradient).max() >= FORCE_TOLERANCE:
        assert steps < 20, f"{case['name']} did not converge"
        n = coords.size
        hessian = numpy.zeros((n, n))
        for i in range(n):
            for s in (1, -1):
                moved = coords.ravel().copy()
                moved[i] += s * HESSIAN_STEP
                hessian[i] += (
                    s * energy_gradient(moved.reshape(coords.shape))[1].ravel()
                )
        hessian = 0.5 * (hessian + hessian.T) / (2.0 * HESSIAN_STEP)
        projector = internal_projector(coords)
        values, vectors = numpy.linalg.eigh(projector @ hessian @ projector)
        internal = numpy.abs(values) > 1e-6
        assert internal.sum() == n - 6 and values[internal].min() > 0.0
        inverse = (vectors[:, internal] / values[internal]) @ vectors[:, internal].T
        step = -inverse @ (projector @ gradient.ravel())
        step *= min(1.0, 0.3 / numpy.abs(step).max())
        coords = coords + step.reshape(coords.shape)
        energy, gradient = energy_gradient(coords)
        steps += 1
    # The recorded energy from PySCF's default guess, not the last one's.
    state["dm"] = None
    final_energy, final_gradient = energy_gradient(coords)
    assert abs(final_energy - energy) < 1e-10
    assert numpy.abs(final_gradient).max() < FORCE_TOLERANCE
    mol, auxmol = molecules(case, coords)
    return {
        "name": case["name"],
        "basis": case["basis"],
        "aux_basis": case["aux_basis"],
        "charge": case["charge"],
        "nao": int(mol.nao),
        "naux": int(auxmol.nao),
        "start": atoms_out(molecules(case, start)[0]),
        "atoms": atoms_out(mol),
        "energy": float(final_energy),
        "max_force": float(numpy.abs(final_gradient).max()),
        "newton_steps": steps,
        "optimizer": "Newton's method over PySCF's analytic DF-RHF gradient "
        "(df.grad.rhf), the Hessian a central difference of it (step "
        f"{HESSIAN_STEP} bohr) recomputed every step, rigid translations and "
        "rotations projected out, steps capped at 0.3 bohr per coordinate; "
        f"converged to max |dE/dR| < {FORCE_TOLERANCE}",
    }


def main() -> None:
    lib.num_threads(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "description": "PySCF/libcint nuclear-gradient references over this "
        "tree's data/basis and data/aux_basis files, spherical. Every "
        "gradient is n_atom x 3 rows, dE/dR in hartree/bohr, in atoms order; "
        "coordinates in bohr, energies in hartree",
        "provenance": {
            "pyscf": pyscf.__version__,
            "libcint": libcint_version(),
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "recipe": {
            "M": "M_mn = 1 / (m + n + 1), nao x nao",
            "W": "W_PQ = 1 / (P + Q + 1), naux x naux",
            "Gamma": "Gamma_mnP = 1 / (m + n + P + 1), nao x nao x naux",
            "indices": "0-based, PySCF's spherical AO order (shells in "
            "`shells` order, m = -l..l, p as x, y, z), unit-normalised "
            "functions",
        },
        "integrals": [integral_case(case) for case in INTEGRAL_CASES],
        "density_fitting": [df_integral_case(case) for case in DF_INTEGRAL_CASES],
        "df_rhf": [gradient_case(case, True) for case in DF_RHF_CASES],
        "rhf": [gradient_case(case, False) for case in RHF_CASES],
        "optimized": [optimized_case(case) for case in OPTIMIZED_CASES],
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
