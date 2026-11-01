#!/usr/bin/env python3
"""Write the reference data the DF SA-CASSCF nonadiabatic-coupling tests read.

For each case below, the density-fitted state-averaged CASSCF is converged,
gated, polished and put in canonical orbitals exactly as
casscf_gradient_reference.py does it (mc1step and mc2step to one minimum,
Newton steps to |g| < 1e-10, `mc.canonicalize(sort=True, cas_natorb=True)`,
the CI re-solved densely), over a DF-RHF `_scf` sharing the CASSCF's
`with_df`, so every two-electron quantity is over this tree's data/aux_basis
file. Then, for every ordered pair (I, J) of its roots, PySCF's analytic
derivative coupling d_IJ = <I|d J/dR> and its pieces are written.

**The DF route is a patch.** PySCF 2.14's `pyscf.nac.sacasscf
.NonAdiabaticCouplings` derives from the non-DF `grad.sacasscf.Gradients`.
`DFNonAdiabaticCouplings` below puts its NAC methods (the transition RDMs in
place of a root's, the core and nuclear terms removed, the CSF term) over
`df.grad.sacasscf.Gradients`, whose Lagrange terms and DF-CASSCF
Hellmann-Feynman gradient are fitted. The tool checks the patch two ways: the
fitted couplings approach PySCF's exact (non-DF) ones as the fitting set grows
(cc-pVDZ-, cc-pVTZ-, cc-pVQZ-JKFIT), and the total coupling matches a
finite difference of the fitted states' overlaps <I(R0)|J(R)> along a random
displacement, which also fixes which state is the bra: PySCF's
`kernel(state=(I, J))` is <I|d J/dR> = h_IJ / (E_J - E_I) + d^CSF_IJ, the
reverse of what its docstring says, and the tool calls it so.

**Signs.** A coupling changes sign with either state's phase. After the dense
re-solve, each CI vector is signed so that its first coefficient, in
row-major (alpha string, beta string) order, of magnitude within 1e-8 of its
largest is positive; the tool refuses a vector with a coefficient between
1e-8 and 1e-4 short of its largest, where that choice would be fragile. The
orbitals' signs are PySCF's, as in casscf_gradient/.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/casscf_nac_reference.py

writes test/gtest/data/casscf_nac/casscf_nac.json. Re-run it to regenerate;
the file is never hand-edited. BLAS and OpenMP are pinned to one thread and
nothing run-dependent (date, host) is written, so a rerun on the same PySCF
and numpy regenerates the file byte for byte.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import itertools  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402

# casscf_reference pins its own thread count on import; numpy's BLAS is
# already loaded at one, and main() pins PySCF's OpenMP.
from casci_reference import aux_basis  # noqa: E402
from casscf_gradient_reference import (  # noqa: E402
    build,
    canonical,
    matrix,
    minimum,
    polish,
    rows,
)
from casscf_reference import (  # noqa: E402
    CONVERGED_AGREEMENT,
    FORMALDEHYDE,
    OUT_DF,
    ROOT_AGREEMENT,
    casscf,
    check_aux,
    molecule,
    optimize,
    spin_roots,
)
from pyscf import fci, gto, lib, scf  # noqa: E402
from pyscf.df.grad import casscf as dfcasscf_grad  # noqa: E402
from pyscf.df.grad import rhf as dfrhf_grad  # noqa: E402
from pyscf.df.grad import sacasscf as dfsacasscf_grad  # noqa: E402
from pyscf.grad import casscf as casscf_grad  # noqa: E402
from pyscf.nac import sacasscf as nac_sacasscf  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "casscf_nac" / "casscf_nac.json"

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/casscf_nac_reference.py"
)

# Formaldehyde pyramidalized and twisted (bohr): C1, so no coupling
# component vanishes by symmetry.
FORMALDEHYDE_SKEW = [
    ("C", (0.0, 0.0, 0.0)),
    ("O", (0.12, -0.08, 2.29)),
    ("H", (0.45, 1.74, -1.13)),
    ("H", (-0.31, -1.81, -1.02)),
]

# All cc-pVDZ over cc-pVDZ-JKFIT, CAS(4,3) frontier (1b1 pi, 2b2 n, 2b1 pi*
# at the planar geometry), equal weights: PySCF refuses unequal ones.
CASES = [
    {
        "name": "h2co_ccpvdz_sa2",
        "from": "h2co_ccpvdz_sa2",
        "atoms": FORMALDEHYDE,
        "basis": "cc-pvdz",
        "cas": (4, 3),
        "weights": [0.5, 0.5],
    },
    {
        "name": "h2co_skew_ccpvdz_sa3",
        "atoms": FORMALDEHYDE_SKEW,
        "basis": "cc-pvdz",
        "cas": (4, 3),
        "weights": [1 / 3, 1 / 3, 1 / 3],
    },
]

# The Lagrange solve's tolerances (PySCF's defaults: rtol 1e-7, atol 1e-12).
LAGRANGE_RTOL = 1e-12
LAGRANGE_ATOL = 1e-14
# The CI sign rule's tie band and the margin it must clear.
SIGN_TIE = 1e-8
SIGN_MARGIN = 1e-4
# PySCF's own NAC rerun, the pieces' sums, d_IJ + d_JI, h_IJ - h_JI and the
# CSF-free coupling's translational sum.
RERUN_TOL = 1e-9
PIECES_TOL = 1e-10
ANTISYMMETRY_TOL = 1e-8
TRANSLATION_TOL = 1e-8
# The overlap finite difference: the five-point stencil's step (bohr) and
# the tolerance on each pair's directional derivative.
FD_STEP = 5e-3
FD_TOL = 1e-6
# The fitting-set ladder, the exact couplings' last stand-in; each rung's
# miss must shrink, and the last be under LADDER_TOL (bohr^-1).
LADDER = ("cc-pvtz-jkfit", "cc-pvqz-jkfit")
LADDER_TOL = 1e-4


class DFNonAdiabaticCouplings(
    nac_sacasscf.NonAdiabaticCouplings, dfsacasscf_grad.Gradients
):
    """PySCF's SA-CASSCF NACs over the DF SA-CASSCF gradient.

    The NAC class calls `sacasscf_grad.Gradients.kernel` by name, past the
    DF class's kernel, so this kernel does what that one does: hand it a
    DF-RHF mf_grad and make `casscf_grad.Gradients` (which both
    `get_ham_response`s build) the DF-CASSCF gradient. `get_LdotJnuc` comes
    from the DF class through the MRO."""

    def __init__(self, mc, state=None, mult_ediff=False, use_etfs=False):
        self.auxbasis_response = True
        nac_sacasscf.NonAdiabaticCouplings.__init__(
            self, mc, state=state, mult_ediff=mult_ediff, use_etfs=use_etfs
        )

    def kernel(self, *args, **kwargs):
        if kwargs.get("mf_grad") is None:
            kwargs["mf_grad"] = dfrhf_grad.Gradients(self.base._scf)
        with lib.temporary_env(casscf_grad, Gradients=dfcasscf_grad.Gradients):
            return nac_sacasscf.NonAdiabaticCouplings.kernel(self, *args, **kwargs)


def signed(name: str, vectors: list) -> list:
    """Each CI vector under the sign rule in the module docstring."""
    out = []
    for vector in vectors:
        flat = numpy.abs(vector.ravel())
        top = flat.max()
        near = flat[(flat < top - SIGN_TIE) & (flat > top - SIGN_MARGIN)]
        assert near.size == 0, f"{name}: the CI sign rule is fragile"
        first = int(numpy.argmax(flat >= top - SIGN_TIE))
        out.append(vector if vector.ravel()[first] > 0 else -vector)
    return out


def sa_state(case: dict, mol: gto.Mole, auxbasis, start: numpy.ndarray):
    """One mc1step DF-SA-CASSCF (exact when auxbasis is None), polished,
    canonical and signed: the ladder's and the displaced geometries' runs."""
    name, weights = case["name"], case["weights"]
    nelecas, ncas = case["cas"]
    if auxbasis is None:
        mf = scf.RHF(mol)
        mf.mo_coeff = start
        mf.mo_occ = numpy.zeros(start.shape[1])
        mf.mo_occ[: mol.nelectron // 2] = 2
        mc = casscf(mf, ncas, nelecas, weights)
    else:
        mc = build(case, mol, auxbasis, start)
    optimize(mc, mc.kernel, start)
    assert mc.converged, f"{name}: mc1step did not converge"
    state = canonical(name, mc, polish(name, mc, mc.mo_coeff, weights), weights)
    state["ci"] = signed(name, state["ci"])
    return mc, state


def couplings(mc, state: dict, nac_class, detail: bool) -> dict:
    """Every ordered pair's couplings at the state, from nac_class; with
    detail, its pieces, transition RDMs and multipliers too."""
    mc.mo_coeff, mc.ci = state["mo"], list(state["ci"])
    mc.fcisolver.e_states = list(state["energies"])
    ncas, nelecas = mc.ncas, mc.nelecas
    out = {}
    for bra, ket in itertools.permutations(range(len(state["ci"])), 2):

        def nac(bra=bra, ket=ket, **kwargs):
            method = nac_class(mc, state=(bra, ket))
            method.verbose = 0
            method.conv_rtol, method.conv_atol = LAGRANGE_RTOL, LAGRANGE_ATOL
            return method, method.kernel(state=(bra, ket), **kwargs)

        _, total = nac(use_etfs=False)
        _, etf = nac(use_etfs=True)
        record = {"total": total, "etf": etf}
        if detail:
            method, h = nac(use_etfs=True, mult_ediff=True)
            record["h"] = h
            record["csf"] = nac_sacasscf.nac_csf(
                method, state=(bra, ket), mf_grad=dfrhf_grad.Gradients(mc._scf)
            )
            record.update(pieces(method, mc, state, bra, ket))
            ci = state["ci"]
            t1, t2 = fci.direct_spin1.trans_rdm12(ci[bra], ci[ket], ncas, nelecas)
            record["rdm"] = {
                "trans_1": t1,
                "trans_2": t2,
                "sym_1": 0.5 * (t1 + t1.T),
                "sym_2": 0.5 * (t2 + t2.transpose(1, 0, 3, 2)),
            }
        out[(bra, ket)] = record
    return out


def pieces(method, mc, state: dict, bra: int, ket: int) -> dict:
    """h_IJ's Hellmann-Feynman and Lagrange halves, the transition orbital
    gradient b and the multipliers z, from the h_IJ run's own solve."""
    mo, ci = state["mo"], list(state["ci"])
    mf_grad = dfrhf_grad.Gradients(mc._scf)
    with lib.temporary_env(casscf_grad, Gradients=dfcasscf_grad.Gradients):
        hellmann_feynman = method.get_ham_response(
            state=(bra, ket), mo=mo, ci=ci, mf_grad=mf_grad, use_etfs=True
        )
        lagrange = method.get_LdotJnuc(
            method.Lvec, state=(bra, ket), mo=mo, ci=ci, mf_grad=mf_grad
        )
        b = method.get_wfn_response(state=(bra, ket), mo=mo, ci=ci)
        aop, _ = method.get_Aop_Adiag(state=(bra, ket), mo=mo, ci=ci)
    assert method.converged, f"({bra}, {ket}): the Lagrange solve did not converge"
    z = method.Lvec
    z_orbital, z_ci = method.unpack_uniq_var(z)
    return {
        "hellmann_feynman": hellmann_feynman,
        "lagrange": lagrange,
        "b": {
            "orbital": b[: method.ngorb],
            "ci_norm": float(numpy.linalg.norm(b[method.ngorb :])),
        },
        "z": {
            "orbital": z[: method.ngorb],
            "ci": z_ci,
            "residual": float(numpy.linalg.norm(aop(z) + b)),
        },
    }


def ci_overlap(mol0, mo0, ci0, mol1, mo1, ci1, ncore, ncas, nelecas) -> numpy.ndarray:
    """<I(mol0)|J(mol1)> for every root pair, by determinant overlaps over
    the two geometries' orbitals and the mixed AO overlap."""
    s = mo0.T @ gto.intor_cross("int1e_ovlp", mol0, mol1) @ mo1
    core = list(range(ncore))

    def strings(n):
        return [
            core + [ncore + p for p in range(ncas) if bits >> p & 1]
            for bits in fci.cistring.make_strings(range(ncas), n)
        ]

    def table(n):
        occ = strings(n)
        return numpy.array(
            [[numpy.linalg.det(s[numpy.ix_(a, b)]) for b in occ] for a in occ]
        )

    da, db = table(nelecas[0]), table(nelecas[1])
    return numpy.array([[numpy.sum(c0 * (da @ c1 @ db.T)) for c1 in ci1] for c0 in ci0])


def finite_difference(case: dict, mol0, state: dict, auxbasis, nac: dict) -> float:
    """The worst miss of each pair's total coupling along a random unit
    displacement against a five-point difference of <I(R0)|J(R)>."""
    name, weights = case["name"], case["weights"]
    nelecas, ncas = case["cas"]
    coords = numpy.array([r for _, r in case["atoms"]], dtype=float)
    direction = numpy.random.default_rng(658).standard_normal(coords.shape)
    direction /= numpy.linalg.norm(direction)
    nroots = len(weights)
    s = {}
    for k in (-2, -1, 1, 2):
        moved = dict(
            case,
            atoms=[
                (sym, tuple(r))
                for (sym, _), r in zip(
                    case["atoms"], coords + k * FD_STEP * direction, strict=True
                )
            ],
        )
        mol = molecule(moved)
        mo = state["mo"]
        overlap = mo.T @ mol.intor("int1e_ovlp") @ mo
        values, vectors = numpy.linalg.eigh(overlap)
        mo = mo @ vectors @ numpy.diag(values**-0.5) @ vectors.T
        mc = build(moved, mol, auxbasis, mo)
        optimize(mc, mc.kernel, mo)
        assert mc.converged, f"{name}: a displaced mc1step did not converge"
        mo = polish(name, mc, mc.mo_coeff, weights)
        _, ci, _ = spin_roots(mc, mo, nroots)
        ncore = mc.ncore
        overlaps = ci_overlap(
            mol0, state["mo"], state["ci"], mol, mo, ci, ncore, ncas, mc.nelecas
        )
        # Each displaced root's phase follows its own root at R0.
        phases = numpy.sign(numpy.diag(overlaps))
        swapped = numpy.abs(numpy.diag(overlaps)).min() < 0.99
        assert not swapped, f"{name}: a root swapped"
        s[k] = overlaps * phases[None, :]
    slope = (s[-2] - 8 * s[-1] + 8 * s[1] - s[2]) / (12 * FD_STEP)
    worst = 0.0
    for (bra, ket), record in nac.items():
        miss = abs(float(numpy.sum(record["total"] * direction)) - slope[bra, ket])
        assert miss < FD_TOL, f"{name}: ({bra}, {ket}) off the overlaps by {miss}"
        worst = max(worst, miss)
    return worst


def aligned_miss(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """max |a - s b|, s the sign that aligns them: two runs' orbitals may
    sign their states apart."""
    s = 1.0 if numpy.sum(a * b) >= 0 else -1.0
    return float(numpy.abs(a - s * b).max())


def ladder(case: dict, mol, start: numpy.ndarray, fitted: dict) -> list[float]:
    """Each fitting set's worst coupling miss against PySCF's exact ones:
    the case's own set, then LADDER's. Each must beat the last."""
    name = case["name"]
    mc, state = sa_state(case, mol, None, start)
    exact = couplings(mc, state, nac_sacasscf.NonAdiabaticCouplings, detail=False)
    sets = [fitted]
    for aux in LADDER:
        mc, state = sa_state(case, mol, aux, start)
        sets.append(couplings(mc, state, DFNonAdiabaticCouplings, detail=False))
    misses = [
        max(aligned_miss(nac[p]["total"], exact[p]["total"]) for p in exact)
        for nac in sets
    ]
    for coarse, fine in itertools.pairwise(misses):
        assert fine < coarse, f"{name}: the fit does not converge, {misses}"
    assert misses[-1] < LADDER_TOL, f"{name}: the largest fit misses by {misses[-1]}"
    return misses


def check_pairs(name: str, nac: dict) -> dict:
    """d_IJ = -d_JI, h_IJ = h_JI, h = HF + Lagrange, d = h/dE + d^CSF and
    the CSF-free coupling's translational sum."""
    worst = {"antisymmetry": 0.0, "pieces": 0.0, "translation": 0.0}
    for (bra, ket), r in nac.items():
        t = nac[(ket, bra)]
        worst["antisymmetry"] = max(
            worst["antisymmetry"],
            float(numpy.abs(r["total"] + t["total"]).max()),
            float(numpy.abs(r["etf"] + t["etf"]).max()),
            float(numpy.abs(r["h"] - t["h"]).max()),
        )
        worst["pieces"] = max(
            worst["pieces"],
            float(numpy.abs(r["hellmann_feynman"] + r["lagrange"] - r["h"]).max()),
            float(numpy.abs(r["etf"] + r["csf"] - r["total"]).max()),
        )
        worst["translation"] = max(
            worst["translation"], float(numpy.abs(r["etf"].sum(axis=0)).max())
        )
    assert worst["antisymmetry"] < ANTISYMMETRY_TOL, f"{name}: {worst}"
    assert worst["pieces"] < PIECES_TOL, f"{name}: {worst}"
    assert worst["translation"] < TRANSLATION_TOL, f"{name}: {worst}"
    return worst


def case_reference(case: dict) -> dict:
    name, weights = case["name"], case["weights"]
    nelecas, ncas = case["cas"]
    mol = molecule(case)
    auxbasis, aux_provenance = aux_basis(case, mol)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, f"{name}: RHF did not converge"
    start = mf.mo_coeff

    (mc, state), _ = minimum(case, mol, auxbasis, start)
    check_aux(name, mc, mol, auxbasis)
    state["ci"] = signed(name, state["ci"])
    if "from" in case:
        (recorded,) = [
            c
            for c in json.loads(OUT_DF.read_text())["cases"]
            if c["name"] == case["from"]
        ]
        miss = numpy.abs(
            numpy.subtract(recorded["casscf"]["energies"], state["energies"])
        ).max()
        assert miss < ROOT_AGREEMENT, f"{name}: off casscf_df.json by {miss}"
        assert abs(
            recorded["casscf"]["energy"] - numpy.dot(weights, state["energies"])
        ) < (CONVERGED_AGREEMENT), f"{name}: off casscf_df.json's weighted energy"

    nac = couplings(mc, state, DFNonAdiabaticCouplings, detail=True)
    rerun = 0.0
    for (bra, ket), record in nac.items():
        fresh = DFNonAdiabaticCouplings(mc, state=(bra, ket))
        fresh.verbose = 0
        fresh.conv_rtol, fresh.conv_atol = LAGRANGE_RTOL, LAGRANGE_ATOL
        rerun = max(rerun, float(numpy.abs(fresh.kernel() - record["total"]).max()))
    assert rerun < RERUN_TOL, f"{name}: PySCF's own default solve by {rerun}"
    checks = check_pairs(name, nac)
    fd = finite_difference(case, mol, state, auxbasis, nac)
    misses = ladder(case, mol, start, nac)

    nmo = state["mo"].shape[1]
    mask = mc.uniq_var_indices(nmo, mc.ncore, mc.ncas, mc.frozen)
    pairs = numpy.argwhere(mask)
    assert all(p > q for p, q in pairs)
    records = []
    for (bra, ket), r in sorted(nac.items()):
        records.append(
            {
                "bra": bra,
                "ket": ket,
                "energy_gap": float(state["energies"][ket] - state["energies"][bra]),
                "total": rows(r["total"]),
                "etf": rows(r["etf"]),
                "csf": rows(r["csf"]),
                "h": rows(r["h"]),
                "hellmann_feynman": rows(r["hellmann_feynman"]),
                "lagrange": rows(r["lagrange"]),
                "rdm": {
                    "trans_1": matrix(r["rdm"]["trans_1"]),
                    "trans_2": r["rdm"]["trans_2"].ravel().tolist(),
                    "sym_1": matrix(r["rdm"]["sym_1"]),
                    "sym_2": r["rdm"]["sym_2"].ravel().tolist(),
                },
                "b": {
                    "orbital": [float(x) for x in r["b"]["orbital"]],
                    "ci_norm": r["b"]["ci_norm"],
                },
                "z": {
                    "orbital": [float(x) for x in r["z"]["orbital"]],
                    "ci": [matrix(z) for z in r["z"]["ci"]],
                    "residual": r["z"]["residual"],
                },
            }
        )
    return {
        "name": name,
        "from": case.get("from"),
        "basis": case["basis"],
        "aux_basis": aux_provenance,
        "charge": 0,
        "atoms": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in case["atoms"]
        ],
        "electrons": int(mol.nelectron),
        "nao": int(mol.nao),
        "nmo": int(nmo),
        "ncore": int(mc.ncore),
        "ncas": ncas,
        "nelecas": nelecas,
        "weights": weights,
        "energies": [float(e) for e in state["energies"]],
        "energy": float(numpy.dot(weights, state["energies"])),
        "orbitals": {
            "core_energies": [float(e) for e in state["core_energies"]],
            "virtual_energies": [float(e) for e in state["virtual_energies"]],
            "natural_occupations": [float(n) for n in state["natural_occupations"]],
        },
        "ci": [matrix(c) for c in state["ci"]],
        "pairs": [[int(p), int(q)] for p, q in pairs],
        "couplings": records,
        "checks": {
            "rerun": rerun,
            "antisymmetry": checks["antisymmetry"],
            "pieces": checks["pieces"],
            "etf_translation": checks["translation"],
            "finite_difference": fd,
            "aux_ladder": misses,
        },
    }


def main() -> None:
    lib.num_threads(1)
    doc = {
        "description": "PySCF analytic nonadiabatic couplings of the "
        "density-fitted state-averaged CASSCF (pyscf.nac.sacasscf over "
        "df.grad.sacasscf), over this tree's data/basis and data/aux_basis "
        "files, each case's CASSCF and its _scf sharing one fit. Per ordered "
        "pair (bra I, ket J): d_IJ = <I|dJ/dR> (total), without the CSF term "
        "(etf), the CSF term (csf), h_IJ, its Hellmann-Feynman and Lagrange "
        "halves, the transition RDMs, the orbital gradient b and the "
        "multipliers z. Hartree, bohr; row A, column k is d/dR_Ak",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "command": COMMAND,
            "threads": 1,
        },
        "cases": [case_reference(case) for case in CASES],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")
    for case in doc["cases"]:
        print(case["name"], json.dumps(case["checks"]))


if __name__ == "__main__":
    main()
