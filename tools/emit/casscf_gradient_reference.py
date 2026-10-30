#!/usr/bin/env python3
"""Write the reference data the DF-CASSCF nuclear-gradient tests read.

For each case below, the density-fitted CASSCF of casscf_reference.py's
casscf_df.json (the same molecule, start, active space and fitting set,
converged by mc1step at conv_tol 1e-12, gated against mc2step and polished to
|g| < 1e-10 as there) is differentiated by PySCF: `df.grad.casscf` for a
single state, `df.grad.sacasscf` for a state average, at **every** root. Every
gradient is n_atom x 3, G[A][k] = dE/dR_Ak in hartree/bohr, atoms in the
order written.

**One fitting set throughout.** `mcscf.CASSCF(mf, ...).density_fit(...)` over
an exact RHF fits the CASSCF's energy, but PySCF's DF gradients build their
generalized Fock with `mc._scf.get_jk`, which is then exact: the gradient
would mix an exact W into a fitted energy's derivative. So the CASSCF is
built over `scf.RHF(mol).density_fit(auxbasis=...)` (carrying the exact RHF's
orbitals, which are only the start) and shares its `with_df`, so every
two-electron quantity, energy and derivative alike, is over this tree's
data/aux_basis file. The tool checks the energy against casscf_df.json's and
each gradient against central differences of the fitted energy.

**The orbitals the multipliers are written in.** After the polish, the core
and virtual orbitals are the eigenvectors of the state-averaged Fock in their
space and the active ones the state-averaged natural orbitals, each sorted
(`mc.canonicalize(..., sort=True, cas_natorb=True)`), and the CI is re-solved
densely over them. Every orbital's sign is PySCF's; a test aligns signs (or
compares magnitudes) by matching orbitals by these energies and occupations,
which the tool checks are non-degenerate.

**The split.** Each gradient is written whole and in five parts, kinetic,
potential (nuclear attraction), overlap (-W at S'), two-electron (Coulomb and
exchange, the active-space 2-RDM term and the auxiliary-basis response) and
nuclear repulsion. The parts are PySCF's own gradient code run once per part
with every other family of derivative integrals zeroed, so they sum to the
total to rounding; a state-averaged root's are over its whole Lagrangian
(Hamiltonian response plus both multiplier terms), which is also written
split as `responses`.

**State-averaged multipliers.** For root I, PySCF solves A z = -b, b the
gradient (`newton_casscf.gen_g_hop`'s, orbital then CI) of root I's energy
and A the state-averaged Hessian projected off the state-averaged space, at
conv_rtol 1e-12. Written: the orbital multipliers z_o with their (p, q)
pairs, PySCF's `uniq_var_indices` read row-major, p > q, kappa_pq = z_pq,
kappa_qp = -z_pq; and z_c, one alpha x beta determinant matrix per root r of
the average, in PySCF's `cistring` order, with the CI vectors c_r they are
paired with. The residual |A z + b| is written beside them. PySCF refuses
unequal weights in SA gradients, so the SA cases are equally weighted.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/casscf_gradient_reference.py

writes test/gtest/data/casscf_gradient/casscf_gradient.json. Re-run it to
regenerate; the file is never hand-edited. BLAS and OpenMP are pinned to one
thread and nothing run-dependent (date, host) is written, so a rerun on the
same PySCF and numpy regenerates the file byte for byte.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import contextlib  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402

# casscf_reference pins its own thread count on import; numpy's BLAS is
# already loaded at one, and main() pins PySCF's OpenMP.
from casci_reference import aux_basis  # noqa: E402
from casscf_reference import (  # noqa: E402
    CONVERGED_AGREEMENT,
    FORMALDEHYDE,
    OUT_DF,
    POLISH_CURVATURE,
    POLISH_GRAD,
    POLISH_STEPS,
    ROOT_AGREEMENT,
    casscf,
    check_aux,
    derivatives,
    molecule,
    optimize,
    spin_roots,
)
from df_reference import WATER_SKEW  # noqa: E402
from pyscf import gto, lib, scf  # noqa: E402
from pyscf.df.grad import casscf as dfcasscf_grad  # noqa: E402
from pyscf.df.grad import rhf as dfrhf_grad  # noqa: E402
from pyscf.df.grad import sacasscf as dfsacasscf_grad  # noqa: E402
from pyscf.mcscf import mc1step  # noqa: E402
from scf_reference import WATER  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "casscf_gradient" / "casscf_gradient.json"

COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/casscf_gradient_reference.py"
)

# A subset of casscf_df.json's cases ("from" names it), all cc-pVDZ over
# cc-pVDZ-JKFIT, plus water off its symmetry, so no component vanishes by
# symmetry: the same CAS(4,4) over the O-H sigma orbitals, its own start.
CASES = [
    {
        "name": "h2o_ccpvdz",
        "from": "h2o_ccpvdz",
        "atoms": WATER,
        "basis": "cc-pvdz",
        "cas": (4, 4),
        "active": [2, 3, 5, 6],
    },
    {
        "name": "h2o_skew_ccpvdz",
        "atoms": WATER_SKEW,
        "basis": "cc-pvdz",
        "cas": (4, 4),
        "active": [2, 3, 5, 6],
    },
    {
        "name": "h2co_ccpvdz_sa2",
        "from": "h2co_ccpvdz_sa2",
        "atoms": FORMALDEHYDE,
        "basis": "cc-pvdz",
        "cas": (4, 3),
        "weights": [0.5, 0.5],
    },
]

PARTS = ("kinetic", "potential", "overlap", "two_electron", "nuclear_repulsion")

# The Lagrange solve's tolerances (PySCF's defaults: rtol 1e-7, atol 1e-12).
LAGRANGE_RTOL = 1e-12
LAGRANGE_ATOL = 1e-14
# Two orbital energies, or natural occupations, closer than this make the
# canonical orbitals ill-defined.
CANONICAL_GAP = 1e-5
# The checks: PySCF's total re-run, the parts' sum, translational invariance,
# mc1step's and mc2step's minima, and the energy against casscf_df.json.
RERUN_TOL = 1e-12
PARTS_TOL = 1e-10
TRANSLATION_TOL = 1e-8
STABILITY_TOL = 1e-9
# Finite differences of the energy along random unit displacements: the
# five-point stencil's step (bohr), the directions, and the tolerance.
FD_STEP = 5e-3
FD_DIRECTIONS = 2
FD_TOL = 1e-7


def rows(gradient: numpy.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in gradient]


def matrix(m: numpy.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in numpy.atleast_2d(m)]


def fitted_scf(mol: gto.Mole, auxbasis: dict, mo: numpy.ndarray) -> scf.hf.SCF:
    """An unrun DF-RHF over `auxbasis`, carrying `mo`: the CASSCF's _scf, so
    PySCF's gradient builds its J and K over the same fit as the energy."""
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
    mf.mo_coeff = mo
    mf.mo_occ = numpy.zeros(mo.shape[1])
    mf.mo_occ[: mol.nelectron // 2] = 2
    return mf


def build(case: dict, mol: gto.Mole, auxbasis: dict, mo: numpy.ndarray):
    """casscf_reference's DF-CASSCF over a DF _scf sharing its fit."""
    nelecas, ncas = case["cas"]
    weights = case.get("weights", [1.0])
    mc = casscf(fitted_scf(mol, auxbasis, mo), ncas, nelecas, weights, auxbasis)
    assert mc.with_df is mc._scf.with_df, "the CASSCF and its _scf fit apart"
    return mc


def polish(name: str, mc, mo: numpy.ndarray, weights: list[float]) -> numpy.ndarray:
    """casscf_reference's polish: Newton steps over the exact orbital
    Hessian, the CI re-solved densely, to |g| < POLISH_GRAD."""
    for _ in range(POLISH_STEPS):
        _, _, _, g, hessian, _ = derivatives(mc, mo, weights)
        if numpy.linalg.norm(g) < POLISH_GRAD:
            break
        mo = mo @ mc1step.expmat(mc.unpack_uniq_var(-numpy.linalg.solve(hessian, g)))
    else:
        raise AssertionError(f"{name}: the polish did not reach |g| {POLISH_GRAD}")
    curvature = numpy.linalg.eigvalsh(hessian)[0]
    assert curvature > POLISH_CURVATURE, f"{name}: H's least eigenvalue {curvature}"
    return mo


def canonical(name: str, mc, mo: numpy.ndarray, weights: list[float]) -> dict:
    """The polished minimum in canonical orbitals, its CI re-solved densely."""
    _, vectors, _ = spin_roots(mc, mo, len(weights))
    ci = vectors if len(weights) > 1 else vectors[0]
    _, dm1, _, _, _, _ = derivatives(mc, mo, weights)
    mo, _, mo_energy = mc.canonicalize(
        mo, ci, mc.ao2mo(mo), sort=True, cas_natorb=True, casdm1=dm1, verbose=0
    )
    energies, vectors, _ = spin_roots(mc, mo, len(weights))
    _, dm1, _, g, _, _ = derivatives(mc, mo, weights)
    assert numpy.linalg.norm(g) < 10 * POLISH_GRAD, f"{name}: |g| {g}"
    ncore, nocc = mc.ncore, mc.ncore + mc.ncas
    occupations = numpy.diag(dm1)
    assert numpy.abs(dm1 - numpy.diag(occupations)).max() < 1e-10, name
    for values in (mo_energy[:ncore], mo_energy[nocc:], occupations):
        if len(values) > 1:
            gap = numpy.abs(numpy.diff(values)).min()
            assert gap > CANONICAL_GAP, f"{name}: a near-degenerate pair, {gap}"
    return {
        "mo": mo,
        "ci": vectors,
        "energies": energies,
        "core_energies": mo_energy[:ncore],
        "virtual_energies": mo_energy[nocc:],
        "natural_occupations": occupations,
    }


def minimum(case: dict, mol: gto.Mole, auxbasis: dict, start: numpy.ndarray):
    """The case's DF-CASSCF minimum from mc1step and from mc2step, each
    gated as casscf_reference gates them, polished and canonicalized."""
    name = case["name"]
    weights = case.get("weights", [1.0])
    found = []
    for kind in ("mc1step", "mc2step"):
        mc = build(case, mol, auxbasis, start)
        optimize(mc, getattr(mc, "kernel" if kind == "mc1step" else "mc2step"), start)
        assert mc.converged, f"{name}: {kind} did not converge"
        state = canonical(name, mc, polish(name, mc, mc.mo_coeff, weights), weights)
        weighted = float(numpy.dot(weights, state["energies"]))
        assert abs(weighted - mc.e_tot) < CONVERGED_AGREEMENT, f"{name}: {kind}"
        found.append((mc, state))
    (_, a), (_, b) = found
    assert numpy.abs(numpy.subtract(a["energies"], b["energies"])).max() < (
        CONVERGED_AGREEMENT
    ), f"{name}: mc1step and mc2step reach different minima"
    return found


@contextlib.contextmanager
def only(part: str | None):
    """PySCF's DF-CASSCF gradient code with every family of derivative
    integrals but `part`'s zeroed (all of them kept when part is None)."""
    if part is None:
        yield
        return
    hcore_generator = dfrhf_grad.Gradients.hcore_generator

    def hcore(self, mol=None):
        mol = self.mol if mol is None else mol
        full = hcore_generator(self, mol)
        ipkin = mol.intor("int1e_ipkin", comp=3)
        slices = mol.aoslice_by_atom()

        def kinetic(atom):
            _, _, p0, p1 = slices[atom]
            out = numpy.zeros_like(ipkin)
            out[:, p0:p1] = -ipkin[:, p0:p1]
            return out + out.transpose(0, 2, 1)

        if part == "kinetic":
            return kinetic
        if part == "potential":
            return lambda atom: full(atom) - kinetic(atom)
        return lambda atom: numpy.zeros_like(ipkin)

    def zeroed(function, keep: bool):
        def wrapped(*args, **kwargs):
            out = function(*args, **kwargs)
            if keep:
                return out
            if isinstance(out, (tuple, list)):
                for x in out:
                    if isinstance(x, numpy.ndarray):
                        x[...] = 0
                        if getattr(x, "aux", None) is not None:
                            x.aux[...] = 0
                return out
            return numpy.zeros_like(out)

        return wrapped

    two = part == "two_electron"
    with contextlib.ExitStack() as stack:
        enter = stack.enter_context
        enter(lib.temporary_env(dfrhf_grad.Gradients, hcore_generator=hcore))
        for cls in (dfrhf_grad.Gradients, dfcasscf_grad.Gradients):
            ovlp = zeroed(cls.get_ovlp, part == "overlap")
            jk = zeroed(cls.get_jk, two)
            enter(lib.temporary_env(cls, get_ovlp=ovlp, get_jk=jk))
        nuc = zeroed(dfcasscf_grad.Gradients.grad_nuc, part == "nuclear_repulsion")
        enter(lib.temporary_env(dfcasscf_grad.Gradients, grad_nuc=nuc))
        for module in (dfcasscf_grad, dfsacasscf_grad):
            enter(
                lib.temporary_env(
                    module,
                    grad_elec_dferi=zeroed(module.grad_elec_dferi, two),
                    grad_elec_auxresponse_dferi=zeroed(
                        module.grad_elec_auxresponse_dferi, two
                    ),
                )
            )
        yield


def split(name: str, run) -> tuple[numpy.ndarray, dict]:
    """run() under each part's mask, checked to sum to the unmasked total."""
    total = run()
    parts = {}
    for part in PARTS:
        with only(part):
            parts[part] = run()
    residual = numpy.abs(sum(parts.values()) - total).max()
    assert residual < PARTS_TOL, f"{name}: the parts miss the total by {residual}"
    return total, parts


def translation(gradient: numpy.ndarray) -> float:
    return float(numpy.abs(gradient.sum(axis=0)).max())


def state_specific(name: str, mc, state: dict) -> dict:
    mc.mo_coeff, mc.ci = state["mo"], state["ci"][0]
    mc.e_tot = state["energies"][0]
    grad = mc.nuc_grad_method()
    assert isinstance(grad, dfcasscf_grad.Gradients), type(grad)
    assert grad.auxbasis_response
    grad.verbose = 0

    def run():
        return grad.kernel(mo_coeff=state["mo"], ci=state["ci"][0])

    total, parts = split(name, run)
    fresh = mc.nuc_grad_method()
    fresh.verbose = 0
    rerun = numpy.abs(fresh.kernel() - total).max()
    assert rerun < RERUN_TOL, f"{name}: PySCF's own total differs by {rerun}"
    return {
        "roots": [{"root": 0, "total": total, "parts": parts}],
        "checks": {"rerun": float(rerun)},
    }


def state_averaged(name: str, mc, state: dict) -> dict:
    mc.mo_coeff, mc.ci = state["mo"], list(state["ci"])
    mc.fcisolver.e_states = list(state["energies"])
    nmo = state["mo"].shape[1]
    mask = mc.uniq_var_indices(nmo, mc.ncore, mc.ncas, mc.frozen)
    pairs = numpy.argwhere(mask)
    assert all(p > q for p, q in pairs)
    roots, rerun_worst = [], 0.0
    for root in range(len(state["energies"])):
        grad = mc.nuc_grad_method()
        assert isinstance(grad, dfsacasscf_grad.Gradients), type(grad)
        assert grad.auxbasis_response
        grad.verbose = 0
        grad.conv_rtol, grad.conv_atol = LAGRANGE_RTOL, LAGRANGE_ATOL
        solve = grad.solve_lagrange
        solved = []

        def cached(*args, solve=solve, solved=solved, **kwargs):
            if not solved:
                solved.append(solve(*args, **kwargs))
            return solved[0]

        grad.solve_lagrange = cached
        captured = {}
        functions = {
            "Lci_dot_dgci_dx": ("ci", dfsacasscf_grad.Lci_dot_dgci_dx),
            "Lorb_dot_dgorb_dx": ("orbital", dfsacasscf_grad.Lorb_dot_dgorb_dx),
        }

        def capture(key, function, captured=captured):
            def wrapped(*args, **kwargs):
                captured[key] = function(*args, **kwargs)
                return captured[key]

            return wrapped

        ham = grad.get_ham_response

        def hamiltonian(*args, ham=ham, captured=captured, **kwargs):
            captured["hamiltonian"] = ham(*args, **kwargs)
            return captured["hamiltonian"]

        grad.get_ham_response = hamiltonian
        with lib.temporary_env(
            dfsacasscf_grad,
            **{n: capture(k, f) for n, (k, f) in functions.items()},
        ):
            total = grad.kernel(state=root)
            responses = dict(captured)
            _, parts = split(name, lambda grad=grad, root=root: grad.kernel(state=root))
        converged, lvec, bvec, aop, _ = solved[0]
        assert converged, f"{name}: root {root}'s Lagrange solve did not converge"
        residual = float(numpy.linalg.norm(aop(lvec) + bvec))
        z_orbital, z_ci = grad.unpack_uniq_var(lvec)
        assert numpy.allclose(z_orbital[tuple(pairs.T)], lvec[: grad.ngorb], 0, 0)
        overlap = max(abs(float(numpy.vdot(z, c))) for z in z_ci for c in state["ci"])
        response = numpy.abs(sum(responses.values()) - total).max()
        assert response < PARTS_TOL, f"{name}: responses miss by {response}"

        fresh = mc.nuc_grad_method()
        fresh.verbose = 0
        fresh.conv_rtol, fresh.conv_atol = LAGRANGE_RTOL, LAGRANGE_ATOL
        rerun = numpy.abs(fresh.kernel(state=root) - total).max()
        assert rerun < 1e3 * RERUN_TOL, f"{name}: PySCF's own total by {rerun}"
        rerun_worst = max(rerun_worst, float(rerun))
        roots.append(
            {
                "root": root,
                "total": total,
                "parts": parts,
                "responses": responses,
                "z": {
                    "orbital": [float(x) for x in lvec[: grad.ngorb]],
                    "ci": [matrix(z) for z in z_ci],
                    "residual": residual,
                    "ci_overlap": overlap,
                },
            }
        )
    return {
        "roots": roots,
        "pairs": [[int(p), int(q)] for p, q in pairs],
        "checks": {"rerun": rerun_worst},
    }


def energy_at(case: dict, coords: numpy.ndarray, reference) -> list[float]:
    """Each root's fitted CASSCF energy at coords, from the reference
    minimum's orbitals carried over (Lowdin) and polished."""
    name = case["name"]
    weights = case.get("weights", [1.0])
    moved = dict(
        case,
        atoms=[(s, tuple(r)) for (s, _), r in zip(case["atoms"], coords, strict=True)],
    )
    mol = molecule(moved)
    auxbasis, _ = aux_basis(moved, mol)
    mo = reference
    overlap = mo.T @ mol.intor("int1e_ovlp") @ mo
    values, vectors = numpy.linalg.eigh(overlap)
    mo = mo @ vectors @ numpy.diag(values**-0.5) @ vectors.T
    mc = build(moved, mol, auxbasis, mo)
    optimize(mc, mc.kernel, mo)
    assert mc.converged, f"{name}: a displaced mc1step did not converge"
    energies, *_ = derivatives(mc, polish(name, mc, mc.mo_coeff, weights), weights)
    return energies


def finite_difference(case: dict, state: dict, totals: list) -> float:
    """The worst miss of each root's gradient along FD_DIRECTIONS random unit
    displacements against a five-point difference of its energy."""
    coords = numpy.array([r for _, r in case["atoms"]], dtype=float)
    rng = numpy.random.default_rng(540)
    worst = 0.0
    for _ in range(FD_DIRECTIONS):
        direction = rng.standard_normal(coords.shape)
        direction /= numpy.linalg.norm(direction)
        e = {
            k: numpy.array(
                energy_at(case, coords + k * FD_STEP * direction, state["mo"])
            )
            for k in (-2, -1, 1, 2)
        }
        slope = (e[-2] - 8 * e[-1] + 8 * e[1] - e[2]) / (12 * FD_STEP)
        for total, fd in zip(totals, slope, strict=True):
            miss = abs(float(numpy.sum(total * direction)) - fd)
            assert miss < FD_TOL, f"{case['name']}: off a finite difference by {miss}"
            worst = max(worst, miss)
    return worst


def case_reference(case: dict) -> dict:
    name = case["name"]
    weights = case.get("weights", [1.0])
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
    if "active" in case:
        start = casscf(mf, ncas, nelecas, [1.0]).sort_mo(case["active"], base=0)

    (mc, state), (other_mc, other) = minimum(case, mol, auxbasis, start)
    check_aux(name, mc, mol, auxbasis)
    ncore = mc.ncore
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

    differentiate = state_specific if len(weights) == 1 else state_averaged
    out = differentiate(name, mc, state)
    again = differentiate(name, other_mc, other)
    stability = max(
        numpy.abs(a["total"] - b["total"]).max()
        for a, b in zip(out["roots"], again["roots"], strict=True)
    )
    assert stability < STABILITY_TOL, f"{name}: mc1step and mc2step by {stability}"
    worst_translation = max(translation(root["total"]) for root in out["roots"])
    assert worst_translation < TRANSLATION_TOL, f"{name}: {worst_translation}"
    worst_parts = max(
        numpy.abs(sum(root["parts"].values()) - root["total"]).max()
        for root in out["roots"]
    )
    fd = finite_difference(case, state, [root["total"] for root in out["roots"]])

    for root in out["roots"]:
        root["energy"] = float(state["energies"][root["root"]])
        root["total"] = rows(root["total"])
        root["parts"] = {key: rows(value) for key, value in root["parts"].items()}
        if "responses" in root:
            root["responses"] = {
                key: rows(value) for key, value in root["responses"].items()
            }
    record = {
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
        "nmo": int(state["mo"].shape[1]),
        "ncore": int(ncore),
        "ncas": ncas,
        "nelecas": nelecas,
        "active": case.get("active"),
        "weights": weights,
        "energies": [float(e) for e in state["energies"]],
        "energy": float(numpy.dot(weights, state["energies"])),
        "orbitals": {
            "core_energies": [float(e) for e in state["core_energies"]],
            "virtual_energies": [float(e) for e in state["virtual_energies"]],
            "natural_occupations": [float(n) for n in state["natural_occupations"]],
        },
        "ci": [matrix(c) for c in state["ci"]],
    }
    if "pairs" in out:
        record["pairs"] = out["pairs"]
    record["roots"] = out["roots"]
    record["checks"] = {
        "rerun": out["checks"]["rerun"],
        "parts_residual": float(worst_parts),
        "translation": worst_translation,
        "mc1step_mc2step": float(stability),
        "finite_difference": fd,
    }
    return record


def main() -> None:
    lib.num_threads(1)
    doc = {
        "description": "PySCF analytic nuclear gradients of the "
        "density-fitted CASSCF (df.grad.casscf; df.grad.sacasscf at every "
        "root), over this tree's data/basis and data/aux_basis files, each "
        "case's CASSCF and its _scf sharing one fit. Each root's total and "
        "its parts (kinetic, potential, overlap, two_electron, "
        "nuclear_repulsion); for a state average, its responses "
        "(hamiltonian, orbital, ci) and multipliers z (orbital over 'pairs', "
        "ci per root of the average over 'ci'). Hartree, bohr; G[A][k] = "
        "dE/dR_Ak",
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
