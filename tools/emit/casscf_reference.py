#!/usr/bin/env python3
"""Write the reference data the CASSCF tests read.

For each case below, PySCF's closed-shell RHF (as casci_reference.py runs
it, over this tree's own basis files) is handed to PySCF's `mcscf.CASSCF` over
a core and an active set of its canonical orbitals: the frontier ones, an
explicit `active` list (0-based indices into the RHF orbitals in ascending
energy, put in place by `sort_mo`: the remaining occupied orbitals, lowest
first, are the core, then `active` in the given order, then the remaining
virtuals), or `mcscf.avas`'s pick over this tree's data/basis/minao.json. A
state-averaged case takes the lowest singlets (`fix_spin_(ss=0)`) at the
given weights. Two kinds of data are recorded.

Each case's start is one whose CASSCF landscape the tool finds clean: mc1step
and mc2step from it reach one minimum, run after run. Water's frontier
CAS(4,4) is not (its active space has to trade the 1b1 lone pair for the 1b2
O-H bond; at one root mc2step can stall 24 mEh up, and at two roots it finds
a second minimum 6 mEh up), so water starts from its sigma O-H bonds,
[2, 3, 5, 6] (1b2, 3a1, 4a1, 2b2). N2's frontier CAS(6,6) is clean at one root
but not at two (about one run in three lands 20 mEh up), and LiH's frontier
CAS(2,4) has an orbital Hessian with a zero eigenvalue at its minimum, so the
two-root cases are formaldehyde's frontier CAS(4,3): pi, n and pi*, the
ground state and the n -> pi* singlet.

**At the starting orbitals** (the CASCI orbitals, before any rotation): the
lowest singlets of the active-space Hamiltonian by a dense diagonalisation of
its M_s = 0 sector, their energies, and what PySCF's `mc1step.gen_g_hop`
gives over their weighted 1- and 2-RDMs. Its gradient g and Hessian H are
over the packed non-redundant pairs x_pq, p > q (active-core, virtual-core,
virtual-active, no internal rotation), in the expansion

    E(C exp(kappa(x))) = E + g.x + x.H.x / 2 + O(x^3),
    kappa_pq = x_pq, kappa_qp = -x_pq,

at fixed RDMs, which the tool checks by finite differences of the energy, so
g_pq = 2 (F_pq - F_qp) with F the generalized Fock. gen_g_hop's own g, h_op
and h_diag are each half of these, and are doubled here. h_diag is H's
diagonal, taken from H itself, one product per unit vector, and checked
against gen_g_hop's.

This tree's AOs differ from PySCF's in normalisation, order and sign within an
atom, so only quantities invariant under each orbital's sign are recorded, and
orbitals are matched by their energies. A degenerate set (orbital energies
within ORBITAL_DEGENERACY, in one space) may also rotate within itself, so
every recorded quantity is one invariant under that too: per pair of sets
(P, Q), sqrt(sum g_pq^2) and sum h_diag_pq over p in P and q in Q, which for
two single orbitals are |g_pq| and h_diag_pq; and over all pairs, |g|,
g.H.g and |H g|. The tool refuses a case with a degenerate set on the edge of
a space.

**Converged**: mc1step's CASSCF at conv_tol 1e-12 and conv_tol_grad 1e-7,
checked for `mc.converged` and against `mc2step` from the same orbitals: the
weighted energy to CONVERGED_AGREEMENT, and each root's, which is not
stationary, only to ROOT_AGREEMENT. Both run with the CI and
augmented-Hessian tolerances `casscf` sets (fcisolver conv_tol 1e-14 and
conv_tol_residual 1e-9, ah_start_tol 1e-10, ah_conv_tol 1e-14): at PySCF's
defaults mc1step stalls near |g| = 3e-7 and never sets `mc.converged`.

At conv_tol_grad 1e-7 the orbitals, and with them the orbital energies, are
only good to about 5e-7, run to run, and PySCF cannot go much tighter (mc2step
does not converge at 1e-8). So mc1step's orbitals are then polished, by
Newton steps over the exact orbital Hessian with the CI re-solved densely at
every step, to |g| < POLISH_GRAD; the tool checks that the polished energy
equals mc1step's to CONVERGED_AGREEMENT and that H is positive definite
there. Recorded from the polished orbitals: each root's energy, the weighted
energy, the eigenvalues of the state-averaged Fock (`mc.get_fock`) within
each of the core, active and virtual spaces, and the state-averaged natural
occupations.

**Density-fitted**: the same five cases again, from the same starts (the
exact RHF's orbitals, the same `active` or AVAS pick), with
`mcscf.CASSCF(...).density_fit(auxbasis=...)`: every two-electron quantity
of the CASSCF, the core's J and K included, over the fitted integrals, and
gated, polished and recorded exactly as above. The fitting set is this
tree's data/aux_basis/cc-pVDZ-JKFIT.json (AUX_FILES), handed to PySCF as
parsed shells as df_reference.py does: `auxbasis="cc-pvdz-jkfit"` would load
PySCF's own library copy, and a test comparing at 1e-8 would then be
comparing two fitting sets. The tool checks that the auxiliary molecule
PySCF fitted over is exactly the one the file gives. The RHF is the exact
one, as above, so the starts are the exact cases' starts; only the CASSCF
is fitted.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/casscf_reference.py [--only exact|df]

writes test/gtest/data/casscf/casscf.json (the exact cases) and
casscf_df.json (the density-fitted ones), or the one `--only` names. Re-run
it to regenerate; the files are never hand-edited. BLAS and OpenMP are
pinned to THREADS, so a run does not depend on the machine's core count; the
output is still not bit-for-bit reproducible (threaded reductions), only to
the drift the data READMEs state.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS.
THREADS = 8
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = str(THREADS)

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import platform  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from active_integrals_reference import NITROGEN  # noqa: E402
from casci_reference import WEIGHT_MARGIN  # noqa: E402
from fci_reference import dense_hamiltonian  # noqa: E402
from pyscf import ao2mo, df, fci, gto, lib, mcscf, scf  # noqa: E402
from pyscf.mcscf import avas, mc1step  # noqa: E402
from scf_reference import BASES, WATER, pyscf_basis  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "casscf" / "casscf.json"
OUT_DF = OUT.with_name("casscf_df.json")

# The fitting set of each orbital basis, under data/aux_basis (pyscf_basis
# reads from data/basis, hence the relative path): df_reference.py's pairing.
# Not imported from there, because importing it pins BLAS and OpenMP to one
# thread.
AUX_FILES = {"cc-pvdz": ("cc-pVDZ-JKFIT", "../aux_basis/cc-pVDZ-JKFIT.json")}

# Formaldehyde in the yz plane, C=O along z: C=O 1.203 A, C-H 1.101 A,
# H-C-H 116.5 degrees. Bohr.
FORMALDEHYDE = [
    ("C", (0.0, 0.0, 0.0)),
    ("O", (0.0, 0.0, 2.273)),
    ("H", (0.0, 1.770, -1.095)),
    ("H", (0.0, -1.770, -1.095)),
]

# N2's frontier CAS(6,6) is 3sigma_g, the 1pi_u and 1pi_g pairs whole, and
# 3sigma_u; formaldehyde's CAS(4,3) is 1b1 (pi), 2b2 (n) and 2b1 (pi*).
# `active` is absent for the frontier orbitals; `weights` defaults to one
# state.
CASES = [
    {
        "name": "h2o_ccpvdz",
        "atoms": WATER,
        "basis": "cc-pvdz",
        "cas": (4, 4),
        "active": [2, 3, 5, 6],
    },
    {"name": "n2_ccpvdz", "atoms": NITROGEN, "basis": "cc-pvdz", "cas": (6, 6)},
    {
        "name": "h2co_ccpvdz_sa2",
        "atoms": FORMALDEHYDE,
        "basis": "cc-pvdz",
        "cas": (4, 3),
        "weights": [0.5, 0.5],
    },
    {
        "name": "h2co_ccpvdz_sa2_weighted",
        "atoms": FORMALDEHYDE,
        "basis": "cc-pvdz",
        "cas": (4, 3),
        "weights": [0.7, 0.3],
    },
    {
        "name": "h2o_ccpvdz_avas",
        "atoms": WATER,
        "basis": "cc-pvdz",
        "avas": {"labels": ["O 2p", "H 1s"], "threshold": 0.2},
    },
]

CONV_TOL = 1e-12
CONV_TOL_GRAD = 1e-7
# Restarts of a stalled mc1step or mc2step, each of up to 50 macro iterations.
RESTARTS = 2
# The polish: |g| to reach, the Newton steps allowed, and the least
# eigenvalue H may have at the end.
POLISH_GRAD = 1e-10
POLISH_STEPS = 50
POLISH_CURVATURE = 1e-3
# Two orbital energies closer than this are one degenerate set (hartree).
ORBITAL_DEGENERACY = 1e-6
# The weighted roots must lie this far from the next singlet, so the
# state average does not depend on a rotation within a level.
ROOT_GAP = 1e-6
# A singlet is a root with <S^2> below this.
SINGLET_S2 = 1e-6
# mc1step, mc2step and the polish must agree on the weighted energy to this
# (hartree). Each root's energy is not stationary, so at conv_tol_grad 1e-7 it
# is only good to a few 1e-8 (formaldehyde's two optimizers differ by 3.4e-8);
# ROOT_AGREEMENT only catches a root that changed, or a flipped order.
CONVERGED_AGREEMENT = 1e-9
ROOT_AGREEMENT = 1e-6
# The finite-difference checks of g and H: their steps, and the tolerance
# relative to |g| and max |H|.
SLOPE_STEP = 1e-4
CURVATURE_STEP = 1e-3
FD_TOL = 1e-5
FD_DIRECTIONS = 3


def molecule(case: dict) -> gto.Mole:
    symbols = {symbol for symbol, _ in case["atoms"]}
    return gto.M(
        atom=[[symbol, coord] for symbol, coord in case["atoms"]],
        unit="Bohr",
        basis=pyscf_basis(BASES[case["basis"]], symbols),
        charge=0,
        spin=0,
        cart=False,
        verbose=0,
    )


def singlets(mc: mcscf.casci.CASCI, mo: numpy.ndarray, count: int):
    """The lowest `count` singlets over mo, by a dense diagonalisation.

    Returns their total energies and CI vectors, and the next singlet's
    energy (None when the sector has no more).
    """
    ncas, nelec = mc.ncas, mc.nelecas
    h1, e_core = mc.get_h1eff(mo)
    eri = ao2mo.restore(1, mc.get_h2eff(mo), ncas)
    h2e = fci.direct_spin1.absorb_h1e(h1, eri, ncas, nelec, 0.5)
    na = fci.cistring.num_strings(ncas, nelec[0])
    values, vectors = numpy.linalg.eigh(dense_hamiltonian(h2e, ncas, nelec))
    energies, kept = [], []
    for value, vector in zip(values, vectors.T, strict=True):
        vector = vector.reshape(na, -1)
        s2, _ = fci.spin_op.spin_square0(vector, ncas, nelec)
        if abs(s2) < SINGLET_S2:
            energies.append(float(value + e_core))
            kept.append(vector)
    assert len(energies) >= count, "too few singlets in the sector"
    following = energies[count] if len(energies) > count else None
    return energies[:count], kept[:count], following


def derivatives(mc: mcscf.mc1step.CASSCF, mo: numpy.ndarray, weights: list[float]):
    """The singlets over mo, their weighted RDMs, and g and H over those.

    Refuses a state average whose last root is degenerate with the next.
    Returns the roots' energies, the weighted 1- and 2-RDMs, g, H, and
    gen_g_hop's own (half) h_diag.
    """
    energies, vectors, following = singlets(mc, mo, len(weights))
    if following is not None:
        assert following - energies[-1] > ROOT_GAP, "the last root is degenerate"
    ncas, nelec = mc.ncas, mc.nelecas
    dm1 = numpy.zeros((ncas, ncas))
    dm2 = numpy.zeros((ncas,) * 4)
    for vector, weight in zip(vectors, weights, strict=True):
        d1, d2 = fci.direct_spin1.make_rdm12(vector, ncas, nelec)
        dm1 += weight * d1
        dm2 += weight * d2
    mc.mo_coeff = mo
    half_g, _, h_op, half_h_diag = mc1step.gen_g_hop(mc, mo, 1, dm1, dm2, mc.ao2mo(mo))
    # gen_g_hop's g and h_op are each half the derivative.
    g = 2 * half_g
    hessian = 2 * numpy.column_stack([h_op(e) for e in numpy.eye(g.size)])
    return energies, dm1, dm2, g, hessian, half_h_diag


def groups(orbital_energies: list[float], spaces: list[range]) -> list[list[int]]:
    """Each space's columns, split into sets of one orbital energy.

    Refuses a degenerate set on the edge of a space: two orbitals within
    ORBITAL_DEGENERACY in different spaces.
    """
    out = []
    for space in spaces:
        columns = sorted(space, key=lambda i: orbital_energies[i])
        for i in columns:
            if (
                out
                and out[-1][0] in space
                and (
                    orbital_energies[i] - orbital_energies[out[-1][-1]]
                    < ORBITAL_DEGENERACY
                )
            ):
                out[-1].append(i)
            else:
                out.append([i])
    for a in range(len(orbital_energies)):
        for b in range(a):
            if abs(orbital_energies[a] - orbital_energies[b]) < ORBITAL_DEGENERACY:
                same = any(a in s and b in s for s in spaces)
                assert same, f"a degenerate set spans two spaces: {b}, {a}"
    return sorted(sorted(members) for members in out)


def intermediates(case, mf, mc, mo, weights) -> dict:
    """The CASCI energies and gen_g_hop's invariants at the starting orbitals."""
    name, ncore, ncas = case["name"], mc.ncore, mc.ncas
    nmo = mo.shape[1]
    energies, dm1, dm2, g, hessian, half_h_diag = derivatives(mc, mo, weights)
    energy = float(numpy.dot(weights, energies))
    npair = g.size
    asym = numpy.abs(hessian - hessian.T).max()
    assert asym < 1e-10 * numpy.abs(hessian).max(), f"{name}: H not symmetric"
    h_diag = numpy.diag(hessian).copy()
    error = numpy.abs(h_diag - 2 * half_h_diag).max()
    assert error < 1e-8 * numpy.abs(h_diag).max(), (
        f"{name}: gen_g_hop's h_diag off H's diagonal by {error}"
    )

    def fixed_rdm_energy(x: numpy.ndarray) -> float:
        rotated = mo @ mc1step.expmat(mc.unpack_uniq_var(x))
        h1, e_core = mc.get_h1eff(rotated)
        eri = ao2mo.restore(1, mc.get_h2eff(rotated), ncas)
        return float(
            e_core
            + numpy.einsum("tu,tu", h1, dm1)
            + 0.5 * numpy.einsum("tuvw,tuvw", eri, dm2)
        )

    e0 = fixed_rdm_energy(numpy.zeros(npair))
    assert abs(e0 - energy) < 1e-10, f"{name}: RDM energy off by {e0 - energy}"
    rng = numpy.random.default_rng(244)
    for _ in range(FD_DIRECTIONS):
        x = rng.standard_normal(npair)
        x /= numpy.linalg.norm(x)
        plus = fixed_rdm_energy(SLOPE_STEP * x)
        minus = fixed_rdm_energy(-SLOPE_STEP * x)
        slope = (plus - minus) / (2 * SLOPE_STEP)
        plus = fixed_rdm_energy(CURVATURE_STEP * x)
        minus = fixed_rdm_energy(-CURVATURE_STEP * x)
        curvature = (plus + minus - 2 * e0) / CURVATURE_STEP**2
        assert abs(slope - g @ x) < FD_TOL * max(abs(g @ x), numpy.linalg.norm(g)), (
            f"{name}: g.x {g @ x} against a finite difference {slope}"
        )
        xhx = x @ hessian @ x
        assert abs(curvature - xhx) < FD_TOL * numpy.abs(hessian).max(), (
            f"{name}: x.H.x {xhx} against a finite difference {curvature}"
        )

    # The starting orbitals' energies, the RHF Fock's diagonal over them.
    fock = mo.T @ mf.get_fock() @ mo
    orbital_energies = [float(v) for v in numpy.diag(fock)]
    offdiag = numpy.abs(fock - numpy.diag(numpy.diag(fock)))
    nocc = ncore + ncas
    spaces = [range(ncore), range(ncore, nocc), range(nocc, nmo)]
    for space in spaces:
        block = offdiag[numpy.ix_(space, space)]
        assert block.size == 0 or block.max() < 1e-8, f"{name}: not canonical"
    sets = groups(orbital_energies, spaces)

    # The packed index of each pair (p, q), p > q, in gen_g_hop's order.
    mask = mc.uniq_var_indices(nmo, ncore, ncas, mc.frozen)
    index = numpy.full((nmo, nmo), -1)
    index[mask] = numpy.arange(npair)
    pairs = []
    for big in sets:
        for small in sets:
            members = [index[p, q] for p in big for q in small if index[p, q] >= 0]
            if not members:
                continue
            assert len(members) == len(big) * len(small)
            pairs.append(
                {
                    "p": big,
                    "q": small,
                    "gradient": float(numpy.linalg.norm(g[members])),
                    "hessian_diagonal": float(h_diag[members].sum()),
                }
            )
    assert sum(len(x["p"]) * len(x["q"]) for x in pairs) == npair
    hg = hessian @ g
    return {
        "orbital_energies": orbital_energies,
        "energies": energies,
        "energy": energy,
        "pairs": pairs,
        "gradient_norm": float(numpy.linalg.norm(g)),
        "g_hessian_g": float(g @ hg),
        "hessian_g_norm": float(numpy.linalg.norm(hg)),
    }


def casscf(mf, ncas, nelecas, weights, auxbasis=None) -> mcscf.mc1step.CASSCF:
    """PySCF's CASSCF at this tool's tolerances; density-fitted over the
    parsed `auxbasis` when one is given."""
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    if auxbasis is not None:
        mc = mc.density_fit(auxbasis=auxbasis)
    mc.conv_tol = CONV_TOL
    mc.conv_tol_grad = CONV_TOL_GRAD
    mc.max_cycle_macro = 50
    # At PySCF's defaults the CI (conv_tol 1e-10, so a residual near 1e-5)
    # and the augmented-Hessian steps are too loose for conv_tol_grad 1e-7:
    # mc1step stalls with |g| near 3e-7 and never sets mc.converged.
    mc.fcisolver.conv_tol = 1e-14
    mc.fcisolver.conv_tol_residual = 1e-9
    mc.ah_start_tol = 1e-10
    mc.ah_conv_tol = 1e-14
    mc.verbose = 0
    if len(weights) > 1:
        mc.fix_spin_(ss=0)
        mc = mc.state_average_(weights)
    return mc


def optimize(mc: mcscf.mc1step.CASSCF, step, mo: numpy.ndarray) -> None:
    """Run mc1step or mc2step from mo, restarting it where it stalls.

    Either can stop moving with |g| just above conv_tol_grad (an orbital
    step of zero, every macro iteration, at the converged energy); a restart
    from where it stopped resets its step guess and finishes.
    """
    step(mo)
    for _ in range(RESTARTS):
        if mc.converged:
            return
        step(mc.mo_coeff, mc.ci)


def converged(case, mf, mo, ncas, nelecas, weights, auxbasis=None) -> dict:
    """The converged CASSCF, checked against mc2step, then polished."""
    name = case["name"]
    mc = casscf(mf, ncas, nelecas, weights, auxbasis)
    optimize(mc, mc.kernel, mo)
    assert mc.converged, f"{name}: mc1step did not converge"
    states = list(mc.e_states) if len(weights) > 1 else [mc.e_tot]

    other = casscf(mf, ncas, nelecas, weights, auxbasis)
    optimize(other, other.mc2step, mo)
    assert other.converged, f"{name}: mc2step did not converge"
    disagreement = abs(other.e_tot - mc.e_tot)
    assert disagreement < CONVERGED_AGREEMENT, (
        f"{name}: mc1step and mc2step differ by {disagreement}"
    )
    other_states = list(other.e_states) if len(weights) > 1 else [other.e_tot]
    disagreement = numpy.abs(numpy.subtract(other_states, states)).max()
    assert disagreement < ROOT_AGREEMENT, (
        f"{name}: mc1step's and mc2step's roots differ by {disagreement}"
    )

    polished = mc.mo_coeff
    for _ in range(POLISH_STEPS):
        energies, dm1, _, g, hessian, _ = derivatives(mc, polished, weights)
        if numpy.linalg.norm(g) < POLISH_GRAD:
            break
        x = -numpy.linalg.solve(hessian, g)
        polished = polished @ mc1step.expmat(mc.unpack_uniq_var(x))
    else:
        raise AssertionError(f"{name}: the polish did not reach |g| {POLISH_GRAD}")
    curvature = numpy.linalg.eigvalsh(hessian)[0]
    assert curvature > POLISH_CURVATURE, f"{name}: H's least eigenvalue {curvature}"
    weighted = float(numpy.dot(weights, energies))
    disagreement = abs(weighted - mc.e_tot)
    assert disagreement < CONVERGED_AGREEMENT, (
        f"{name}: the polish moved the energy by {disagreement}"
    )
    disagreement = numpy.abs(numpy.subtract(energies, states)).max()
    assert disagreement < ROOT_AGREEMENT, (
        f"{name}: the polish moved a root by {disagreement}"
    )

    ncore = mc.ncore
    nocc = ncore + ncas
    # Without eris, get_fock builds J and K by the RHF's own (exact) get_jk;
    # a density-fitted Fock needs the fitted integrals passed in.
    eris = mc.ao2mo(polished) if auxbasis is not None else None
    fock = polished.T @ mc.get_fock(polished, casdm1=dm1, eris=eris) @ polished
    spaces = {
        "core": slice(0, ncore),
        "active": slice(ncore, nocc),
        "virtual": slice(nocc, fock.shape[0]),
    }
    orbital_energies = {
        key: [float(v) for v in numpy.linalg.eigvalsh(fock[s, s])]
        for key, s in spaces.items()
    }
    # PySCF canonicalizes core and virtual by default, at its own orbitals.
    for key in ("core", "virtual"):
        error = numpy.abs(
            numpy.sort(mc.mo_energy[spaces[key]]) - orbital_energies[key]
        ).max(initial=0)
        assert error < 1e-5, f"{name}: {key} orbital energies off by {error}"
    return {
        "energies": energies,
        "energy": weighted,
        "orbital_energies": orbital_energies,
        "natural_occupations": [float(v) for v in numpy.linalg.eigvalsh(dm1)[::-1]],
        "gradient_norm": float(numpy.linalg.norm(g)),
    }


def aux_basis(case: dict, mol: gto.Mole) -> tuple[dict, dict]:
    """The case's fitting set as PySCF shells, and its provenance."""
    name, file = AUX_FILES[case["basis"]]
    symbols = {symbol for symbol, _ in case["atoms"]}
    path = (ROOT / "data" / "basis" / file).resolve()
    provenance = {
        "name": name,
        "file": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "naux": int(df.addons.make_auxmol(mol, pyscf_basis(file, symbols)).nao),
    }
    return pyscf_basis(file, symbols), provenance


def check_aux(name: str, mc, mol: gto.Mole, auxbasis: dict) -> None:
    """Refuse a fit over any set but exactly `auxbasis`'s shells."""
    expected = df.addons.make_auxmol(mol, auxbasis)
    used = mc.with_df.auxmol
    assert used is not None, f"{name}: the CASSCF was not density-fitted"
    assert used.nao == expected.nao, f"{name}: aux size {used.nao}"
    assert numpy.array_equal(used._bas, expected._bas), f"{name}: aux shells"
    assert numpy.array_equal(used._env, expected._env), f"{name}: aux exponents"


def case_reference(case: dict, fitted: bool = False) -> dict:
    """One case's record; density-fitted over AUX_FILES's set if `fitted`."""
    name = case["name"]
    mol = molecule(case)
    auxbasis, aux_provenance = aux_basis(case, mol) if fitted else (None, None)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, f"{name}: RHF did not converge"

    weights = case.get("weights", [1.0])
    if "avas" in case:
        symbols = {symbol for symbol, _ in case["atoms"]}
        selection = avas.AVAS(
            mf,
            case["avas"]["labels"],
            threshold=case["avas"]["threshold"],
            minao=pyscf_basis("minao.json", symbols),
            verbose=0,
        )
        ncas, nelecas, mo = selection.kernel()
        ncas, nelecas = int(ncas), int(nelecas)
        projected = numpy.concatenate([selection.occ_weights, selection.vir_weights])
        margin = numpy.abs(projected - case["avas"]["threshold"]).min()
        assert margin > WEIGHT_MARGIN, f"{name}: a weight within {margin}"
    else:
        nelecas, ncas = case["cas"]
        mo = mf.mo_coeff
        if "active" in case:
            mo = casscf(mf, ncas, nelecas, [1.0]).sort_mo(case["active"], base=0)
    ncore = (mol.nelectron - nelecas) // 2

    start = casscf(mf, ncas, nelecas, [1.0], auxbasis)
    out = {
        "name": name,
        "basis": case["basis"],
        "charge": 0,
        "atoms": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in case["atoms"]
        ],
        "electrons": int(mol.nelectron),
        "nmo": int(mo.shape[1]),
        "ncas": ncas,
        "nelecas": nelecas,
        "ncore": ncore,
        "active": case.get("active"),
        "avas": case.get("avas"),
        "weights": weights,
        "rhf_energy": float(mf.e_tot),
        "mo_energies": [float(e) for e in mf.mo_energy],
        "casci": intermediates(case, mf, start, mo, weights),
        "casscf": converged(case, mf, mo, ncas, nelecas, weights, auxbasis),
    }
    if fitted:
        check_aux(name, start, mol, auxbasis)
        out["aux_basis"] = aux_provenance
    return out


def provenance() -> dict:
    return {
        "pyscf": pyscf.__version__,
        "numpy": numpy.__version__,
        "date": datetime.date.today().isoformat(),
        "host": platform.node(),
        "command": "uv run --group tools tools/emit/casscf_reference.py",
        "threads": THREADS,
    }


def exact() -> dict:
    return {
        "description": "PySCF closed-shell RHF (spherical, conv_tol 1e-14, "
        "conv_tol_grad 1e-11) over this tree's data/basis files, and PySCF "
        "mcscf.CASSCF (conv_tol 1e-12, conv_tol_grad 1e-7, the lowest "
        "singlets at the given weights, checked against mc2step, then "
        "polished to |g| < 1e-10) from its frontier orbitals, an explicit "
        "'active' set (0-based RHF orbital indices, by sort_mo) or "
        "mcscf.avas's over data/basis/minao.json: at those starting "
        "orbitals, the CASCI energies and the sign-invariant parts of "
        "mc1step.gen_g_hop's gradient and orbital Hessian (doubled, the "
        "derivatives of E) over the weighted RDMs; converged, each root's "
        "energy, the weighted energy, the state-averaged Fock's eigenvalues "
        "in each space and the state-averaged natural occupations. Hartree; "
        "coordinates in bohr",
        "provenance": provenance(),
        "cases": [case_reference(case) for case in CASES],
    }


def fitted() -> dict:
    return {
        "description": "As casscf.json's cases, from the same starts (the "
        "exact RHF's orbitals and the same 'active' or AVAS pick), with "
        "PySCF mcscf.CASSCF(...).density_fit(auxbasis=...): every "
        "two-electron quantity of the CASSCF, the core's J and K included, "
        "over the fitted integrals, gated against mc2step and polished to "
        "|g| < 1e-10 as there. The fitting set is this tree's "
        "data/aux_basis file each case's 'aux_basis' names, handed to PySCF "
        "as parsed shells, not its library copy. Hartree; coordinates in bohr",
        "provenance": provenance(),
        "cases": [case_reference(case, fitted=True) for case in CASES],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        choices=("exact", "df"),
        help="write only casscf.json (exact) or casscf_df.json (df)",
    )
    only = parser.parse_args().only
    lib.num_threads(THREADS)
    for kind, out, build in (("exact", OUT, exact), ("df", OUT_DF, fitted)):
        if only not in (None, kind):
            continue
        doc = build()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
