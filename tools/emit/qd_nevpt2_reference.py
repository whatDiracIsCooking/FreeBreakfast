#!/usr/bin/env python3
"""Write the reference data the QD-NEVPT2 tests read: SC-QD-NEVPT2.

Neither PySCF (2.14) nor block2 (0.5.4) implements quasi-degenerate NEVPT2,
so this is this tree's own, built over PySCF's `mrpt.nevpt2` class
arithmetic and `dfnevpt2`'s fitted integrals, and checked two ways below.

**The states.** Per case, a density-fitted state-averaged CASSCF over this
tree's data/basis and data/aux_basis files, then a multi-root DFCASCI over
its orbitals, its CI vectors replaced by the dense singlets of the same
Hamiltonian and signed by casscf_nac_reference.py's rule (each vector's
first coefficient within 1e-8 of its largest is positive). The core and
virtual orbitals are canonicalized over the **state-averaged** Fock (the
case's weights; what `CasscfOutput::mo_energies` holds) and shared by every
state; the active orbitals are not rotated, so the CI vectors stand. The
existing cases start, polish and gate exactly as nevpt2_reference.py does
(mc1step, Newton steps to |g| < 1e-10, the energies checked against the
file the case comes from); the two tiny brute-force cases are mc1step DF
SA-CASSCFs, converged but not polished, since nothing they check needs a
stationary point. The cases and why each is there are at CASES.

**What is computed.** For every ordered pair (I, J) of a case's roots:

- the transition dm1, dm2, dm3 from `fci.rdm.make_dm123('FCI3pdm_kern_sf',
  ci_I, ci_J, ...)`, raw (not reordered): dm1[p, q] = <I|E_pq|J>,
  dm2[p, q, r, s] = <I|E_pq E_rs|J>, dm3 likewise, so I is the bra. dm1 is
  the transpose of make_dm123's own, which at I != J is <I|E_qp|J> at
  [p, q] (trans_rdm1's convention) while its dm2 and dm3 are not;
- the transition f3ca and f3ac: `NEVPTkern_cedf_aedf` /
  `NEVPTkern_aedf_ecdf`'s contractions of the transition 4-RDM
  (`make_dm1234('FCI4pdm_kern_sf', ci_I, ci_J)`) with h2e, as the einsums
  `make_a16` documents them (F3CA, F3AC below; at I = J they are checked
  against PySCF's `_contract4pdm`);
- per class k and per perturber t of it, PySCF's own tuples, the transition
  norm T^t_IJ = <I|V P_t V|J>: the class's norm einsums over the transition
  RDMs, with two changes, both found by the brute force below. Their
  zero-body terms (the 2 delta of hdm1 and dm1_h, the delta delta of hdm2,
  Sir's h1e_v h1e_v and the whole of Sijrs) are <I|J> = delta_IJ times a
  number. And Sr's, Si's and Sir's one-electron cross terms fold two
  Hermitian-conjugate terms into one (PySCF's "* 2.0"), which holds only at
  I = J, so T^t_IJ is the mean of the einsums over (I, J) and over (J, I);
  the other classes' einsums are already symmetric in I and J;
- the coupling <I|V|Psi_J^(1)>_k = -sum_t T^t_IJ / (Delta_t + H^J_t / N^J_t),
  J's own norms N^J_t, Dyall energies H^J_t and orbital-energy gaps Delta_t,
  over the t whose N^J_t passes PySCF's `_norm_to_energy` guard (|N| > 1e-14;
  Sijrs unguarded, as there);
- H_eff[I, J] = E_I delta_IJ + sum_k <I|V|Psi_J^(1)>_k, unsymmetrized (each
  column carries its ket's denominators) and 1/2 (H + H^T), whose eigenvalues
  and eigenvectors are the SC-QD-NEVPT2 energies and states.

**Validation**, in this script, every case:

1. Each diagonal element, per class (norm and energy), equals PySCF's
   `mrpt.NEVPT(casci, root=I)` with `canonicalized=True` on the
   SA-canonical orbitals, density-fitted (`dfnevpt2._ERIS`), to DIAGONAL.
2. Every transition RDM, dm1 to dm4, equals <I|E E ...|J> built by applying
   the spin-summed excitation operators to the CI vectors, to RDM_TOL.

And on the brute-force cases (all orbitals, the full determinant space):
the perturbers P_t V|J> are built explicitly as the full-space H|J>
restricted to each external occupation pattern t, their norms, their
overlaps <I|V P_t V|J> and their Dyall energies (H_D applied in the full
space) give every element of every class's T and coupling, and of H_eff,
to BRUTE_FORCE.

Every array is written as a '<f8' Fortran-order .npy (the layout
`src/io/npy` reads), test/gtest/data/qd_nevpt2/<case>/<array>.npy, and the
scalars, provenance and achieved validation errors to qd_nevpt2.json.
qd_nevpt2/README.md has the layouts and conventions. BLAS and OpenMP run on
one thread, and nothing run-dependent is written, so a rerun is
byte-identical:

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/qd_nevpt2_reference.py

writes test/gtest/data/qd_nevpt2/. Never hand-edited.

The case setup (`State`), the transition RDMs and the integral blocks are
method-independent: a PC-QD-NEVPT2 part reads them as the SC part does.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import copy  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import shutil  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from pyscf import df, fci, gto, lib, mcscf, mrpt, scf  # noqa: E402
from pyscf.mcscf.df import _DFCAS  # noqa: E402
from pyscf.mrpt import dfnevpt2, nevpt2  # noqa: E402

# isort: split
# The sibling tools pin their own thread counts on import; numpy is loaded
# by then, and main() pins PySCF's OpenMP.
from casci_reference import aux_basis  # noqa: E402
from casscf_nac_reference import CASES as NAC_CASES  # noqa: E402
from casscf_nac_reference import OUT as NAC_OUT  # noqa: E402
from casscf_nac_reference import signed  # noqa: E402
from casscf_reference import CASES as CASSCF_CASES  # noqa: E402
from casscf_reference import (  # noqa: E402
    OUT_DF,
    casscf,
    check_aux,
    molecule,
    optimize,
    spin_roots,
)
from nevpt2_reference import CLASSES, DEGENERATE, Recorder, polished  # noqa: E402
from scf_reference import pyscf_basis  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "qd_nevpt2"
COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/qd_nevpt2_reference.py"
)

# The polished CASSCF against its source file: each root to ROOT_AGREEMENT
# and the weighted energy to CASSCF_AGREEMENT (hartree), as
# casscf_nac_reference.py holds its own.
CASSCF_AGREEMENT = 1e-9
ROOT_AGREEMENT = 1e-6
# The Davidson CASCI, the dense CI and the CASSCF energies against each
# other (hartree); the dense vectors against the Davidson ones.
CASCI_AGREEMENT = 1e-10
SAME_STATE = 1e-8
# Validation 1: the diagonal, per class norm and energy, against PySCF's.
DIAGONAL = 1e-10
# Validation 2: the brute-force full-space H_eff, every element of every
# class's T and coupling and of H_eff (hartree).
BRUTE_FORCE = 1e-9
# The transition RDMs against the explicit operator products; the einsum
# f3ca/f3ac against PySCF's `_contract4pdm` at I = J; the fitted B blocks
# against dfnevpt2's integrals.
RDM_TOL = 1e-12
F3_TOL = 1e-12
B_TOL = 1e-12

# check-added-large-files' --maxkb in .pre-commit-config.yaml.
LIMIT = 3072 * 1024

# The transition f3ca and f3ac from the raw 4-RDM (make_dm1234's, before
# reordering), as two of make_a16's comments give them:
#   a16 += einsum('ijka,rpqbjcik->pqrabc', h2e, dm4) == f3ac.transpose(1,2,0,4,3,5)
#   a16 -= einsum('kcij,rpqbajki->pqrabc', h2e, dm4) == f3ca.transpose(1,2,0,4,3,5)
# solved for f3ac and f3ca. h2e is the physicist block, h2e[p,q,r,s] = (pr|qs).
# (Its third, 'kbij,rpqjkiac->pqrabc' == f3ca.transpose(1,4,0,2,5,3), does
# not hold even at I = J: that term rests on a symmetry of a <c|..|c> 4-RDM.)
# The two above are checked against `_contract4pdm` at I = J below.
F3CA = "kcij,rpqbajki->rpqbac"
F3AC = "ijka,rpqbjcik->rpqbac"

# The aux files of the tiny case's orbital basis, which casci_reference's
# AUX_FILES (cc-pVDZ only) does not pair.
TINY_AUX = ("cc-pVDZ-JKFIT", "../aux_basis/cc-pVDZ-JKFIT.json")
TINY_BASES = {"6-31g": "6-31g.json", "sto-3g": "sto-3g.json"}

# Six hydrogens, three stretched and twisted H2 units, no symmetry (bohr).
# 6-31G, 12 orbitals: a core of two, CAS(2,3), seven virtuals, so every
# class has distinct pairs, and 48 400 determinants in the full space.
HEXAHYDROGEN = [
    ("H", (0.00, 0.00, 0.00)),
    ("H", (0.10, 0.05, 1.55)),
    ("H", (2.70, 0.30, 0.40)),
    ("H", (2.95, -0.20, 2.10)),
    ("H", (0.90, 2.60, 0.75)),
    ("H", (1.30, 3.05, 2.35)),
]

# Two stretched H2 "chromophores" 6 bohr apart, A (1.7 bohr) and B (1.85
# bohr, slightly tilted), a short H2 (1.4 bohr) beside A and another far from
# both, the two short ones the core (bohr). STO-3G, 8 orbitals: a core of
# two, CAS(4,4) over the chromophores' sigma and sigma*, two virtuals, 4900
# determinants in the full space. The lowest singlet excitation is local to
# either chromophore, so roots 1 and 2 are a near-degenerate pair (19 mEh
# apart), and the neighbour on A correlates them differently: their H_eff
# coupling is about a quarter of their gap, and it moves the SC-QD energies
# by 1.7 mEh from the diagonal's.
OCTAHYDROGEN = [
    ("H", (0.00, 3.00, -0.85)),
    ("H", (0.05, 3.00, 0.85)),
    ("H", (0.10, -3.00, -0.92)),
    ("H", (0.00, -3.00, 0.93)),
    ("H", (1.50, 4.50, 2.50)),
    ("H", (1.60, 4.50, 3.90)),
    ("H", (-3.40, -0.40, -4.10)),
    ("H", (-3.50, -0.20, -5.50)),
]

# The cases: two of casscf_df.json's (C2v formaldehyde, equal and unequal
# weights: its two roots are 1A1 and 1A2, so their couplings vanish by
# symmetry, a zero the C++ must reproduce), casscf_nac.json's skewed
# three-root formaldehyde (C1: no coupling vanishes; the roots are 0.11 and
# 0.19 Eh apart), and two brute-force cases, the only ones whose full
# determinant space is small enough: H6, generic, and H8, whose roots 1 and
# 2 are near-degenerate, so the off-diagonals matter. No existing DF
# SA-CASSCF case has a near-degeneracy.
CASES = [
    {"from": "casscf_df", "case": "h2co_ccpvdz_sa2"},
    {"from": "casscf_df", "case": "h2co_ccpvdz_sa2_weighted"},
    {"from": "casscf_nac", "case": "h2co_skew_ccpvdz_sa3"},
    {
        "from": None,
        "case": {
            "name": "h6_631g_sa2",
            "atoms": HEXAHYDROGEN,
            "basis": "6-31g",
            "cas": (2, 3),
            "weights": [0.5, 0.5],
        },
        "brute_force": True,
    },
    {
        "from": None,
        "case": {
            "name": "h8_sto3g_sa3",
            "atoms": OCTAHYDROGEN,
            "basis": "sto-3g",
            "cas": (4, 4),
            "weights": [1 / 3, 1 / 3, 1 / 3],
        },
        "brute_force": True,
    },
]

NCLASS = len(CLASSES)


@dataclasses.dataclass
class State:
    """One case's shared orbitals and integrals and its roots: everything
    an SC or PC part reads."""

    name: str
    record: dict
    casci: object
    mo: numpy.ndarray
    mo_energy: numpy.ndarray
    ci: list
    energies: numpy.ndarray
    weights: list
    eris: dict
    blocks: dict
    full: dict | None = None


# ---------------------------------------------------------------- the states


def tiny_molecule(case: dict) -> tuple[gto.Mole, dict, dict]:
    """The brute-force case's molecule, its fitting set and provenance."""
    symbols = {symbol for symbol, _ in case["atoms"]}
    mol = gto.M(
        atom=[[symbol, coord] for symbol, coord in case["atoms"]],
        unit="Bohr",
        basis=pyscf_basis(TINY_BASES[case["basis"]], symbols),
        charge=0,
        spin=0,
        cart=False,
        verbose=0,
    )
    name, file = TINY_AUX
    path = (ROOT / "data" / "basis" / file).resolve()
    auxbasis = pyscf_basis(file, symbols)
    provenance = {
        "name": name,
        "file": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "naux": int(df.addons.make_auxmol(mol, auxbasis).nao),
    }
    return mol, auxbasis, provenance


def source_case(entry: dict) -> tuple[dict, dict | None]:
    """The case's definition and the energies its source file records."""
    if entry["from"] is None:
        return entry["case"], None
    if entry["from"] == "casscf_df":
        case = next(c for c in CASSCF_CASES if c["name"] == entry["case"])
        doc = json.loads(OUT_DF.read_text())
        rec = next(c for c in doc["cases"] if c["name"] == entry["case"])
        return case, {"file": OUT_DF, "energies": rec["casscf"]["energies"]}
    case = next(c for c in NAC_CASES if c["name"] == entry["case"])
    doc = json.loads(NAC_OUT.read_text())
    rec = next(c for c in doc["cases"] if c["name"] == entry["case"])
    return case, {"file": NAC_OUT, "energies": rec["energies"]}


def fitted_b(with_df, mo: numpy.ndarray) -> numpy.ndarray:
    """B[L, p, q] over all orbitals, (pq|rs) = sum_L B[L,p,q] B[L,r,s]: the
    fitting object's metric-folded three-index tensor, transformed."""
    cderi = numpy.vstack([lib.unpack_tril(block) for block in with_df.loop()])
    half = numpy.einsum("Lmn,mp->Lpn", cderi, mo, optimize=True)
    return numpy.einsum("Lpn,nq->Lpq", half, mo, optimize=True)


def build_state(entry: dict) -> State:
    """The case's SA-canonical orbitals, roots and fitted integrals."""
    case, source = source_case(entry)
    name = case["name"]
    weights = case["weights"]
    nroots = len(weights)
    nelecas, ncas = case["cas"]
    if entry["from"] is None:
        mol, auxbasis, provenance = tiny_molecule(case)
    else:
        mol = molecule(case)
        auxbasis, provenance = aux_basis(case, mol)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, f"{name}: RHF did not converge"
    assert "active" not in case and "avas" not in case, name

    if source is not None:
        orbitals, casscf_energies = polished(
            name, mf, mf.mo_coeff, ncas, nelecas, weights, auxbasis
        )
        miss = numpy.abs(numpy.subtract(casscf_energies, source["energies"])).max()
        assert miss < ROOT_AGREEMENT, f"{name}: a root off its source by {miss}"
        weighted = numpy.dot(weights, casscf_energies)
        miss_w = abs(weighted - numpy.dot(weights, source["energies"]))
        assert miss_w < CASSCF_AGREEMENT, f"{name}: off its source by {miss_w}"
    else:
        mc = casscf(mf, ncas, nelecas, weights, auxbasis)
        optimize(mc, mc.kernel, mf.mo_coeff)
        assert mc.converged, f"{name}: mc1step did not converge"
        orbitals = mc.mo_coeff
        casscf_energies, _, _ = spin_roots(mc, orbitals, nroots)
        miss, miss_w = 0.0, 0.0

    # A DFCASCI over the exact RHF wrapped as a DF-RHF over the same set,
    # so NEVPT's core Fock is fitted too (nevpt2_reference.py's).
    casci = mcscf.CASCI(mf.density_fit(auxbasis=auxbasis), ncas, nelecas)
    assert isinstance(casci, _DFCAS), f"{name}: the CASCI is not fitted"
    casci.fcisolver.nroots = nroots
    casci.fcisolver.conv_tol = 1e-14
    casci.fix_spin_(ss=0)
    casci.verbose = 0
    casci.kernel(orbitals)
    check_aux(name, casci, mol, auxbasis)
    energies, vectors, following = spin_roots(casci, orbitals, nroots)
    for root, (old, new) in enumerate(zip(casci.ci, vectors, strict=True)):
        overlap = abs(numpy.vdot(old, new))
        assert overlap > 1 - SAME_STATE, f"{name} root {root}: overlap {overlap}"
    error = numpy.abs(numpy.subtract(energies, casci.e_tot)).max()
    assert error < CASCI_AGREEMENT, f"{name}: Davidson off the dense CI by {error}"
    error = numpy.abs(numpy.subtract(energies, casscf_energies)).max()
    assert error < CASCI_AGREEMENT, f"{name}: CASCI off the CASSCF by {error}"
    vectors = signed(name, vectors)

    # Core and virtuals canonicalized over the state-averaged Fock; the
    # active orbitals, and so the CI vectors, unchanged.
    ncore = casci.ncore
    nocc = ncore + ncas
    sa_dm1 = sum(
        w * casci.fcisolver.make_rdm1(v, ncas, casci.nelecas)
        for w, v in zip(weights, vectors, strict=True)
    )
    mo, _, mo_energy = casci.canonicalize(
        orbitals, ci=vectors[0], casdm1=sa_dm1, cas_natorb=False, verbose=0
    )
    assert numpy.array_equal(mo[:, ncore:nocc], orbitals[:, ncore:nocc]), name
    for space, values in (("core", mo_energy[:ncore]), ("virtual", mo_energy[nocc:])):
        gaps = numpy.diff(values)
        assert gaps.size == 0 or gaps.min() >= DEGENERATE, (
            f"{name}: two canonical {space} energies {gaps.min():.1e} apart"
        )
    again, _, _ = spin_roots(casci, mo, nroots)
    error = numpy.abs(numpy.subtract(again, energies)).max()
    assert error < CASCI_AGREEMENT, f"{name}: canonicalizing moved the roots by {error}"
    casci.mo_coeff = mo
    casci.mo_energy = mo_energy
    casci.ci = list(vectors)

    eris = dfnevpt2._ERIS(casci, mo, casci.with_df)
    blocks = integral_blocks(name, casci, mo, eris)
    record = {
        "name": name,
        "source": None if source is None else str(source["file"].relative_to(ROOT)),
        "basis": case["basis"],
        "aux_basis": provenance,
        "atoms": [[s, list(c)] for s, c in case["atoms"]],
        "ncore": ncore,
        "ncas": ncas,
        "nelecas": nelecas,
        "nvirt": int(mo.shape[1] - nocc),
        "naux": int(blocks["B_aa"].shape[0]),
        "nroots": nroots,
        "weights": weights,
        "casscf_energies": [float(e) for e in casscf_energies],
        "casscf_agreement": {"root": float(miss), "weighted": float(miss_w)},
        "casci_energies": [float(e) for e in energies],
        "next_root_gap": None if following is None else float(following - energies[-1]),
    }
    return State(
        name=name,
        record=record,
        casci=casci,
        mo=mo,
        mo_energy=mo_energy,
        ci=list(vectors),
        energies=numpy.asarray(energies),
        weights=weights,
        eris=eris,
        blocks=blocks,
    )


def integral_blocks(name: str, casci, mo: numpy.ndarray, eris: dict) -> dict:
    """The shared inputs `nevpt2::EnergyInputs` and the slabs read, as
    test/gtest/data/nevpt2/ names them, from dfnevpt2's integrals; and the
    fitted B blocks, checked against those integrals."""
    ncore, ncas = casci.ncore, casci.ncas
    nocc = ncore + ncas
    nmo = mo.shape[1]
    nvir = nmo - nocc
    c, a, v = slice(0, ncore), slice(ncore, nocc), slice(nocc, nmo)
    h1eff = numpy.asarray(eris["h1eff"])
    ppaa = numpy.asarray(eris["ppaa"]).reshape(nmo, nmo, ncas, ncas)
    papa = numpy.asarray(eris["papa"]).reshape(nmo, ncas, nmo, ncas)
    pacv = numpy.asarray(eris["pacv"]).reshape(nmo, ncas, ncore, nvir)
    cvcv = numpy.asarray(eris["cvcv"]).reshape(ncore, nvir, ncore, nvir)

    b = fitted_b(casci.with_df, mo)
    worst = max(
        numpy.abs(numpy.einsum("Lpq,Lrs->pqrs", b, b[:, a, a]) - ppaa).max(),
        numpy.abs(numpy.einsum("Lpq,Lrs->pqrs", b[:, :, a], b[:, :, a]) - papa).max(),
        numpy.abs(numpy.einsum("Lpq,Lrs->pqrs", b[:, :, a], b[:, c, v]) - pacv).max(),
        numpy.abs(numpy.einsum("Lpq,Lrs->pqrs", b[:, c, v], b[:, c, v]) - cvcv).max(),
    )
    assert worst < B_TOL, f"{name}: B blocks off dfnevpt2's integrals by {worst}"

    h2e_v_sr = ppaa[v, a].transpose(0, 2, 1, 3)
    return {
        "B_aa": b[:, a, a],
        "B_ca": b[:, c, a],
        "B_va": b[:, v, a],
        "B_cv": b[:, c, v],
        "h1e": h1eff[a, a],
        "h2e": ppaa[a, a].transpose(0, 2, 1, 3),
        "e_core": casci.mo_energy[:ncore],
        "e_virt": casci.mo_energy[nocc:],
        "h1e_v_Sr": h1eff[v, a] - numpy.einsum("mbbn->mn", h2e_v_sr),
        "h1e_v_Si": h1eff[a, c],
        "h1e_v_Sir": h1eff[v, c],
        # The external blocks each class takes from eris, exactly as
        # PySCF's eris branch slices them (not written; the slabs are B's).
        "_h2e_v_Sr": h2e_v_sr,
        "_h2e_v_Si": ppaa[a, c].transpose(0, 2, 1, 3),
        "_cvcv": cvcv,
        "_h2e_v_Sijr": pacv[c].transpose(3, 1, 2, 0),
        "_h2e_v_Srsi": pacv[v].transpose(3, 0, 2, 1),
        "_h2e_v_Srs": papa[v, :, v].transpose(0, 2, 1, 3),
        "_h2e_v_Sij": papa[c, :, c].transpose(1, 3, 0, 2),
        "_h2e_v1_Sir": ppaa[v, c].transpose(0, 2, 1, 3),
        "_h2e_v2_Sir": papa[v, :, c].transpose(0, 3, 1, 2),
    }


# ---------------------------------------------------------- transition RDMs


def excitation_operators(ncas: int, nelec) -> tuple[numpy.ndarray, numpy.ndarray]:
    """E^alpha_pq and E^beta_pq as dense string matrices: [p, q, out, in]."""
    out = []
    for n in nelec:
        link = fci.cistring.gen_linkstr_index(range(ncas), n)
        e = numpy.zeros((ncas, ncas, len(link), len(link)))
        for source, row in enumerate(link):
            for p, q, target, sign in row:
                e[p, q, target, source] += sign
        out.append(e)
    return out[0], out[1]


def explicit_rdms(bra, ket, ncas, nelec) -> list[numpy.ndarray]:
    """<bra|E_pq|ket> ... <bra|E_pq E_rs E_tu E_vw|ket>, each operator applied
    to the vectors explicitly."""
    ea, eb = excitation_operators(ncas, nelec)

    def apply(x):  # E_pq x for every p, q: [p, q, na, nb]
        return numpy.einsum("pqij,...jb->...pqib", ea, x) + numpy.einsum(
            "pqij,...aj->...pqai", eb, x
        )

    k1 = apply(ket)  # E_tu |ket>
    k2 = apply(k1)  # E_tu E_vw |ket>, [t,u,v,w] -> E_vw then E_tu: see below
    # apply() puts the new operator's indices after the old ones, so k2's
    # [v, w, t, u] is E_tu E_vw |ket>; reorder to [t, u, v, w].
    k2 = k2.transpose(2, 3, 0, 1, 4, 5)
    b1 = apply(bra)  # E_qp |bra> at [q, p]: <bra|E_pq = (E_qp |bra>)^T
    b2 = apply(b1).transpose(2, 3, 0, 1, 4, 5)  # E_sr E_qp |bra> at [s, r, q, p]
    dm1 = numpy.einsum("ab,pqab->pq", bra, k1)
    dm2 = numpy.einsum("qpab,rsab->pqrs", b1, k1)
    dm3 = numpy.einsum("srqpab,tuab->pqrstu", b2, k1)
    dm4 = numpy.einsum("srqpab,tuvwab->pqrstuvw", b2, k2)
    return [dm1, dm2, dm3, dm4]


def transition_rdms(state: State) -> tuple[dict, float, float]:
    """Per ordered pair (I, J): dm1, dm2, dm3 (bra I), f3ca and f3ac, and
    the overlap <I|J>. Returns them with the largest misses of the RDMs
    against the explicit operator products and of the einsum f3 against
    PySCF's `_contract4pdm` at I = J."""
    ncas, nelec = state.casci.ncas, state.casci.nelecas
    h2e = state.blocks["h2e"]
    aaaa = numpy.ascontiguousarray(h2e.transpose(0, 2, 1, 3))
    link = tuple(fci.cistring.gen_linkstr_index(range(ncas), n) for n in nelec)
    out, rdm_miss, f3_miss = {}, 0.0, 0.0
    n = len(state.ci)
    for i, j in itertools.product(range(n), repeat=2):
        bra = numpy.ascontiguousarray(state.ci[i])
        ket = numpy.ascontiguousarray(state.ci[j])
        dm1t, dm2, dm3 = fci.rdm.make_dm123("FCI3pdm_kern_sf", bra, ket, ncas, nelec)
        # make_dm123's dm1 is <I|E_qp|J> at [p, q] (the transpose, as
        # trans_rdm1's is), while its dm2 and dm3 are <I|E_pq E_rs ..|J>; dm1
        # is taken in dm2's convention, which the norm einsums read.
        dm1 = dm1t.T
        _, _, _, dm4 = fci.rdm.make_dm1234("FCI4pdm_kern_sf", bra, ket, ncas, nelec)
        explicit = explicit_rdms(bra, ket, ncas, nelec)
        for got, want in zip((dm1, dm2, dm3, dm4), explicit, strict=True):
            rdm_miss = max(rdm_miss, numpy.abs(got - want).max())
        f3ca = numpy.einsum(F3CA, h2e, dm4, optimize=True)
        f3ac = numpy.einsum(F3AC, h2e, dm4, optimize=True)
        if i == j:
            for got, kern in (
                (f3ca, "NEVPTkern_cedf_aedf"),
                (f3ac, "NEVPTkern_aedf_ecdf"),
            ):
                want = nevpt2._contract4pdm(kern, aaaa, ket, ncas, nelec, link)
                f3_miss = max(f3_miss, numpy.abs(got - want).max())
        out[i, j] = {
            "dm1": dm1,
            "dm2": dm2,
            "dm3": dm3,
            "f3ca": f3ca,
            "f3ac": f3ac,
            "overlap": float(numpy.vdot(bra, ket)),
        }
    assert rdm_miss < RDM_TOL, f"{state.name}: transition RDMs off by {rdm_miss}"
    assert f3_miss < F3_TOL, f"{state.name}: f3ca/f3ac off _contract4pdm by {f3_miss}"
    return out, rdm_miss, f3_miss


# ------------------------------------------------------------- SC-QD-NEVPT2


def hdm1(dm1, s):
    """make_hdm1, its constant times the overlap s."""
    return 2.0 * s * numpy.eye(dm1.shape[0]) - dm1.T


def hdm2(dm1, dm2, s):
    """make_hdm2, its delta-delta terms times the overlap s."""
    delta = numpy.eye(dm2.shape[0])
    d2 = numpy.einsum("ikjl->ijkl", dm2) - numpy.einsum("jk,il->ijkl", delta, dm1)
    return (
        numpy.einsum("klij->ijkl", d2)
        + numpy.einsum("il,kj->ijkl", delta, dm1)
        + numpy.einsum("jk,li->ijkl", delta, dm1)
        - 2.0 * numpy.einsum("ik,lj->ijkl", delta, dm1)
        - 2.0 * numpy.einsum("jl,ki->ijkl", delta, dm1)
        + s * (-2.0 * numpy.einsum("il,jk->ijkl", delta, delta))
        + s * (4.0 * numpy.einsum("ik,jl->ijkl", delta, delta))
    )


def class_norms(blk: dict, d: dict, s: float) -> dict[str, numpy.ndarray]:
    """Each class's per-perturber <bra|V P_t V|ket> over the RDMs d of
    (bra, ket) and their overlap s: PySCF's norm einsums, eris branch, the
    zero-body terms times s. At bra = ket these are PySCF's norms."""
    dm1, dm2, dm3 = d["dm1"], d["dm2"], d["dm3"]
    ncas = dm1.shape[0]
    delta = numpy.eye(ncas)
    out = {}

    v, hv = blk["_h2e_v_Sr"], blk["h1e_v_Sr"]
    out["Sr"] = (
        numpy.einsum("ipqr,rpqbac,iabc->i", v, dm3, v)
        + numpy.einsum("ipqr,rpqa,ia->i", v, dm2, hv) * 2.0
        + numpy.einsum("ip,pa,ia->i", hv, dm1, hv)
    )

    v, hv = blk["_h2e_v_Si"], blk["h1e_v_Si"]
    dm3_h = numpy.einsum("abef,cd->abcdef", dm2, delta) * 2 - dm3.transpose(
        0, 1, 3, 2, 4, 5
    )
    dm2_h = numpy.einsum("ab,cd->abcd", dm1, delta) * 2 - dm2.transpose(0, 1, 3, 2)
    dm1_h = hdm1(dm1, s)
    out["Si"] = (
        numpy.einsum("qpir,rpqbac,baic->i", v, dm3_h, v)
        + numpy.einsum("qpir,rpqa,ai->i", v, dm2_h, hv) * 2.0
        + numpy.einsum("pi,pa,ai->i", hv, dm1_h, hv)
    )

    g = blk["_cvcv"]  # g[i, a, j, b] = (ia|jb)
    out["Sijrs"] = s * (g * (2.0 * g - g.transpose(0, 3, 2, 1))).ravel()

    v = blk["_h2e_v_Sijr"]
    h1 = hdm1(dm1, s)
    norm = 2.0 * numpy.einsum("rpji,raji,pa->rji", v, v, h1) - numpy.einsum(
        "rpji,raij,pa->rji", v, v, h1
    )
    out["Sijr"] = triu_pairs(norm, axes=(1, 2))

    v = blk["_h2e_v_Srsi"]
    norm = 2.0 * numpy.einsum("rsip,rsia,pa->rsi", v, v, dm1) - numpy.einsum(
        "rsip,sria,pa->rsi", v, v, dm1
    )
    out["Srsi"] = triu_pairs(norm, axes=(0, 1))

    v = blk["_h2e_v_Srs"]
    rm2 = numpy.einsum("iljk->ijkl", dm2) - numpy.einsum("ik,jl->ijkl", dm1, delta)
    out["Srs"] = 0.5 * numpy.einsum("rsqp,rsba,pqba->rs", v, v, rm2).ravel()

    v = blk["_h2e_v_Sij"]
    out["Sij"] = (
        0.5 * numpy.einsum("qpij,baij,pqab->ij", v, v, hdm2(dm1, dm2, s)).ravel()
    )

    v1, v2, hv = blk["_h2e_v1_Sir"], blk["_h2e_v2_Sir"], blk["h1e_v_Sir"]
    out["Sir"] = (
        numpy.einsum("rpiq,raib,qpab->ir", v1, v1, dm2) * 2.0
        - numpy.einsum("rpiq,rabi,qpab->ir", v1, v2, dm2)
        - numpy.einsum("rpqi,raib,qpab->ir", v2, v1, dm2)
        + numpy.einsum("raqi,rabi,qb->ir", v2, v2, dm1) * 2.0
        - numpy.einsum("rpqi,rabi,qbap->ir", v2, v2, dm2)
        + numpy.einsum("rpqi,raai,qp->ir", v2, v2, dm1)
        + numpy.einsum("rpiq,ri,qp->ir", v1, hv, dm1) * 4.0
        - numpy.einsum("rpqi,ri,qp->ir", v2, hv, dm1) * 2.0
        + s * numpy.einsum("ri,ri->ir", hv, hv) * 2.0
    ).ravel()
    return out


def triu_pairs(x: numpy.ndarray, axes: tuple[int, int]) -> numpy.ndarray:
    """PySCF's Sijr / Srsi symmetrization over a pair of axes, x + x^T with
    the diagonal halved, kept on the upper triangle, flattened."""
    perm = list(range(3))
    perm[axes[0]], perm[axes[1]] = perm[axes[1]], perm[axes[0]]
    x = x + x.transpose(perm)
    n = x.shape[axes[0]]
    diag = numpy.diag_indices(n)
    triu = numpy.triu_indices(n)
    if axes == (1, 2):
        x[:, diag[0], diag[1]] *= 0.5
        return x[:, triu[0], triu[1]].ravel()
    x[diag] *= 0.5
    return x[triu].ravel()


def class_hams(blk: dict, d: dict) -> dict[str, numpy.ndarray]:
    """Each class's per-perturber Dyall part H_t (PySCF's `h` / `ener`) and
    gap Delta_t, over one state's own RDMs d: PySCF's arithmetic, eris branch."""
    h1e, h2e = blk["h1e"], blk["h2e"]
    dm1, dm2, dm3 = d["dm1"], d["dm2"], d["dm3"]
    ncas = dm1.shape[0]
    e_core, e_virt = blk["e_core"], blk["e_virt"]
    dms = {
        "1": dm1,
        "2": dm2,
        "3": dm3,
        "4": None,
        "f3ca": d["f3ca"],
        "f3ac": d["f3ac"],
    }
    out = {}

    v, hv = blk["_h2e_v_Sr"], blk["h1e_v_Sr"]
    a16 = nevpt2.make_a16(h1e, h2e, dms, None, ncas, None)
    a17 = nevpt2.make_a17(h1e, h2e, dm2, dm3)
    a19 = nevpt2.make_a19(h1e, h2e, dm1, dm2)
    h = (
        numpy.einsum("ipqr,pqrabc,iabc->i", v, a16, v)
        + numpy.einsum("ipqr,pqra,ia->i", v, a17, hv) * 2.0
        + numpy.einsum("ip,pa,ia->i", hv, a19, hv)
    )
    out["Sr"] = (h, e_virt)

    v, hv = blk["_h2e_v_Si"], blk["h1e_v_Si"]
    a22 = nevpt2.make_a22(h1e, h2e, dms, None, ncas, None)
    a23 = nevpt2.make_a23(h1e, h2e, dm1, dm2, dm3)
    a25 = nevpt2.make_a25(h1e, h2e, dm1, dm2)
    h = (
        numpy.einsum("qpir,pqrabc,baic->i", v, a22, v)
        + numpy.einsum("qpir,pqra,ai->i", v, a23, hv) * 2.0
        + numpy.einsum("pi,pa,ai->i", hv, a25, hv)
    )
    out["Si"] = (h, -e_core)

    nc, nv = len(e_core), len(e_virt)
    gap = (
        -e_core[:, None, None, None]
        + e_virt[None, :, None, None]
        - e_core[None, None, :, None]
        + e_virt[None, None, None, :]
    )
    out["Sijrs"] = (numpy.zeros(nc * nv * nc * nv), gap.ravel())

    v = blk["_h2e_v_Sijr"]
    h1 = nevpt2.make_hdm1(dm1)
    a3 = nevpt2.make_a3(h1e, h2e, dm1, dm2, h1)
    h = 2.0 * numpy.einsum("rpji,raji,pa->rji", v, v, a3) - numpy.einsum(
        "rpji,raij,pa->rji", v, v, a3
    )
    gap = e_virt[:, None, None] - e_core[None, :, None] - e_core[None, None, :]
    triu = numpy.triu_indices(nc)
    out["Sijr"] = (triu_pairs(h, axes=(1, 2)), gap[:, triu[0], triu[1]].ravel())

    v = blk["_h2e_v_Srsi"]
    k27 = nevpt2.make_k27(h1e, h2e, dm1, dm2)
    h = 2.0 * numpy.einsum("rsip,rsia,pa->rsi", v, v, k27) - numpy.einsum(
        "rsip,sria,pa->rsi", v, v, k27
    )
    gap = e_virt[:, None, None] + e_virt[None, :, None] - e_core[None, None, :]
    triu = numpy.triu_indices(nv)
    out["Srsi"] = (triu_pairs(h, axes=(0, 1)), gap[triu].ravel())

    v = blk["_h2e_v_Srs"]
    _, a7 = nevpt2.make_a7(h1e, h2e, dm1, dm2, dm3)
    h = 0.5 * numpy.einsum("rsqp,rsba,pqab->rs", v, v, a7)
    out["Srs"] = (h.ravel(), (e_virt[:, None] + e_virt[None, :]).ravel())

    v = blk["_h2e_v_Sij"]
    h1 = nevpt2.make_hdm1(dm1)
    h2 = nevpt2.make_hdm2(dm1, dm2)
    h3 = nevpt2.make_hdm3(dm1, dm2, dm3, h1, h2)
    a9 = nevpt2.make_a9(h1e, h2e, h1, h2, h3)
    h = 0.5 * numpy.einsum("qpij,baij,pqab->ij", v, v, a9)
    out["Sij"] = (h.ravel(), -(e_core[:, None] + e_core[None, :]).ravel())

    v1, v2 = blk["_h2e_v1_Sir"], blk["_h2e_v2_Sir"]
    a12 = nevpt2.make_a12(h1e, h2e, dm1, dm2, dm3)
    a13 = nevpt2.make_a13(h1e, h2e, dm1, dm2, dm3)
    h = (
        numpy.einsum("rpiq,raib,pqab->ir", v1, v1, a12) * 2.0
        - numpy.einsum("rpiq,rabi,pqab->ir", v1, v2, a12)
        - numpy.einsum("rpqi,raib,pqab->ir", v2, v1, a12)
        + numpy.einsum("rpqi,rabi,pqab->ir", v2, v2, a13)
    )
    out["Sir"] = (h.ravel(), (e_virt[None, :] - e_core[:, None]).ravel())
    return out


def coupling(t: numpy.ndarray, n: numpy.ndarray, h, gap, guarded: bool) -> float:
    """-sum_t T_t / (Delta_t + H_t / N_t) over the ket's perturbers whose norm
    passes `_norm_to_energy`'s guard (Sijrs has none, and no H)."""
    if not guarded:
        return float(-(t / gap).sum())
    idx = numpy.abs(n) > nevpt2.NUMERICAL_ZERO
    return float(-(t[idx] / (gap[idx] + h[idx] / n[idx])).sum())


def sc_qd(state: State, rdms: dict) -> tuple[dict, numpy.ndarray]:
    """SC-QD-NEVPT2 over the case: per class the transition norms and the
    couplings, and H_eff, its symmetrization and their spectrum."""
    n = len(state.ci)
    t_norms = numpy.zeros((n, n, NCLASS))
    couplings = numpy.zeros((n, n, NCLASS))
    diag_norms = numpy.zeros((n, NCLASS))
    diag_h = {}
    for j in range(n):
        own = class_norms(state.blocks, rdms[j, j], 1.0)
        hams = class_hams(state.blocks, rdms[j, j])
        diag_h[j] = (own, hams)
        for k, key in enumerate(CLASSES):
            diag_norms[j, k] = own[key].sum()
    asymmetry = numpy.zeros(NCLASS)
    for i, j in itertools.product(range(n), repeat=2):
        s = 1.0 if i == j else 0.0
        forward = class_norms(state.blocks, rdms[i, j], s)
        backward = class_norms(state.blocks, rdms[j, i], s)
        own, hams = diag_h[j]
        for k, key in enumerate(CLASSES):
            # Sr's, Si's and Sir's one-electron cross terms (PySCF's "* 2.0"
            # and "* 4.0 ... * 2.0") are two Hermitian-conjugate terms
            # folded into one, which holds only at I = J: per perturber,
            # <I|V P_t V|J> is the mean of the einsums over (I, J) and (J, I).
            # The other classes' einsums are already symmetric.
            asymmetry[k] = max(
                asymmetry[k], numpy.abs(forward[key] - backward[key]).max()
            )
            norms = 0.5 * (forward[key] + backward[key])
            h, gap = hams[key]
            t_norms[i, j, k] = norms.sum()
            couplings[i, j, k] = coupling(norms, own[key], h, gap, key != "Sijrs")
    heff2 = couplings.sum(axis=2)
    heff = numpy.diag(state.energies) + heff2
    heff_sym = 0.5 * (heff + heff.T)
    values, vectors = numpy.linalg.eigh(heff_sym)
    for col in range(n):
        if vectors[numpy.argmax(numpy.abs(vectors[:, col])), col] < 0:
            vectors[:, col] *= -1
    return {
        "t_norms": t_norms,
        "couplings": couplings,
        "class_norms": diag_norms,
        "class_energies": couplings[range(n), range(n)],
        "heff2": heff2,
        "heff": heff,
        "heff_sym": heff_sym,
        "eigenvalues": values,
        "eigenvectors": vectors,
    }, asymmetry


def pyscf_diagonal(state: State) -> tuple[numpy.ndarray, numpy.ndarray]:
    """PySCF's own SC-NEVPT2 per root on the SA-canonical orbitals,
    `canonicalized=True`, density-fitted: (norms, energies), each (n, 8)."""
    n = len(state.ci)
    norms = numpy.zeros((n, NCLASS))
    energies = numpy.zeros((n, NCLASS))
    for root in range(n):
        state.casci.ci = copy.deepcopy(state.ci)
        pt = mrpt.NEVPT(state.casci, root=root).set(canonicalized=True)
        pt.verbose = 0
        pt.mo_coeff = state.mo
        pt.mo_energy = state.mo_energy
        with Recorder() as recorder:
            pt.kernel()
        assert recorder.df_eris == 1, f"{state.name}: NEVPT did not take its DF branch"
        for k, key in enumerate(CLASSES):
            norms[root, k], energies[root, k] = recorder.classes[key]
    state.casci.ci = list(state.ci)
    return norms, energies


# ------------------------------------------------------------- brute force


def brute_force(state: State, sc: dict) -> float:
    """Validation 2: every element of every class's T and coupling, and of
    H_eff, from the explicit perturbers in the full determinant space.

    V|J> restricted to an external occupation pattern t is H|J> restricted to
    it (H_D keeps every pattern, and |J> is the internal one). The Dyall
    Hamiltonian is sum_c e_c n_c + sum_r e_r n_r + H_cas, H_cas over h1eff's
    active block and (aa|aa), in the full space. Returns the largest miss."""
    casci, mo = state.casci, state.mo
    ncore, ncas = casci.ncore, casci.ncas
    nocc = ncore + ncas
    nmo = mo.shape[1]
    nelec = casci.mol.nelec
    b = fitted_b(casci.with_df, mo)
    eri = numpy.einsum("Lpq,Lrs->pqrs", b, b)
    h1 = mo.T @ casci.get_hcore() @ mo
    h_full = fci.direct_spin1.absorb_h1e(h1, eri, nmo, nelec, 0.5)

    a = slice(ncore, nocc)
    h1d = numpy.zeros((nmo, nmo))
    h1d[range(ncore), range(ncore)] = state.mo_energy[:ncore]
    h1d[range(nocc, nmo), range(nocc, nmo)] = state.mo_energy[nocc:]
    h1d[a, a] = state.blocks["h1e"]
    erid = numpy.zeros_like(eri)
    erid[a, a, a, a] = eri[a, a, a, a]
    h_dyall = fci.direct_spin1.absorb_h1e(h1d, erid, nmo, nelec, 0.5)

    # The core h1eff the CASCI and NEVPT read must be this H's.
    core = 2.0 * numpy.einsum("ppqr->qr", eri[:ncore, :ncore]) - numpy.einsum(
        "pqpr->qr", eri[:ncore, :, :ncore]
    )
    miss_h1 = numpy.abs((h1 + core)[a, a] - state.blocks["h1e"]).max()
    assert miss_h1 < B_TOL, f"{state.name}: h1eff off the full H by {miss_h1}"

    strings = [fci.cistring.make_strings(range(nmo), n) for n in nelec]
    cas_strings = [fci.cistring.make_strings(range(ncas), n) for n in casci.nelecas]
    core_bits = (1 << ncore) - 1
    embed = [
        fci.cistring.strs2addr(nmo, n, core_bits | (s << ncore))
        for n, s in zip(nelec, cas_strings, strict=True)
    ]
    occ = [((s[:, None] >> numpy.arange(nmo)) & 1) for s in strings]
    ext = list(range(ncore)) + list(range(nocc, nmo))
    digits = 3 ** numpy.arange(len(ext))
    key = (occ[0][:, ext] @ digits)[:, None] + (occ[1][:, ext] @ digits)[None, :]
    holes = 2 * ncore - (occ[0][:, :ncore].sum(1)[:, None] + occ[1][:, :ncore].sum(1))
    parts = occ[0][:, nocc:].sum(1)[:, None] + occ[1][:, nocc:].sum(1)
    kinds = {
        "Sr": (0, 1),
        "Si": (1, 0),
        "Sijrs": (2, 2),
        "Sijr": (2, 1),
        "Srsi": (1, 2),
        "Srs": (0, 2),
        "Sij": (2, 0),
        "Sir": (1, 1),
    }
    assert list(kinds) == list(CLASSES)

    n = len(state.ci)
    na, nb = len(strings[0]), len(strings[1])
    hv, hdv, e0 = [], [], []
    for j in range(n):
        full = numpy.zeros((na, nb))
        full[numpy.ix_(embed[0], embed[1])] = state.ci[j]
        e_full = numpy.vdot(
            full, fci.direct_spin1.contract_2e(h_full, full, nmo, nelec)
        )
        miss = abs(e_full + casci.energy_nuc() - state.energies[j])
        assert miss < CASCI_AGREEMENT, f"{state.name}: <J|H|J> off E_J by {miss}"
        e0.append(
            numpy.vdot(full, fci.direct_spin1.contract_2e(h_dyall, full, nmo, nelec))
        )
        x = fci.direct_spin1.contract_2e(h_full, full, nmo, nelec)
        outside = (holes > 2) | (parts > 2)
        assert numpy.abs(x[outside]).max(initial=0.0) < 1e-12, state.name
        hv.append(x)
        hdv.append(fci.direct_spin1.contract_2e(h_dyall, x, nmo, nelec))

    worst = 0.0
    total = numpy.zeros((n, n))
    for k, (cls, (nh, npart)) in enumerate(kinds.items()):
        mask = (holes == nh) & (parts == npart)
        _, inverse = numpy.unique(key[mask], return_inverse=True)
        per = []
        for j in range(n):
            norm = numpy.bincount(inverse, weights=hv[j][mask] ** 2)
            dyall = numpy.bincount(inverse, weights=hv[j][mask] * hdv[j][mask])
            per.append((norm, dyall))
        for i, j in itertools.product(range(n), repeat=2):
            t = numpy.bincount(inverse, weights=hv[i][mask] * hv[j][mask])
            norm, dyall = per[j]
            idx = numpy.abs(norm) > nevpt2.NUMERICAL_ZERO
            gap = dyall[idx] / norm[idx] - e0[j]
            c = float(-(t[idx] / gap).sum())
            total[i, j] += c
            worst = max(
                worst,
                abs(t.sum() - sc["t_norms"][i, j, k]),
                abs(c - sc["couplings"][i, j, k]),
            )
            assert worst < BRUTE_FORCE, (
                f"{state.name} {cls} ({i}, {j}): T {t.sum()!r} vs "
                f"{sc['t_norms'][i, j, k]!r}, coupling {c!r} vs "
                f"{sc['couplings'][i, j, k]!r}"
            )
    worst = max(worst, numpy.abs(total - sc["heff2"]).max())
    assert worst < BRUTE_FORCE, f"{state.name}: H_eff off the brute force by {worst}"
    state.full = {"ndet": int(na * nb), "nmo": nmo, "nelec": list(nelec)}
    return float(worst)


# ------------------------------------------------------------------ output


def save(path: pathlib.Path, array) -> int:
    """One '<f8' Fortran-order .npy; its size in bytes."""
    numpy.save(path, numpy.asfortranarray(array, dtype="<f8"), allow_pickle=False)
    return path.stat().st_size


def write_case(state: State, rdms: dict, sc: dict) -> int:
    """The case's directory: the shared inputs, the CI vectors, the per-pair
    transition RDMs and the SC answers. Returns the bytes written."""
    out = OUT / state.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    written = 0
    for key, value in state.blocks.items():
        if not key.startswith("_"):
            written += save(out / f"{key}.npy", value)
    written += save(out / "ci.npy", numpy.stack(state.ci))
    written += save(out / "casci_energies.npy", state.energies)
    written += save(out / "weights.npy", numpy.asarray(state.weights))
    for key, value in sc.items():
        written += save(out / f"{key}.npy", value)
    for (i, j), d in rdms.items():
        pair = out / f"pair_{i}_{j}"
        pair.mkdir()
        for key in ("dm1", "dm2", "dm3", "f3ca", "f3ac"):
            written += save(pair / f"{key}.npy", d[key])
    return written


def main() -> None:
    lib.num_threads(1)
    assert lib.num_threads() == 1
    OUT.mkdir(parents=True, exist_ok=True)
    records, total = [], 0
    worst = {"diagonal": 0.0, "brute_force": 0.0, "rdm": 0.0, "f3": 0.0}
    for entry in CASES:
        state = build_state(entry)
        rdms, rdm_miss, f3_miss = transition_rdms(state)
        sc, asymmetry = sc_qd(state, rdms)

        norms, energies = pyscf_diagonal(state)
        diag_miss = max(
            numpy.abs(norms - sc["class_norms"]).max(),
            numpy.abs(energies - sc["class_energies"]).max(),
        )
        assert diag_miss < DIAGONAL, f"{state.name}: diagonal off PySCF by {diag_miss}"
        brute = brute_force(state, sc) if entry.get("brute_force") else None

        n = len(state.ci)
        off = numpy.abs(sc["heff2"] - numpy.diag(numpy.diag(sc["heff2"])))
        gaps = numpy.abs(state.energies[:, None] - state.energies[None, :])
        gaps[range(n), range(n)] = numpy.inf
        record = state.record | {
            "e_sc_nevpt2": [float(x) for x in sc["class_energies"].sum(axis=1)],
            "heff": sc["heff"].tolist(),
            "eigenvalues": sc["eigenvalues"].tolist(),
            "max_off_diagonal": float(off.max()),
            "max_off_diagonal_over_gap": float((off / gaps).max()),
            "checks": {
                "diagonal_vs_pyscf": float(diag_miss),
                "brute_force": brute,
                "transition_rdms_vs_explicit": float(rdm_miss),
                "f3_vs_contract4pdm": float(f3_miss),
            },
            "norm_einsum_asymmetry": dict(
                zip(CLASSES, (float(x) for x in asymmetry), strict=True)
            ),
        }
        if state.full is not None:
            record["full_space"] = state.full
        records.append(record)
        total += write_case(state, rdms, sc)
        worst["diagonal"] = max(worst["diagonal"], diag_miss)
        worst["rdm"] = max(worst["rdm"], rdm_miss)
        worst["f3"] = max(worst["f3"], f3_miss)
        if brute is not None:
            worst["brute_force"] = max(worst["brute_force"], brute)
        print(
            f"{state.name}: diagonal vs PySCF NEVPT {diag_miss:.1e}, "
            + (f"brute-force H_eff {brute:.1e}, " if brute is not None else "")
            + f"transition RDMs {rdm_miss:.1e}, f3 {f3_miss:.1e}, "
            + f"max |H_IJ| {off.max():.2e} "
            + f"(/gap {record['max_off_diagonal_over_gap']:.2e})"
        )

    doc = {
        "description": "SC-QD-NEVPT2 (this tree's, over PySCF's mrpt.nevpt2 "
        "class arithmetic and dfnevpt2's fitted integrals) on the "
        "SA-canonical orbitals of density-fitted SA-CASSCFs over this tree's "
        "data/basis and data/aux_basis files: per case the transition "
        "RDMs, per-class transition norms and couplings, and H_eff. "
        "Arrays in <case>/*.npy; qd_nevpt2/README.md has the conventions. "
        "Hartree",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND,
        },
        "classes": list(CLASSES),
        "validation": {
            "diagonal": "each root's per-class norm and energy against PySCF's "
            "mrpt.NEVPT(casci, root=I) with canonicalized=True on the "
            "SA-canonical orbitals, density-fitted",
            "diagonal_tolerance": DIAGONAL,
            "diagonal_max_difference": worst["diagonal"],
            "brute_force": "every class's T and coupling and H_eff from the "
            "explicit perturbers in the full determinant space (all orbitals), "
            "brute-force cases only",
            "brute_force_tolerance": BRUTE_FORCE,
            "brute_force_max_difference": worst["brute_force"],
            "transition_rdms": "dm1-dm4 against explicit products of the "
            "spin-summed excitation operators on the CI vectors",
            "transition_rdm_tolerance": RDM_TOL,
            "transition_rdm_max_difference": worst["rdm"],
            "f3": "the einsum f3ca/f3ac at I = J against PySCF's _contract4pdm",
            "f3_tolerance": F3_TOL,
            "f3_max_difference": worst["f3"],
        },
        "cases": records,
    }
    (OUT / "qd_nevpt2.json").write_text(json.dumps(doc, indent=1) + "\n")
    over = [p for p in OUT.rglob("*") if p.is_file() and p.stat().st_size > LIMIT]
    assert not over, f"over the {LIMIT}-byte limit: {over}"
    print(
        f"wrote {OUT} ({total} bytes of arrays): diagonal vs PySCF "
        f"{worst['diagonal']:.1e} (tol {DIAGONAL:.0e}), brute-force H_eff "
        f"{worst['brute_force']:.1e} (tol {BRUTE_FORCE:.0e})"
    )


if __name__ == "__main__":
    main()
