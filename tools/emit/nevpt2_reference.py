#!/usr/bin/env python3
"""Write the reference data the DF-NEVPT2 tests read.

For each density-fitted case of casscf_reference.py (casscf_df.json) but
those in EXCLUDED (N2: see there), from
the same start, over the same fitting set (this tree's
data/aux_basis/cc-pVDZ-JKFIT.json, parsed as that tool parses it, never
PySCF's library copy): PySCF's DF (SA-)CASSCF, mc1step polished by Newton
steps to |g| < POLISH_GRAD exactly as there, then strongly contracted
NEVPT2 per root on those orbitals, as PySCF's
examples/mrpt/41-for_state_average.py does it: a multi-root CASCI over them
(the lowest singlets, `fix_spin_(ss=0)`), then `mrpt.NEVPT(casci, root=r)`.
The CASCI's CI vectors are swapped for the dense singlets of the same
Hamiltonian (see SAME_STATE): its spin-penalised Davidson stalls short of
them, and NEVPT2 is first order in the CI.

Everything two-electron is fitted. The CASCI is built over the exact RHF's
converged orbitals wrapped as a DF-RHF (`mf.density_fit`, not re-run), so
`mcscf.CASCI` is PySCF's DFCASCI over its fitting object and NEVPT takes its
DF branch (`dfnevpt2._ERIS`, whose core Fock is the DF-RHF's get_jk too).
The tool refuses a run whose fitting set is not exactly the file's shells,
or where NEVPT did not take the DF branch (it silently falls back to exact
integrals when it judges memory short).

Per root `kernel` canonicalizes the core and virtuals over that root's own
Fock (its own 1-RDM, not the state average) and rotates the active space to
that root's natural orbitals (`cas_natorb=True`). Two PySCF facts:

- `NEVPT.kernel` writes the canonicalized CI vector back into the CASCI's
  shared `ci` list, so each root starts from a deep copy of the CASCI's CI.
- SC-NEVPT2 is invariant to the active-space rotation. The tool checks it on
  every root: the same root with core and virtuals canonicalized over its
  Fock but the active orbitals left as they are (`cas_natorb=False`), every
  class's norm and energy and the total, to INVARIANCE. The C++ step relies
  on this (it does not rotate the active space), so a miss fails emission.

PySCF has no partially contracted NEVPT2, so each root's PC-NEVPT2 is
block2's (`pyblock2.icmr.icnevpt2_full.WickICNEVPT2`), on exactly the state
the C++ step computes from: core and virtuals canonicalized over the root's
Fock, the active orbitals unrotated (`cas_natorb=False`), the same CI vector,
and PySCF's own fitted integrals (`dfnevpt2._ERIS`) injected in place of
block2's transform (block2's `eri_helper.init_eris` always transforms the
exact 4-index ERIs, and `kernel(eris=...)` raises in 0.5.4: its `teris`
timer is bound only on the build path). Three gates before it is written:
block2's SC on the same state must equal PySCF's per class to BLOCK2_SC, so
block2 sees the state the SC classes were recorded on; PC Sijrs, which has no
active index, must equal SC Sijrs to BLOCK2_SC. There is no gap gate: the
C++ step cuts its metric at pc_cut(d) and needs no gap there, since the real
modes that may run through the cut carry no energy (src/nevpt2/energy/README.md,
"PC-NEVPT2", #853); block2's own solves are its lstsq.
PC is invariant to the active rotation as SC is (each class's metric and
Dyall matrix move by one congruence), and the tool checks that too: block2's
PC again on the root's natural orbitals, to INVARIANCE.

Recorded: per case, the polished CASSCF's energies (checked against
casscf_df.json's to CASSCF_AGREEMENT, so the starts are those cases'), the
CASCI's per-root energies over the polished orbitals, and per root the eight
classes' norms and energies and their total, and block2's eight PC class
energies and their total. BLAS and OpenMP run on one
thread, and nothing run-dependent (date, host) is written, so a rerun is
byte-identical:

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/nevpt2_reference.py

writes test/gtest/data/nevpt2/nevpt2_df.json. Never hand-edited.

**Frozen core** (#859), with `--frozen-core`: the cases of FROZEN_CASES
polished and gated exactly as above, then each root's SC- and PC-NEVPT2
with its lowest n_f canonical core orbitals frozen, n_f each variant's
count or "auto" (PySCF's `chemcore`, 1s for B to Ne). Neither PySCF's
NEVPT (`NEVPT.kernel` raises NotImplementedError for a frozen `_mc`) nor
anything else here has it, so it is PySCF's own eight class functions on
the root's canonical state (core and virtuals over its Fock, the whole
core canonicalized together, active unrotated) with the frozen orbitals
sliced out of `dfnevpt2._ERIS`'s integrals and orbital energies: h1eff
is still over the whole core's density, which is what freezing means
here (BAGEL's SMITH and block2's `frozen` do the same). Gates before it
is written: the same function at n_f = 0 equals PySCF's own kernel per
class to FROZEN_ZERO; Sr and Srs, which have no core index, equal the
all-electron ones to FROZEN_ZERO; block2's SC with its own `frozen=n_f`
equals it per class to BLOCK2_SC; and block2's PC, `frozen=n_f`, is
recorded, its Sijrs gated against SC's and its active-rotation
invariance to INVARIANCE as above:

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/nevpt2_reference.py --frozen-core

writes test/gtest/data/nevpt2/nevpt2_df_frozen_core.json, and leaves
nevpt2_df.json alone.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import copy  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402
from importlib.metadata import version  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from pyblock2.icmr import eri_helper, icnevpt2_full, scnevpt2  # noqa: E402
from pyscf import fci, lib, mcscf, mrpt, scf  # noqa: E402
from pyscf.data import elements  # noqa: E402
from pyscf.mcscf import avas, mc1step  # noqa: E402
from pyscf.mcscf.df import _DFCAS  # noqa: E402
from pyscf.mrpt import dfnevpt2, nevpt2  # noqa: E402

# isort: split
# casscf_reference pins its own thread count on import; numpy and PySCF are
# loaded by then, so this module's single thread stands.
from casci_reference import WEIGHT_MARGIN  # noqa: E402
from casscf_reference import (  # noqa: E402
    CASES,
    OUT_DF,
    POLISH_CURVATURE,
    POLISH_GRAD,
    POLISH_STEPS,
    aux_basis,
    casscf,
    check_aux,
    derivatives,
    molecule,
    optimize,
    spin_roots,
)
from scf_reference import pyscf_basis  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "nevpt2" / "nevpt2_df.json"
COMMAND = "uv run --group tools tools/emit/nevpt2_reference.py"
OUT_FROZEN = ROOT / "test" / "gtest" / "data" / "nevpt2" / "nevpt2_df_frozen_core.json"

# The frozen-core cases (--frozen-core): case name -> the frozen_core values
# recorded, each as the [nevpt2] key spells it. H2O freezes its O 1s by
# "auto"; H2CO its C and O 1s, and by count 1 only the O 1s, the lowest.
FROZEN_CASES = {"h2o_ccpvdz": ["auto"], "h2co_ccpvdz_sa2": ["auto", 1]}
# The frozen function at n_f = 0 against PySCF's kernel, and the frozen Sr
# and Srs against the all-electron ones, per class (hartree): the same
# einsums over the same arrays, sliced.
FROZEN_ZERO = 1e-12

# The polished CASSCF must reproduce casscf_df.json's energies to this
# (hartree): that file's stated run-to-run tolerance.
CASSCF_AGREEMENT = 1e-9
# The multi-root CASCI over the polished orbitals, its Davidson and the
# dense CI must agree with each other and with the polish's root energies
# to this (hartree).
CASCI_AGREEMENT = 1e-10
# The CI vectors enter NEVPT2 to first order, and PySCF's spin-penalised
# Davidson (fix_spin_) stalls short of them: N2's residual stays at 8e-7
# with fcisolver.converged unset, which moves its NEVPT2 energy by 1.1e-6.
# So the CASCI's vectors are replaced by casscf_reference's dense singlets
# over the same Hamiltonian, each first matched to the Davidson root it
# replaces by |<davidson|dense>| > 1 - SAME_STATE.
SAME_STATE = 1e-8
# cas_natorb=True and False must agree on every class's norm and energy and
# on the total to this (hartree for energies).
INVARIANCE = 1e-10

# block2's SC must equal PySCF's per class, and PC Sijrs SC Sijrs, to this
# (hartree) on the same state and integrals. block2 builds Sr and Si from the
# dense 4-RDM where PySCF contracts it into f3ca and f3ac first.
BLOCK2_SC = 1e-10
# nevpt2::pc_class_energies' cut on a class's d x d metric S: its modes at or
# below pc_cut(d) of its largest are null, the larger of PC_TAU and PC_NOISE
# eps sqrt(d). Every mode round-off or the RDMs decide has measured at or
# below 7.4 eps sqrt(d), so the cut sits at least 13 times above it; there is
# no gap rule (src/nevpt2/energy/README.md, "PC-NEVPT2", #853).
PC_TAU = 1e-13
PC_NOISE = 100.0


def pc_cut(d: int) -> float:
    """The cut on a d x d metric, relative to its largest eigenvalue."""
    return max(PC_TAU, PC_NOISE * float(numpy.finfo(float).eps) * float(numpy.sqrt(d)))


# block2's sub-space keys, as PySCF names the classes.
B2_TO_CLASS = {
    "r": "Sr",
    "i": "Si",
    "ijrs": "Sijrs",
    "ijr": "Sijr",
    "rsi": "Srsi",
    "rs": "Srs",
    "ij": "Sij",
    "ir": "Sir",
}

# casscf_df.json's cases left out, with the reason. N2's canonical virtuals
# hold six exactly degenerate pi* pairs, and SC-NEVPT2 is not invariant to a
# rotation inside one: its Srs class contracts per virtual pair, so its
# energy depends on how each pair is split, which canonicalization leaves
# arbitrary. Measured on PySCF 2.14.0's own N2 DF-CASSCF(6,6) with the CI and
# integrals fixed: one pair rotated by 0.1, 0.3 and pi/4 moves Srs by
# -3.2e-8, -6.2e-7 and -4.6e-6, random relative rotations of all six by up
# to 5.0e-6, and two PySCF runs landed at opposite ends of that band. So
# PySCF's N2 value is defined only to ~5e-6, and no test can hold the C++
# step to it. The invariance check below rotates only the active space, so
# it cannot see this; DEGENERATE refuses any other such case.
EXCLUDED = {
    "n2_ccpvdz": "degenerate pi* virtual pairs (see above)",
    "n2_ccpvdz_triplet": "not a singlet: the step refuses multiplicity != 1",
    "nh2_ccpvdz_rohf": "open shell: the step refuses n_alpha != n_beta",
    "ch2_ccpvdz_rohf": "open shell: the step refuses n_alpha != n_beta",
}
# Two canonical core or two canonical virtual orbital energies of a root
# closer than this (hartree) refuse emission: the split of a degenerate pair
# is arbitrary, and SC-NEVPT2 depends on it (EXCLUDED).
DEGENERATE = 1e-6

# PySCF's SC-NEVPT2 classes, in the order kernel sums them, with the change
# in the number of active electrons each perturber carries.
CLASSES = {
    "Sr": "(-1)'",
    "Si": "(+1)'",
    "Sijrs": "(0)",
    "Sijr": "(+1)",
    "Srsi": "(-1)",
    "Srs": "(-2)",
    "Sij": "(+2)",
    "Sir": "(0)'",
}


class Recorder:
    """Wraps nevpt2's class functions and dfnevpt2._ERIS, which
    NEVPT.kernel looks up at call time, so a kernel's per-class
    (norm, energy) are kept and its use of the DF integrals is seen."""

    def __init__(self):
        self.classes: dict[str, tuple[float, float]] = {}
        self.df_eris = 0
        self.saved = {name: getattr(nevpt2, name) for name in CLASSES}
        self.saved_eris = dfnevpt2._ERIS

    def __enter__(self):
        for name, function in self.saved.items():
            setattr(nevpt2, name, self.wrap(name, function))
        saved_eris = self.saved_eris

        def eris(*args, **kwargs):
            self.df_eris += 1
            return saved_eris(*args, **kwargs)

        dfnevpt2._ERIS = eris
        return self

    def __exit__(self, *exc):
        for name, function in self.saved.items():
            setattr(nevpt2, name, function)
        dfnevpt2._ERIS = self.saved_eris

    def wrap(self, name, function):
        def recorded(*args, **kwargs):
            norm, energy = function(*args, **kwargs)
            self.classes[name] = (float(norm), float(energy))
            return norm, energy

        return recorded


def nevpt(casci, ci, root: int, natorb: bool) -> tuple[float, dict]:
    """One root's SC-NEVPT2 over a deep copy of the CASCI's CI: by
    kernel's own canonicalization (natorb) or with the active space left
    unrotated. Returns the total and {class: (norm, energy)}."""
    casci.ci = copy.deepcopy(ci)
    pt = mrpt.NEVPT(casci, root=root)
    pt.verbose = 0
    if not natorb:
        pt.mo_coeff, root_ci, pt.mo_energy = casci.canonicalize(
            casci.mo_coeff, ci=pt.load_ci(), cas_natorb=False, verbose=0
        )
        if casci.fcisolver.nroots == 1:
            pt.ci = root_ci
        else:
            pt.ci[root] = root_ci
        pt.canonicalized = True
        nocc = casci.ncore + casci.ncas
        for space, energies in (
            ("core", pt.mo_energy[: casci.ncore]),
            ("virtual", pt.mo_energy[nocc:]),
        ):
            gaps = numpy.diff(numpy.sort(energies))
            assert gaps.size == 0 or gaps.min() >= DEGENERATE, (
                f"root {root}: two canonical {space} energies {gaps.min():.1e} apart, "
                f"below DEGENERATE {DEGENERATE}: SC-NEVPT2 depends on their split"
            )
    with Recorder() as recorder:
        total = float(pt.kernel())
    assert recorder.df_eris == 1, "NEVPT did not take its DF branch"
    assert set(recorder.classes) == set(CLASSES), sorted(recorder.classes)
    parts = sum(energy for _, energy in recorder.classes.values())
    assert abs(parts - total) < 1e-12, f"the classes sum to {parts}, not {total}"
    return total, recorder.classes


def frozen_count(mol, key) -> int:
    """The orbitals a frozen_core value freezes: a count, or "auto", PySCF's
    chemical core of the molecule."""
    return int(elements.chemcore(mol)) if key == "auto" else int(key)


def frozen_classes(casci, state, nfrozen: int) -> dict[str, tuple[float, float]]:
    """PySCF's eight SC class functions on one root's canonical `state`
    with its lowest `nfrozen` core orbitals frozen: `dfnevpt2._ERIS` over
    every orbital (so h1eff is over the whole core), then those orbitals'
    rows and columns sliced out of every integral and of the orbital
    energies, as NEVPT.kernel feeds the class functions. {class: (norm,
    energy)}."""
    mo, ci, mo_energy = state
    # _contract4pdm hands the CI vector's buffer to C as it lies, and
    # canonicalize's can come back Fortran-ordered.
    ci = numpy.ascontiguousarray(ci)
    ncore, ncas, nelecas = casci.ncore, casci.ncas, casci.nelecas
    nocc = ncore + ncas
    mc = copy.copy(casci)
    mc.mo_coeff, mc.mo_energy = mo, mo_energy
    eris = dfnevpt2._ERIS(mc, mo, casci.with_df)
    f = nfrozen
    nmo = mo.shape[1]
    nvir = nmo - nocc
    ppaa = numpy.asarray(eris["ppaa"]).reshape(nmo, nmo, ncas, ncas)
    papa = numpy.asarray(eris["papa"]).reshape(nmo, ncas, nmo, ncas)
    pacv = numpy.asarray(eris["pacv"]).reshape(nmo, ncas, ncore, nvir)
    cvcv = numpy.asarray(eris["cvcv"]).reshape(ncore, nvir, ncore, nvir)
    sliced = {
        "ppaa": numpy.ascontiguousarray(ppaa[f:, f:]),
        "papa": numpy.ascontiguousarray(papa[f:, :, f:]),
        "pacv": numpy.ascontiguousarray(pacv[f:, :, f:]),
        "cvcv": numpy.ascontiguousarray(cvcv[f:, :, f:]).reshape(
            (ncore - f) * nvir, (ncore - f) * nvir
        ),
        "h1eff": numpy.asarray(eris["h1eff"])[f:, f:],
    }
    pt = types.SimpleNamespace(
        ncore=ncore - f,
        ncas=ncas,
        nelecas=nelecas,
        mo_coeff=mo[:, f:],
        mo_energy=mo_energy[f:],
        fcisolver=casci.fcisolver,
        mol=casci.mol,
        verbose=0,
    )
    dm1, dm2, dm3 = fci.rdm.make_dm123("FCI3pdm_kern_sf", ci, ci, ncas, nelecas)
    link = tuple(fci.cistring.gen_linkstr_index(range(ncas), n) for n in nelecas)
    aaaa = ppaa[ncore:nocc, ncore:nocc].copy()
    dms = {
        "1": dm1,
        "2": dm2,
        "3": dm3,
        "4": None,
        "f3ca": nevpt2._contract4pdm(
            "NEVPTkern_cedf_aedf", aaaa, ci, ncas, nelecas, link
        ),
        "f3ac": nevpt2._contract4pdm(
            "NEVPTkern_aedf_ecdf", aaaa, ci, ncas, nelecas, link
        ),
    }
    out = {
        "Sr": nevpt2.Sr(pt, ci, dms, sliced),
        "Si": nevpt2.Si(pt, ci, dms, sliced),
        "Sijrs": nevpt2.Sijrs(pt, sliced),
        "Sijr": nevpt2.Sijr(pt, dms, sliced),
        "Srsi": nevpt2.Srsi(pt, dms, sliced),
        "Srs": nevpt2.Srs(pt, dms, sliced),
        "Sij": nevpt2.Sij(pt, dms, sliced),
        "Sir": nevpt2.Sir(pt, dms, sliced),
    }
    return {key: (float(norm), float(energy)) for key, (norm, energy) in out.items()}


def canonical(casci, ci, natorb: bool):
    """One root's orbitals, CI vector and orbital energies: core and
    virtuals canonicalized over the root's own Fock, the active space
    rotated to its natural orbitals or left as it is."""
    return casci.canonicalize(casci.mo_coeff, ci=ci, cas_natorb=natorb, verbose=0)


def block2(casci, state, kernel: str, frozen: int = 0) -> dict[str, float]:
    """block2's SC (`kernel="sc"`) or PC (`"pc"`) per class on one root's
    `state` (canonical's), `canonicalized` so block2 takes the orbitals and
    energies it is handed, over PySCF's DF integrals for those orbitals.
    With `frozen`, block2's own frozen core: its lowest `frozen` orbitals
    sliced out of the integrals as its `init_eris(frozen=)` slices them
    (h1eff over the whole core), and `frozen` handed to the solver, which
    slices its orbital energies."""
    mo, ci, mo_energy = state
    mc = copy.copy(casci)
    mc.mo_coeff, mc.ci, mc.mo_energy = mo, ci, mo_energy
    e = dfnevpt2._ERIS(mc, mo, casci.with_df)
    ncore, ncas = mc.ncore, mc.ncas
    nmo = mo.shape[1]
    nvir = nmo - ncore - ncas
    f = frozen
    eris = eri_helper._ChemistsERIs()
    eris.ncore, eris.ncas, eris.nocc = ncore - f, ncas, ncore - f + ncas
    eris.h1eff = numpy.asarray(e["h1eff"])[f:, f:]
    eris.known = ["ppaa", "papa", "pacv", "cvcv"]
    eris.ppaa = numpy.asarray(e["ppaa"]).reshape(nmo, nmo, ncas, ncas)[f:, f:]
    eris.papa = numpy.asarray(e["papa"]).reshape(nmo, ncas, nmo, ncas)[f:, :, f:]
    eris.pacv = numpy.asarray(e["pacv"]).reshape(nmo, ncas, ncore, nvir)[f:, :, f:]
    eris.cvcv = numpy.asarray(e["cvcv"]).reshape(ncore, nvir, ncore, nvir)[f:, :, f:]

    init_eris = eri_helper.init_eris
    eri_helper.init_eris = lambda *args, **kwargs: eris
    try:
        cls = {"sc": scnevpt2.WickSCNEVPT2, "pc": icnevpt2_full.WickICNEVPT2}[kernel]
        ic = cls(mc, frozen=f)
        ic.verbose = 0
        ic.canonicalized = True
        ic.kernel()
    finally:
        eri_helper.init_eris = init_eris
    return {B2_TO_CLASS[k]: float(v) for k, v in ic.sub_eners.items()}


def polished(name, mf, mo, ncas, nelecas, weights, auxbasis):
    """casscf_reference's DF mc1step from mo, polished to |g| < POLISH_GRAD.
    Returns the polished orbitals and each root's energy there."""
    mc = casscf(mf, ncas, nelecas, weights, auxbasis)
    optimize(mc, mc.kernel, mo)
    assert mc.converged, f"{name}: mc1step did not converge"
    orbitals = mc.mo_coeff
    for _ in range(POLISH_STEPS):
        energies, _, _, g, hessian, _ = derivatives(mc, orbitals, weights)
        if numpy.linalg.norm(g) < POLISH_GRAD:
            break
        x = -numpy.linalg.solve(hessian, g)
        orbitals = orbitals @ mc1step.expmat(mc.unpack_uniq_var(x))
    else:
        raise AssertionError(f"{name}: the polish did not reach |g| {POLISH_GRAD}")
    curvature = numpy.linalg.eigvalsh(hessian)[0]
    assert curvature > POLISH_CURVATURE, f"{name}: H's least eigenvalue {curvature}"
    return orbitals, energies


def prepared(case: dict, recorded: dict):
    """One case's polished CASSCF, checked against casscf_df.json, and the
    multi-root DFCASCI over it with the dense singlets as its roots:
    (mol, casci, ci, casci_energies, casscf_energies, weights, ncas,
    nelecas, aux_provenance), ci a list over the roots or the one vector."""
    name = case["name"]
    mol = molecule(case)
    auxbasis, aux_provenance = aux_basis(case, mol)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, f"{name}: RHF did not converge"

    # The start, exactly as casscf_reference.case_reference takes it.
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
    assert ncas == recorded["ncas"] and nelecas == recorded["nelecas"], name

    orbitals, casscf_energies = polished(name, mf, mo, ncas, nelecas, weights, auxbasis)
    error = numpy.abs(
        numpy.subtract(casscf_energies, recorded["casscf"]["energies"])
    ).max()
    assert error < CASSCF_AGREEMENT, f"{name}: off casscf_df.json by {error}"

    # The exact RHF's orbitals as a DF-RHF over the same fitting set, so the
    # CASCI is PySCF's DFCASCI and NEVPT's core Fock is fitted too.
    casci = mcscf.CASCI(mf.density_fit(auxbasis=auxbasis), ncas, nelecas)
    assert isinstance(casci, _DFCAS), f"{name}: the CASCI is not fitted"
    assert casci.with_df is casci._scf.with_df, f"{name}: two fitting objects"
    casci.fcisolver.nroots = len(weights)
    casci.fcisolver.conv_tol = 1e-14
    casci.fix_spin_(ss=0)
    casci.verbose = 0
    casci.kernel(orbitals)
    check_aux(name, casci, mol, auxbasis)
    davidson = casci.ci if len(weights) > 1 else [casci.ci]
    casci_energies, vectors, _ = spin_roots(casci, casci.mo_coeff, len(weights))
    for root, (old, new) in enumerate(zip(davidson, vectors, strict=True)):
        overlap = abs(numpy.vdot(old, new))
        assert overlap > 1 - SAME_STATE, f"{name} root {root}: overlap {overlap}"
    error = numpy.abs(numpy.subtract(casci_energies, casci.e_tot)).max()
    assert error < CASCI_AGREEMENT, f"{name}: Davidson off the dense CI by {error}"
    error = numpy.abs(numpy.subtract(casci_energies, casscf_energies)).max()
    assert error < CASCI_AGREEMENT, f"{name}: CASCI off the polish by {error}"
    ci = vectors if len(weights) > 1 else vectors[0]
    return (
        mol,
        casci,
        ci,
        casci_energies,
        casscf_energies,
        weights,
        ncas,
        nelecas,
        aux_provenance,
    )


def case_reference(case: dict, recorded: dict) -> tuple[dict, float, float]:
    """One case's record, its largest cas_natorb disagreement, SC's and PC's."""
    name = case["name"]
    (
        mol,
        casci,
        ci,
        casci_energies,
        casscf_energies,
        weights,
        ncas,
        nelecas,
        aux_provenance,
    ) = prepared(case, recorded)

    roots, worst, pc_worst = [], 0.0, 0.0
    for root in range(len(weights)):
        root_ci = ci[root] if len(weights) > 1 else ci
        total, classes = nevpt(casci, ci, root, natorb=True)
        other, other_classes = nevpt(casci, ci, root, natorb=False)
        misses = [abs(other - total)]
        for key, (norm, energy) in classes.items():
            misses += [
                abs(other_classes[key][0] - norm),
                abs(other_classes[key][1] - energy),
            ]
        worst = max(worst, *misses)
        assert max(misses) < INVARIANCE, (
            f"{name} root {root}: cas_natorb=False differs by {max(misses)}"
        )

        # block2's PC on the state the C++ step computes from (active
        # orbitals unrotated), gated by its SC there against PySCF's.
        state = canonical(casci, root_ci, natorb=False)
        b2_sc = block2(casci, state, "sc")
        b2_miss = max(abs(b2_sc[key] - other_classes[key][1]) for key in CLASSES)
        assert b2_miss < BLOCK2_SC, (
            f"{name} root {root}: block2 SC off PySCF's by {b2_miss}"
        )
        pc = block2(casci, state, "pc")
        assert abs(pc["Sijrs"] - b2_sc["Sijrs"]) < BLOCK2_SC, (
            f"{name} root {root}: PC Sijrs"
        )
        natural = block2(casci, canonical(casci, root_ci, natorb=True), "pc")
        pc_miss = max(abs(natural[key] - pc[key]) for key in CLASSES)
        pc_worst = max(pc_worst, pc_miss)
        assert pc_miss < INVARIANCE, (
            f"{name} root {root}: PC on the natural orbitals differs by {pc_miss}"
        )
        print(
            f"{name} root {root}: block2 SC - PySCF SC {b2_miss:.1e}, "
            f"PC cas_natorb {pc_miss:.1e}, PC - SC {sum(pc.values()) - total:+.3e}"
        )

        roots.append(
            {
                "casci_energy": casci_energies[root],
                "classes": [
                    {
                        "name": key,
                        "excitation": CLASSES[key],
                        "norm": classes[key][0],
                        "energy": classes[key][1],
                    }
                    for key in CLASSES
                ],
                "e_corr": total,
                "energy": casci_energies[root] + total,
                "pc_class_energies": [
                    {"name": key, "energy": pc[key]} for key in CLASSES
                ],
                "e_pc_total": sum(pc[key] for key in CLASSES),
            }
        )
    return (
        {
            "name": name,
            "ncas": ncas,
            "nelecas": nelecas,
            "ncore": int((mol.nelectron - nelecas) // 2),
            "weights": weights,
            "aux_basis": aux_provenance,
            "casscf_energies": [float(e) for e in casscf_energies],
            "casscf_energy": float(numpy.dot(weights, casscf_energies)),
            "roots": roots,
        },
        worst,
        pc_worst,
    )


def frozen_reference(case: dict, recorded: dict, keys: list) -> tuple[dict, dict]:
    """One case's frozen-core record, one variant per frozen_core value in
    `keys`, and its largest gate misses."""
    name = case["name"]
    (
        mol,
        casci,
        ci,
        casci_energies,
        casscf_energies,
        weights,
        ncas,
        nelecas,
        aux_provenance,
    ) = prepared(case, recorded)
    ncore = int((mol.nelectron - nelecas) // 2)
    misses = {"zero": 0.0, "no_core_class": 0.0, "block2_sc": 0.0, "pc_natorb": 0.0}
    states, everything = [], []
    for root in range(len(weights)):
        root_ci = ci[root] if len(weights) > 1 else ci
        state = canonical(casci, root_ci, natorb=False)
        states.append(state)
        # The frozen function at n_f = 0 is PySCF's own kernel on this state.
        _, own = nevpt(casci, ci, root, natorb=False)
        zero = frozen_classes(casci, state, 0)
        miss = max(abs(zero[k][i] - own[k][i]) for k in CLASSES for i in range(2))
        misses["zero"] = max(misses["zero"], miss)
        assert miss < FROZEN_ZERO, f"{name} root {root}: n_f = 0 off PySCF by {miss}"
        everything.append(zero)
    variants = []
    for key in keys:
        nfrozen = frozen_count(mol, key)
        assert 0 < nfrozen <= ncore, (
            f"{name}: frozen_core {key} is {nfrozen} of {ncore}"
        )
        roots = []
        for root, state in enumerate(states):
            classes = frozen_classes(casci, state, nfrozen)
            total = sum(energy for _, energy in classes.values())
            # No core index: unchanged by freezing.
            for k in ("Sr", "Srs"):
                miss = max(
                    abs(classes[k][i] - everything[root][k][i]) for i in range(2)
                )
                misses["no_core_class"] = max(misses["no_core_class"], miss)
                assert miss < FROZEN_ZERO, (
                    f"{name} root {root}: frozen {k} moved by {miss}"
                )
            b2_sc = block2(casci, state, "sc", nfrozen)
            b2_miss = max(abs(b2_sc[k] - classes[k][1]) for k in CLASSES)
            misses["block2_sc"] = max(misses["block2_sc"], b2_miss)
            assert b2_miss < BLOCK2_SC, (
                f"{name} root {root}: block2 frozen SC off by {b2_miss}"
            )
            pc = block2(casci, state, "pc", nfrozen)
            assert abs(pc["Sijrs"] - b2_sc["Sijrs"]) < BLOCK2_SC, (
                f"{name} root {root}: Sijrs"
            )
            root_ci = ci[root] if len(weights) > 1 else ci
            natural = block2(
                casci, canonical(casci, root_ci, natorb=True), "pc", nfrozen
            )
            pc_miss = max(abs(natural[k] - pc[k]) for k in CLASSES)
            misses["pc_natorb"] = max(misses["pc_natorb"], pc_miss)
            assert pc_miss < INVARIANCE, (
                f"{name} root {root}: frozen PC natorb {pc_miss}"
            )
            all_electron = sum(energy for _, energy in everything[root].values())
            print(
                f"{name} frozen_core {key} ({nfrozen}) root {root}: "
                f"E_corr {total:.10f} (all-electron {all_electron:.10f}), "
                f"block2 SC {b2_miss:.1e}, PC natorb {pc_miss:.1e}"
            )
            roots.append(
                {
                    "casci_energy": casci_energies[root],
                    "classes": [
                        {
                            "name": k,
                            "excitation": CLASSES[k],
                            "norm": classes[k][0],
                            "energy": classes[k][1],
                        }
                        for k in CLASSES
                    ],
                    "e_corr": total,
                    "energy": casci_energies[root] + total,
                    "pc_class_energies": [
                        {"name": k, "energy": pc[k]} for k in CLASSES
                    ],
                    "e_pc_total": sum(pc[k] for k in CLASSES),
                }
            )
        variants.append({"frozen_core": key, "n_frozen": nfrozen, "roots": roots})
    return (
        {
            "name": name,
            "ncas": ncas,
            "nelecas": nelecas,
            "ncore": ncore,
            "weights": weights,
            "aux_basis": aux_provenance,
            "casscf_energies": [float(e) for e in casscf_energies],
            "variants": variants,
        },
        misses,
    )


def main_frozen() -> None:
    """--frozen-core: nevpt2_df_frozen_core.json."""
    lib.num_threads(1)
    assert lib.num_threads() == 1
    recorded = {c["name"]: c for c in json.loads(OUT_DF.read_text())["cases"]}
    cases = []
    worst = {"zero": 0.0, "no_core_class": 0.0, "block2_sc": 0.0, "pc_natorb": 0.0}
    for case in CASES:
        if case["name"] not in FROZEN_CASES:
            continue
        record, misses = frozen_reference(
            case, recorded[case["name"]], FROZEN_CASES[case["name"]]
        )
        cases.append(record)
        worst = {k: max(worst[k], misses[k]) for k in worst}
    doc = {
        "description": "Frozen-core SC- and PC-NEVPT2 per root on the cases "
        "of nevpt2_df.json named here, the same polished DF (SA-)CASSCF "
        "orbitals and multi-root DFCASCI: per frozen_core value (as the "
        "[nevpt2] key spells it) and its count n_frozen, each root's eight "
        "SC classes' norms and energies (PySCF's class functions with the "
        "lowest n_frozen canonical core orbitals sliced out of dfnevpt2's "
        "integrals; h1eff over the whole core), their sum e_corr and the "
        "total, and block2's PC with frozen=n_frozen per class and its "
        "total, on the root's canonical state with the active orbitals "
        "unrotated. Hartree",
        "provenance": {
            "pyscf": pyscf.__version__,
            "block2": version("block2"),
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND + " --frozen-core",
            "casscf": str(OUT_DF.relative_to(ROOT)),
        },
        "checks": {
            "zero": "the frozen class functions at n_frozen = 0 against "
            "PySCF's NEVPT.kernel (cas_natorb=False), per class norm and energy",
            "no_core_class": "frozen Sr and Srs, which have no core index, "
            "against the all-electron ones, norm and energy",
            "tolerance": FROZEN_ZERO,
            "block2_sc": "block2's SC with frozen=n_frozen against the frozen "
            "PySCF classes, per class energy",
            "block2_sc_tolerance": BLOCK2_SC,
            "pc_natorb": "block2's frozen PC on the root's natural orbitals "
            "against the recorded unrotated one, per class",
            "pc_natorb_tolerance": INVARIANCE,
            "max_difference": worst,
        },
        "block2": {"pc_tau": PC_TAU, "pc_noise": PC_NOISE},
        "cases": cases,
    }
    OUT_FROZEN.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT_FROZEN}: {worst}")


def main() -> None:
    lib.num_threads(1)
    assert lib.num_threads() == 1
    recorded = {c["name"]: c for c in json.loads(OUT_DF.read_text())["cases"]}
    cases, worst, pc_worst = [], 0.0, 0.0
    for case in CASES:
        if case["name"] in EXCLUDED:
            continue
        record, miss, pc_miss = case_reference(case, recorded[case["name"]])
        cases.append(record)
        worst = max(worst, miss)
        pc_worst = max(pc_worst, pc_miss)
    doc = {
        "description": "PySCF strongly contracted NEVPT2 (mrpt.NEVPT, density "
        "fitted throughout) per root over a multi-root DFCASCI on the "
        "polished DF (SA-)CASSCF orbitals of casscf_df.json's cases, from "
        "the same starts and over the same data/aux_basis fitting set: each "
        "root's eight classes' norms and energies, their sum e_corr and the "
        "total, and block2's PC-NEVPT2 per class and pc total on the same "
        "state with the active orbitals unrotated. Hartree",
        "provenance": {
            "pyscf": pyscf.__version__,
            "block2": version("block2"),
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND,
            "casscf": str(OUT_DF.relative_to(ROOT)),
        },
        "invariance": {
            "check": "every root again with cas_natorb=False (core and "
            "virtuals canonicalized over the root's Fock, active orbitals "
            "unrotated): every class's norm and energy and e_corr",
            "tolerance": INVARIANCE,
            "max_difference": worst,
            "pc_check": "block2's PC on each root's natural orbitals "
            "(cas_natorb=True) against the recorded unrotated one, per class",
            "pc_max_difference": pc_worst,
        },
        "block2": {
            "sc_tolerance": BLOCK2_SC,
            "pc_tau": PC_TAU,
            "pc_noise": PC_NOISE,
        },
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT} (cas_natorb agreement {worst:.2e}, PC {pc_worst:.2e})")


if __name__ == "__main__":
    if sys.argv[1:] == ["--frozen-core"]:
        main_frozen()
    else:
        assert not sys.argv[1:], f"unknown arguments {sys.argv[1:]}"
        main()
