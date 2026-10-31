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
active index, must equal SC Sijrs to BLOCK2_SC; and every external tuple's
solve's matrix must have a gap of at least PC_MIN_GAP at the C++ step's
cut, its own refusal rule, or the PC energy would depend on where the cut fell.
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
from importlib.metadata import version  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from pyblock2.icmr import eri_helper, icnevpt2_full, scnevpt2  # noqa: E402
from pyscf import lib, mcscf, mrpt, scf  # noqa: E402
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
# Every PC solve's matrix must have a gap of at least PC_MIN_GAP at PC_TAU:
# its singular values at or below PC_TAU of its largest are null, and the
# smallest kept over the largest cut (over PC_TAU where none is) must reach
# PC_MIN_GAP. That is nevpt2::pc_class_energies' refusal rule. lstsq(rcond=None)
# cuts lower, at eps * d, and the structural null modes' noise can straddle
# that cut on natural orbitals; a mode in that noise carries no energy, so only
# the gap above it matters.
PC_TAU = 1e-13
PC_MIN_GAP = 1e3

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


def canonical(casci, ci, natorb: bool):
    """One root's orbitals, CI vector and orbital energies: core and
    virtuals canonicalized over the root's own Fock, the active space
    rotated to its natural orbitals or left as it is."""
    return casci.canonicalize(casci.mo_coeff, ci=ci, cas_natorb=natorb, verbose=0)


def pc_gap(a: numpy.ndarray) -> float:
    """The smallest gap at PC_TAU over one block of per-tuple matrices:
    smallest kept / largest cut relative singular value, or / PC_TAU where a
    tuple cuts nothing. All-zero tuples have nothing to solve and are
    skipped."""
    if len(a) == 0:
        return numpy.inf
    sv = numpy.linalg.svd(a, compute_uv=False)
    live = sv[:, 0] > 0
    rel = sv[live] / sv[live, :1]
    if not len(rel):
        return numpy.inf
    kept = rel > PC_TAU
    smallest_kept = numpy.where(kept, rel, numpy.inf).min(axis=1)
    largest_cut = numpy.where(kept, 0.0, rel).max(axis=1)
    floor = numpy.where((~kept).any(axis=1), largest_cut, PC_TAU)
    return float((smallest_kept / numpy.maximum(floor, numpy.finfo(float).tiny)).min())


def block2(casci, state, kernel: str) -> dict[str, float]:
    """block2's SC (`kernel="sc"`) or PC (`"pc"`) per class on one root's
    `state` (canonical's), `canonicalized` so block2 takes the orbitals and
    energies it is handed, over PySCF's DF integrals for those orbitals. PC
    checks every solve's gap against PC_MIN_GAP."""
    mo, ci, mo_energy = state
    mc = copy.copy(casci)
    mc.mo_coeff, mc.ci, mc.mo_energy = mo, ci, mo_energy
    e = dfnevpt2._ERIS(mc, mo, casci.with_df)
    ncore, ncas = mc.ncore, mc.ncas
    nmo = mo.shape[1]
    nvir = nmo - ncore - ncas
    eris = eri_helper._ChemistsERIs()
    eris.ncore, eris.ncas, eris.nocc = ncore, ncas, ncore + ncas
    eris.h1eff = numpy.asarray(e["h1eff"])
    eris.known = ["ppaa", "papa", "pacv", "cvcv"]
    eris.ppaa = numpy.asarray(e["ppaa"]).reshape(nmo, nmo, ncas, ncas)
    eris.papa = numpy.asarray(e["papa"]).reshape(nmo, ncas, nmo, ncas)
    eris.pacv = numpy.asarray(e["pacv"]).reshape(nmo, ncas, ncore, nvir)
    eris.cvcv = numpy.asarray(e["cvcv"]).reshape(ncore, nvir, ncore, nvir)

    gaps = []
    solve = icnevpt2_full._linear_solve

    def gapped_solve(a, b):
        gaps.append(pc_gap(a))
        return solve(a, b)

    init_eris = eri_helper.init_eris
    eri_helper.init_eris = lambda *args, **kwargs: eris
    icnevpt2_full._linear_solve = gapped_solve
    try:
        cls = {"sc": scnevpt2.WickSCNEVPT2, "pc": icnevpt2_full.WickICNEVPT2}[kernel]
        ic = cls(mc)
        ic.verbose = 0
        ic.canonicalized = True
        ic.kernel()
    finally:
        eri_helper.init_eris = init_eris
        icnevpt2_full._linear_solve = solve
    if kernel == "pc":
        assert gaps and min(gaps) >= PC_MIN_GAP, (
            f"a PC solve's gap at {PC_TAU:.0e} is {min(gaps):.1e} < {PC_MIN_GAP:.0e}: "
            "the PC energy would depend on where the cut fell"
        )
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


def case_reference(case: dict, recorded: dict) -> tuple[dict, float, float]:
    """One case's record, its largest cas_natorb disagreement, SC's and PC's."""
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
            "pc_min_gap": PC_MIN_GAP,
        },
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT} (cas_natorb agreement {worst:.2e}, PC {pc_worst:.2e})")


if __name__ == "__main__":
    main()
