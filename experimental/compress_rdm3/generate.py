#!/usr/bin/env python3
"""Write one NEVPT2 case for C1 glycolaldehyde, DF-CASSCF(10,10)/cc-pVDZ, in
the layout of test/gtest/data/nevpt2/ (that README has the conventions), plus
dm3, f3ca and f3ac: the input for experiments on compressing the 3-RDM.

The state is a DF-CASCI on DF-CASSCF(10,10) orbitals, canonicalised as
`pyscf.mrpt.NEVPT.kernel` does (`cas_natorb`), density-fitted throughout with
cc-pVDZ-JKFIT: RHF, CASSCF, CASCI, `h1eff` and every integral block, as the
tree's `*_casscf_df` golden is. The answer is PySCF's DF-NEVPT2
(`pyscf.mrpt.dfnevpt2`), rebuilt class by class from the very arrays written
here and checked against `NEVPT.kernel`'s own total before anything is saved.

There is no PC-NEVPT2 reference: the tree's goldens take theirs from block2,
which this script does not run, so `pc_class_energies` and `e_pc_total` are
not written.

dm3, f3ca and f3ac are 8 MB each at CAS(10,10), over the tree's 3 MB
per-file limit (`check-added-large-files`), so the output directory is
written but not meant to be committed as it is.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        experimental/compress_rdm3/generate.py

writes experimental/compress_rdm3/glycolaldehyde_ccpvdz_cas1010_casscf_df/.
"""

from __future__ import annotations

import argparse
import os
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
CASE = "glycolaldehyde_ccpvdz_cas1010_casscf_df"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=pathlib.Path, default=HERE / CASE)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="BLAS/OpenMP threads; only the default of 1 reruns bit for bit",
    )
    return parser.parse_args()


# Parsed before numpy loads its BLAS, which reads the thread count once. At
# one thread every reduction runs in one order and a rerun is reproducible.
ARGS = parse_args()
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = str(ARGS.threads)


import numpy  # noqa: E402
from pyscf import fci, gto, lib, mcscf, mp, mrpt, scf  # noqa: E402
from pyscf.mrpt import dfnevpt2, nevpt2  # noqa: E402

# Glycolaldehyde, not optimised: standard bond lengths, the O=C-C-O torsion
# at 30 degrees and the hydroxyl hydrogen out of every plane, so the point
# group is C1 and no CI coefficient or RDM element vanishes by symmetry.
GEOMETRY = """
C
C 1 1.51
O 2 1.21 1 122.0
H 2 1.10 1 116.0 3 180.0
O 1 1.41 2 111.0 3 30.0
H 5 0.96 1 108.0 2 70.0
H 1 1.09 2 109.0 5 120.0
H 1 1.09 2 109.0 5 -120.0
"""
BASIS = "cc-pVDZ"
AUX_BASIS = "cc-pVDZ-JKFIT"
# Only for the MP2 natural orbitals the CASSCF starts from.
MP2_AUX_BASIS = "cc-pVDZ-RI"
NCAS, NELECAS = 10, 10

# The class order of class_norms / class_energies, PySCF's.
CLASSES = ("Sr", "Si", "Sijrs", "Sijr", "Srsi", "Srs", "Sij", "Sir")

# Each class rebuilt from the saved arrays against NEVPT.kernel's total.
CHECK_TOL = 1e-10


def reference():
    """The canonicalised NEVPT object over a DF-CASCI on DF-CASSCF orbitals,
    and the DF-CASSCF itself."""
    mol = gto.M(atom=GEOMETRY, basis=BASIS, symmetry=False, verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUX_BASIS)
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise SystemExit("RHF did not converge")
    # From DF-MP2 natural orbitals, not the RHF frontier: from RHF orbitals the
    # 1-step CASSCF drifts (dE ~ -2e-6 still at macro iteration 50, 18 mEh
    # above this minimum); from these it converges in about 18.
    _, natorbs = mcscf.addons.make_natural_orbitals(
        mp.MP2(mf).density_fit(auxbasis=MP2_AUX_BASIS).run()
    )
    cas = mcscf.DFCASSCF(mf, NCAS, NELECAS, auxbasis=AUX_BASIS)
    cas.conv_tol = 1e-11
    cas.conv_tol_grad = 1e-6
    cas.max_cycle_macro = 100
    cas.fcisolver.conv_tol = 1e-12
    cas.kernel(natorbs)
    if not cas.converged:
        raise SystemExit("CASSCF did not converge")
    ci = mcscf.DFCASCI(mf, NCAS, NELECAS, auxbasis=AUX_BASIS)
    ci.fcisolver.conv_tol = 1e-13
    ci.kernel(cas.mo_coeff)
    pt = mrpt.NEVPT(ci)
    pt.verbose = 0
    pt.mo_coeff, civec, pt.mo_energy = pt.canonicalize(
        pt.mo_coeff, ci=pt.load_ci(), cas_natorb=True, verbose=0
    )
    pt.ci = civec
    pt.canonicalized = True
    return pt, cas


def three_index(with_df, mo: numpy.ndarray) -> numpy.ndarray:
    """B[L, p, q] over all MOs, (pq|rs) = sum_L B[L,p,q] B[L,r,s]."""
    out = []
    for cderi in with_df.loop():
        ao = lib.unpack_tril(cderi)
        out.append(numpy.einsum("Lmn,mp,nq->Lpq", ao, mo, mo, optimize=True))
    return numpy.concatenate(out)


def case_arrays(pt) -> tuple[dict[str, numpy.ndarray], object]:
    """Every array of the case, and the eris they were sliced from."""
    ncore, ncas = pt.ncore, pt.ncas
    nocc = ncore + ncas
    nelec = pt.nelecas
    ci = numpy.ascontiguousarray(pt.load_ci())
    eris = dfnevpt2._ERIS(pt._mc, pt.mo_coeff, pt._mc.with_df)

    h1e = eris["h1eff"][ncore:nocc, ncore:nocc]
    aaaa = numpy.ascontiguousarray(eris["ppaa"][ncore:nocc, ncore:nocc])
    dm1, dm2, dm3 = fci.rdm.make_dm123("FCI3pdm_kern_sf", ci, ci, ncas, nelec)
    link = tuple(fci.cistring.gen_linkstr_index(range(ncas), n) for n in nelec)
    f3ca = nevpt2._contract4pdm("NEVPTkern_cedf_aedf", aaaa, ci, ncas, nelec, link)
    f3ac = nevpt2._contract4pdm("NEVPTkern_aedf_ecdf", aaaa, ci, ncas, nelec, link)

    # Each external block exactly as PySCF's Sxxx slices it from eris.
    h2e_v_sr = eris["ppaa"][nocc:, ncore:nocc].transpose(0, 2, 1, 3)
    nvirt = pt.mo_coeff.shape[1] - nocc
    b = three_index(pt._mc.with_df, pt.mo_coeff)
    a = {
        "ci": ci,
        "dm1": dm1,
        "dm2": dm2,
        "dm3": dm3,
        "f3ca": f3ca,
        "f3ac": f3ac,
        "h1e": h1e,
        "h2e": aaaa.transpose(0, 2, 1, 3),
        "e_core": pt.mo_energy[:ncore],
        "e_virt": pt.mo_energy[nocc:],
        "h1e_v_Sr": eris["h1eff"][nocc:, ncore:nocc]
        - numpy.einsum("mbbn->mn", h2e_v_sr),
        "h1e_v_Si": eris["h1eff"][ncore:nocc, :ncore],
        "h1e_v_Sir": eris["h1eff"][nocc:, :ncore],
        "h2e_v_Sr": h2e_v_sr,
        "h2e_v_Si": eris["ppaa"][ncore:nocc, :ncore].transpose(0, 2, 1, 3),
        "cvcv": numpy.asarray(eris["cvcv"]).reshape(ncore, nvirt, ncore, nvirt),
        "h2e_v_Sijr": eris["pacv"][:ncore].transpose(3, 1, 2, 0),
        "h2e_v_Srsi": eris["pacv"][nocc:].transpose(3, 0, 2, 1),
        "h2e_v_Srs": eris["papa"][nocc:, :, nocc:].transpose(0, 2, 1, 3),
        "h2e_v_Sij": eris["papa"][:ncore, :, :ncore].transpose(1, 3, 0, 2),
        "h2e_v1_Sir": eris["ppaa"][nocc:, :ncore].transpose(0, 2, 1, 3),
        "h2e_v2_Sir": eris["papa"][nocc:, :, :ncore].transpose(0, 3, 1, 2),
        "B_aa": b[:, ncore:nocc, ncore:nocc],
        "B_ca": b[:, :ncore, ncore:nocc],
        "B_va": b[:, nocc:, ncore:nocc],
        "B_cv": b[:, :ncore, nocc:],
    }
    return {k: numpy.ascontiguousarray(v) for k, v in a.items()}, eris


def class_answers(pt, a: dict, eris) -> numpy.ndarray:
    """(8, 2) norms and energies: NEVPT.kernel's eight calls, on the eris and
    RDMs the case was cut from."""
    dms = {"1": a["dm1"], "2": a["dm2"], "3": a["dm3"], "4": None}
    dms |= {"f3ca": a["f3ca"], "f3ac": a["f3ac"]}
    ci = a["ci"]
    return numpy.array(
        [
            nevpt2.Sr(pt, ci, dms, eris),
            nevpt2.Si(pt, ci, dms, eris),
            nevpt2.Sijrs(pt, eris),
            nevpt2.Sijr(pt, dms, eris),
            nevpt2.Srsi(pt, dms, eris),
            nevpt2.Srs(pt, dms, eris),
            nevpt2.Sij(pt, dms, eris),
            nevpt2.Sir(pt, dms, eris),
        ]
    )


def check(pt, a: dict, answers: numpy.ndarray) -> list[str]:
    """The checks before writing; a line for each, or SystemExit on a miss."""
    lines = []
    total = answers[:, 1].sum()
    # A fresh NEVPT over the same canonical CASCI: its own DF path end to end.
    fresh = mrpt.NEVPT(pt._mc)
    fresh.verbose = 0
    want = fresh.kernel()
    if not abs(total - want) <= CHECK_TOL:
        raise SystemExit(f"class sum {total!r} against NEVPT.kernel's {want!r}")
    lines.append(f"sum of classes vs NEVPT.kernel: {abs(total - want):.1e}")

    # The three-index blocks reproduce the eris they stand beside.
    eri = numpy.einsum("Lpq,Lrs->pqrs", a["B_aa"], a["B_aa"])
    err = abs(eri - a["h2e"].transpose(0, 2, 1, 3)).max()
    if not err <= CHECK_TOL:
        raise SystemExit(f"B_aa B_aa misses (tu|vw) by {err:.1e}")
    lines.append(f"B_aa B_aa vs (tu|vw): {err:.1e}")
    eri = numpy.einsum("Lia,Ljb->iajb", a["B_cv"], a["B_cv"])
    err = abs(eri - a["cvcv"]).max()
    if not err <= CHECK_TOL:
        raise SystemExit(f"B_cv B_cv misses cvcv by {err:.1e}")
    lines.append(f"B_cv B_cv vs cvcv: {err:.1e}")

    norm = numpy.linalg.norm(a["ci"])
    if not abs(norm - 1.0) <= 1e-12:
        raise SystemExit(f"ci has norm {norm!r}")
    trace = numpy.einsum("ppqqrr->", a["dm3"])
    n = sum(pt.nelecas)
    if not abs(trace - n**3) <= 1e-9:
        raise SystemExit(f"tr dm3 = {trace!r}, not N^3 = {n**3}")
    lines.append(
        f"|ci| - 1: {abs(norm - 1):.1e}; tr dm3 - N^3: {abs(trace - n**3):.1e}"
    )
    return lines


def save(path: pathlib.Path, array: numpy.ndarray) -> int:
    """One '<f8' Fortran-order .npy, as tools/emit/nevpt2_golden.py writes."""
    numpy.save(path, numpy.asfortranarray(array, dtype="<f8"), allow_pickle=False)
    return path.stat().st_size


def main() -> None:
    args = ARGS
    pt, cas = reference()
    a, eris = case_arrays(pt)
    answers = class_answers(pt, a, eris)
    lines = check(pt, a, answers)

    occ = numpy.linalg.eigvalsh(a["dm1"])[::-1]
    print(f"{CASE}: E_CASSCF {cas.e_tot:.10f}, E_CASCI {pt._mc.e_tot:.10f}")
    print("  natural occupations " + " ".join(f"{x:.4f}" for x in occ))
    print(
        f"  ncore {pt.ncore}, ncas {pt.ncas}, nvirt {a['e_virt'].size}, "
        f"naux {a['B_aa'].shape[0]}, ndet {a['ci'].size}"
    )
    for name, (norm, energy) in zip(CLASSES, answers, strict=True):
        print(f"  {name:6s} norm {norm:.12e}  E {energy:+.12e}")
    print(f"  SC-NEVPT2 correlation energy {answers[:, 1].sum():+.12f}")
    for line in lines:
        print(f"  {line}")

    a["class_norms"] = answers[:, 0]
    a["class_energies"] = answers[:, 1]
    a["e_nevpt2_total"] = numpy.array([answers[:, 1].sum()])
    a["e_casci"] = numpy.array([pt._mc.e_tot])
    args.out.mkdir(parents=True, exist_ok=True)
    written = sum(save(args.out / f"{k}.npy", v) for k, v in a.items())
    print(f"  {len(a)} arrays, {written} bytes, in {args.out}")


if __name__ == "__main__":
    main()
