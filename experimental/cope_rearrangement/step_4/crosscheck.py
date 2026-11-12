"""PySCF/block2 NEVPT2 on free-breakfast's own CASSCF orbitals, and its CI.

Two layers of difference sit between free-breakfast and step 2, and this
script takes each away in turn (README "Step 4a").

1. **Orbitals.** Step 2's CASSCF stopped at ||g|| ~ 1e-6 (pt2.py's
   CONV_TOL_GRAD); free-breakfast's at ~1e-10. NEVPT2 is not variational in
   the orbitals, so it carries that residual to first order, while the CASSCF
   energy does not. Here PySCF runs a DF-CASCI(6,6) on free-breakfast's
   converged orbitals (runs/R<r>/casscf_orbitals.npy), projected back onto
   PySCF's built-in cc-pVDZ (exact: the spans agree), with no orbital
   optimisation.
2. **The CI vector.** That CASCI's CI is converged to 1e-13; the CASSCF's is
   converged to a residual scaled to conv_tol_grad (mcscf/solve's
   ci_tolerance_for), ~1e-6 away in norm. NEVPT2 is linear in the CI error
   where the energy is quadratic, so that too shows at ~1e-8. The second run
   hands PySCF/block2 free-breakfast's own CI (runs/R<r>/casscf_ci.npy, in
   PySCF's cistring order) on the same orbitals.

What is left after both is the two implementations' difference.

  energies.py run R ...   first (it writes the orbitals and the CI), then
  crosscheck.py R ...     -> crosscheck/R<r>.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parent.parent
sys.path.insert(0, str(STUDY / "step_2"))
sys.path.insert(0, str(HERE))

import pt2  # noqa: E402
from energies import RUNS, STEP1, parse, tag, tree_mols  # noqa: E402


def check(r: float) -> dict:
    from pyscf import gto, mcscf, scf

    point = json.loads((STEP1 / "points" / f"{tag(r)}.json").read_text())
    atoms = [(s, tuple(xyz)) for s, xyz in point["geometry"]]
    run = RUNS / tag(r)
    c_tree = numpy.load(run / "casscf_orbitals.npy")
    fb = parse((run / "log.txt").read_text())

    builtin, tree, d = tree_mols(atoms)
    s_pp = builtin.intor("int1e_ovlp")
    s_pt = gto.intor_cross("int1e_ovlp", builtin, tree)
    # D first: this tree's AOs are PySCF's normalised JSON AOs times D (#1220).
    c_p = numpy.linalg.solve(s_pp, s_pt @ (d[:, None] * c_tree))
    defect = float(abs(c_p.T @ s_pp @ c_p - numpy.eye(c_p.shape[1])).max())

    mf = scf.RHF(builtin).density_fit(auxbasis=pt2.AUXBASIS)
    mc = mcscf.DFCASCI(mf, pt2.cope.NCAS, pt2.cope.NELECAS)
    mc.fix_spin_(ss=0)
    mc.fcisolver.conv_tol = 1e-13
    mc.verbose = 0
    mc.kernel(c_p)
    own = pt2.nevpt2(mc)

    ci_p = mc.ci / numpy.linalg.norm(mc.ci)
    ci_fb = numpy.load(run / "casscf_ci.npy").reshape(ci_p.shape)
    ci_fb = ci_fb / numpy.linalg.norm(ci_fb)
    overlap = float((ci_p * ci_fb).sum())
    ci_fb *= numpy.sign(overlap)
    m2 = copy.copy(mc)
    m2.ci = ci_fb
    same_ci = pt2.nevpt2(m2)

    out = {
        "r": r,
        "projection_defect": defect,
        "e_casci_pyscf": float(mc.e_tot),
        "e_casscf_fb": fb["e_casscf"],
        "ci_distance": float(numpy.linalg.norm(ci_p - ci_fb)),
        "e_corr_pc_fb": fb["e_corr_pc"],
        "e_corr_sc_fb": fb["e_corr_sc"],
        "e_corr_pc_block2": own["e_corr_pc"],
        "e_corr_sc_pyscf": own["e_corr_sc"],
        "e_corr_pc_block2_fb_ci": same_ci["e_corr_pc"],
        "e_corr_sc_pyscf_fb_ci": same_ci["e_corr_sc"],
        "d_e": fb["e_casscf"] - float(mc.e_tot),
        "d_e_corr_pc": fb["e_corr_pc"] - own["e_corr_pc"],
        "d_e_corr_sc": fb["e_corr_sc"] - own["e_corr_sc"],
        "d_e_corr_pc_fb_ci": fb["e_corr_pc"] - same_ci["e_corr_pc"],
        "d_e_corr_sc_fb_ci": fb["e_corr_sc"] - same_ci["e_corr_sc"],
        "pc_classes_block2": own["pc_classes"],
    }
    (HERE / "crosscheck").mkdir(exist_ok=True)
    (HERE / "crosscheck" / f"{tag(r)}.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("r", nargs="+", type=float)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        from pyscf import lib

        lib.num_threads(args.threads)
    for r in args.r:
        o = check(r)
        print(
            f"{tag(r)}: dE {o['d_e']:+.2e}"
            f"  own CI: PC {o['d_e_corr_pc']:+.2e} SC {o['d_e_corr_sc']:+.2e}"
            f"  fb CI: PC {o['d_e_corr_pc_fb_ci']:+.2e}"
            f" SC {o['d_e_corr_sc_fb_ci']:+.2e}"
            f"  (|dci| {o['ci_distance']:.1e})",
            flush=True,
        )


if __name__ == "__main__":
    main()
