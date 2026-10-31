#!/usr/bin/env python3
"""Setup: the C2h chair guess and the symmetry-labelled CAS(6,6).

1. Build an exactly C2h chair (R16 = R34 = 1.95 A) and optimise it with
   DF-B3LYP/cc-pVDZ under C2h. The chair TS's transition mode is a_u, so
   inside C2h the TS is a minimum and a plain minimisation finds it.
2. DF-RHF at that geometry; pick the six canonical MOs with the most allyl-pi
   character irrep by irrep, 1a_g + 1b_u + 1a_u occupied and 1a_g + 1b_u +
   1b_g virtual (2a_g + 2b_u + 1a_u + 1b_g in all).
3. DF-CASSCF(6,6), 1A_g singlet, from those MOs.

Writes, beside this script:
  b3lyp_c2h_chair.xyz   the DF-B3LYP C2h chair (PySCF's C2h frame)
  setup.json            B3LYP and CASSCF energies, R16/R34, the chosen MOs,
                        the active irreps and natural occupations
  setup_orbitals.npy    the CASSCF MO coefficients scan.py starts from

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        experimental/cope_rearrangement/step_1/setup.py [--threads N]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cope  # noqa: E402

BASIS = "cc-pvdz"
AUXBASIS = "cc-pvdz-jkfit"


def write_xyz(path, mol, comment):
    atoms = cope.mol_atoms(mol)
    lines = [str(len(atoms)), comment]
    lines += [f"{s:2s} {x:16.10f} {y:16.10f} {z:16.10f}" for s, (x, y, z) in atoms]
    path.write_text("\n".join(lines) + "\n")


def casscf(mf, mo):
    from pyscf import mcscf

    mc = mcscf.DFCASSCF(mf, cope.NCAS, cope.NELECAS)
    mc.fcisolver.wfnsym = "Ag"
    mc.fix_spin_(ss=0)
    mc.conv_tol = 1e-9
    mc.conv_tol_grad = 1e-5
    mc.kernel(mo)
    return mc


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    from pyscf import dft, gto, lib, mcscf, scf
    from pyscf.geomopt.geometric_solver import optimize

    if args.threads:
        lib.num_threads(args.threads)

    mol = gto.M(
        atom=cope.atom_string(cope.chair_c2h(r_inter=1.95)),
        basis=BASIS,
        symmetry=True,
        verbose=3,
    )
    assert mol.topgroup == "C2h", mol.topgroup

    ks = dft.RKS(mol, xc="b3lyp").density_fit(auxbasis=AUXBASIS)
    ks.conv_tol = 1e-10
    ks.grids.level = 4
    mol_ts = optimize(ks, maxsteps=100)
    ks = dft.RKS(mol_ts, xc="b3lyp").density_fit(auxbasis=AUXBASIS)
    ks.conv_tol = 1e-10
    ks.grids.level = 4
    e_b3lyp = ks.kernel()
    r16, r34 = cope.interallyl(mol_ts)
    write_xyz(
        HERE / "b3lyp_c2h_chair.xyz",
        mol_ts,
        f"DF-B3LYP/cc-pVDZ C2h chair, E = {e_b3lyp:.10f}, R16 = R34 = {r16:.4f} A",
    )

    mf = scf.RHF(mol_ts).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = 1e-10
    mf.kernel()
    chosen, report = cope.select_cas(mf)
    mo = mcscf.CASSCF(mf, cope.NCAS, cope.NELECAS).sort_mo(
        chosen, mo_coeff=mf.mo_coeff, base=0
    )
    mc = casscf(mf, mo)
    assert mc.converged
    counts = cope.active_irrep_counts(mc)
    assert counts == cope.CAS_IRREP_NCAS, counts
    nat = cope.active_natural(mc)

    np.save(HERE / "setup_orbitals.npy", mc.mo_coeff)
    out = {
        "basis": BASIS,
        "auxbasis": AUXBASIS,
        "b3lyp": {"e_tot": e_b3lyp, "r16": r16, "r34": r34},
        "rhf": {"e_tot": float(mf.e_tot), "cas_guess_mos": report},
        "casscf": {
            "e_tot": float(mc.e_tot),
            "ncore": int(mc.ncore),
            "active_irreps": counts,
            "natural_occupations": nat,
        },
    }
    (HERE / "setup.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
