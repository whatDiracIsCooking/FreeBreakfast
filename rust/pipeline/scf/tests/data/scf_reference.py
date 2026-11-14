"""PySCF's density-fitted RHF energy, for the Rust SCF loop's tests:
`scf.RHF(mol).density_fit(auxbasis=...)` over the same orbital and fitting
documents the tree compiles in, converged to `conv_tol = 1e-12`.

Usage (from the checkout root; PySCF is in the `tools` dependency group):

    D=rust/pipeline/scf/tests/data
    OMP_NUM_THREADS=8 uv run --group tools python $D/scf_reference.py \\
        rust/integrals/one_electron/overlap/tests/data/h2o.xyz \\
        cc-pvdz cc-pvdz-jkfit $D/scf_h2o_cc-pvdz_cc-pvdz-jkfit.npy
    OMP_NUM_THREADS=8 uv run --group tools python $D/scf_reference.py \\
        $D/benzene.xyz cc-pvdz cc-pvdz-jkfit \\
        $D/scf_benzene_cc-pvdz_cc-pvdz-jkfit.npy
    OMP_NUM_THREADS=8 uv run --group tools python $D/scf_reference.py \\
        $D/ethylene_meci856.xyz cc-pvdz cc-pvdz-jkfit \\
        $D/scf_ethylene_meci856_cc-pvdz_cc-pvdz-jkfit.npy newton

A fifth argument `newton` runs PySCF's second-order solver
(`mf.newton()`) in place of its DIIS, at the same `conv_tol` and its
default `conv_tol_grad`. Ethylene at #856's MECI needs it: its RHF orbital
Hessian has an eigenvalue near zero (6.4e-6 Eh), on which PySCF's DIIS
stalls at an orbital gradient of 1e-6 and reports itself converged
2.1e-7 Eh above the minimum. The second-order energy moves by under 1e-12
between `conv_tol_grad` 1e-6 and 1e-10.

It writes one float64 .npy of shape (1,): the total energy in hartree,
electronic plus nuclear repulsion. The molecule and both sets come from the
three-center crate's reference.py (`named`): the BSE JSON documents of
data/basis/ and data/aux_basis/, the coordinates in bohr by CODATA 2018.
The energy does not depend on the scale of each basis function, so the
tree's unrenormalized contractions need no correction here, unlike a
matrix. It prints the energy, PySCF's iteration count and whether it
converged; a run that did not converge writes nothing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from pyscf import scf

ROOT = Path(__file__).resolve().parents[5]
THREE_CENTER = (
    ROOT / "rust/integrals/eri/density_fitting/three_center/tests/data/reference.py"
)


def three_center_reference():
    """The three-center crate's reference.py, as a module."""
    spec = importlib.util.spec_from_file_location(
        "three_center_reference", THREE_CENTER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if len(sys.argv) not in (5, 6) or sys.argv[5:] not in ([], ["newton"]):
        raise SystemExit(__doc__)
    xyz, orbital, aux_spec, out = (
        Path(sys.argv[1]),
        sys.argv[2],
        sys.argv[3],
        Path(sys.argv[4]),
    )
    newton = sys.argv[5:] == ["newton"]
    ref = three_center_reference()
    orb, _, _, _ = ref.named(xyz, orbital, aux_spec)
    atoms = ref.read_xyz(xyz)
    aux_doc = ref.bse_document(aux_spec, "aux_basis")
    auxbasis = {s: ref.basis_from_bse(aux_doc, s) for s in {a for a, _ in atoms}}

    mf = scf.RHF(orb).density_fit(auxbasis=auxbasis)
    if newton:
        mf = mf.newton()
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    cycles = []
    # One call a cycle, or a macro iteration of the second-order solver.
    mf.callback = lambda envs: cycles.append(None)
    energy = mf.kernel()
    print(
        f"nao {orb.nao}, naux {mf.with_df.get_naoaux()}, nocc {orb.nelectron // 2}:"
        f" E = {energy:.12f}, {len(cycles)} cycles, converged {mf.converged}"
    )
    if not mf.converged:
        raise SystemExit("not converged: nothing written")
    np.save(out, np.array([energy]))


if __name__ == "__main__":
    main()
