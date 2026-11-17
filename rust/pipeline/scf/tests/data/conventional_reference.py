"""PySCF's exact RHF energy, no density fitting, for the Rust SCF loop's
conventional case: `scf.RHF(mol)` over the same orbital document the tree
compiles in, converged to `conv_tol = 1e-12`.

Usage (from the checkout root; PySCF is in the `tools` dependency group):

    D=rust/pipeline/scf/tests/data
    OMP_NUM_THREADS=8 uv run --group tools python $D/conventional_reference.py \\
        rust/integrals/one_electron/overlap/tests/data/h2o.xyz \\
        cc-pvdz $D/scf_h2o_cc-pvdz_conventional.npy

It writes one float64 .npy of shape (1,): the total energy in hartree,
electronic plus nuclear repulsion. The molecule and the set come as
scf_reference.py's do, through the three-center crate's reference.py
helpers: the BSE JSON document of data/basis/, the coordinates in bohr by
CODATA 2018. The energy does not depend on the scale of each basis
function, so the tree's unrenormalized contractions need no correction. It
prints the energy, PySCF's iteration count and whether it converged; a run
that did not converge writes nothing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from pyscf import gto, scf

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
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    xyz, orbital, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    ref = three_center_reference()
    atoms = ref.read_xyz(xyz)
    doc = ref.bse_document(orbital, "basis")
    basis = {s: ref.basis_from_bse(doc, s) for s in {a for a, _ in atoms}}
    mol = gto.M(atom=atoms, basis=basis, unit="Bohr", cart=False, verbose=0)

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    cycles = []
    mf.callback = lambda envs: cycles.append(None)
    energy = mf.kernel()
    print(
        f"nao {mol.nao}, nocc {mol.nelectron // 2}:"
        f" E = {energy:.12f}, {len(cycles)} cycles, converged {mf.converged}"
    )
    if not mf.converged:
        raise SystemExit("not converged: nothing written")
    np.save(out, np.array([energy]))


if __name__ == "__main__":
    main()
