"""One geometry's DF-CASSCF(6,6) and PC-NEVPT2, in C1.

C1, not C2h: the a_u displacements step 2 differentiates along break C2h to
C2, and PySCF re-orients a symmetric molecule into its own frame, which would
turn the orbitals carried in from the reference point. In C1 the frame is the
one the coordinates are written in, so a converged neighbour's MO
coefficients are a valid guess as they stand. The reference (undisplaced)
point is run the same way, so every energy in a finite difference comes from
the same code path.

PC-NEVPT2 is block2's (`pyblock2.icmr.icnevpt2_full.WickICNEVPT2`); PySCF
implements only SC. As in tools/emit/nevpt2_reference.py, block2 is handed
the orbitals already canonicalised and PySCF's own density-fitted integrals
(`dfnevpt2._ERIS`, cc-pVDZ-JKFIT, the fit the CASSCF uses), in place of its
own exact transform. SC-NEVPT2 is PySCF's `mrpt.NEVPT` on the same state,
recorded beside it.
"""

from __future__ import annotations

import copy
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "step_1"))

import cope  # noqa: E402

BASIS = "cc-pvdz"
AUXBASIS = "cc-pvdz-jkfit"

# Tight enough for a finite difference of h = 0.02 bohr: an a_u energy step
# there is ~1e-5 to 1e-4 Eh, and NEVPT2, not variational in the orbitals,
# carries the orbital gradient's residual to first order. (1e-7 on the
# gradient is past what mc1step reaches with DF here: it stalls unconverged.)
CONV_TOL = 1e-11
CONV_TOL_GRAD = 1e-6


def make_mol(atoms_bohr):
    from pyscf import gto

    return gto.M(
        atom=[(s, tuple(x)) for s, x in atoms_bohr],
        unit="Bohr",
        basis=BASIS,
        symmetry=False,
        verbose=0,
    )


def orthonormal(mol, mo):
    """`mo` made orthonormal in `mol`'s AO overlap (symmetric, Loewdin).

    Load-bearing: MO coefficients converged at one geometry are not
    orthonormal at a displaced one (the AOs move with the atoms), and PySCF's
    mc1step only rotates the orbitals it is handed, so it would carry the
    defect to convergence. At 0.02 bohr along s that was an 11 mEh error, an
    apparent a_u curvature of -56 Eh/bohr^2."""
    s = mo.T @ mol.intor("int1e_ovlp") @ mo
    w, v = np.linalg.eigh(s)
    return mo @ (v * w**-0.5) @ v.T


def casscf(mol, mo, ci=None):
    from pyscf import mcscf, scf

    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mc = mcscf.DFCASSCF(mf, cope.NCAS, cope.NELECAS)
    mc.fix_spin_(ss=0)
    mc.conv_tol = CONV_TOL
    mc.conv_tol_grad = CONV_TOL_GRAD
    mc.fcisolver.conv_tol = 1e-13
    mc.kernel(orthonormal(mol, mo), ci)
    if not mc.converged:
        # Once, from where it stopped, with more room: one a_u pair point at
        # R = 1.65 A ran out of mc1step's default 50 macro iterations.
        mc.max_cycle_macro = 200
        mc.kernel(mc.mo_coeff, mc.ci)
    assert mc.converged, "CASSCF did not converge"
    return mc


def nevpt2(mc, pc: bool = True) -> dict:
    """PC- and SC-NEVPT2 on mc's state: correlation energies and PC's classes.

    PC is invariant to rotations inside the active space; SC is not, so SC
    runs on the active natural orbitals (unique here: no two occupations are
    close), where PC takes the CASSCF's active orbitals as they come.
    `pc=False` runs SC alone."""
    from pyblock2.icmr import eri_helper, icnevpt2_full
    from pyscf.mrpt import dfnevpt2

    mo, ci, mo_energy = mc.canonicalize(
        mc.mo_coeff, ci=mc.ci, cas_natorb=False, verbose=0
    )
    m2 = copy.copy(mc)
    m2.mo_coeff, m2.ci, m2.mo_energy = mo, ci, mo_energy
    e = dfnevpt2._ERIS(m2, mo, mc.with_df)
    ncore, ncas = mc.ncore, mc.ncas
    nmo = mo.shape[1]
    nvir = nmo - ncore - ncas
    eris = eri_helper._ChemistsERIs()
    eris.ncore, eris.ncas, eris.nocc = ncore, ncas, ncore + ncas
    eris.h1eff = np.asarray(e["h1eff"])
    eris.known = ["ppaa", "papa", "pacv", "cvcv"]
    eris.ppaa = np.asarray(e["ppaa"]).reshape(nmo, nmo, ncas, ncas)
    eris.papa = np.asarray(e["papa"]).reshape(nmo, ncas, nmo, ncas)
    eris.pacv = np.asarray(e["pacv"]).reshape(nmo, ncas, ncore, nvir)
    eris.cvcv = np.asarray(e["cvcv"]).reshape(ncore, nvir, ncore, nvir)

    t0 = time.perf_counter()
    if not pc:
        return {"e_corr_sc": _sc(mc), "seconds_sc": time.perf_counter() - t0}
    init_eris = eri_helper.init_eris
    eri_helper.init_eris = lambda *args, **kwargs: eris
    try:
        ic = icnevpt2_full.WickICNEVPT2(m2)
        ic.verbose = 0
        ic.canonicalized = True
        ic.kernel()
    finally:
        eri_helper.init_eris = init_eris
    t1 = time.perf_counter()

    e_sc = _sc(mc)
    t2 = time.perf_counter()
    return {
        "e_corr_pc": float(ic.e_corr),
        "e_corr_sc": e_sc,
        "pc_classes": {k: float(v) for k, v in ic.sub_eners.items()},
        "seconds_pc": t1 - t0,
        "seconds_sc": t2 - t1,
    }


def _sc(mc) -> float:
    from pyscf import mrpt

    mo, ci, mo_energy = mc.canonicalize(
        mc.mo_coeff, ci=mc.ci, cas_natorb=True, verbose=0
    )
    m2 = copy.copy(mc)
    m2.mo_coeff, m2.ci, m2.mo_energy = mo, ci, mo_energy
    sc = mrpt.NEVPT(m2)
    sc.verbose = 0
    sc.canonicalized = True
    return float(sc.kernel())
