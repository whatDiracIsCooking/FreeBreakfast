# `tools/`

Standalone utilities around the C++ tree: Python scripts, and one C++
generator. Not tests, not a package, not on the push gate.

```
basis/bse_download.py
                  Fetch basis sets from the Basis Set Exchange as the JSON
                  src/io/basis reads. Standard library only.
basis/cartesian_to_spherical.py
                  Relabel a BSE JSON document's gto_cartesian shells
                  gto_spherical (5D), in place, exponents and coefficients
                  untouched, so src/io/basis reads it: how data/basis/
                  carries the Pople sets. Standard library only.
minao_from_pyscf.py
                  Write data/basis/minao.json: the MINAO free-atom minimal
                  basis (the IAOs' B2), converted from PySCF's
                  pyscf/gto/basis/minao.py, since the Basis Set Exchange
                  does not serve it. Needs PySCF.
jkfit_from_pyscf.py
                  Write data/aux_basis/cc-pV{D,T}Z-JKFIT.json and
                  def2-universal-JKFIT.json: Weigend's JK fitting sets,
                  converted from PySCF's pyscf/gto/basis/*-jkfit.dat, since
                  the Basis Set Exchange has no cc-pVDZ-JKFIT and PySCF is
                  the oracle. Needs PySCF.
atomic_cd/        Generate an auxiliary basis for any orbital basis: aCD /
                  acCD atomic Cholesky decomposition plus SVD contraction
                  (Lehtola 2021, 2023) over PySCF integrals; writes BSE
                  JSON, one element (cli) or a whole named set (aux_set,
                  which wrote data/aux_basis/cc-pVDZ-acCD.json and the
                  *-acCD-6-6.json sets).
                  atomic_cd/README.md has the method, measured accuracy
                  and references.
emit/one_electron_kernels.py
                  Regenerate src/integrals/one_electron/{overlap,
                  overlap_mixed,kinetic,potential_en,dipole}/kernels/
                  (the dipole through d only), and the mmd record bound
                  src/integrals/mmd/kernels/pair_bound/, from the
                  FelizJueves submodule. Standard library only.
emit/coulomb_kernels.py
                  Regenerate src/integrals/eri/integral_direct/coulomb/
                  {dq,e_hermite,j_coulomb}/kernels/ from the FelizJueves
                  submodule. Standard library only.
emit/exchange_kernels.py
                  Regenerate src/integrals/eri/integral_direct/exchange/
                  kernels/{kb,kbr}/, and the dressed record bound
                  src/integrals/mmd/kernels/record_bound/, from the
                  FelizJueves submodule. Standard library only.
emit/df_kernels.py
                  Regenerate src/integrals/eri/density_fitting/{metric,
                  three_center}/kernels/ from the FelizJueves submodule.
                  Standard library only.
extract_integral_golden.py
                  Copy the libcint golden data test/gtest/data/{one_electron,
                  coulomb,exchange}/ hold out of FelizJueves's fixtures.
                  Standard library only.
emit/nonsymmetric_jk_golden.py
                  Write test/gtest/data/exchange/*_nonsymmetric.json:
                  PySCF's get_jk(hermi=0), vj and vk, of a random
                  non-symmetric density (fixed seed) over the H2O exchange
                  goldens' molecules and bases, which FelizJueves's fixtures
                  do not carry. Needs PySCF.
emit/record_bound_golden.py
                  Write test/gtest/data/record_bound/: FelizJueves's host
                  ao.k.record_bounds for every ordered shell pair of the
                  exchange goldens' bases. Needs numpy (FelizJueves's
                  reference Boys).
emit/sad_reference.py
                  Write test/sad/data/atomic_scf.json: PySCF's
                  spherically averaged atomic RHF densities, the reference
                  src/sad's guess is tested against. Needs PySCF.
emit/scf_reference.py
                  Write test/gtest/data/scf/rhf.json and rohf.json: PySCF's
                  closed-shell RHF and open-shell ROHF energies and orbital
                  energies over this tree's data/basis files, the reference
                  the Scf step is tested against. Needs PySCF.
emit/df_reference.py
                  Write test/gtest/data/density_fitting/: libcint's spherical
                  (P|Q) and (mn|P), PySCF's DF vj/vk of one density and its
                  DF-RHF and DF-ROHF energies, over this tree's data/basis and
                  data/aux_basis files, the reference density fitting is
                  tested against. Needs PySCF.
emit/cphf_reference.py
                  Write test/gtest/data/cphf/: over df_reference's h2o_ccpvdz
                  integrals, PySCF's DF-RHF orbitals, the nuclear CPHF
                  right-hand sides and pyscf.df.hessian.rhf solve_mo1's mo1,
                  and the dipole right-hand sides and polarizability, the
                  reference response::cphf is tested against. Needs PySCF.
emit/rhf_hessian_reference.py
                  Write test/gtest/data/rhf_hessian/: the response part of
                  pyscf.df.hessian.rhf's DF-RHF Hessian (its total less
                  partial_hess_elec and hess_nuc, CPHF column by column to
                  1e-12, checked against a dense solve), the orbitals and
                  the CPHF solutions U^x, for a skew water and formaldehyde
                  in cc-pVDZ / cc-pVDZ-JKFIT, the reference
                  hessian::df_rhf_response is tested against. Needs PySCF.
emit/cis_reference.py
                  Write test/gtest/data/cis/: PySCF's RHF + TDA (CIS)
                  singlet and triplet roots and singlet oscillator
                  strengths, exact and density-fitted over cc-pVDZ-JKFIT,
                  each checked against a dense diagonalization of the
                  explicit A matrix, and the spherical AO dipole int1e_r,
                  the reference the CIS sigma is tested against. Needs
                  PySCF.
emit/localization_reference.py
                  Write test/gtest/data/localization/{pipek_mezey,ibo,rohf}.json:
                  PySCF's Mulliken Pipek-Mezey and IBOs of those RHF orbitals, both
                  exponents, as the functional and per-orbital atomic
                  populations, and of two DF-ROHF references per occupied
                  space, the reference the Localization step is tested
                  against. Needs PySCF.
emit/fci_reference.py
                  Write test/gtest/data/ci/direct_spin1.json: PySCF's
                  active-space integrals in its own MO basis, and
                  fci.direct_spin1's lowest roots over them, with <S^2> and
                  1-RDMs, and state averages (CI vectors with their weighted
                  make_rdm12 1- and 2-RDMs), the reference src/ci is tested
                  against. Needs PySCF.
emit/active_integrals_reference.py
                  Write test/gtest/data/active_space/active_space.json:
                  PySCF CASCI's E_core and the sign-invariant elements of
                  h_eff and (tu|vw) over this tree's basis files, the
                  reference integrals.active_space is tested against. Needs
                  PySCF.
emit/casci_reference.py
                  Write test/gtest/data/casci/casci.json: PySCF mcscf.CASCI
                  over its RHF in this tree's basis files (and mcscf.avas's
                  active spaces over this tree's MINAO), the lowest
                  singlets' (or a triplet case's) energies, <S^2> and natural occupations, the
                  reference the Casci step is tested against; and over a
                  DF-ROHF's orbitals, a doublet and a triplet, by index and
                  by mcscf.avas (its active orbitals recorded up to
                  sign). --append
                  keeps the file's cases and adds only new ones. Needs
                  PySCF.
emit/fci_step_reference.py
                  Write test/gtest/data/fci/fci.json: PySCF fci.FCI over
                  every orbital of its RHF in this tree's basis files, the
                  lowest singlets' (or a triplet case's) energies, <S^2> and natural occupations,
                  the reference the Fci step is tested against. Needs PySCF.
emit/casscf_reference.py
                  Write test/gtest/data/casscf/casscf.json: PySCF
                  mcscf.CASSCF (one root, or two singlets state-averaged)
                  over its RHF in this tree's basis files (and, fitted only,
                  from a DF-ROHF's orbitals, a doublet and a triplet),
                  checked against
                  mc2step and polished by Newton steps; at the start, the
                  CASCI energies and the sign-invariant parts of gen_g_hop's
                  gradient and orbital Hessian; converged, the energies,
                  each space's orbital energies and the natural occupations,
                  the reference src/response, src/mcscf and the Casscf step
                  are tested against. --append keeps each file's cases and
                  adds only new ones. Needs PySCF.
emit/nevpt2_reference.py
                  Write test/gtest/data/nevpt2/nevpt2_df.json: PySCF's SC
                  DF-NEVPT2 per root on the polished DF CASSCF orbitals of
                  casscf_df.json's cases, each class's norm and energy and
                  the total, and block2's PC-NEVPT2 per class on the same
                  state (active orbitals unrotated), with the cas_natorb
                  invariance of both checked. Needs PySCF and block2.
emit/nevpt2_gradient_reference.py
                  Write test/gtest/data/nevpt2_gradient/: the numerical
                  nuclear gradients (central differences at three steps
                  plus one Richardson step) of re-converged single-state
                  DF-CASSCF, SC-NEVPT2 (PySCF) and PC-NEVPT2 (block2)
                  energies, every displaced CASSCF through a continuity
                  guard, the oracle the NEVPT2 gradient is tested against.
                  Needs PySCF and block2.
emit/nevpt2_golden.py
                  Write test/gtest/data/nevpt2/<case>/<array>.npy: the
                  nevpt2-gpu-demo's density-fitted goldens as .npy, plus
                  PySCF's dm3, f3ca and f3ac (CAS(4,4) and CAS(8,8) only),
                  each checked against the golden's Sr and Si before
                  writing. Needs PySCF and a clone of the demo.
emit/frequencies_reference.py
                  Write test/gtest/data/frequencies/frequencies.json: PySCF
                  RHF/cc-pVDZ Hessians of water, CO2, HCN and linear water
                  and pyscf.hessian.thermo.harmonic_analysis of each, the
                  oracle geometry::harmonic_analysis is tested against.
                  Needs PySCF.
emit/ci_bench_integrals.py
                  Write the given JSON path (untracked): PySCF's CAS(10,10)
                  and CAS(12,12) integrals of N2 cc-pVDZ at 1.0977 and
                  2.0 A, in direct_spin1.json's schema, the input
                  bench/ci_sigma_precision times ci_solve over. Needs PySCF.
sad/              C++: free-breakfast-sad-stamps, built with the tree. Write
                  data/sad/<basis>/<Z>.npy, the SAD atomic density of every
                  element of the compiled-in sets named (or --all), by src/sad
                  itself. data/sad/README.md has when and how.
```

Re-run the four `emit/*_kernels.py`, `extract_integral_golden.py` and then
`emit/record_bound_golden.py` and `emit/nonsymmetric_jk_golden.py` (both read
the exchange goldens' shells) after
bumping `deps/FelizJueves`. Everything but `basis/`, `atomic_cd` and
`emit/ci_bench_integrals.py` writes tracked files, so the diff is the review.

## Running

`basis/` needs nothing beyond Python:

```sh
tools/basis/bse_download.py cc-pVDZ def2-SVP -o data/basis
tools/basis/bse_download.py --list
tools/basis/cartesian_to_spherical.py data/basis/6-31g.json
```

`atomic_cd`, `minao_from_pyscf.py`, `jkfit_from_pyscf.py` and the
`emit/*_reference.py` scripts need PySCF and numpy, which sit in the opt-in
`tools` dependency group in `pyproject.toml` rather than in `dev`, so the
image and the push gate do not carry them. Inside the devcontainer, add
`UV_PROJECT_ENVIRONMENT=$PWD/.venv` so the group goes into a worktree-local
venv rather than the image's `/opt/venv`:

```sh
uv run --group tools tools/minao_from_pyscf.py
uv run --group tools tools/jkfit_from_pyscf.py
uv run --group tools tools/emit/sad_reference.py
uv run --group tools tools/emit/scf_reference.py
uv run --group tools tools/emit/df_reference.py
uv run --group tools tools/emit/cis_reference.py
uv run --group tools tools/emit/localization_reference.py
uv run --group tools tools/emit/casci_reference.py
uv run --group tools tools/emit/fci_step_reference.py
uv run --group tools tools/emit/casscf_reference.py
uv run --group tools tools/emit/nevpt2_reference.py
uv run --group tools tools/emit/nevpt2_gradient_reference.py
uv run --group tools tools/emit/nevpt2_golden.py <demo clone>
uv run --group tools tools/emit/record_bound_golden.py
uv run --group tools tools/emit/nonsymmetric_jk_golden.py
uv run --group tools python -m tools.atomic_cd.cli C sto-3g
uv run --group tools python -m tools.atomic_cd.cli C cc-pVDZ \
    --bse-out c_ccpvdz_accd.bse.json --df-check
uv run --group tools python -m tools.atomic_cd.aux_set \
    data/basis/cc-pVDZ.json cc-pVDZ-acCD --max-l 4
```

The two compose: an orbital basis fetched by `bse_download.py` can be fed back
in with `--bse-in`, which makes the positional basis name a display label only.

```sh
tools/basis/bse_download.py cc-pVDZ --elements C -o /tmp/basis
uv run --group tools python -m tools.atomic_cd.cli C cc-pVDZ \
    --bse-in /tmp/basis/cc-pVDZ.json
```

`atomic_cd/README.md` covers the method, options and measured accuracy, with
the literature; each module's docstring carries the derivation and
conventions, and `atomic_cd/__init__.py` lists them in dependency order.
