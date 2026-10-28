# Pipeline: Nevpt2

Strongly contracted NEVPT2 of every root of the state-averaged CASSCF, on its
orbitals, over the density-fitted B the CASSCF ran on: PySCF's
`mrpt.NEVPT(mc, root=r)` with `dfnevpt2`'s fitted integrals. The work is the
[`nevpt2`](../../nevpt2/README.md) layer's; this step builds what that layer
reads of each root.

## Module

`free_breakfast.pipeline.nevpt2`, target `free_breakfast::pipeline::nevpt2`.
It re-exports `free_breakfast.pipeline.casscf` (the `CasscfOutput` and every
product it re-exports, the `EriOutput` among them) and
`free_breakfast.nevpt2.energy` (`nevpt2::ClassEnergies`, in the product).
`run` is defined in the implementation unit, so `integrals.active_space`,
`response.fock`, `ci.strings`, `ci.rdm`, `nevpt2.density_fitting` and
`nevpt2.rdm` are PRIVATE links.

## The step

```cpp
struct Nevpt2Output {
  std::vector<nevpt2::ClassEnergies> roots; // per root: norms[8], energies[8], total
};

struct Nevpt2Step {
  static constexpr std::string_view kName = "nevpt2";

  Result<Nevpt2Output> run(View<MoleculeOutput, BasisSetOutput, EriOutput,
                                CoreHamiltonianOutput, CasscfOutput> in,
                           DeviceResources &res) const;
};
```

No options: `[nevpt2]` is the request. Root r's total energy is
`CasscfOutput::energies[r] + roots[r].total`; `main` prints both.

`run`, all on `res.stream()`:

1. **Refuses integral-direct ERIs**, `Unsupported`, naming `[eri]`, before
   anything is queued: the density-fitted variant is the only one ported, and
   an integral-direct source is a later driver (`nevpt2/density_fitting`
   "No `IntegralSource` interface"). The config refuses it too, when it is
   read (`config/README.md`), so a run does not converge an SCF and a CASSCF
   first.
2. **One `casscf_integrals`** over the CASSCF's orbitals, fitted: (pq|uv) and
   (pu|qv), which every root's Fock reads, and (tu|vw), which
   `nevpt2::rdms` contracts. The per-root rotations below leave the active
   columns alone, so the three are every root's.
3. **Per root r:**
   1. γ_r and Γ_r of the root's CI vector alone, by `ci::Rdm::rdm12` at
      weight 1.
   2. **The root's own Fock**, F = F^I + F^A(γ_r), by `response::fock` over
      the step-2 integrals, fitted, and the core and virtuals canonicalized
      over it by `canonicalize_core_virtual` (`pipeline/casscf`'s, the
      CASSCF step's own over the state average): R, block-diagonal, and
      C_r = C R. `CasscfOutput::mo_energies` is the **state-averaged** Fock's,
      which is not what PySCF's NEVPT2 reads; e_core and e_virt are F's
      eigenvalues here. **The active orbitals are not rotated**: PySCF's
      `NEVPT.kernel` rotates them to the root's natural orbitals
      (`cas_natorb`), and SC-NEVPT2 is invariant to that, which
      `nevpt2_reference.py` checks for every root of the reference to 1e-10
      (`test/gtest/data/README.md`).
   3. **The one-electron blocks**: F^I in the root's orbitals, Rᵀ F^I R
      (two gemms), sliced into h1e (its active block, PySCF's
      `h1eff[a, a]`), `h1e_v_si` (a × c), `h1e_v_sir` (v × c) and
      `h1e_v_sr` (v × a), from which Σ_b (vb|ba) is subtracted: one gemm per
      active b over B_va and B_aa, PySCF's `einsum('mbbn->mn', h2e_v)`.
   4. **The MO blocks** of B over C_r, `nevpt2::mo_blocks`, and h2e from them,
      `nevpt2::h2e`.
   5. **The RDMs** in PySCF's `make_dm123` convention: dm1 = γ_r, dm2 =
      Γ_r + δ_qr γ_ps (⟨E_pq E_rs⟩, `ci::Rdm`'s Γ is `make_rdm12`'s), and dm3,
      f3ca, f3ac by `nevpt2::rdms` over (tu|vw).
   6. `nevpt2::class_energies` over them.

## Memory

The step measures the device's free memory once its integrals and the
CASSCF's products are held, and

- **gives `nevpt2::rdms` half of it**, not the layer's default (the free
  memory less 512 MiB). The device workspace never shrinks, so at the default
  a large active space would leave `class_energies` almost nothing for its own
  buffers, and a failed allocation aborts. `rdms` grows the workspace only to
  the tile it plans: all of the determinants when they fit, which they do
  with room to spare at every size the tests run (CAS(6,6), 400 of them, is
  4.4 MB). Half is a cap, not a reservation, so a sibling process on the card
  is not starved by it;
- **sizes the slab batch** of `class_energies` to an eighth of what is free
  after `rdms`: the most of a class's external index whose widest slab fits.
  A class holds two slabs at once (Sir, Srsi), and `nevpt2::slab` grows the
  workspace to a third for its gemm product. At the tests' sizes every class
  takes one slab.

The per-root arrays (the RDMs, the one-electron blocks, R, Rᵀ F^I R, C_r)
are the step's own buffers, not the workspace's: each is held across calls
that use it. The largest are dm3, f3ca and f3ac, n_act⁶ doubles each.

## Errors

`Unsupported` under an integral-direct `EriOutput`; the module calls' own
(`casscf_integrals`', `ci::DeviceStringSet::create`'s and `ci::Rdm`'s,
`response::fock`'s, the canonicalization's, `nevpt2::mo_blocks`', `h2e`'s,
`rdms`' and `class_energies`'); GpuRuntime for a failed copy or gemm.

## Not done

- **Integral-direct ERIs** (step 1).
- **One `casscf_integrals` too many.** The Casscf step builds the same
  (pq|uv) and (pu|qv) at its solution and drops them; the product could
  carry them, at nmo² n_act² doubles each held for the rest of the run.
- **F^I per root.** `response::fock` builds F^I again for every root,
  though it is the same for all: one `df_coulomb` and one `df_exchange` per
  root, against the RDMs' n_act⁶ and the slabs, small.

## Tests

`test/gtest/pipeline_nevpt2_gpu.cpp`, through the real registry from a
config naming `[casscf]`, a density-fitted `[eri]` over the compiled-in
cc-pVDZ-JKFIT and `[nevpt2]`: each `data/casscf/casscf_df.json` case through
our own DF-CASSCF at `conv_tol_grad` 1e-9, from its recorded start, then per
root every class's norm and energy and E_corr against PySCF's in
`data/nevpt2/nevpt2_df.json`, at 1e-9, derived in the test from how closely
our CASSCF reaches PySCF's polished orbitals (\|g\| and each root's energy,
both asserted). N₂ is not a case: SC-NEVPT2 depends on the arbitrary split
of its degenerate π* virtual pairs by up to 5e-6, PySCF's included, so the
reference leaves it out (`test/gtest/data/README.md`). The step, like PySCF,
does nothing about such a split: within a degenerate core or virtual set its
answer is as arbitrary as the eigenvectors `diagonalize` returns. And
`[nevpt2]` with an integral-direct `[eri]`, written or filled in, refused
`Unsupported` naming `[eri]` when the config is read.

## Files

| File | Role |
|------|------|
| `nevpt2.cppm` | Module interface: `Nevpt2Output`, `Nevpt2Step` |
| `nevpt2.cpp` | `run`: the refusal, the integrals, and per root the Fock, the canonicalization, the one-electron blocks, the MO blocks and RDMs, and the classes |
| `CMakeLists.txt` | The module library |
