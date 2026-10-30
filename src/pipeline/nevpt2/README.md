# Pipeline: Nevpt2

Strongly and partially contracted NEVPT2 of every root of the
state-averaged CASSCF, on its orbitals, over the density-fitted B the CASSCF
ran on: PySCF's `mrpt.NEVPT(mc, root=r)` with `dfnevpt2`'s fitted integrals
for SC, and PC-NEVPT2 (block2's `WickICNEVPT2`) over the same inputs. The work is the
[`nevpt2`](../../nevpt2/README.md) layer's; this step builds what that layer
reads of each root.

## Module

`free_breakfast.pipeline.nevpt2`, target `free_breakfast::pipeline::nevpt2`.
It re-exports `free_breakfast.pipeline.casscf` (the `CasscfOutput` and every
product it re-exports, the `EriOutput` among them) and
`free_breakfast.nevpt2.energy` (`nevpt2::ClassEnergies` and
`nevpt2::PcClassEnergies`, in the product).
`run` is defined in the implementation unit, so `integrals.active_space`,
`response.fock`, `ci.strings`, `ci.rdm`, `nevpt2.density_fitting` and
`nevpt2.rdm` are PRIVATE links.

## The step

```cpp
struct Nevpt2Output {
  std::vector<nevpt2::ClassEnergies> roots;      // SC per root: norms[8], energies[8], total
  // PC per root: energies[8], dropped[8], total; empty where PC was refused
  std::vector<std::optional<nevpt2::PcClassEnergies>> pc_roots;
  std::vector<std::string> pc_refusals; // why, where pc_roots[r] is empty
};

struct Nevpt2Step {
  static constexpr std::string_view kName = "nevpt2";

  Result<Nevpt2Output> run(View<MoleculeOutput, BasisSetOutput, EriOutput,
                                CoreHamiltonianOutput, ScfOutput, CasscfOutput> in,
                           DeviceResources &res) const;
};
```

No options: `[nevpt2]` is the request, and it gets both variants. The
`ScfOutput` is read for its spin occupations alone: an open shell,
`n_alpha != n_beta`, is refused `Unsupported` before anything is queued:
the Casscf step takes an open shell, but this step's strings are
`nelecas / 2` of each spin. For the
same reason a molecule of multiplicity other than 1, whose CASSCF roots are of
its spin S (`../../mcscf/solve/README.md`), is refused `Unsupported` too. They share
every input (the RDMs, the Fock blocks, the MO blocks and slabs), so a choice
between them would save only PC's energy stage, about 1.2–1.5× SC's, on top
of RDMs that dominate as the active space grows. Root r's SC total energy is
`CasscfOutput::energies[r] + roots[r].total`, and its PC one
`CasscfOutput::energies[r] + pc_roots[r]->total`; `main` prints both, or
`PC-NEVPT2 root r:  refused (<message>)`, the message naming the class, for
a root whose PC was refused (Errors).

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
   6. `nevpt2::class_energies` over them, then `nevpt2::pc_class_energies`
      over the same blocks, inputs and slab batch. **PC is invariant to the
      active orbitals too**, so not rotating them is as right for PC as for
      SC: an active rotation U transforms each class's metric S and Dyall
      matrix K by the same congruence (each internal index carries one U),
      which leaves the span and the eigenvalues of the pencil (K, S), and so
      every class energy, unchanged; the cut, relative to S's largest
      eigenvalue, is unchanged with them. Measured twice:
      `Nevpt2PcInvariance` (`nevpt2_energy_gpu.cpp`) rotates every input of
      the energy layer by a random U at CAS(4,4) and CAS(8,8), both of which
      cut modes, and gets every class back to 4e-15 with the same dropped
      counts; and `nevpt2_reference.py` checks block2's PC on each root's
      natural orbitals against the unrotated one it records, to 1.8e-14.

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
  takes one slab. PC reuses that batch. Its solve's d × d matrices, d a
  class's internal dimension (n_act³ for Sr and Si), are allocated past that
  budget and freed when the class returns: at n_act = 12 each is 24 MB, the
  size of dm3.

The per-root arrays (the RDMs, the one-electron blocks, R, Rᵀ F^I R, C_r)
are the step's own buffers, not the workspace's: each is held across calls
that use it. The largest are dm3, f3ca and f3ac, n_act⁶ doubles each.

## Errors

`Unsupported` under an integral-direct `EriOutput`; the module calls' own
(`casscf_integrals`', `ci::DeviceStringSet::create`'s and `ci::Rdm`'s,
`response::fock`'s, the canonicalization's, `nevpt2::mo_blocks`', `h2e`'s,
`rdms`', `class_energies`' and `pc_class_energies`'); GpuRuntime for a
failed copy or gemm. Each fails the step.

**A PC refusal does not.** `nevpt2::pc_class_energies` refuses a class as
`Numerical` when its metric has no clean gap above its null space (no
positive eigenvalue, a smallest kept eigenvalue under 1e3 times the largest
cut one, or a non-positive λ + Δ: `nevpt2/energy` "The cut, and refusals").
That says PC is not defined to the precision asked of it on this state; SC,
which solves no such system, is. So the step logs a warning naming the root
and the layer's message (which names the class), leaves that root's
`pc_roots[r]` empty with the message in `pc_refusals[r]`, and goes on; SC's
energies for the root stand. Any other error from `pc_class_energies` still
fails the step. A state with an all but doubly occupied orbital in the
active space is the usual cause: the demo's N₂ CAS(12,12), with a 1s
combination active, and the test's HF CAS(10,7), with the F 1s.

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
answer is as arbitrary as the eigenvectors `diagonalize` returns.
`[nevpt2]` with an integral-direct `[eri]`, written or filled in, refused
`Unsupported` naming `[eri]` when the config is read, is
`test/pipeline/config_host.cpp`'s.

PC against block2's, the same file's `pc_class_energies` and `e_pc_total`
(`nevpt2_reference.py`, on the reference's state with the active orbitals
unrotated, as the step leaves them): every class and the total per root at
the same 1e-9, which the same derivation sets, since PC is first order in
the orbitals and the CI vectors as SC is. Measured on an RTX 3080 (CUDA):
every class within 7.4e-12. And per root what must hold of PC beside SC:
none above SC's (PC minimizes the same Hylleraas functional over a span
holding SC's function), Sijrs equal to SC's (it has no active index), and
the total the sum, to 1e-12.

`PipelineNevpt2.PcRefusalKeepsSc`: HF in cc-pVDZ, CAS(10,7) over every
electron, from its own config. The F 1s is active and all but doubly
occupied, and Sr's metric has eigenvalues running through the cut with no
gap (smallest kept / largest dropped 1.7, far under 1e3), so PC is refused.
The step succeeds, SC's eight classes are there and sum to the total, and
`pc_roots[0]` is empty with a `pc_refusals[0]` naming Sr. 441 determinants,
3 s, in the fast tier.

## Files

| File | Role |
|------|------|
| `nevpt2.cppm` | Module interface: `Nevpt2Output`, `Nevpt2Step` |
| `nevpt2.cpp` | `run`: the refusal, the integrals, and per root the Fock, the canonicalization, the one-electron blocks, the MO blocks and RDMs, and the classes |
| `CMakeLists.txt` | The module library |
