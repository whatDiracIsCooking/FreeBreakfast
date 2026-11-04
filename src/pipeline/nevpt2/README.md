# Pipeline: Nevpt2

Strongly and partially contracted NEVPT2 of every root of the
state-averaged CASSCF, on its orbitals, over the density-fitted B the CASSCF
ran on: PySCF's `mrpt.NEVPT(mc, root=r)` with `dfnevpt2`'s fitted integrals
for SC, and PC-NEVPT2 (block2's `WickICNEVPT2`) over the same inputs; and,
with `quasi_degenerate = true`, SC- and PC-QD-NEVPT2 of the roots together.
The work is the [`nevpt2`](../../nevpt2/README.md) layer's; this step builds
what that layer reads of each root and of each pair of roots.

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

  // QD-NEVPT2, empty without quasi_degenerate; n roots, [I n + J] bra I
  std::vector<nevpt2::ClassEnergies> qd_couplings;      // SC <I|V|Psi_J^(1)> per class
  std::vector<double> qd_heff, qd_energies, qd_vectors; // H_eff, eigenvalues, eigenvectors
  std::vector<nevpt2::PcClassEnergies> pc_qd_couplings; // PC's; empty, as the three
  std::vector<double> pc_qd_heff, pc_qd_energies, pc_qd_vectors; // below, when refused
  std::string pc_qd_refusal;                            // why, where PC-QD is empty
};

struct Nevpt2Step {
  static constexpr std::string_view kName = "nevpt2";
  bool quasi_degenerate = false;

  Result<Nevpt2Output> run(View<MoleculeOutput, BasisSetOutput, EriOutput,
                                CoreHamiltonianOutput, ScfOutput, CasscfOutput> in,
                           DeviceResources &res) const;
};
```

One option, `quasi_degenerate` (QD-NEVPT2, below); without it `[nevpt2]`
is the request, and it gets both variants per root. The
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

## QD-NEVPT2

[`examples/nevpt2.toml`](../../../examples/nevpt2.toml) runs the per-root
results alone, and [`examples/qd_nevpt2.toml`](../../../examples/qd_nevpt2.toml)
adds `quasi_degenerate = true` over a CASSCF of two roots.

With `quasi_degenerate`, after the per-root results, `run` forms the second-
order effective Hamiltonian of the n roots together, SC's and PC's, over the
layer's `nevpt2::sc_effective_hamiltonian` and `pc_effective_hamiltonian`
(`nevpt2/energy/README.md` "SC-QD-NEVPT2", "PC-QD-NEVPT2"):

1. **The orbitals are the CASSCF's own**, `CasscfOutput::c`, whose core and
   virtuals are canonical over the **state-averaged** Fock, with
   `CasscfOutput::mo_energies` as e_core and e_virt: one set shared by every
   root, as QD needs (every pair's inputs but the RDMs are the same
   pointers). Not the per-root Fock the state-specific results use, so **the
   QD diagonal differs from `roots[r]`**: `qd_couplings[r n + r]` is root r's
   SC-NEVPT2 on the shared orbitals, `roots[r]` on its own. F^I comes from
   one more `response::fock` (over the state average's γ and Γ, though F^I
   reads neither), and h1e, the `h1e_v_*` blocks, the MO blocks and h2e are
   built from it once, as per root (step 3.3–3.4 above), with R = 1.
2. **Each root's own RDMs are kept** from the per-root pass (dm1, dm2, dm3,
   f3ca, f3ac do not depend on the core and virtual rotation, which leaves
   the active orbitals and (tu|vw) alone), for the diagonal.
3. **The diagonal, once**: both effective Hamiltonians over n × n inputs
   with every root's own entry and every off-diagonal omitted (null RDMs),
   which walks each root's classes and solves its PC metrics once and hands
   back each root's first-order wave function (`ScAmplitudes`,
   `PcAmplitudes`): the kets. Then **per unordered pair I < J**, the
   transition dm1, dm2 and dm3 of (I, J) and of (J, I) by `nevpt2::rdms`
   with a bra and a ket, and null f3ca and f3ac (read on the diagonal
   only), and both effective Hamiltonians over those two entries alone, the
   diagonal omitted and the kets passed back: a coupling ⟨I|V|Ψ_J⁽¹⁾⟩ reads
   only the pair's transition RDMs and ket J's own wave function, so each
   pair walks its transition norms and nothing of a root's own
   (`nevpt2/energy/README.md` "SC-QD-NEVPT2", #689). The step logs both
   parts' wall times.
4. **H_IJ = E_I δ_IJ + ½ (C_IJ + C_JI)**, E_I the CASSCF root energy and C
   the couplings' totals, and its eigenvalues (ascending: the QD total
   energies) and eigenvectors over the roots, each signed so that its
   largest-magnitude component is positive (the goldens' rule), by
   `diagonalize` on the device; n is small.

**A PC refusal does not fail the step**, as per root: a `Numerical` error
from `pc_effective_hamiltonian` (a state's metric with no positive
eigenvalue, or a denominator not positive, naming the class and the state) is logged, PC is
not attempted for the pairs, and `pc_qd_refusal` holds the message, whose
"state J" is root J (the diagonal call's states are the roots);
`pc_qd_couplings`, `pc_qd_heff`, `pc_qd_energies` and
`pc_qd_vectors` are empty. SC-QD stands. Any other error fails the step.

Fewer than two roots is refused `InvalidConfig` when the config is read
(`config/README.md`), and by `run` before anything is queued, for a step
driven without the config: with one root the state-averaged Fock is the
root's own, so QD would only repeat the state-specific result.

`main` prints each QD state's energy and its mixing vector over the CASSCF
roots, SC's then PC's, or `PC-QD-NEVPT2: refused (<message>)`.

**The cost** over the per-root pass: one Fock and one set of MO blocks, n(n
− 1) transition builds of `nevpt2::rdms` (each dm3 and the dm1 and dm2
gemvs alone, about a third of a per-root build's gemm work), one call of
each effective Hamiltonian over the n diagonals, and n(n − 1)/2 over two
transition entries each: every root's class walk and PC solves run once.
On the RTX 3080 the QD part of the step takes 31 ms on the skew H2CO SA3
case and 40 ms on the polished H8 ring SA3, against 47 and 67 when each
pair's 2 × 2 call walked both roots' diagonals again.

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

**QD** keeps every root's dm3, f3ca and f3ac (3n of them), and per pair two
transition dm3 (the transition builds skip f3, which the effective
Hamiltonians read only on the diagonal): 3n + 2 n_act⁶ tensors, not the 3n²
of every pair's transition set at once. Each pair's are consumed by its
call before the next pair is built, and the layer's own per-pair
intermediates (dm3_h, ...; the PC transition rows) are two pairs' rather
than n²'s. The kets held across the pairs are the PC U_J on the device (d ×
m_J per class and root, at most n_act⁶ for Sr and Si, as a dm3) and the
amplitudes on the host (`nevpt2/energy/README.md` "The amplitude
hand-over"). The slab batch is sized as per root.

## Errors

`InvalidConfig` for `quasi_degenerate` over fewer than two roots;
`Unsupported` under an integral-direct `EriOutput`; the module calls' own
(`casscf_integrals`', `ci::DeviceStringSet::create`'s and `ci::Rdm`'s,
`response::fock`'s, the canonicalization's, `nevpt2::mo_blocks`', `h2e`'s,
`rdms`', `class_energies`' and `pc_class_energies`', and with QD the
effective Hamiltonians' and `diagonalize`'s); GpuRuntime for a
failed copy or gemm. Each fails the step.

**A PC refusal does not.** `nevpt2::pc_class_energies` refuses a class as
`Numerical` when its metric has no positive eigenvalue or a denominator
λ + Δ is not positive (`nevpt2/energy` "The cut, and refusals"; since #853
there is no gap test, and a metric whose real modes run through the cut is
solved).
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

`PipelineNevpt2.PcAcceptedThroughAContinuum`: HF in cc-pVDZ, CAS(10,7)
over every electron, from its own config. The F 1s is active and all but
doubly occupied, and Sr's metric (d = 343) has real eigenvalues running
continuously through the cut, with no gap, from 5.8e-14 of its largest up.
Until #853 PC was refused (smallest kept / largest dropped 1.7, under the
gap rule's 1e3); those modes carry no energy (E_corr moves 2.6e-9 with the
cut anywhere in 1e-15 … 1e-12, `nevpt2/energy` "PC-NEVPT2"), and PC is
accepted: every class at or below SC's, the total their sum, and E_corr
within 1e-8 of that band's middle, −0.1415607176. 441 determinants, 3 s, in
the fast tier.

`PipelineNevpt2.PcRefusalKeepsSc`: water in STO-3G with CAS(2,1), its one
active orbital doubly occupied, so Si's metric is zero and has no positive
eigenvalue. The step succeeds, SC's eight classes are there and sum to the
total, and `pc_roots[0]` is empty with a `pc_refusals[0]` naming Si.

`PipelineQdNevpt2.MatchesGoldens`, from configs with `quasi_degenerate =
true` written from `data/qd_nevpt2/qd_nevpt2.json` (molecule, basis,
fitting set, state average; the Casci step's start, as the reference's
PySCF default): the three H2CO cases, `h2co_skew_ccpvdz_sa3` (C1, three
roots, every coupling non-zero) and the two C2v ones (both off-diagonals
zero by symmetry), each under a second on an RTX 3080, all in the fast
tier. Each case asserts our
CASSCF's |g| and root energies, then: the QD diagonal's eight classes per
root against the goldens' `class_energies` at the state-specific 1e-9; the
SC and PC eigenvalues against `eigenvalues` and `pc_eigenvalues` at 1e-8,
the issue's bar, derived in the test (each H_eff element within 1e-9 by the
state-specific derivation, an eigenvalue within n times that); H_eff's
diagonal at 1e-8 and its off-diagonal and eigenvectors in magnitude (their
signs are the CI phases'); and, where `nevpt2_df.json` has the case
(`h2co_ccpvdz_sa2`), each root's state-specific E_corr against PySCF's at
1e-9, unchanged by the option. The hydrogen cases are not run: their
references are mc1step CASSCFs converged to |g| < 1e-7 and not polished, so
our orbitals at 1e-9 could differ from theirs by more than 1e-8 allows.
Measured on an RTX 3080 (CUDA): |g| 7e-13 to 8e-11, every SC and PC
eigenvalue within 9.6e-12 of the goldens' (skew; 6.8e-13 on the C2v
cases).
`PipelineNevpt2.MatchesPyscf` asserts the QD product empty without the
option.

`PipelineQdNevpt2.PcQdAcceptedThroughAContinuum`:
`PcAcceptedThroughAContinuum`'s HF CAS(10,7) averaged over three roots (the
ground state and both components of the lowest ¹Π: over two, the average
splits the pair, and its CASSCF converged erratically, #716; see
`../../mcscf/solve/README.md`) with `quasi_degenerate`, in at most 20 macro
iterations. The run succeeds, and SC-QD and PC-QD are each there and held
alike (H_eff symmetric, its diagonal each root's CASSCF energy plus its own
coupling, its eigenvalues ascending and summing to its trace), each PC
diagonal coupling at or below SC's; until #853 PC-QD was refused here.
**PC-QD's refusal path (PC-QD empty, SC-QD kept) has no pipeline test**: no
plain config reaches it since #853. Its refusals are a metric with no
positive eigenvalue or a non-positive denominator, and the configs that give
the first (an active space fully occupied or empty) have one orbital and so
one singlet, where QD needs two roots. It is covered at the layer:
`Nevpt2PcQdEnergyAdjoint.RefusesWhatTheEnergyRefuses` (`nevpt2_energy_gpu.cpp`)
and `Nevpt2Pt2DensityRefusals.PcRefusalRefuses` (`nevpt2_pt2_density_gpu.cpp`)
refuse a state with zero RDMs, and the step's handling of a refusal is the
single-state `PcRefusalKeepsSc`'s. QD over a single-root CASSCF, the step
driven by hand, is refused `InvalidConfig` in
`PipelineNevpt2.AnOpenShellOrANonSingletIsUnsupported`; through the config,
in `test/pipeline/config_host.cpp`, with the option's parsing.

## Files

| File | Role |
|------|------|
| `nevpt2.cppm` | Module interface: `Nevpt2Output`, `Nevpt2Step` |
| `nevpt2.cpp` | `run`: the refusals, the integrals, and per root the Fock, the canonicalization, the one-electron blocks, the MO blocks and RDMs, and the classes; then QD's shared inputs, per pair the transition RDMs and the two effective Hamiltonians, and their eigenpairs |
| `CMakeLists.txt` | The module library |
