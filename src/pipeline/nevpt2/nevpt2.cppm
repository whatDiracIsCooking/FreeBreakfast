/// @file
/// @brief The `Nevpt2` step: strongly and partially contracted NEVPT2 of
/// each root of the @ref CasscfOutput, on its state-averaged orbitals, over the density-fitted
/// @ref EriOutput the CASSCF itself ran on.
///
/// Device-shaped. `nevpt2`'s layer does the work (the MO blocks and slabs,
/// dm3 and f3, the eight classes, SC and PC); the step builds what it reads of each
/// root: the root's own canonical core and virtuals, its RDMs and the
/// one-electron blocks of the inactive Fock. With @ref Nevpt2Step::quasi_degenerate,
/// SC- and PC-QD-NEVPT2 beside them, over the state-averaged orbitals the
/// roots share. See README.md.

export module free_breakfast.pipeline.nevpt2;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.footprint;
export import free_breakfast.nevpt2.energy; // nevpt2::{Pc,}ClassEnergies, in the product
export import free_breakfast.pipeline.casscf;

export namespace free_breakfast {

/// @brief SC- and PC-NEVPT2 per root of a state-averaged CASSCF.
struct Nevpt2Output {
  /// @brief Root r's eight class norms and energies and their sum, the
  /// root's NEVPT2 correlation energy, in @ref CasscfOutput::energies' order:
  /// root r's total energy is CasscfOutput::energies[r] + roots[r].total.
  std::vector<nevpt2::ClassEnergies> roots;
  /// @brief Root r's eight PC-NEVPT2 class energies and their sum, over the
  /// same RDMs and slabs as roots[r]: root r's PC total energy is
  /// CasscfOutput::energies[r] + pc_roots[r]->total. Empty for a root whose
  /// PC was refused as Numerical (a class's metric with no positive
  /// eigenvalue, or a pole: an intruder, |t| > 1, or a denominator at
  /// round-off); its SC energies stand.
  std::vector<std::optional<nevpt2::PcClassEnergies>> pc_roots;
  /// @brief Why root r's PC was refused, `nevpt2::pc_class_energies`'
  /// message naming the class; empty where pc_roots[r] holds a value.
  std::vector<std::string> pc_refusals;
  /// @brief The core orbitals left uncorrelated, @ref Nevpt2Step::frozen_core
  /// resolved: the lowest of each root's canonical core (and of the
  /// state-averaged one, for QD). 0 correlates every core orbital.
  int32_t frozen_core = 0;

  // QD-NEVPT2, @ref Nevpt2Step::quasi_degenerate's: every field below is
  // empty without it. n is the root count; the core and the virtuals are the
  // CASSCF's own, canonical over the state-averaged Fock and shared by every
  // root, so the QD diagonal is not roots[r]'s, which is over root r's Fock.

  /// @brief SC's ⟨I|V|Ψ_J⁽¹⁾⟩ per class, n x n row-major, bra I at [I n + J]:
  /// `nevpt2::sc_effective_hamiltonian`'s. Its diagonal is each root's
  /// SC-NEVPT2 on the shared orbitals.
  std::vector<nevpt2::ClassEnergies> qd_couplings;
  /// @brief SC's symmetrized effective Hamiltonian, n x n,
  /// H_IJ = E_I δ_IJ + ½ (C_IJ + C_JI) with E_I CasscfOutput::energies[I] and
  /// C_IJ qd_couplings[I n + J].total (hartree).
  std::vector<double> qd_heff;
  /// @brief qd_heff's eigenvalues, ascending: the SC-QD-NEVPT2 total
  /// energies (hartree).
  std::vector<double> qd_energies;
  /// @brief qd_heff's eigenvectors over the CASSCF roots, n x n column-major:
  /// state k's at [k n, (k + 1) n), signed so that its largest-magnitude
  /// component is positive.
  std::vector<double> qd_vectors;
  /// @brief PC's ⟨I|V|Ψ_J⁽¹⁾⟩ per class and ket J's cut counts, as
  /// @ref qd_couplings: `nevpt2::pc_effective_hamiltonian`'s. Empty, as are
  /// the three below, when PC was refused (@ref pc_qd_refusal).
  std::vector<nevpt2::PcClassEnergies> pc_qd_couplings;
  /// @brief PC's symmetrized effective Hamiltonian, as @ref qd_heff.
  std::vector<double> pc_qd_heff;
  /// @brief pc_qd_heff's eigenvalues, ascending (hartree).
  std::vector<double> pc_qd_energies;
  /// @brief pc_qd_heff's eigenvectors, as @ref qd_vectors.
  std::vector<double> pc_qd_vectors;
  /// @brief Why PC-QD was refused, `nevpt2::pc_effective_hamiltonian`'s
  /// message naming the class and the state; empty where it was not.
  std::string pc_qd_refusal;
};

/// @brief Strongly and partially contracted NEVPT2 of every root of the
/// @ref CasscfOutput, on its orbitals, over the @ref EriOutput's B. Both
/// variants come from one pass, since they share every input.
struct Nevpt2Step {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "nevpt2";

  /// @brief Also SC- and PC-QD-NEVPT2 of the roots together: their effective
  /// Hamiltonians over the state-averaged orbitals, and its eigenvalues and
  /// eigenvectors (@ref Nevpt2Output::qd_heff, ...). Needs two roots or
  /// more.
  bool quasi_degenerate = false;

  /// @brief @ref frozen_core's value for "auto": each element's chemical
  /// core, PySCF's `chemcore` (none for H to Be, 1s for B to Mg, ...).
  static constexpr int32_t kFrozenCoreAuto = -1;

  /// @brief How many of the lowest canonical core orbitals NEVPT2 leaves
  /// uncorrelated, or @ref kFrozenCoreAuto. They are excluded from every
  /// perturber class with a core index (Si, Sijrs, Sijr, Srsi, Sij, Sir) and
  /// stay in the core density, so in the inactive Fock that Dyall's
  /// Hamiltonian and the one-electron blocks are built over. The core is
  /// canonicalized whole, then the lowest are frozen (README "Frozen
  /// core"). 0, the default, correlates every core orbital.
  int32_t frozen_core = 0;

  /// @brief A `frozen_core` value (a count, or @ref kFrozenCoreAuto)
  /// resolved over @p molecule, each atom's chemical core for "auto", and
  /// checked against the CASSCF's @p ncore core orbitals. One rule for every
  /// step that reads the key (`[nevpt2]`, `[nevpt2_gradient]`, and the QD
  /// MECI of `[optimize]` through the latter).
  ///
  /// @return The count frozen, 0 to @p ncore; InvalidConfig, naming
  ///         @p step, for one past @p ncore ("auto" resolving past it
  ///         included) or below kFrozenCoreAuto.
  [[nodiscard]] static Result<int32_t> resolve_frozen_core(std::string_view step,
                                                           int32_t frozen_core,
                                                           const Molecule &molecule, int32_t ncore);
  /// @brief The same over the molecule's @p atoms alone, as a footprint has
  /// them (FootprintDimensions::atoms).
  [[nodiscard]] static Result<int32_t> resolve_frozen_core(std::string_view step,
                                                           int32_t frozen_core,
                                                           std::span<const Atom> atoms,
                                                           int32_t ncore);

  /// @brief Each root's SC-NEVPT2, PySCF's `mrpt.NEVPT(mc, root=r)` over
  /// density fitting, and its PC-NEVPT2. Per root, on res.stream():
  ///
  /// - the core and the virtuals canonicalized over **that root's** Fock
  ///   F^I + F^A(γ_r), its own 1-RDM's, not the state average's
  ///   (`CasscfOutput::mo_energies`), by @ref canonicalize_core_virtual; the
  ///   active orbitals are left as optimized;
  /// - γ_r and Γ_r by `ci::Rdm`, dm3, f3ca and f3ac by `nevpt2::rdms`;
  /// - the MO blocks of B over the root's orbitals by `nevpt2::mo_blocks`,
  ///   h2e by `nevpt2::h2e`, and h1e and the `h1e_v_*` blocks from the
  ///   inactive Fock F^I (`response::fock`'s) in the root's orbitals;
  /// - `nevpt2::class_energies` and `nevpt2::pc_class_energies` over them.
  ///
  /// The (pq|uv) and (pu|qv) the Fock reads are one `casscf_integrals` over
  /// the CASSCF's orbitals, fitted, shared by every root.
  ///
  /// With @ref frozen_core, the lowest n_f canonical core orbitals are left
  /// out of what the layer walks: the MO blocks are over the orbitals from
  /// n_f on, and e_core, `h1e_v_si` and `h1e_v_sir` over the correlated core
  /// alone. F^I, so h1e and every `h1e_v_*` value, is still over the whole
  /// core.
  ///
  /// With @ref quasi_degenerate, then: the inputs above once over the
  /// CASSCF's own orbitals (core and virtuals canonical over the
  /// state-averaged Fock, `CasscfOutput::mo_energies`); per unordered pair
  /// I < J the transition RDMs ⟨I|…|J⟩ and ⟨J|…|I⟩ by `nevpt2::rdms`, and
  /// `nevpt2::sc_effective_hamiltonian` and `pc_effective_hamiltonian` over
  /// the pair's 2 x 2 block, each root's own RDMs on its diagonal; then each
  /// symmetrized H_eff's eigenpairs, by `diagonalize`. Synchronizes
  /// res.stream() many times.
  ///
  /// @return The class energies per root; InvalidConfig, before anything is
  ///         queued, for @ref quasi_degenerate over fewer than two roots, or
  ///         a @ref frozen_core (resolved) past the CASSCF's core or below
  ///         kFrozenCoreAuto;
  ///         Unsupported, before anything is
  ///         queued, for an open-shell ScfOutput (n_alpha != n_beta), for a
  ///         molecule of multiplicity other than 1, and under an
  ///         integral-direct EriOutput, naming `[eri]`; or
  ///         the error of the first module call that failed:
  ///         `casscf_integrals`', `ci`'s, `response::fock`'s, the
  ///         canonicalization's or `nevpt2`'s, and GpuRuntime for a failed
  ///         copy or BLAS call. A Numerical error from
  ///         `nevpt2::pc_class_energies` alone does not fail the step: it is
  ///         logged as a warning and that root's PC left empty
  ///         (@ref Nevpt2Output::pc_roots); nor does one from
  ///         `nevpt2::pc_effective_hamiltonian`, which leaves PC-QD empty
  ///         (@ref Nevpt2Output::pc_qd_refusal).
  [[nodiscard]] Result<Nevpt2Output> run(View<MoleculeOutput, BasisSetOutput, EriOutput,
                                              CoreHamiltonianOutput, ScfOutput, CasscfOutput> in,
                                         DeviceResources &res) const;

  /// @brief What @ref run holds at @p dims, all of it freed on return (the
  /// product is on the host): the fitted (pq|uv), (pu|qv) and Hamiltonian,
  /// the strings, the CI vectors of the planned CASSCF's roots
  /// (dims.casscf_nroots) and the per-root arrays (every root's RDM slots
  /// with @ref quasi_degenerate); then per root, its γ and Γ, its Fock and
  /// canonical orbitals, its MO blocks and h2e, nevpt2::rdms at one
  /// determinant, and the SC and PC classes at a batch of one (the adaptive
  /// parts at the least they run with); with @ref quasi_degenerate, the
  /// state-averaged inputs, both effective Hamiltonians' diagonal walk with
  /// PC's kets kept, a pair's transition RDMs and couplings, and the two
  /// eigensolves. PC's every class is counted at its whole basis, no mode
  /// cut, the most it holds. Sized over the active space at @p dims, ncore
  /// from the electrons, and @ref frozen_core over dims.atoms. Empty for what
  /// @ref run refuses before allocating: an open shell, no density fitting,
  /// @ref quasi_degenerate under two roots, or a frozen core past the core.
  [[nodiscard]] MemoryFootprint footprint(const FootprintDimensions &dims) const;
};

} // namespace free_breakfast
