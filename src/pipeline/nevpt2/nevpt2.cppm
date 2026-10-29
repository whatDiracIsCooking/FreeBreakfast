/// @file
/// @brief The `Nevpt2` step: strongly and partially contracted NEVPT2 of
/// each root of the @ref CasscfOutput, on its state-averaged orbitals, over the density-fitted
/// @ref EriOutput the CASSCF itself ran on.
///
/// Device-shaped. `nevpt2`'s layer does the work (the MO blocks and slabs,
/// dm3 and f3, the eight classes, SC and PC); the step builds what it reads of each
/// root: the root's own canonical core and virtuals, its RDMs and the
/// one-electron blocks of the inactive Fock. See README.md.

export module free_breakfast.pipeline.nevpt2;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
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
  /// PC was refused as Numerical (a class's metric with no clean gap above
  /// its null space); its SC energies stand.
  std::vector<std::optional<nevpt2::PcClassEnergies>> pc_roots;
  /// @brief Why root r's PC was refused, `nevpt2::pc_class_energies`'
  /// message naming the class; empty where pc_roots[r] holds a value.
  std::vector<std::string> pc_refusals;
};

/// @brief Strongly and partially contracted NEVPT2 of every root of the
/// @ref CasscfOutput, on its orbitals, over the @ref EriOutput's B. No
/// options: the table is the request, and both variants come from one pass,
/// since they share every input.
struct Nevpt2Step {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "nevpt2";

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
  /// the CASSCF's orbitals, fitted, shared by every root. Synchronizes
  /// res.stream() many times.
  ///
  /// @return The class energies per root; Unsupported, before anything is
  ///         queued, for an open-shell ScfOutput (n_alpha != n_beta), and
  ///         under an integral-direct EriOutput, naming `[eri]`; or
  ///         the error of the first module call that failed:
  ///         `casscf_integrals`', `ci`'s, `response::fock`'s, the
  ///         canonicalization's or `nevpt2`'s, and GpuRuntime for a failed
  ///         copy or BLAS call. A Numerical error from
  ///         `nevpt2::pc_class_energies` alone does not fail the step: it is
  ///         logged as a warning and that root's PC left empty
  ///         (@ref Nevpt2Output::pc_roots).
  [[nodiscard]] Result<Nevpt2Output> run(View<MoleculeOutput, BasisSetOutput, EriOutput,
                                              CoreHamiltonianOutput, ScfOutput, CasscfOutput> in,
                                         DeviceResources &res) const;
};

} // namespace free_breakfast
