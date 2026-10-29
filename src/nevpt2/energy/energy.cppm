/// @file
/// @brief The eight SC- and PC-NEVPT2 class energies on the GPU, over
/// density-fitted slabs of their external integrals.
///
/// A transcription of the NEVPT2 demo's energy.cpp, energy_sc_classes.cppm and
/// PC classes, themselves the NumPy reference's einsums, each issued through
/// recipes.einsum with its subscript string unchanged. The README has the
/// layouts, the class order and what is not ported.

export module free_breakfast.nevpt2.energy;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.nevpt2.density_fitting;

export namespace free_breakfast::nevpt2 {

/// @brief What the eight classes read besides the external integrals, all on
/// the device. c core, a active, v virtual orbitals, as in the MoBlocks the
/// slabs come from.
///
/// The integrals are column-major, first index fastest, as
/// nevpt2.density_fitting writes them. The RDMs are **row-major**, PySCF's
/// layout, as nevpt2::rdms writes dm3, f3ca and f3ac.
struct EnergyInputs {
  /// @brief The active h1eff, a x a.
  const double *h1e = nullptr;
  /// @brief h2e[t, u, v, w] = (tv|uw), a⁴, as nevpt2::h2e writes it.
  const double *h2e = nullptr;
  /// @brief The canonical core orbital energies, c.
  const double *e_core = nullptr;
  /// @brief The canonical virtual orbital energies, v.
  const double *e_virt = nullptr;
  /// @brief h1eff[v, a] - Σ_b (vb|ba), v x a: Sr's one-electron part.
  const double *h1e_v_sr = nullptr;
  /// @brief h1eff[a, c], a x c: Si's.
  const double *h1e_v_si = nullptr;
  /// @brief h1eff[v, c], v x c: Sir's.
  const double *h1e_v_sir = nullptr;
  /// @brief The 1-RDM, a², row-major.
  const double *dm1 = nullptr;
  /// @brief The 2-RDM, a⁴, row-major, `make_dm123`'s convention.
  const double *dm2 = nullptr;
  /// @brief The 3-RDM, a⁶, row-major, `make_dm123`'s, not reordered.
  const double *dm3 = nullptr;
  /// @brief PySCF's f3ca, a⁶, row-major.
  const double *f3ca = nullptr;
  /// @brief PySCF's f3ac, a⁶, row-major.
  const double *f3ac = nullptr;
};

/// @brief Each class's norm and energy in PySCF's order -- Sr, Si, Sijrs,
/// Sijr, Srsi, Srs, Sij, Sir -- and the correlation energy, their sum.
struct ClassEnergies {
  /// @brief Σ_k N_k per class.
  std::array<double, 8> norms{};
  /// @brief -Σ_k N_k / (Δ_k + H_k / N_k) per class, in hartree.
  std::array<double, 8> energies{};
  /// @brief The SC-NEVPT2 correlation energy, Σ of @ref energies.
  double total = 0.0;
};

/// @brief The eight SC-NEVPT2 class energies of one state, each class walking
/// its external index in batches of at most @p batch, building each slab with
/// nevpt2::slab over @p blocks.
///
/// @p batch <= 0 takes each class's whole extent in one slab. A class's
/// RDM-only intermediates are built once, before its walk; its slab, and for
/// Sir and Srsi the second slab, are device buffers of the batch's size,
/// allocated per class. Every launch is on res.stream(), and each slab's norm
/// and H vectors are read back to the host for PySCF's `_norm_to_energy`, so
/// the call synchronizes once per slab.
///
/// @return The norms and energies; or the first error of a slab build, an
///         einsum (InvalidConfig only for a size past its limits) or a launch.
[[nodiscard]] Result<ClassEnergies> class_energies(DeviceResources &res, const MoBlocks &blocks,
                                                   const EnergyInputs &in, int32_t batch);

/// @brief Each class's PC-NEVPT2 energy in PySCF's order -- Sr, Si, Sijrs,
/// Sijr, Srsi, Srs, Sij, Sir -- and the correlation energy, their sum.
struct PcClassEnergies {
  /// @brief -Σ_t b_tᵀ (K + Δ_t S)⁺ b_t per class, in hartree.
  std::array<double, 8> energies{};
  /// @brief The modes of each class's metric S cut as null; 0 for Sijrs,
  /// which has none.
  std::array<int64_t, 8> dropped{};
  /// @brief The PC-NEVPT2 correlation energy, Σ of @ref energies.
  double total = 0.0;
};

/// @brief The eight PC-NEVPT2 class energies of one state: the classes of
/// class_energies, each over the whole span of its internal functions rather
/// than SC's one contraction of them.
///
/// Each class's metric S and Dyall Hamiltonian K are SC's, built once and
/// solved once (two symmetric eigendecompositions on the device); each slab
/// is then one einsum into that eigenbasis, finished on the host. Sijrs, with
/// no active index, is SC's. The walk, @p batch and the synchronization are
/// class_energies'.
///
/// S is singular by construction. Its modes at or below 1e-13 of its largest
/// eigenvalue are cut as null, and a class whose smallest kept eigenvalue is
/// not 1e3 times its largest cut one (or, nothing cut, the cut itself) is
/// refused, since its energy would depend on where the cut fell.
///
/// @return The energies; a Numerical error naming the class when its metric
///         has no positive eigenvalue or no gap at the cut, or a denominator
///         λ + Δ is not positive; or class_energies' errors, and the
///         eigensolver's.
[[nodiscard]] Result<PcClassEnergies> pc_class_energies(DeviceResources &res,
                                                        const MoBlocks &blocks,
                                                        const EnergyInputs &in, int32_t batch);

} // namespace free_breakfast::nevpt2
