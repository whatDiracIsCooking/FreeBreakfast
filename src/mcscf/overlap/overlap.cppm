/// @file
/// @brief The overlaps S_IJ = <I(R)|J(R')> between the CAS states of two
/// CASSCF solutions, each in its own basis (two geometries, or two basis
/// sets): the mixed AO overlap, the MO overlap over core and active, and the
/// determinant overlaps contracted with the CI vectors. See README.md for the
/// conventions.

export module free_breakfast.mcscf.overlap;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.mcscf.solve;

export namespace free_breakfast::mcscf {

/// @brief The overlaps S_IJ = <I|J> between every root I of @p bra, a CASSCF
/// solution over @p bra_basis, and every root J of @p ket, one over
/// @p ket_basis: row-major n_bra x n_ket, element I n_ket + J.
///
/// The two bases may differ in geometry, basis set or both; each solution's
/// orbitals are over its own. Both have @p ncore core and @p ncas active
/// orbitals (columns [0, ncore) and [ncore, ncore + ncas) of each C) and
/// @p nelecas active electrons of the spin 2S + 1 = @p multiplicity, so the
/// same determinant space. The steps: the spherical mixed AO overlap
/// (@c overlap_mixed_spherical), the MO overlap over the ncore + ncas
/// occupied orbitals (two gemms), downloaded; on the host, the core block's
/// determinant once and the active block with the core eliminated (its Schur
/// complement), then each pair of alpha strings' and of beta strings'
/// determinant of that block's occupied rows and columns; uploaded, and
/// contracted with the two roots' CI vectors in three gemms. README.md has
/// the determinant order, why the loop is on the host, and the core
/// condition.
///
/// No phase is fixed: S_IJ changes sign with either root's CI phase as the
/// solutions carry it, and with nothing else (the orbitals' own signs and
/// any rotation among the core or among the active orbitals cancel). With
/// the same solution and basis on both sides S is the identity to rounding.
///
/// Queued on res.stream(), which the call synchronizes; the mixed AO overlap
/// is a buffer of its own, the other intermediates res's workspace.
///
/// @return The overlaps; InvalidConfig if a solution has no roots, a C is not
///         its basis's nao_sphe() square or not on @p res's device, ncore or
///         ncas is out of range for either, @p nelecas is no electron count of
///         @p multiplicity in ncas orbitals, or a solution's CI vectors are
///         not whole roots of that determinant space; Unsupported if the two
///         core blocks overlap with |det| below 1e-6 (the core orbitals not
///         carried into each other, README.md); otherwise
///         @c overlap_mixed_spherical's or @c ci::StringSet::create's error,
///         or the GpuRuntime error of the first copy or gemm that failed.
[[nodiscard]] Result<std::vector<double>>
state_overlaps(DeviceResources &res, const BasisSet &bra_basis, const Solution &bra,
               const BasisSet &ket_basis, const Solution &ket, int32_t ncore, int32_t ncas,
               int32_t nelecas, int32_t multiplicity);

} // namespace free_breakfast::mcscf
