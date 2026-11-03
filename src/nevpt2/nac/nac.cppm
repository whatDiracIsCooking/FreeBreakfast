/// @file
/// @brief The QD-NEVPT2 nonadiabatic coupling's pieces. For now the PT2
/// transition density D^PT2_xy = Σ_MN T_MN ⟨Ψ_M⁽¹⁾|E_xy|N⟩, SC and PC, on
/// the device, over the effective Hamiltonians' own amplitudes.
///
/// The README has the per-class contractions and the conventions.

export module free_breakfast.nevpt2.nac;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.nevpt2.density_fitting;
import free_breakfast.nevpt2.energy;

export namespace free_breakfast::nevpt2 {

/// @brief The SC-QD-NEVPT2 PT2 transition density of n states that share
/// orbitals,
///
///     D_xy = Σ_MN T_MN ⟨Ψ_M⁽¹⁾|E_xy|N⟩,
///
/// E_xy spin summed, Ψ_M⁽¹⁾ state M's SC first-order wave function as
/// sc_effective_hamiltonian builds it, and |N⟩ state N's reference.
///
/// @p pairs is sc_effective_hamiltonian's, n x n, [M n + N] bra M, ket N,
/// read the same way: dm1 and dm2 off the diagonal are the transition RDMs
/// ⟨M|E_pq|N⟩, ..., and every pair shares the integrals and orbital energies.
/// @p t is T, n x n row-major and antisymmetric (|T_MN + T_NM| at most 1e-12
/// of its largest element, taken as ½(T − Tᵀ)), so the diagonal, which would
/// need ⟨Ψ_M⁽¹⁾|E_xy|M⟩, never enters. @p batch is the energy's slab batch.
///
/// The result is nmo x nmo row-major, nmo = ncore + ncas + nvirt over
/// @p blocks' orbitals in that order, D_xy at [x nmo + y]. It is zero
/// outside the (active, core), (virtual, active) and (virtual, core) blocks:
/// E_xy|N⟩ lies outside the first-order interacting space everywhere else.
/// Those blocks are the Si, Sr and Sir classes' (the README has each
/// contraction), over the T-weighted transition RDMs Σ_N T_MN γ^(MN) and
/// sc_effective_hamiltonian's amplitudes, which this call runs to get them
/// (ScAmplitudes); every launch is on res.stream().
///
/// @return The density; InvalidConfig when @p t is not n x n or not
///         antisymmetric; or sc_effective_hamiltonian's errors, a slab's and
///         einsum's.
[[nodiscard]] Result<Buffers::device<double>>
sc_pt2_transition_density(DeviceResources &res, const MoBlocks &blocks,
                          std::span<const EnergyInputs> pairs, std::span<const double> t,
                          int32_t batch);

/// @brief The PC-QD-NEVPT2 PT2 transition density: sc_pt2_transition_density
/// with Ψ_M⁽¹⁾ state M's PC first-order wave function, as
/// pc_effective_hamiltonian solves it (its cut modes cut, PcAmplitudes).
///
/// Its arguments and result are sc_pt2_transition_density's. No slab is
/// walked: each block is a product of the amplitudes, the solve's eigenbasis
/// and the T-weighted transition RDMs.
///
/// @return The density; sc_pt2_transition_density's InvalidConfig; or
///         pc_effective_hamiltonian's errors, a refused class of any state
///         included: the density is refused with the energy's error.
[[nodiscard]] Result<Buffers::device<double>>
pc_pt2_transition_density(DeviceResources &res, const MoBlocks &blocks,
                          std::span<const EnergyInputs> pairs, std::span<const double> t,
                          int32_t batch);

} // namespace free_breakfast::nevpt2
