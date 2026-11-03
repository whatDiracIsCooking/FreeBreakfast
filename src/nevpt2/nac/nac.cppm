/// @file
/// @brief The QD-NEVPT2 nonadiabatic (derivative) coupling, SC and PC, on
/// the device (Park 2020, Eqs. 45-54):
///
///     d_PQ = <Psi_P|dPsi_Q/dR> = h_PQ / (E_Q - E_P) + d^CAS + d^PT2,
///
/// h_PQ the interstate coupling (qd_sc_coupling, qd_pc_coupling), d^CAS and
/// d^PT2 mcscf::df_nac's combination at T^PQ = 1/2 (R_P R_Q^T - R_Q R_P^T)
/// and at the PT2 transition density D^PT2_xy = Σ_MN T_MN ⟨Ψ_M⁽¹⁾|E_xy|N⟩,
/// which this module also computes, over the effective Hamiltonians' own
/// amplitudes.
///
/// The README has the per-class contractions and the conventions.

export module free_breakfast.nevpt2.nac;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.mcscf.solve;
import free_breakfast.mcscf.nac;
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

/// @brief A QD-NEVPT2 coupling between two QD states and its pieces, each
/// flat and row-major as @c (n_atom, 3), as mcscf::Nac's: element 3 A + k is
/// the derivative along R_{A,k}, atoms in the molecule's order.
struct QdNac {
  /// @brief h_PQ = R_P^T (dH_eff/dR) R_Q (hartree / bohr), unscaled:
  /// symmetric in P and Q, and translation-invariant. Empty under the fold,
  /// as are cas and pt2.
  std::vector<double> numerator;
  /// @brief d^CAS = Σ_MN T^PQ_MN d_MN (1 / bohr), d_MN the roots' total
  /// couplings (their CSF terms in it), unscaled whatever the options.
  std::vector<double> cas;
  /// @brief d^PT2 = Σ_xy D^PT2_xy <phi_x|d phi_y / dR> with the orbitals
  /// relaxed (1 / bohr): the basis functions' motion and the SA orbitals'
  /// response, unscaled whatever the options.
  std::vector<double> pt2;
  /// @brief The CSF parts of both: d^CAS's Σ_MN T^PQ_MN d^CSF_MN plus
  /// d^PT2's basis motion, what use_etfs leaves out (1 / bohr).
  std::vector<double> csf;
  /// @brief The coupling as mcscf::NacOptions asks: by default d_PQ =
  /// numerator / (E_Q - E_P) + cas + pt2; under mult_ediff numerator +
  /// (E_Q - E_P) (cas + pt2); use_etfs takes csf out of cas + pt2.
  std::vector<double> total;
  /// @brief |E_Q - E_P|, H_eff's eigenvalues: E_Q - E_P is this for P < Q,
  /// its negative for P > Q (the eigenvalues ascend).
  double heff_gap = 0.0;
  /// @brief The smallest |E_N - E_M| of the CASSCF roots over the pairs
  /// with T^PQ_MN != 0 (hartree).
  double root_gap = std::numeric_limits<double>::infinity();
  /// @brief h_PQ's max |zeta_sr|.
  double max_zeta = 0.0;
  /// @brief The three Z-vectors' conjugate-gradient products: h_PQ's, then
  /// d^CAS's and d^PT2's (mcscf::df_nac's). Under the fold, the one solve's
  /// in slot 0, and 0 in the others.
  std::array<int32_t, 3> iterations{};
  /// @brief The three Z-vectors' residuals, in @ref iterations' order.
  std::array<double, 3> residual{};
};

/// @brief The SC-QD-NEVPT2 derivative coupling d_PQ = <Psi_P|dPsi_Q/dR>
/// between QD states @p p (the bra) and @p q (the ket), H_eff's eigenvalues
/// ascending, on the DF-SA-CASSCF @p solution:
///
///     d_PQ = h_PQ / (E_Q - E_P) + d^CAS + d^PT2.
///
/// In order: h_PQ by qd_sc_coupling, whose R (each column's largest
/// component positive) gives T^PQ = 1/2 (R_P R_Q^T - R_Q R_P^T); D^PT2 by
/// sc_pt2_transition_density at T^PQ over nevpt2::qd_inputs; then
/// mcscf::df_nac's combination twice, at (T^PQ, no D^X) for d^CAS and at
/// (T = 0, D^PT2) for d^PT2, so that the two are reported apart: three
/// Z-vector solves in all. README.md has the derivation and the
/// conventions.
///
/// The arguments are qd_sc_coupling's, with @p options mcscf::df_nac's.
/// d_PQ = -d_QP. Its sign follows R's and the solution's CI phases; the
/// call fixes none.
///
/// @p fold solves one Z-vector instead of three (Park 2020 Eqs. 51-52):
/// qd_sc_coupling with ΔE times mcscf::nac_combination_terms' RDMs at
/// (T^PQ, D^PT2) and D^PT2's source folded into its Lagrangian. Then only
/// total, csf, heff_gap, root_gap and max_zeta are valid: numerator, cas
/// and pt2 are empty, and iterations and residual are the one solve's, in
/// slot 0 (slots 1 and 2 zero). It is off by default: README.md "The fold"
/// has the measurement.
///
/// @return The coupling; qd_sc_coupling's errors (Numerical, naming both,
///         for QD states within mcscf::kRootGapFloor; InvalidConfig for
///         p = q or a state not one of the roots'); Numerical, naming the
///         pair, for two CASSCF roots closer than kRootGapFloor with
///         T^PQ_MN != 0 (mcscf::df_nac's); or the first error of
///         qd_inputs, the density or mcscf::df_nac.
[[nodiscard]] Result<QdNac>
qd_sc_nac(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
          std::span<const PrimitivePairClass> classes, const BasisSet &aux,
          const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
          const mcscf::Solution &solution, int32_t ncore, int32_t ncas, int32_t p, int32_t q,
          int32_t batch, const mcscf::NacOptions &options = {}, bool fold = false);

/// @brief The PC-QD-NEVPT2 derivative coupling: qd_sc_nac with
/// qd_pc_coupling and pc_pt2_transition_density, R PC's H_eff's
/// eigenvectors signed as SC's. The arguments are qd_sc_nac's.
///
/// @return The coupling; a refused PC class's Numerical error,
///         qd_pc_energy's unchanged, never a coupling with the class zeroed;
///         qd_sc_nac's other errors.
[[nodiscard]] Result<QdNac>
qd_pc_nac(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
          std::span<const PrimitivePairClass> classes, const BasisSet &aux,
          const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
          const mcscf::Solution &solution, int32_t ncore, int32_t ncas, int32_t p, int32_t q,
          int32_t batch, const mcscf::NacOptions &options = {}, bool fold = false);

} // namespace free_breakfast::nevpt2
