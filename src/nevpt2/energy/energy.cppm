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

/// @brief The SC-QD-NEVPT2 second-order effective Hamiltonian of n states that
/// share orbitals: element [I n + J] holds, per class, ⟨I|V|Ψ_J⁽¹⁾⟩, the
/// unsymmetrized correction with I the bra (row) and J the ket (column).
///
/// @p pairs is n x n row-major, pairs[I n + J] the EnergyInputs of the ordered
/// pair (I, J): every field but the RDMs is the states' shared one, the same
/// pointer in every pair, and dm1, dm2 and dm3 are the transition RDMs
/// between ⟨I| and |J⟩ in nevpt2::rdms' bra/ket convention (dm1[p, q] =
/// ⟨I|E_pq|J⟩, dm2's order), so pairs[J n + J] is what class_energies takes
/// for state J. f3ca and f3ac are read only on the diagonal and may be null
/// off it: a coupling's denominator is the ket's own.
///
/// Per class and perturber t, with N^J_t, H^J_t and Δ_t state J's own, as
/// class_energies forms them, the coupling is
/// -Σ_t T^t_IJ / (Δ_t + H^J_t / N^J_t) over the t with |N^J_t| > 1e-14, and
/// T^t_IJ = ⟨I|V P_t V|J⟩ is the class's norm einsum over the transition
/// RDMs with two changes at I != J: its terms with no RDM, which carry
/// ⟨I|J⟩ = 0, are dropped (hdm1's and dm1_h's 2δ, hdm2's δδ, Sir's
/// h1e_v·h1e_v, all of Sijrs); and T is the mean of the einsum over (I, J)
/// and (J, I), since PySCF folds Sr's, Si's and Sir's Hermitian-conjugate
/// cross terms into one, which holds only at I = J. So T is symmetric in I
/// and J; the matrix is not.
///
/// Each element is a ClassEnergies: norms[k] is Σ_t T^t_IJ, energies[k] the
/// coupling and total their sum; the diagonal element is state J's
/// class_energies. Each class walks its slabs once for every pair, in batches
/// of at most @p batch as class_energies does, holding every state's Dyall
/// blocks (a16, ...) and every pair's norm intermediates (dm3_h, ...) at
/// once, and its per-slab tables grow as n² + n.
///
/// @return The n² elements; InvalidConfig when @p pairs is empty or not n²
///         long, or when a pair's shared field is not the first pair's; or
///         class_energies' errors.
[[nodiscard]] Result<std::vector<ClassEnergies>>
sc_effective_hamiltonian(DeviceResources &res, const MoBlocks &blocks,
                         std::span<const EnergyInputs> pairs, int32_t batch);

/// @brief Where class_energies_adjoint adds the bars dE/dx of the class-level
/// inputs, all on the device, each in its EnergyInputs field's layout (or,
/// for @ref lambda, nevpt2::slab_adjoint's). Every bar is added onto, so the
/// caller zeroes it; a null one is not wanted and is not formed.
///
/// Every EnergyInputs field has one. The bars on h1e, h2e and the RDMs are
/// the dm1-, ..., f3ac-shaped raw derivatives of the energy as a function of
/// each array's every element: no symmetry of an RDM is assumed or imposed.
struct EnergyBars {
  /// @brief On EnergyInputs::h1e, a x a column-major.
  double *h1e = nullptr;
  /// @brief On EnergyInputs::h2e, a⁴ column-major.
  double *h2e = nullptr;
  /// @brief On EnergyInputs::dm1, a² row-major.
  double *dm1 = nullptr;
  /// @brief On EnergyInputs::dm2, a⁴ row-major.
  double *dm2 = nullptr;
  /// @brief On EnergyInputs::dm3, a⁶ row-major.
  double *dm3 = nullptr;
  /// @brief On EnergyInputs::f3ca, a⁶ row-major.
  double *f3ca = nullptr;
  /// @brief On EnergyInputs::f3ac, a⁶ row-major.
  double *f3ac = nullptr;
  /// @brief On EnergyInputs::h1e_v_sr, v x a.
  double *h1e_v_sr = nullptr;
  /// @brief On EnergyInputs::h1e_v_si, a x c.
  double *h1e_v_si = nullptr;
  /// @brief On EnergyInputs::h1e_v_sir, v x c.
  double *h1e_v_sir = nullptr;
  /// @brief On the core orbital energies, c.
  double *e_core = nullptr;
  /// @brief On the virtual orbital energies, v.
  double *e_virt = nullptr;
  /// @brief Λ = dE/dB through every slab, naux x nmo² in B's MO layout, as
  /// nevpt2::slab_adjoint adds onto it.
  double *lambda = nullptr;
};

/// @brief class_energies_adjoint's result: the energies it walked, and each
/// class's Park Eq. 40 assembled from its bars.
struct ClassEnergiesAdjoint {
  /// @brief The norms and energies, as class_energies computes them.
  ClassEnergies energies;
  /// @brief Σ_t (P_t N_t + Q_t H_t) per class, assembled from the bars: each
  /// class's Σ_G <M_G, G> over the blocks G it contracts, plus its
  /// block-free terms against their multipliers (Sir's one-electron norm
  /// term), and for Sijrs, which reads no block, ½ <g-bar, g>. Equal to
  /// @ref energies' energies to round-off.
  std::array<double, 8> eq40{};
};

/// @brief The SC reverse pass of class_energies: @p bars += dE/dx for E the
/// SC correlation energy and x each input @p bars names, every EnergyInputs
/// field and B (through Λ).
///
/// Each class's walk is class_energies', in batches of at most @p batch.
/// Per slab, the norms and H are evaluated, the finish's multipliers P, Q
/// and R (Park Eqs. 36-38) taken on the host, and every norm and H einsum
/// differentiated per operand by rewriting its subscripts; the slab's bar is
/// then pulled back onto @p bars.lambda through nevpt2::slab_adjoint before
/// the next slab is built. R goes onto the orbital energies. Srsi walks its
/// slabs twice, since its symmetrized pairs span them. The bars on the
/// class blocks (a16, ...) are formed as well, for Eq. 40, and then swept
/// back through the blocks' builders, in reverse topological order, onto
/// h1e, h2e and the RDMs; that sweep is skipped when none of those seven
/// bars is wanted.
///
/// @return The energies and Eq. 40; or class_energies' errors, and the
///         slab adjoint's.
[[nodiscard]] Result<ClassEnergiesAdjoint>
class_energies_adjoint(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                       int32_t batch, const EnergyBars &bars);

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

/// @brief pc_class_energies_adjoint's result: the energies it walked, and each
/// class's degree-one homogeneity assembled from its bars.
struct PcClassEnergiesAdjoint {
  /// @brief The energies and dropped counts, as pc_class_energies computes
  /// them.
  PcClassEnergies energies;
  /// @brief Σ_G <M_G, G> per class over the blocks G its S, K and S₁ are
  /// built from (a16, dm3, dm2 for Sr, ...): a class energy is homogeneous of
  /// degree one in (S, S₁, K) at fixed slabs and orbital energies, so this
  /// equals @ref energies' energies to round-off. Sijrs, which has no solve,
  /// carries SC's ½ <g-bar, g>.
  std::array<double, 8> homogeneity{};
};

/// @brief The PC reverse pass of pc_class_energies: @p bars += dE/dx for E the
/// PC correlation energy and x each input @p bars names, as
/// class_energies_adjoint does for SC.
///
/// Each class solves as pc_class_energies does and walks its slabs in batches
/// of at most @p batch. A tuple's energy is -c bᵀ (K + Δ S)⁺ b, so its
/// derivative is the solve's adjoint, -c (2 xᵀ db - xᵀ dA x) with
/// x = (K + Δ S)⁺ b, the solve the energy ran: no eigenvalue or eigenvector
/// derivative. Per slab, the eigenbasis rows y are evaluated, the multipliers
/// taken on the host, and every slab einsum differentiated per operand onto
/// the slab (and through nevpt2::slab_adjoint onto @p bars.lambda), h1e_v's
/// rows, and T and t1; the orbital energies take dE/dΔ. After the walk the
/// bars on S, K and S₁ go onto the class blocks SC's reverse pass has (a16,
/// ..., and the RDMs Sr, Srsi and Sir read directly), and SC's sweep takes
/// them onto h1e, h2e and the RDMs, skipped when none of those seven bars is
/// wanted. Sijrs is SC's.
///
/// The derivative is exact where the energy is defined: S's kept rank locally
/// constant (the gap at the cut) and, for the bars on the RDMs, K annihilating
/// S's dropped modes, which holds at an eigenstate. A class the energy refuses
/// is refused here with the energy's error, never given a zero.
///
/// @return The energies and the homogeneity sums; or pc_class_energies'
///         errors, and the slab adjoint's. On an error @p bars is unspecified.
[[nodiscard]] Result<PcClassEnergiesAdjoint>
pc_class_energies_adjoint(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                          int32_t batch, const EnergyBars &bars);

} // namespace free_breakfast::nevpt2
