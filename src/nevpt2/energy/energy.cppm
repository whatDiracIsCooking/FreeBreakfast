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
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.nevpt2.density_fitting;

export namespace free_breakfast::nevpt2 {

/// @brief What the eight classes read besides the external integrals, all on
/// the device. c core, a active, v virtual orbitals, as in the MoBlocks the
/// slabs come from. Under a frozen core c is the correlated core alone, and
/// h1e and every h1e_v_* value stay the inactive Fock's over the whole core
/// (README "Frozen core").
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
  /// @brief PC's conditioning floor, relative to each metric's largest
  /// eigenvalue: S's modes at or below max(the default cut, @ref pc_floor)
  /// of it are cut (README "The conditioning floor", #917). 0, the default,
  /// leaves the cut as it is, bitwise; the default cut itself is #923's
  /// decision. Read by PC alone, by its energy, reverse pass and tangent
  /// alike; unread by SC and in an EnergyDirection.
  double pc_floor = 0.0;
};

/// @brief Each class's norm and energy in PySCF's order -- Sr, Si, Sijrs,
/// Sijr, Srsi, Srs, Sij, Sir -- and the correlation energy, their sum.
struct ClassEnergies {
  /// @brief Σ_k N_k per class.
  std::array<double, 8> norms{};
  /// @brief -Σ_k N_k / (Δ_k + H_k / N_k) per class, in hartree; a class
  /// whose perturbers span one function each (Srs at two active electrons,
  /// Sij at two active holes) keeps every k at H_k / N_k = η, its one
  /// function's Dyall energy read from its blocks (README "One-function
  /// classes", #949).
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

/// @brief Each state's SC first-order wave function in the seven classes
/// with an active index: per perturber t, Ψ_J⁽¹⁾ holds c_t P_t V|J⟩ with
///
///     c_t = -1 / (Δ_t + H^J_t / N^J_t),
///
/// state J's own denominator, and c_t = 0 where its norm N^J_t vanishes
/// (|N^J_t| <= 1e-14, the energy's guard); in a one-function class
/// (ClassEnergies::energies) -1 / (Δ_t + η^J) for every t.
/// sc_effective_hamiltonian hands it over: for the PT2 transition density
/// ⟨Ψ_J⁽¹⁾|E_xy|K⟩ (Sr, Si and Sir), and as the kets a later call over the same orbitals reads with its
/// diagonal omitted. A class with no perturber holds n empty lists.
struct ScAmplitudes {
  /// @brief [J]: Sr's c over the virtual r, nvirt.
  std::vector<std::vector<double>> sr;
  /// @brief [J]: Si's c over the core i, ncore.
  std::vector<std::vector<double>> si;
  /// @brief [J]: Sijr's c over (r, j <= i), r outermost, then j, then i.
  std::vector<std::vector<double>> sijr;
  /// @brief [J]: Srsi's c over (r <= s, i), r outermost, then s, then i.
  std::vector<std::vector<double>> srsi;
  /// @brief [J]: Srs's c over (r, s), nvirt x nvirt row-major.
  std::vector<std::vector<double>> srs;
  /// @brief [J]: Sij's c over (i, j), ncore x ncore row-major.
  std::vector<std::vector<double>> sij;
  /// @brief [J]: Sir's c over (r, i), nvirt x ncore row-major, [r ncore + i].
  std::vector<std::vector<double>> sir;
};

/// @brief Each state's PC first-order wave function in the seven classes with
/// an active index, in its own solve's eigenbasis: per tuple t, Ψ_J⁽¹⁾ holds -Σ_k g_tk Σ_μ U_J[μ, k]
/// |φ^J_μ⟩, φ^J_μ the class's internal functions on |J⟩ (the basis S_J is
/// the metric of), with
///
///     g_tk = y_{J,tk} / (λ_{J,k} + Δ_t),   y_{J,t} = U_Jᵀ b_{J,t},
///
/// the amplitudes pc_effective_hamiltonian's energy is -Σ_k y_tk g_tk of.
/// S_J's cut modes are not in U_J, as the solve cut them. Besides the PT2
/// transition density (Sr, Si and Sir), they are the kets a later
/// pc_effective_hamiltonian over the same orbitals reads with its diagonal
/// omitted: its couplings need only U_J and g.
struct PcAmplitudes {
  /// @brief One class's amplitudes, one entry per state J; empty when the
  /// class has no tuple (no core orbital, say).
  struct Class {
    /// @brief U_J, d x m_J column-major, as the solve leaves it.
    std::vector<Buffers::device<double>> u;
    /// @brief g, tuples x m_J row-major: Sr's over r, Si's over i, Sir's over
    /// (r, i) at r ncore + i.
    std::vector<std::vector<double>> g;
  };
  /// @brief d = n³, basis (p, q, r) row-major.
  Class sr;
  /// @brief d = n³, basis (p, q, r) row-major.
  Class si;
  /// @brief d = n, basis p; tuples (r, j, i) row-major, every ordered (j, i).
  Class sijr;
  /// @brief d = n, basis p; tuples (r, s, i) row-major, every ordered (r, s).
  Class srsi;
  /// @brief d = n², basis (p, q) row-major; tuples (r, s) row-major.
  Class srs;
  /// @brief d = n², basis (p, q) row-major; tuples (i, j) row-major.
  Class sij;
  /// @brief d = 2 n², basis (x, p, q) row-major, x the spin coupling.
  Class sir;
};

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
/// -Σ_t T^t_IJ / (Δ_t + H^J_t / N^J_t) over the t with |N^J_t| > 1e-14 (every
/// t, at η^J, in a one-function class: ClassEnergies::energies), and
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
/// When @p amplitudes is not null it is filled with each state's first-order
/// wave function (ScAmplitudes); the elements are the same either way.
///
/// **An entry may be omitted**, its dm1 null: nothing of it is built or
/// walked, and its element is zero. Either every diagonal entry is given or
/// none is, and an off-diagonal (I, J) only beside (J, I), whose mean it
/// reads. With the diagonal omitted, ket J's c_t are read from
/// @p amplitudes, which must then hold what a call over the same blocks and
/// shared inputs with those diagonals filled: element (I, J) is
/// Σ_t T^t_IJ c^J_t, the same sum, and Sijrs, diagonal alone, is not walked.
/// So n states' couplings take one call over the n diagonals (off-diagonals
/// omitted), then one per pair over (I, J) and (J, I) alone, each state's
/// class walk run once and only two transition sets held at a time.
///
/// @return The n² elements; InvalidConfig when @p pairs is empty or not n²
///         long, when a pair's shared field is not the first pair's, when the
///         omitted entries are not as above, or when the diagonal is omitted
///         and @p amplitudes is null or not n states'; or class_energies'
///         errors.
[[nodiscard]] Result<std::vector<ClassEnergies>>
sc_effective_hamiltonian(DeviceResources &res, const MoBlocks &blocks,
                         std::span<const EnergyInputs> pairs, int32_t batch,
                         ScAmplitudes *amplitudes = nullptr);

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

/// @brief One direction class_energies_tangent (and pc_class_energies_tangent)
/// differentiates along: a tangent of every EnergyInputs field and of B. A
/// null field is zero.
struct EnergyDirection {
  /// @brief The fields' tangents, each in its EnergyInputs field's layout.
  EnergyInputs in;
  /// @brief B's tangent over the MO blocks, shaped as the call's blocks and
  /// laid out as mo_blocks lays B out (vc B_cv's pair swapped); null is zero.
  const MoBlocks *blocks = nullptr;
};

/// @brief class_energies_tangent's result: the energies it walked, and per
/// class each perturber's norm N, H and Δ with their tangents, and Ė.
///
/// A class's perturbers are ScAmplitudes' order (Sir's at r ncore + i), the
/// same whatever the batch. Sijrs has no N, H or Δ tables (no guard, a
/// kernel of its own): its tables are empty and its Ė alone is given.
struct ClassEnergiesTangent {
  /// @brief The norms and energies, as class_energies computes them.
  ClassEnergies energies;
  /// @brief [k][t]: perturber t's N, H and Δ in class k, as the finish reads
  /// them (Sijr's and Srsi's symmetrized pairs combined).
  std::array<std::vector<double>, 8> norm, h, delta;
  /// @brief [k][t m + u], m directions: their tangents along direction u, as
  /// the walk forms them, also where N is guarded.
  std::array<std::vector<double>, 8> norm_dot, h_dot, delta_dot;
  /// @brief [u][k]: Ė per class along direction u, the guarded perturbers
  /// contributing nothing.
  std::vector<std::array<double, 8>> energy_dot;
  /// @brief [u]: Σ_k energy_dot[u][k].
  std::vector<double> total_dot;
};

/// @brief The SC forward tangent of class_energies along each of
/// @p directions: per perturber (Ṅ, Ḣ, Δ̇), and Ė per class.
///
/// Ė is the finish differentiated in the (N, d) form, d = NΔ + H and
/// E = -N² / d per perturber, so no 1/N appears:
///
///     Ė = (-2N/d + N²Δ/d²) Ṅ + (N²/d²) Ḣ + (N³/d²) Δ̇,
///
/// over the perturbers whose norm the energy keeps (|N| > 1e-14); a guarded,
/// empty one gives zero. A one-function class's is c Ṅ + N c² (Δ̇ + η̇),
/// c = -1 / (Δ + η), η̇ from its blocks' tangents. Ṅ and Ḣ are every norm
/// and H einsum differentiated operand by operand, each at its operand's
/// tangent: the slabs' (bilinear in B), h1e_v's rows, the RDMs a class reads directly, and every class block's
/// (a16, ...). A block is of degree at most two in (h1e, h2e, RDMs), one
/// integral times one RDM, so its tangent is its own builder's central
/// polarization, (G(x + s ẋ) - G(x - s ẋ)) / 2s, exact; a slab's is
/// (slab(B + s Ḃ) - slab(B) - s² slab(Ḃ)) / s, exact. Each s is a power of
/// two that sizes s ẋ to x, so the differences cancel nothing past
/// round-off. Sijrs's Ė is one kernel's.
///
/// Each class is walked once for all the directions, in batches of at most
/// @p batch as class_energies: its blocks built once, and per direction
/// their tangents, held through the walk (memory grows as the directions;
/// a caller with many chunks them); per slab, the slab built once, then per
/// direction its tangent and the tangent tables, read back.
///
/// @return The energies and tangents; InvalidConfig when a direction's
///         blocks are not shaped as @p blocks; or class_energies' errors.
[[nodiscard]] Result<ClassEnergiesTangent>
class_energies_tangent(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                       int32_t batch, std::span<const EnergyDirection> directions);

/// @brief An empty perturber (|N| <= 1e-14) is populated by a direction w
/// when its N^{ww} is above this fraction of w's population scale: between
/// the round-off a direction that keeps the symmetry leaves on it and the
/// smallest real population (README, "The SC Hessian-vector product").
constexpr double kEmptyPopulationFloor = 1e-10;

/// @brief class_energies_hvp's result: the energies it walked, and per
/// direction what its empty perturbers took.
struct ClassEnergiesHvp {
  /// @brief The norms and energies, as class_energies computes them.
  ClassEnergies energies;
  /// @brief [u][k]: class k's empty perturbers (|N| <= 1e-14) direction u
  /// populates, each given K6's multipliers.
  std::vector<std::array<int64_t, 8>> populated;
  /// @brief [u]: the largest N^{uu} over the empty perturbers u populates
  /// (0 when none), and u's population scale, the larger of its largest
  /// |N^{uu}| over every perturber and the largest |N| over (8 s)^2, which
  /// @ref kEmptyPopulationFloor is a fraction of (0 when nothing is empty).
  /// A direction that moves nothing has a round-off scale of its own, so a
  /// caller with several directions of one size compares population
  /// against the largest scale.
  std::vector<double> population;
  std::vector<double> scale;
  /// @brief [u]: the empty perturbers' ambiguity on the pair (u, its
  /// partner) relative to the pair's Hessian scale: max_t ρ_τ w_t / S over
  /// the empty perturbers both populate, w_t = √(N^{uu} N^{vv}) / |Δ + η|
  /// and S the largest such w over every perturber. 0 with no partner, or
  /// none both populate. Not "no Hessian on the pair": a bound on the
  /// relative error K6 at the mean η leaves there.
  std::vector<double> defect;
  /// @brief The kept perturbers' worst relative rounding bound on N,
  /// max_t σ_t / |N_t| over every class whose finish divides by its N (not
  /// Sijrs, nor a one-function class): σ_t = ε/2 absolute in Sr, Si and Sir,
  /// whose norm and H einsums cancel O(1) terms, and 4ε |N_t| in the others
  /// (README "The SC-QD reverse pass", Conditioning; #1055). Each walk of the
  /// tables shifts η_t = H_t / N_t by about that, so a second derivative
  /// assembled from separate walks inherits it (hessian's README, "The SC
  /// conditioning refusal"). The ket is 0.
  double rounding = 0.0;
  /// @brief Where @ref rounding is: {ket, class, perturber}, the ket -1 when
  /// nothing is kept.
  std::array<int64_t, 3> rounding_at{-1, 0, 0};
  /// @brief That perturber's N.
  double rounding_norm = 0.0;
};

/// @brief The SC energy's Hessian-vector product in its inputs, forward over
/// the reverse pass: @p bars[u] += the tangent of class_energies_adjoint's
/// bars along @p directions[u], every field the bars name (Λ included).
///
/// Per perturber, with (P, Q, R) the finish's derivatives in (N, H, Δ), the
/// bars are Σ_t P_t ∂N_t + Q_t ∂H_t + R_t ∂Δ_t, so their tangent is two
/// terms. The multipliers' tangent: the finish's 3 x 3 Hessian in (N, H, Δ),
/// in the (N, d) form (d = NΔ + H, no 1/N), times class_energies_tangent's
/// (Ṅ, Ḣ, Δ̇), through one more reverse pass at those multipliers. And the
/// bars at fixed multipliers differenced along the direction: every table is
/// a polynomial of degree at most six in (B, h1e, h2e, h1e_v, RDMs), so a
/// symmetric difference at ±s, ±2s, ±3s, Richardson-weighted, is exact, six
/// reverse passes at shifted inputs. Sijrs's share is one kernel's.
///
/// **Empty perturbers.** One the energy's guard skips (|N| <= 1e-14; a
/// one-function class has none, and its η's bars on its blocks are fixed
/// multipliers like P and Q), empty
/// at a symmetric geometry, has N, H and their first derivatives zero; a
/// direction that breaks the symmetry populates it at second order, where
/// its energy is homogeneous of degree two. Along u alone that is a
/// quadratic form, and the perturber takes K6: the finish's multipliers at
/// their limit along the ray, P̄ = 2T̄ + ΔT̄² and Q̄ = T̄², T̄ = -1 / (Δ + η),
/// η = H^{uu} / N^{uu} from N's and H's second derivatives along u. With
/// @p partners, one per direction, the pair (u, v) is checked: on a
/// perturber both populate, the energy is a quadratic form on the pair only
/// when H is proportional to N there. Each takes η the mean of η_u and η_v,
/// and its scale-free defect ρ_τ (zero exactly then), weighted by its share
/// of the pair's Hessian, √(N^{uu} N^{vv}) / |Δ + η|, against the largest
/// such share over every perturber, refuses the call past 1e-6.
/// ⟨v, HVP u⟩ is then the Hessian element to that relative bound.
///
/// Memory and cost: per direction, seven reverse passes over the whole
/// inputs, and with any empty perturber six class_energies_tangent calls
/// (twelve with a partner).
///
/// @return The energies and, per direction, the populated empty perturbers;
///         InvalidConfig when @p bars is not one per direction, @p partners
///         is neither empty nor one per direction, or a direction's blocks
///         are not shaped as @p blocks; Numerical, naming the class and the
///         perturber, when an empty perturber's weighted defect on a pair
///         is past 1e-6; or the tangent's and reverse pass's errors.
[[nodiscard]] Result<ClassEnergiesHvp>
class_energies_hvp(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                   int32_t batch, std::span<const EnergyDirection> directions,
                   std::span<const EnergyBars> bars,
                   std::span<const EnergyDirection> partners = {});

/// @brief sc_effective_hamiltonian_adjoint's result: the matrix it walked,
/// and each class's Eq. 40 assembled from its bars.
struct ScEffectiveHamiltonianAdjoint {
  /// @brief The n² elements, as sc_effective_hamiltonian computes them.
  std::vector<ClassEnergies> elements;
  /// @brief Σ_IJ H̄_IJ times class k's element (I, J), assembled from the
  /// bars as class_energies_adjoint's Eq. 40 is: each coupling is
  /// homogeneous of degree one in its transition norm and its ket's N and H.
  std::array<double, 8> eq40{};
};

/// @brief The reverse pass of sc_effective_hamiltonian: the bars dL/dx of
/// L = Σ_IJ H̄_IJ H_IJ, H_IJ element (I, J)'s total, on every input of every
/// pair. class_energies_adjoint is n = 1 with H̄ = 1, einsum for einsum.
///
/// @p pairs is sc_effective_hamiltonian's, every entry given (none
/// omitted). @p hbar is H̄, n x n row-major,
/// [I n + J] on the unsymmetrized element (I, J). @p bars holds n
/// EnergyBars, one per state, each added onto as class_energies_adjoint's:
/// bars[J]'s dm1, dm2, dm3, f3ca and f3ac take state J's own RDMs' bars; the
/// rest (h1e, h2e, h1e_v_*, the orbital energies, Λ) are the bars on what
/// every pair shares, summed over the pairs, read from bars[0] and the same
/// pointer in every entry. The transition RDMs' bars are not held: each
/// ordered pair (I, J), I != J, is handed to @p transition as an EnergyBars
/// whose dm1, dm2 and dm3 alone are set, device buffers queued on
/// res.stream() and freed when the walk moves on, so the callee works on that
/// stream or synchronizes. A null @p transition forms none of them.
///
/// The walk is sc_effective_hamiltonian's per class, with its finish
/// differentiated: a coupling -T_IJ / D_J, D_J = Δ + H^J / N^J, puts
/// -H̄_IJ / D_J on T_IJ, and so half on each of the norm einsums of (I, J)
/// and (J, I), and A_J = Σ_I H̄_IJ T_IJ / D_J² on D_J, which state J's own
/// tables then take as -A_J H^J / (N^J)² on N^J, A_J / N^J on H^J and A_J on
/// Δ. It runs in three parts. Each state's own norm and H tables, perturber
/// by perturber, on the host (n > 1 only). Then one unordered pair M < N at a
/// time: every class's slabs walked once, the norm einsums of (M, N) and
/// (N, M) evaluated and differentiated onto the slabs (and so Λ), h1e_v's
/// rows and the pair's metrics, A added to both kets' tables, and the
/// metrics' bars swept onto the pair's transition RDMs and handed to
/// @p transition, (M, N) then (N, M), before the next pair. Last, each
/// state's own walk, as class_energies_adjoint's with A_J's share added,
/// and its sweep onto bars[J]. Sijrs reads no RDM: its bars are Σ_J H̄_JJ
/// times class_energies_adjoint's. Every slab is rebuilt in each part.
///
/// @return The elements and Eq. 40; InvalidConfig when @p pairs is not n²
///         (sc_effective_hamiltonian's checks), @p hbar not n² or @p bars not
///         n long, or an entry of @p bars does not share the first's
///         h1e, h2e, h1e_v_*, orbital-energy and Λ bars; or
///         class_energies_adjoint's errors, and @p transition's.
[[nodiscard]] Result<ScEffectiveHamiltonianAdjoint> sc_effective_hamiltonian_adjoint(
    DeviceResources &res, const MoBlocks &blocks, std::span<const EnergyInputs> pairs,
    std::span<const double> hbar, int32_t batch, std::span<const EnergyBars> bars,
    const std::function<Status(int64_t bra, int64_t ket, const EnergyBars &transition)>
        &transition);

/// @brief sc_effective_hamiltonian_tangent's result: the matrix it walked,
/// every pair's transition tables and their tangents, and Ċ per element.
///
/// A class's tuples are ClassEnergiesTangent's order (ScAmplitudes', Sir's
/// at r ncore + i), whatever the batch; Sijrs has no tables (it couples no
/// pair), its diagonal Ċ the kets'.
struct ScEffectiveHamiltonianTangent {
  /// @brief The n² elements, as sc_effective_hamiltonian computes them.
  std::vector<ClassEnergies> elements;
  /// @brief [J]: state J's own walk, class_energies_tangent of pair (J, J)
  /// along its directions: the N^J, H^J and Δ every coupling with ket J
  /// divides by, their tangents, and the diagonal element's Ė.
  std::vector<ClassEnergiesTangent> kets;
  /// @brief [I n + J][k][t]: T^t_IJ, the mean of the norm einsums of (I, J)
  /// and (J, I), so the same at (J, I); on the diagonal N^J_t, kets[J]'s.
  std::vector<std::array<std::vector<double>, 8>> transition;
  /// @brief [I n + J][k][t m + u], m directions: Ṫ^t_IJ along direction u,
  /// also where ket J's norm is guarded; on the diagonal kets[J]'s Ṅ.
  std::vector<std::array<std::vector<double>, 8>> transition_dot;
  /// @brief [J][k]: ket J's tuples the guard drops (|N^J_t| <= 1e-14),
  /// ascending; none in a one-function class. Every coupling (I, J) skips
  /// them, so they put nothing in @ref element_dot: their Ṫ^t_IJ, in
  /// @ref transition_dot, is ⟨Φ^I_t|∂Φ^J_t⟩, the overlap a refusal of the
  /// coupling's second order reads.
  std::vector<std::array<std::vector<int64_t>, 8>> empty;
  /// @brief [u][I n + J][k]: Ċ_IJ per class along direction u; on the
  /// diagonal kets[J]'s Ė.
  std::vector<std::vector<std::array<double, 8>>> element_dot;
  /// @brief [u][I n + J]: Σ_k element_dot[u][I n + J][k].
  std::vector<std::vector<double>> total_dot;
};

/// @brief The SC forward tangent of sc_effective_hamiltonian along each of
/// m directions: per pair and tuple Ṫ^t_IJ, per ket (Ṅ, Ḣ, Δ̇), and Ċ per
/// element and class.
///
/// @p pairs is sc_effective_hamiltonian's, every entry given (none omitted).
/// @p directions holds m n² EnergyDirections, [u n² + I n + J] pair (I, J)'s
/// along direction u: its RDMs' tangents those of the pair's transition
/// RDMs (bilinear in the bra and ket, so the bra/ket nevpt2::rdms at the
/// dotted bra plus at the dotted ket), f3ca's and f3ac's read on the
/// diagonal alone; and, like the pairs, every other field and the blocks'
/// tangent the same in every pair of one direction.
///
/// Each state's own walk is class_energies_tangent's. Each unordered pair
/// M < N then walks every class's slabs once for all the directions: the
/// norm einsums of (M, N) and (N, M) differentiated operand by operand, each
/// side's metric blocks at their own builders' polarization along that
/// side's RDM tangents (class_energies_tangent's rules), T and Ṫ their
/// means. A coupling C = -T N / d, d = N Δ + H (ket N's own), is finished in
/// that (N, d) form, with no 1/N:
///
///     Ċ = -(N/d) Ṫ - (T H/d²) Ṅ + (T N/d²) Ḣ + (T N²/d²) Δ̇,
///
/// over the tuples whose ket norm the coupling keeps (|N| > 1e-14); a
/// one-function class's is c Ṫ + T c² (Δ̇ + η̇), c = -1 / (Δ + η) and η̇
/// from the ket's blocks' tangents, as class_energies_tangent's. A guarded
/// tuple gives zero and is reported (@ref ScEffectiveHamiltonianTangent::empty).
///
/// @return The elements and tangents; InvalidConfig when @p pairs is not as
///         sc_effective_hamiltonian_adjoint takes it, @p directions is not a
///         whole number of n², or one direction's shared fields or blocks
///         differ between two of its pairs; or class_energies_tangent's
///         errors.
[[nodiscard]] Result<ScEffectiveHamiltonianTangent>
sc_effective_hamiltonian_tangent(DeviceResources &res, const MoBlocks &blocks,
                                 std::span<const EnergyInputs> pairs, int32_t batch,
                                 std::span<const EnergyDirection> directions);

/// @brief sc_effective_hamiltonian_hvp's result: the matrix it walked, and
/// per direction what the kets' empty tuples took.
struct ScEffectiveHamiltonianHvp {
  /// @brief The n² elements, as sc_effective_hamiltonian computes them.
  std::vector<ClassEnergies> elements;
  /// @brief [u][J][k]: ket J's empty tuples (|N^J| <= 1e-14) direction u
  /// populates, each given K6 on its own tables and in every coupling with
  /// ket J.
  std::vector<std::vector<std::array<int64_t, 8>>> populated;
  /// @brief [u]: the largest of ClassEnergiesHvp::defect over the kets, on
  /// (u, its partner); 0 with no partner.
  std::vector<double> defect;
  /// @brief [u]: the same ρ_τ weighted by the tuple's share of a coupling's
  /// first-order scale, max(|Ṫ_u|, |Ṫ_v|) / |Δ + η| against the largest over
  /// the coupling's tuples, the largest over every coupling (I, J), I != J;
  /// 0 with no partner.
  std::vector<double> coupling_defect;
  /// @brief [u]: K6's share of ⟨u, HVP u⟩ through the couplings,
  /// Σ_{I != J} H̄_IJ (T̄ T^{uu} - ρ η T̄² N^{uu} + ρ T̄² H^{uu}) over ket J's
  /// tuples empty for bra I too that u populates; what a check of the
  /// product sees of it.
  std::vector<double> coupling_k6;
  /// @brief The kept perturbers' worst relative rounding bound on N,
  /// max_t σ_t / |N_t| over every class whose finish divides by its N (not
  /// Sijrs, nor a one-function class): σ_t = ε/2 absolute in Sr, Si and Sir,
  /// whose norm and H einsums cancel O(1) terms, and 4ε |N_t| in the others
  /// (README "The SC-QD reverse pass", Conditioning; #1055). Each walk of the
  /// tables shifts η_t = H_t / N_t by about that, so a second derivative
  /// assembled from separate walks inherits it (hessian's README, "The SC
  /// conditioning refusal"). Over the kets
  /// H̄ weighs (Σ_I |H̄_IJ| > 0), whose bars the product forms.
  double rounding = 0.0;
  /// @brief Where @ref rounding is: {ket, class, perturber}, the ket -1 when
  /// nothing is kept.
  std::array<int64_t, 3> rounding_at{-1, 0, 0};
  /// @brief That perturber's N.
  double rounding_norm = 0.0;
};

/// @brief The Hessian-vector product of L = Σ_IJ H̄_IJ H_IJ in its inputs,
/// forward over sc_effective_hamiltonian_adjoint: along each of m
/// directions, the tangent of every bar that reverse pass forms.
/// class_energies_hvp is its n = 1, H̄ = 1, Sijrs's kernel and all.
///
/// @p pairs and @p hbar are sc_effective_hamiltonian_adjoint's, @p directions
/// sc_effective_hamiltonian_tangent's (m n², [u n² + I n + J]). @p bars holds
/// m n EnergyBars, [u n + J] direction u's as that reverse pass's bars[J]:
/// the shared fields' bars one pointer within each u (checked). Each ordered
/// pair's transition bars' tangent is handed to @p transition(u, bra, ket,
/// bars), dm1, dm2 and dm3 alone set, as the reverse pass hands its own
/// (device buffers on res.stream(), freed after the call); null: not formed.
/// @p partners is none or m n², pair entries as @p directions'.
///
/// Per tuple, the coupling g = -T N / d (d = N Δ + H, the ket's) and the
/// diagonal (g at T = N) are differentiated twice in the (N, d) form
/// (docs/nevpt2/nevpt2.tex Eq. qd-scfinish): the multipliers' tangent times
/// sc_effective_hamiltonian_tangent's (Ṫ, Ṅ, Ḣ, Δ̇), through one reverse
/// pass at those multipliers; the T bars split ½/½ onto (M, N)'s and (N,
/// M)'s norm einsums, the kets' own tables collecting from every bra. The
/// bars at fixed multipliers are then differenced along each direction,
/// exactly, as class_energies_hvp's are, each pair moved along its own
/// RDMs' tangent. A one-function class is c = -1 / (Δ + η^J) throughout.
///
/// **Empty tuples, on every root.** A tuple a ket's guard drops that a
/// direction populates (class_energies_hvp's rule, over that ket's tables)
/// takes K6 in every coupling with that ket, its own diagonal included
/// (Part I's), whatever H̄ weighs it: T̄ = -1 / (Δ + η) on T, and where the
/// bra's tuple is empty too, -ρ η T̄² on N and ρ T̄² on H, ρ = T^{uu} /
/// N^{uu}, the limit along the ray. With @p partners each ket's tuples
/// both populate are scored as class_energies_hvp scores them, and per bra
/// by their share of the coupling's first order: past 1e-6 the call is
/// refused, the coupling's score first (Eq. qd-scfail).
///
/// Memory and cost: per direction, n own walks and n(n - 1)/2 pair walks at
/// fixed multipliers, each seven times; with any empty tuple, Part I's six
/// tangent calls per ket (twelve with a partner), and six pair walks per
/// pair whose tuples are empty for both.
///
/// @return The elements and the empty tuples' counts and scores;
///         InvalidConfig as sc_effective_hamiltonian_tangent's, for @p bars not
///         m n or not sharing one direction's shared bars, or @p partners not
///         empty or m n²; Numerical, naming the coupling (bra, ket), the class,
///         the tuple and the directions, when a populated empty tuple with a
///         first-order transition tangent is not proportional on a pair, or
///         naming the state, the class and the perturber for the diagonal's
///         defect; or the tangent's and reverse pass's errors, and
///         @p transition's.
[[nodiscard]] Result<ScEffectiveHamiltonianHvp>
sc_effective_hamiltonian_hvp(DeviceResources &res, const MoBlocks &blocks,
                             std::span<const EnergyInputs> pairs, std::span<const double> hbar,
                             int32_t batch, std::span<const EnergyDirection> directions,
                             std::span<const EnergyBars> bars,
                             const std::function<Status(int64_t direction, int64_t bra, int64_t ket,
                                                        const EnergyBars &transition)> &transition,
                             std::span<const EnergyDirection> partners = {});

/// @brief One perturber function of a PC class, mode k of tuple t in its
/// ket's eigenbasis: its denominator λ_k + Δ_t and its first-order amplitude
/// t = y_tk / (λ_k + Δ_t), the weight of the normalized perturber in Ψ⁽¹⁾.
struct PcMode {
  double denominator = 0.0;
  double amplitude = 0.0;
};

/// @brief Each class's PC-NEVPT2 energy in PySCF's order -- Sr, Si, Sijrs,
/// Sijr, Srsi, Srs, Sij, Sir -- and the correlation energy, their sum.
struct PcClassEnergies {
  /// @brief -Σ_t b_tᵀ (K + Δ_t S)⁺ b_t per class, in hartree.
  std::array<double, 8> energies{};
  /// @brief The modes of each class's metric S cut as null; 0 for Sijrs,
  /// which has none.
  std::array<int64_t, 8> dropped{};
  /// @brief Each class's mode of largest |amplitude|, what the refusal
  /// bounds (|t| <= 1), and its mode nearest a pole, smallest
  /// |denominator|; zero for Sijrs, a class with no tuple, and a ket read
  /// from cached amplitudes. In an effective Hamiltonian, ket J's at every
  /// [I n + J], as @ref dropped.
  std::array<PcMode, 8> strongest{};
  std::array<PcMode, 8> nearest{};
  /// @brief Each class's conditioning, relative to its metric's largest
  /// eigenvalue: @ref cut the largest eigenvalue the cut dropped (a few
  /// ε √d for exact null modes, more for real ones; 0 when nothing was
  /// dropped), @ref kept the smallest it kept, what the solve's conditioning
  /// follows. Both 0 for Sijrs, a class with no tuple, and a ket read from
  /// cached amplitudes; per ket as @ref dropped. README "The conditioning
  /// floor": a PC Hessian's ladder needs @ref kept well above the default
  /// cut, and the ratio kept / cut does not tell (it is near 1 wherever real
  /// modes run through the cut, at any floor).
  std::array<double, 8> cut{};
  std::array<double, 8> kept{};
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
/// S is singular by construction. Its modes at or below max(1e-13, 100 ε √d,
/// in.pc_floor) of its largest eigenvalue are cut as null: at least 13 times
/// above every mode round-off decides, measured (README "PC-NEVPT2"). There
/// is no gap test: the real modes may run through the cut, and carry no
/// energy there. Each class's conditioning is reported (PcClassEnergies::cut,
/// kept).
///
/// @return The energies; a Numerical error naming the class when its metric
///         has no positive eigenvalue, or a ket meets a pole (a denominator
///         λ + Δ at round-off, or an intruder, |y / (λ + Δ)| > 1; no sign
///         test, README "Poles, not signs"); or class_energies' errors, and
///         the eigensolver's.
[[nodiscard]] Result<PcClassEnergies> pc_class_energies(DeviceResources &res,
                                                        const MoBlocks &blocks,
                                                        const EnergyInputs &in, int32_t batch);

/// @brief The PC-QD-NEVPT2 second-order effective Hamiltonian of n states that
/// share orbitals: element [I n + J] holds, per class, ⟨I|V|Ψ_J⁽¹⁾⟩, the
/// unsymmetrized correction with I the bra (row) and J the ket (column).
///
/// @p pairs is sc_effective_hamiltonian's, read the same way, f3ca and f3ac
/// on the diagonal only.
///
/// Per class, state J's metric S_J and Dyall Hamiltonian K_J are its own, as
/// pc_class_energies builds them, and are cut and solved once per state
/// (K_J U_J = S_J U_J diag(λ_J)), n solves per class. Per pair and tuple t,
/// with b_{J,t} = S_J x_t + S₁_J h1_t,
///
///     ⟨I|V|Ψ_J⁽¹⁾⟩_t = -c b_{I←J,t}ᵀ (K_J + Δ_t S_J)⁺ b_{J,t},
///     b_{I←J,t} = S^{(J,I)} x_t + S₁^{(J,I)} h1_t,
///
/// S^{(J,I)} and S₁^{(J,I)} the class's S and S₁ over the transition RDMs of
/// pair (J, I) (bra J, ket I), unsymmetrized and without their terms that
/// read no RDM, which carry ⟨J|I⟩ = 0 (hdm1's 2δ, hdm2's δδ). Sijr's and
/// Srsi's "2 direct - exchange" becomes the bilinear 2 y_I y_J - y_I y_J′
/// over the tuple's exchange partner; Sijrs, with no active index, is SC's,
/// zero off the diagonal. No (I, J) / (J, I) mean is taken, so the matrix is
/// not symmetric.
///
/// Each element is a PcClassEnergies: energies[k] the coupling, dropped[k]
/// ket J's cut count, total their sum; the diagonal element is state J's
/// pc_class_energies. Each class walks its slabs once for every pair, in
/// batches of at most @p batch, one einsum per pair per slab into J's
/// eigenbasis, holding every state's solve and every pair's transition rows
/// (d x m each) across the walk.
///
/// When @p amplitudes is not null it is filled with each state's first-order
/// wave function, in its own solve's eigenbasis (PcAmplitudes); the elements
/// are the same either way, and a refusal leaves it unspecified.
///
/// Entries may be omitted as sc_effective_hamiltonian's: with the diagonal
/// omitted, ket J's U_J and g are read from @p amplitudes, which a call with
/// those diagonals filled, and element (I, J) is -c Σ_k y_{I←J,tk} g_{J,tk}
/// (Sijr's and Srsi's y (2 g - g′)); no state is solved, so none is refused.
///
/// @return The n² elements; sc_effective_hamiltonian's InvalidConfig for
///         malformed @p pairs or omissions; or pc_class_energies' errors, a refused class
///         (S_J with no positive eigenvalue, or a pole: λ + Δ at round-off, or an
///         intruder) naming the class and the state J.
[[nodiscard]] Result<std::vector<PcClassEnergies>>
pc_effective_hamiltonian(DeviceResources &res, const MoBlocks &blocks,
                         std::span<const EnergyInputs> pairs, int32_t batch,
                         PcAmplitudes *amplitudes = nullptr);

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
/// The derivative is exact where S's kept rank is locally constant and, for
/// the bars on the RDMs, K annihilates S's dropped modes, which holds at an
/// eigenstate for its null modes. With real modes dropped at the cut it is
/// validated at PSB3 to the numerical-gradient bar, and keeping them does not
/// move it (README "PC-NEVPT2"). A class the energy refuses
/// is refused here with the energy's error, never given a zero.
///
/// @return The energies and the homogeneity sums; or pc_class_energies'
///         errors, and the slab adjoint's. On an error @p bars is unspecified.
[[nodiscard]] Result<PcClassEnergiesAdjoint>
pc_class_energies_adjoint(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                          int32_t batch, const EnergyBars &bars);

/// @brief pc_class_energies_tangent's result: the energies it walked, and Ė
/// per class along each direction.
struct PcClassEnergiesTangent {
  /// @brief The energies, dropped counts and conditioning (cut, kept), as
  /// pc_class_energies computes them.
  PcClassEnergies energies;
  /// @brief [u][k]: Ė per class along direction u.
  std::vector<std::array<double, 8>> energy_dot;
  /// @brief [u]: Σ_k energy_dot[u][k].
  std::vector<double> total_dot;
};

/// @brief The PC forward tangent of pc_class_energies along each of
/// @p directions (EnergyDirection, as class_energies_tangent's): Ė per class.
///
/// Per class, the energy's own solve is reused, with no eigenvalue or
/// eigenvector derivative: per tuple -c (2 ẏᵀ g - gᵀ Ḋ g), g = y / (λ + Δ),
/// ẏ = Ṫᵀ x + Tᵀ ẋ + ṫ1ᵀ h1 + t1ᵀ ḣ1 the slab einsums at their operands'
/// tangents, and Ḋ the projected pencil's tangent Uᵀ (K̇ + Δ Ṡ) U + Δ̇ 1
/// (Sijr's and Srsi's 2 direct - exchange bilinearly). Ṡ, K̇ and Ṡ₁ are their
/// builders' polarizations, exact as class_energies_tangent's blocks are.
/// Where the cut drops real modes, U also rotates into them at first order,
/// U̇ = V_n Θ V_kᵀ U with Θ_nl = (V_nᵀ Ṡ V_k)_nl / (σ_l - σ_n): the forward
/// side of the reverse pass's kept-span term (#862), so the two are dual;
/// U̇ enters Ṫ and ṫ1 as S U̇ and S₁ᵀ U̇, and Ḋ as U̇ᵀ K U + Uᵀ K U̇. Sijrs is
/// SC's kernel. @p span false leaves U̇ out, the fixed-span tangent alone:
/// not a derivative where real modes are dropped, there for the tests that
/// show U̇'s share.
///
/// Each class is walked once for all the directions, in batches of at most
/// @p batch: its matrices' tangents and Ṫ, ṫ1 and Ḋ per direction once
/// solved (memory grows as the directions, by d x m and two m x m each, and
/// while one is built, the class's S, K and S₁ three times), and per slab
/// the slab built once, then per direction its tangent and ẏ, read back.
///
/// @return The energies and tangents; InvalidConfig when a direction's
///         blocks are not shaped as @p blocks; or pc_class_energies' errors,
///         its refusals among them (no positive metric eigenvalue, a pole),
///         never a class summed as zero.
[[nodiscard]] Result<PcClassEnergiesTangent>
pc_class_energies_tangent(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                          int32_t batch, std::span<const EnergyDirection> directions,
                          bool span = true);

/// @brief pc_class_energies_hvp's result: the energies it walked.
struct PcClassEnergiesHvp {
  /// @brief The energies, dropped counts and conditioning (cut, kept), as
  /// pc_class_energies computes them.
  PcClassEnergies energies;
};

/// @brief The PC energy's Hessian-vector product in its inputs, forward over
/// the reverse pass: @p bars[u] += the tangent of pc_class_energies_adjoint's
/// bars along @p directions[u], every field the bars name (Λ included).
///
/// Per class the bars are a polynomial in the inputs at fixed multipliers:
/// the amplitudes' bars ȳ (and the exchange rows'), M = Σ g wᵀ and
/// M_Δ = Σ Δ g wᵀ, the solve's U, and S's eigenvectors V and eigenvalues σ
/// for the kept span's rotation (#862). So the tangent is three terms.
/// - **The multipliers' tangent**, through one reverse pass at them: per
///   tuple ġ = (ẏ - Ḋ g) / (λ + Δ), the solve differentiated through its own
///   factorization, with ẏ and Ḋ pc_class_energies_tangent's (the span's
///   rotation in them); and the span bar's derivative in V and σ at fixed
///   matrices, closed form (README "The PC Hessian-vector product").
/// - **U's tangent**, U̇ = V_n Θ V_kᵀ U: the bars are quadratic in U, so one
///   symmetric difference in U is exact, two reverse passes.
/// - **The inputs' tangent at fixed multipliers**: a polynomial of degree at
///   most six along the direction, so class_energies_hvp's ±s, ±2s, ±3s
///   difference is exact, six reverse passes at shifted inputs.
/// Sijrs's share is SC's (class_energies_hvp's kernel).
///
/// The kept span's rotation is taken at second order for every dropped mode,
/// null or real: Θ and the span bar's pairs within σ's round-off rotate
/// nothing, as at first order. @p span false leaves the rotation out at both
/// orders, the fixed-span product: not a derivative where real modes are
/// dropped, there for the tests that show the rotation's share.
///
/// Cost per direction: nine reverse passes over the whole inputs, after one
/// tangent walk for every direction at once.
///
/// @return The energies; InvalidConfig when @p bars is not one per
///         direction or a direction's blocks are not shaped as @p blocks;
///         Numerical, naming the class, when a class keeps a metric mode
///         below 1e-10 of its largest, where the second derivative is not
///         attainable (#917; EnergyInputs::pc_floor raises the cut past it);
///         or pc_class_energies_tangent's errors, its
///         refusals among them.
[[nodiscard]] Result<PcClassEnergiesHvp>
pc_class_energies_hvp(DeviceResources &res, const MoBlocks &blocks, const EnergyInputs &in,
                      int32_t batch, std::span<const EnergyDirection> directions,
                      std::span<const EnergyBars> bars, bool span = true);

/// @brief pc_metric_lifts' result: per class, how S's dropped modes rise at
/// second order along one direction of the RDMs.
struct PcMetricLifts {
  /// @brief [k]: the largest eigenvalue of the Schur lift L (below), over
  /// S's largest eigenvalue; 0 where the cut drops nothing, and for Sijrs.
  std::array<double, 8> lift{};
  /// @brief [k]: the class's cut over S's largest eigenvalue,
  /// max(1e-13, 100 ε √d, pc_floor); 0 for Sijrs and a class with no tuple.
  std::array<double, 8> cut{};
  /// @brief [k]: the class's smallest kept eigenvalue over its largest
  /// (PcClassEnergies::kept); 0 where nothing was solved.
  std::array<double, 8> kept{};
  /// @brief QD only (empty at one state): [M][k], per bra M, how the ket's
  /// null modes (those the call's step reaches) reach M's rows at first
  /// order along the path, in units of
  /// its round-off ε √d / kept (V_n's accuracy):
  /// max_e ‖Eᵀ ρ‖_e / (‖A'‖ + ‖W‖ ‖A U‖) over L's eigenvectors e, with
  ///     ρ = V_nᵀ A'ᵀ - W (A U)ᵀ,  W = V_nᵀ S' U,
  /// A pair (N, M)'s transition metric (and S₁ the same), the rows' part
  /// along a null mode v moving as v' = -S⁺ S' v: zero to round-off when
  /// the lifted functions have no first-order overlap with bra M's
  /// (⟨φ|V|M⟩, Eq. qd-pcfail), of order the lift's root otherwise. 0 at
  /// M = N, for a class with no null mode, and where nothing moves.
  std::vector<std::array<double, 8>> overlap;
};

/// @brief Each PC class's Schur lift along a path of the RDMs (README "The
/// Schur lift"): with S = V diag(σ) Vᵀ cut as pc_class_energies cuts it
/// (@p in, its pc_floor among them), V_n the dropped modes and S⁺ = U Uᵀ
/// over the kept ones, and S' and S'' the metric's first and second
/// derivatives along the path,
///
///     L = V_nᵀ (S'' - 2 S' S⁺ S') V_n,
///
/// whose eigenvalues are the second derivatives of the dropped modes' own:
/// a null mode of S rises along the path as ½ t² L. @p first and @p second
/// hold the RDMs' first and second derivatives along the path (their dm1,
/// dm2 and dm3; no other field is read): S' and S'' are each class's metric
/// builder over them, linear in the RDMs.
///
/// @return The lifts; pc_class_energies' refusal of a class with no positive
///         metric eigenvalue; or a builder's or the solver's error.
[[nodiscard]] Result<PcMetricLifts> pc_metric_lifts(DeviceResources &res, const MoBlocks &blocks,
                                                    const EnergyInputs &in,
                                                    const EnergyInputs &first,
                                                    const EnergyInputs &second);

/// @brief pc_metric_lifts on every model state (../gradient/README.md, "The
/// Schur lift on every model state"): ket @p ket's lifts over pc_effective_hamiltonian's n²
/// @p pairs, its own path's first and second derivatives
/// @p first[ket n + ket] and @p second, and per bra M != ket the null modes'
/// first-order overlap with M's rows (PcMetricLifts::overlap), over pair
/// (ket, M)'s transition metrics and their first derivatives
/// @p first[ket n + M] (each EnergyInputs' dm1, dm2 and dm3 read alone),
/// taken over the null modes that reach the cut within @p step along the
/// path (½ ℓ_e step² at least the cut; infinity: every null mode). At one
/// state it is the overload above.
///
/// @return The lifts and overlaps; InvalidConfig when @p pairs and @p first
///         are not n² each or @p ket is not a state; pc_class_energies'
///         refusal; or a builder's or the solver's error.
[[nodiscard]] Result<PcMetricLifts> pc_metric_lifts(DeviceResources &res, const MoBlocks &blocks,
                                                    std::span<const EnergyInputs> pairs,
                                                    std::span<const EnergyInputs> first,
                                                    const EnergyInputs &second, int64_t ket,
                                                    double step);

/// @brief pc_effective_hamiltonian_adjoint's result: the matrix it walked, and
/// each class's homogeneity sum assembled from its bars.
struct PcEffectiveHamiltonianAdjoint {
  /// @brief The n² elements, as pc_effective_hamiltonian computes them.
  std::vector<PcClassEnergies> elements;
  /// @brief Σ_G <M_G, G> per class over the blocks of every state's own S,
  /// K and S₁ and of every transition's S and S₁: each coupling is
  /// homogeneous of degree one in its ket's (S_J, K_J, S₁_J) and pair
  /// (J, I)'s (S, S₁) together, so this equals Σ_IJ H̄_IJ times class k's
  /// element (I, J) to round-off. Sijrs carries SC's ½ <g-bar, g>.
  std::array<double, 8> homogeneity{};
};

/// @brief The reverse pass of pc_effective_hamiltonian: the bars dL/dx of
/// L = Σ_IJ H̄_IJ H_IJ, H_IJ element (I, J)'s total, on every input of every
/// pair. pc_class_energies_adjoint is n = 1 with H̄ = 1, einsum for einsum.
///
/// @p pairs, @p hbar, @p bars and @p transition are
/// sc_effective_hamiltonian_adjoint's, read the same way: bars[J]'s dm1, dm2,
/// dm3, f3ca and f3ac take state J's own RDMs' bars, bars[0]'s other fields
/// the shared ones' (the same pointer in every entry), and each ordered pair
/// (I, J), I != J, is handed to @p transition once, an EnergyBars whose dm1,
/// dm2 and dm3 alone are set, device buffers queued on res.stream() and freed
/// when the call returns. A null @p transition forms none of them.
///
/// Per class, every ket J's solve of (K_J, S_J, S₁_J) runs backwards as
/// pc_class_energies_adjoint's does: with g = y / (λ + Δ), E_IJ is
/// -c b_{I←J}ᵀ (K_J + Δ S_J)⁺ b_J, bilinear in the two rows, so pair (I, J)
/// puts -c H̄_IJ g_J on y_{I←J}, -c H̄_IJ g_{I←J} on y_J, and
/// c H̄_IJ g_{I←J} g_Jᵀ on J's M (and Δ times it on M_Δ): still no
/// eigenvalue or eigenvector derivative. The bars on T_{I←J} =
/// S^{(J,I)ᵀ} U_J and t1_{I←J} go back through that product, U_J held, onto
/// S^{(J,I)} and S₁^{(J,I)} unsymmetrized, and so onto pair (J, I)'s metric
/// blocks and its transition RDMs. There is no (I, J) / (J, I) mean, unlike
/// SC's. Each class walks its slabs once, every pair at once, as the forward
/// does; every transition's metric bars are held until all eight classes
/// have walked, then swept onto its transition RDMs and handed over in
/// row-major pair order. Sijrs is SC's, weight Σ_J H̄_JJ.
///
/// @return The elements and the homogeneity sums; InvalidConfig as
///         sc_effective_hamiltonian_adjoint's; or a refused class (S_J with no
///         positive eigenvalue, a pole), refused with
///         pc_effective_hamiltonian's error naming the class and the state,
///         never given a zero; or the slab adjoint's errors, and
///         @p transition's. On an error the bars are unspecified.
[[nodiscard]] Result<PcEffectiveHamiltonianAdjoint> pc_effective_hamiltonian_adjoint(
    DeviceResources &res, const MoBlocks &blocks, std::span<const EnergyInputs> pairs,
    std::span<const double> hbar, int32_t batch, std::span<const EnergyBars> bars,
    const std::function<Status(int64_t bra, int64_t ket, const EnergyBars &transition)>
        &transition);

/// @brief pc_effective_hamiltonian_tangent's result: the matrix it walked,
/// and Ċ per element and class.
struct PcEffectiveHamiltonianTangent {
  /// @brief The n² elements, as pc_effective_hamiltonian computes them.
  std::vector<PcClassEnergies> elements;
  /// @brief [u][I n + J][k]: Ċ_IJ per class along direction u; on the
  /// diagonal state J's pc_class_energies_tangent Ė, Sijrs's there alone.
  std::vector<std::vector<std::array<double, 8>>> element_dot;
  /// @brief [u][I n + J]: Σ_k element_dot[u][I n + J][k].
  std::vector<std::vector<double>> total_dot;
};

/// @brief The PC forward tangent of pc_effective_hamiltonian along each of
/// m directions: Ċ per element and class (docs/nevpt2/nevpt2.tex
/// Eq. qd-pcfinish's first order, over Eq. qd-rdm2's RDM tangents).
///
/// @p pairs and @p directions are sc_effective_hamiltonian_tangent's: every
/// pair given, m n² directions, [u n² + I n + J] pair (I, J)'s along u, its
/// RDMs' tangents those of the pair's transition RDMs and every other
/// field and the blocks' the same in every pair of one direction (checked).
///
/// Per class, every ket J's solve is the energy's, reused, with no
/// eigenvalue or eigenvector derivative. A coupling is -c b_{I←J}ᵀ A_J⁺ b_J,
/// A_J = K_J + Δ S_J, so with x = A_J⁺ b_J and u = A_J⁺ b_{I←J} its tangent
/// is -c (xᵀ ḃ_{I←J} + uᵀ ḃ_J - uᵀ Ȧ_J x), read in J's eigenbasis as
/// pc_class_energies_tangent reads it: ẏ_{I←J} and ẏ_J the slab einsums at
/// their operands' tangents, and Ȧ_J's projection U_Jᵀ (K̇_J + Δ Ṡ_J) U_J +
/// Δ̇. Ṡ_J, K̇_J, Ṡ₁_J are ket J's builders' polarization along its own
/// pair's directions, and S^{(J,I)} and S₁^{(J,I)}, which T_{I←J} =
/// S^{(J,I)ᵀ} U_J reads unsymmetrized, along pair (J, I)'s. Where S_J's cut
/// drops real modes U_J rotates into them, U̇_J = V_n Θ' as at n = 1, and
/// each transition row takes it through its leak V_nᵀ S^{(J,I)}: the
/// forward side of pc_effective_hamiltonian_adjoint's kept-span bar (#862).
/// @p span false leaves U̇ out, as pc_class_energies_tangent's, for the tests
/// alone. Sijrs is SC's kernel, on the diagonal.
///
/// Each class walks its slabs once for every pair and direction, as the
/// energy does: per direction, the n solves' tangents (d x m_J, two m_J x m_J
/// each) and the n² - n transition rows' (d x m_J), and per pair M and M_Δ
/// (m_J x m_J), held across the walk.
///
/// @return The elements and tangents; InvalidConfig when @p pairs or
///         @p directions are not as sc_effective_hamiltonian_tangent takes
///         them, or a direction's blocks are not shaped as @p blocks; or
///         pc_effective_hamiltonian's errors, its refusals among them (a
///         ket's metric with no positive eigenvalue, a pole), naming the
///         class and the state, never a coupling summed as zero.
[[nodiscard]] Result<PcEffectiveHamiltonianTangent>
pc_effective_hamiltonian_tangent(DeviceResources &res, const MoBlocks &blocks,
                                 std::span<const EnergyInputs> pairs, int32_t batch,
                                 std::span<const EnergyDirection> directions, bool span = true);

/// @brief The Hessian-vector product of L = Σ_IJ H̄_IJ H_IJ, H_IJ
/// pc_effective_hamiltonian's element (I, J) total, in its inputs, forward
/// over pc_effective_hamiltonian_adjoint: along each of m directions, the
/// tangent of every bar that reverse pass forms (docs/nevpt2/nevpt2.tex
/// Eq. qd-pcfinish). pc_class_energies_hvp is its n = 1, H̄ = 1.
///
/// @p pairs and @p hbar are pc_effective_hamiltonian_adjoint's, @p directions
/// pc_effective_hamiltonian_tangent's (m n², [u n² + I n + J]). @p bars holds
/// m n EnergyBars, [u n + J] direction u's as that reverse pass's bars[J],
/// added onto: the shared fields' bars one pointer within each u (checked).
/// Each ordered pair's transition bars' tangent is handed to
/// @p transition(u, bra, ket, bars), dm1, dm2 and dm3 alone set, as the
/// reverse pass hands its own (device buffers on res.stream(), freed after
/// the call); null: not formed.
///
/// pc_class_energies_hvp's three terms over every ket's solve, each a fixed
/// walk of the pair reverse pass: the multipliers' tangent (per ket J and bra
/// I, ġ_I = (ẏ_{I←J} - Ḋ_J g_I) / (λ + Δ) through J's factorization, ẏ and
/// Ḋ pc_effective_hamiltonian_tangent's, the transition rows' leak through
/// the kept span's rotation in them); every ket's U along U̇ = V_n Θ', one
/// symmetric difference; and the inputs' tangent at fixed multipliers, every
/// pair along its own direction, ±s, ±2s, ±3s. Each ket's span bar collects
/// every bra's T̄_{I←J}, and so does its Daleckii–Krein derivative. Sijrs's
/// share is SC's at Σ_J H̄_JJ. @p span false leaves the rotation out at both
/// orders, for the tests alone.
///
/// @return The n² elements, as pc_effective_hamiltonian computes them;
///         InvalidConfig as pc_effective_hamiltonian_tangent's, or for @p bars
///         not m n or not sharing one direction's shared bars; Numerical,
///         naming the state and the class, when a ket keeps a metric mode
///         below 1e-10 of its largest (pc_class_energies_hvp's #917 rule);
///         pc_effective_hamiltonian's refusals; or the tangent's and reverse
///         pass's errors, and @p transition's.
[[nodiscard]] Result<std::vector<PcClassEnergies>>
pc_effective_hamiltonian_hvp(DeviceResources &res, const MoBlocks &blocks,
                             std::span<const EnergyInputs> pairs, std::span<const double> hbar,
                             int32_t batch, std::span<const EnergyDirection> directions,
                             std::span<const EnergyBars> bars,
                             const std::function<Status(int64_t direction, int64_t bra, int64_t ket,
                                                        const EnergyBars &transition)> &transition,
                             bool span = true);

} // namespace free_breakfast::nevpt2
