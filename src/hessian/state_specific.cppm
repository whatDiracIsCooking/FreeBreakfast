/// @file
/// @brief What df_casscf and df_nevpt2_tangents share about a state-specific
/// DF-CASSCF solution: its inputs checked, and its active electrons by spin
/// from trace(gamma) and the molecule's multiplicity; the same for a state
/// average, which df_sacasscf and df_qd_nevpt2_tangents share, and the
/// mixing-rate bound they put on its roots. Beside them, the CP solve
/// df_casscf and df_sacasscf share: zvectors with the probe column that
/// refuses a singular H~, and its footprint with the helper both footprints
/// end on; H_eff's eigenpairs, which
/// df_qd_eigenvector_response and the QD Hessians (df_qd_nevpt2.cpp) share;
/// SC's conditioning refusal, which df_sc_nevpt2 and df_qd_sc_nevpt2
/// share; and PC's degenerate-pair step, which df_pc_nevpt2 and
/// df_qd_pc_nevpt2 share (#1277). And the NEVPT2 Hessians' footprints'
/// shared pieces (#1186): a Ledger to walk a body with, and the footprints
/// of the routines they compose that export none.
///
/// An internal partition (`module`, not `export module`): only this module's
/// implementation units import it, so nothing here reaches an importer of
/// free_breakfast.hessian. It declares the functions; df_casscf.cpp defines
/// them, but state_averaged_electrons and sacasscf_footprint, which
/// df_sacasscf.cpp does, qd_response_eigen and its footprints, which
/// df_qd_eigenvector_response.cpp does, sc_rounding_refusal and the SC/PC
/// assembly's footprint, which df_sc_nevpt2.cpp does, and the others in the
/// file of the routine each is of. See
/// docs/style.md, "Module-private code".

module free_breakfast.hessian:state_specific;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers
import free_breakfast.core.resources;
import free_breakfast.core.footprint;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.response.hessian;
import free_breakfast.mcscf.solve; // mcscf::Solution, ci::Roots
import free_breakfast.response.relaxed_hessian;
import free_breakfast.response.zvector;

namespace free_breakfast::hessian {

/// @brief (nalpha, nbeta) of a one-root DF-CASSCF @p solution, its inputs
/// checked first: one root, C nao square on res's device, the molecule and
/// both bases on the same atoms, ncore >= 0, ncas >= 1, ncore + ncas <= nao,
/// @p b naux nao^2 long, the RDMs ncas^2 and ncas^4 long, and trace(gamma)
/// an electron count of the molecule's multiplicity in ncas orbitals. Each
/// refusal an InvalidConfig whose message starts with @p who.
Result<std::pair<int32_t, int32_t>>
state_specific_electrons(std::string_view who, const DeviceResources &res, const Molecule &molecule,
                         const BasisSet &basis, const BasisSet &aux,
                         const Buffers::device<double> &b, const mcscf::Solution &solution,
                         int32_t ncore, int32_t ncas);

/// @brief (nalpha, nbeta) of a state-averaged DF-CASSCF @p solution at
/// @p weights, its inputs checked first: two roots or more, one positive
/// weight per root, then state_specific_electrons' checks but the root
/// count; and no two roots under mcscf::kRootGapFloor apart (Numerical,
/// naming them: a degeneracy). Each refusal's message starts with @p who.
Result<std::pair<int32_t, int32_t>>
state_averaged_electrons(std::string_view who, const DeviceResources &res, const Molecule &molecule,
                         const BasisSet &basis, const BasisSet &aux,
                         const Buffers::device<double> &b, const mcscf::Solution &solution,
                         std::span<const double> weights, int32_t ncore, int32_t ncas);

// The largest rotation rate between two roots, per bohr, at which a state
// average's Hessian still describes its states: mcscf::root_rotation's
// default max_rotation, put on lambda^x's rotations at unequal weights and on
// eq. qd-theta's at equal ones (README.md).
constexpr double kSaCasscfMaxRotation = 10.0;

// H~'s CI response is solved to this, relative to its right-hand side
// (RelaxedHessian::create's default, named for probed_zvectors' bar).
constexpr double kCasscfCpResponseTolerance = 1e-8;

/// @brief response::zvectors' Schur solve of the @p n_x columns @p gradients,
/// @p ci_rhs and @p theta_rhs (each as zvectors takes them) over @p relaxed,
/// built with kCasscfCpResponseTolerance, with one more column beside them,
/// the probe: H~ u = -s for a fixed pseudo-random unit s, no CI or rotation
/// part. The coordinates' right-hand sides can be orthogonal to a null
/// direction of H~ by symmetry, which leaves the direction to round-off; s is
/// not (README.md "A singular H~").
///
/// @return The coordinates' columns alone; a Divergence whose message starts
///         with @p who and calls H~ singular when the probe's Rayleigh
///         quotient rho = -u.s / u.u is not above kCasscfCpResponseTolerance
///         max diag(H), or when zvectors meets a direction H~ is not positive
///         along or runs out of products; or zvectors' other errors.
Result<response::ZVectors>
probed_zvectors(DeviceResources &res, std::string_view who, response::RelaxedHessian &relaxed,
                const response::Hessian &hessian, const ci::Roots &roots, size_t n_x,
                const Buffers::device<double> &gradients, const Buffers::device<double> &ci_rhs,
                std::span<const double> theta_rhs, double tolerance, int32_t max_iterations);

/// @brief What probed_zvectors holds over @p pairs orbital pairs, @p nroots CI
/// vectors of @p dim determinants and @p n_x coordinates, past the
/// RelaxedHessian's own: the coordinates' columns returned, persistent; the
/// right-hand sides with the probe's column, and response::zvectors_footprint
/// over n_x + 1 columns with @p product a block product's, scratch.
/// @p rotating whether the weights rotate (and so a rotation right-hand side
/// is given).
MemoryFootprint probed_zvectors_footprint(size_t pairs, size_t dim, int32_t nroots, size_t n_x,
                                          const MemoryFootprint &product, bool rotating);

/// @brief @p chain with only @p kept bytes of its persistent ones kept: the
/// rest freed when it returns, its peak unchanged.
MemoryFootprint keeping(const MemoryFootprint &chain, size_t kept);

/// @brief df_sacasscf_footprint's body, or with @p cp_only df_sacasscf_cp's
/// (its CP step, lambda^x returned), its arguments df_sacasscf_footprint's.
MemoryFootprint sacasscf_footprint(const BasisSet &basis, const BasisSet &aux, int32_t ncore,
                                   int32_t ncas, int32_t nalpha, int32_t nbeta, int32_t nroots,
                                   bool rotating, size_t syevd_lwork, bool cp_only);

/// @brief What df_qd_zeta holds over @p basis and @p aux with @p ncore core
/// and @p ncas active orbitals, @p nroots roots over the determinants of
/// @p nalpha and @p nbeta electrons and @p k columns; @p syevd_lwork at order
/// ncas. Its L_xlambda persistent.
MemoryFootprint qd_zeta_footprint(const BasisSet &basis, const BasisSet &aux, int32_t ncore,
                                  int32_t ncas, int32_t nalpha, int32_t nbeta, int32_t nroots,
                                  int32_t k, size_t syevd_lwork);

/// @brief What qd_response_eigen holds over @p nroots roots: H_eff and its
/// eigenvalues, scratch; diagonalize's work at order nroots (@p syevd_lwork
/// by order), the workspace.
MemoryFootprint qd_response_eigen_footprint(int32_t nroots, std::span<const size_t> syevd_lwork);

/// @brief What df_qd_eigenvector_response holds, SC's or with @p pc PC's at
/// @p dropped (PcClassEnergies' per state, or empty for the most): the
/// effective Hamiltonian over nevpt2::qd_inputs, its eigenpairs, then
/// @p n_w couplings' several-W gradient, bounded by n_w times one trace-zero
/// W's (nevpt2::qd_sc_w_gradient_footprint, exact at one W). The counts as
/// the QD Hessians'.
MemoryFootprint qd_eigenvector_response_footprint(const BasisSet &basis, const BasisSet &aux,
                                                  int32_t ncore, int32_t ncas, int32_t nalpha,
                                                  int32_t nbeta, int32_t nroots, bool rotating,
                                                  int32_t batch,
                                                  std::span<const size_t> syevd_lwork, bool pc,
                                                  int32_t n_w,
                                                  std::span<const std::array<int64_t, 8>> dropped);

/// @brief A routine's footprint, built in its allocations' order: what is
/// live, the most that ever was, and the largest workspace a callee borrows.
/// The NEVPT2 Hessians' footprints walk their bodies with it.
struct Ledger {
  size_t live = 0;
  size_t peak = 0;
  size_t workspace = 0;
  size_t pinned = 0;
  void take(const size_t bytes) {
    live += bytes;
    peak = std::max(peak, live);
  }
  void give(const size_t bytes) { live -= bytes; }
  /// @brief A callee: its peak beside what is live, then its persistent
  /// bytes kept.
  void call(const MemoryFootprint &f) {
    take(f.device.persistent + f.device.scratch);
    give(f.device.scratch);
    borrow(f);
  }
  /// @brief A callee whose product is freed as soon as it returns.
  void run(const MemoryFootprint &f) {
    call(f);
    give(f.device.persistent);
  }
  /// @brief A callee's workspaces alone.
  void borrow(const MemoryFootprint &f) {
    workspace = std::max(workspace, f.device.workspace);
    pinned = std::max(pinned, f.pinned.peak());
  }
  /// @brief The routine's footprint, @p kept of what is still live its
  /// product.
  [[nodiscard]] MemoryFootprint footprint(const size_t kept) const {
    return {.device = {.persistent = kept, .scratch = peak - kept, .workspace = workspace},
            .pinned = {.workspace = pinned}};
  }
};

/// @brief What casscf_z_terms holds over @p basis and @p aux with @p ncore
/// core and @p ncas active orbitals, @p nroots roots over the determinants of
/// @p nalpha and @p nbeta electrons and @p n_theta rotations among them in
/// ThirdDerivative's layout, with a right-hand side (@p rhs) or without, and
/// the third derivative's action over @p k columns; @p syevd_lwork at order
/// ncas. L_xlambda persistent; the rest df_sacasscf_z_footprint describes,
/// the action's Y (size k) and its k x k product last.
MemoryFootprint casscf_z_terms_footprint(const BasisSet &basis, const BasisSet &aux, int32_t ncore,
                                         int32_t ncas, int32_t nalpha, int32_t nbeta,
                                         int32_t nroots, size_t n_theta, bool rhs, int32_t k,
                                         size_t syevd_lwork);

/// @brief One chunk of @p count coordinates of df_nevpt2_tangents' consumer
/// over @p naux fitting functions and @p nmo orbitals with @p ncore core and
/// @p ncas active ones, all of it scratch: h_dot, B_dot and kappa,
/// nevpt2::builders_tangent's, then kappa_oo. Its peak is what the chunk
/// takes of the memory budget.
MemoryFootprint nevpt2_tangent_chunk_footprint(int32_t naux, int32_t nmo, int32_t ncore,
                                               int32_t ncas, int32_t count);

/// @brief One chunk of @p count coordinates of df_qd_nevpt2_tangents'
/// consumer over @p nroots roots of @p nalpha and @p nbeta electrons, as
/// nevpt2_tangent_chunk_footprint's: h_dot, B_dot, kappa and c_dot,
/// nevpt2::qd_builders_tangent's, the first chunk's transition RDMs, then
/// kappa_oo.
MemoryFootprint qd_nevpt2_tangent_chunk_footprint(int32_t naux, int32_t nmo, int32_t ncore,
                                                  int32_t ncas, int32_t nalpha, int32_t nbeta,
                                                  int32_t nroots, int32_t count);

/// @brief What mcscf::df_integral_derivatives holds when it hands its first
/// chunk of @p count coordinates over @p basis and @p aux to a consumer: the
/// MO factor, h_MO, tau^x and h1^x, the metric derivative, the chunk's W^x
/// and one aux atom's block (integral_derivative.cpp's held buffers).
size_t integral_derivatives_held_bytes(const BasisSet &basis, const BasisSet &aux, size_t count);

/// @brief The NEVPT2 Hessians' derivative-integral sizing, df_nevpt2_tangents'
/// and df_qd_nevpt2_tangents': the max_bytes mcscf::df_integral_derivatives
/// takes when the caller gives @p max_bytes (a third of it, at least one
/// byte, as before), or else a third of the memory budget less
/// kPoolHeadroomBytes, never past what leaves the consumer @p reserve bytes
/// (one coordinate's chunk) and never below one coordinate's own need.
Result<size_t> integral_derivatives_bytes(const DeviceResources &res, const BasisSet &basis,
                                          const BasisSet &aux, size_t reserve, size_t max_bytes);

/// @brief What df_sc_nevpt2's and df_pc_nevpt2's one assembly holds (their
/// footprints' body): @p gradient the Lagrangian's calls with their strings
/// taken off (z^c and the multipliers persistent), @p product(count,
/// partners) the Hessian-vector product's; with @p copies a pair's
/// directions and partners (SC's), and with @p pairs its product too; with
/// @p degenerate PC's degenerate-pair step (a turned basis and one product).
/// The counts as df_sc_nevpt2_footprint's.
MemoryFootprint
nevpt2_hessian_footprint(const BasisSet &basis, const BasisSet &aux, int32_t ncore, int32_t ncas,
                         int32_t nalpha, int32_t nbeta, int32_t k,
                         std::span<const size_t> syevd_lwork, const MemoryFootprint &gradient,
                         const std::function<MemoryFootprint(int32_t, int32_t)> &product,
                         bool copies, bool pairs, bool degenerate = false);

/// @brief @p gradient, a nevpt2 gradient's footprint over @p ncas active
/// orbitals of @p nalpha and @p nbeta electrons, less the strings it makes
/// first and holds throughout: what its calls hold beside a caller's own.
MemoryFootprint without_strings(const MemoryFootprint &gradient, int32_t ncas, int32_t nalpha,
                                int32_t nbeta);

/// @brief The widest of @p count coordinates whose chunk @p bytes (a
/// count's bytes, rising with it) fits the memory budget less
/// kPoolHeadroomBytes; Unsupported, naming @p who and the bytes, when one
/// does not.
Result<size_t> widest_fitting(const DeviceResources &res, std::string_view who, size_t count,
                              const std::function<size_t(size_t)> &bytes);

// SC's conditioning refusal (README.md "The SC conditioning refusal",
// #1102): a kept perturber's relative rounding bound on its N
// (nevpt2::ClassEnergiesHvp::rounding) past this is refused. The Hessian is
// assembled from separate walks of the tables whose parts cancel, and each
// moves that perturber's H / N by about the bound. The bound sits at the
// log-midpoint of the measured gap: accepted cases reach 1.2e-8 and repeat
// to 1e-11; h2nb_sto3g_sa2's state 1, at 9.2e-7 and 3.8e-6, spreads 1e3
// bars run to run.
constexpr double kScNevpt2Rounding = 1e-7;

/// @brief A Numerical error whose message starts with @p who when
/// @p rounding, a product's worst kept perturber's sigma / |N|, is past
/// kScNevpt2Rounding, naming the reference root, class and perturber
/// @p at ({root, class, perturber}, nevpt2's rounding_at) and its N @p norm;
/// nothing otherwise.
Status sc_rounding_refusal(std::string_view who, double rounding, std::array<int64_t, 3> at,
                           double norm);

/// @brief H_eff = diag(E_r) + 1/2 (C + C^T) of a QD-NEVPT2 state average, from
/// its root energies @p energies and the n x n correlation elements' totals
/// @p totals ([M n + N] bra M, ket N), diagonalized: the eigenvalues ascending
/// and R, column K at [K n], each signed so that its largest-magnitude
/// component is positive (nevpt2::qd_sc_coupling's convention).
Result<std::pair<std::vector<double>, std::vector<double>>>
qd_response_eigen(DeviceResources &res, std::span<const double> energies,
                  std::span<const double> totals);

/// @brief The reach (Eh) of PC's pair rule in the Hessians' tangents (README
/// "Where a pair is taken", #1296): a core or virtual pair closer than this
/// is left unrotated and taken by its off-diagonal f, not divided.
constexpr double kPcPairReach = 1e-4;

/// @brief The most orbitals a near-degenerate cluster is taken whole with
/// (README "A near-degenerate cluster", #1314): its m (m - 1) / 2 pairs take
/// three products each and every two of them one more, about m^4 / 4 in all
/// (some 120 at six).
constexpr size_t kPcClusterLimit = 6;

/// @brief The core or virtual pairs (p, q), p > q, of the orbital energies
/// @p eps (nmo, the core [0, nc), the virtuals [nocc, nmo)) that PC's tangents
/// take by their off-diagonal f: every pair within a cluster, a connected
/// component of the pairs closer than @p reach (README "A near-degenerate
/// cluster", #1314), so that no pair inside the reach is divided; in a
/// cluster of more than kPcClusterLimit orbitals, the pairs closer than
/// @p reach, closest first, each orbital in one at most (#1296's rule).
/// Closest first. df_nevpt2_tangents' and df_qd_nevpt2_tangents' (PC).
std::vector<std::array<size_t, 2>> pc_taken_pairs(std::span<const double> eps, size_t nc,
                                                  size_t nocc, double reach);

/// @brief The distinct (p, q) of a PC Hessian's degenerate-pair records
/// @p records, in first-seen order; they may share orbitals (a cluster,
/// README "A near-degenerate cluster"). Unsupported, its message starting
/// with @p who and naming the orbitals, for a cluster of more than
/// kPcClusterLimit (one that pc_taken_pairs did not take whole, and whose
/// pairs under 1e-6 Eh still join it).
/// df_pc_nevpt2's and df_qd_nevpt2's (PC).
Result<std::vector<std::array<int32_t, 2>>>
degenerate_pair_blocks(std::string_view who, std::span<const std::array<int32_t, 2>> records);

/// @brief The degenerate pairs' part of a PC Hessian (README "A degenerate
/// inactive pair", #1268), k x k column-major, and the largest |energy|
/// change of turning the pairs in the bases it reads them in.
struct DegeneratePairPart {
  std::vector<double> part;
  double rotation_shift{0.0};
};

/// @brief df_pc_nevpt2's step 4b, which df_qd_nevpt2 (PC) shares at its W:
/// H_ij += sum_P (phi^P_i K^P_j + phi^P_j K^P_i) + sum_PP' M_PP' phi^P_i
/// phi^P'_j over the @p blocks (degenerate_pair_blocks', which may share
/// orbitals: README "A near-degenerate cluster"), @p phi
/// and @p split their f^(i)_pq and (f^(i)_pp - f^(i)_qq) / 2, [P k + i],
/// over @p k coordinates. Each term is read from @p product(c_turned,
/// direction, consume), the energy's Hessian-vector product at orbitals
/// c_turned (nmo x nmo on the device, C U) with z = 0 along the h direction
/// @p direction alone, handing consume its (h_bar, Lambda, y) tangent and
/// the energy at c_turned; @p pair_with(h_bar, Lambda, y, into) puts
/// <t_i, them> over every i into k doubles on the device. @p c is C, and
/// @p energy the canonical energy rotation_shift is against.
Result<DegeneratePairPart> degenerate_pair_part(
    DeviceResources &res, int32_t nmo, int32_t naux, size_t k, const double *c, double energy,
    std::span<const std::array<int32_t, 2>> blocks, std::span<const double> phi,
    std::span<const double> split,
    const std::function<Status(const double *, const double *,
                               const std::function<Status(const double *, const double *,
                                                          const double *, double)> &)> &product,
    const std::function<Status(const double *, const double *, const double *, double *)>
        &pair_with);

} // namespace free_breakfast::hessian
