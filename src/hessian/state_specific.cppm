/// @file
/// @brief What df_casscf and df_nevpt2_tangents share about a state-specific
/// DF-CASSCF solution: its inputs checked, and its active electrons by spin
/// from trace(gamma) and the molecule's multiplicity; the same for a state
/// average, which df_sacasscf and df_qd_nevpt2_tangents share, and the
/// mixing-rate bound they put on its roots. Beside them, the CP solve
/// df_casscf and df_sacasscf share: zvectors with the probe column that
/// refuses a singular H~; H_eff's eigenpairs, which
/// df_qd_eigenvector_response and the QD Hessians (df_qd_nevpt2.cpp) share;
/// and SC's conditioning refusal, which df_sc_nevpt2 and df_qd_sc_nevpt2
/// share.
///
/// An internal partition (`module`, not `export module`): only this module's
/// implementation units import it, so nothing here reaches an importer of
/// free_breakfast.hessian. It declares the functions; df_casscf.cpp defines
/// them, but state_averaged_electrons, which df_sacasscf.cpp does,
/// qd_response_eigen, which df_qd_eigenvector_response.cpp does, and
/// sc_rounding_refusal, which df_sc_nevpt2.cpp does. See
/// docs/style.md, "Module-private code".

module free_breakfast.hessian:state_specific;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers
import free_breakfast.core.resources;
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

} // namespace free_breakfast::hessian
