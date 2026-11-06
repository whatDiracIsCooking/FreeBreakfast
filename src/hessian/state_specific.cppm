/// @file
/// @brief What df_casscf and df_nevpt2_tangents share about a state-specific
/// DF-CASSCF solution: its inputs checked, and its active electrons by spin
/// from trace(gamma) and the molecule's multiplicity. Beside it, the CP solve
/// df_casscf and df_sacasscf share: zvectors with the probe column that
/// refuses a singular H~.
///
/// An internal partition (`module`, not `export module`): only this module's
/// implementation units import it, so nothing here reaches an importer of
/// free_breakfast.hessian. It declares the functions; df_casscf.cpp defines
/// them. See
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
import free_breakfast.mcscf.relaxed_hessian;
import free_breakfast.mcscf.zvector;

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

// H~'s CI response is solved to this, relative to its right-hand side
// (RelaxedHessian::create's default, named for probed_zvectors' bar).
constexpr double kCasscfCpResponseTolerance = 1e-8;

/// @brief mcscf::zvectors' Schur solve of the @p n_x columns @p gradients,
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
Result<mcscf::ZVectors>
probed_zvectors(DeviceResources &res, std::string_view who, mcscf::RelaxedHessian &relaxed,
                const response::Hessian &hessian, const ci::Roots &roots, size_t n_x,
                const Buffers::device<double> &gradients, const Buffers::device<double> &ci_rhs,
                std::span<const double> theta_rhs, double tolerance, int32_t max_iterations);

} // namespace free_breakfast::hessian
