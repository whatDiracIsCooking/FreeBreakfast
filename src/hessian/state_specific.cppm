/// @file
/// @brief What df_casscf and df_nevpt2_tangents share about a state-specific
/// DF-CASSCF solution: its inputs checked, and its active electrons by spin
/// from trace(gamma) and the molecule's multiplicity.
///
/// An internal partition (`module`, not `export module`): only this module's
/// implementation units import it, so nothing here reaches an importer of
/// free_breakfast.hessian. It only declares; df_casscf.cpp defines. See
/// docs/style.md, "Module-private code".

module free_breakfast.hessian:state_specific;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.mcscf.solve;

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

} // namespace free_breakfast::hessian
