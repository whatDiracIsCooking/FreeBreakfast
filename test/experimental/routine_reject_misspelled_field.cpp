// Must not compile (#1246 case d): a call site with a misspelled In field
// (.ncroe for .ncore). The key is passed down, as a step's callee would have
// it, so the only error is the designator. CMakeLists.txt registers a ctest
// entry that builds this TU and passes on that diagnostic.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.mcscf;
import free_breakfast.experimental.routine;

namespace fb = free_breakfast;

fb::Result<fb::mcscf::RootRotation> routine_reject_misspelled_field(
    const fb::routine::AllocKey &key, fb::DeviceResources &res, const fb::Molecule &molecule,
    const fb::BasisSet &basis, const fb::SphericalMatrixAO &h, const fb::Buffers::device<double> &b,
    const fb::SphericalMatrixAO &c, const double *roots, std::span<const double> weights,
    std::span<const double> energies, const fb::mcscf::Derivatives &derivatives,
    const double *kappa) {
  return fb::routine::root_rotation(key, {.res = res,
                                          .molecule = molecule,
                                          .basis = basis,
                                          .h = h,
                                          .b = b,
                                          .c = c,
                                          .ncroe = 1,
                                          .ncas = 4,
                                          .nalpha = 2,
                                          .nbeta = 2,
                                          .roots = roots,
                                          .weights = weights,
                                          .energies = energies,
                                          .derivatives = derivatives,
                                          .kappa = kappa});
}
