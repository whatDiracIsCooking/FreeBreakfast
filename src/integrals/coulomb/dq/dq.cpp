/// @file
/// @brief The ket pre-contraction of one class.

module;

// The kernel launcher, defined in dq_launch.cu: declared in the GMF so it
// keeps external linkage (see the header).
#include "dq_bridge.h"

module free_breakfast.integrals.coulomb.dq;

import std;
import free_breakfast.core.wwr;
import free_breakfast.integrals.coulomb.pair_class;
import free_breakfast.linalg.matrix_ao;

namespace free_breakfast {

wwr::wwrError_t dq(CoulombPairClass &ket, const CartesianMatrixAO &density) {
  return device::dq_class(ket.device_stream()->stream(), ket.view(),
                          static_cast<int32_t>(density.n()), density.data(), ket.w.data(),
                          ket.dq.data(), ket.dmax.data());
}

} // namespace free_breakfast
