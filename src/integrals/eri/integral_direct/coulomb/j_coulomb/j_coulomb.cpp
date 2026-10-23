/// @file
/// @brief One bra class's back-transform into J.

module;

// The kernel launchers, defined in j_coulomb_launch.cu: declared in the GMF so
// they keep external linkage (see the header).
#include "j_coulomb_bridge.h"

module free_breakfast.integrals.eri.integral_direct.coulomb.j_coulomb;

import std;
import free_breakfast.core.types;
import free_breakfast.core.wwr;
import free_breakfast.integrals.eri.integral_direct.coulomb.pair_class;
import free_breakfast.recipes.matrix_ao;

namespace free_breakfast {

wwr::wwrError_t j_coulomb_rows(const wwr::wwrStream_t stream, const std::span<int32_t> rows) {
  return device::j_coulomb_rows(stream, static_cast<int32_t>(rows.size()), rows.data());
}

wwr::wwrError_t j_coulomb(const CoulombPairClass &bra, const std::span<const int32_t> rows,
                          CartesianMatrixAO &j) {
  return device::j_coulomb_class(bra.device_stream()->stream(), bra.view(),
                                 static_cast<int32_t>(j.n()), rows.data(), bra.jp.data(), j.data());
}

} // namespace free_breakfast
