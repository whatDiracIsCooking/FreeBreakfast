/// @file
/// @brief One ket class's [p|q] contraction against every bra class.

module;

// The kernel launcher, defined in e_hermite_launch.cu: declared in the GMF so
// it keeps external linkage (see the header).
#include "e_hermite_bridge.h"

module free_breakfast.integrals.eri.integral_direct.coulomb.e_hermite;

import std;
import free_breakfast.core.wwr;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.integral_direct.coulomb.pair_class;

namespace free_breakfast {

wwr::wwrError_t e_hermite(const CoulombPairClass &ket, const std::span<CoulombPairClass> bras,
                          const double tau) {
  const wwr::wwrStream_t s = ket.device_stream()->stream();
  const PrimitivePairClassView k = ket.ket_view();
  for (CoulombPairClass &bra : bras) {
    const PrimitivePairClassView b = bra.view();
    if (const wwr::wwrError_t code = device::e_hermite_class(
            s, b, k, tau, b.bound, k.bound, ket.ket_dq.data(), ket.ket_dmax.data(), bra.jp.data());
        code != wwr::wwrSuccess) {
      return code;
    }
  }
  return wwr::wwrSuccess;
}

} // namespace free_breakfast
