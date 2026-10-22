/// @file
/// @brief SortedKets: the scratch, and the launcher's arguments.

module;

// The launcher, defined in sort_bounds_launch.cu: declared in the GMF so it
// keeps external linkage (see the header).
#include "sort_bounds_bridge.h"

module free_breakfast.integrals.coulomb.sort_bounds;

import std;
import free_breakfast.core.types;
import free_breakfast.core.wwr;
import free_breakfast.integrals.mmd;

namespace free_breakfast {

SortedKets::SortedKets(const std::shared_ptr<Device> &device, const PrimitivePairClassView &cls,
                       const int32_t n_hermite)
    : cls_(cls), n_hermite_(n_hermite), key_(static_cast<size_t>(cls.n_rec), device),
      key_alt_(static_cast<size_t>(cls.n_rec), device),
      perm_(static_cast<size_t>(cls.n_rec), device),
      perm_alt_(static_cast<size_t>(cls.n_rec), device),
      zeta_(static_cast<size_t>(cls.n_rec), device), p_x_(static_cast<size_t>(cls.n_rec), device),
      p_y_(static_cast<size_t>(cls.n_rec), device), p_z_(static_cast<size_t>(cls.n_rec), device),
      bound_(static_cast<size_t>(cls.n_rec), device), dmax_(static_cast<size_t>(cls.n_rec), device),
      dq_(static_cast<size_t>(cls.n_rec) * static_cast<size_t>(n_hermite), device) {}

wwr::wwrError_t SortedKets::sort(const wwr::wwrStream_t stream, const double *bound,
                                 const double *dmax, const double *dq) {
  return device::sort_bounds_kets(stream, cls_, n_hermite_, bound, dmax, dq, key_.data(),
                                  key_alt_.data(), perm_.data(), perm_alt_.data(), zeta_.data(),
                                  p_x_.data(), p_y_.data(), p_z_.data(), bound_.data(),
                                  dmax_.data(), dq_.data());
}

PrimitivePairClassView SortedKets::view() const {
  return {
      .la = cls_.la,
      .lb = cls_.lb,
      .n_rec = cls_.n_rec,
      .zeta = zeta_.data(),
      .p_x = p_x_.data(),
      .p_y = p_y_.data(),
      .p_z = p_z_.data(),
      .pa_x = nullptr,
      .pa_y = nullptr,
      .pa_z = nullptr,
      .pb_x = nullptr,
      .pb_y = nullptr,
      .pb_z = nullptr,
      .s = nullptr,
      .eb = nullptr,
      .bound = bound_.data(),
      .off_a = nullptr,
      .off_b = nullptr,
  };
}

} // namespace free_breakfast
