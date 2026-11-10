/// @file fb_overlap.cpp
/// @brief fb_overlap.h over the pipeline's setup steps and OverlapStep. Not a
/// module unit: a plain TU that imports modules and exports C symbols, as
/// rust/gemm's fb_gemm.cpp is. Every function is noexcept, for that file's
/// reason.
#include "fb_overlap.h"

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step; // View
import free_breakfast.pipeline.molecule;
import free_breakfast.pipeline.basis_set;
import free_breakfast.pipeline.spherical_transform;
import free_breakfast.pipeline.primitive_pair_classes;
import free_breakfast.pipeline.overlap;

#include "diagnostics.h" // FBF_TRY
#include "fb_handles.hpp"

namespace fb = free_breakfast;

/// The steps, in order, each reading what the ones before it made: the same
/// chain a TOML run with an [overlap] table walks.
static fb::Result<fb::OverlapOutput> build_overlap(fb::DeviceResources &res,
                                                   const std::filesystem::path &xyz,
                                                   const std::string &basis) {
  const fb::MoleculeOutput molecule = FBF_TRY(fb::MoleculeStep{.xyz = xyz}.run(fb::View<>{}, res));
  const fb::BasisSetOutput basis_set =
      FBF_TRY(fb::BasisSetStep{.basis = basis}.run(fb::View<fb::MoleculeOutput>{molecule}));
  const fb::View<fb::BasisSetOutput> on_basis{basis_set};
  const fb::SphericalTransformOutput transform =
      FBF_TRY(fb::SphericalTransformStep{}.run(on_basis, res));
  const fb::PrimitivePairClassesOutput pairs =
      FBF_TRY(fb::PrimitivePairClassesStep{}.run(on_basis, res));
  return fb::OverlapStep{}.run(
      fb::View<fb::BasisSetOutput, fb::SphericalTransformOutput, fb::PrimitivePairClassesOutput>{
          basis_set, transform, pairs},
      res);
}

extern "C" {

fb_status fb_overlap(fb_resources *const res, const char *const xyz_path, const char *const basis,
                     fb_buffer **const out, size_t *const nao) noexcept {
  if (res == nullptr || xyz_path == nullptr || basis == nullptr || out == nullptr ||
      nao == nullptr) {
    return FB_INVALID_ARGUMENT;
  }
  auto built = build_overlap(res->res, xyz_path, basis);
  if (!built) {
    return fb_fail(std::move(built).error().message);
  }
  const fb::SphericalMatrixAO &s = built->s;
  auto *buf = new fb_buffer{fb::Buffers::device<double>(s.num_elements(), res->res.device())};
  // S's own storage is the step's; a device copy hands Rust a buffer it owns.
  if (wwr::wwrMemcpyAsync(buf->data.data(), s.data(), s.num_elements() * sizeof(double),
                          wwr::wwrMemcpyDeviceToDevice, res->res.stream()) != wwr::wwrSuccess ||
      wwr::wwrStreamSynchronize(res->res.stream()) != wwr::wwrSuccess) {
    delete buf;
    return FB_GPU_ERROR;
  }
  *out = buf;
  *nao = s.n();
  return FB_OK;
}

} // extern "C"
