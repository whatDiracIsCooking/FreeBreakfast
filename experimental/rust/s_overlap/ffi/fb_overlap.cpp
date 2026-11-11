/// @file fb_overlap.cpp
/// @brief fb_overlap.h over the pipeline's setup steps and OverlapStep. Not a
/// module unit: a plain TU that imports modules and exports C symbols, built
/// into a static archive the Rust crate links. Every function is noexcept,
/// for the header's reason.
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

namespace fb = free_breakfast;

/// S and the DeviceResources it lives on; `res` is declared first so it is
/// destroyed last.
struct fb_overlap {
  fb::DeviceResources res;
  std::optional<fb::OverlapOutput> out;
};

/// fb_last_error()'s message, per thread, as errno is.
static thread_local std::string last_error;

static fb_status fail(std::string message) noexcept {
  last_error = std::move(message);
  return FB_FAILED;
}

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

const char *fb_last_error(void) noexcept {
  return last_error.c_str();
}

fb_status fb_overlap_new(const int32_t dev_idx, const char *const xyz_path, const char *const basis,
                         fb_overlap **const out, size_t *const nao) noexcept {
  if (dev_idx < 0 || xyz_path == nullptr || basis == nullptr || out == nullptr || nao == nullptr) {
    return FB_INVALID_ARGUMENT;
  }
  // Workspaces sized at construction (src/core/resources/README.md): the
  // overlap borrows at most a few cc-pVTZ AO squares.
  auto s = std::unique_ptr<fb_overlap>(
      new fb_overlap{fb::DeviceResources(dev_idx, {.device = {.workspace = std::size_t{64} << 20},
                                                   .pinned = {.workspace = std::size_t{1} << 20}}),
                     std::nullopt});
  auto built = build_overlap(s->res, xyz_path, basis);
  if (!built) {
    return fail(std::move(built).error().message);
  }
  s->out.emplace(std::move(*built));
  if (wwr::wwrStreamSynchronize(s->res.stream()) != wwr::wwrSuccess) {
    return FB_GPU_ERROR;
  }
  *nao = s->out->s.n();
  *out = s.release();
  return FB_OK;
}

fb_status fb_overlap_copy(const fb_overlap *const s, double *const dst, const size_t len) noexcept {
  if (s == nullptr || dst == nullptr || len != s->out->s.num_elements()) {
    return FB_INVALID_ARGUMENT;
  }
  const auto stream = s->res.stream();
  if (wwr::wwrMemcpyAsync(dst, s->out->s.data(), len * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                          stream) != wwr::wwrSuccess ||
      wwr::wwrStreamSynchronize(stream) != wwr::wwrSuccess) {
    return FB_GPU_ERROR;
  }
  return FB_OK;
}

void fb_overlap_free(fb_overlap *const s) noexcept {
  delete s;
}

} // extern "C"
