/// @file
/// @brief The MO blocks (b_mo and a permute), h2e, and the slab builder: one
/// strided-batched gemm and a permute per slab, over the demo's (X, Y, axes)
/// table.

module;

#include "nevpt2_permute_bridge.h"

module free_breakfast.nevpt2.density_fitting;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemmStridedBatched, wwrMemsetAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.integrals.eri.density_fitting.mo;

#include "diagnostics.h"

namespace free_breakfast::nevpt2 {

// One slab's operands: C[x, y, z, w] = sum_Q X_Q,xy Y_Q,zw, X nx x ny and
// Y nz x nw orbital pairs, then slab axis k = C axis axes[k] (numpy's
// transpose convention).
struct SlabOperands {
  const double *x = nullptr;
  int32_t nx = 0;
  int32_t ny = 0;
  const double *y = nullptr;
  int32_t nz = 0;
  int32_t nw = 0;
  std::array<int32_t, 4> axes{};
};

// The demo's table (nevpt2-gpu-demo src/df_integrals/df_integrals.cpp, carried
// over unchanged). Each row was derived from the PySCF `eris[...]` slice its
// generate_golden.py's _integral_blocks takes for that block, and
// nevpt2_density_fitting_gpu checks every one against the golden four-index
// arrays. C[x, y, z, w] = (xy|zw); the batched index is always the FIRST of
// X's pair, which is why B_vc is kept beside B_cv.
//
//   block  physicist slab              X      Y      C          axes
//   Sr     [r,a,b,c] = (rb|ac)         B_va   B_aa   [r,b,a,c]  0 2 1 3
//   Si     [t,u,i,v] = (it|uv)         B_ca   B_aa   [i,t,u,v]  1 2 0 3
//   Sijrs  [i,a,j,b] = (ia|jb)         B_cv   B_cv   [i,a,j,b]  0 1 2 3
//   Sijr   [r,p,j,i] = (rj|ip)         B_vc   B_ca   [r,j,i,p]  0 3 1 2
//   Srsi   [x,y,i,t] = (xi|yt)         B_vc   B_va   [x,i,y,t]  0 2 1 3
//   SrsiT  [y,x,i,t] = (xt|yi)         B_va   B_vc   [x,t,y,i]  2 0 3 1
//   Srs    [r,s,t,u] = (rt|su)         B_va   B_va   [r,t,s,u]  0 2 1 3
//   Sij    [t,u,i,j] = (it|ju)         B_ca   B_ca   [i,t,j,u]  1 3 0 2
//   Sir1   [r,t,i,u] = (ri|tu)         B_vc   B_aa   [r,i,t,u]  0 2 1 3
//   Sir2   [r,u,t,i] = (rt|iu)         B_va   B_ca   [r,t,i,u]  0 3 1 2
// The table's axes column.
std::array<int32_t, 4> slab_axes(const ExtBlock block) {
  switch (block) {
  case ExtBlock::Sijrs:
    return {0, 1, 2, 3};
  case ExtBlock::Si:
    return {1, 2, 0, 3};
  case ExtBlock::Sijr:
  case ExtBlock::Sir2:
    return {0, 3, 1, 2};
  case ExtBlock::SrsiT:
    return {2, 0, 3, 1};
  case ExtBlock::Sij:
    return {1, 3, 0, 2};
  case ExtBlock::Sr:
  case ExtBlock::Srsi:
  case ExtBlock::Srs:
  case ExtBlock::Sir1:
    return {0, 2, 1, 3};
  }
  std::unreachable();
}

// The table's X and Y columns.
SlabOperands slab_operands(const MoBlocks &m, const ExtBlock block) {
  const int32_t c = m.ncore;
  const int32_t a = m.ncas;
  const int32_t v = m.nvirt;
  const std::array<int32_t, 4> axes = slab_axes(block);
  switch (block) {
  case ExtBlock::Sr:
    return {m.va.data(), v, a, m.aa.data(), a, a, axes};
  case ExtBlock::Si:
    return {m.ca.data(), c, a, m.aa.data(), a, a, axes};
  case ExtBlock::Sijrs:
    return {m.cv.data(), c, v, m.cv.data(), c, v, axes};
  case ExtBlock::Sijr:
    return {m.vc.data(), v, c, m.ca.data(), c, a, axes};
  case ExtBlock::Srsi:
    return {m.vc.data(), v, c, m.va.data(), v, a, axes};
  case ExtBlock::SrsiT:
    return {m.va.data(), v, a, m.vc.data(), v, c, axes};
  case ExtBlock::Srs:
    return {m.va.data(), v, a, m.va.data(), v, a, axes};
  case ExtBlock::Sij:
    return {m.ca.data(), c, a, m.ca.data(), c, a, axes};
  case ExtBlock::Sir1:
    return {m.vc.data(), v, c, m.aa.data(), a, a, axes};
  case ExtBlock::Sir2:
    return {m.va.data(), v, a, m.ca.data(), c, a, axes};
  }
  std::unreachable();
}

// out = in permuted by axes: out axis k is in axis axes[k], in column-major
// with dimensions dims.
Status permute_slab(DeviceResources &res, const std::array<int64_t, 4> &dims,
                    const std::array<int32_t, 4> &axes, const double *in, double *out) {
  std::array<int64_t, 4> stride{1, 0, 0, 0};
  for (size_t k = 1; k < 4; ++k) {
    stride[k] = stride[k - 1] * dims[k - 1];
  }
  const auto ax = [&](const size_t k) { return static_cast<size_t>(axes[k]); };
  return gpu_status(device::permute(res.stream(), dims[ax(0)], dims[ax(1)], dims[ax(2)],
                                    dims[ax(3)], stride[ax(0)], stride[ax(1)], stride[ax(2)],
                                    stride[ax(3)], in, out));
}

// C[x, y, z, w] for x in [b0, b1), then its permute into out. C, column-major
// (nb, ny, nz, nw), is out itself for the identity, else workspace scratch.
//
// X's element (Q, x, y) is at Q + naux (x + nx y), so the batch's x for one
// y is the naux x nb matrix at naux (b0 + nx y), and consecutive y are
// naux nx apart: one gemm per y, C_y = X_y^T Y, batched over y. C_y lands at
// nb y with ldc nb ny, so C's element (x, y, z + nz w) is at
// x + nb (y + ny (z + nz w)): the demo's C[x, y, z, w], read column-major.
Status build_slab(DeviceResources &res, const int32_t naux, const SlabOperands &op,
                  const int32_t b0, const int32_t b1, double *out) {
  const int32_t nb = b1 - b0;
  const int64_t rows = int64_t{nb} * op.ny;
  const int64_t cols = int64_t{op.nz} * op.nw;
  const int64_t count = rows * cols;
  if (count == 0) {
    return {};
  }
  if (rows > std::numeric_limits<int32_t>::max() || cols > std::numeric_limits<int32_t>::max()) {
    return err_config("nevpt2::slab: the gemm's {} x {} product exceeds int32_t", rows, cols);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const bool identity = op.axes == std::array<int32_t, 4>{0, 1, 2, 3};
  double *const c =
      identity ? out : std::get<0>(res.device_regions<double>(static_cast<size_t>(count)));

  if (naux == 0) {
    // No fitting function to sum over: C = 0, which a gemm of inner dimension
    // 0 is not promised to write on both backends.
    FBF_TRY(gpu_status(
        wwr::wwrMemsetAsync(c, 0, sizeof(double) * static_cast<size_t>(count), res.stream())));
  } else {
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        res.blas().get(), wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, nb, static_cast<int32_t>(cols),
        naux, &kOne<double>, op.x + (int64_t{naux} * b0), naux, int64_t{naux} * op.nx, op.y, naux,
        0, &kZero<double>, c, static_cast<int32_t>(rows), nb, op.ny)));
  }
  if (identity) {
    return {};
  }
  return permute_slab(res, {nb, op.ny, op.nz, op.nw}, op.axes, c, out);
}

Result<MoBlocks> mo_blocks(DeviceResources &res, const int32_t naux, const int32_t nao,
                           const int32_t nmo, const double *b, const double *c, const int32_t ncore,
                           const int32_t ncas) {
  if (ncore < 0 || ncas < 0 || int64_t{ncore} + ncas > nmo) {
    return err_config("nevpt2::mo_blocks: ncore = {} and ncas = {} must be non-negative and fit "
                      "C's {} columns",
                      ncore, ncas, nmo);
  }
  const int32_t nvirt = nmo - ncore - ncas;
  const OrbitalRange core{.first = 0, .count = ncore};
  const OrbitalRange active{.first = ncore, .count = ncas};
  const OrbitalRange virt{.first = ncore + ncas, .count = nvirt};
  const auto n = [&](const OrbitalRange p, const OrbitalRange q) {
    return static_cast<size_t>(std::max(naux, 0)) * static_cast<size_t>(p.count) *
           static_cast<size_t>(q.count);
  };
  MoBlocks m{.naux = naux,
             .ncore = ncore,
             .ncas = ncas,
             .nvirt = nvirt,
             .aa = Buffers::device<double>(n(active, active), res.device()),
             .ca = Buffers::device<double>(n(core, active), res.device()),
             .va = Buffers::device<double>(n(virt, active), res.device()),
             .cv = Buffers::device<double>(n(core, virt), res.device()),
             .vc = Buffers::device<double>(n(virt, core), res.device())};
  FBF_TRY(b_mo(res, naux, nao, nmo, b, c, active, active, m.aa.data()));
  FBF_TRY(b_mo(res, naux, nao, nmo, b, c, core, active, m.ca.data()));
  FBF_TRY(b_mo(res, naux, nao, nmo, b, c, virt, active, m.va.data()));
  FBF_TRY(b_mo(res, naux, nao, nmo, b, c, core, virt, m.cv.data()));
  if (n(virt, core) > 0) {
    const ScopedDeviceIndex on{res.dev_idx()};
    FBF_TRY(permute_slab(res, {naux, ncore, nvirt, 1}, {0, 2, 1, 3}, m.cv.data(), m.vc.data()));
  }
  return m;
}

Status h2e(DeviceResources &res, const MoBlocks &blocks, double *out) {
  // h2e[t, u, v, w] = (tv|uw): C[t, v, u, w] from B_aa x B_aa, axes 0 2 1 3.
  const int32_t a = blocks.ncas;
  return build_slab(res, blocks.naux,
                    {blocks.aa.data(), a, a, blocks.aa.data(), a, a, {0, 2, 1, 3}}, 0, a, out);
}

int32_t batch_axis(const ExtBlock block) {
  const std::array<int32_t, 4> axes = slab_axes(block);
  return static_cast<int32_t>(std::ranges::find(axes, 0) - axes.begin());
}

std::array<int32_t, 4> block_shape(const MoBlocks &blocks, const ExtBlock block) {
  const SlabOperands op = slab_operands(blocks, block);
  const std::array<int32_t, 4> c{op.nx, op.ny, op.nz, op.nw};
  return {c[static_cast<size_t>(op.axes[0])], c[static_cast<size_t>(op.axes[1])],
          c[static_cast<size_t>(op.axes[2])], c[static_cast<size_t>(op.axes[3])]};
}

Status slab(DeviceResources &res, const MoBlocks &blocks, const ExtBlock block, const int32_t b0,
            const int32_t b1, double *out) {
  const SlabOperands op = slab_operands(blocks, block);
  if (b0 < 0 || b1 < b0 || b1 > op.nx) {
    return err_config("nevpt2::slab: [{}, {}) is outside the block's batch extent {}", b0, b1,
                      op.nx);
  }
  return build_slab(res, blocks.naux, op, b0, b1, out);
}

} // namespace free_breakfast::nevpt2
