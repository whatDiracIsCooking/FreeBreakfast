/// @file
/// @brief The MO blocks (b_mo and a permute), h2e, and the slab builder: one
/// strided-batched gemm and a permute per slab, over the demo's (X, Y, axes)
/// table. And their adjoints onto Lambda: the transposed gemms, the inverse
/// permute, and in-place geams for bars on the blocks themselves.

module;

#include "nevpt2_permute_bridge.h"

module free_breakfast.nevpt2.density_fitting;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemmStridedBatched, geam, wwrMemsetAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.integrals.eri.density_fitting.mo;

#include "diagnostics.h"

namespace free_breakfast::nevpt2 {

// One of the MO blocks as an operand: its data, its pair's extents nx x ny,
// and where its element (Q, x, y) lies in Lambda, the MO-basis B's adjoint:
// at + x sx + y sy (Q fastest, at stride 1). B_vc's pair is B_cv's swapped,
// so its x is Lambda's slow orbital index.
struct BlockOperand {
  const double *data = nullptr;
  int32_t nx = 0;
  int32_t ny = 0;
  int64_t at = 0;
  int64_t sx = 0;
  int64_t sy = 0;
};

// The MO blocks, as BlockOperands.
enum class MoBlock { Aa, Ca, Va, Cv, Vc };

BlockOperand block_operand(const MoBlocks &m, const MoBlock which) {
  const int64_t naux = m.naux;
  const int64_t nmo = int64_t{m.ncore} + m.ncas + m.nvirt;
  const int32_t a0 = m.ncore;
  const int32_t v0 = m.ncore + m.ncas;
  // A pair (p, q) over p from p0, q from q0, not swapped.
  const auto plain = [&](const double *data, const int32_t p0, const int32_t np, const int32_t q0,
                         const int32_t nq) {
    return BlockOperand{data, np, nq, naux * (p0 + (nmo * q0)), naux, naux * nmo};
  };
  switch (which) {
  case MoBlock::Aa:
    return plain(m.aa.data(), a0, m.ncas, a0, m.ncas);
  case MoBlock::Ca:
    return plain(m.ca.data(), 0, m.ncore, a0, m.ncas);
  case MoBlock::Va:
    return plain(m.va.data(), v0, m.nvirt, a0, m.ncas);
  case MoBlock::Cv:
    return plain(m.cv.data(), 0, m.ncore, v0, m.nvirt);
  case MoBlock::Vc:
    return {m.vc.data(), m.nvirt, m.ncore, naux * nmo * v0, naux * nmo, naux};
  }
  std::unreachable();
}

// One slab's operands: C[x, y, z, w] = sum_Q X_Q,xy Y_Q,zw, X nx x ny and
// Y nz x nw orbital pairs, then slab axis k = C axis axes[k] (numpy's
// transpose convention).
struct SlabOperands {
  BlockOperand x;
  BlockOperand y;
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
  const auto op = [&](const MoBlock x, const MoBlock y) {
    return SlabOperands{block_operand(m, x), block_operand(m, y), slab_axes(block)};
  };
  switch (block) {
  case ExtBlock::Sr:
    return op(MoBlock::Va, MoBlock::Aa);
  case ExtBlock::Si:
    return op(MoBlock::Ca, MoBlock::Aa);
  case ExtBlock::Sijrs:
    return op(MoBlock::Cv, MoBlock::Cv);
  case ExtBlock::Sijr:
    return op(MoBlock::Vc, MoBlock::Ca);
  case ExtBlock::Srsi:
    return op(MoBlock::Vc, MoBlock::Va);
  case ExtBlock::SrsiT:
    return op(MoBlock::Va, MoBlock::Vc);
  case ExtBlock::Srs:
    return op(MoBlock::Va, MoBlock::Va);
  case ExtBlock::Sij:
    return op(MoBlock::Ca, MoBlock::Ca);
  case ExtBlock::Sir1:
    return op(MoBlock::Vc, MoBlock::Aa);
  case ExtBlock::Sir2:
    return op(MoBlock::Va, MoBlock::Ca);
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
  const int64_t rows = int64_t{nb} * op.x.ny;
  const int64_t cols = int64_t{op.y.nx} * op.y.ny;
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
        naux, &kOne<double>, op.x.data + (int64_t{naux} * b0), naux, int64_t{naux} * op.x.nx,
        op.y.data, naux, 0, &kZero<double>, c, static_cast<int32_t>(rows), nb, op.x.ny)));
  }
  if (identity) {
    return {};
  }
  return permute_slab(res, {nb, op.x.ny, op.y.nx, op.y.ny}, op.axes, c, out);
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
  const BlockOperand aa = block_operand(blocks, MoBlock::Aa);
  return build_slab(res, blocks.naux, {aa, aa, {0, 2, 1, 3}}, 0, a, out);
}

int32_t batch_axis(const ExtBlock block) {
  const std::array<int32_t, 4> axes = slab_axes(block);
  return static_cast<int32_t>(std::ranges::find(axes, 0) - axes.begin());
}

std::array<int32_t, 4> block_shape(const MoBlocks &blocks, const ExtBlock block) {
  const SlabOperands op = slab_operands(blocks, block);
  const std::array<int32_t, 4> c{op.x.nx, op.x.ny, op.y.nx, op.y.ny};
  return {c[static_cast<size_t>(op.axes[0])], c[static_cast<size_t>(op.axes[1])],
          c[static_cast<size_t>(op.axes[2])], c[static_cast<size_t>(op.axes[3])]};
}

Status slab(DeviceResources &res, const MoBlocks &blocks, const ExtBlock block, const int32_t b0,
            const int32_t b1, double *out) {
  const SlabOperands op = slab_operands(blocks, block);
  if (b0 < 0 || b1 < b0 || b1 > op.x.nx) {
    return err_config("nevpt2::slab: [{}, {}) is outside the block's batch extent {}", b0, b1,
                      op.x.nx);
  }
  return build_slab(res, blocks.naux, op, b0, b1, out);
}

// The adjoints. Each forward map is C = X_b^T Y over the batch, then a
// permute, so with C-bar the slab bar permuted back,
//   X-bar_y += Y C-bar_y^T     (one gemm per y, batched over y)
//   Y-bar   += X_b C-bar       (one gemm per w, batched over w)
// written straight into Lambda at the operands' places there: Lambda's
// element (Q, x, y) of a block lies at at + x sx + y sy, Q fastest, so each
// block's x columns (y for B_vc's swapped pair) are a strided run of
// Lambda's naux-long columns and a gemm's ldc reaches them. The bars on the
// blocks are never formed.

// lambda += the adjoint of build_slab at op, for the slab bar over [b0, b1).
// C-bar, when a permute was taken, and X_b, when the batch is not all of X's
// first index (its columns are then not uniformly strided), are workspace
// scratch.
Status build_slab_adjoint(DeviceResources &res, const int32_t naux, const SlabOperands &op,
                          const int32_t b0, const int32_t b1, const double *bar, double *lambda) {
  const int32_t nb = b1 - b0;
  const int64_t rows = int64_t{nb} * op.x.ny;
  const int64_t cols = int64_t{op.y.nx} * op.y.ny;
  const int64_t count = rows * cols;
  if (count == 0 || naux == 0) {
    return {};
  }
  constexpr int64_t kIntMax = std::numeric_limits<int32_t>::max();
  if (rows > kIntMax || cols > kIntMax || op.x.sx > kIntMax || op.y.sx > kIntMax) {
    return err_config("nevpt2::slab_adjoint: the gemms' {} x {} product or Lambda's leading "
                      "dimension {} exceeds int32_t",
                      rows, cols, std::max(op.x.sx, op.y.sx));
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const bool identity = op.axes == std::array<int32_t, 4>{0, 1, 2, 3};
  const bool whole = nb == op.x.nx;
  const auto [c_scratch, x_scratch] =
      res.device_regions<double>(identity ? size_t{0} : static_cast<size_t>(count),
                                 whole ? size_t{0} : static_cast<size_t>(int64_t{naux} * rows));

  // C-bar: the slab bar permuted back, out axis axes[k] = slab axis k.
  const double *c_bar = bar;
  if (!identity) {
    const std::array<int64_t, 4> c_dims{nb, op.x.ny, op.y.nx, op.y.ny};
    std::array<int64_t, 4> slab_dims{};
    std::array<int32_t, 4> inverse{};
    for (size_t k = 0; k < 4; ++k) {
      const auto j = static_cast<size_t>(op.axes[k]);
      slab_dims[k] = c_dims[j];
      inverse[j] = static_cast<int32_t>(k);
    }
    FBF_TRY(permute_slab(res, slab_dims, inverse, bar, c_scratch));
    c_bar = c_scratch;
  }
  // X_b, naux x nb x ny packed: X's columns b0 + x of each y, gathered.
  const double *x_b = op.x.data;
  if (!whole) {
    FBF_TRY(gpu_status(device::permute(res.stream(), naux, nb, op.x.ny, 1, 1, naux,
                                       int64_t{naux} * op.x.nx, 0, op.x.data + (int64_t{naux} * b0),
                                       x_scratch)));
    x_b = x_scratch;
  }

  // X-bar_y (naux x nb) += Y (naux x nz nw) C-bar_y^T: C-bar_y is the nb x
  // (nz nw) matrix at nb y, leading dimension nb ny.
  FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, naux, nb, static_cast<int32_t>(cols),
      &kOne<double>, op.y.data, naux, 0, c_bar, static_cast<int32_t>(rows), nb, &kOne<double>,
      lambda + op.x.at + (b0 * op.x.sx), static_cast<int32_t>(op.x.sx), op.x.sy, op.x.ny)));
  // Y-bar_w (naux x nz) += X_b (naux x nb ny) C-bar_w: C-bar_w is the
  // (nb ny) x nz matrix at nb ny nz w.
  return gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, naux, op.y.nx,
      static_cast<int32_t>(rows), &kOne<double>, x_b, naux, 0, c_bar, static_cast<int32_t>(rows),
      rows * op.y.nx, &kOne<double>, lambda + op.y.at, static_cast<int32_t>(op.y.sx), op.y.sy,
      op.y.ny));
}

Status h2e_adjoint(DeviceResources &res, const MoBlocks &blocks, const double *h2e_bar,
                   double *lambda) {
  const BlockOperand aa = block_operand(blocks, MoBlock::Aa);
  return build_slab_adjoint(res, blocks.naux, {aa, aa, {0, 2, 1, 3}}, 0, blocks.ncas, h2e_bar,
                            lambda);
}

Status slab_adjoint(DeviceResources &res, const MoBlocks &blocks, const ExtBlock block,
                    const int32_t b0, const int32_t b1, const double *slab_bar, double *lambda) {
  const SlabOperands op = slab_operands(blocks, block);
  if (b0 < 0 || b1 < b0 || b1 > op.x.nx) {
    return err_config("nevpt2::slab_adjoint: [{}, {}) is outside the block's batch extent {}", b0,
                      b1, op.x.nx);
  }
  return build_slab_adjoint(res, blocks.naux, op, b0, b1, slab_bar, lambda);
}

Status mo_blocks_adjoint(DeviceResources &res, const MoBlocks &bar, const double *core_diagonal_bar,
                         const double *virt_diagonal_bar, double *lambda) {
  if (bar.naux < 0 || bar.ncore < 0 || bar.ncas < 0 || bar.nvirt < 0) {
    return err_config("nevpt2::mo_blocks_adjoint: naux = {}, ncore = {}, ncas = {} and nvirt = "
                      "{} must be non-negative",
                      bar.naux, bar.ncore, bar.ncas, bar.nvirt);
  }
  const int64_t naux = bar.naux;
  const int64_t nmo = int64_t{bar.ncore} + bar.ncas + bar.nvirt;
  // The diagonal's stride, naux (nmo + 1), is the largest leading dimension.
  if (naux * (nmo + 1) > std::numeric_limits<int32_t>::max()) {
    return err_config("nevpt2::mo_blocks_adjoint: Lambda's leading dimension {} exceeds int32_t",
                      naux * (nmo + 1));
  }
  const std::array<std::pair<MoBlock, const char *>, 5> all{{{MoBlock::Aa, "aa"},
                                                             {MoBlock::Ca, "ca"},
                                                             {MoBlock::Va, "va"},
                                                             {MoBlock::Cv, "cv"},
                                                             {MoBlock::Vc, "vc"}}};
  const auto buffer = [&](const MoBlock which) -> const Buffers::device<double> & {
    switch (which) {
    case MoBlock::Aa:
      return bar.aa;
    case MoBlock::Ca:
      return bar.ca;
    case MoBlock::Va:
      return bar.va;
    case MoBlock::Cv:
      return bar.cv;
    case MoBlock::Vc:
      return bar.vc;
    }
    std::unreachable();
  };
  for (const auto &[which, name] : all) {
    const BlockOperand op = block_operand(bar, which);
    const size_t n = buffer(which).num_elements();
    const auto want = static_cast<size_t>(naux * op.nx * op.ny);
    if (n != 0 && n != want) {
      return err_config("nevpt2::mo_blocks_adjoint: bar.{} holds {} doubles, not {} or none", name,
                        n, want);
    }
  }
  if (naux == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto blas = res.blas().get();
  const auto ld_lambda = static_cast<int32_t>(naux * nmo);

  // lambda's (naux nx) x ny run at op += the packed (naux nx) x ny bar: a
  // plain pair's x columns are contiguous in Lambda, so one in-place geam.
  const auto add_plain = [&](const BlockOperand &op, const double *b) -> Status {
    if (int64_t{op.nx} * op.ny == 0) {
      return {};
    }
    const auto m = static_cast<int32_t>(naux * op.nx);
    return gpu_status(wwr::geam<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, op.ny, &kOne<double>, b, m, &kOne<double>,
        lambda + op.at, ld_lambda, lambda + op.at, ld_lambda));
  };
  for (const MoBlock which : {MoBlock::Aa, MoBlock::Ca, MoBlock::Va, MoBlock::Cv}) {
    if (buffer(which).num_elements() != 0) {
      FBF_TRY(add_plain(block_operand(bar, which), buffer(which).data()));
    }
  }
  if (bar.vc.num_elements() != 0) {
    // B_vc's bar, swapped back into B_cv's layout, then added at B_cv's place.
    double *const cv = std::get<0>(res.device_regions<double>(bar.vc.num_elements()));
    FBF_TRY(permute_slab(res, {naux, bar.nvirt, bar.ncore, 1}, {0, 2, 1, 3}, bar.vc.data(), cv));
    FBF_TRY(add_plain(block_operand(bar, MoBlock::Cv), cv));
  }

  // B_Q,pp at Q + naux (nmo + 1) p: the diagonal is a naux x n matrix of
  // leading dimension naux (nmo + 1).
  const auto ld_diagonal = static_cast<int32_t>(naux * (nmo + 1));
  const auto add_diagonal = [&](const int64_t p0, const int32_t n, const double *b) -> Status {
    if (b == nullptr || n == 0) {
      return {};
    }
    double *const at = lambda + (naux * (nmo + 1) * p0);
    return gpu_status(wwr::geam<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, static_cast<int32_t>(naux), n, &kOne<double>, b,
        static_cast<int32_t>(naux), &kOne<double>, at, ld_diagonal, at, ld_diagonal));
  };
  FBF_TRY(add_diagonal(0, bar.ncore, core_diagonal_bar));
  return add_diagonal(int64_t{bar.ncore} + bar.ncas, bar.nvirt, virt_diagonal_bar);
}

} // namespace free_breakfast::nevpt2
