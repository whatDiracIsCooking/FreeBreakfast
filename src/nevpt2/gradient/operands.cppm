/// @file
/// @brief The array plumbing both implementation units of
/// free_breakfast.nevpt2.gradient share: column-major einsum operands and
/// contractions, zeroed buffers, an in-place block add, the host copies, and
/// the entry points' count check.
///
/// An internal partition (`module`, not `export module`): only this module's
/// implementation units import it, so nothing here reaches an importer. See
/// docs/style.md, "Module-private code".

module free_breakfast.nevpt2.gradient:operands;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // Buffers, geam, wwrMemcpyAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.recipes.einsum;
import free_breakfast.ci.strings;

#include "diagnostics.h"

namespace free_breakfast::nevpt2 {

// A packed column-major einsum operand, its first index fastest: every array
// here but the RDMs, which are row-major and so enter with their labels
// reversed ("ut" is dm1[t][u]).
EinsumOperand column_operand(const double *data, std::vector<int64_t> dims) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (size_t k = 1; k < dims.size(); ++k) {
    strides[k] = strides[k - 1] * dims[k - 1];
  }
  return {.data = data, .dims = std::move(dims), .strides = std::move(strides)};
}

// @p out, packed column-major over @p out_dims, += alpha einsum(@p subscripts).
// einsum writes its output row-major, so the output's labels and extents go
// to it reversed. Nothing is launched when an extent is zero: the sum, or the
// output, is then empty.
Status contract(DeviceResources &res, const std::string_view subscripts,
                const std::vector<EinsumOperand> &operands, double *out,
                std::vector<int64_t> out_dims, const double alpha) {
  const auto empty = [](const std::vector<int64_t> &dims) {
    return std::ranges::find(dims, 0) != dims.end();
  };
  if (empty(out_dims) ||
      std::ranges::any_of(operands, [&](const EinsumOperand &op) { return empty(op.dims); })) {
    return {};
  }
  const size_t arrow = subscripts.find("->");
  std::string reversed{subscripts.substr(0, arrow + 2)};
  const std::string_view rhs = subscripts.substr(arrow + 2);
  reversed.append(rhs.rbegin(), rhs.rend());
  std::ranges::reverse(out_dims);
  return einsum(res.stream(), reversed, operands, out, out_dims, alpha);
}

// A zeroed device buffer of @p count doubles, which the accumulating einsums
// and geams add onto.
Result<Buffers::device<double>> zeroed(DeviceResources &res, const int64_t count) {
  Buffers::device<double> out(static_cast<size_t>(count), res.device());
  if (count > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(
        out.data(), 0, sizeof(double) * static_cast<size_t>(count), res.stream())));
  }
  return out;
}

// The packed m x n @p from added onto the m x n block at @p to, leading
// dimension @p ld: one in-place geam. A diagonal is m = 1 with ld + 1.
Status add_onto(DeviceResources &res, const int64_t m, const int64_t n, const double *from,
                double *to, const int64_t ld) {
  if (m * n == 0) {
    return {};
  }
  return gpu_status(wwr::geam<double, int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, static_cast<int32_t>(m),
      static_cast<int32_t>(n), &kOne<double>, from, static_cast<int32_t>(m), &kOne<double>, to,
      static_cast<int32_t>(ld), to, static_cast<int32_t>(ld)));
}

// @p count device doubles at @p data, on the host once res.stream() has
// reached them.
Result<std::vector<double>> to_host(DeviceResources &res, const double *data, const size_t count) {
  std::vector<double> host(count);
  if (count > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                           wwr::wwrMemcpyDeviceToHost, res.stream())));
    FBF_TRY(res.sync());
  }
  return host;
}

// @p host on the device, the copy landed before return: it reads pageable
// memory, which may go with the caller's frame.
Result<Buffers::device<double>> to_device(DeviceResources &res, std::span<const double> host) {
  Buffers::device<double> out(host.size(), res.device());
  if (!host.empty()) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.data(), host.data(), host.size_bytes(),
                                           wwr::wwrMemcpyHostToDevice, res.stream())));
    FBF_TRY(res.sync());
  }
  return out;
}

// The counts every entry point takes, against each other and the strings:
// InvalidConfig, naming @p caller, for counts that do not fit.
Status check_counts(const std::string_view caller, const ci::DeviceStringSet &alpha,
                    const ci::DeviceStringSet &beta, const int32_t naux, const int32_t nao,
                    const int32_t nmo, const int32_t ncore, const int32_t ncas) {
  if (naux < 0 || nao <= 0 || ncore < 0 || ncas <= 0 || nmo < ncore + ncas) {
    return err_config("{}: naux {}, nao {}, nmo {}, ncore {} and ncas {} do not fit: each "
                      "positive (naux and ncore may be 0) and ncore + ncas <= nmo",
                      caller, naux, nao, nmo, ncore, ncas);
  }
  if (alpha.n_orbitals() != ncas || beta.n_orbitals() != ncas) {
    return err_config("{}: strings over {} and {} orbitals, and {} are active", caller,
                      alpha.n_orbitals(), beta.n_orbitals(), ncas);
  }
  return {};
}

} // namespace free_breakfast::nevpt2
