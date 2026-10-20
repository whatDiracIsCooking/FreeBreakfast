/// @file
/// @brief What every one-electron build shares beyond mmd's primitive-pair
/// records: the nuclei, once per geometry; the per-class launch over the
/// records' streams; and the check every build runs before it launches
/// anything, so each family's module (overlap, ...) refuses the same inputs
/// with the same messages.
///
/// The nuclei's owner here holds wwr buffers; the non-owning views are
/// device_views.h's, included in the global module fragment and re-exported so
/// importers and the launchers' #include share one entity (atom.cppm has the
/// why of the in-namespace using-declarations).

module;

#include "device_views.h"

export module free_breakfast.integrals.one_electron.upload;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.integrals.mmd;
import free_breakfast.molecule;
import free_breakfast.linalg.matrix_ao;

export namespace free_breakfast {

using free_breakfast::NucleiView;
using free_breakfast::PrimitivePairClassView;

/// @brief Call `launch(stream, cls)` for every class of @p classes, with that
/// class's own stream, between a fork from and a join into @p parent.
///
/// Every class's work is ordered after everything already queued on
/// @p parent -- the matrix's zeroing -- and everything queued on @p parent
/// afterwards after every class's work, so the classes run concurrently and
/// what follows sees the whole matrix. The classes are not ordered among
/// themselves; the kernels add atomically, so classes running at once into one
/// matrix do not race.
///
/// After the first failure no further class is launched, but every stream
/// already forked is still joined back, so @p parent never runs ahead of a
/// kernel in flight: a matrix freed on it afterwards is not freed under one.
/// The forks and joins record events on the classes' streams, which are
/// shared, so this mutates them although @p classes is const.
///
/// @return wwrSuccess, or the first error of a fork, launch or join.
template <typename Launch>
[[nodiscard]] wwr::wwrError_t launch_each(std::span<const PrimitivePairClass> classes,
                                          const wwr::wwrStream_t parent, Launch &&launch) {
  wwr::wwrError_t first = wwr::wwrSuccess;
  size_t forked = 0;
  for (const PrimitivePairClass &cls : classes) {
    ForkStream &fork = cls.device_stream()->fork_stream();
    if (first = fork.fork_from(parent); first != wwr::wwrSuccess) {
      break;
    }
    ++forked;
    if (first = launch(static_cast<wwr::wwrStream_t>(fork.stream()), cls);
        first != wwr::wwrSuccess) {
      break;
    }
  }
  for (size_t i = 0; i < forked; ++i) {
    ForkStream &fork = classes[i].device_stream()->fork_stream();
    wwr::wwrError_t code = fork.record_join();
    if (code == wwr::wwrSuccess) {
      code = fork.join_into(parent);
    }
    if (first == wwr::wwrSuccess) {
      first = code;
    }
  }
  return first;
}

/// @brief A molecule's nuclei on the device: the point charges V attracts
/// to.
struct DeviceNuclei {
  /// @brief The device the buffers are on (@c DeviceResources::dev_idx).
  int32_t dev_idx;
  /// @brief The number of nuclei, positive.
  int32_t n_nuc;
  /// @brief Z_C, each nucleus's charge (its atomic number).
  Buffers::device<double> z;
  /// @brief C, each nucleus's position in bohr (3 per nucleus).
  Buffers::device<double> c;

  /// @brief The nuclei by pointer, for a launcher. Valid while this is alive
  /// and unmoved.
  [[nodiscard]] NucleiView view() const { return {.n_nuc = n_nuc, .z = z.data(), .c = c.data()}; }
};

/// @brief Copy @p molecule's nuclei, in atom order, to @p res's device.
///
/// Synchronizes res.stream() once, after the copies. The device res.dev_idx()
/// names is made current for the call and the previous one restored.
///
/// @return The device nuclei; InvalidConfig for a molecule with no atoms, or
///         more than an int32_t counts; or the GpuRuntime error of the first
///         copy that failed.
[[nodiscard]] Result<DeviceNuclei> upload_nuclei(DeviceResources &res, const Molecule &molecule);

/// @brief What every one-electron build checks before it launches anything.
///
/// @p m and every class of @p classes must be on @p res's device, @p m's order
/// within the generated kernels' 32-bit `m * n_ao + n` index (n_ao^2 <=
/// INT32_MAX, about 46k AOs), and every class one with a kernel: la <= lb <=
/// @c kMaxAngularMomentum, the one limit every family is generated to. @p what
/// names the build in the messages.
///
/// @return Success, or InvalidConfig naming the first input refused.
[[nodiscard]] Status check_one_electron_build(const DeviceResources &res,
                                              std::span<const PrimitivePairClass> classes,
                                              const CartesianMatrixAO &m, std::string_view what);

/// @brief What a build that reads the nuclei checks beyond
/// @ref check_one_electron_build: @p nuclei on @p res's device. @p what names
/// the build in the message.
///
/// @return Success, or InvalidConfig if the nuclei are on another device.
[[nodiscard]] Status check_nuclei(const DeviceResources &res, const DeviceNuclei &nuclei,
                                  std::string_view what);

} // namespace free_breakfast
