/// @file
/// @brief The device inputs every one-electron build reads: the primitive-pair
/// records, once per basis, and the nuclei, once per geometry.
///
/// Also the check every build runs before it launches anything, so each
/// family's module (overlap, ...) refuses the same inputs with the same
/// messages.
///
/// The owners here hold wwr buffers; the non-owning views they hand out are
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
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.molecule;
import free_breakfast.linalg.matrix_ao;

export namespace free_breakfast {

using free_breakfast::NucleiView;
using free_breakfast::PrimitivePairClassView;

/// @brief One @ref PrimitivePairClass on the device: the same fields, the same
/// layout, @ref n_rec records, and the work stream they live on.
struct DevicePrimitivePairClass {
  /// @brief The bra shell's angular momentum, <= @ref lb.
  int32_t la;
  /// @brief The ket shell's angular momentum.
  int32_t lb;
  /// @brief The number of records, positive.
  int32_t n_rec;
  /// @brief The class's own work stream on the Device: the records were
  /// allocated and copied on it, and every build launches the class on it.
  std::shared_ptr<DeviceStream> device_stream;
  /// @brief zeta = alpha + beta.
  StreamBuffers::device<double> zeta;
  /// @brief P: x.
  StreamBuffers::device<double> p_x;
  /// @brief P: y.
  StreamBuffers::device<double> p_y;
  /// @brief P: z.
  StreamBuffers::device<double> p_z;
  /// @brief P - A: x.
  StreamBuffers::device<double> pa_x;
  /// @brief P - A: y.
  StreamBuffers::device<double> pa_y;
  /// @brief P - A: z.
  StreamBuffers::device<double> pa_z;
  /// @brief P - B: x.
  StreamBuffers::device<double> pb_x;
  /// @brief P - B: y.
  StreamBuffers::device<double> pb_y;
  /// @brief P - B: z.
  StreamBuffers::device<double> pb_z;
  /// @brief c_a c_b (pi / zeta)^{3/2}.
  StreamBuffers::device<double> s;
  /// @brief beta, the ket primitive's exponent.
  StreamBuffers::device<double> eb;
  /// @brief The bra shell's first Cartesian AO.
  StreamBuffers::device<int32_t> off_a;
  /// @brief The ket shell's first Cartesian AO.
  StreamBuffers::device<int32_t> off_b;

  /// @brief The records by pointer, for a launcher. Valid while this is alive
  /// and unmoved.
  [[nodiscard]] PrimitivePairClassView view() const {
    return {
        .la = la,
        .lb = lb,
        .n_rec = n_rec,
        .zeta = zeta.data(),
        .p_x = p_x.data(),
        .p_y = p_y.data(),
        .p_z = p_z.data(),
        .pa_x = pa_x.data(),
        .pa_y = pa_y.data(),
        .pa_z = pa_z.data(),
        .pb_x = pb_x.data(),
        .pb_y = pb_y.data(),
        .pb_z = pb_z.data(),
        .s = s.data(),
        .eb = eb.data(),
        .off_a = off_a.data(),
        .off_b = off_b.data(),
    };
  }
};

/// @brief Every primitive-pair class of a basis on one device, each on its own
/// work stream, so a build runs the classes concurrently.
///
/// A build forks every class's stream from its own (@ref launch_each), so the
/// classes start after what is already queued there -- the matrix's zeroing --
/// and joins them back, so what follows on that stream sees every class's
/// contributions. The kernels add atomically, so classes running at once into
/// one matrix do not race.
///
/// Held once per basis. Movable, not copyable.
class DevicePrimitivePairs {
public:
  /// @brief @p classes, every one on device @p dev_idx.
  DevicePrimitivePairs(int32_t dev_idx, std::vector<DevicePrimitivePairClass> classes)
      : dev_idx_(dev_idx), classes_(std::move(classes)) {}

  /// @brief The device every class is on (@c DeviceResources::dev_idx).
  int32_t dev_idx() const noexcept { return dev_idx_; }

  /// @brief The classes, in @ref primitive_pair_classes order.
  std::span<const DevicePrimitivePairClass> classes() const noexcept { return classes_; }

  /// @brief Call `launch(stream, cls)` for every class, with that class's own
  /// stream, between a fork from and a join into @p parent.
  ///
  /// Every class's work is ordered after everything already queued on
  /// @p parent, and everything queued on @p parent afterwards after every
  /// class's work. The classes are not ordered among themselves.
  ///
  /// After the first failure no further class is launched, but every stream
  /// already forked is still joined back, so @p parent never runs ahead of a
  /// kernel in flight: a matrix freed on it afterwards is not freed under one.
  /// The forks and joins record events on the classes' streams, which are
  /// shared, so this mutates them although it is const.
  ///
  /// @return wwrSuccess, or the first error of a fork, launch or join.
  template <typename Launch>
  [[nodiscard]] wwr::wwrError_t launch_each(wwr::wwrStream_t parent, Launch &&launch) const {
    wwr::wwrError_t first = wwr::wwrSuccess;
    size_t forked = 0;
    for (const DevicePrimitivePairClass &cls : classes_) {
      ForkStream &fork = cls.device_stream->fork_stream();
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
      ForkStream &fork = classes_[i].device_stream->fork_stream();
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

private:
  int32_t dev_idx_;
  std::vector<DevicePrimitivePairClass> classes_;
};

/// @brief Copy @p classes to @p res's device, each class on a new
/// @c DeviceStream of res.device().
///
/// Synchronizes every class's stream once, after the copies, so @p classes may
/// be released when this returns. The device res.dev_idx() names is made
/// current for the call and the previous one restored.
///
/// @return The device classes, in the same order; InvalidConfig for an empty
///         class; or the GpuRuntime error of the first copy or synchronization
///         that failed.
[[nodiscard]] Result<DevicePrimitivePairs>
upload_primitive_pairs(DeviceResources &res, std::span<const PrimitivePairClass> classes);

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
/// @p m and @p pairs must be on @p res's device, @p m's order within the
/// generated kernels' 32-bit `m * n_ao + n` index (n_ao^2 <= INT32_MAX, about
/// 46k AOs), and every class one with a kernel: la <= lb <=
/// @c kMaxAngularMomentum, the one limit every family is generated to. @p what
/// names the build in the messages.
///
/// @return Success, or InvalidConfig naming the first input refused.
[[nodiscard]] Status check_one_electron_build(const DeviceResources &res,
                                              const DevicePrimitivePairs &pairs,
                                              const CartesianMatrixAO &m, std::string_view what);

/// @brief What a build that reads the nuclei checks beyond
/// @ref check_one_electron_build: @p nuclei on @p res's device. @p what names
/// the build in the message.
///
/// @return Success, or InvalidConfig if the nuclei are on another device.
[[nodiscard]] Status check_nuclei(const DeviceResources &res, const DeviceNuclei &nuclei,
                                  std::string_view what);

} // namespace free_breakfast
