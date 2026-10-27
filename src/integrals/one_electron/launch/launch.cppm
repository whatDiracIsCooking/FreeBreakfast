/// @file
/// @brief What every one-electron build shares beyond mmd's primitive-pair
/// records and the molecule's nuclei: the per-class launch over the records'
/// streams, and the checks every build runs before it launches anything, so
/// each family's module (overlap, ...) refuses the same inputs with the same
/// messages.

export module free_breakfast.integrals.one_electron.launch;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.integrals.mmd;
import free_breakfast.molecule;
import free_breakfast.recipes.matrix_ao;

export namespace free_breakfast {

/// @brief Call `launch(cls)` for every class of @p classes, between a fork of
/// the class's own stream from, and a join of it into, @p parent; @p launch
/// enqueues on `cls.device_stream()->stream()`.
///
/// Any random-access range of classes that carry a @c device_stream() will
/// do: mmd's @c PrimitivePairClass, and the exchange build's
/// @c OrderedPairClass.
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
template <std::ranges::random_access_range Classes, typename Launch>
  requires std::is_invocable_r_v<wwr::wwrError_t, Launch &,
                                 std::ranges::range_reference_t<const Classes>>
[[nodiscard]] wwr::wwrError_t launch_each(const Classes &classes, const wwr::wwrStream_t parent,
                                          Launch &&launch) {
  wwr::wwrError_t first = wwr::wwrSuccess;
  size_t forked = 0;
  for (const auto &cls : classes) {
    ForkStream &fork = cls.device_stream()->fork_stream();
    if (first = fork.fork_from(parent); first != wwr::wwrSuccess) {
      break;
    }
    ++forked;
    if (first = launch(cls); first != wwr::wwrSuccess) {
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

/// @brief What every one-electron build checks before it launches anything.
///
/// @p m and every class of @p classes must be on @p res's device, @p m's order
/// within the generated kernels' 32-bit `m * n_ao + n` index (n_ao^2 <=
/// INT32_MAX, about 46k AOs), and every class one with a kernel: la <= lb <=
/// @c kMaxAngularMomentum, the tree-wide limit (each family's launcher is
/// bounded by its own kernel table; this moves to them with #344). @p what
/// names the build in the messages.
///
/// @return Success, or InvalidConfig naming the first input refused.
[[nodiscard]] Status check_one_electron_build(const DeviceResources &res,
                                              std::span<const PrimitivePairClass> classes,
                                              const CartesianMatrixAO &m, std::string_view what);

/// @brief What a build that reads the nuclei checks beyond
/// @ref check_one_electron_build: @p molecule's nuclei on @p res's device.
/// @p what names the build in the message.
///
/// @return Success, or InvalidConfig if the nuclei are on another device.
[[nodiscard]] Status check_nuclei(const DeviceResources &res, const Molecule &molecule,
                                  std::string_view what);

} // namespace free_breakfast
