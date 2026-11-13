/// @file
/// @brief `free-breakfast-mmd-records`: the primitive-pair records of
///        H2O/cc-pVDZ and CO/cc-pVTZ, as `primitive_pair_classes` forms them,
///        copied back and written as raw little-endian files: the reference
///        `rust/integrals/mmd`'s tier-4 test compares its records against.
///
/// Each molecule's basis is the BasisSet step's: the compiled-in set through
/// @c resolve_basis, its primitives normalized (@c primitive_normalized) and
/// placed on the atoms. The geometries, in bohr, are those of
/// `test/gtest/data/coulomb/{h2o_ccpvdz,co_ccpvtz}.json`, the molecules
/// `mmd_gpu.cpp` checks the bounds of, copied here as literals, and copied
/// again into the Rust test, so the two trees form the same records.
///
/// `<out>/<case>.bin` is, all little-endian, with no padding:
///
/// - the 8 bytes `fbmmd v1`, then the class count as a `uint32`;
/// - per class, in `primitive_pair_classes`' (ascending) order: `la`, `lb`
///   and `n_rec` as `int32`; then 16 fields of `n_rec` doubles each, `zeta`,
///   `p_x`, `p_y`, `p_z`, `pa_x`, `pa_y`, `pa_z`, `pb_x`, `pb_y`, `pb_z`,
///   `s`, `ea`, `eb`, `bound`, `bound_ab`, `bound_ba`; then 4 fields of
///   `n_rec` `int32` each, `off_a`, `off_b`, `rank_a`, `rank_b`.
///
/// `rust/integrals/mmd/tests/data/README.md` has the command. Needs a GPU
/// (device 0).

#include "diagnostics.h"

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.atom;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.basis;
import free_breakfast.integrals.mmd;

namespace {

namespace fb = free_breakfast;

static_assert(std::endian::native == std::endian::little,
              "the files are written as the host lays them out");

// One molecule: its file stem, its basis's name and its atoms (bohr).
struct Case {
  std::string_view stem;
  std::string_view basis;
  std::vector<fb::Atom> atoms;
};

std::vector<Case> cases() {
  using V = fb::Vec<double, 3>;
  return {
      {.stem = "h2o_ccpvdz",
       .basis = "cc-pvdz",
       .atoms = {{V{0.0, 0.0, 0.0}, fb::Element::O},
                 {V{1.43, 0.21, -1.07}, fb::Element::H},
                 {V{-1.38, 0.35, -1.12}, fb::Element::H}}},
      {.stem = "co_ccpvtz",
       .basis = "cc-pvtz",
       .atoms = {{V{0.13, -0.21, 0.17}, fb::Element::C}, {V{0.77, 0.86, 1.86}, fb::Element::O}}},
  };
}

// The raw bytes of values, appended to out.
template <typename T>
void put(std::vector<char> &out, const std::span<const T> values) {
  const auto bytes = std::as_bytes(values);
  std::ranges::transform(bytes, std::back_inserter(out),
                         [](const std::byte b) { return static_cast<char>(b); });
}

// field copied back on stream and appended to out, once the stream is
// drained (the caller's sync).
template <typename T>
fb::Status copy_back(const fb::StreamBuffers::device<T> &field, const std::size_t n,
                     const wwr::wwrStream_t stream, std::vector<T> &host) {
  host.resize(n);
  return fb::gpu_status(wwr::wwrMemcpyAsync(host.data(), field.data(), sizeof(T) * n,
                                            wwr::wwrMemcpyDeviceToHost, stream));
}

fb::Status write_case(const std::filesystem::path &out_dir, const Case &c) {
  fb::DeviceResources res{0, fb::MemoryFootprint{.device = {.workspace = std::size_t{64} << 20},
                                                 .pinned = {.workspace = std::size_t{4} << 20}}};
  const fb::Molecule molecule(res, c.atoms);
  const fb::BasisStamp stamp = FBF_TRY(fb::resolve_basis(c.basis));
  const fb::BasisSet basis = FBF_TRY(fb::place(stamp.primitive_normalized(), molecule));
  const std::vector<fb::PrimitivePairClass> classes =
      FBF_TRY(fb::primitive_pair_classes(res, basis));

  std::vector<char> out;
  put(out, std::span<const char>{"fbmmd v1", 8});
  const auto n_classes = static_cast<std::uint32_t>(classes.size());
  put(out, std::span{&n_classes, 1});
  std::size_t n_total = 0;
  for (const fb::PrimitivePairClass &cls : classes) {
    const std::array<std::int32_t, 3> head{cls.la, cls.lb, cls.n_rec};
    put(out, std::span<const std::int32_t>{head});
    const auto n = static_cast<std::size_t>(cls.n_rec);
    n_total += n;
    const wwr::wwrStream_t stream = cls.device_stream()->stream();
    const std::array doubles{&cls.zeta, &cls.p_x,   &cls.p_y,      &cls.p_z,
                             &cls.pa_x, &cls.pa_y,  &cls.pa_z,     &cls.pb_x,
                             &cls.pb_y, &cls.pb_z,  &cls.s,        &cls.ea,
                             &cls.eb,   &cls.bound, &cls.bound_ab, &cls.bound_ba};
    const std::array ints{&cls.off_a, &cls.off_b, &cls.rank_a, &cls.rank_b};
    std::vector<std::vector<double>> d(doubles.size());
    std::vector<std::vector<std::int32_t>> i(ints.size());
    for (std::size_t f = 0; f < doubles.size(); ++f) {
      FBF_TRY(copy_back(*doubles[f], n, stream, d[f]));
    }
    for (std::size_t f = 0; f < ints.size(); ++f) {
      FBF_TRY(copy_back(*ints[f], n, stream, i[f]));
    }
    FBF_TRY(fb::gpu_status(wwr::wwrStreamSynchronize(stream)));
    for (const std::vector<double> &field : d) {
      put(out, std::span<const double>{field});
    }
    for (const std::vector<std::int32_t> &field : i) {
      put(out, std::span<const std::int32_t>{field});
    }
  }

  const std::filesystem::path path = out_dir / std::format("{}.bin", c.stem);
  std::ofstream file(path, std::ios::binary);
  file.write(out.data(), static_cast<std::streamsize>(out.size()));
  if (!file) {
    return fb::err_io("cannot write {}", path.string());
  }
  std::println("{}: {} classes, {} records, {} bytes", path.string(), classes.size(), n_total,
               out.size());
  return {};
}

fb::Status run(std::span<char *const> args) {
  if (args.size() != 2) {
    return fb::err_config("usage: {} <out dir>",
                          args.empty() ? "free-breakfast-mmd-records" : args[0]);
  }
  const std::filesystem::path out_dir{args[1]};
  for (const Case &c : cases()) {
    FBF_TRY(write_case(out_dir, c));
  }
  return {};
}

} // namespace

// int-type-ok: argc's type is fixed by the standard's signature for main.
int main(int argc, char **argv) {
  const fb::Status status = run(std::span<char *const>(argv, static_cast<std::size_t>(argc)));
  if (!status) {
    std::println(std::cerr, "error: {}", status.error().message);
    return fb::to_exit_code(status.error().kind);
  }
  return 0;
}
