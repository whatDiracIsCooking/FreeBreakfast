/// @file
/// @brief pipek_mezey_localize: the sweep driver, and its two instantiations.

module;

// The launchers, defined in pm_kernels.cu, declared in the GMF so they keep
// external linkage (see the header); and pm_fill_round_robin, which the driver
// runs on the host.
#include "pm_bridge.h"
#include "pm_math.h"

module free_breakfast.localization.pm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemm, wwrMemcpyAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;

#include "diagnostics.h"

namespace free_breakfast {

// Atom A's spherical AOs are [offsets[A], offsets[A + 1]). A placed basis
// holds each atom's shells together, in atom order, and the AO layout follows
// the shell order, so an atom's AOs are one contiguous range.
std::vector<int32_t> pm_atom_ao_offsets(const BasisSet &basis) {
  const uint32_t n_atoms = basis.natoms();
  const uint32_t n_shells = basis.nshells();
  std::vector<int32_t> offsets(static_cast<size_t>(n_atoms) + 1);
  for (uint32_t a = 0; a <= n_atoms; ++a) {
    // An atom with no shells, or the end, starts where the next AO would.
    const uint32_t shell = a < n_atoms ? basis.atom_shell_offset(a) : n_shells;
    offsets[a] =
        static_cast<int32_t>(shell < n_shells ? basis.ao_offset_sphe(shell) : basis.nao_sphe());
  }
  return offsets;
}

// The call's device scratch, one region per buffer, carved from one
// workspace request. Each region is a whole number of alignment steps, so
// every one starts aligned for its type. Nothing here calls another workspace
// user while these are live.
template <typename T>
struct PmScratch {
  T *w;                     // S C, n_ao x n_occ
  T *pops_a;                // the population tensor, ping
  T *pops_b;                // and pong
  T *u_a;                   // the accumulated U, ping
  T *u_b;                   // and pong
  T *cos;                   // a round's per-orbital rotation
  T *sin;                   //
  T *functional;            // L, one element
  int32_t *partner;         // a round's per-orbital partner
  int32_t *schedule;        // every round's pairs
  int32_t *atom_ao_offsets; // n_atoms + 1

  PmScratch(DeviceResources &res, const size_t n_ao, const size_t n_occ, const size_t n_atoms,
            const size_t schedule_size) {
    const size_t pops = n_atoms * n_occ * n_occ;
    size_t bytes = 0;
    const auto region = [&bytes](const size_t count, const size_t size) {
      const size_t at = bytes;
      bytes += math::align_up(count * size, DeviceResources::kWorkspaceAlignment);
      return at;
    };
    const size_t at_w = region(n_ao * n_occ, sizeof(T));
    const size_t at_pops_a = region(pops, sizeof(T));
    const size_t at_pops_b = region(pops, sizeof(T));
    const size_t at_u_a = region(n_occ * n_occ, sizeof(T));
    const size_t at_u_b = region(n_occ * n_occ, sizeof(T));
    const size_t at_cos = region(n_occ, sizeof(T));
    const size_t at_sin = region(n_occ, sizeof(T));
    const size_t at_functional = region(1, sizeof(T));
    const size_t at_partner = region(n_occ, sizeof(int32_t));
    const size_t at_schedule = region(schedule_size, sizeof(int32_t));
    const size_t at_offsets = region(n_atoms + 1, sizeof(int32_t));

    std::byte *const base = res.device_workspace(bytes).data();
    w = reinterpret_cast<T *>(base + at_w);
    pops_a = reinterpret_cast<T *>(base + at_pops_a);
    pops_b = reinterpret_cast<T *>(base + at_pops_b);
    u_a = reinterpret_cast<T *>(base + at_u_a);
    u_b = reinterpret_cast<T *>(base + at_u_b);
    cos = reinterpret_cast<T *>(base + at_cos);
    sin = reinterpret_cast<T *>(base + at_sin);
    functional = reinterpret_cast<T *>(base + at_functional);
    partner = reinterpret_cast<int32_t *>(base + at_partner);
    schedule = reinterpret_cast<int32_t *>(base + at_schedule);
    atom_ao_offsets = reinterpret_cast<int32_t *>(base + at_offsets);
  }
};

template <real_fp T>
Result<PipekMezeyInfo> pipek_mezey_localize(DeviceResources &res, const BasisSet &basis,
                                            const int32_t n_occ, const T *const s, const T *const c,
                                            T *const c_local, T *const u,
                                            const PipekMezeyOptions &options) {
  // Everything that can be refused is refused before the device is touched.
  const uint32_t n_atoms = basis.natoms();
  const auto n_ao = static_cast<int32_t>(basis.nao_sphe());
  if (n_atoms == 0) {
    return err_config("pipek_mezey_localize: the basis has no atoms; place it over a molecule");
  }
  if (s == nullptr || c == nullptr || c_local == nullptr) {
    return err_config("pipek_mezey_localize: s, c and c_local must not be null");
  }
  if (c_local == c) {
    return err_config("pipek_mezey_localize: c_local must not alias c (C U is a gemm)");
  }
  if (n_occ < 0 || n_occ > n_ao) {
    return err_config("pipek_mezey_localize: n_occ {} is outside 0..{}", n_occ, n_ao);
  }
  if (n_occ > device::kPmMaxOccupied) {
    return err_config("pipek_mezey_localize: n_occ {} is past the grid's limit {}", n_occ,
                      device::kPmMaxOccupied);
  }
  if (options.max_sweeps < 1) {
    return err_config("pipek_mezey_localize: max_sweeps {} is below 1", options.max_sweeps);
  }

  PipekMezeyInfo info{};
  if (n_occ == 0) {
    return info;
  }

  const wwr::wwrStream_t stream = res.stream();
  const auto exponent = static_cast<int32_t>(options.exponent);
  const auto n_atoms_i = static_cast<int32_t>(n_atoms);
  const std::vector<int32_t> atom_ao_offsets = pm_atom_ao_offsets(basis);

  // The schedule: an odd occupied count is padded to m with one dummy index,
  // which becomes the pivot, so each round one real orbital sits out.
  const int32_t m = (n_occ % 2 == 0) ? n_occ : n_occ + 1;
  const int32_t rounds = m - 1;
  std::vector<int32_t> schedule(static_cast<size_t>(rounds) * static_cast<size_t>(m));
  pm_fill_round_robin(m, schedule.data());

  PmScratch<T> scratch(res, static_cast<size_t>(n_ao), static_cast<size_t>(n_occ), n_atoms,
                       schedule.size());
  // Pageable sources: they live to the end of the call, and the first read of
  // L below synchronizes the stream before either is touched again.
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(scratch.schedule, schedule.data(),
                                         schedule.size() * sizeof(int32_t),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(scratch.atom_ao_offsets, atom_ao_offsets.data(),
                                         atom_ao_offsets.size() * sizeof(int32_t),
                                         wwr::wwrMemcpyHostToDevice, stream)));

  // W = S C, the populations, and U = I. The coefficients are not touched again
  // until C U: the sweeps run on the populations alone.
  FBF_TRY(gpu_status(wwr::gemm<T, int32_t>(res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N,
                                           n_ao, n_occ, n_ao, &kOne<T>, s, n_ao, c, n_ao, &kZero<T>,
                                           scratch.w, n_ao)));
  T *pops_cur = scratch.pops_a;
  T *pops_next = scratch.pops_b;
  T *u_cur = scratch.u_a;
  T *u_next = scratch.u_b;
  FBF_TRY(gpu_status(device::pm_populations(stream, n_atoms_i, n_occ, n_ao, c, scratch.w,
                                            scratch.atom_ao_offsets, pops_cur)));
  FBF_TRY(gpu_status(device::pm_identity(stream, n_occ, u_cur)));

  const auto read_functional = [&](const T *const pops) -> Result<double> {
    FBF_TRY(gpu_status(
        device::pm_functional(stream, exponent, n_atoms_i, n_occ, pops, scratch.functional)));
    T value{};
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(&value, scratch.functional, sizeof(T),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    return static_cast<double>(value);
  };

  info.initial_functional = FBF_TRY(read_functional(pops_cur));
  info.functional = info.initial_functional;

  // Fewer than two orbitals have no pair: U stays the identity.
  bool converged = n_occ < 2;
  double change = 0.0; // the last sweep's
  const auto gain_threshold = static_cast<T>(options.rotation_gain_threshold);
  for (int32_t sweep = 1; !converged && sweep <= options.max_sweeps; ++sweep) {
    // A round is three launches however many pairs it holds; the rounds queue
    // back to back, and only the convergence test below waits on the host.
    for (int32_t round = 0; round < rounds; ++round) {
      const int32_t *const pairs =
          scratch.schedule + (static_cast<size_t>(round) * static_cast<size_t>(m));
      FBF_TRY(gpu_status(device::pm_pair_angles(stream, exponent, n_atoms_i, n_occ, pops_cur, pairs,
                                                m / 2, gain_threshold, scratch.cos, scratch.sin,
                                                scratch.partner)));
      FBF_TRY(
          gpu_status(device::pm_rotate_populations(stream, n_atoms_i, n_occ, pops_cur, scratch.cos,
                                                   scratch.sin, scratch.partner, pops_next)));
      FBF_TRY(gpu_status(device::pm_rotate_columns(stream, n_occ, u_cur, scratch.cos, scratch.sin,
                                                   scratch.partner, u_next)));
      std::swap(pops_cur, pops_next);
      std::swap(u_cur, u_next);
    }
    const double previous = info.functional;
    info.functional = FBF_TRY(read_functional(pops_cur));
    info.sweeps = sweep;
    change = std::abs(info.functional - previous);
    converged = change < options.functional_tolerance;
  }

  // Written whether or not the sweeps converged, so a caller handed
  // MaxIterationsExceeded still has the last sweep's orbitals.
  FBF_TRY(gpu_status(wwr::gemm<T, int32_t>(res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N,
                                           n_ao, n_occ, n_occ, &kOne<T>, c, n_ao, u_cur, n_occ,
                                           &kZero<T>, c_local, n_ao)));
  if (u != nullptr) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(
        u, u_cur, static_cast<size_t>(n_occ) * static_cast<size_t>(n_occ) * sizeof(T),
        wwr::wwrMemcpyDeviceToDevice, stream)));
  }
  if (!converged) {
    return err_max_iterations_exceeded("pipek_mezey_localize: L still moved {:g} in sweep {}",
                                       change, options.max_sweeps);
  }
  return info;
}

template Result<PipekMezeyInfo> pipek_mezey_localize<float>(DeviceResources &, const BasisSet &,
                                                            int32_t, const float *, const float *,
                                                            float *, float *,
                                                            const PipekMezeyOptions &);
template Result<PipekMezeyInfo> pipek_mezey_localize<double>(DeviceResources &, const BasisSet &,
                                                             int32_t, const double *,
                                                             const double *, double *, double *,
                                                             const PipekMezeyOptions &);

} // namespace free_breakfast
