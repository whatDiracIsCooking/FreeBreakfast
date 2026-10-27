/// @file
/// @brief pm_localize: the sweep driver.

module;

// The launchers, defined in pipek_mezey_kernels.cu, declared in the GMF so they
// keep external linkage (see the header).
#include "pipek_mezey_bridge.h"

module free_breakfast.localization.pipek_mezey;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // wwrMemcpyAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.recipes.blas; // gemm

#include "diagnostics.h"

namespace free_breakfast::localization {

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

// Fill a round-robin ("circle method") tournament over m indices.
//
// A sweep visits each of the m(m-1)/2 orbital pairs once, and keeps the pairs
// of each round disjoint, so a whole round applies as one block-diagonal
// rotation. That is a round-robin tournament: m-1 rounds of m/2 disjoint
// pairs, index m-1 the fixed pivot and the rest rotating past it.
//
// m is even and at least 2. An odd occupied count is padded with the dummy
// index n_occ, which is then the pivot, so in each round exactly one real
// orbital draws it and sits the round out.
//
// pairs holds (m - 1) * m entries: round r starts at pairs + r * m and is
// m/2 consecutive (first, second) pairs. Built once on
// the host and uploaded.
void pm_fill_round_robin(const int32_t m, int32_t *const pairs) {
  const int32_t pivot = m - 1;
  for (int32_t r = 0; r < pivot; ++r) {
    int32_t *slot = pairs + (static_cast<size_t>(r) * static_cast<size_t>(m));
    // Pair 0 draws the pivot against the round number; the rest fan out
    // around it, one step further apart per pair.
    slot[0] = pivot;
    slot[1] = r;
    for (int32_t i = 1; i < m / 2; ++i) {
      slot += 2;
      slot[0] = (r + i) % pivot;
      slot[1] = (((r - i) % pivot) + pivot) % pivot;
    }
  }
}

// The call's device scratch, one region per buffer, carved from one
// workspace request. Nothing here calls another workspace user while these
// are live.
struct PmScratch {
  double *w;                // S C, n_ao x n_occ
  double *pops_a;           // the population tensor, ping
  double *pops_b;           // and pong
  double *u_a;              // the accumulated U, ping
  double *u_b;              // and pong
  double *cos;              // a round's per-orbital rotation
  double *sin;              //
  double *functional;       // L, one element
  int32_t *partner;         // a round's per-orbital partner
  int32_t *schedule;        // every round's pairs
  int32_t *atom_ao_offsets; // n_atoms + 1

  PmScratch(DeviceResources &res, const size_t n_ao, const size_t n_occ, const size_t n_atoms,
            const size_t schedule_size) {
    const size_t pops = n_atoms * n_occ * n_occ;
    std::tie(w, pops_a, pops_b, u_a, u_b, cos, sin, functional, partner, schedule,
             atom_ao_offsets) = res.device_regions<double, double, double, double, double, double,
                                                   double, double, int32_t, int32_t, int32_t>(
        n_ao * n_occ, pops, pops, n_occ * n_occ, n_occ * n_occ, n_occ, n_occ, size_t{1}, n_occ,
        schedule_size, n_atoms + 1);
  }
};

Result<Info> pm_localize(DeviceResources &res, const BasisSet &basis, const int32_t n_occ,
                         const double *const s, const double *const c, double *const c_local,
                         double *const u, const PmOptions &options) {
  // Everything that can be refused is refused before the device is touched.
  const uint32_t n_atoms = basis.natoms();
  const auto n_ao = static_cast<int32_t>(basis.nao_sphe());
  if (n_atoms == 0) {
    return err_config(
        "localization::pm_localize: the basis has no atoms; place it over a molecule");
  }
  if (s == nullptr || c == nullptr || c_local == nullptr) {
    return err_config("localization::pm_localize: s, c and c_local must not be null");
  }
  if (c_local == c) {
    return err_config("localization::pm_localize: c_local must not alias c (C U is a gemm)");
  }
  if (n_occ < 0 || n_occ > n_ao) {
    return err_config("localization::pm_localize: n_occ {} is outside 0..{}", n_occ, n_ao);
  }
  if (n_occ > device::kMaxOccupied) {
    return err_config("localization::pm_localize: n_occ {} is past the grid's limit {}", n_occ,
                      device::kMaxOccupied);
  }
  if (options.max_sweeps < 1) {
    return err_config("localization::pm_localize: max_sweeps {} is below 1", options.max_sweeps);
  }

  Info info{};
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

  PmScratch scratch(res, static_cast<size_t>(n_ao), static_cast<size_t>(n_occ), n_atoms,
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
  FBF_TRY(gpu_status(
      gemm(res.blas(), n_ao, n_occ, n_ao, &kOne<double>, s, c, &kZero<double>, scratch.w)));
  double *pops_cur = scratch.pops_a;
  double *pops_next = scratch.pops_b;
  double *u_cur = scratch.u_a;
  double *u_next = scratch.u_b;
  FBF_TRY(gpu_status(device::populations(stream, n_atoms_i, n_occ, n_ao, c, scratch.w,
                                         scratch.atom_ao_offsets, pops_cur)));
  FBF_TRY(gpu_status(device::identity(stream, n_occ, u_cur)));

  const auto read_functional = [&](const double *const pops) -> Result<double> {
    FBF_TRY(gpu_status(
        device::functional(stream, exponent, n_atoms_i, n_occ, pops, scratch.functional)));
    double value{};
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(&value, scratch.functional, sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    return value;
  };

  info.initial_functional = FBF_TRY(read_functional(pops_cur));
  info.functional = info.initial_functional;

  // Fewer than two orbitals have no pair: U stays the identity.
  bool converged = n_occ < 2;
  double change = 0.0; // the last sweep's
  const double gain_threshold = options.rotation_gain_threshold;
  for (int32_t sweep = 1; !converged && sweep <= options.max_sweeps; ++sweep) {
    // A round is three launches however many pairs it holds; the rounds queue
    // back to back, and only the convergence test below waits on the host.
    for (int32_t round = 0; round < rounds; ++round) {
      const int32_t *const pairs =
          scratch.schedule + (static_cast<size_t>(round) * static_cast<size_t>(m));
      FBF_TRY(gpu_status(device::pair_angles(stream, exponent, n_atoms_i, n_occ, pops_cur, pairs,
                                             m / 2, gain_threshold, scratch.cos, scratch.sin,
                                             scratch.partner)));
      FBF_TRY(gpu_status(device::rotate_populations(stream, n_atoms_i, n_occ, pops_cur, scratch.cos,
                                                    scratch.sin, scratch.partner, pops_next)));
      FBF_TRY(gpu_status(device::rotate_columns(stream, n_occ, u_cur, scratch.cos, scratch.sin,
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
  FBF_TRY(gpu_status(
      gemm(res.blas(), n_ao, n_occ, n_occ, &kOne<double>, c, u_cur, &kZero<double>, c_local)));
  if (u != nullptr) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(
        u, u_cur, static_cast<size_t>(n_occ) * static_cast<size_t>(n_occ) * sizeof(double),
        wwr::wwrMemcpyDeviceToDevice, stream)));
  }
  if (!converged) {
    return err_max_iterations_exceeded("localization::pm_localize: L still moved {:g} in sweep {}",
                                       change, options.max_sweeps);
  }
  return info;
}

} // namespace free_breakfast::localization
