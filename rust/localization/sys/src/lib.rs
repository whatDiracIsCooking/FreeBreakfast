//! `libfb_localization_kernels.so`'s C ABI: the launch wrappers rust/ has
//! forked from `src/localization`, as raw FFI (`docs/gpu_layer.md`,
//! "Layering"), and the Pipek-Mezey tasks that are their one safe caller.
//!
//! The bindings are generated from `localization/kernels/fb_localization.h`
//! by `devtools/rust-bindings.py` and committed; `fb-device-check`
//! regenerates and diffs them. Every function is `unsafe`: the C side trusts
//! its device pointers and lengths, and three of its launches trust indices
//! held in device memory (the atoms' AO offsets, a round's orbital pairs and
//! the partners the pairs give), which no host check can read.
//!
//! | C++ (`src/localization/pipek_mezey`) | Here |
//! |---|---|
//! | `device::populations(stream, n_atoms, n_occ, ld, c, w, atom_ao_offsets, pops)` | [`fb_pm_populations`], counts below 1 and null pointers refused |
//! | `device::pair_angles(stream, exponent, …)`, any exponent but 4 taken as 2 | [`fb_pm_pair_angles`], an exponent other than 2 or 4 refused |
//! | `device::rotate_populations`, `device::rotate_columns` | [`fb_pm_rotate_populations`], [`fb_pm_rotate_columns`] |
//! | `device::functional`, `device::identity` | [`fb_pm_functional`], [`fb_pm_identity`] |
//! | `device::kMaxOccupied` | [`FB_PM_MAX_OCCUPIED`] |
//! | `pm_fill_round_robin(m, pairs)`, `pm_schedule_size(n_occ)` | the `pm tables` task of [`add_pm_start`], which writes the same tournament straight into its pinned row |
//! | `pm_atom_ao_offsets(basis)` | the caller's: [`add_pm_start`] takes the offsets and checks them |
//! | `PmScratch`, carved from `res.device_workspace()` | the rows of `pm start` and `pm sweep`, laid out by the plan |
//! | the driver's populations, `U = I` and first `read_functional`, one stream sync | `pm start`, its `L` a copy into a pinned row a CPU node reads |
//! | one sweep's `n_occ - 1` rounds and `read_functional`, one stream sync | `pm sweep` ([`add_pm_sweep`]), the same, in a loop body |
//! | `wwrError_t` | [`fb_launch_status`]: 0, a positive `wwr_status`, or a negative refusal ([`FB_LAUNCH_INVALID_ARGUMENT`]) |
//! | nothing: the C++ tree compiles its kernels into the binary | [`check_library`], the library's [`fb_localization_abi_version`] and [`fb_localization_backend`] against the bindings' |
//! | `localization/ibo`'s IAO populations and the rest of `src/localization` | not forked yet |
//!
//! **The safe layer is two builders**, since the indices the kernels trust
//! must come from where this crate wrote them (`docs/gpu_layer.md`, "The
//! safe wrapper"), as `fb-integrals-sys`'s metric forms its pairs itself:
//!
//! - [`add_pm_start`], before a sweep loop: the atoms' AO offsets, checked
//!   on the host, and the round-robin schedule, formed here, staged and
//!   uploaded into device rows only this crate names; then `pm start`, the
//!   population tensor, `U = I` and the functional of the orbitals as
//!   given. It returns a [`PmRows`], whose index rows are private.
//! - [`add_pm_sweep`], in the loop's body: one sweep of every round over a
//!   [`PmRows`], and its functional.
//!
//! **Why a launch may trust its indices.** The offsets row is the upload of
//! a pinned row `pm tables` wrote from offsets [`add_pm_start`] checked
//! (non-decreasing, the last within `n_ao`, the leading dimension of C and
//! of `W = S C`), and the schedule row the upload of one it filled with a
//! round-robin tournament over `m` (`n_occ`, or one more when odd): every
//! index below `m`, `n_occ` only as the padding the kernel sits out, and
//! each of the `n_occ` orbitals in exactly one pair of every round. So the
//! pair kernel writes a cosine, a sine and an in-range partner for every
//! orbital each round, into rows of `pm sweep`'s own, which the rotations
//! read next on the same stream. No other task can name the index rows: a
//! [`PmRows`] holds them in private fields.

#![allow(non_camel_case_types, reason = "the C header's names, kept as written")]

#[rustfmt::skip]
mod bindings;

pub use bindings::*;

use contracts::debug_requires;
use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_d2p};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::blas::gemm;
use fb_core_wwr::common::wwr_status_from_code;
use fb_core_wwr::copy;
use fb_core_wwr::device_slice::DeviceSlice;
use fb_core_wwr::gpu_status_at;
use fb_core_wwr::handle::DeviceHandleStream;
use fb_kernel_abi::Backend;
use wwr_sys::wwr_blas_operation::WWR_BLAS_OP_N as OP_N;

/// Checks that the `libfb_localization_kernels.so` this process loaded is
/// the one these bindings describe: its [`fb_localization_abi_version`]
/// against [`FB_LOCALIZATION_ABI_VERSION`], and its
/// [`fb_localization_backend`] against the backend of the CMake build this
/// crate linked (`docs/gpu_layer.md`, "What the port changes"). Call it
/// once, before the first launch. Needs no device. The comparison is
/// [`fb_kernel_abi::check`], tested where no library is built.
///
/// # Errors
///
/// A mismatch, with a message naming the found and the expected values.
pub fn check_library() -> Result<(), String> {
    // SAFETY: a host function with no arguments, which calls no runtime.
    let version = unsafe { fb_localization_abi_version() };
    // SAFETY: as above.
    let backend = unsafe { fb_localization_backend() };
    let built_for = const { Backend::named(env!("FB_KERNELS_BACKEND")) };
    fb_kernel_abi::check(
        "localization",
        (version, backend),
        (FB_LOCALIZATION_ABI_VERSION, built_for),
    )
}

/// A launch status as a [`Status`] naming the wrapper: a launch error by
/// its `wwr_status` name, at the caller.
#[track_caller]
fn launched(what: &str, status: fb_launch_status) -> Status {
    fb_kernel_abi::launch_status(what, status, |code, at| {
        Some(gpu_status_at(wwr_status_from_code(code)?, at))
    })
}

/// The schedule's index count `m`: `n_occ`, padded to even with one dummy
/// index, `n_occ` itself, which becomes the pivot.
fn padded(n_occ: usize) -> usize {
    n_occ + n_occ % 2
}

/// Fill `pairs`, `(m - 1) m` entries, with a round-robin ("circle method")
/// tournament over `m` indices, `m` even and at least 2: `m - 1` rounds of
/// `m / 2` disjoint pairs, round `r` at `r m` as consecutive (first,
/// second) entries, index `m - 1` the fixed pivot and the rest rotating
/// past it. The C++ `pm_fill_round_robin`, the same entries in the same
/// order.
fn fill_round_robin(m: usize, pairs: &mut [i32]) {
    let pivot = m - 1;
    for r in 0..pivot {
        let round = &mut pairs[r * m..(r + 1) * m];
        // Pair 0 draws the pivot against the round number; the rest fan out
        // around it, one step further apart per pair.
        round[0] = pivot as i32;
        round[1] = r as i32;
        for i in 1..m / 2 {
            round[2 * i] = ((r + i) % pivot) as i32;
            round[2 * i + 1] = ((r + pivot - i) % pivot) as i32;
        }
    }
}

/// The rows a Pipek-Mezey sweep loop works on, as [`add_pm_start`] returned
/// them: the population tensor and `U`, which [`add_pm_sweep`] carries from
/// sweep to sweep, `L` of the orbitals as given, and the schedule the pair
/// kernel trusts, which only this crate names.
#[derive(Clone, Copy, Debug)]
pub struct PmRows {
    tensor: RowId<Device, f64>,
    u: RowId<Device, f64>,
    initial: RowId<Pinned, f64>,
    schedule: RowId<Device, i32>,
    n_atoms: usize,
    n_occ: usize,
    exponent: i32,
}

impl PmRows {
    /// The accumulated rotation `U`, `n_occ²` doubles, column-major: the
    /// identity after `pm start`, and every sweep's rotation after it, so
    /// the localized orbitals are `C U`.
    pub fn u(&self) -> RowId<Device, f64> {
        self.u
    }

    /// `L` of the orbitals as given, one double in a pinned row.
    pub fn initial(&self) -> RowId<Pinned, f64> {
        self.initial
    }
}

/// Add the start of a Pipek-Mezey localization of `n_occ` orbitals over
/// `n_atoms = atom_ao_offsets.len() - 1` atoms to `dag` on GPU worker
/// `worker`, and return its [`PmRows`].
///
/// Atom `A` owns the AOs `atom_ao_offsets[A]..atom_ao_offsets[A + 1]` of the
/// `n_ao`. `s` is the `n_ao × n_ao` overlap and `c` holds the orbitals,
/// column-major at leading dimension `n_ao`, of which the first `n_occ`
/// columns are read, so the square C an SCF produces serves as it is.
/// `exponent` is the functional's power, `L = Σ_i Σ_A (Q^A_ii)^exponent`.
///
/// It adds **`pm tables`** (CPU: the offsets and the round-robin schedule
/// into pinned rows), their uploads (**`upload pm atom offsets`**,
/// **`upload pm schedule`**), and **`pm start`** (stream): `W = S C` by a
/// gemm into scratch, the population tensor `Q^A_ij` ([`fb_pm_populations`])
/// into its output row, `U = I` ([`fb_pm_identity`]) into its output row,
/// and `L` of the tensor ([`fb_pm_functional`]) copied into the pinned row
/// [`PmRows::initial`].
///
/// # Panics
/// If the offsets are fewer than two, decrease anywhere, or end past
/// `n_ao`; if `n_occ` is 0, past `n_ao` or past [`FB_PM_MAX_OCCUPIED`];
/// or if `n_ao` does not fit an `i32`: every index the kernels follow,
/// checked here, at planning. As [`Dag::add_gpu`], if a row is of another
/// worker's table; and when the plan runs, if `c` is shorter than `n_ao`
/// `n_occ` doubles.
#[debug_requires(exponent == 2 || exponent == 4, "the exponent is 2 or 4")]
#[allow(
    clippy::too_many_arguments,
    reason = "the basis's dimensions and the two input rows"
)]
#[track_caller]
pub fn add_pm_start(
    dag: &mut Dag<'_>,
    worker: usize,
    atom_ao_offsets: &[usize],
    n_ao: usize,
    n_occ: usize,
    exponent: i32,
    s: RowId<Device, f64>,
    c: RowId<Device, f64>,
) -> PmRows {
    assert!(
        i32::try_from(n_ao).is_ok(),
        "add_pm_start: n_ao = {n_ao} does not fit the kernels' i32"
    );
    assert!(
        atom_ao_offsets.len() >= 2
            && atom_ao_offsets.is_sorted()
            && atom_ao_offsets.last().is_some_and(|&end| end <= n_ao),
        "add_pm_start: the atom offsets {atom_ao_offsets:?} are not a non-decreasing run of \
         two or more within n_ao = {n_ao}"
    );
    assert!(
        (1..=n_ao.min(FB_PM_MAX_OCCUPIED as usize)).contains(&n_occ),
        "add_pm_start: n_occ = {n_occ} is outside 1..={}",
        n_ao.min(FB_PM_MAX_OCCUPIED as usize)
    );
    let n_atoms = atom_ao_offsets.len() - 1;
    let m = padded(n_occ);
    let staged = dag.add_cpu(
        "pm tables",
        worker,
        Tables {
            // In range: each is at most n_ao, which fits an i32.
            offsets: atom_ao_offsets.iter().map(|&o| o as i32).collect(),
            m,
        },
    );
    let offsets = dag.upload(worker, staged.offsets, "pm atom offsets", n_atoms + 1);
    let schedule = dag.upload(worker, staged.schedule, "pm schedule", (m - 1) * m);
    let start = dag.add_gpu(
        "pm start",
        worker,
        Start {
            s,
            c,
            offsets,
            n_ao,
            n_atoms,
            n_occ,
            exponent,
        },
    );
    PmRows {
        tensor: start.tensor,
        u: start.u,
        initial: start.functional,
        schedule,
        n_atoms,
        n_occ,
        exponent,
    }
}

/// Add one Pipek-Mezey sweep over `rows` to `dag` on GPU worker `worker`, to
/// sit in a loop's body ([`Dag::add_loop`]), and return its `L`: one double
/// in a pinned output row, for the loop's decision.
///
/// **`pm sweep`** (stream), the population tensor's and `U`'s one user in
/// the body, which it writes, so the loop carries both: each of the
/// schedule's `m - 1` rounds is three launches, the round's angles
/// ([`fb_pm_pair_angles`]; a pair that gains no more than
/// `gain_threshold` left alone), `Q ← Rᵀ Q R` ([`fb_pm_rotate_populations`])
/// and `U ← U R` ([`fb_pm_rotate_columns`]), each out of place between the
/// carried row and a scratch row of its own. The rounds are odd in number,
/// so the last lands in scratch and is copied back. Then `L` of the tensor
/// ([`fb_pm_functional`]), copied to the pinned row.
///
/// # Panics
/// As [`Dag::add_gpu`], if `rows` are of another worker's table.
#[debug_requires(!gain_threshold.is_nan(), "the gain threshold is a number")]
#[track_caller]
pub fn add_pm_sweep(
    dag: &mut Dag<'_>,
    worker: usize,
    rows: &PmRows,
    gain_threshold: f64,
) -> RowId<Pinned, f64> {
    dag.add_gpu(
        "pm sweep",
        worker,
        Sweep {
            rows: *rows,
            gain_threshold,
        },
    )
    .functional
}

/// The pinned rows `pm tables` writes.
#[derive(Clone, Copy)]
struct TableRows {
    offsets: RowId<Pinned, i32>,
    schedule: RowId<Pinned, i32>,
}

/// CPU: the atoms' AO offsets and the round-robin schedule over `m`, each
/// written straight into its pinned row.
struct Tables {
    offsets: Vec<i32>,
    m: usize,
}

impl CpuTask for Tables {
    type Rows = TableRows;

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> TableRows {
        TableRows {
            offsets: host.output("pm atom offsets", self.offsets.len()),
            schedule: host.output("pm schedule", (self.m - 1) * self.m),
        }
    }

    fn run(&self, rows: &TableRows, host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        host.take_mut(rows.offsets).copy_from_slice(&self.offsets);
        fill_round_robin(self.m, host.take_mut(rows.schedule));
        Ok(())
    }
}

/// GPU: `W = S C`, the population tensor, `U = I` and `L`.
struct Start {
    s: RowId<Device, f64>,
    c: RowId<Device, f64>,
    offsets: RowId<Device, i32>,
    n_ao: usize,
    n_atoms: usize,
    n_occ: usize,
    exponent: i32,
}

#[derive(Clone, Copy)]
struct StartRows {
    w: RowId<Device, f64>,
    scalar: RowId<Device, f64>,
    tensor: RowId<Device, f64>,
    u: RowId<Device, f64>,
    functional: RowId<Pinned, f64>,
}

impl GpuTask for Start {
    type Rows = StartRows;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> StartRows {
        device.read(self.s);
        device.read(self.c);
        device.read(self.offsets);
        StartRows {
            w: device.scratch("S C", self.n_ao * self.n_occ),
            scalar: device.scratch("pm L device", 1),
            tensor: device.output("pm populations", self.n_atoms * self.n_occ * self.n_occ),
            u: device.output("pm U", self.n_occ * self.n_occ),
            functional: host.output("pm L before", 1),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("n_ao", self.n_ao),
            ("n_atoms", self.n_atoms),
            ("n_occ", self.n_occ),
        ]
    }

    fn run(
        &self,
        rows: &StartRows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        // In range, as every cast below: add_pm_start checked n_ao and
        // n_occ <= n_ao against an i32, and the atoms are at most n_ao + 1.
        let (n_ao, n_occ, n_atoms) = (self.n_ao as i32, self.n_occ as i32, self.n_atoms as i32);
        let (s, c, offsets) = (
            device.get(self.s),
            device.get(self.c),
            device.get(self.offsets),
        );
        assert!(
            c.len() >= self.n_ao * self.n_occ,
            "pm start: C holds {} doubles, short of {} x {}",
            c.len(),
            self.n_ao,
            self.n_occ
        );
        let mut w = device.take_mut(rows.w);
        gemm(
            worker.blas(),
            OP_N,
            OP_N,
            n_ao,
            n_occ,
            n_ao,
            1.0,
            &s,
            n_ao,
            &c,
            n_ao,
            0.0,
            &mut w,
            n_ao,
        )?;
        let (tensor, u, scalar) = (
            device.take_mut(rows.tensor),
            device.take_mut(rows.u),
            device.take_mut(rows.scalar),
        );
        // SAFETY: c holds the n_ao x n_occ block (checked above) and w is
        // that block's own scratch; offsets is the upload of the n_atoms + 1
        // offsets add_pm_start checked to be non-decreasing and within
        // n_ao, the leading dimension of both; tensor is n_atoms n_occ²
        // doubles of this task's output; the stream follows the copy that
        // wrote the offsets; the launch only enqueues.
        let status = unsafe {
            fb_pm_populations(
                worker.stream(),
                n_atoms,
                n_occ,
                n_ao,
                c.as_ptr(),
                w.as_ptr(),
                offsets.as_ptr(),
                tensor.as_mut_ptr(),
            )
        };
        launched("fb_pm_populations", status)?;
        // SAFETY: u is n_occ² doubles of this task's output.
        let status = unsafe { fb_pm_identity(worker.stream(), n_occ, u.as_mut_ptr()) };
        launched("fb_pm_identity", status)?;
        functional(worker, self.exponent, n_atoms, n_occ, &tensor, &scalar)?;
        memcpy_d2p(host, rows.functional, &scalar, worker)
    }
}

/// `L` of `tensor` into the one-double `scalar`, on `worker`'s stream.
fn functional(
    worker: &DeviceStream,
    exponent: i32,
    n_atoms: i32,
    n_occ: i32,
    tensor: &impl DeviceSlice<f64>,
    scalar: &fb_core_arena::DeviceMut<'_, f64>,
) -> Status {
    // INVARIANT: both rows are laid out by the task calling this, at these
    // dimensions.
    debug_assert!(
        tensor.len() == n_atoms as usize * (n_occ as usize).pow(2) && scalar.len() == 1,
        "the tensor and the scalar are the task's own rows"
    );
    // SAFETY: tensor is n_atoms n_occ² doubles and scalar one, rows of the
    // calling task; the launch only enqueues.
    let status = unsafe {
        fb_pm_functional(
            worker.stream(),
            exponent,
            n_atoms,
            n_occ,
            tensor.as_ptr(),
            scalar.as_mut_ptr(),
        )
    };
    launched("fb_pm_functional", status)
}

/// GPU: one sweep's rounds over the carried tensor and `U`, then `L`.
struct Sweep {
    rows: PmRows,
    gain_threshold: f64,
}

#[derive(Clone, Copy)]
struct SweepRows {
    tensor_pong: RowId<Device, f64>,
    u_pong: RowId<Device, f64>,
    cos: RowId<Device, f64>,
    sin: RowId<Device, f64>,
    partner: RowId<Device, i32>,
    scalar: RowId<Device, f64>,
    functional: RowId<Pinned, f64>,
}

impl GpuTask for Sweep {
    type Rows = SweepRows;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> SweepRows {
        let r = &self.rows;
        device.read(r.schedule);
        device.write(r.tensor);
        device.write(r.u);
        SweepRows {
            tensor_pong: device.scratch("pm populations pong", r.n_atoms * r.n_occ * r.n_occ),
            u_pong: device.scratch("pm U pong", r.n_occ * r.n_occ),
            cos: device.scratch("pm cos", r.n_occ),
            sin: device.scratch("pm sin", r.n_occ),
            partner: device.scratch("pm partner", r.n_occ),
            scalar: device.scratch("pm L device", 1),
            functional: host.output("pm L", 1),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n_atoms", self.rows.n_atoms), ("n_occ", self.rows.n_occ)]
    }

    fn uses_libraries(&self) -> bool {
        false
    }

    fn run(
        &self,
        rows: &SweepRows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let r = &self.rows;
        // In range: add_pm_start checked n_occ and n_ao, so the atoms too,
        // against an i32.
        let (n_atoms, n_occ) = (r.n_atoms as i32, r.n_occ as i32);
        let m = padded(r.n_occ);
        let schedule = device.get(r.schedule);
        let (mut tensor, mut u) = (device.take_mut(r.tensor), device.take_mut(r.u));
        let (tensor_pong, u_pong) = (
            device.take_mut(rows.tensor_pong),
            device.take_mut(rows.u_pong),
        );
        let (cos, sin, partner) = (
            device.take_mut(rows.cos),
            device.take_mut(rows.sin),
            device.take_mut(rows.partner),
        );
        let (mut q_in, mut q_out) = (tensor.as_mut_ptr(), tensor_pong.as_mut_ptr());
        let (mut u_in, mut u_out) = (u.as_mut_ptr(), u_pong.as_mut_ptr());
        // A round is three launches however many pairs it holds; the rounds
        // queue back to back on the stream.
        for round in 0..m - 1 {
            let pairs = schedule.as_ptr().wrapping_add(round * m);
            // SAFETY: the tensor and its pong are n_atoms n_occ² doubles,
            // cos, sin and partner n_occ each, all this task's rows; pairs is
            // round `round` of the schedule row, m entries inside its
            // (m - 1) m, a tournament `pm tables` wrote (the crate doc's
            // argument), so every orbital is in exactly one pair and every
            // index the kernel follows is below n_occ or is the padding it
            // sits out; the launch only enqueues.
            let status = unsafe {
                fb_pm_pair_angles(
                    worker.stream(),
                    r.exponent,
                    n_atoms,
                    n_occ,
                    q_in,
                    pairs,
                    (m / 2) as i32,
                    self.gain_threshold,
                    cos.as_mut_ptr(),
                    sin.as_mut_ptr(),
                    partner.as_mut_ptr(),
                )
            };
            launched("fb_pm_pair_angles", status)?;
            // SAFETY: the launch above wrote every orbital's partner, each
            // below n_occ, before this one on the stream; the two tensors
            // are distinct rows of n_atoms n_occ² doubles.
            let status = unsafe {
                fb_pm_rotate_populations(
                    worker.stream(),
                    n_atoms,
                    n_occ,
                    q_in,
                    cos.as_ptr(),
                    sin.as_ptr(),
                    partner.as_ptr(),
                    q_out,
                )
            };
            launched("fb_pm_rotate_populations", status)?;
            // SAFETY: as above, over the two n_occ² rows of U.
            let status = unsafe {
                fb_pm_rotate_columns(
                    worker.stream(),
                    n_occ,
                    u_in,
                    cos.as_ptr(),
                    sin.as_ptr(),
                    partner.as_ptr(),
                    u_out,
                )
            };
            launched("fb_pm_rotate_columns", status)?;
            std::mem::swap(&mut q_in, &mut q_out);
            std::mem::swap(&mut u_in, &mut u_out);
        }
        // INVARIANT: m is even, so the rounds are odd in number and the last
        // wrote the pongs.
        debug_assert!(
            q_in == tensor_pong.as_mut_ptr(),
            "the last round wrote the pong"
        );
        copy::d2d(&mut tensor, &tensor_pong, worker)?;
        copy::d2d(&mut u, &u_pong, worker)?;
        let scalar = device.take_mut(rows.scalar);
        functional(worker, r.exponent, n_atoms, n_occ, &tensor, &scalar)?;
        memcpy_d2p(host, rows.functional, &scalar, worker)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every round of the tournament over m = 2 to 12 pairs each index below
    /// m exactly once, and the m - 1 rounds together pair every two indices
    /// exactly once. What the kernels trust of the schedule: catches a pair
    /// repeated or an orbital left out of a round, which would leave its
    /// partner unwritten.
    #[test]
    fn each_round_pairs_every_index_once_and_a_sweep_every_pair() {
        for m in (2..=12).step_by(2) {
            let mut pairs = vec![-1; (m - 1) * m];
            fill_round_robin(m, &mut pairs);
            let mut met = vec![0; m * m];
            for round in pairs.chunks(m) {
                let mut seen = vec![0; m];
                for pair in round.chunks(2) {
                    let (p, q) = (pair[0] as usize, pair[1] as usize);
                    seen[p] += 1;
                    seen[q] += 1;
                    met[p.min(q) * m + p.max(q)] += 1;
                }
                assert!(seen.iter().all(|&n| n == 1), "m = {m}: {round:?}");
            }
            for p in 0..m {
                for q in p + 1..m {
                    assert_eq!(met[p * m + q], 1, "m = {m}: ({p}, {q})");
                }
            }
        }
    }

    /// The padding: an odd count's one dummy index is the pivot.
    #[test]
    fn an_odd_count_is_padded_by_one() {
        assert_eq!(padded(5), 6);
        assert_eq!(padded(1), 2);
        assert_eq!(padded(8), 8);
    }
}
