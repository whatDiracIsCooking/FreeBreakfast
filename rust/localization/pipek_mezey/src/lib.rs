//! Pipek-Mezey orbital localization on the GPU, as tasks of a [`Dag`]: the
//! occupied orbitals rotated, within their own span, to sit on as few atoms
//! as their Mulliken populations allow. The C++
//! `src/localization/pipek_mezey` is the reference design, and its README
//! the algorithm at length.
//!
//! **What it maximizes.** Over orthogonal `U` of the occupied space,
//!
//! ```text
//! L_p(U) = Σ_i Σ_A (Q^A_ii)^p
//! Q^A_ij = 1/2 Σ_{μ ∈ A} (C_μi W_μj + C_μj W_μi),   W = S C
//! ```
//!
//! `Q^A` is atom A's symmetrized Mulliken population matrix, and because
//! `Σ_A Q^A_ij = (Cᵀ S C)_ij = δ_ij` for S-orthonormal orbitals, raising an
//! orbital's populations to a power and summing rewards concentrating it on
//! few atoms. `p = 2` is Pipek and Mezey's (J. Chem. Phys. 90, 4916 (1989)),
//! with an exact rotation angle per pair; `p = 4` is Knizia's IBO exponent
//! (J. Chem. Theory Comput. 9, 4834 (2013), eq. 4, the angle of Appendix
//! D), which breaks `p = 2`'s exact zero mode on a symmetric π system, its
//! angle approximate and a rotation kept only where it raises `L_4`
//! exactly. Over Mulliken populations, not IAO ones: this is Pipek-Mezey
//! with Knizia's exponent, not IBO.
//!
//! **An atom owns its shells' AOs, as one range**: a basis placed over a
//! molecule holds each atom's shells together, in atom order
//! ([`BasisSet::atom_shell_offset`]), and the spherical AOs follow the
//! shells, so no centre is matched to a nucleus, and a basis with no atoms
//! is refused.
//!
//! **The sweep runs on the populations**, which transform as `Uᵀ Q^A U`,
//! so the coefficients are touched twice, `W = S C` at the start and `C U`
//! at the end, and **a round of disjoint pairs rotates at once**: a
//! round-robin tournament splits a sweep into `m - 1` rounds of `m / 2`
//! disjoint pairs (`m` the occupied count padded to even), and disjoint
//! rotations commute into one block-diagonal `R`, three launches a round.
//! A pair's rotation changes only its own two diagonals, from its own
//! three elements, so a round gains the sum of its pairs' gains, each
//! non-negative, and `L` rises round by round as under sequential Jacobi.
//!
//! [`add_pm_localize`] adds, after the kernel library's load check at
//! planning:
//!
//! - **Before the loop**, [`add_pm_start`]'s `pm tables`, its two uploads
//!   and `pm start` (the populations, `U = I` and `L` of the orbitals as
//!   given, in a pinned row), then **`pm before`** (CPU): the decision's
//!   carried heap row `[L before, L, sweeps]`, `[L, L, 0]` here.
//! - **The body**, one sweep: [`add_pm_sweep`]'s `pm sweep`, every round
//!   and `L` after them, in a pinned row. **The decision**, `pm converged`
//!   (CPU): `L` and the sweep count written into the heap row; converged
//!   when the sweep moved `L` by less than the tolerance. Fewer than two
//!   orbitals have no pair, and the plan has no loop.
//! - **After the loop**, **`pm rotate`** (stream): `C_local = C U` by a
//!   gemm.
//!
//! | C++ | Here |
//! |---|---|
//! | `pm_localize(res, basis, n_occ, s, c, c_local, u, options)` | [`add_pm_localize`]`(dag, worker, options, basis, n_occ, s, c)`, returning `C_local` and `U` as output rows |
//! | `PmOptions`, `PmExponent` | [`PmOptions`], [`PmExponent`] |
//! | `Info`: sweeps, the functional before and after | the heap row `[L before, L after, sweeps]` |
//! | `pm_atom_ao_offsets(basis)` | the offsets, formed at planning and handed to [`add_pm_start`], which checks them |
//! | the round-robin schedule built on the host and uploaded, the scratch carved from `res.device_workspace()` | [`add_pm_start`]'s `pm tables` and uploads, and the tasks' rows, laid out by the plan |
//! | the sweep loop, one stream sync a sweep for `L` | a loop of the plan ([`Dag::add_loop`]): [`add_pm_sweep`] its body, `pm converged` its decision |
//! | `MaxIterationsExceeded` after `max_sweeps`, `c_local` and `u` holding the last sweep | `MaxIterationsExceeded` from the decision at `max_sweeps`; the run fails, so `pm rotate` does not run and `C_local` is not written |
//! | `InvalidConfig` for a basis with no atoms, `n_occ` outside `0..n_ao` or past 65535, `max_sweeps` below 1 | the same, at planning, adding nothing; `n_occ` 0 refused too, a plan of nothing |
//! | `InvalidConfig` for a null pointer or `c_local` aliasing `c` | nothing: rows, which the planner keeps apart |
//! | `pm_localize_footprint(basis, n_occ)` | nothing: the layouts declare every row, and the arena's size is the plan's golden |
//! | `localization/ibo`, Loewdin populations, the virtual space | not ported |

#![forbid(unsafe_code)]

use std::ops::ControlFlow;

use fb_basis::BasisSet;
use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{CpuTask, Dag, Decision, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::blas::gemm;
use fb_localization_sys::{FB_PM_MAX_OCCUPIED, add_pm_start, add_pm_sweep};
use wwr_sys::wwr_blas_operation::WWR_BLAS_OP_N as OP_N;

/// The power `p` of `L_p = Σ_i Σ_A (Q^A_ii)^p`.
///
/// Two is Pipek and Mezey's functional, with an exact rotation angle per
/// pair. Its rotation Hessian can have an exact zero mode for a symmetric
/// π system (benzene's six equivalent π orbitals), leaving a continuum of
/// equally localized answers. Four is Knizia's IBO exponent, which
/// concentrates the populations harder and breaks that degeneracy; its
/// angle is approximate, but a rotation is kept only where it raises `L_4`
/// exactly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PmExponent {
    /// `L_2`, Pipek and Mezey (1989).
    #[default]
    Two = 2,
    /// `L_4`, Knizia (2013).
    Four = 4,
}

/// The options of one localization, each at the C++ default
/// ([`PmOptions::default`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PmOptions {
    /// Sweeps before the run fails with `MaxIterationsExceeded`; each visits
    /// every orbital pair once. Default 64; at least 1.
    pub max_sweeps: usize,
    /// Converged when a whole sweep moves `L` by less than this. Default
    /// 1e-10.
    pub functional_tolerance: f64,
    /// A pair whose gain in `L` does not exceed this is left alone, rather
    /// than spending a rotation, and its rounding, on nothing. Default
    /// 1e-14.
    pub rotation_gain_threshold: f64,
    /// The functional's power. Default [`PmExponent::Two`].
    pub exponent: PmExponent,
}

impl Default for PmOptions {
    fn default() -> PmOptions {
        PmOptions {
            max_sweeps: 64,
            functional_tolerance: 1e-10,
            rotation_gain_threshold: 1e-14,
            exponent: PmExponent::Two,
        }
    }
}

/// Add the Pipek-Mezey localization of the first `n_occ` orbitals of `c`
/// over the spherical AOs of `basis` to `dag` on GPU worker `worker`, and
/// return `(info, [c_local, u])`: a heap row of three doubles, `[L before,
/// L after, sweeps]` (`sweeps` a whole number, 0 for one orbital), and
/// device output rows of `C_local = C U`, `n_ao × n_occ`, and the
/// orthogonal `U`, `n_occ × n_occ`, both column-major.
///
/// `s` is the overlap, `n_ao²` doubles, and `c` the orbitals, column-major
/// at leading dimension `n_ao`: its first `n_occ` columns are read, so the
/// square C of an SCF serves as it is. They are expected S-orthonormal,
/// `Cᵀ S C = I`, which is what makes an orbital's populations sum to one;
/// that is the caller's to keep, and not checked.
///
/// The tasks are the crate doc's. The loop's bound is
/// `options.max_sweeps`; each sweep is logged at `debug`, and the result at
/// `info`.
///
/// # Errors
/// At planning, before anything is added: `GpuRuntime` with the check's
/// message if the loaded `libfb_localization_kernels.so` is not the one
/// the bindings describe; `InvalidConfig` for a basis with no atoms, an
/// `n_occ` of 0, past `n_ao` or past the kernels' 65535, or a
/// `max_sweeps` of 0. At run time, the tasks' errors, and
/// `MaxIterationsExceeded` when `max_sweeps` sweeps have not converged.
///
/// # Panics
/// As [`Dag::add_gpu`], if a row is of another worker's table; and when the
/// plan runs, if `c` is shorter than `n_ao n_occ` doubles.
#[allow(
    clippy::type_complexity,
    reason = "the products as rows, in the crate tuple style of add_scf"
)]
#[track_caller]
pub fn add_pm_localize(
    dag: &mut Dag<'_>,
    worker: usize,
    options: &PmOptions,
    basis: &BasisSet,
    n_occ: usize,
    s: RowId<Device, f64>,
    c: RowId<Device, f64>,
) -> Result<(RowId<Heap, f64>, [RowId<Device, f64>; 2])> {
    // Not `map_err(Error::gpu)`: through a function pointer the error's
    // origin would be core's, not this call's caller.
    if let Err(e) = fb_localization_sys::check_library() {
        return Err(Error::gpu(e));
    }
    let n_ao = basis.nao_sphe() as usize;
    let n_atoms = basis.natoms();
    if n_atoms == 0 {
        return Err(Error::config(
            "add_pm_localize: the basis has no atoms; place it over a molecule",
        ));
    }
    let limit = n_ao.min(FB_PM_MAX_OCCUPIED as usize);
    if !(1..=limit).contains(&n_occ) {
        return Err(Error::config(format!(
            "add_pm_localize: n_occ {n_occ} is outside 1..={limit} (n_ao {n_ao}, the kernels' \
             {FB_PM_MAX_OCCUPIED})"
        )));
    }
    if options.max_sweeps < 1 {
        return Err(Error::config("add_pm_localize: max_sweeps is 0, below 1"));
    }

    // Atom A's spherical AOs start at its first shell's; an atom with no
    // shells, and the end, where the next AO would.
    let n_shells = basis.nshells();
    let offsets: Vec<usize> = (0..=n_atoms)
        .map(|a| {
            let shell = if a < n_atoms {
                basis.atom_shell_offset(a)
            } else {
                n_shells
            };
            if shell < n_shells {
                basis.ao_offset_sphe(shell) as usize
            } else {
                n_ao
            }
        })
        .collect();
    let exponent = options.exponent as i32;
    let rows = add_pm_start(dag, worker, &offsets, n_ao, n_occ, exponent, s, c);
    let info = dag.add_cpu(
        "pm before",
        worker,
        Before {
            initial: rows.initial(),
        },
    );
    if n_occ >= 2 {
        let body = dag.len();
        let functional = add_pm_sweep(dag, worker, &rows, options.rotation_gain_threshold);
        dag.add_loop(
            "pm converged",
            worker,
            body,
            options.max_sweeps,
            Converged {
                functional,
                info,
                tolerance: options.functional_tolerance,
                max_sweeps: options.max_sweeps,
            },
        );
    }
    let c_local = dag.add_gpu(
        "pm rotate",
        worker,
        Rotate {
            c,
            u: rows.u(),
            n_ao,
            n_occ,
        },
    );
    Ok((info, [c_local, rows.u()]))
}

/// CPU, before the loop: the decision's carried `[L before, L, sweeps]`,
/// from `L` of the orbitals as given, a heap row: host state no copy engine
/// meets.
struct Before {
    initial: RowId<Pinned, f64>,
}

impl CpuTask for Before {
    type Rows = RowId<Heap, f64>;

    fn layout(&self, host: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) -> Self::Rows {
        host.read(self.initial);
        heap.output("pm info", 3)
    }

    fn run(&self, row: &Self::Rows, host: &Bufs<'_, Pinned>, heap: &Bufs<'_, Heap>) -> Status {
        // An assembly, not a copy: L before and the L the first sweep is
        // measured from are both the orbitals' as given, and no sweep has
        // run.
        let l = host.get(self.initial)[0];
        let info = heap.take_mut(*row);
        info[0] = l;
        info[1] = l;
        info[2] = 0.0;
        Ok(())
    }
}

/// The loop's decision: the sweep recorded, logged and tested.
struct Converged {
    functional: RowId<Pinned, f64>,
    info: RowId<Heap, f64>,
    tolerance: f64,
    max_sweeps: usize,
}

impl Decision for Converged {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) {
        host.read(self.functional);
        heap.write(self.info);
    }

    fn decide(
        &self,
        _: &(),
        host: &Bufs<'_, Pinned>,
        heap: &Bufs<'_, Heap>,
        iteration: usize,
    ) -> Result<ControlFlow<()>> {
        let l = host.get(self.functional)[0];
        let info = heap.take_mut(self.info);
        let change = (l - info[1]).abs();
        let sweeps = iteration + 1;
        info[1] = l;
        // A count well under 2^53, so exact as a double.
        info[2] = sweeps as f64;
        log::debug!("pm: sweep {sweeps:3}  L = {l:.12}  dL = {change:.3e}");
        if change < self.tolerance {
            log::info!(
                "pm: converged in {sweeps} sweeps, L = {l:.12} (from {:.12})",
                info[0]
            );
            return Ok(ControlFlow::Break(()));
        }
        if sweeps == self.max_sweeps {
            return Err(Error::max_iterations_exceeded(format!(
                "pm_localize: L still moved {change:e} in sweep {sweeps}"
            )));
        }
        Ok(ControlFlow::Continue(()))
    }
}

/// GPU, after the loop: `C_local = C U`.
struct Rotate {
    c: RowId<Device, f64>,
    u: RowId<Device, f64>,
    n_ao: usize,
    n_occ: usize,
}

impl GpuTask for Rotate {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.c);
        device.read(self.u);
        device.output("C local", self.n_ao * self.n_occ)
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n_ao", self.n_ao), ("n_occ", self.n_occ)]
    }

    fn run(
        &self,
        c_local: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        // In range: add_pm_start checked n_ao, and n_occ <= n_ao, against an
        // i32.
        let (n_ao, n_occ) = (self.n_ao as i32, self.n_occ as i32);
        let (c, u) = (device.get(self.c), device.get(self.u));
        let mut c_local = device.take_mut(*c_local);
        gemm(
            worker.blas(),
            OP_N,
            OP_N,
            n_ao,
            n_occ,
            n_occ,
            1.0,
            &c,
            n_ao,
            &u,
            n_occ,
            0.0,
            &mut c_local,
            n_ao,
        )
    }
}
