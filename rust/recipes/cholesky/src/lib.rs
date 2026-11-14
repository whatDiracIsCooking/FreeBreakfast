//! The Cholesky factorization `A = L Lᵀ` of a symmetric positive definite
//! matrix on the GPU, as tasks of a [`Dag`]: the tree's own blocked
//! factorization, never the dense solver's `potrf`. The C++
//! `src/recipes/cholesky` is the reference design, and its README has why
//! not `potrf`: on ROCm 7.2 rocSOLVER's `potf2_kernel_small` reads the pivot
//! with no barrier before thread 0 overwrites it, so beside other work on
//! the card the DF metric came out "not positive definite" at a pivot that
//! changed from run to run, or its factor slightly wrong (#404, #379). One
//! code path for both backends, so CUDA runs it too.
//!
//! [`add_cholesky`] adds, for one symmetric matrix row A:
//!
//! - **`cholesky`**, on the worker's stream: A copied into the output row
//!   L, the flag (a device scratch row) zeroed, then right-looking and
//!   blocked, [`CHOLESKY_BLOCK`] columns at a time over L: the diagonal
//!   tile by `libfb_recipes_kernels.so`'s one-block kernel
//!   ([`cholesky_diagonal`], which also leaves the factored tile packed in a
//!   scratch row), the panel below it `A21 ← A21 L11⁻ᵀ` ([`trsm`]), and the
//!   trailing matrix `A22 ← A22 - L21 L21ᵀ` ([`syrk`], lower). The flag is
//!   copied into a pinned row as the task's last work.
//! - **`cholesky check`**, on the CPU: the flag read from that pinned row,
//!   after the executor waited for `cholesky`. Non-zero fails the run with
//!   `Numerical`, naming the 1-based leading minor that is not positive. On
//!   success it writes the pinned row it returns beside L, which a task that
//!   must not run on a failed factor reads.
//!
//! Two tasks because the C++ call synchronizes to read the flag, and a task
//! has no host wait inside it: the wait is the edge from `cholesky` to the
//! check.
//!
//! **The blocks of one row.** The panel's trsm reads L11 and writes the
//! panel below it, in the same columns of L, and the syrk reads that panel
//! and writes the trailing matrix. A safe library call takes each operand
//! as a [`DeviceSlice`](fb_core_wwr::device_slice::DeviceSlice) it holds
//! alone, so L is cut into disjoint parts by
//! [`split_at_mut`]: the panel ends at element `(n - 1, j + jb - 1)` and the
//! trailing matrix starts at `(j + jb, j + jb)`, so a cut there parts them.
//! L11 and the panel interleave column by column, so the trsm reads the
//! packed copy of L11 the diagonal kernel wrote instead.
//!
//! **After a minor that is not positive** the later tiles leave themselves
//! alone (the kernel reads the flag), so the flag names the first failing
//! minor, as `potrf`'s does, and the panels carry on over a matrix that no
//! longer matters. L is then unspecified, and the run's result is the
//! check's error.
//!
//! | C++ | Here |
//! |---|---|
//! | `cholesky(res, n, a)`, A overwritten by L | [`add_cholesky`]'s `cholesky` task, A copied into its own output row L first, since a task writes only its own rows |
//! | `res.info()`, zeroed by `wwrMemsetAsync`, and `res.fetch_info()`, one stream sync | the `info` device scratch row, zeroed by [`memset::zero`], its pinned copy, and the `cholesky check` task the executor waits for `cholesky` before |
//! | the returned flag, `0` or the failing minor | `cholesky check`'s `Numerical` error naming the minor; the pinned row it writes on success, which a later task reads to run only on a good factor |
//! | `device::kCholeskyBlock` | [`CHOLESKY_BLOCK`], the kernel library's `FB_CHOLESKY_TILE` |
//! | `device::cholesky_diagonal_block(stream, jb, j, diagonal, ld, info)` | [`cholesky_diagonal`], over a part of L, the tile also written packed into the `tile` scratch row |
//! | `wwr::trsm<double, int32_t>(…, diagonal, n, panel, n)` | [`trsm`] against the packed tile, over the part of L that starts at the panel |
//! | `wwr::syrk<double, int32_t>(…, panel, n, …, panel + jb ld, n)` | [`syrk`], the panel and the trailing matrix the two sides of one cut of L |
//! | `res.blas()` | the stream worker's handle, [`DeviceStream::blas`], bound to its stream |
//! | `ScopedDeviceIndex` for the call | nothing: a stream worker's tasks run on its device |
//! | `n < 0`: `InvalidConfig`; `n == 0`: an empty success | refused at planning: `n` is at least 1 |
//! | nothing: the C++ tree compiles its kernels into the binary | [`fb_recipes_sys::check_library`] first, at planning |
//!
//! [`split_at_mut`]: fb_core_wwr::device_slice::split_at_mut
//! [`trsm`]: fb_core_wwr::blas::trsm
//! [`syrk`]: fb_core_wwr::blas::syrk
//! [`memset::zero`]: fb_core_wwr::memset::zero

#![forbid(unsafe_code)]

use contracts::debug_requires;
use fb_core_arena::{Bufs, Device, HostKind, Layout, Pinned, RowId, memcpy_d2d, memcpy_d2p};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::blas::{syrk, trsm};
use fb_core_wwr::device_slice::split_at_mut;
use fb_core_wwr::memset;
use fb_recipes_sys::{FB_CHOLESKY_TILE, cholesky_diagonal};
use wwr_sys::{wwr_blas_diag_type, wwr_blas_fill_mode, wwr_blas_operation, wwr_blas_side_mode};

/// The order of the diagonal tiles, and so the width of the panels: the
/// kernel library's `FB_CHOLESKY_TILE`, the C++ `kCholeskyBlock`.
pub const CHOLESKY_BLOCK: usize = FB_CHOLESKY_TILE as usize;

/// Add the factorization `A = L Lᵀ` of the symmetric positive definite
/// `n` × `n` matrix `a` to `dag` on GPU worker `worker`, and return
/// `(L, checked)`: an output row of `n²` doubles, column-major, named
/// `name`, L in its lower triangle and A's strict upper triangle above it as
/// it was; and a pinned output row of one `i32` the check writes when the
/// factor is good, named `"{name} checked"`.
///
/// `a` is an earlier task's output of `n²` doubles, column-major; only its
/// lower triangle is factored, and it is left as it was. The tasks added
/// are `cholesky` (stream), its scratch `"{name} tile"` (the packed
/// diagonal tile) and `"{name} info"` (the flag, on the device and pinned),
/// and `cholesky check` (CPU), which fails the run with `Numerical` naming
/// the first leading minor of A that is not positive.
///
/// A task reading L alone is ordered after `cholesky`, not after the check:
/// on a failed flag it may run on what the factorization left, and the
/// run's result is the check's error either way. A task that must not run
/// at all then reads `checked` too, which puts it after the check, and so
/// among the check's descendants a failure skips.
///
/// First, before anything is added, the kernel library's load check
/// ([`fb_recipes_sys::check_library`]): planning is a run's first call into
/// the library.
///
/// # Errors
/// A `GpuRuntime` error with the check's message, if the loaded
/// `libfb_recipes_kernels.so` is not the one the bindings describe.
///
/// # Panics
/// If `n` does not fit an `i32`, the BLAS's integer, or `a` is a row of
/// another worker's table ([`Dag::add`]).
#[debug_requires(n > 0, "a matrix of at least one row")]
#[track_caller]
pub fn add_cholesky(
    dag: &mut Dag<'_>,
    worker: usize,
    name: &str,
    n: usize,
    a: RowId<Device, f64>,
) -> Result<(RowId<Device, f64>, RowId<Pinned, i32>)> {
    assert!(
        i32::try_from(n).is_ok(),
        "add_cholesky: n = {n} does not fit the BLAS's i32"
    );
    // Not `map_err(Error::gpu)`: through a function pointer the error's
    // origin would be core's, not this call's caller.
    if let Err(e) = fb_recipes_sys::check_library() {
        return Err(Error::gpu(e));
    }
    let rows = dag.add(
        "cholesky",
        worker,
        Factor {
            a,
            n,
            name: name.to_owned(),
        },
    );
    let checked = dag.add(
        "cholesky check",
        worker,
        Check {
            info: rows.info,
            name: name.to_owned(),
        },
    );
    Ok((rows.l, checked))
}

/// GPU: A copied into L, the flag zeroed, the blocked loop over L, then the
/// flag into a pinned row.
struct Factor {
    a: RowId<Device, f64>,
    n: usize,
    /// L's row name, and the other rows' prefix.
    name: String,
}

#[derive(Clone, Copy)]
struct FactorRows {
    l: RowId<Device, f64>,
    tile: RowId<Device, f64>,
    flag: RowId<Device, i32>,
    info: RowId<Pinned, i32>,
}

impl Task for Factor {
    type Rows = FactorRows;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> FactorRows {
        device.read(self.a);
        let tile = self.n.min(CHOLESKY_BLOCK);
        FactorRows {
            l: device.output(&self.name, self.n * self.n),
            tile: device.scratch(&format!("{} tile", self.name), tile * tile),
            flag: device.scratch(&format!("{} info", self.name), 1),
            info: host.output(&format!("{} info", self.name), 1),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(
        &self,
        rows: &FactorRows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let n = self.n;
        let mut l = device.take_mut(rows.l);
        memcpy_d2d(&l, device.get(self.a), worker)?;
        let mut flag = device.take_mut(rows.flag);
        memset::zero(&mut flag, worker)?;
        let mut tile = device.take_mut(rows.tile);
        // add_cholesky checked that n fits an i32, so every count below
        // does; the handle is the worker's, bound to its stream, so each
        // call enqueues there in order.
        let blas = worker.blas();
        let ld = n as i32;
        for j in (0..n).step_by(CHOLESKY_BLOCK) {
            let jb = CHOLESKY_BLOCK.min(n - j);
            // Element (j, j) of L, and the tile that starts there.
            let (_, mut diagonal) = split_at_mut(&mut l, j * (n + 1));
            cholesky_diagonal(worker, jb, j, &mut diagonal, n, &mut tile, &mut flag)?;
            let rest = n - j - jb;
            if rest == 0 {
                break;
            }
            // Everything before (j + jb, j + jb), the trailing matrix's
            // first element, holds the panel; the trailing matrix is the
            // rest. Within the first part, the panel starts at (j + jb, j).
            let (mut head, mut trailing) = split_at_mut(&mut l, (j + jb) * (n + 1));
            let (_, mut panel) = split_at_mut(&mut head, j * (n + 1) + jb);
            let (jb, rest) = (jb as i32, rest as i32);
            trsm(
                blas,
                wwr_blas_side_mode::WWR_BLAS_SIDE_RIGHT,
                wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER,
                wwr_blas_operation::WWR_BLAS_OP_T,
                wwr_blas_diag_type::WWR_BLAS_DIAG_NON_UNIT,
                rest,
                jb,
                1.0,
                &tile,
                jb,
                &mut panel,
                ld,
            )?;
            syrk(
                blas,
                wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER,
                wwr_blas_operation::WWR_BLAS_OP_N,
                rest,
                jb,
                -1.0,
                &panel,
                ld,
                1.0,
                &mut trailing,
                ld,
            )?;
        }
        memcpy_d2p(host, rows.info, &flag, worker)
    }
}

/// CPU: the flag read from its pinned row, after the executor waited for
/// the copy, and the `checked` row written when it is zero. Generic over
/// the host kind and stream, so the tier-3 test hands it a flag from a
/// fixture on `Heap` rows and `HostOnly` workers.
struct Check<H: HostKind> {
    info: RowId<H, i32>,
    /// The factor's row name, `checked`'s prefix.
    name: String,
}

impl<H: HostKind, S> Task<H, S> for Check<H> {
    type Rows = RowId<H, i32>;
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, H>) -> RowId<H, i32> {
        host.read(self.info);
        host.output(&format!("{} checked", self.name), 1)
    }

    fn run(
        &self,
        checked: &RowId<H, i32>,
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, H>,
        _: Lane<'_, S>,
    ) -> Status {
        // The kernel writes only first + k + 1, so the flag is 0 or a
        // positive 1-based minor.
        match host.get(self.info)[0] {
            0 => {
                host.take_mut(*checked)[0] = 0;
                Ok(())
            }
            minor => Err(Error::numerical(format!(
                "cholesky: the leading minor {minor} is not positive"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fb_core_arena::Heap;
    use fb_core_error_handling::ErrorKind;
    use fb_core_thread_pool::HostOnly;
    use std::sync::{Arc, Mutex};

    /// CPU: a fixture's flag written into a host row.
    struct Flag(i32);

    impl Task<Heap, ()> for Flag {
        type Rows = RowId<Heap, i32>;
        const GPU: bool = false;

        fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Heap>) -> Self::Rows {
            host.output("info", 1)
        }

        fn run(
            &self,
            info: &Self::Rows,
            _: &Bufs<'_, Device>,
            host: &Bufs<'_, Heap>,
            _: Lane<'_, ()>,
        ) -> Status {
            host.take_mut(*info)[0] = self.0;
            Ok(())
        }
    }

    /// CPU: `checked` read out, if it runs at all.
    struct Reader {
        checked: RowId<Heap, i32>,
        seen: Arc<Mutex<Option<i32>>>,
    }

    impl Task<Heap, ()> for Reader {
        type Rows = ();
        const GPU: bool = false;

        fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Heap>) {
            host.read(self.checked);
        }

        fn run(
            &self,
            _: &(),
            _: &Bufs<'_, Device>,
            host: &Bufs<'_, Heap>,
            _: Lane<'_, ()>,
        ) -> Status {
            *self.seen.lock().unwrap() = Some(host.get(self.checked)[0]);
            Ok(())
        }
    }

    /// The check of the flag `info`, with a reader of `checked` after it,
    /// run once on `Heap` rows and no device: the run's status, and what
    /// the reader saw if it ran.
    fn check(info: i32) -> (Status, Option<i32>) {
        let mut dag: Dag<'static, Heap, ()> = Dag::new();
        let row = dag.add("flag", 0, Flag(info));
        let checked = dag.add(
            "cholesky check",
            0,
            Check {
                info: row,
                name: "L".to_owned(),
            },
        );
        let seen = Arc::new(Mutex::new(None));
        dag.add(
            "reader",
            0,
            Reader {
                checked,
                seen: Arc::clone(&seen),
            },
        );
        let result = dag
            .plan()
            .instantiate(&[0])
            .run(&mut HostOnly::new(0), 1)
            .result;
        let seen = *seen.lock().unwrap();
        (result, seen)
    }

    /// Tier 3: a zero flag passes and `checked` reaches its reader as 0; a
    /// non-zero one is `Numerical`, naming the minor and the check, and the
    /// reader of `checked` does not run.
    #[test]
    fn the_check_reads_the_flag_and_gates_its_readers() {
        let (result, seen) = check(0);
        result.unwrap();
        assert_eq!(seen, Some(0));

        let (result, seen) = check(3);
        let e = result.unwrap_err();
        assert_eq!(e.kind, ErrorKind::Numerical);
        assert_eq!(e.message, "cholesky: the leading minor 3 is not positive");
        assert_eq!(e.context, ["cholesky check@cpu0"]);
        assert_eq!(seen, None, "the reader of checked ran after a failed check");
    }
}
