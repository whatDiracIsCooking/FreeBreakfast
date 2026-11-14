//! One closed-shell SCF iteration's dense nodes on a stream: the density
//! from the orbitals, and the Fock matrix, the energy and the DIIS residual
//! from J and K.

use contracts::debug_requires;
use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2p};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::blas::{geam, gemm, gemv};
use fb_molecule::Molecule;
use wwr_sys::wwr_blas_operation::{WWR_BLAS_OP_N as OP_N, WWR_BLAS_OP_T as OP_T};

/// The number of doubly occupied orbitals of `molecule` as a closed shell:
/// half its electron count ([`Molecule::electron_count`]), which a plan
/// fixes before anything is added, so C's occupied factor is sized by it.
///
/// # Errors
/// An `InvalidConfig` error naming the count, if it is odd (no open shell
/// is ported) or not positive (no electrons, or a charge past `Σ Z`).
pub fn occupied_orbitals(molecule: &Molecule) -> Result<usize> {
    let electrons = molecule.electron_count();
    if electrons <= 0 || electrons % 2 != 0 {
        return Err(Error::config(format!(
            "scf: {electrons} electrons (charge {}): a closed shell needs a positive even count",
            molecule.charge()
        )));
    }
    // Positive, so the cast is exact.
    Ok(electrons as usize / 2)
}

/// Add the density of the orbitals `c` to `dag` on GPU worker `worker`, as
/// the task `density` (stream): C's first `nocc` columns copied into
/// `c_occ` (a geam at `beta = 0`), then `D = 2 C_occ C_occᵀ` into `d` (one
/// gemm).
///
/// `c` is an earlier task's output of `n²` doubles, column-major, column
/// `j` the orbital of the `j`-th orbital energy in ascending order
/// ([`add_solve_fock`](crate::add_solve_fock)'s C), left as it was; `nocc`
/// is [`occupied_orbitals`]'s. `d` (`n²` doubles) and `c_occ` (`n · nocc`,
/// column-major, n × nocc) are rows the loop around the task carries, each
/// produced before it (the guess's D and its factor): the task writes both
/// as their last user in the loop's body
/// ([`Layout::write`](fb_core_arena::Layout::write)), so the next
/// iteration's J reads this D and its K this factor.
///
/// # Panics
/// If `n²` does not fit an `i32`, the BLAS's integer; or as [`Dag::add`],
/// if a row is of another worker's table, or `d` or `c_occ` is written
/// where no loop carries it.
#[debug_requires(n > 0, "a basis of at least one function")]
#[debug_requires((1..=n).contains(&nocc), "between one occupied orbital and n")]
#[track_caller]
pub fn add_density(
    dag: &mut Dag<'_>,
    worker: usize,
    n: usize,
    nocc: usize,
    c: RowId<Device, f64>,
    d: RowId<Device, f64>,
    c_occ: RowId<Device, f64>,
) {
    assert_square_fits(n);
    dag.add(
        "density",
        worker,
        Density {
            c,
            d,
            c_occ,
            n,
            nocc,
        },
    );
}

/// Add the Fock matrix of J and K, its energy and its DIIS residual to
/// `dag` on GPU worker `worker`, and return `(F, e, scalars)`: F and e
/// device output rows of `n²` doubles, column-major, named `"F"` and `"e"`,
/// and scalars a pinned output row of two doubles named `"E, |e|"`:
///
/// ```text
/// F = H + J - K/2
/// E = tr(D (H + F))/2 + E_nuc        scalars[0]
/// e = Xᵀ (F D S - S D F) X
/// ‖e‖_F                              scalars[1]
/// ```
///
/// `h`, `j`, `k`, `d`, `s` and `x` are earlier tasks' outputs of `n²`
/// doubles, column-major and symmetric up to rounding, all left as they
/// were: the core Hamiltonian, the Coulomb and exchange matrices of the
/// density `d` (`D = 2 C_occ C_occᵀ`), the overlap and the orthogonalizer
/// `X = S^(-1/2)`. `e_nuc` is the nuclear repulsion energy, in hartree.
/// The tasks added are:
///
/// - **`fock`** (stream): `H + J` into the scratch `"H + J"` and then F, by
///   two geams, the C++'s order of the sums; `tr(D H)/2 + tr(D F)/2` by two
///   gemvs (D, H and F are symmetric, so each trace is the elementwise dot
///   of `vec(D)` with the other), into a device scratch and down to the
///   pinned output `"E elec"`.
/// - **`residual`** (stream): `M = F D S` by two gemms through the scratch
///   `"F D"`, `M - Mᵀ` by one geam into `"F D S - S D F"` (`S D F` is
///   `(F D S)ᵀ` for symmetric matrices), `e = Xᵀ (M - Mᵀ) X` by two gemms,
///   and `‖e‖²` by one gemv of `vec(e)` with itself, down to the pinned
///   output `"|e|²"`.
/// - **`energy and norm`** (CPU): scalars from the two pinned rows, E_nuc
///   added and the square root taken.
///
/// # Panics
/// If `n²` does not fit an `i32`, the BLAS's integer; or as [`Dag::add`],
/// if a row is of another worker's table.
#[debug_requires(n > 0, "a basis of at least one function")]
#[allow(
    clippy::too_many_arguments,
    reason = "one iteration's six operator rows, each its own"
)]
#[track_caller]
pub fn add_fock(
    dag: &mut Dag<'_>,
    worker: usize,
    n: usize,
    e_nuc: f64,
    h: RowId<Device, f64>,
    j: RowId<Device, f64>,
    k: RowId<Device, f64>,
    d: RowId<Device, f64>,
    s: RowId<Device, f64>,
    x: RowId<Device, f64>,
) -> (RowId<Device, f64>, RowId<Device, f64>, RowId<Pinned, f64>) {
    assert_square_fits(n);
    let fock = dag.add("fock", worker, Fock { h, j, k, d, n });
    let residual = dag.add(
        "residual",
        worker,
        Residual {
            f: fock.f,
            d,
            s,
            x,
            n,
        },
    );
    let scalars = dag.add(
        "energy and norm",
        worker,
        Scalars {
            e_elec: fock.e_elec,
            norm2: residual.norm2,
            e_nuc,
        },
    );
    (fock.f, residual.e, scalars)
}

/// Panic unless `n²` fits an `i32`: every count the BLAS is handed here is
/// at most `n²`.
#[track_caller]
fn assert_square_fits(n: usize) {
    assert!(
        n.checked_mul(n).is_some_and(|n2| i32::try_from(n2).is_ok()),
        "scf: n = {n}: n² does not fit BLAS's i32"
    );
}

/// GPU: `C_occ = C[:, :nocc]`, then `D = 2 C_occ C_occᵀ`.
struct Density {
    c: RowId<Device, f64>,
    d: RowId<Device, f64>,
    c_occ: RowId<Device, f64>,
    n: usize,
    nocc: usize,
}

impl Task for Density {
    type Rows = ();
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) {
        device.read(self.c);
        device.write(self.c_occ);
        device.write(self.d);
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n), ("nocc", self.nocc)]
    }

    fn run(
        &self,
        _: &(),
        device: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        // add_density checked that n² fits an i32, and nocc <= n.
        let (n, nocc) = (self.n as i32, self.nocc as i32);
        let c = device.get(self.c);
        let (mut c_occ, mut d) = (device.take_mut(self.c_occ), device.take_mut(self.d));
        // The handle is the worker's, bound to its stream: the gemm follows
        // the copy.
        geam(
            worker.blas(),
            OP_N,
            OP_N,
            n,
            nocc,
            1.0,
            &c,
            n,
            0.0,
            &c,
            n,
            &mut c_occ,
            n,
        )?;
        gemm(
            worker.blas(),
            OP_N,
            OP_T,
            n,
            n,
            nocc,
            2.0,
            &c_occ,
            n,
            &c_occ,
            n,
            0.0,
            &mut d,
            n,
        )
    }
}

/// GPU: `F = (H + J) - K/2`, and `tr(D (H + F))/2` down to a pinned row.
struct Fock {
    h: RowId<Device, f64>,
    j: RowId<Device, f64>,
    k: RowId<Device, f64>,
    d: RowId<Device, f64>,
    n: usize,
}

/// The Fock task's rows: F and the pinned electronic energy, and the
/// scratch `H + J` and the trace on the device.
#[derive(Clone, Copy)]
struct FockRows {
    f: RowId<Device, f64>,
    e_elec: RowId<Pinned, f64>,
    hj: RowId<Device, f64>,
    trace: RowId<Device, f64>,
}

impl Task for Fock {
    type Rows = FockRows;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> FockRows {
        for row in [self.h, self.j, self.k, self.d] {
            device.read(row);
        }
        FockRows {
            hj: device.scratch("H + J", self.n * self.n),
            trace: device.scratch("tr(D (H + F))/2", 1),
            f: device.output("F", self.n * self.n),
            e_elec: host.output("E elec", 1),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(
        &self,
        rows: &FockRows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let blas = worker.blas();
        // add_fock checked that n² fits an i32.
        let (n, n2) = (self.n as i32, (self.n * self.n) as i32);
        let (h, j, k, d) = (
            device.get(self.h),
            device.get(self.j),
            device.get(self.k),
            device.get(self.d),
        );
        let (mut hj, mut f, mut trace) = (
            device.take_mut(rows.hj),
            device.take_mut(rows.f),
            device.take_mut(rows.trace),
        );
        // Each call follows the one before on the handle's stream.
        geam(blas, OP_N, OP_N, n, n, 1.0, &h, n, 1.0, &j, n, &mut hj, n)?;
        geam(blas, OP_N, OP_N, n, n, 1.0, &hj, n, -0.5, &k, n, &mut f, n)?;
        // vec(D)ᵀ vec(M): D as an n² × 1 matrix, transposed.
        gemv(blas, OP_T, n2, 1, 0.5, &d, n2, &h, 1, 0.0, &mut trace, 1)?;
        gemv(blas, OP_T, n2, 1, 0.5, &d, n2, &f, 1, 1.0, &mut trace, 1)?;
        memcpy_d2p(host, rows.e_elec, &trace, worker)
    }
}

/// GPU: `e = Xᵀ (F D S - S D F) X`, and `‖e‖²` down to a pinned row.
struct Residual {
    f: RowId<Device, f64>,
    d: RowId<Device, f64>,
    s: RowId<Device, f64>,
    x: RowId<Device, f64>,
    n: usize,
}

/// The residual task's rows: e and its pinned squared norm, and the
/// scratch products on the device.
#[derive(Clone, Copy)]
struct ResidualRows {
    e: RowId<Device, f64>,
    norm2: RowId<Pinned, f64>,
    fd: RowId<Device, f64>,
    fds: RowId<Device, f64>,
    commutator: RowId<Device, f64>,
    square: RowId<Device, f64>,
}

impl Task for Residual {
    type Rows = ResidualRows;
    const GPU: bool = true;

    fn layout(
        &self,
        device: &mut Layout<'_, Device>,
        host: &mut Layout<'_, Pinned>,
    ) -> ResidualRows {
        for row in [self.f, self.d, self.s, self.x] {
            device.read(row);
        }
        let n2 = self.n * self.n;
        ResidualRows {
            fd: device.scratch("F D", n2),
            fds: device.scratch("F D S", n2),
            commutator: device.scratch("F D S - S D F", n2),
            square: device.scratch("|e|² on the device", 1),
            e: device.output("e", n2),
            norm2: host.output("|e|²", 1),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(
        &self,
        rows: &ResidualRows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let blas = worker.blas();
        // add_fock checked that n² fits an i32.
        let (n, n2) = (self.n as i32, (self.n * self.n) as i32);
        let (f, d, s, x) = (
            device.get(self.f),
            device.get(self.d),
            device.get(self.s),
            device.get(self.x),
        );
        let (mut fd, mut fds, mut commutator, mut e, mut square) = (
            device.take_mut(rows.fd),
            device.take_mut(rows.fds),
            device.take_mut(rows.commutator),
            device.take_mut(rows.e),
            device.take_mut(rows.square),
        );
        // Each call follows the one before on the handle's stream.
        gemm(
            blas, OP_N, OP_N, n, n, n, 1.0, &f, n, &d, n, 0.0, &mut fd, n,
        )?;
        gemm(
            blas, OP_N, OP_N, n, n, n, 1.0, &fd, n, &s, n, 0.0, &mut fds, n,
        )?;
        geam(
            blas,
            OP_N,
            OP_T,
            n,
            n,
            1.0,
            &fds,
            n,
            -1.0,
            &fds,
            n,
            &mut commutator,
            n,
        )?;
        // (M - Mᵀ) X into "F D", free again, then Xᵀ times it.
        gemm(
            blas,
            OP_N,
            OP_N,
            n,
            n,
            n,
            1.0,
            &commutator,
            n,
            &x,
            n,
            0.0,
            &mut fd,
            n,
        )?;
        gemm(
            blas, OP_T, OP_N, n, n, n, 1.0, &x, n, &fd, n, 0.0, &mut e, n,
        )?;
        gemv(blas, OP_T, n2, 1, 1.0, &e, n2, &e, 1, 0.0, &mut square, 1)?;
        memcpy_d2p(host, rows.norm2, &square, worker)
    }
}

/// CPU: `[E elec + E_nuc, sqrt(|e|²)]`, after the executor waited for both
/// copies down.
struct Scalars {
    e_elec: RowId<Pinned, f64>,
    norm2: RowId<Pinned, f64>,
    e_nuc: f64,
}

impl Task for Scalars {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.read(self.e_elec);
        host.read(self.norm2);
        host.output("E, |e|", 2)
    }

    fn run(
        &self,
        scalars: &Self::Rows,
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        let (e_elec, norm2) = (host.get(self.e_elec)[0], host.get(self.norm2)[0]);
        host.take_mut(*scalars)
            .copy_from_slice(&[e_elec + self.e_nuc, norm2.sqrt()]);
        Ok(())
    }
}
