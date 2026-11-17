//! The dense conventional ERI over the run's GPU workers: [`add_dense`] and
//! [`add_dense_jk`], the crate doc's dense storage.

use contracts::debug_requires;
use fb_basis::BasisSet;
use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Dag, GpuTask, Partition};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::blas::gemm;
use fb_core_wwr::device_slice::{split_at, split_at_mut};
use fb_integrals_mmd::{add_mixed_records, mixed_record_tables};
use fb_integrals_sys::{DenseBlocks, FB_CONVENTIONAL_MAX_MOMENTUM, add_dense_tasks, dense_row_dot};
use fb_recipes_matrix_ao::add_harmonics;
use fb_recipes_symmetrize::symmetrize;
use wwr_sys::wwr_blas_operation::{WWR_BLAS_OP_N as OP_N, WWR_BLAS_OP_T as OP_T};

/// The dense conventional integrals of a plan, [`add_dense`]'s product:
/// what [`add_dense_jk`] reads.
#[derive(Clone, Debug)]
pub struct Dense {
    /// Each GPU worker's row blocks of `M_J` and `M_K`, worker `w`'s the
    /// rows `split.range(w)`; `None` for a worker of no rows.
    pub blocks: Vec<Option<DenseBlocks>>,
    /// The rows of `M_J` and `M_K`, `n_cart²` of them, cut over the
    /// workers at shell boundaries: also the entries of `vec(J)` and
    /// `vec(K)` each worker's products write.
    pub split: Partition,
    /// U, the basis's Cartesian-to-spherical transform: worker 0's device
    /// row, `n_cart` × `n_sph`, column-major.
    pub u: RowId<Device, f64>,
    /// The basis's Cartesian AO count.
    pub n_cart: usize,
    /// The basis's spherical AO count, the order of D, J and K.
    pub n_sph: usize,
}

/// The shells of `basis` cut into `workers` consecutive ranges whose
/// Cartesian AOs, and so rows of `M_J` and `M_K`, are as even as shell
/// boundaries let them be: range `w` ends at the boundary nearest
/// `(w + 1) / workers` of the AOs, never before the last one ended. A
/// worker past the shells holds none.
fn shell_split(basis: &BasisSet, workers: usize) -> Vec<std::ops::Range<usize>> {
    let ends: Vec<usize> = basis
        .shells()
        .iter()
        .scan(0, |at, s| {
            *at += s.function_count_cart() as usize;
            Some(*at)
        })
        .collect();
    let total = ends.last().copied().unwrap_or(0);
    let mut ranges = Vec::with_capacity(workers);
    let mut start = 0;
    for w in 0..workers {
        let end = if w + 1 == workers {
            ends.len()
        } else {
            let target = total * (w + 1) / workers;
            // The boundary nearest the target: shells [start, e) end at
            // ends[e - 1].
            (start..=ends.len())
                .min_by_key(|&e| {
                    let at = if e == 0 { 0 } else { ends[e - 1] };
                    at.abs_diff(target)
                })
                .unwrap_or(start)
        };
        ranges.push(start..end);
        start = end;
    }
    ranges
}

/// Add the dense conventional integrals of `basis` to `dag` over GPU
/// workers `0..workers`, and return them ([`Dense`]): worker `w`'s row
/// blocks of `M_J` and `M_K` (the crate doc's dense storage), its shells
/// of `m` a range [`shell_split`] balances by rows, and U's row on
/// worker 0.
///
/// Each worker with rows gets the records between `basis` and itself
/// ([`add_mixed_records`]) and its blocks
/// ([`fb_integrals_sys::add_dense_tasks`]: `dense tables`, `dense eri`).
/// Only worker 0's tasks call the BLAS, whose workspace the plan's caller
/// declares ([`Dag::blas_workspace`]). `u` is the plan's slot for U over `basis`
/// on worker 0, shared with every other reader, as
/// [`add_conventional`](crate::add_conventional) takes it.
///
/// First, before anything is added, the kernel libraries' load checks,
/// and then what the kernels can take, both at planning. The blocks are
/// output rows of each worker's table, which every reader's edge keeps
/// live, so a basis whose blocks do not fit the cards is refused at
/// planning by admission ([`fb_core_arena::admit`]), naming the worker,
/// the bytes over and the rows at the peak: `M_J` and `M_K` are each
/// `rows · n_cart²` doubles on a worker of `rows` of them.
///
/// # Errors
/// A `GpuRuntime` error with the check's message, if a loaded kernel
/// library is not the one the bindings describe; an `Unsupported` one
/// naming the shell, if a shell is above d
/// ([`FB_CONVENTIONAL_MAX_MOMENTUM`]), or naming the count, if `n_cart²`
/// does not fit the kernels' 32-bit column index. Nothing is added to
/// `dag` on an error.
///
/// # Panics
/// If the row `u` holds is of another worker's or another plan's table
/// ([`Dag::add_gpu`]).
#[debug_requires(basis.nshells() > 0, "a basis of at least one shell")]
#[debug_requires(workers > 0, "at least one GPU worker")]
#[track_caller]
pub fn add_dense(
    dag: &mut Dag<'_>,
    workers: usize,
    basis: &BasisSet,
    u: &mut Option<RowId<Device, f64>>,
) -> Result<Dense> {
    // Not `map_err(Error::gpu)`, as in add_conventional.
    if let Err(e) = fb_integrals_sys::check_library() {
        return Err(Error::gpu(e));
    }
    if let Err(e) = fb_recipes_sys::check_library() {
        return Err(Error::gpu(e));
    }
    if let Some((s, shell)) = (0..)
        .zip(basis.shells())
        .find(|(_, shell)| shell.angular_momentum() > FB_CONVENTIONAL_MAX_MOMENTUM)
    {
        return Err(Error::unsupported(format!(
            "add_dense: shell {s} has momentum {}, but the dense ERI's kernels cover shells \
             through l = {FB_CONVENTIONAL_MAX_MOMENTUM} (d)",
            shell.angular_momentum()
        )));
    }
    let n_cart = basis.nao_cart() as usize;
    if n_cart
        .checked_mul(n_cart)
        .is_none_or(|n2| i32::try_from(n2).is_err())
    {
        return Err(Error::unsupported(format!(
            "add_dense: the basis has {n_cart} Cartesian AOs, whose {n_cart}² columns pass the \
             kernels' 32-bit index"
        )));
    }
    let u = add_harmonics(dag, 0, basis, u);
    let ranges = shell_split(basis, workers);
    let mut counts = Vec::with_capacity(workers);
    let mut blocks = Vec::with_capacity(workers);
    for (w, shells) in ranges.into_iter().enumerate() {
        if shells.is_empty() {
            counts.push(0);
            blocks.push(None);
            continue;
        }
        let classes = add_mixed_records(dag, w, basis, basis)?;
        let tables = mixed_record_tables(basis, basis);
        let b = add_dense_tasks(dag, w, &classes, &tables, shells);
        counts.push(b.blocks.iter().map(|r| r.len()).sum::<usize>() * n_cart);
        blocks.push(Some(b));
    }
    Ok(Dense {
        blocks,
        split: Partition::new(&counts),
        u,
        n_cart,
        n_sph: basis.nao_sphe() as usize,
    })
}

/// Add J and K of the spherical density `density` over the dense
/// integrals `eri` to `dag`, and return them, `(J, K)`: two output rows of
/// `n_sph²` doubles on worker 0 named `"J"` and `"K"`, column-major,
/// `J_mn = Σ_ls (mn|ls) D_ls` and `K_mn = Σ_ls (ml|ns) D_ls`, each averaged
/// with its transpose, so exactly symmetric.
///
/// `density` is worker 0's row of `n_sph²` doubles, an earlier task's
/// output, left as it was. The tasks, in order:
///
/// - **`dense density`** (worker 0's stream): `D_cart = U D Uᵀ` by two
///   gemms into the row `"D cartesian"`, `n_cart²`.
/// - **`broadcast D cartesian`** ([`Dag::broadcast`]): it, to every worker.
/// - **`dense jk`** (each worker's stream): its pieces of `vec(J)` and
///   `vec(K)`, each of its sub-blocks' rows by `vec(D)` per matrix
///   ([`dense_row_dot`]), into the rows `"J piece"` and `"K piece"`, its
///   `split.range(w)`. The arrays are column-major, so `vec(D)` read
///   row-major is Dᵀ, and the product read back column-major is J and K of
///   D itself, for any D.
/// - **`gather J cartesian`**, **`gather K cartesian`** ([`Dag::gather`]):
///   the pieces onto worker 0, exactly, `n_cart²` each.
/// - **`dense spherical`** (worker 0's stream): `Uᵀ · U` around each by two
///   gemms into J and K, each then averaged with its transpose
///   ([`symmetrize`]).
///
/// A collective spans every GPU worker, so a plan holding this has exactly
/// `eri.blocks.len()` GPU workers. Each entry of a piece is its row's
/// products summed in an order `n_cart²` alone fixes ([`dense_row_dot`]),
/// not the BLAS's GEMV, whose order may follow the call's shape, and the
/// gathers are exact (`docs/programming_model.md`, "Two kinds of
/// distribution": an ordered reduce). J and K on several workers are held
/// to one worker's within `2 n_cart² ε` of their scale, not bitwise.
///
/// # Panics
/// If `density` is not a row of worker 0's table ([`Dag::add_gpu`]).
#[track_caller]
pub fn add_dense_jk(
    dag: &mut Dag<'_>,
    eri: &Dense,
    density: RowId<Device, f64>,
) -> (RowId<Device, f64>, RowId<Device, f64>) {
    let (nc, workers) = (eri.n_cart, eri.blocks.len());
    let d_cart = dag
        .add_gpu(
            "dense density",
            0,
            Cartesian {
                u: eri.u,
                density,
                n_cart: nc,
                n_sph: eri.n_sph,
            },
        )
        .d;
    let ds = dag.broadcast("D cartesian", 0, d_cart, nc * nc, workers);
    let mut j_pieces = Vec::with_capacity(workers);
    let mut k_pieces = Vec::with_capacity(workers);
    for (w, blocks) in eri.blocks.iter().enumerate() {
        let task = Product {
            blocks: blocks.clone(),
            d: ds[w],
            n_cart: nc,
            rows: eri.split.range(w).len(),
        };
        let [j, k] = dag.add_gpu("dense jk", w, task);
        j_pieces.push(j);
        k_pieces.push(k);
    }
    let j_cart = dag.gather("J cartesian", 0, &j_pieces, &eri.split);
    let k_cart = dag.gather("K cartesian", 0, &k_pieces, &eri.split);
    let out = dag.add_gpu(
        "dense spherical",
        0,
        Spherical {
            u: eri.u,
            j_cart,
            k_cart,
            n_cart: nc,
            n_sph: eri.n_sph,
        },
    );
    (out.j, out.k)
}

/// GPU, worker 0: `D_cart = U D Uᵀ`.
struct Cartesian {
    u: RowId<Device, f64>,
    density: RowId<Device, f64>,
    n_cart: usize,
    n_sph: usize,
}

#[derive(Clone, Copy)]
struct CartesianRows {
    work: RowId<Device, f64>,
    d: RowId<Device, f64>,
}

impl GpuTask for Cartesian {
    type Rows = CartesianRows;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.u);
        device.read(self.density);
        CartesianRows {
            work: device.scratch("dense U", self.n_cart * self.n_sph),
            d: device.output("D cartesian", self.n_cart * self.n_cart),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n_cart", self.n_cart), ("n_ao", self.n_sph)]
    }

    fn run(
        &self,
        rows: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        let handle = stream.blas();
        let (u, d) = (device.get(self.u), device.get(self.density));
        let mut work = device.take_mut(rows.work);
        let mut d_cart = device.take_mut(rows.d);
        // In range: n_cart² fits an i32 (add_dense), and n_sph <= n_cart.
        let (nc, ns) = (self.n_cart as i32, self.n_sph as i32);
        // work = U D (n_cart × n_sph), then D_cart = work Uᵀ.
        gemm(
            handle, OP_N, OP_N, nc, ns, ns, 1.0, &u, nc, &d, ns, 0.0, &mut work, nc,
        )?;
        gemm(
            handle,
            OP_N,
            OP_T,
            nc,
            nc,
            ns,
            1.0,
            &work,
            nc,
            &u,
            nc,
            0.0,
            &mut d_cart,
            nc,
        )
    }
}

/// GPU, each worker: its pieces of `vec(J)` and `vec(K)`, a row dot per
/// sub-block per matrix.
struct Product {
    blocks: Option<DenseBlocks>,
    d: RowId<Device, f64>,
    n_cart: usize,
    rows: usize,
}

impl GpuTask for Product {
    type Rows = [RowId<Device, f64>; 2];

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        if let Some(b) = &self.blocks {
            device.read(b.j);
            device.read(b.k);
            device.read(self.d);
        }
        [
            device.output("J piece", self.rows),
            device.output("K piece", self.rows),
        ]
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n_cart", self.n_cart), ("rows", self.rows)]
    }

    fn uses_libraries(&self) -> bool {
        false
    }

    fn run(
        &self,
        rows: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        let Some(b) = &self.blocks else {
            return Ok(());
        };
        let d = device.get(self.d);
        let n = self.n_cart;
        let n_col = n * n;
        for (matrix, piece) in [(b.j, rows[0]), (b.k, rows[1])] {
            let m = device.get(matrix);
            let mut piece = device.take_mut(piece);
            let first = b.blocks.first().map_or(0, |r| r.start);
            for block in &b.blocks {
                let (r0, r) = ((block.start - first) * n, block.len() * n);
                let (_, from) = split_at(&m, r0 * n_col);
                let (a, _) = split_at(&from, r * n_col);
                let (_, mut y) = split_at_mut(&mut piece, r0);
                let (mut y, _) = split_at_mut(&mut y, r);
                // The block is row-major: the sub-block's rows of M vec(D),
                // each in an order n_cart² alone fixes.
                dense_row_dot(stream, r, n_col, &a, &d, &mut y)?;
            }
        }
        Ok(())
    }
}

/// GPU, worker 0: J and K back to the spherical AOs, each symmetrized.
struct Spherical {
    u: RowId<Device, f64>,
    j_cart: RowId<Device, f64>,
    k_cart: RowId<Device, f64>,
    n_cart: usize,
    n_sph: usize,
}

#[derive(Clone, Copy)]
struct SphericalRows {
    work: RowId<Device, f64>,
    j: RowId<Device, f64>,
    k: RowId<Device, f64>,
}

impl GpuTask for Spherical {
    type Rows = SphericalRows;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.u);
        device.read(self.j_cart);
        device.read(self.k_cart);
        SphericalRows {
            work: device.scratch("dense spherical U", self.n_cart * self.n_sph),
            j: device.output("J", self.n_sph * self.n_sph),
            k: device.output("K", self.n_sph * self.n_sph),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n_cart", self.n_cart), ("n_ao", self.n_sph)]
    }

    fn run(
        &self,
        rows: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        let handle = stream.blas();
        let u = device.get(self.u);
        let mut work = device.take_mut(rows.work);
        // In range: n_cart² fits an i32 (add_dense), and n_sph <= n_cart.
        let (nc, ns) = (self.n_cart as i32, self.n_sph as i32);
        for (cart, out) in [(self.j_cart, rows.j), (self.k_cart, rows.k)] {
            let cart = device.get(cart);
            let mut out = device.take_mut(out);
            // Uᵀ V U: work = V U (n_cart × n_sph), then Uᵀ work.
            gemm(
                handle, OP_N, OP_N, nc, ns, nc, 1.0, &cart, nc, &u, nc, 0.0, &mut work, nc,
            )?;
            gemm(
                handle, OP_T, OP_N, ns, ns, nc, 1.0, &u, nc, &work, nc, 0.0, &mut out, ns,
            )?;
            // The two gemms round (m, n) and (n, m) in different orders, and
            // the products' J and K are each symmetric only to rounding.
            symmetrize(stream, self.n_sph, &mut out, self.n_sph)?;
        }
        Ok(())
    }
}
