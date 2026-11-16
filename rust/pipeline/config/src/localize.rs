//! The localization step's tasks over an SCF's products: the C++
//! `LocalizationStep::run`'s Pipek-Mezey mode, each occupied space
//! localized among itself and the localized orbitals sorted by their Fock
//! expectation value within it.

use fb_basis::BasisSet;
use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_p2d};
use fb_core_error_handling::{Result, Status};
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::blas::gemm;
use fb_core_wwr::copy;
use fb_core_wwr::device_slice::split_at_mut;
use fb_localization_pipek_mezey::{PmOptions, add_pm_localize};
use wwr_sys::wwr_blas_operation::WWR_BLAS_OP_N as OP_N;

/// One occupied space's localization: its first column in C, its width,
/// and [`add_pm_localize`]'s info, `C_local` and `U`.
#[derive(Clone, Copy)]
struct Space {
    first: usize,
    width: usize,
    info: RowId<Heap, f64>,
    c_local: RowId<Device, f64>,
    u: RowId<Pinned, f64>,
}

/// Add the localization of an SCF's occupied orbitals to `dag` on GPU
/// worker `worker` and return `(info, C_local)`: a heap row
/// `[L before, L after, sweeps]`, each summed over the spaces, as the C++
/// `Info` is, and a device output row of the `n_alpha` localized orbitals,
/// `n × n_alpha` column-major, the `n_beta` doubly occupied first, then the
/// singly occupied, each space in ascending order of `<i|F|i>`.
///
/// `s` and `c` are the SCF's S and orbitals (`n²` each, C column-major,
/// its occupied columns first, doubly before singly), and `eps` its `n`
/// orbital energies. ROHF's energy is invariant only under rotations
/// within each occupied space, so each is localized among itself
/// ([`add_pm_localize`]): C's first `n_beta` columns, then its next
/// `n_alpha - n_beta`; a closed shell (`n_alpha = n_beta`) is the first
/// space alone, and an empty space is skipped (no doubly occupied orbital
/// in the H atom). The singly occupied columns are first copied into a row
/// of their own by **`singly occupied`** (stream), unless they start at
/// column 0. Then, for the C++ sort:
///
/// - each space's U copied to the pinned row `"<space> U host"` by the
///   planner's copy node, and ε to `"epsilon host"`;
/// - **`localization order`** (CPU): each localized orbital's
///   `<i|F|i> = Σ_k U_ki² ε_k`, over its space's canonical orbitals `k`, the
///   diagonal of the C++ `Uᵀ diag(ε_occ) U`, stably sorted within its
///   space, so orbitals of equal energy keep the sweeps' order, as the C++
///   `merge_sort` keeps them; written as the permutation matrix
///   `"localization permutation"`, `n_alpha × n_alpha`, block-diagonal by
///   space, and, with two spaces, their infos summed into the heap row
///   `"localization info"`;
/// - **`localization sort`** (stream): the permutation copied to the device,
///   each space's `C_local` copied into its columns of a scratch row, and
///   `C_local P` by a gemm into the output row `"C local"`, a pure
///   permutation of the columns, since every product is by 1 or 0 and every
///   sum adds zeros.
///
/// # Errors
/// [`add_pm_localize`]'s, at planning.
pub(crate) fn add_localization(
    dag: &mut Dag<'_>,
    worker: usize,
    options: &PmOptions,
    basis: &BasisSet,
    [n_alpha, n_beta]: [usize; 2],
    [s, c, eps]: [RowId<Device, f64>; 3],
) -> Result<(RowId<Heap, f64>, RowId<Device, f64>)> {
    let n = basis.nao_sphe() as usize;
    let mut spaces = Vec::with_capacity(2);
    for (name, first, width) in [("doubly", 0, n_beta), ("singly", n_beta, n_alpha - n_beta)] {
        if width == 0 {
            continue;
        }
        let columns = if first == 0 {
            c
        } else {
            dag.add_gpu("singly occupied", worker, Columns { c, n, first, width })
                .columns
        };
        let (info, [c_local, u]) = add_pm_localize(dag, worker, options, basis, width, s, columns)?;
        let u = dag.download(worker, u, &format!("{name} U host"), width * width);
        spaces.push(Space {
            first,
            width,
            info,
            c_local,
            u,
        });
    }
    let eps = dag.download(worker, eps, "epsilon host", n);
    let order = dag.add_cpu(
        "localization order",
        worker,
        Order {
            spaces: spaces.clone(),
            eps,
            n_occ: n_alpha,
        },
    );
    let SortRows { c_local, .. } = dag.add_gpu(
        "localization sort",
        worker,
        Sort {
            spaces: spaces.clone(),
            permutation: order.permutation,
            n,
            n_occ: n_alpha,
        },
    );
    Ok((order.info.unwrap_or(spaces[0].info), c_local))
}

/// GPU: C's columns `[first, first + width)` into a row of their own.
struct Columns {
    c: RowId<Device, f64>,
    n: usize,
    first: usize,
    width: usize,
}

/// The columns task's rows: the output, and the scratch copy of C it is
/// cut from.
#[derive(Clone, Copy)]
struct ColumnsRows {
    columns: RowId<Device, f64>,
    c_copy: RowId<Device, f64>,
}

impl GpuTask for Columns {
    type Rows = ColumnsRows;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> ColumnsRows {
        device.read(self.c);
        ColumnsRows {
            c_copy: device.scratch("C copy", self.n * self.n),
            columns: device.output("C singly", self.n * self.width),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n), ("first", self.first), ("width", self.width)]
    }

    fn run(
        &self,
        rows: &ColumnsRows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let mut c_copy = device.take_mut(rows.c_copy);
        // A read row is not cut, so the columns come from a copy of C. The
        // second copy follows the first on the stream; the singly occupied
        // columns end at n_alpha <= n.
        copy::d2d(&mut c_copy, &device.get(self.c), worker)?;
        let (_, mut tail) = split_at_mut(&mut c_copy, self.n * self.first);
        let (columns, _) = split_at_mut(&mut tail, self.n * self.width);
        copy::d2d(&mut device.take_mut(rows.columns), &columns, worker)
    }
}

/// CPU: the sort's permutation, and with two spaces their summed info.
struct Order {
    spaces: Vec<Space>,
    eps: RowId<Pinned, f64>,
    n_occ: usize,
}

/// The order task's rows: the permutation, and the summed info when there
/// are two spaces.
#[derive(Clone, Copy)]
struct OrderRows {
    permutation: RowId<Pinned, f64>,
    info: Option<RowId<Heap, f64>>,
}

impl CpuTask for Order {
    type Rows = OrderRows;

    fn layout(&self, host: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) -> OrderRows {
        host.read(self.eps);
        for space in &self.spaces {
            host.read(space.u);
            heap.read(space.info);
        }
        OrderRows {
            permutation: host.output("localization permutation", self.n_occ * self.n_occ),
            info: (self.spaces.len() > 1).then(|| heap.output("localization info", 3)),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n_occ", self.n_occ)]
    }

    fn run(&self, rows: &OrderRows, host: &Bufs<'_, Pinned>, heap: &Bufs<'_, Heap>) -> Status {
        let eps = host.get(self.eps);
        let p = host.take_mut(rows.permutation);
        p.fill(0.0);
        for space in &self.spaces {
            let (first, width) = (space.first, space.width);
            let u = host.get(space.u);
            // <i|F|i> = Σ_k U_ki² ε_k, column i of the space's U.
            let energy = |i: usize| -> f64 {
                (0..width)
                    .map(|k| u[k + width * i].powi(2) * eps[first + k])
                    .sum()
            };
            let mut order: Vec<usize> = (0..width).collect();
            // Stable: orbitals of equal energy keep the sweeps' order.
            order.sort_by(|&a, &b| energy(a).total_cmp(&energy(b)));
            // Column first + j of the sorted C is column first + order[j]
            // of the unsorted.
            for (j, &i) in order.iter().enumerate() {
                p[(first + i) + self.n_occ * (first + j)] = 1.0;
            }
        }
        if let Some(info) = rows.info {
            let total = heap.take_mut(info);
            total.fill(0.0);
            for space in &self.spaces {
                for (t, v) in total.iter_mut().zip(heap.get(space.info)) {
                    *t += v;
                }
            }
        }
        Ok(())
    }
}

/// GPU: the spaces' localized orbitals side by side, permuted by the
/// order's matrix.
struct Sort {
    spaces: Vec<Space>,
    permutation: RowId<Pinned, f64>,
    n: usize,
    n_occ: usize,
}

/// The sort's rows: the output, and the scratch permutation and unsorted
/// orbitals.
#[derive(Clone, Copy)]
struct SortRows {
    c_local: RowId<Device, f64>,
    permutation: RowId<Device, f64>,
    unsorted: RowId<Device, f64>,
}

impl GpuTask for Sort {
    type Rows = SortRows;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> SortRows {
        host.read(self.permutation);
        for space in &self.spaces {
            device.read(space.c_local);
        }
        let (n, n_occ) = (self.n, self.n_occ);
        SortRows {
            permutation: device.scratch("localization permutation", n_occ * n_occ),
            unsorted: device.scratch("C local unsorted", n * n_occ),
            c_local: device.output("C local", n * n_occ),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n), ("n_occ", self.n_occ)]
    }

    fn run(
        &self,
        rows: &SortRows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        // The plan's n² fits an i32 (add_scf), and n_occ <= n.
        let (n, n_occ) = (self.n as i32, self.n_occ as i32);
        let [mut permutation, mut unsorted, mut c_local] =
            [rows.permutation, rows.unsorted, rows.c_local].map(|r| device.take_mut(r));
        // The permutation as `guess factor` takes its floor: a stream node
        // does not read a pinned row on the host. Each call follows the one
        // before on the stream.
        memcpy_p2d(&mut permutation, host, self.permutation, worker)?;
        for space in &self.spaces {
            let (_, mut tail) = split_at_mut(&mut unsorted, self.n * space.first);
            let (mut columns, _) = split_at_mut(&mut tail, self.n * space.width);
            copy::d2d(&mut columns, &device.get(space.c_local), worker)?;
        }
        gemm(
            worker.blas(),
            OP_N,
            OP_N,
            n,
            n_occ,
            n_occ,
            1.0,
            &unsorted,
            n,
            &permutation,
            n_occ,
            0.0,
            &mut c_local,
            n,
        )
    }
}
