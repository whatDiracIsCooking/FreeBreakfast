//! Write nodes: a device matrix copied into a pinned row on the stream,
//! then written to disk by a CPU node that feeds nothing, or a pinned row
//! written by that node alone (`docs/design.md`, "What it writes"); and
//! the SCF's `scf.json`, the same two nodes over ε.

use std::path::PathBuf;

use fb_core_arena::{Bufs, Device, Heap, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, Status};
use fb_core_planner::{CpuTask, Dag};
use serde_json::json;

/// Add the writing of the device matrix `m`, `n × n` doubles column-major,
/// to the `.npy` file `path` to `dag` on GPU worker `worker`.
///
/// Two nodes, both named after the file: `download <file>` (the planner's
/// copy node, [`Dag::download`]: `m` copied into a pinned row of that
/// name) and `write <file>` (CPU, after
/// the executor waited for the copy: the pinned row written in place as a
/// float64 `.npy` in Fortran order, [`fb_io_npy::write_npy`]). The write feeds
/// nothing, so no later task waits on the disk. The directory must exist
/// when the plan runs.
///
/// # Errors
/// None at planning. At run time the download's GPU status, and
/// [`fb_io_npy::write_npy`]'s `Io` error naming a file that cannot be
/// written.
///
/// # Panics
/// If `m` is a row of another worker's table ([`Dag::download`]).
pub fn add_write_npy(
    dag: &mut Dag<'_>,
    worker: usize,
    m: RowId<Device, f64>,
    n: usize,
    path: PathBuf,
) {
    let file = path
        .file_name()
        .map_or_else(|| path.display().to_string(), |f| f.display().to_string());
    let pinned = dag.download(worker, m, &file, n * n);
    add_write_host_npy(dag, worker, pinned, n, path);
}

/// Add the writing of the host matrix `m`, `n × n` doubles column-major,
/// to the `.npy` file `path` to `dag` on GPU worker `worker`: the write
/// node of [`add_write_npy`] alone, `write <file>` on the CPU, for a
/// product that is already a pinned row. It feeds nothing, and the
/// directory must exist when the plan runs.
///
/// # Errors
/// None at planning. At run time [`fb_io_npy::write_npy`]'s `Io` error
/// naming a file that cannot be written.
///
/// # Panics
/// If `m` is a row of another worker's table ([`Dag::add_gpu`]).
pub fn add_write_host_npy(
    dag: &mut Dag<'_>,
    worker: usize,
    m: RowId<Pinned, f64>,
    n: usize,
    path: PathBuf,
) {
    let file = path
        .file_name()
        .map_or_else(|| path.display().to_string(), |f| f.display().to_string());
    dag.add_cpu(
        format!("write {file}"),
        worker,
        Write { pinned: m, n, path },
    );
}

/// Add the writing of an SCF's results to the JSON file `path` to `dag` on
/// GPU worker `worker`: `scalars`, the heap row `[E, ‖e‖, builds]` of
/// [`fb_pipeline_scf::add_scf`], and `eps`, its device row of the `n`
/// orbital energies. Two nodes: `download epsilon` (the planner's copy
/// node, [`Dag::download`]: ε copied into a pinned row `epsilon`) and
/// `write scf.json` (CPU: the object `{"energy",
/// "residual", "iterations", "epsilon"}`, each double at full precision).
/// The write feeds nothing, and the directory must exist when the plan
/// runs.
///
/// # Panics
/// If a row is of another worker's table ([`Dag::download`]).
pub(crate) fn add_write_scf_json(
    dag: &mut Dag<'_>,
    worker: usize,
    scalars: RowId<Heap, f64>,
    eps: RowId<Device, f64>,
    n: usize,
    path: PathBuf,
) {
    let epsilon = dag.download(worker, eps, "epsilon", n);
    dag.add_cpu(
        "write scf.json",
        worker,
        WriteScf {
            scalars,
            epsilon,
            n,
            path,
        },
    );
}

/// CPU: the pinned copy written to disk as a `.npy`.
struct Write {
    pinned: RowId<Pinned, f64>,
    n: usize,
    path: PathBuf,
}

impl CpuTask for Write {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {
        host.read(self.pinned);
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(&self, _: &(), host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        fb_io_npy::write_npy(&self.path, &[self.n, self.n], host.get(self.pinned))
    }
}

/// CPU: the SCF's scalars, from the heap, and ε written to disk as JSON.
struct WriteScf {
    scalars: RowId<Heap, f64>,
    epsilon: RowId<Pinned, f64>,
    n: usize,
    path: PathBuf,
}

impl CpuTask for WriteScf {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) {
        heap.read(self.scalars);
        host.read(self.epsilon);
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(&self, _: &(), host: &Bufs<'_, Pinned>, heap: &Bufs<'_, Heap>) -> Status {
        let [energy, residual, builds] = heap.get(self.scalars)[..] else {
            unreachable!("add_scf's scalars are [E, |e|, builds]")
        };
        let text = json!({
            "energy": energy,
            "residual": residual,
            // A whole number the loop's decision wrote.
            "iterations": builds as usize,
            "epsilon": host.get(self.epsilon),
        });
        std::fs::write(&self.path, format!("{text:#}\n"))
            .map_err(|e| Error::io(format!("cannot write '{}': {e}", self.path.display())))
    }
}
