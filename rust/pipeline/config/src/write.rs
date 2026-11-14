//! Write nodes: a device matrix copied into a pinned row on the stream,
//! then written to disk by a CPU node that feeds nothing, or a pinned row
//! written by that node alone (`docs/design.md`, "What it writes"); and
//! the SCF's `scf.json`, the same two nodes over ε.

use std::path::PathBuf;

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2h};
use fb_core_error_handling::{Error, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_io_npy::NpyArray;
use serde_json::json;

/// Add the writing of the device matrix `m`, `n × n` doubles column-major,
/// to the `.npy` file `path` to `dag` on GPU worker `worker`.
///
/// Two tasks, both named after the file: `download <file>` (stream: `m`
/// copied into a pinned row of that name) and `write <file>` (CPU, after
/// the executor waited for the copy: the pinned row written as a float64
/// `.npy` in Fortran order, [`fb_io_npy::write_npy`]). The write feeds
/// nothing, so no later task waits on the disk. The directory must exist
/// when the plan runs.
///
/// # Errors
/// None at planning. At run time the download's GPU status, and
/// [`fb_io_npy::write_npy`]'s `Io` error naming a file that cannot be
/// written.
///
/// # Panics
/// If `m` is a row of another worker's table ([`Dag::add`]).
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
    let pinned = dag.add(
        format!("download {file}"),
        worker,
        Download {
            m,
            row: file.clone(),
            n,
            len: n * n,
        },
    );
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
/// If `m` is a row of another worker's table ([`Dag::add`]).
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
    dag.add(
        format!("write {file}"),
        worker,
        Write { pinned: m, n, path },
    );
}

/// Add the writing of an SCF's results to the JSON file `path` to `dag` on
/// GPU worker `worker`: `scalars`, the pinned `[E, ‖e‖, builds]` of
/// [`fb_pipeline_scf::add_scf`], and `eps`, its device row of the `n`
/// orbital energies. Two tasks: `download scf.json` (stream: ε copied into
/// a pinned row) and `write scf.json` (CPU: the object `{"energy",
/// "residual", "iterations", "epsilon"}`, each double at full precision).
/// The write feeds nothing, and the directory must exist when the plan
/// runs.
///
/// # Panics
/// If a row is of another worker's table ([`Dag::add`]).
pub(crate) fn add_write_scf_json(
    dag: &mut Dag<'_>,
    worker: usize,
    scalars: RowId<Pinned, f64>,
    eps: RowId<Device, f64>,
    n: usize,
    path: PathBuf,
) {
    let epsilon = dag.add(
        "download scf.json",
        worker,
        Download {
            m: eps,
            row: "epsilon".into(),
            n,
            len: n,
        },
    );
    dag.add(
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

/// GPU: a device row of `len` doubles, of a dimension `n`, copied into a
/// pinned row on the stream.
struct Download {
    m: RowId<Device, f64>,
    row: String,
    n: usize,
    len: usize,
}

impl Task for Download {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.m);
        host.output(&self.row, self.len)
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(
        &self,
        pinned: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        // The write that reads the pinned row runs after the executor
        // waited for this task's event.
        memcpy_d2p(host, *pinned, device.get(self.m), stream)
    }
}

/// CPU: the pinned copy written to disk as a `.npy`.
struct Write {
    pinned: RowId<Pinned, f64>,
    n: usize,
    path: PathBuf,
}

impl Task for Write {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) {
        host.read(self.pinned);
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(
        &self,
        _: &(),
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        fb_io_npy::write_npy(
            &self.path,
            &NpyArray {
                shape: vec![self.n, self.n],
                data: memcpy_p2h(host, self.pinned),
            },
        )
    }
}

/// CPU: the SCF's scalars and ε written to disk as JSON.
struct WriteScf {
    scalars: RowId<Pinned, f64>,
    epsilon: RowId<Pinned, f64>,
    n: usize,
    path: PathBuf,
}

impl Task for WriteScf {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) {
        host.read(self.scalars);
        host.read(self.epsilon);
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(
        &self,
        _: &(),
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        let [energy, residual, builds] = host.get(self.scalars)[..] else {
            unreachable!("add_scf's scalars are [E, |e|, builds]")
        };
        let text = json!({
            "energy": energy,
            "residual": residual,
            // A whole number the loop's decision wrote.
            "iterations": builds as usize,
            "epsilon": memcpy_p2h(host, self.epsilon),
        });
        std::fs::write(&self.path, format!("{text:#}\n"))
            .map_err(|e| Error::io(format!("cannot write '{}': {e}", self.path.display())))
    }
}
