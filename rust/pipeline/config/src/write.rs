//! Write nodes: a device matrix copied into a pinned row on the stream,
//! then written to disk by a CPU node that feeds nothing, or a pinned row
//! written by that node alone (`docs/design.md`, "What it writes").

use std::path::PathBuf;

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::gpu_status;
use fb_core_wwr::handle::DeviceHandleStream;
use fb_io_npy::NpyArray;
use wwr_sys::wwr_memcpy_kind;

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

/// GPU: the matrix, `n²` doubles, copied into a pinned row on the stream.
struct Download {
    m: RowId<Device, f64>,
    row: String,
    n: usize,
}

impl Task for Download {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.m);
        host.output(&self.row, self.n * self.n)
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
        let (src, dst) = (device.get(self.m), host.take_mut(*pinned));
        // SAFETY: two rows of the same length, on the task's stream; a
        // stream task hands the pinned row to the copy and no more, and the
        // write that reads it runs after the executor waited for this
        // task's event.
        gpu_status(unsafe {
            wwr_sys::wwr_memcpy_async(
                dst.as_mut_ptr().cast(),
                src.as_ptr().cast(),
                size_of_val(dst),
                wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_HOST,
                stream.stream(),
            )
        })
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
                data: host.get(self.pinned).to_vec(),
            },
        )
    }
}
