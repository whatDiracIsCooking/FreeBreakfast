//! A whole run: an XYZ file and a basis name to S on the host.

use std::path::Path;
use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::{Result, Status};
use fb_core_planner::{Dag, Planned, Task};
use fb_core_thread_pool::{self as thread_pool, DeviceStream, Lane, Record};
use fb_core_wwr::gpu_status;
use fb_core_wwr::handle::DeviceHandleStream;
use fb_recipes_matrix_ao::add_blas_workspace;
use wwr_sys::wwr_memcpy_kind;

use fb_integrals_mmd::add_records;

use super::tasks::add_overlap;

/// The overlap of one molecule in one basis, planned once: the records
/// ([`add_records`]), the tasks of [`add_overlap`] reading them, and two
/// more that bring S to the host, `download` (stream: S copied into a
/// pinned row) and `collect` (CPU: the pinned row copied out).
/// [`Run::planned`] is the plan, its arena sizes among it, with no device;
/// [`Run::run`] lays the arena out on a device and runs it on one stream.
pub struct Run {
    planned: Planned<'static>,
    n_ao: usize,
    s: Arc<Mutex<Vec<f64>>>,
}

impl Run {
    /// Plan the overlap of the molecule in the XYZ file `xyz` (Angstrom) in
    /// the basis `basis`, a compiled-in name or a BSE JSON path
    /// ([`fb_io_basis::resolve_basis`]), its primitives normalized and
    /// placed on the atoms as the BasisSet step does. Touches no device.
    ///
    /// # Errors
    /// The readers' refusals: an XYZ file that cannot be read or parsed, a
    /// basis that is neither a carried name nor a readable document, an
    /// element the basis lacks; and the kernel library's load check
    /// ([`add_records`]), a `GpuRuntime` error naming the found and the
    /// expected ABI version and backend.
    pub fn new(xyz: &Path, basis: &str) -> Result<Run> {
        // Charge and multiplicity do not enter the overlap.
        let molecule = fb_io_xyz::read_xyz(xyz, 0, 1)?;
        let basis = fb_io_basis::resolve_basis(basis)?
            .primitive_normalized()
            .place(&molecule)?;
        let n_ao = basis.nao_sphe() as usize;
        let mut dag = Dag::new();
        add_blas_workspace(&mut dag, 0);
        let classes = add_records(&mut dag, 0, &basis)?;
        let s = add_overlap(&mut dag, 0, &basis, &classes, &mut None);
        let pinned = dag.add(
            "download",
            0,
            Download {
                s,
                len: n_ao * n_ao,
            },
        );
        let out = Arc::new(Mutex::new(Vec::new()));
        dag.add(
            "collect",
            0,
            Collect {
                pinned,
                out: Arc::clone(&out),
            },
        );
        Ok(Run {
            planned: dag.plan(),
            n_ao,
            s: out,
        })
    }

    /// The plan: its tasks, and each worker's device and pinned tables.
    pub fn planned(&self) -> &Planned<'static> {
        &self.planned
    }

    /// The number of spherical AOs: S is `n_ao` × `n_ao`.
    pub fn n_ao(&self) -> usize {
        self.n_ao
    }

    /// Lay the arena out on `device` and run the plan once, on one stream.
    /// Returns the run's [`Record`], its result and every task's times, and
    /// S, `n_ao²` doubles column-major, which is empty unless the result is
    /// `Ok`.
    pub fn run(self, device: i32) -> (Record, Vec<f64>) {
        let instance = self.planned.instantiate(&[device]);
        let backend = thread_pool::Device::new(&[device], instance.plan());
        let record = instance.run(&backend, 1);
        let s = if record.result.is_ok() {
            std::mem::take(&mut *self.s.lock().unwrap_or_else(|e| e.into_inner()))
        } else {
            Vec::new()
        };
        (record, s)
    }
}

/// GPU: S, `len` doubles, copied into a pinned row on the stream.
struct Download {
    s: RowId<Device, f64>,
    len: usize,
}

impl Task for Download {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.s);
        host.output("S", self.len)
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
        let (src, dst) = (device.get(self.s), host.take_mut(*pinned));
        // SAFETY: two rows of the same length, on the task's stream; a
        // stream task hands the pinned row to the copy and no more.
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

/// CPU: the pinned copy of S read out, after the executor waited for the
/// download.
struct Collect {
    pinned: RowId<Pinned, f64>,
    out: Arc<Mutex<Vec<f64>>>,
}

impl Task for Collect {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) {
        host.read(self.pinned);
    }

    fn run(
        &self,
        _: &(),
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        *self.out.lock().unwrap_or_else(|e| e.into_inner()) = host.get(self.pinned).to_vec();
        Ok(())
    }
}
