//! The overlap's three fixtures, which both test binaries run: H2O in
//! STO-3G and cc-pVDZ, CO in cc-pVTZ (f on both atoms, the bond off every
//! axis), each an XYZ file and a compiled-in basis, with PySCF's H beside
//! them (`tests/data/README.md`); and the plan they run: the nuclei's upload
//! ([`add_nuclei`]), [`add_core_hamiltonian`] over it, and two tasks that
//! bring H to the host.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Planned, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::gpu_status;
use fb_core_wwr::handle::DeviceHandleStream;
use fb_integrals_one_electron_core_hamiltonian::add_core_hamiltonian;
use fb_molecule_nuclei::add_nuclei;
use wwr_sys::wwr_memcpy_kind;

/// Each fixture: its stem, naming its golden and its reference, its XYZ
/// file and its basis.
pub const FIXTURES: [(&str, &str, &str); 3] = [
    ("h2o_sto-3g", "h2o.xyz", "sto-3g"),
    ("h2o_cc-pvdz", "h2o.xyz", "cc-pvdz"),
    ("co_cc-pvtz", "co.xyz", "cc-pvtz"),
];

/// A file of `tests/data/`.
pub fn data(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(file)
}

/// A fixture planned: the nuclei's and H's tasks, then `download` (stream:
/// H copied into a pinned row) and `collect` (CPU: the pinned row copied
/// into `h`).
#[allow(dead_code, reason = "the plan binary reads the plan alone")]
pub struct Fixture {
    pub planned: Planned<'static>,
    pub n_ao: usize,
    pub h: Arc<Mutex<Vec<f64>>>,
}

/// The fixture's plan: the molecule read (charge and multiplicity do not
/// enter H) and the basis placed on it as the BasisSet step does.
pub fn plan(xyz: &str, basis: &str) -> Fixture {
    let molecule = fb_io_xyz::read_xyz(&data(xyz), 0, 1).expect("the XYZ file reads");
    let basis = fb_io_basis::resolve_basis(basis)
        .expect("a carried basis")
        .primitive_normalized()
        .place(&molecule)
        .expect("the basis covers the molecule");
    let n_ao = basis.nao_sphe() as usize;
    let mut dag = Dag::new();
    let nuclei = add_nuclei(&mut dag, 0, &molecule);
    let h =
        add_core_hamiltonian(&mut dag, 0, &basis, nuclei, &mut None).expect("the library loads");
    let pinned = dag.add(
        "download",
        0,
        Download {
            h,
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
    Fixture {
        planned: dag.plan(),
        n_ao,
        h: out,
    }
}

/// GPU: H, `len` doubles, copied into a pinned row on the stream.
struct Download {
    h: RowId<Device, f64>,
    len: usize,
}

impl Task for Download {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.h);
        host.output("H", self.len)
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
        let (src, dst) = (device.get(self.h), host.take_mut(*pinned));
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

/// CPU: the pinned copy of H read out, after the executor waited for the
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
