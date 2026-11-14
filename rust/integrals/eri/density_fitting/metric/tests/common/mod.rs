//! The two fixtures both test binaries run, each an XYZ file and a
//! compiled-in fitting set with PySCF's `int2c2e` beside them
//! (`tests/data/README.md`): water in cc-pVDZ-JKFIT (aux f, every shell one
//! primitive), and iron beside water in def2-universal-JKFIT (aux h and i on
//! Fe, contracted shells on every atom, the Fe–O bond off every axis); and
//! the plan they run: [`add_metric`], and two tasks that bring M to the host.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2p};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Planned, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_integrals_eri_density_fitting_metric::add_metric;

/// Each fixture: its stem, naming its golden and its reference, its XYZ
/// file, its fitting set and the largest momentum that set reaches on it.
pub const FIXTURES: [(&str, &str, &str, i32); 2] = [
    ("h2o_cc-pvdz-jkfit", "h2o.xyz", "cc-pvdz-jkfit", 3),
    (
        "feh2o_def2-universal-jkfit",
        "feh2o.xyz",
        "def2-universal-jkfit",
        6,
    ),
];

/// A file of `tests/data/`.
pub fn data(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(file)
}

/// A fixture planned: M's tasks, then `download` (stream: M copied into a
/// pinned row) and `collect` (CPU: the pinned row copied into `m`).
#[allow(dead_code, reason = "the plan binary reads the plan alone")]
pub struct Fixture {
    pub planned: Planned<'static>,
    pub n_aux: usize,
    pub l_max: i32,
    pub m: Arc<Mutex<Vec<f64>>>,
}

/// The fixture's plan: the molecule read (charge and multiplicity do not
/// enter M) and the fitting set placed on it as the DF steps will place it.
pub fn plan(xyz: &str, aux: &str) -> Fixture {
    let molecule = fb_io_xyz::read_xyz(&data(xyz), 0, 1).expect("the XYZ file reads");
    let aux = fb_io_basis::resolve_aux_basis(aux)
        .expect("a carried fitting set")
        .primitive_normalized()
        .place(&molecule)
        .expect("the fitting set covers the molecule");
    let n_aux = aux.nao_sphe() as usize;
    let l_max = aux
        .shells()
        .iter()
        .map(fb_basis::Shell::angular_momentum)
        .max()
        .expect("a shell");
    let mut dag = Dag::new();
    let m = add_metric(&mut dag, 0, &aux, &mut None).expect("the library loads");
    let pinned = dag.add(
        "download",
        0,
        Download {
            m,
            len: n_aux * n_aux,
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
        n_aux,
        l_max,
        m: out,
    }
}

/// GPU: M, `len` doubles, copied into a pinned row on the stream.
struct Download {
    m: RowId<Device, f64>,
    len: usize,
}

impl Task for Download {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.m);
        host.output("M", self.len)
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
        memcpy_d2p(host, *pinned, device.get(self.m), stream)
    }
}

/// CPU: the pinned copy of M read out, after the executor waited for the
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
