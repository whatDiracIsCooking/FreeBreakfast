//! The overlap's three fixtures, which both test binaries run: H2O in
//! STO-3G and cc-pVDZ, CO in cc-pVTZ (f on both atoms, the bond off every
//! axis), each an XYZ file and a compiled-in basis, read from
//! `integrals/one_electron/overlap/tests/data/` beside PySCF's S, with
//! PySCF's H in `integrals/one_electron/core_hamiltonian/tests/data/` and
//! the reference orbital energies in `tests/data/` here. Each is planned by
//! [`CoreRun`], the core Hamiltonian's orbitals as a whole run: the plan
//! the Roothaan step was first tested in, before the SCF loop took its
//! place in the crate, kept here as the step's own fixture.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2h};
use fb_core_error_handling::{Result, Status};
use fb_core_planner::{Dag, Planned, Task};
use fb_core_thread_pool::{DeviceStream, Lane, Record};
use fb_integrals_mmd::add_records;
use fb_integrals_one_electron_core_hamiltonian::add_core_hamiltonian;
use fb_integrals_one_electron_overlap::add_overlap;
use fb_molecule_nuclei::add_nuclei;
use fb_pipeline_scf::add_solve_fock;
use fb_recipes_orthogonalize::{ORTHOGONALIZE_EIGENVALUE_FLOOR, add_orthogonalize};

/// Each fixture: its stem, naming its golden and its references, its XYZ
/// file and its basis.
pub const FIXTURES: [(&str, &str, &str); 3] = [
    ("h2o_sto-3g", "h2o.xyz", "sto-3g"),
    ("h2o_cc-pvdz", "h2o.xyz", "cc-pvdz"),
    ("co_cc-pvtz", "co.xyz", "cc-pvtz"),
];

/// A file of `rust/<dir>`, relative to the workspace root.
pub fn data(dir: &str, file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(dir)
        .join(file)
}

/// A fixture's XYZ file.
pub fn xyz(file: &str) -> PathBuf {
    data("integrals/one_electron/overlap/tests/data", file)
}

/// The fixture's run, planned with `lwork` for syevd's work.
pub fn run(xyz_file: &str, basis: &str, lwork: impl FnOnce(usize) -> Result<usize>) -> CoreRun {
    CoreRun::new(&xyz(xyz_file), basis, lwork).expect("the fixture plans")
}

/// The orbitals of the core Hamiltonian, `H C = S C ε`, of one molecule in
/// one basis, planned once: the nuclei's upload ([`add_nuclei`]), the
/// records ([`add_records`]), S ([`add_overlap`]) and H
/// ([`add_core_hamiltonian`]) sharing that one build of the records and one
/// staging of the spherical transform, `X = S^(-1/2)` ([`add_orthogonalize`], at
/// the default floor), the step ([`add_solve_fock`]) with H for F, and two
/// tasks that bring ε and C to the host, `download` (stream: both copied
/// into pinned rows) and `collect` (CPU: the pinned rows copied out).
/// [`CoreRun::planned`] is the plan, its arena sizes among it, with no device;
/// [`CoreRun::run`] lays the arena out on a device and runs it on one stream.
pub struct CoreRun {
    planned: Planned<'static>,
    n_ao: usize,
    out: Arc<Mutex<(Vec<f64>, Vec<f64>)>>,
}

impl CoreRun {
    /// Plan the core Hamiltonian's orbitals of the molecule in the XYZ file
    /// `xyz` (Angstrom) in the basis `basis`, a compiled-in name or a BSE
    /// JSON path ([`fb_io_basis::resolve_basis`]), its primitives
    /// normalized and placed on the atoms as the BasisSet step does.
    /// `lwork` maps the number of spherical AOs `n` to syevd's work at `n`
    /// on the device the plan will run on,
    /// [`syevd_lwork`](fb_recipes_diagonalize::syevd_lwork) on a solver
    /// handle there; X and the step both diagonalize at `n`, so it is asked
    /// once. Touches no device unless `lwork` does.
    ///
    /// # Errors
    /// The readers' refusals: an XYZ file that cannot be read or parsed, a
    /// basis that is neither a carried name nor a readable document, an
    /// element the basis lacks; `lwork`'s error; and the kernel libraries'
    /// load checks ([`add_records`], [`add_orthogonalize`]).
    pub fn new(
        xyz: &Path,
        basis: &str,
        lwork: impl FnOnce(usize) -> Result<usize>,
    ) -> Result<CoreRun> {
        // Charge and multiplicity enter neither S nor H.
        let molecule = fb_io_xyz::read_xyz(xyz, 0, 1)?;
        let basis = fb_io_basis::resolve_basis(basis)?
            .primitive_normalized()
            .place(&molecule)?;
        let n_ao = basis.nao_sphe() as usize;
        let lwork = lwork(n_ao)?;
        let mut dag = Dag::new();
        dag.blas_workspace(0);
        let nuclei = add_nuclei(&mut dag, 0, &molecule);
        let mut u = None;
        let classes = add_records(&mut dag, 0, &basis)?;
        let s = add_overlap(&mut dag, 0, &basis, &classes, &mut u);
        let h = add_core_hamiltonian(&mut dag, 0, &basis, &classes, nuclei, &mut u)?;
        let x = add_orthogonalize(
            &mut dag,
            0,
            "X",
            n_ao,
            lwork,
            s,
            ORTHOGONALIZE_EIGENVALUE_FLOOR,
        )?;
        let (eps, c) = add_solve_fock(&mut dag, 0, n_ao, lwork, h, x);
        let pinned = dag.add("download", 0, Download { eps, c, n: n_ao });
        let out = Arc::new(Mutex::new((Vec::new(), Vec::new())));
        dag.add(
            "collect",
            0,
            Collect {
                pinned,
                out: Arc::clone(&out),
            },
        );
        Ok(CoreRun {
            planned: dag.plan(),
            n_ao,
            out,
        })
    }

    /// The plan: its tasks, and each worker's device and pinned tables.
    pub fn planned(&self) -> &Planned<'static> {
        &self.planned
    }

    /// The number of spherical AOs: ε has `n_ao` entries and C is
    /// `n_ao` × `n_ao`.
    pub fn n_ao(&self) -> usize {
        self.n_ao
    }

    /// Lay the arena out on `device` and run the plan once, on one stream.
    /// Returns the run's [`Record`], its result and every task's times, and
    /// `(ε, C)`: the `n_ao` orbital energies ascending and the orbitals,
    /// `n_ao²` doubles column-major, column `j` the orbital of `ε[j]`; both
    /// empty unless the result is `Ok`.
    pub fn run(self, device: i32) -> (Record, (Vec<f64>, Vec<f64>)) {
        let mut instance = self.planned.instantiate(&[device]);
        let mut backend = instance.backend();
        let record = instance.run(&mut backend, 1);
        let out = if record.result.is_ok() {
            std::mem::take(&mut *self.out.lock().unwrap_or_else(|e| e.into_inner()))
        } else {
            (Vec::new(), Vec::new())
        };
        (record, out)
    }
}

/// GPU: ε, `n` doubles, and C, `n²`, copied into pinned rows on the stream.
struct Download {
    eps: RowId<Device, f64>,
    c: RowId<Device, f64>,
    n: usize,
}

#[derive(Clone, Copy)]
struct HostRows {
    eps: RowId<Pinned, f64>,
    c: RowId<Pinned, f64>,
}

impl Task for Download {
    type Rows = HostRows;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> HostRows {
        device.read(self.eps);
        device.read(self.c);
        HostRows {
            eps: host.output("epsilon", self.n),
            c: host.output("C", self.n * self.n),
        }
    }

    fn run(
        &self,
        pinned: &HostRows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        for (src, dst) in [(self.eps, pinned.eps), (self.c, pinned.c)] {
            memcpy_d2p(host, dst, device.get(src), stream)?;
        }
        Ok(())
    }
}

/// CPU: the pinned copies of ε and C read out, after the executor waited
/// for the download.
struct Collect {
    pinned: HostRows,
    out: Arc<Mutex<(Vec<f64>, Vec<f64>)>>,
}

impl Task for Collect {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) {
        host.read(self.pinned.eps);
        host.read(self.pinned.c);
    }

    fn run(
        &self,
        _: &(),
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        *self.out.lock().unwrap_or_else(|e| e.into_inner()) = (
            memcpy_p2h(host, self.pinned.eps),
            memcpy_p2h(host, self.pinned.c),
        );
        Ok(())
    }
}
