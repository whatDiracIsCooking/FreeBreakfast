//! A whole run: an XYZ file, a basis and a fitting set to the converged
//! DF-RHF energy, orbitals and density on the host.

use std::path::Path;
use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2h};
use fb_core_error_handling::{Result, Status};
use fb_core_planner::{Dag, Planned, Task};
use fb_core_thread_pool::{DeviceStream, Lane, Record};
use fb_integrals_mmd::add_records;
use fb_integrals_one_electron_core_hamiltonian::add_core_hamiltonian;
use fb_integrals_one_electron_overlap::add_overlap;
use fb_molecule_nuclei::add_nuclei;
use fb_pipeline_eri::add_df_eri;
use fb_pipeline_guess::{add_guess_upload, add_sad_guess};
use fb_recipes_orthogonalize::{ORTHOGONALIZE_EIGENVALUE_FLOOR, add_orthogonalize};

use super::scf::{ScfStep, add_scf};

/// What a converged run brings back: the C++ `ScfOutput`'s closed-shell
/// fields.
#[derive(Clone, Debug, PartialEq)]
pub struct ScfOutput {
    /// The total energy, electronic plus nuclear repulsion, in hartree:
    /// that of the density the last Fock matrix was built from.
    pub energy: f64,
    /// The residual's norm `‖Xᵀ (F D S - S D F) X‖_F` at that build.
    pub residual: f64,
    /// The Fock builds, the loop's iterations.
    pub iterations: usize,
    /// The `n` orbital energies, ascending.
    pub epsilon: Vec<f64>,
    /// The orbitals, `n²` doubles column-major, column `j` the orbital of
    /// `epsilon[j]`.
    pub c: Vec<f64>,
    /// `D = 2 C_occ C_occᵀ`, `n²` doubles column-major.
    pub d: Vec<f64>,
}

/// The DF-RHF of one molecule in one basis over one fitting set, from the
/// SAD guess, planned once: the nuclei ([`add_nuclei`]), the records
/// ([`add_records`]), S ([`add_overlap`]) and H ([`add_core_hamiltonian`])
/// over one build of the records and one staging of the spherical
/// transform, `X = S^(-1/2)` ([`add_orthogonalize`], at the default floor),
/// the fitted integrals ([`add_df_eri`]), the SAD guess and its upload
/// ([`add_sad_guess`], [`add_guess_upload`]), the loop ([`add_scf`]), and
/// two tasks that bring the products to the host, `download` (stream: ε,
/// C and D copied into pinned rows) and `collect` (CPU: the pinned rows
/// copied out). [`Run::planned`] is the plan, its arena sizes among it,
/// with no device; [`Run::run`] lays the arena out on a device and runs it
/// on one stream.
pub struct Run {
    planned: Planned<'static>,
    n_ao: usize,
    out: Arc<Mutex<Option<ScfOutput>>>,
}

impl Run {
    /// Plan the DF-RHF of the neutral singlet in the XYZ file `xyz`
    /// (Angstrom) in the basis `basis`, a compiled-in name or a BSE JSON
    /// path ([`fb_io_basis::resolve_basis`]), over the compiled-in fitting
    /// set `aux_basis`, with the options `step`. `lwork` maps the number of
    /// spherical AOs `n` to syevd's work at `n` on the device the plan will
    /// run on, [`syevd_lwork`](fb_recipes_diagonalize::syevd_lwork) on a
    /// solver handle there; every eigensolve is at `n`, so it is asked once.
    /// Touches no device unless `lwork` does.
    ///
    /// # Errors
    /// The readers' refusals: an XYZ file that cannot be read or parsed, a
    /// basis that is neither a carried name nor a readable document, an
    /// element the basis or the fitting set lacks; `lwork`'s error; and the
    /// planning errors of [`add_records`], [`add_orthogonalize`],
    /// [`add_df_eri`], [`add_sad_guess`] and [`add_scf`], an odd electron
    /// count and an option out of range among them.
    pub fn new(
        xyz: &Path,
        basis: &str,
        aux_basis: &str,
        step: &ScfStep,
        lwork: impl FnOnce(usize) -> Result<usize>,
    ) -> Result<Run> {
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
        let eri = add_df_eri(&mut dag, 0, &molecule, &basis, &classes, aux_basis)?;
        let guess = add_sad_guess(&mut dag, 0, &basis, molecule.atoms())?;
        let guess = add_guess_upload(&mut dag, 0, &basis, guess);
        let (scalars, [eps, c, d]) = add_scf(
            &mut dag,
            0,
            step,
            &molecule,
            n_ao,
            lwork,
            &eri,
            [h, s, x],
            guess,
        )?;
        let pinned = dag.add(
            "download",
            0,
            Download {
                rows: [(eps, n_ao), (c, n_ao * n_ao), (d, n_ao * n_ao)],
            },
        );
        let out = Arc::new(Mutex::new(None));
        dag.add(
            "collect",
            0,
            Collect {
                scalars,
                pinned,
                out: Arc::clone(&out),
            },
        );
        Ok(Run {
            planned: dag.plan(),
            n_ao,
            out,
        })
    }

    /// The plan: its tasks, and each worker's device and pinned tables.
    pub fn planned(&self) -> &Planned<'static> {
        &self.planned
    }

    /// The number of spherical AOs: ε has `n_ao` entries, C and D
    /// `n_ao²`.
    pub fn n_ao(&self) -> usize {
        self.n_ao
    }

    /// Lay the arena out on `device` and run the plan once, on one stream.
    /// Returns the run's [`Record`], its result and every task's times, and
    /// the products, `None` unless the result is `Ok`.
    pub fn run(self, device: i32) -> (Record, Option<ScfOutput>) {
        let mut instance = self.planned.instantiate(&[device]);
        let mut backend = instance.backend();
        let record = instance.run(&mut backend, 1);
        let mut out = None;
        if record.result.is_ok() {
            out = self.out.lock().unwrap_or_else(|e| e.into_inner()).take();
        }
        (record, out)
    }
}

/// GPU: ε, C and D, each with its length, copied into pinned rows.
struct Download {
    rows: [(RowId<Device, f64>, usize); 3],
}

impl Task for Download {
    type Rows = [RowId<Pinned, f64>; 3];
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        let names = ["epsilon", "C", "D"];
        let mut pinned = [None; 3];
        for (slot, (&(row, len), name)) in pinned.iter_mut().zip(self.rows.iter().zip(names)) {
            device.read(row);
            *slot = Some(host.output(name, len));
        }
        pinned.map(|p| p.expect("each row declared"))
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
        for (&(src, _), &dst) in self.rows.iter().zip(pinned) {
            memcpy_d2p(host, dst, device.get(src), stream)?;
        }
        Ok(())
    }
}

/// CPU: the last build's `[E, ‖e‖, builds]` and the pinned copies read out,
/// after the executor waited for the download.
struct Collect {
    scalars: RowId<Pinned, f64>,
    pinned: [RowId<Pinned, f64>; 3],
    out: Arc<Mutex<Option<ScfOutput>>>,
}

impl Task for Collect {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) {
        host.read(self.scalars);
        for &row in &self.pinned {
            host.read(row);
        }
    }

    fn run(
        &self,
        _: &(),
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        let scalars = host.get(self.scalars);
        let [epsilon, c, d] = self.pinned.map(|row| memcpy_p2h(host, row));
        *self.out.lock().unwrap_or_else(|e| e.into_inner()) = Some(ScfOutput {
            energy: scalars[0],
            residual: scalars[1],
            // A whole number the decision wrote.
            iterations: scalars[2] as usize,
            epsilon,
            c,
            d,
        });
        Ok(())
    }
}
