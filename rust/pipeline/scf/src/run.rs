//! A whole run: a molecule, a basis and a fitting set to the converged
//! DF-RHF or DF-ROHF energy, orbitals and density on the host.

use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, Heap, Layout, Pinned, RowId, memcpy_p2h};
use fb_core_error_handling::{Result, Status};
use fb_core_planner::{CpuTask, Dag, Planned};
use fb_core_thread_pool::Record;
use fb_integrals_mmd::add_records;
use fb_integrals_one_electron_core_hamiltonian::add_core_hamiltonian;
use fb_integrals_one_electron_overlap::add_overlap;
use fb_molecule::Molecule;
use fb_molecule_nuclei::add_nuclei;
use fb_pipeline_eri::add_df_eri;
use fb_pipeline_guess::{add_guess_upload, add_sad_guess};
use fb_recipes_orthogonalize::{ORTHOGONALIZE_EIGENVALUE_FLOOR, add_orthogonalize};

use super::iteration::occupied_orbitals;
use super::scf::{ScfStep, add_scf};

/// What a converged run brings back: the C++ `ScfOutput`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScfOutput {
    /// The total energy, electronic plus nuclear repulsion, in hartree:
    /// that of the density the last Fock matrix was built from.
    pub energy: f64,
    /// The residual's norm `‖Xᵀ (F D S - S D F) X‖_F` at that build.
    pub residual: f64,
    /// The Fock builds, the loop's iterations.
    pub iterations: usize,
    /// The occupied orbitals in alpha: C's first `n_alpha` columns.
    pub n_alpha: usize,
    /// The occupied in beta, C's first `n_beta` columns, the doubly
    /// occupied ones; `n_alpha` for a closed shell.
    pub n_beta: usize,
    /// The `n` orbital energies, ascending; an open shell's by occupation
    /// class, its open ones in their columns ([`add_scf`]).
    pub epsilon: Vec<f64>,
    /// The orbitals, `n²` doubles column-major, column `j` the orbital of
    /// `epsilon[j]`.
    pub c: Vec<f64>,
    /// The spin-summed density, `n²` doubles column-major:
    /// `D = 2 C_occ C_occᵀ`, or an open shell's `Da + Db`.
    pub d: Vec<f64>,
}

/// The DF-RHF or DF-ROHF of one molecule in one basis over one fitting
/// set, from the SAD guess, planned once: the nuclei ([`add_nuclei`]), the records
/// ([`add_records`]), S ([`add_overlap`]), H ([`add_core_hamiltonian`])
/// and the fitted integrals ([`add_df_eri`]) over one build of the records
/// and one staging of the orbital basis's spherical transform,
/// `X = S^(-1/2)` ([`add_orthogonalize`], at the default floor), the SAD
/// guess and its upload
/// ([`add_sad_guess`], [`add_guess_upload`]), the loop ([`add_scf`]), and
/// the nodes that bring the products to the host, `download epsilon`,
/// `download C` and `download D` (the planner's copy nodes,
/// [`Dag::download`]) and `collect` (CPU: the pinned rows copied out). [`Run::planned`] is the plan, its arena sizes among it,
/// with no device; [`Run::run`] lays the arena out on a device and runs it
/// on one stream.
pub struct Run {
    planned: Planned<'static>,
    n_ao: usize,
    out: Arc<Mutex<Option<ScfOutput>>>,
}

impl Run {
    /// Plan the DF-RHF of `molecule`, or its DF-ROHF when its multiplicity
    /// is above 1, in the basis `basis`, a compiled-in name or a BSE JSON
    /// path ([`fb_io_basis::resolve_basis`]), over the compiled-in fitting
    /// set `aux_basis`, with the options `step`. `lwork` maps the number of
    /// spherical AOs `n` to syevd's work at `n` on the device the plan will
    /// run on, [`syevd_lwork`](fb_recipes_diagonalize::syevd_lwork) on a
    /// solver handle there; every eigensolve is at `n` or under, so it is
    /// asked once. Touches no device unless `lwork` does.
    ///
    /// # Errors
    /// The readers' refusals: a basis that is neither a carried name nor a
    /// readable document, an element the basis or the fitting set lacks;
    /// `lwork`'s error; and the planning errors of [`add_records`],
    /// [`add_orthogonalize`], [`add_df_eri`], [`add_sad_guess`] and
    /// [`add_scf`], an electron count of the wrong parity for the
    /// multiplicity and an option out of range among them.
    pub fn new(
        molecule: &Molecule,
        basis: &str,
        aux_basis: &str,
        step: &ScfStep,
        lwork: impl FnOnce(usize) -> Result<usize>,
    ) -> Result<Run> {
        let (n_alpha, n_beta) = occupied_orbitals(molecule)?;
        let basis = fb_io_basis::resolve_basis(basis)?
            .primitive_normalized()
            .place(molecule)?;
        let n_ao = basis.nao_sphe() as usize;
        let lwork = lwork(n_ao)?;
        let mut dag = Dag::new();
        dag.blas_workspace(0);
        let nuclei = add_nuclei(&mut dag, 0, molecule);
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
        let eri = add_df_eri(&mut dag, 0, molecule, &basis, &classes, &mut u, aux_basis)?;
        let guess = add_sad_guess(&mut dag, 0, &basis, molecule.atoms())?;
        let guess = add_guess_upload(&mut dag, 0, &basis, guess);
        let (scalars, [eps, c, d]) = add_scf(
            &mut dag,
            0,
            step,
            molecule,
            n_ao,
            lwork,
            &eri,
            [h, s, x],
            guess,
        )?;
        let pinned = [
            dag.download(0, eps, "epsilon", n_ao),
            dag.download(0, c, "C", n_ao * n_ao),
            dag.download(0, d, "D", n_ao * n_ao),
        ];
        let out = Arc::new(Mutex::new(None));
        dag.add_cpu(
            "collect",
            0,
            Collect {
                scalars,
                pinned,
                occupied: [n_alpha, n_beta],
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

/// CPU: the last build's `[E, ‖e‖, builds]`, a heap row, and the pinned
/// copies read out, after the executor waited for the download. The copies
/// are the run's exit, so a run's one allocation (`docs/memory.md`, "The
/// three kinds").
struct Collect {
    scalars: RowId<Heap, f64>,
    pinned: [RowId<Pinned, f64>; 3],
    occupied: [usize; 2],
    out: Arc<Mutex<Option<ScfOutput>>>,
}

impl CpuTask for Collect {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) {
        heap.read(self.scalars);
        for &row in &self.pinned {
            host.read(row);
        }
    }

    fn run(&self, _: &(), host: &Bufs<'_, Pinned>, heap: &Bufs<'_, Heap>) -> Status {
        let scalars = heap.get(self.scalars);
        let [epsilon, c, d] = self.pinned.map(|row| memcpy_p2h(host, row));
        *self.out.lock().unwrap_or_else(|e| e.into_inner()) = Some(ScfOutput {
            energy: scalars[0],
            residual: scalars[1],
            // A whole number the decision wrote.
            iterations: scalars[2] as usize,
            n_alpha: self.occupied[0],
            n_beta: self.occupied[1],
            epsilon,
            c,
            d,
        });
        Ok(())
    }
}
