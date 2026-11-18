//! The DF-RHF and DF-ROHF loop, and the RHF over the conventional
//! integrals: one iteration's tasks as a loop's body, replayed until its
//! decision finds the energy and the residual converged.

use std::ops::ControlFlow;

use contracts::requires;
use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_p2d};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_log::fits_i32;
use fb_core_planner::{CpuTask, Dag, Decision, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::blas::{dgmm, geam};
use fb_core_wwr::copy;
use fb_core_wwr::device_slice::split_at_mut;
use fb_core_wwr::memset;
use fb_integrals_eri_conventional::{add_conventional_jk, add_dense_jk};
use fb_integrals_eri_density_fitting_jk::{Contraction, add_streamed_jk};
use fb_molecule::{Molecule, nuclear_repulsion_energy};
use fb_pipeline_eri::Eri;
use fb_recipes_diagonalize::add_diagonalize;
use fb_recipes_diis::{DIIS_MAX_HISTORY_LIMIT, Diis, add_diis};
use fb_recipes_sys::inverse_sqrt;
use serde::Deserialize;
use wwr_sys::wwr_blas_operation::WWR_BLAS_OP_N as OP_N;
use wwr_sys::wwr_blas_side_mode::{WWR_BLAS_SIDE_LEFT as LEFT, WWR_BLAS_SIDE_RIGHT as RIGHT};

use super::iteration::{add_density, add_fock, add_published_density, occupied_orbitals};
use super::open::{add_open_density, add_open_fock, add_publish_open, add_spin_densities};
use super::tasks::add_solve_fock;

/// The C++ `kDensityFactorFloor`: an eigenvalue of the guess density at or
/// under this fraction of its largest is dropped from the guess's factor,
/// and one below minus this fraction is warned of. What it drops is the
/// eigensolver's round-off on a positive semidefinite D, about `n ε` of the
/// largest eigenvalue, so the floor is relative to it, as in the C++.
const GUESS_FLOOR: f64 = 1e-12;

/// The closed-shell SCF's options: the C++ `ScfStep`'s keys that the
/// density-fitted closed-shell loop reads, each at the C++ default
/// ([`ScfStep::default`]).
///
/// It is also the config's `[scf]` table: each field a key, every key
/// optional at its default, and any other key refused by name.
///
/// The C++ `fock_rebuild_interval` and `tau_j` are the integral-direct
/// builder's, which every DF build ignores, and `molden` a product writer:
/// none is ported.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ScfStep {
    /// The loop's bound: the most Fock builds before the run fails with
    /// `MaxIterationsExceeded`. Default 100; at least 1.
    pub max_iterations: usize,
    /// `|ΔE|` between two Fock builds under which the energy is converged,
    /// in hartree. Default 1e-10; non-negative.
    pub energy_tolerance: f64,
    /// `‖Xᵀ (F D S - S D F) X‖_F` under which the residual is converged.
    /// Default 1e-7; non-negative.
    pub commutator_tolerance: f64,
    /// The most (F, e) pairs DIIS extrapolates over. Default 8; 1 to
    /// [`DIIS_MAX_HISTORY_LIMIT`].
    pub diis_history: usize,
}

impl Default for ScfStep {
    fn default() -> ScfStep {
        ScfStep {
            max_iterations: 100,
            energy_tolerance: 1e-10,
            commutator_tolerance: 1e-7,
            diis_history: 8,
        }
    }
}

impl ScfStep {
    /// Check each option against its range, as the C++ step does; [`add_scf`]
    /// checks them too, so a reader of a config can refuse them before
    /// anything is planned.
    ///
    /// # Errors
    /// `InvalidConfig` naming the first option out of range.
    pub fn check(&self) -> Status {
        if self.max_iterations < 1 {
            return Err(Error::config(format!(
                "[scf] max_iterations is {}, below 1",
                self.max_iterations
            )));
        }
        let refused = |t: f64| t.is_nan() || t < 0.0;
        if refused(self.energy_tolerance) || refused(self.commutator_tolerance) {
            return Err(Error::config(format!(
                "[scf] energy_tolerance ({}) and commutator_tolerance ({}) must be non-negative",
                self.energy_tolerance, self.commutator_tolerance
            )));
        }
        if !(1..=DIIS_MAX_HISTORY_LIMIT).contains(&self.diis_history) {
            return Err(Error::config(format!(
                "[scf] diis_history is {}, outside 1..{DIIS_MAX_HISTORY_LIMIT}",
                self.diis_history
            )));
        }
        Ok(())
    }
}

/// Add the DF-RHF of `molecule` to `dag` on GPU worker `worker`, or its
/// DF-ROHF when its multiplicity is above 1 ([`occupied_orbitals`]), from
/// the guess density `guess`, and return `(scalars, [ε, C, D])`: a heap
/// row of three doubles, `[E, ‖e‖_F, builds]` of the last Fock build (E in
/// hartree, electronic plus nuclear repulsion; `builds` the loop's
/// iterations, a whole number), and device output rows of the `n` orbital
/// energies ascending, the orbitals (`n²` doubles, column-major, column
/// `j` the orbital of `ε[j]`) and `D = 2 C_occ C_occᵀ` (`n²`), all of the
/// converged Fock matrix as built; an open shell's are below.
///
/// `h`, `s` and `x` are earlier tasks' outputs of `n²` doubles: the core
/// Hamiltonian, the overlap and `X = S^(-1/2)`. `guess` is the guess
/// density, `n²` doubles, symmetric positive semidefinite (a SAD guess;
/// `fb_pipeline_guess::add_guess_upload`'s row); the loop carries D in it,
/// so it holds the last iteration's density after the run, unless it is
/// rescaled (below). `eri` is the two-electron integrals of the same basis
/// (`fb_pipeline_eri::add_eri`): the density-fitted packed B on the host,
/// whose J and K are one streamed pass over it per build
/// ([`add_streamed_jk`]: `df jk pack`, the aux batches, `df jk finish`),
/// or the conventional
/// tensors, over which each build is one `conventional jk` of the carried
/// D ([`add_conventional_jk`]), closed shell only; or the dense row blocks
/// over the run's GPU workers, over which each build is
/// [`add_dense_jk`]'s tasks (`dense density`, the broadcast, a `dense jk`
/// per worker, the two gathers, `dense spherical`), the loop's body then
/// spanning every worker, and `worker` 0. K over the tensors is
/// of D itself, so a conventional plan builds no rest's K and adds no
/// `guess exchange`: the first build's K is the guess's whole.
/// `lwork` is [`syevd_lwork`](fb_recipes_diagonalize::syevd_lwork) at `n`.
///
/// **Before the loop**, when the molecule's electron count `N` is not
/// `Σ Z` (an ion), **`guess rescale`** (stream): the guess scaled by
/// `N / Σ Z` into the row `"D rescaled"`, which the loop carries in its
/// place, since SAD places `Σ Z` electrons whatever the charge. The guess
/// enters as its factor, as the C++ first build does, since K is built
/// from a factor of D, never from D: the
/// eigendecomposition `D = U diag(w) Uᵀ` ([`add_diagonalize`] on the rows
/// `"guess U"`, `"guess w"`); **`download guess w host`**, the planner's
/// copy node ([`Dag::download`]), `w` copied to the pinned row
/// `"guess w host"`; **`guess check`** (CPU), which reads it
/// and writes the floor, 1e-12 of the largest eigenvalue (0 if none is
/// positive), to the pinned row `"guess floor"`, and warns of an eigenvalue
/// below minus that floor, possible only for a guess loaded from a file,
/// as in the C++; then **`guess factor`** (stream), which copies the floor
/// to the device and has the kernel read it there: `C̃ = U diag((w/2)^(1/2))`, an eigenvalue at or under the
/// floor a zero column, so `D = 2 C̃ C̃ᵀ` to round-off and a negative part
/// of D is left out of the first K, as in the C++. The guess's rank is
/// about its minimal basis, above `nocc`, but the loop's factor row is
/// `n × nocc`, the width of every later K. So C̃'s `nocc` columns of the
/// largest eigenvalues become the carried factor `"C occ"`, and the other
/// `n - nocc` the row `"C rest"`, whose K (a streamed pass, once) the
/// first iteration adds to its own: the C++ takes the same factor `nocc`
/// columns at a time and sums their K. Then `diis history` ([`add_diis`],
/// the rings of `"DIIS F"`), and **`energy before`** (CPU): the decision's
/// carried row `"last build"`, its energy 0 as the C++ starts it, on the
/// heap, since it is host state no copy engine meets (`docs/memory.md`,
/// "The three kinds").
///
/// **The body**, one iteration: one streamed pass ([`add_streamed_jk`]),
/// J of D and K of the factor; **`guess exchange`** (stream), when `nocc < n`: K plus
/// the rest's K into `"K of D"`, then the rest's K zeroed, so only the first
/// iteration has it; [`add_fock`]'s `fock`, `residual` and `energy and
/// norm`; `diis push` ([`Diis::add_push`]) of (F, e); [`add_solve_fock`] on
/// the extrapolated F; [`add_density`], which writes D and the factor for
/// the next iteration. **The decision**, `converged` (CPU): the build's E
/// and `‖e‖` and the iteration count written into `"last build"`, the row
/// returned as `scalars`; converged when `|ΔE| < energy_tolerance` and
/// `‖e‖ < commutator_tolerance`, from the second iteration on; otherwise a
/// singular DIIS extrapolation restarts the history from this iteration's
/// pair, every residual from it on scaled by `1/‖e‖` ([`Diis::restart`]),
/// as the C++ loop does. Each iteration is logged at `info`.
///
/// **After the loop**, the published orbitals are the converged F's as
/// built, not of its extrapolation, as in the C++: [`add_solve_fock`] on
/// F, then **`published density`** (stream), D from C's first `nocc`
/// columns by one gemm.
///
/// **An open shell** (`n_alpha > n_beta`) changes the C++ loop's steps 1,
/// 2 and 6, as its README's open shell paragraph has them; `nocc` above is
/// `n_alpha`. Before the loop, C̃'s last `n_alpha` columns are the carried
/// alpha factor `"C a"` and the rest `"C a rest"`, its last `n_beta` the
/// carried beta factor `"C b"` and the rest `"C b rest"`, each rest's K
/// built once, both by one streamed pass; **`spin densities`**
/// (stream), `Da = Db = D/2` into the
/// carried `"Da"` and `"Db"`; and **`identity`** (CPU) with its upload,
/// `"I"`. In the body, a second K in the pass, of `"C b"`, each K with its
/// rest's added by a `guess exchange` (`"Ka of D"`, `"Kb of D"`). With no
/// beta electron (H, H₂⁺) there is no `"C b"` and no second K:
/// the beta `guess exchange` copies the rest's K alone, the guess's
/// `Db = D/2`, into the first build's Kb, and every later Kb is 0, as
/// `open density` zeroes Db, the C++ `K(Db) = 0`. Then in
/// place of `fock`, **`open fock`** (`Fa = H + J - Ka/2`,
/// `Fb = H + J - Kb/2`, `E = tr(D H + Da Fa + Db Fb)/2 + E_nuc`) and
/// **`roothaan fock`** (the Roothaan effective F, its open block
/// `(Fa + 3 Fb)/4`, not PySCF's `(Fa + Fb)/2`; the C++ tree's issue #476),
/// with `residual` and `energy and norm` on `(F, D)`, and **`built
/// densities`**, Da and Db kept past the loop; DIIS and the solve
/// unchanged; and in place of `density`, **`open density`**, which writes
/// the factors, `Da = Ca Caᵀ`, `Db = Cb Cbᵀ` and `D = Da + Db`. After the
/// loop, the C++ `publish_open`: F's canonical orbitals with the open ones,
/// columns `[n_beta, n_alpha)`, rotated among themselves to diagonalize
/// `Fc = (Fa + Fb)/2` there (**`open block`**, the eigensolver on it,
/// **`canonical open`**), whose energies are PySCF's ROHF `mo_energy` by
/// occupation class; and the canonical orbitals of PySCF's own Roothaan
/// matrix at the last build's densities (**`pyscf roothaan fock`** and the
/// Roothaan step on it). When the first set's energies ascend (**`open
/// order`**, on the CPU) the second is published, otherwise the first, its
/// open energies in their columns, by **`publish open`**, with D their
/// `Da + Db`. A closed shell adds none of it.
///
/// # Errors
/// At planning, before anything is added: `InvalidConfig` for an option
/// out of range ([`ScfStep`]), an electron count of the wrong parity for
/// the multiplicity ([`occupied_orbitals`]) or more alpha electrons than
/// the `n` orbitals; `Unsupported` for an open shell over the conventional
/// integrals.
/// Then the kernel library's load check
/// ([`add_diis`], [`add_streamed_jk`]), as `GpuRuntime`; `dag` may hold
/// some of the tasks after it. At run time, the tasks' errors, and
/// `MaxIterationsExceeded` when the loop has not converged in
/// `max_iterations` builds.
///
/// # Panics
/// As [`add_streamed_jk`]; or as [`Dag::add_gpu`], if a row is of another
/// worker's table.
#[allow(
    clippy::too_many_arguments,
    reason = "the step's inputs, each a row or a product of its own"
)]
#[allow(
    clippy::type_complexity,
    reason = "the step's four products as rows, in the crate's tuple style"
)]
#[requires(fits_i32!(n.checked_mul(n)), "n² fits the BLAS's i32")]
#[track_caller]
pub fn add_scf(
    dag: &mut Dag<'_>,
    worker: usize,
    step: &ScfStep,
    molecule: &Molecule,
    n: usize,
    lwork: usize,
    eri: &Eri,
    [h, s, x]: [RowId<Device, f64>; 3],
    guess: RowId<Device, f64>,
) -> Result<(RowId<Heap, f64>, [RowId<Device, f64>; 3])> {
    step.check()?;
    let (n_alpha, n_beta) = occupied_orbitals(molecule)?;
    let electrons = molecule.electron_count();
    if n_alpha > n {
        return Err(Error::config(format!(
            "scf: {electrons} electrons at multiplicity {} do not fit {n} spatial orbitals",
            molecule.multiplicity()
        )));
    }
    let open = n_alpha != n_beta;
    if open && matches!(eri, Eri::Conventional(_) | Eri::Dense(_)) {
        return Err(Error::unsupported(format!(
            "scf: multiplicity {} is an open shell, but [eri] mode = \"conventional\" builds \
             the closed-shell RHF alone",
            molecule.multiplicity()
        )));
    }
    let nocc = n_alpha;
    let e_nuc = nuclear_repulsion_energy(molecule);

    // Before the loop: the guess rescaled to the electron count, its factor,
    // the rest's K, an open shell's spin densities, DIIS's rings and the
    // decision's last energy.
    let z = molecule.total_nuclear_charge();
    let guess = if z != electrons && z > 0 {
        let scale = f64::from(electrons) / f64::from(z);
        dag.add_gpu("guess rescale", worker, GuessRescale { d: guess, scale, n })
    } else {
        guess
    };
    let (u, w) = add_diagonalize(dag, worker, "guess", n, lwork, guess);
    let w_host = dag.download(worker, w, "guess w host", n);
    let floor = dag.add_cpu("guess check", worker, GuessCheck { w: w_host });
    let factor = dag.add_gpu(
        "guess factor",
        worker,
        GuessFactor {
            u,
            w,
            floor,
            n,
            nocc,
            n_beta: open.then_some(n_beta),
        },
    );
    // The rests' K, the density-fitted plan's alone (K over the stored
    // tensors is of D itself): one streamed pass over B for both.
    let df = match eri {
        Eri::DensityFitted(df) => Some(df),
        Eri::Conventional(_) | Eri::Dense(_) => None,
    };
    let (rest_k, beta) = match df {
        Some(df) => {
            let rests: Vec<Contraction> = [(factor.rest, n - nocc), (factor.beta_rest, n - n_beta)]
                .into_iter()
                .filter_map(|(rest, nocc)| rest.map(|c| Contraction::Exchange { c, nocc }))
                .collect();
            let mut ks = if rests.is_empty() {
                Vec::new()
            } else {
                add_streamed_jk(
                    dag,
                    worker,
                    df.map.sphe(),
                    df.tables,
                    df.aux.nao_sphe() as usize,
                    df.b,
                    &rests,
                )?
            }
            .into_iter();
            let rest_k = factor.rest.and_then(|_| ks.next());
            let beta = factor
                .beta_rest
                .and_then(|_| ks.next())
                .map(|k| (factor.cb, k));
            (rest_k, beta)
        }
        None => (None, None),
    };
    let spin = open.then(|| add_spin_densities(dag, worker, n, guess));
    let diis = add_diis(dag, worker, "DIIS F", n * n, step.diis_history)?;
    let previous = dag.add_cpu("energy before", worker, EnergyBefore);

    // The body: one iteration.
    let body = dag.len();
    // A density-fitted plan's J, K and an open shell's beta K: one
    // streamed pass over B.
    let mut kb = None;
    let (j, mut k) = match eri {
        Eri::DensityFitted(df) => {
            let mut set = vec![
                Contraction::Coulomb(guess),
                Contraction::Exchange {
                    c: factor.c_occ,
                    nocc,
                },
            ];
            if let Some(cb) = factor.cb {
                set.push(Contraction::Exchange {
                    c: cb,
                    nocc: n_beta,
                });
            }
            let jk = add_streamed_jk(
                dag,
                worker,
                df.map.sphe(),
                df.tables,
                df.aux.nao_sphe() as usize,
                df.b,
                &set,
            )?;
            kb = jk.get(2).copied();
            (jk[0], jk[1])
        }
        Eri::Conventional(c) => add_conventional_jk(dag, worker, c, guess),
        Eri::Dense(d) => add_dense_jk(dag, d, guess),
    };
    if let Some(rest) = rest_k {
        let name = if open { "Ka of D" } else { "K of D" };
        let guess_exchange = GuessExchange {
            k: Some(k),
            rest,
            n,
            name,
        };
        k = dag.add_gpu("guess exchange", worker, guess_exchange);
    }
    let (f, spin_fock, e, scalars) = match (beta, spin) {
        (Some((cb, rest)), Some([da, db, identity])) => {
            // No beta electron, no beta factor: Kb is the rest's alone.
            // Only a density-fitted plan reaches here (an open shell over
            // the conventional integrals is refused above).
            let kb = dag.add_gpu(
                "guess exchange",
                worker,
                GuessExchange {
                    k: kb,
                    rest,
                    n,
                    name: "Kb of D",
                },
            );
            let (f, fs, e, scalars) = add_open_fock(
                dag,
                worker,
                n,
                e_nuc,
                [h, j, k, kb],
                [guess, da, db],
                [s, x, identity],
            );
            (f, Some((fs, [da, db, identity], cb)), e, scalars)
        }
        _ => {
            let (f, e, scalars) = add_fock(dag, worker, n, e_nuc, h, j, k, guess, s, x);
            (f, None, e, scalars)
        }
    };
    let (f_star, singular) = diis.add_push(dag, f, e);
    let (_, c_next) = add_solve_fock(dag, worker, n, lwork, f_star, x);
    match spin_fock {
        Some((_, [da, db, _], cb)) => add_open_density(
            dag,
            worker,
            n,
            [n_alpha, n_beta],
            c_next,
            [guess, da, db],
            (factor.c_occ, cb),
        ),
        None => add_density(dag, worker, n, nocc, c_next, guess, factor.c_occ),
    }
    // The decision writes the last build's E, ‖e‖ and count into `previous`.
    dag.add_loop(
        "converged",
        worker,
        body,
        step.max_iterations,
        Converged {
            scalars,
            singular,
            previous,
            diis,
            energy_tolerance: step.energy_tolerance,
            commutator_tolerance: step.commutator_tolerance,
        },
    );

    // After the loop: the converged F's own orbitals and their density, an
    // open shell's made PySCF's.
    let (eps, c) = add_solve_fock(dag, worker, n, lwork, f, x);
    let published = match spin_fock {
        Some((fs, [.., identity], _)) => add_publish_open(
            dag,
            worker,
            n,
            lwork,
            [n_alpha, n_beta],
            fs,
            [s, x, identity],
            eps,
            c,
        ),
        None => [eps, c, add_published_density(dag, worker, n, nocc, c)],
    };
    Ok((previous, published))
}

/// GPU, before the loop: the guess scaled by `N / Σ Z` into a row of its
/// own, which the loop then carries: SAD places `Σ Z` electrons whatever
/// the charge, and the SCF needs the molecule's own count.
struct GuessRescale {
    d: RowId<Device, f64>,
    scale: f64,
    n: usize,
}

impl GpuTask for GuessRescale {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.d);
        device.output("D rescaled", self.n * self.n)
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(
        &self,
        out: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        // add_scf checked that n² fits an i32.
        let n = self.n as i32;
        let d = device.get(self.d);
        geam(
            worker.blas(),
            OP_N,
            OP_N,
            n,
            n,
            self.scale,
            &d,
            n,
            0.0,
            &d,
            n,
            &mut device.take_mut(*out),
            n,
        )
    }
}

/// CPU, before the loop: the guess's eigenvalues, ascending, read for the
/// factor's floor, [`GUESS_FLOOR`] of the largest, written to a pinned row
/// of one double; and for a negative eigenvalue below minus the floor,
/// which is warned of as the C++ first build does.
struct GuessCheck {
    w: RowId<Pinned, f64>,
}

impl CpuTask for GuessCheck {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        host.read(self.w);
        host.output("guess floor", 1)
    }

    fn run(&self, floor_row: &Self::Rows, host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        let w = host.get(self.w);
        let (smallest, largest) = (
            w.first().copied().unwrap_or(0.0),
            w.last().copied().unwrap_or(0.0),
        );
        let floor = GUESS_FLOOR * largest.max(0.0);
        host.take_mut(*floor_row)[0] = floor;
        if smallest < -floor {
            // Only a density loaded from a file can be: the SAD guess is
            // PSD. Its negative part only moves the first iterate.
            log::warn!(
                "scf: the first density has an eigenvalue {smallest:.3e}, below \
                 -{GUESS_FLOOR:.1e} of its largest; its exchange is of the positive part"
            );
        }
        Ok(())
    }
}

/// GPU: `C̃ = U diag((w/2)^(1/2))`, an eigenvalue at or under the floor a
/// zero column, its last `nocc` columns into the carried factor and the
/// rest into a row of their own; for an open shell, `nocc` is `n_alpha`,
/// and its last `n_beta` columns are the carried beta factor and the rest
/// another row.
struct GuessFactor {
    u: RowId<Device, f64>,
    w: RowId<Device, f64>,
    floor: RowId<Pinned, f64>,
    n: usize,
    nocc: usize,
    n_beta: Option<usize>,
}

#[derive(Clone, Copy)]
struct GuessFactorRows {
    half: RowId<Device, f64>,
    root: RowId<Device, f64>,
    weights: RowId<Device, f64>,
    factor: RowId<Device, f64>,
    c_occ: RowId<Device, f64>,
    rest: Option<RowId<Device, f64>>,
    /// An open shell's beta factor, none with no beta electron.
    cb: Option<RowId<Device, f64>>,
    /// An open shell's rest of C̃'s columns beside its beta factor.
    beta_rest: Option<RowId<Device, f64>>,
}

impl GpuTask for GuessFactor {
    type Rows = GuessFactorRows;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.u);
        device.read(self.w);
        host.read(self.floor);
        let n = self.n;
        let (occ, rest) = match self.n_beta {
            Some(_) => ("C a", "C a rest"),
            None => ("C occ", "C rest"),
        };
        GuessFactorRows {
            half: device.scratch("w/2", n),
            root: device.scratch("(w/2)^(-1/2)", n),
            weights: device.scratch("(w/2)^(1/2)", n),
            factor: device.scratch("guess factor", n * n),
            c_occ: device.output(occ, n * self.nocc),
            rest: (self.nocc < n).then(|| device.output(rest, n * (n - self.nocc))),
            cb: self
                .n_beta
                .filter(|&nb| nb > 0)
                .map(|nb| device.output("C b", n * nb)),
            beta_rest: self
                .n_beta
                .map(|nb| device.output("C b rest", n * (n - nb))),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        match self.n_beta {
            Some(nb) => vec![("n", self.n), ("n_alpha", self.nocc), ("n_beta", nb)],
            None => vec![("n", self.n), ("nocc", self.nocc)],
        }
    }

    fn run(
        &self,
        rows: &GuessFactorRows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let blas = worker.blas();
        // add_scf checked that n² fits an i32.
        let n = self.n as i32;
        let (u, w) = (device.get(self.u), device.get(self.w));
        let (mut half, mut root, mut weights, mut factor, mut c_occ) = (
            device.take_mut(rows.half),
            device.take_mut(rows.root),
            device.take_mut(rows.weights),
            device.take_mut(rows.factor),
            device.take_mut(rows.c_occ),
        );
        // The floor `guess check` wrote, copied on this stream into the
        // first element of `weights`, which the kernel reads it from and the
        // dgmm after it overwrites: a stream node does not read a pinned row
        // on the host. Halved, as the map is of w/2, by the kernel's
        // `0.5 * floor`; the halving is exact, so the test is the C++'s
        // `w <= floor`.
        let (mut floor, _) = split_at_mut(&mut weights, 1);
        memcpy_p2d(&mut floor, host, self.floor, worker)?;
        // Each call follows the one before on the stream. (w/2)^(1/2) as
        // (w/2) (w/2)^(-1/2): the kernel maps an eigenvalue at or under the
        // floor, so a negative one too, to 0, and so drops its column.
        geam(blas, OP_N, OP_N, n, 1, 0.5, &w, n, 0.0, &w, n, &mut half, n)?;
        copy::d2d(&mut root, &half, worker)?;
        inverse_sqrt(worker, self.n, &mut root, 0.5, Some(&floor))?;
        dgmm(blas, LEFT, n, 1, &half, n, &root, 1, &mut weights, n)?;
        dgmm(blas, RIGHT, n, n, &u, n, &weights, 1, &mut factor, n)?;
        // Ascending: the largest eigenvalues are the last nocc columns.
        {
            let (head, tail) = split_at_mut(&mut factor, self.n * (self.n - self.nocc));
            copy::d2d(&mut c_occ, &tail, worker)?;
            if let Some(rest) = rows.rest {
                copy::d2d(&mut device.take_mut(rest), &head, worker)?;
            }
        }
        if let (Some(nb), Some(rest)) = (self.n_beta, rows.beta_rest) {
            let (head, tail) = split_at_mut(&mut factor, self.n * (self.n - nb));
            if let Some(cb) = rows.cb {
                copy::d2d(&mut device.take_mut(cb), &tail, worker)?;
            }
            copy::d2d(&mut device.take_mut(rest), &head, worker)?;
        }
        Ok(())
    }
}

/// GPU, in the body: `K of D = K + K rest`, then `K rest = 0`, so the
/// rest of the guess's factor enters the first iteration's K alone. The
/// output row is `name`: `"K of D"`, or an open shell's `"Ka of D"` and
/// `"Kb of D"`. With no K, an open shell's beta with no beta electron,
/// `K of D` is `K rest`: the guess's Db is D/2, every later one 0.
struct GuessExchange {
    k: Option<RowId<Device, f64>>,
    rest: RowId<Device, f64>,
    n: usize,
    name: &'static str,
}

impl GpuTask for GuessExchange {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        if let Some(k) = self.k {
            device.read(k);
        }
        device.write(self.rest);
        device.output(self.name, self.n * self.n)
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(
        &self,
        total: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let n = self.n as i32;
        let (mut rest, mut total) = (device.take_mut(self.rest), device.take_mut(*total));
        // The zero follows the sum, or the copy, on the stream.
        match self.k {
            Some(k) => geam(
                worker.blas(),
                OP_N,
                OP_N,
                n,
                n,
                1.0,
                &device.get(k),
                n,
                1.0,
                &rest,
                n,
                &mut total,
                n,
            )?,
            None => copy::d2d(&mut total, &rest, worker)?,
        }
        memset::zero(&mut rest, worker)
    }
}

/// CPU, before the loop: the decision's carried `[E, ‖e‖, builds]`, its
/// energy 0, a heap row: host state no copy engine meets.
struct EnergyBefore;

impl CpuTask for EnergyBefore {
    type Rows = RowId<Heap, f64>;

    fn layout(&self, _: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) -> Self::Rows {
        heap.output("last build", 3)
    }

    fn run(&self, row: &Self::Rows, _: &Bufs<'_, Pinned>, heap: &Bufs<'_, Heap>) -> Status {
        heap.take_mut(*row).fill(0.0);
        Ok(())
    }
}

/// The loop's decision: the iteration recorded and logged, convergence
/// tested, and a singular DIIS subspace restarted at residual scale
/// `1/‖e‖`.
struct Converged {
    scalars: RowId<Pinned, f64>,
    singular: RowId<Pinned, i32>,
    previous: RowId<Heap, f64>,
    diis: Diis,
    energy_tolerance: f64,
    commutator_tolerance: f64,
}

impl Decision for Converged {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) {
        host.read(self.scalars);
        host.read(self.singular);
        heap.write(self.previous);
    }

    fn decide(
        &self,
        _: &(),
        host: &Bufs<'_, Pinned>,
        heap: &Bufs<'_, Heap>,
        iteration: usize,
    ) -> Result<ControlFlow<()>> {
        let [energy, norm] = host.get(self.scalars)[..] else {
            unreachable!("the scalars row is [E, |e|]")
        };
        let last = heap.take_mut(self.previous);
        let delta = energy - last[0];
        // A count well under 2^53, so exact as a double.
        last.copy_from_slice(&[energy, norm, (iteration + 1) as f64]);
        log::info!(
            "scf: iteration {:3}  E = {energy:.12}  dE = {delta:+.3e}  |[F,D]| = {norm:.3e}",
            iteration + 1
        );
        if iteration > 0 && delta.abs() < self.energy_tolerance && norm < self.commutator_tolerance
        {
            log::info!(
                "scf: converged in {} iterations, E = {energy:.12}",
                iteration + 1
            );
            return Ok(ControlFlow::Break(()));
        }
        if host.get(self.singular)[0] != 0 && norm > 0.0 {
            // The push already returned F as built, the plain step. The
            // history restarts from this pair, its residual scaled to unit
            // norm, and every later residual by the same 1/‖e‖, so the next
            // push extrapolates over two pairs again.
            let scale = 1.0 / norm;
            log::info!(
                "scf: DIIS subspace singular at |[F,D]| = {norm:.3e}; restarted with its \
                 residuals scaled by {scale:.3e}"
            );
            self.diis.restart(scale);
        }
        Ok(ControlFlow::Continue(()))
    }
}
