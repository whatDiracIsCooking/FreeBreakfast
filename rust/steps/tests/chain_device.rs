//! The DF-RHF chain on a card (tier 4): every step of `fb-steps` through
//! [`fb_steps::run`], cc-pVDZ over cc-pVDZ-JKFIT.
//!
//! The energies are `../experimental/rust/v0.0.2/README.md`'s recorded
//! ones, each within 1.2e-12 of PySCF's DF-RHF over the same documents
//! (`pipeline/scf/tests/data/scf_reference.py`'s `.npy`, printed beside
//! each). The bar on an energy is 1e-9 Eh, set by the loop's own
//! convergence: `[scf] energy_tolerance` is 1e-10 between two builds and
//! the commutator 1e-7, whose square bounds the energy's distance from the
//! stationary point, so a converged energy is a few 1e-11 from it, and two
//! runs that converge in different orders (batches, rings, GPUs, workers)
//! agree within the same. B's bar is 1e-10 of its largest element
//! (`docs/testing.md`, "The steps' tests"): the same three-centre integrals and the same unique
//! Cholesky factor, its triangular solve's forward error a modest multiple
//! of `ε κ(L)` on a metric whose condition water's JKFIT keeps under 1e5.
//!
//! One run each; the whole binary takes under 3 s on an RTX 3080, so no
//! case is a slow one (`docs/testing.md`, "Slow cases").

#![allow(
    clippy::disallowed_methods,
    reason = "a test reads its result back on the host: the harness waits, not a task"
)]
#![allow(
    clippy::disallowed_macros,
    reason = "a test's case lists and host reference, outside any step's run"
)]

#[allow(
    dead_code,
    reason = "the shared module's orthogonalizer check is the other binaries'"
)]
mod common;

use std::path::Path;

use fb_core_budget::{Backing, LockId};
use fb_core_flow::Bus;
use fb_steps::{Config, EriStep, MainBus, Run, ScfStep, ScfTable, for_each_pair, run};

/// The bar on an energy, hartree.
const ENERGY_BAR: f64 = 1e-9;

/// `(file, recorded, PySCF)`: v0.0.2's recorded energy and PySCF's DF-RHF.
const WATER: (&str, f64, f64) = ("h2o.xyz", -76.026_751_140_543, -76.026_751_140_542_9);
const BENZENE: (&str, f64, f64) = ("benzene.xyz", -230.721_729_270_921, -230.721_729_270_919_78);
const ETHYLENE: (&str, f64, f64) = (
    "ethylene_meci856.xyz",
    -77.867_545_824_336,
    -77.867_545_824_336_02,
);

/// The chain's config of `xyz`: cc-pVDZ, `[eri]` over cc-pVDZ-JKFIT, and
/// `[scf]` at its defaults.
fn chain(xyz: &str) -> Config {
    let mut config = common::config(xyz, "cc-pvdz", Some("cc-pvdz-jkfit"));
    config.scf = Some(ScfTable::default());
    config
}

/// The least `[eri] mem_per_gpu` of `config`'s plan at `ring`.
fn least(config: &Config, ring: usize) -> u64 {
    let bus = MainBus::setup(config.clone(), Backing::Dry).unwrap();
    EriStep::least_mem_per_gpu(&bus.shared().dims, ring)
}

/// `config` run on the card, converged.
fn converged(config: Config) -> Run {
    let run = run(config, Backing::Real).expect("the chain runs");
    assert!(run.plan.output(ScfStep).converged());
    run
}

/// The SCF's energy and iterations of `run`.
fn energy(run: &Run) -> (f64, usize) {
    let scf = run.plan.output(ScfStep);
    (scf.energy(), scf.iterations())
}

/// `case` on one GPU against its recorded energy, its J/K passes one per
/// iteration (exact).
fn against_the_record((xyz, recorded, pyscf): (&str, f64, f64)) {
    let run = converged(chain(xyz));
    let (e, iterations) = energy(&run);
    assert_eq!(
        run.plan.output(EriStep).passes(),
        iterations,
        "{xyz}: one J/K pass per iteration"
    );
    println!(
        "{xyz}: E = {e:.12} in {iterations} iterations, |E - recorded| = {:.1e}, \
         |E - PySCF| = {:.1e}",
        (e - recorded).abs(),
        (e - pyscf).abs()
    );
    assert!(
        (e - recorded).abs() < ENERGY_BAR,
        "{xyz}: {e} against {recorded}"
    );
}

/// The first J/K pass is given the guess's factor, not its D: on water,
/// benzene and ethylene the factor's rank is below the AOs and at least
/// the occupied orbitals (the SAD guess holds every electron in its
/// atoms' occupied shells), and each converges in the iterations it took
/// from D (11, 11 and 51, `main` at c35c6c9b), within one: the dropped
/// eigenvalues are round-off, so the path is the same. The energies are
/// [`against_the_record`]'s.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_first_pass_is_the_guesss_factor() {
    for ((xyz, ..), from_d) in [(WATER, 11), (BENZENE, 11), (ETHYLENE, 51)] {
        let run = converged(chain(xyz));
        let dims = &run.plan.bus().shared().dims;
        let (n, n_occ) = (dims.n_ao, dims.n_occ);
        let scf = run.plan.output(ScfStep);
        let (rank, iterations) = (scf.guess_rank(), scf.iterations());
        println!(
            "{xyz}: the guess's factor of rank {rank} of {n} AOs, {n_occ} occupied; \
             {iterations} iterations, {from_d} from D"
        );
        assert!(
            n_occ <= rank && rank < n,
            "{xyz}: rank {rank}, {n_occ} occupied of {n}"
        );
        assert!(
            iterations.abs_diff(from_d) <= 1,
            "{xyz}: {iterations} iterations against {from_d} from D"
        );
    }
}

/// Water's energy, the chain's representative: every step, one GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn waters_energy_matches_the_recorded_one() {
    against_the_record(WATER);
}

/// Water's dication, a closed shell by `[molecule] charge = 2` alone,
/// eight electrons in four orbitals, against PySCF's DF-RHF over the same
/// documents (`scf_reference.py`'s molecule built again at `charge = 2`,
/// whose neutral is [`WATER`]'s to 1e-13), at the energy bar.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_water_dications_energy_matches_pyscfs() {
    const PYSCF: f64 = -74.596_854_781_569_2;
    let mut config = chain("h2o.xyz");
    config.molecule.charge = 2;
    let run = converged(config);
    assert_eq!(run.plan.bus().shared().dims.n_occ, 4);
    let (e, iterations) = energy(&run);
    println!(
        "water's dication: E = {e:.12} in {iterations} iterations, |E - PySCF| = {:.1e}",
        (e - PYSCF).abs()
    );
    assert!((e - PYSCF).abs() < ENERGY_BAR, "{e} against {PYSCF}");
}

/// Benzene and ethylene, whose DIIS subspace goes singular and restarts.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn benzenes_and_ethylenes_energies_match_the_recorded_ones() {
    against_the_record(BENZENE);
    against_the_record(ETHYLENE);
}

/// Water at the least `mem_per_gpu` (many pair batches, aux batches past
/// the ring's depth, so every ring buffer is reused), NaN-filled, then at
/// both rings' depth 1: each holds the default's energy, so a ring buffer
/// rewritten before its reader ran shows here.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn forced_batches_and_rings_of_one_give_waters_energy() {
    let default = converged(chain(WATER.0));
    let (e0, _) = energy(&default);
    {
        let eri = default.plan.output(EriStep);
        assert_eq!(eri.jk_width(), eri.n_aux(), "one aux batch by default");
    }
    drop(default);
    for ring in [2, 1] {
        let mut config = chain(WATER.0);
        let at = least(&config, ring);
        let eri = config.eri.as_mut().unwrap();
        (eri.mem_per_gpu, eri.ring) = (Some(at), ring);
        config.resources.nan_fill = true;
        let run = converged(config);
        let (e, iterations) = energy(&run);
        let eri = run.plan.output(EriStep);
        println!(
            "ring {ring} at {at} B: {} pair batches of {} columns, aux batches of {} of {}; \
             E in {iterations} iterations, |E - default| = {:.1e}",
            eri.batches(),
            eri.width(),
            eri.jk_width(),
            eri.n_aux(),
            (e - e0).abs()
        );
        assert!(eri.batches() > 6, "the build is batched");
        assert!(eri.jk_width() * 4 < eri.n_aux(), "the passes are batched");
        assert!((e - e0).abs() < ENERGY_BAR, "ring {ring}: {e} against {e0}");
    }
}

/// Benzene on two GPUs (two logical ones on device 0, then every count of
/// real ones up to four), each share one aux batch and then forced
/// batches, NaN-filled, on two workers and on eight: each the energy of
/// one GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn several_gpus_give_benzenes_energy_on_one() {
    let (one, _) = energy(&converged(chain(BENZENE.0)));
    assert!((one - BENZENE.1).abs() < ENERGY_BAR);
    let mut lists = vec![vec![0, 0]];
    let seen = fb_core_device::device_count().unwrap();
    lists.extend((2..=seen.min(4)).map(|n| (0..n).collect::<Vec<i32>>()));
    let forced = least(&chain(BENZENE.0), 2) * 3 / 2;
    for devices in lists {
        for (mem_per_gpu, threads) in [(None, 8), (Some(forced), 2)] {
            let mut config = chain(BENZENE.0);
            config.eri.as_mut().unwrap().mem_per_gpu = mem_per_gpu;
            config.resources.devices.clone_from(&devices);
            (config.resources.nan_fill, config.resources.threads) = (true, threads);
            let run = converged(config);
            let (e, iterations) = energy(&run);
            println!(
                "devices {devices:?}, mem_per_gpu {mem_per_gpu:?}, {threads} workers: E in \
                 {iterations} iterations, {} pair batches, |E - one GPU| = {:.1e}",
                run.plan.output(EriStep).batches(),
                (e - one).abs()
            );
            assert!(
                (e - one).abs() < ENERGY_BAR,
                "{devices:?}: {e} against {one}"
            );
        }
    }
}

/// The executor's worker count moves no result: water on one worker and on
/// eight, NaN-filled, gives the same energy, so a step that read a slot
/// before its writer finished, or a lock taken out of order, shows here.
/// On eight workers, the lock acquisitions the execute records: every
/// step's in rank order, gpu0's three holders (the orthogonalizer, `eri`
/// and the SCF) never holding it at once, and a step recorded as having
/// waited asking while another held one of its locks.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_worker_count_does_not_move_waters_energy() {
    let at = |threads: usize| {
        let mut config = chain(WATER.0);
        (config.resources.threads, config.resources.nan_fill) = (threads, true);
        converged(config)
    };
    let ((one, n1), eight) = (energy(&at(1)), at(8));
    let (e8, n8) = energy(&eight);
    println!(
        "one worker: E in {n1} iterations; eight: in {n8}; |E1 - E8| = {:.1e}",
        (one - e8).abs()
    );
    assert!((one - e8).abs() < ENERGY_BAR && (one - WATER.1).abs() < ENERGY_BAR);

    let steps = &eight.steps;
    for t in steps {
        assert!(
            t.locks.windows(2).all(|w| w[0] < w[1]),
            "{}: {:?} out of rank order",
            t.name,
            t.locks
        );
        assert!(t.began <= t.held && t.held <= t.released && t.released <= t.ended);
    }
    let gpu0: Vec<_> = steps
        .iter()
        .filter(|t| t.locks.contains(&LockId::Gpu(0)))
        .collect();
    let mut names: Vec<_> = gpu0.iter().map(|t| t.name).collect();
    names.sort_unstable();
    assert_eq!(names, ["eri", "orthogonalizer", "scf"]);
    for (i, x) in gpu0.iter().enumerate() {
        for y in &gpu0[i + 1..] {
            assert!(
                x.released <= y.held || y.released <= x.held,
                "{} and {} held gpu0 at once",
                x.name,
                y.name
            );
        }
    }
    for x in steps.iter().filter(|t| t.waited) {
        assert!(
            steps.iter().any(|o| o.name != x.name
                && o.locks.iter().any(|l| x.locks.contains(l))
                && o.held <= x.held
                && o.released >= x.began),
            "{} is recorded as having waited with no other holder",
            x.name
        );
    }
    println!(
        "{}",
        steps
            .iter()
            .filter(|t| !t.locks.is_empty())
            .map(|t| format!(
                "{}: {:?} held {:?}..{:?}{}",
                t.name,
                t.locks,
                t.held,
                t.released,
                if t.waited { ", waited" } else { "" }
            ))
            .collect::<Vec<_>>()
            .join("; ")
    );
}

/// A replay of water at the same dimensions starts from the orbitals the
/// run left, converges to its energy again, and allocates nothing: the
/// ledger's count is unchanged, exactly. B is built again, so the passes
/// are counted again: the replay's equal its iterations, exactly.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_replay_gives_waters_energy_and_allocates_nothing() {
    let mut run = converged(chain(WATER.0));
    let (first, n_first) = energy(&run);
    let allocations = run.plan.bus().ledger().allocations();
    run.replay().expect("the replay runs");
    let (again, n_again) = energy(&run);
    println!(
        "first: {n_first} iterations; replay: {n_again}, |E - first| = {:.1e}; {allocations} \
         allocations",
        (again - first).abs()
    );
    assert!(run.plan.output(ScfStep).converged());
    assert!((again - first).abs() < ENERGY_BAR);
    assert!(
        n_again < n_first,
        "the replay starts from the converged orbitals"
    );
    assert_eq!(run.plan.bus().ledger().allocations(), allocations);
    assert_eq!(
        run.plan.output(EriStep).passes(),
        n_again,
        "the replay's J/K passes, one per iteration"
    );
}

/// Water's B against PySCF's `L⁻¹ (P|mn)` over the same documents
/// (`tests/data/b_reference.py`), element for element at each pair
/// column's AOs: a column of B mapped to the wrong pair, or a fit against
/// the wrong factor, shows here and in no energy.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn waters_b_matches_pyscf() {
    let mut config = chain(WATER.0);
    config.scf = None;
    let run = run(config, Backing::Real).unwrap();
    let shared = run.plan.bus().shared();
    let eri = run.plan.output(EriStep);
    let (n_aux, n_col) = (eri.n_aux(), eri.n_col());
    let reference = fb_io_npy::read_npy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/b_h2o_cc-pvdz_cc-pvdz-jkfit.npy"),
    )
    .unwrap();
    assert_eq!(
        reference.shape,
        [n_aux, shared.dims.n_ao * (shared.dims.n_ao + 1) / 2]
    );
    // Each pair column's AOs, in B's order: the classes ascending, each
    // class's pairs in the walk's order, a pair's columns bra-major.
    let classes = &shared.dims.classes;
    let mut first: Vec<usize> = classes
        .iter()
        .scan(0, |at, c| {
            let here = *at;
            *at += c.n_pair * c.sph();
            Some(here)
        })
        .collect();
    let mut aos = vec![(0usize, 0usize); n_col];
    for_each_pair(&shared.basis, |la, lb, bra, ket| {
        let c = classes
            .iter()
            .position(|c| (c.la, c.lb) == (la, lb))
            .unwrap();
        let (a0, b0) = (
            shared.basis.ao_offset_sphe(bra) as usize,
            shared.basis.ao_offset_sphe(ket) as usize,
        );
        let mut col = first[c];
        for ia in 0..(2 * la + 1) as usize {
            for ib in 0..(2 * lb + 1) as usize {
                aos[col] = (a0 + ia, b0 + ib);
                col += 1;
            }
        }
        first[c] = col;
    });
    let (b, mut largest, mut worst) = (eri.b(), 0.0f64, 0.0f64);
    for (col, &(m, n)) in aos.iter().enumerate() {
        let p = m.max(n) * (m.max(n) + 1) / 2 + m.min(n);
        for q in 0..n_aux {
            let want = reference.data[q + n_aux * p];
            largest = largest.max(want.abs());
            worst = worst.max((b[q * n_col + col] - want).abs());
        }
    }
    println!("B: {n_aux} x {n_col}, max |B| = {largest:.6}, max |B - PySCF| = {worst:.1e}");
    assert!(largest > 0.1 && !worst.is_nan() && worst <= 1e-10 * largest);
}
