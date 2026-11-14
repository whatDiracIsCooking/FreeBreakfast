//! Each fixture's core-Hamiltonian orbitals run once on a card (tier 4,
//! device 0), syevd's work asked of a solver handle there first, against
//! LAPACK's generalized eigenvalues of PySCF's H and S
//! (`tests/data/<stem>_epsilon.npy`, written by `tests/data/reference.py`
//! with κ₂(S) and ‖H‖₂ beside them in `<stem>_scales.npy`).
//!
//! The bars are fixed by analysis, one run, n the number of AOs and ε_mach
//! the double's epsilon:
//!
//! - `max_i |ε_i - ε_ref,i| <= 100 n ε_mach κ₂(S) ‖H‖₂`: the generalized
//!   problem's eigenvalue sensitivity through `S^(-1/2)`, whose backward
//!   error `n ε_mach` is amplified by `κ₂(S)` on the scale of H, with two
//!   decades of margin.
//! - `max |Cᵀ S C - I| <= 100 n ε_mach κ₂(S)`: the orthogonalizer's bar,
//!   since `C = X C'` with C' orthogonal.
//! - `max |H C - S C diag(ε)| <= 100 n ε_mach κ₂(S) ‖H‖₂`.
//!
//! The residual and the S-orthonormality are measured with the references'
//! H and S, which the run's own meet to `1e-12 * max |H_ref|` and
//! `1e-12 * max |S_ref|` (the core Hamiltonian's and the overlap's tier 4,
//! measured at 4e-13 and 1e-15 at most): decades under every bar here,
//! whose smallest is 3e-11. C itself is not compared with PySCF's: it is
//! fixed only up to each column's sign and the mixing inside a degenerate
//! level, and the residual and the orthonormality pin what is fixed.
//!
//! The run's timing record has an entry per task, and every stream task's
//! entry has a stream scope.

mod common;

use std::rc::Rc;

use fb_core_thread_pool::Placement;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::solver::SolverHandle;
use fb_recipes_diagonalize::syevd_lwork;

/// A float64 `.npy` of `rust/<dir>`: `(shape, data)`, data column-major.
fn npy(dir: &str, file: &str) -> (Vec<usize>, Vec<f64>) {
    let a = fb_io_npy::read_npy(&common::data(dir, file)).unwrap();
    (a.shape, a.data)
}

/// `a b` for column-major `n` × `n` `a` and `n` × `n` `b`, on the host.
fn matmul(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    (0..n * n)
        .map(|ij| {
            let (i, j) = (ij % n, ij / n);
            (0..n).map(|k| a[i + k * n] * b[k + j * n]).sum()
        })
        .collect()
}

/// Each fixture planned, run once on device 0 and held to its three bars;
/// its measures and task times are printed, and every miss named before
/// the test fails. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn core_orbitals_match_lapack() {
    let mut misses = Vec::new();
    for (stem, xyz, basis) in common::FIXTURES {
        let run = common::run(xyz, basis, |n| {
            syevd_lwork(&SolverHandle::new(Rc::new(Stream::new(0))), n)
        });
        let n = run.n_ao();
        let planned = run.planned();
        let stream_tasks: Vec<bool> = (0..planned.len())
            .map(|p| matches!(planned.placement(p), Placement::Stream(_)))
            .collect();
        let (record, (eps, c)) = run.run(0);
        record.result.unwrap();

        // One entry per task, in rank order; a stream scope on each stream
        // task's, none on a CPU task's.
        assert_eq!(
            record.tasks.len(),
            stream_tasks.len(),
            "{stem}: tasks timed"
        );
        for (t, &on_stream) in record.tasks.iter().zip(&stream_tasks) {
            println!(
                "{stem} {}@{}: host {:?}, device {:?}",
                t.name, t.worker, t.host, t.device
            );
            assert_eq!(
                t.device.is_some(),
                on_stream,
                "{stem} {}: a stream scope iff on a stream",
                t.name
            );
        }

        let (shape, eps_ref) = npy("pipeline/scf/tests/data", &format!("{stem}_epsilon.npy"));
        assert_eq!(shape, [n], "{stem}: the reference energies");
        let (_, scales) = npy("pipeline/scf/tests/data", &format!("{stem}_scales.npy"));
        let [kappa, norm_h] = scales[..] else {
            panic!("{stem}: the scales are [κ₂(S), ‖H‖₂]")
        };
        let (shape, s) = npy(
            "integrals/one_electron/overlap/tests/data",
            &format!("{stem}.npy"),
        );
        assert_eq!(shape, [n, n], "{stem}: S");
        let (_, h) = npy(
            "integrals/one_electron/core_hamiltonian/tests/data",
            &format!("{stem}.npy"),
        );
        assert_eq!((eps.len(), c.len()), (n, n * n), "{stem}: ε and C");

        let orthonormal_bar = 100.0 * n as f64 * f64::EPSILON * kappa;
        let bar = orthonormal_bar * norm_h;

        let eps_err = eps
            .iter()
            .zip(&eps_ref)
            .fold(0.0_f64, |m, (g, w)| m.max((g - w).abs()));
        let sc = matmul(n, &s, &c);
        let hc = matmul(n, &h, &c);
        let mut orthonormal_err = 0.0_f64;
        let mut residual = 0.0_f64;
        for j in 0..n {
            for i in 0..n {
                let ctsc: f64 = (0..n).map(|k| c[k + i * n] * sc[k + j * n]).sum();
                let identity = if i == j { 1.0 } else { 0.0 };
                orthonormal_err = orthonormal_err.max((ctsc - identity).abs());
                residual = residual.max((hc[i + j * n] - sc[i + j * n] * eps[j]).abs());
            }
        }
        println!(
            "{stem}: n {n}, κ₂(S) {kappa:e}, ‖H‖₂ {norm_h:e}, max |ε - ε_ref| = {eps_err:e}, \
             max |CᵀSC - I| = {orthonormal_err:e} (bar {orthonormal_bar:e}), \
             max |HC - SCε| = {residual:e} (bar {bar:e})"
        );
        for (what, got, limit) in [
            ("max |ε - ε_ref|", eps_err, bar),
            ("max |CᵀSC - I|", orthonormal_err, orthonormal_bar),
            ("max |HC - SCε|", residual, bar),
        ] {
            if got.is_nan() || got > limit {
                misses.push(format!("{stem}: {what} = {got:e} > {limit:e}"));
            }
        }
    }
    assert!(misses.is_empty(), "past the bar:\n{}", misses.join("\n"));
}
