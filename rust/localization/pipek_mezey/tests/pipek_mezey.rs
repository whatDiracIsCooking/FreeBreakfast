//! The localization on a card (tier 4, device 0, one run per plan), from the
//! committed fixtures `tests/data/<case>_*.npy`, PySCF's RHF orbitals and
//! its Pipek-Mezey of them over Mulliken populations, written by
//! `tests/data/reference.py`, which says how each is made: water (five
//! occupied orbitals, so the schedule's padding runs) and borane (four) in
//! cc-pVDZ. Each case stages S and C, plans the localization once and
//! brings `[L before, L, sweeps]`, `C_local` and `U` back.
//!
//! **The bars, by analysis, one run.**
//!
//! - **L before**, of the orbitals as given: a sum of `n_occ n_atoms`
//!   powers of populations that are each a dot of at most `n_ao` products,
//!   the same numbers PySCF sums; `1e-12` relative, `n_ao ε` (5e-15) on
//!   each population carried through a power of at most 4 and a sum of at
//!   most 15 terms of at most 1, two decades over it.
//! - **L after**, against PySCF's maximum: both codes stop at a maximum,
//!   this one when a sweep moves L by less than its tolerance, set here to
//!   1e-13 rather than the default 1e-10 so that the bars below are tight,
//!   and PySCF when an iteration moves it by less than `conv_tol = 1e-12`,
//!   then its stability check. Each converges at least linearly, faster
//!   than halving the change a step near the maximum, so each code's L is
//!   within its own last change of the maximum: `2e-12` absolute, the sum
//!   of the two stopping changes rounded up.
//! - **Populations**, matched up to orbital order: L is quadratic in a
//!   rotation `δ` away from the maximum, `ΔL ≈ h δ²`, so a code within
//!   `ΔL ≈ 1e-12` of it sits at `δ ≈ (1e-12 / h)^(1/2)`, and a population
//!   moves linearly in `δ`. The softest mode here, water's two lone pairs,
//!   both on O, has `h` of order the square of their populations on H,
//!   1e-2, so `δ` is about 1e-5 there and less elsewhere. `1e-5`.
//! - **U and C_local**: `UᵀU = I` and `C_local = C U` to `100 n_occ ε` and
//!   `100 n_occ ε max|C|`: U is a product of `n_occ - 1` rounds a sweep of
//!   exact 2 × 2 rotations each rounded once, over a few sweeps, and the
//!   gemm's rounding is `γ_{n_occ}` of its terms.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use fb_basis::BasisSet;
use fb_core_arena::{Bufs, Device, Heap, Layout, Pinned, RowId};
use fb_core_error_handling::{ErrorKind, Result, Status};
use fb_core_planner::{CpuTask, Dag};
use fb_localization_pipek_mezey::{PmExponent, PmOptions, add_pm_localize};

/// CPU: an input written into a pinned row.
struct Stage {
    name: &'static str,
    data: Vec<f64>,
}

impl CpuTask for Stage {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        host.output(self.name, self.data.len())
    }

    fn run(&self, row: &Self::Rows, host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        host.take_mut(*row).copy_from_slice(&self.data);
        Ok(())
    }
}

/// CPU: the heap row of the info, then pinned rows, read out in order.
struct Collect {
    info: RowId<Heap, f64>,
    rows: Vec<RowId<Pinned, f64>>,
    out: Arc<Mutex<Vec<Vec<f64>>>>,
}

impl CpuTask for Collect {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) {
        heap.read(self.info);
        for &row in &self.rows {
            host.read(row);
        }
    }

    fn run(&self, _: &(), host: &Bufs<'_, Pinned>, heap: &Bufs<'_, Heap>) -> Status {
        let mut out = vec![heap.get(self.info).to_vec()];
        out.extend(self.rows.iter().map(|&r| host.get(r).to_vec()));
        *self.out.lock().unwrap() = out;
        Ok(())
    }
}

fn data(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(file)
}

/// The fixture `<file>.npy`: its data, column-major, and its shape.
fn fixture(file: &str) -> (Vec<f64>, Vec<usize>) {
    let a = fb_io_npy::read_npy(&data(&format!("{file}.npy"))).unwrap();
    (a.data, a.shape)
}

/// The case's molecule in cc-pVDZ, as the tree places it.
fn basis(case: &str) -> BasisSet {
    let molecule = fb_io_xyz::read_xyz(&data(&format!("{case}.xyz")), 0, 1).unwrap();
    fb_io_basis::resolve_basis("cc-pvdz")
        .unwrap()
        .primitive_normalized()
        .place(&molecule)
        .unwrap()
}

/// `data` staged and uploaded into a device row named `name`.
fn upload(dag: &mut Dag<'_>, name: &'static str, data: Vec<f64>) -> RowId<Device, f64> {
    let len = data.len();
    let staged = dag.add_cpu(format!("stage {name}"), 0, Stage { name, data });
    dag.upload(0, staged, name, len)
}

/// What one localization run returns.
struct Localized {
    info: Vec<f64>,
    c_local: Vec<f64>,
    u: Vec<f64>,
}

/// The localization of the first `n_occ` columns of `c` over `basis`, planned
/// and run once on device 0.
fn localize(
    basis: &BasisSet,
    s: Vec<f64>,
    c: Vec<f64>,
    n_occ: usize,
    options: &PmOptions,
) -> Result<Localized> {
    let n_ao = basis.nao_sphe() as usize;
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let s = upload(&mut dag, "S", s);
    let c = upload(&mut dag, "C", c);
    let (info, [c_local, u]) = add_pm_localize(&mut dag, 0, options, basis, n_occ, s, c)?;
    let rows = vec![
        dag.download(0, c_local, "C local", n_ao * n_occ),
        dag.download(0, u, "U", n_occ * n_occ),
    ];
    let out = Arc::new(Mutex::new(Vec::new()));
    dag.add_cpu(
        "collect",
        0,
        Collect {
            info,
            rows,
            out: Arc::clone(&out),
        },
    );
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result?;
    let mut got = std::mem::take(&mut *out.lock().unwrap()).into_iter();
    Ok(Localized {
        info: got.next().unwrap(),
        c_local: got.next().unwrap(),
        u: got.next().unwrap(),
    })
}

/// Each orbital's Mulliken population on each atom, orbital-major:
/// `Q^A_ii = Σ_{μ ∈ A} C_μi (S C)_μi`.
fn populations(basis: &BasisSet, s: &[f64], c: &[f64], n_occ: usize) -> Vec<Vec<f64>> {
    let n = basis.nao_sphe() as usize;
    let n_atoms = basis.natoms();
    let start = |a: u32| {
        if a < n_atoms {
            basis.ao_offset_sphe(basis.atom_shell_offset(a)) as usize
        } else {
            n
        }
    };
    (0..n_occ)
        .map(|i| {
            let col = &c[i * n..(i + 1) * n];
            let w: Vec<f64> = (0..n)
                .map(|mu| (0..n).map(|nu| s[mu + n * nu] * col[nu]).sum())
                .collect();
            (0..n_atoms)
                .map(|a| (start(a)..start(a + 1)).map(|mu| col[mu] * w[mu]).sum())
                .collect()
        })
        .collect()
}

/// The largest difference between each of `got`'s rows and the nearest
/// unclaimed row of `want`, each row of `want` claimed once.
fn matched_miss(got: &[Vec<f64>], want: &[Vec<f64>]) -> f64 {
    let mut free: Vec<usize> = (0..want.len()).collect();
    let mut worst = 0.0_f64;
    for row in got {
        let (k, d) = free
            .iter()
            .enumerate()
            .map(|(k, &j)| {
                let d = row
                    .iter()
                    .zip(&want[j])
                    .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
                (k, d)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        free.swap_remove(k);
        worst = if d.is_nan() { d } else { worst.max(d) };
    }
    worst
}

/// L before and after, and every localized orbital's populations, against
/// PySCF's, for water's odd count and borane's even one at both exponents;
/// and U orthogonal with `C_local = C U`. Catches a wrong angle or a wrong
/// sign in a rotation (L or the populations off), the padding mishandled
/// (water), a round that skips or repeats an orbital, a sweep that stops
/// short, and U or C_local not the rotation the tensor saw. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn matches_pyscf_at_both_exponents() {
    for case in ["h2o", "bh3"] {
        let basis = basis(case);
        let (s, _) = fixture(&format!("{case}_s"));
        let (c, shape) = fixture(&format!("{case}_c"));
        let (n, n_occ) = (shape[0], shape[1]);
        assert_eq!(n, basis.nao_sphe() as usize, "{case}");
        let (functional, _) = fixture(&format!("{case}_functional"));
        for (k, exponent) in [PmExponent::Two, PmExponent::Four].into_iter().enumerate() {
            let p = exponent as i32;
            let options = PmOptions {
                exponent,
                functional_tolerance: 1e-13,
                ..PmOptions::default()
            };
            let got = localize(&basis, s.clone(), c.clone(), n_occ, &options).unwrap();
            let [before, after, sweeps] = got.info[..] else {
                unreachable!("the info row is three doubles")
            };
            let (want_before, want_after) = (functional[2 * k], functional[2 * k + 1]);
            let miss_before = (before - want_before).abs() / want_before;
            let miss_after = (after - want_after).abs();
            let (q_want, q_shape) = fixture(&format!("{case}_populations_p{p}"));
            let n_atoms = q_shape[1];
            let want: Vec<Vec<f64>> = (0..n_occ)
                .map(|i| (0..n_atoms).map(|a| q_want[i + n_occ * a]).collect())
                .collect();
            let miss_q = matched_miss(&populations(&basis, &s, &got.c_local, n_occ), &want);
            let mut miss_u = 0.0_f64;
            for i in 0..n_occ {
                for j in 0..n_occ {
                    let dot: f64 = (0..n_occ)
                        .map(|r| got.u[r + n_occ * i] * got.u[r + n_occ * j])
                        .sum();
                    miss_u = miss_u.max((dot - f64::from(u8::from(i == j))).abs());
                }
            }
            let mut miss_c = 0.0_f64;
            for j in 0..n_occ {
                for r in 0..n {
                    let cu: f64 = (0..n_occ)
                        .map(|i| c[r + n * i] * got.u[i + n_occ * j])
                        .sum();
                    miss_c = miss_c.max((cu - got.c_local[r + n * j]).abs());
                }
            }
            let max_c = c.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let ulp = 100.0 * n_occ as f64 * f64::EPSILON;
            println!(
                "{case} p={p}: {sweeps} sweeps; |dL before| {miss_before:.1e}, \
                 |dL| {miss_after:.1e}, |dQ| {miss_q:.1e}, |UtU - I| {miss_u:.1e}, \
                 |CU - C_local| {miss_c:.1e}"
            );
            assert!(
                miss_before <= 1e-12,
                "{case} p={p}: L before {before} vs {want_before}"
            );
            assert!(
                miss_after <= 2e-12,
                "{case} p={p}: L {after} vs {want_after}"
            );
            assert!(
                miss_q <= 1e-5,
                "{case} p={p}: populations off by {miss_q:e}"
            );
            assert!(miss_u <= ulp, "{case} p={p}: |UᵀU - I| = {miss_u:e}");
            assert!(
                miss_c <= ulp * max_c,
                "{case} p={p}: |C U - C_local| = {miss_c:e}"
            );
            assert!(
                sweeps >= 1.0 && after >= before,
                "{case} p={p}: {:?}",
                got.info
            );
        }
    }
}

/// One orbital has no pair: no loop is planned, the info is `[L, L, 0]`,
/// and `C_local` is C bit for bit (`C I`, every product exact). Catches the
/// one-orbital plan running a sweep or leaving U unwritten. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn one_orbital_is_left_as_it_is() {
    let basis = basis("h2o");
    let (s, _) = fixture("h2o_s");
    let (c, shape) = fixture("h2o_c");
    let first = c[..shape[0]].to_vec();
    let got = localize(&basis, s, first.clone(), 1, &PmOptions::default()).unwrap();
    assert_eq!(got.info[0], got.info[1]);
    assert_eq!(got.info[2], 0.0);
    assert_eq!(got.u, [1.0]);
    assert_eq!(got.c_local, first);
}

/// A cap of one sweep, short of convergence from water's canonical
/// orbitals, fails the run with `MaxIterationsExceeded`, as the C++ does.
/// Catches a decision that stops at the bound and calls it converged.
/// Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_sweep_cap_fails_the_run() {
    let basis = basis("h2o");
    let (s, _) = fixture("h2o_s");
    let (c, shape) = fixture("h2o_c");
    let options = PmOptions {
        max_sweeps: 1,
        ..PmOptions::default()
    };
    let Err(e) = localize(&basis, s, c, shape[1], &options) else {
        panic!("one sweep converged");
    };
    assert_eq!(e.kind, ErrorKind::MaxIterationsExceeded, "{e}");
    assert!(e.message.contains("in sweep 1"), "{e}");
}

/// The refusals, at planning, before anything is added: a basis with no
/// atoms, an occupied count of 0 or past the basis, and a cap of no sweeps.
/// Plans only; it loads the kernel library for its load check.
#[test]
fn refusals_add_nothing() {
    let placed = basis("h2o");
    let bare = BasisSet::new(placed.shells().to_vec());
    let n = placed.nao_sphe() as usize;
    let mut dag = Dag::new();
    let row = upload(&mut dag, "S", vec![0.0; n * n]);
    let before = dag.len();
    let options = PmOptions::default();
    let no_sweeps = PmOptions {
        max_sweeps: 0,
        ..options
    };
    for (basis, n_occ, options, what) in [
        (&bare, 5, &options, "no atoms"),
        (&placed, 0, &options, "outside 1..=24"),
        (&placed, n + 1, &options, "outside 1..=24"),
        (&placed, 5, &no_sweeps, "max_sweeps is 0"),
    ] {
        let Err(e) = add_pm_localize(&mut dag, 0, options, basis, n_occ, row, row) else {
            panic!("{what}: accepted");
        };
        assert_eq!(e.kind, ErrorKind::InvalidConfig, "{e}");
        assert!(e.message.contains(what), "{what}: {e}");
        assert_eq!(dag.len(), before, "{what}: added a task");
    }
}
