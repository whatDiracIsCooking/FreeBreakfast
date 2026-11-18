//! The projection inequality the density-fitting screen rests on, on a
//! card (tier 4, device 0): with the Coulomb metric,
//! `Σ_Q B_Q,mn² = (mn|mn)_DF <= (mn|mn)`, so a shell pair whose every
//! Cartesian `sqrt((mn|mn))` is below the threshold has columns of B below
//! it too.
//!
//! B is the device's own, packed with every pair kept as the DF step
//! builds it (`fb_integrals_eri_density_fitting_three_center::add_packed_b`
//! over the metric's Cholesky factor, into a heap row), over the spherical
//! functions; the bound is on each column, so the packed B serves as the
//! square one did. Those are
//! fixed combinations of the Cartesian ones, `|μν) = Σ_mn U_mμ U_nν |mn)`
//! (U, [`fb_basis::spherical_transform_matrix`]), so the statement
//! through U is, by the triangle inequality in the Coulomb norm,
//!
//! ```text
//! Σ_Q B_Q,μν² <= (μν|μν) <= (Σ_mn |U_mμ| |U_nν| sqrt((mn|mn)))²
//! ```
//!
//! and for s and p shells, whose blocks of U are the identity, it is the
//! Cartesian statement itself, element by element. The pairs are
//! `fb_integrals_mmd::record_tables`' (its `ClassPairs`), in its order, so
//! the walk over the diagonal's blocks is the one the screen's caller
//! makes.
//!
//! The bar is the rounding of both sides, `1e-10` relative, the libcint
//! comparison's: the fit's and the integrals' rounding is near 1e-14 of
//! max, far inside it. One run.

use std::sync::{Arc, Mutex};

use fb_basis::BasisSet;
use fb_core_arena::{Bufs, Heap, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag};
use fb_integrals_eri_density_fitting_metric::add_metric;
use fb_integrals_eri_density_fitting_pair_map::PairMap;
use fb_integrals_eri_density_fitting_three_center::add_packed_b;
use fb_recipes_cholesky::add_cholesky;

/// The bar on `Σ_Q B² / bound² - 1`.
const BAR: f64 = 1e-10;

/// Water in cc-pVDZ and cc-pVDZ-JKFIT, placed as the DF step places them.
fn water() -> (BasisSet, BasisSet) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/h2o.xyz");
    let molecule = fb_io_xyz::read_xyz(&path, 0, 1).expect("the XYZ file reads");
    let basis = fb_io_basis::resolve_basis("cc-pvdz")
        .expect("a carried set")
        .primitive_normalized()
        .place(&molecule)
        .expect("the set covers water");
    let aux = fb_io_basis::resolve_aux_basis("cc-pvdz-jkfit")
        .expect("a carried fitting set")
        .primitive_normalized()
        .place(&molecule)
        .expect("the fitting set covers water");
    (basis, aux)
}

/// B of `basis` fitted over `aux` on device 0, packed over `map`, one
/// run, brought back from its heap row: `naux · n_columns` doubles,
/// element `(column, Q)` at `column + n_columns Q`.
fn fitted(basis: &BasisSet, aux: &BasisSet, map: &PairMap) -> Vec<f64> {
    let naux = aux.nao_sphe() as usize;
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let classes = fb_integrals_mmd::add_records(&mut dag, 0, basis).expect("the library loads");
    let mut u_aux = None;
    let metric = add_metric(&mut dag, 0, aux, &mut u_aux).expect("the kernels cover the aux set");
    let factor = add_cholesky(&mut dag, 0, "L", naux, metric).expect("the library loads");
    let b = add_packed_b(
        &mut dag, 0, basis, &classes, aux, map, factor, &mut None, &mut u_aux,
    )
    .expect("the kernels cover both sets");
    let out = Arc::new(Mutex::new(Vec::new()));
    dag.add_cpu("keep B", 0, Keep(b, Arc::clone(&out)));
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result.unwrap();
    let got = std::mem::take(&mut *out.lock().unwrap());
    assert_eq!(got.len(), naux * map.sphe().n_columns(), "B kept");
    got
}

/// Water in cc-pVDZ / cc-pVDZ-JKFIT: every spherical pair's column of the
/// device's B within its bound through U from the host diagonal, the
/// Cartesian statement exactly on the s and p shells. Catches a diagonal
/// below the true `(mn|mn)` (a normalisation smaller than the kernels', a
/// quartet dropped), and a diagonal walked in another order than the
/// tables' pairs. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn device_b_is_within_the_host_diagonal() {
    let (basis, aux) = water();
    let tables = fb_integrals_mmd::record_tables(&basis);
    let map = PairMap::new(&tables, None, 0.0);
    let b = fitted(&basis, &aux, &map);
    let (nc, ns, naux) = (
        basis.nao_cart() as usize,
        basis.nao_sphe() as usize,
        aux.nao_sphe() as usize,
    );
    let n_columns = map.sphe().n_columns();
    let sphe = map.sphe().blocks();
    let mut u = vec![0.0; nc * ns];
    fb_basis::spherical_transform_matrix(&basis, &mut u);

    let pairs: Vec<(u32, u32)> = tables
        .classes()
        .iter()
        .flat_map(|c| c.bra.iter().copied().zip(c.ket.iter().copied()))
        .collect();
    let q = fb_integrals_schwarz::schwarz_diagonal(&basis, &pairs);

    let shells = basis.shells();
    let (mut at, mut worst, mut tightest, mut misses) = (0, f64::MIN, f64::MAX, Vec::new());
    // Every pair kept, so the map's pair p is the tables' pair p.
    for (p, &(a, c)) in pairs.iter().enumerate() {
        let (sa, sc) = (&shells[a as usize], &shells[c as usize]);
        let (na, nb) = (
            sa.function_count_cart() as usize,
            sc.function_count_cart() as usize,
        );
        let block = &q[at..at + na * nb];
        at += na * nb;
        let (ca, cc) = (
            basis.ao_offset_cart(a) as usize,
            basis.ao_offset_cart(c) as usize,
        );
        let (pa, pc) = (
            basis.ao_offset_sphe(a) as usize,
            basis.ao_offset_sphe(c) as usize,
        );
        for mu in pa..pa + sa.function_count_sphe() as usize {
            for nu in pc..pc + sc.function_count_sphe() as usize {
                let mut bound = 0.0;
                for m in 0..na {
                    for n in 0..nb {
                        bound +=
                            (u[ca + m + nc * mu] * u[cc + n + nc * nu]).abs() * block[m * nb + n];
                    }
                }
                let column = sphe[p].column + (mu - pa) * sphe[p].n[1] + (nu - pc);
                let norm2: f64 = (0..naux).map(|k| b[column + n_columns * k].powi(2)).sum();
                let ratio = norm2 / (bound * bound) - 1.0;
                worst = worst.max(ratio);
                tightest = tightest.min(ratio.abs());
                if ratio.is_nan() || ratio > BAR {
                    misses.push(format!(
                        "shells ({a}, {c}), functions ({mu}, {nu}): Σ B² {norm2:e} > bound² {:e}",
                        bound * bound
                    ));
                }
            }
        }
    }
    assert_eq!(at, q.len(), "every block walked");
    println!(
        "water cc-pVDZ / cc-pVDZ-JKFIT: {} pairs, max Σ B² / bound² - 1 = {worst:e}, closest \
         |Σ B² / bound² - 1| = {tightest:e}, bar {BAR:e}",
        pairs.len()
    );
    assert!(
        misses.is_empty(),
        "{} columns past the bound:\n{}",
        misses.len(),
        misses.join("\n")
    );
}

/// CPU: the heap row of B copied out, after the build's last write.
struct Keep(RowId<Heap, f64>, Arc<Mutex<Vec<f64>>>);

impl CpuTask for Keep {
    type Rows = ();

    fn layout(&self, _: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) {
        heap.read(self.0);
    }

    fn run(&self, _: &(), _: &Bufs<'_, Pinned>, heap: &Bufs<'_, Heap>) -> Status {
        *self.1.lock().unwrap_or_else(|e| e.into_inner()) = heap.get(self.0).to_vec();
        Ok(())
    }
}
