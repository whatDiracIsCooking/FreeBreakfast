//! B built from packed batches of shell pairs (tier 4, device 0): the
//! packed (mn|P) of [`add_packed_three_center`], then [`add_batch_fit`]
//! against one L, for water in cc-pVDZ / cc-pVDZ-JKFIT with every pair
//! kept (threshold 0), held against the host's forward substitution
//! `L⁻¹ (mn|P)` over the same run's L and one-batch tensor, brought back.
//!
//! The bar is `max |Δ| <= 1e-12 max |B|`, one run, the bar this case held
//! against the resident fit (#1689), which it replaces (#1721): the two
//! solve the same triangular system on the same inputs in another order,
//! the device's blocked trsm against the host's column-by-column
//! substitution, so what differs is rounding alone, a few ulps of the
//! largest element carried through L⁻¹.
//!
//! And the batches' promise: the columns of every step are independent,
//! so B over two disjoint batches, one class cut between them, is the
//! one-batch B within the same bar. Not bitwise: the two widths give the
//! transform's GEMMs and the fit's trsm other shapes, and the BLAS picks
//! its kernel by shape, so they round apart (10,745 of 37,352 elements
//! differ in their last bits on the gfx1200, #1750). A batch fitted
//! against the wrong columns is off by far more than the bar.
//! `tests/build.rs` holds B bitwise against batches of the same shapes.

mod common;

use fb_core_planner::Dag;
use fb_integrals_eri_density_fitting_fit::add_batch_fit;
use fb_integrals_eri_density_fitting_metric::add_metric;
use fb_integrals_eri_density_fitting_pair_map::PairMap;
use fb_integrals_eri_density_fitting_three_center::add_packed_three_center;
use fb_integrals_mmd::record_tables;
use fb_recipes_cholesky::add_cholesky;

/// The bar, relative to `max |B|` of the host's substitution.
const BAR: f64 = 1e-12;

/// Water's B, packed over every pair as one batch and as two, against the
/// host's `L⁻¹ (mn|P)` within the bar column by column, a diagonal block's
/// upper columns included, and the two batches' within the bar of the one's. Catches a
/// packed column fitted against the wrong pair or aux function, a batch's
/// column offset taken from the map instead of the batch, and a step
/// whose columns are not independent. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn packed_batches_are_the_hosts_fit() {
    let (basis, aux) = common::water();
    let (nao, naux) = (basis.nao_sphe() as usize, aux.nao_sphe() as usize);
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let classes = fb_integrals_mmd::add_records(&mut dag, 0, &basis).expect("the library loads");
    let (mut u, mut u_aux) = (None, None);
    let metric = add_metric(&mut dag, 0, &aux, &mut u_aux).expect("the kernels cover the aux set");

    let map = PairMap::new(&record_tables(&basis), None, 0.0);
    let n = map.pairs().len();
    // Cut the largest class in two, so the second batch starts mid-class.
    let class = map
        .classes()
        .iter()
        .max_by_key(|c| c.pairs.len())
        .expect("a class");
    let split = class.pairs.start + class.pairs.len() / 2;
    println!(
        "water: nao {nao}, naux {naux}, {n} kept pairs, {} spherical columns; second batch from \
         pair {split}, in class ({}, {})",
        map.sphe().n_columns(),
        class.la,
        class.lb
    );
    let factor = add_cholesky(&mut dag, 0, "L packed", naux, metric).expect("the library loads");
    let l = common::download(&mut dag, factor.0, naux * naux);
    let blocks = map.sphe().blocks();
    let mut batches = Vec::new();
    let mut tensor = None;
    for pairs in [0..n, 0..split, split..n] {
        let columns = blocks[pairs.end - 1].end() - blocks[pairs.start].column;
        let packed = add_packed_three_center(
            &mut dag,
            0,
            &basis,
            &classes,
            &aux,
            &map,
            pairs.clone(),
            pairs.len(),
            &mut u,
        )
        .expect("the kernels cover both sets");
        if tensor.is_none() {
            tensor = Some(common::download(&mut dag, packed, naux * columns));
        }
        let b = add_batch_fit(&mut dag, 0, naux, columns, columns, factor, packed);
        batches.push((common::download(&mut dag, b, naux * columns), columns));
    }

    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result.unwrap();
    let take = |out: &std::sync::Mutex<Vec<f64>>| std::mem::take(&mut *out.lock().unwrap());
    let (l, tensor) = (take(&l), take(tensor.as_ref().expect("the one batch")));
    let [one, first, second] = [0, 1, 2].map(|i| {
        let got = take(&batches[i].0);
        assert_eq!(got.len(), naux * batches[i].1, "batch {i} downloaded");
        got
    });
    assert_eq!(l.len(), naux * naux, "L downloaded");
    assert_eq!(tensor.len(), one.len(), "the one batch's tensor downloaded");

    // The host's L⁻¹ T, column by column: forward substitution over L's
    // lower triangle, column-major, element (i, j) at i + naux j.
    let mut want = tensor.clone();
    for column in want.chunks_mut(naux) {
        for i in 0..naux {
            let mut x = column[i];
            for j in 0..i {
                x -= l[i + naux * j] * column[j];
            }
            column[i] = x / l[i + naux * i];
        }
    }
    let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut worst = 0.0_f64;
    for (g, w) in one.iter().zip(&want) {
        // Not f64::max, which would drop a NaN.
        let d = (g - w).abs();
        if d.is_nan() || d > worst {
            worst = d;
        }
    }
    println!(
        "water: max |B| {scale:.6}, max |Δ| / max |B| = {:e}, bar {BAR:e}",
        worst / scale
    );
    let two = [first, second].concat();
    assert_eq!(two.len(), one.len(), "the two batches cover the one");
    let mut apart = 0.0_f64;
    for (a, b) in one.iter().zip(&two) {
        // Not f64::max, which would drop a NaN.
        let d = (a - b).abs();
        if d.is_nan() || d > apart {
            apart = d;
        }
    }
    println!(
        "water: two batches against one, max |Δ| / max |B| = {:e}, bar {BAR:e}",
        apart / scale
    );
    let mut misses = Vec::new();
    if worst.is_nan() || worst > BAR * scale {
        misses.push(format!("max |Δ| = {worst:e} > {:e}", BAR * scale));
    }
    if apart.is_nan() || apart > BAR * scale {
        misses.push(format!(
            "two batches against one: max |Δ| = {apart:e} > {:e}",
            BAR * scale
        ));
    }
    assert!(misses.is_empty(), "water failed:\n{}", misses.join("\n"));
}
