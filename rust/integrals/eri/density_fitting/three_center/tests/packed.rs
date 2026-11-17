//! B built from packed batches of shell pairs (tier 4, device 0): the
//! packed (mn|P) of [`add_packed_three_center`], then [`add_batch_fit`]
//! against one L, for water in cc-pVDZ / cc-pVDZ-JKFIT with every pair
//! kept (threshold 0), held against the resident B of [`add_three_center`]
//! and [`add_density_fit`] in the same run.
//!
//! The bar is `max |Δ| <= 1e-12 max |B|`, one run, by the issue's
//! analysis (#1689): the Cartesian integrals are bitwise the square
//! kernels' (#1686's test), so what differs is rounding alone, the
//! spherical transform's summation order (three dense GEMMs and the
//! (m, n) average against the per-class strided GEMMs, #1688 measured
//! 1.4e-14 of max there) and the trsm's over a different column count,
//! each a few ulps of the largest element carried through L⁻¹.
//!
//! And the batches' promise, exactly: the columns of every step are
//! independent, so B over two disjoint batches, one class cut between
//! them, is the one-batch B bitwise. Both run in this one process, so
//! the BLAS's choice of kernel per shape is the same library's.

mod common;

use fb_core_planner::Dag;
use fb_integrals_eri_density_fitting_fit::{add_batch_fit, add_density_fit};
use fb_integrals_eri_density_fitting_metric::add_metric;
use fb_integrals_eri_density_fitting_pair_map::PairMap;
use fb_integrals_eri_density_fitting_three_center::{add_packed_three_center, add_three_center};
use fb_integrals_mmd::record_tables;
use fb_recipes_cholesky::add_cholesky;

/// The bar, relative to `max |B|` of the resident fit.
const BAR: f64 = 1e-12;

/// Water's B, packed over every pair as one batch and as two, against the
/// resident B within the bar column by column, a diagonal block's upper
/// columns included, and the two batches' bitwise the one's. Catches a
/// packed column fitted against the wrong pair or aux function, a batch's
/// column offset taken from the map instead of the batch, and a step
/// whose columns are not independent. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn packed_batches_are_the_resident_fit() {
    let (basis, aux) = common::water();
    let (nao, naux) = (basis.nao_sphe() as usize, aux.nao_sphe() as usize);
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let classes = fb_integrals_mmd::add_records(&mut dag, 0, &basis).expect("the library loads");
    let (mut u, mut u_aux) = (None, None);
    let tensor = add_three_center(&mut dag, 0, &basis, &classes, &aux, &mut u, &mut u_aux)
        .expect("the kernels cover both sets");
    let metric = add_metric(&mut dag, 0, &aux, &mut u_aux).expect("the kernels cover the aux set");
    let (resident, _) =
        add_density_fit(&mut dag, 0, nao, naux, metric, tensor).expect("the library loads");
    let resident = common::download(&mut dag, resident, naux * nao * nao);

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
    let blocks = map.sphe().blocks();
    let mut batches = Vec::new();
    for pairs in [0..n, 0..split, split..n] {
        let columns = blocks[pairs.end - 1].end() - blocks[pairs.start].column;
        let packed = add_packed_three_center(
            &mut dag, 0, &basis, &classes, &aux, &map, pairs, &mut u, &mut u_aux,
        )
        .expect("the kernels cover both sets");
        let b = add_batch_fit(&mut dag, 0, naux, columns, factor, packed);
        batches.push((common::download(&mut dag, b, naux * columns), columns));
    }

    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result.unwrap();
    let take = |out: &std::sync::Mutex<Vec<f64>>| std::mem::take(&mut *out.lock().unwrap());
    let resident = take(&resident);
    let [one, first, second] = [0, 1, 2].map(|i| {
        let got = take(&batches[i].0);
        assert_eq!(got.len(), naux * batches[i].1, "batch {i} downloaded");
        got
    });
    assert_eq!(resident.len(), naux * nao * nao, "B downloaded");

    // Element (Q, column) of a packed B at Q + naux · column, against the
    // resident B's (Q, n + nao m), every column of every block.
    let scale = resident.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut worst = 0.0_f64;
    for block in blocks {
        for a in 0..block.n[0] {
            for b in 0..block.n[1] {
                let (column, m, k) = (
                    block.column + a * block.n[1] + b,
                    block.ao[0] + a,
                    block.ao[1] + b,
                );
                for q in 0..naux {
                    let want = resident[q + naux * (k + nao * m)];
                    worst = worst.max((one[q + naux * column] - want).abs());
                }
            }
        }
    }
    println!(
        "water: max |B| {scale:.6}, max |Δ| / max |B| = {:e}, bar {BAR:e}",
        worst / scale
    );
    let two = [first, second].concat();
    let differ = one
        .iter()
        .zip(&two)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let mut misses = Vec::new();
    if worst > BAR * scale {
        misses.push(format!("max |Δ| = {worst:e} > {:e}", BAR * scale));
    }
    if differ > 0 {
        misses.push(format!(
            "{differ} of {} elements of the two-batch B differ in their bits from the one-batch B",
            one.len()
        ));
    }
    assert!(misses.is_empty(), "water failed:\n{}", misses.join("\n"));
}
