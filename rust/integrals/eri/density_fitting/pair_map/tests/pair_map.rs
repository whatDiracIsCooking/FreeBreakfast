//! The packed map over random bases (tier 3, host only; the crate links
//! the kernel library through `RecordTables`' crate, so it must be built):
//! pack then unpack of a symmetric matrix is the identity on the kept
//! pairs, bitwise; each `(m, n)` a kept pair covers is one column and each
//! column one `(m, n)`, or its mirror in a diagonal block; every pair is
//! kept with no bounds or at threshold 0; a higher threshold never adds a
//! pair; and the map of one small basis is a committed golden.

use std::collections::BTreeMap;
use std::fmt::Write;

use fb_basis::{BasisSet, Shell};
use fb_core_types::Double3;
use fb_integrals_eri_density_fitting_pair_map::{Packing, PairMap};
use fb_integrals_mmd::record_tables;
use fb_test_support::golden;
use proptest::prelude::*;

/// A basis of 1 to 8 shells, each of momentum 0 to 3 with 1 or 2
/// primitives, on one of three centres, in a random order.
fn arb_basis() -> impl Strategy<Value = BasisSet> {
    prop::collection::vec((0_i32..=3, 1_usize..=2, 0_usize..3), 1..=8).prop_map(|shells| {
        BasisSet::new(
            shells
                .into_iter()
                .map(|(l, n_prim, centre)| {
                    let at = Double3 {
                        x: centre as f64,
                        y: 0.5 * centre as f64,
                        z: -1.0,
                    };
                    Shell::zipped(l, &[2.0, 0.5][..n_prim], &[0.6, 0.4][..n_prim], at)
                })
                .collect(),
        )
    })
}

/// A basis, a bound in `[0, 1)` per shell pair, and a threshold in
/// `[0, 1)`.
fn arb_screened() -> impl Strategy<Value = (BasisSet, Vec<f64>, f64)> {
    arb_basis().prop_flat_map(|basis| {
        let n = basis.nshells() as usize;
        (
            Just(basis),
            prop::collection::vec(0.0_f64..1.0, n * (n + 1) / 2),
            0.0_f64..1.0,
        )
    })
}

/// A symmetric `n × n` matrix of distinct values, so a misplaced copy
/// shows.
fn symmetric(n: usize, seed: u64) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for m in 0..n {
        for k in 0..=m {
            let v = ((m * n + k) as f64 + 0.25) * (1.0 + seed as f64 * 1e-3);
            a[m * n + k] = v;
            a[k * n + m] = v;
        }
    }
    a
}

/// Whether each `(m, n)` lies in a kept pair's block or its mirror, from
/// the blocks themselves.
fn support(packing: &Packing) -> Vec<bool> {
    let nao = packing.nao();
    let mut s = vec![false; nao * nao];
    for b in packing.blocks() {
        for a in 0..b.n[0] {
            for k in 0..b.n[1] {
                let (m, n) = (b.ao[0] + a, b.ao[1] + k);
                s[m * nao + n] = true;
                s[n * nao + m] = true;
            }
        }
    }
    s
}

/// `packing`'s index tables: each column's row, column and weight.
fn index_tables(packing: &Packing) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let n = packing.n_columns();
    let (mut rows, mut cols, mut weights) = (vec![-1; n], vec![-1; n], vec![f64::NAN; n]);
    packing.write_tables(&mut rows, &mut cols, &mut weights);
    (rows, cols, weights)
}

/// Every property that holds of one map, over both packings.
fn check_map(map: &PairMap, n_pairs: usize) -> Result<(), TestCaseError> {
    // The kept pairs ascend in the tables' order, the classes tile them,
    // each of one shape, ascending in (la, lb).
    prop_assert!(map.pairs().windows(2).all(|w| w[0].pair < w[1].pair));
    prop_assert!(map.pairs().iter().all(|p| p.pair < n_pairs));
    let mut next = 0;
    for (i, c) in map.classes().iter().enumerate() {
        prop_assert!(c.la <= c.lb);
        prop_assert!(i == 0 || (map.classes()[i - 1].la, map.classes()[i - 1].lb) < (c.la, c.lb));
        prop_assert_eq!(c.pairs.start, next);
        prop_assert!(!c.pairs.is_empty());
        next = c.pairs.end;
        for sphe in [false, true] {
            let packing = if sphe { map.sphe() } else { map.cart() };
            let l = |l: i32| {
                if sphe {
                    2 * l as usize + 1
                } else {
                    ((l + 1) * (l + 2) / 2) as usize
                }
            };
            let blocks = &packing.blocks()[c.pairs.clone()];
            prop_assert!(blocks.iter().all(|b| b.n == [l(c.la), l(c.lb)]));
        }
    }
    prop_assert_eq!(next, map.pairs().len());

    for sphe in [false, true] {
        let packing = if sphe { map.sphe() } else { map.cart() };
        let nao = packing.nao();
        let n_cols = packing.n_columns();
        prop_assert_eq!(packing.blocks().len(), map.pairs().len());
        let (rows, cols, weights) = index_tables(packing);

        // The blocks lie side by side, and each column's (m, n) and weight
        // are its block's element.
        let mut column = 0;
        for (p, b) in map.pairs().iter().zip(packing.blocks()) {
            prop_assert_eq!(b.column, column);
            let diagonal = p.shells[0] == p.shells[1];
            prop_assert_eq!(b.is_diagonal(), diagonal);
            for a in 0..b.n[0] {
                for k in 0..b.n[1] {
                    let c = b.column + a * b.n[1] + k;
                    prop_assert_eq!(rows[c] as usize, b.ao[0] + a);
                    prop_assert_eq!(cols[c] as usize, b.ao[1] + k);
                    prop_assert_eq!(weights[c], if diagonal { 1.0 } else { 2.0 });
                }
            }
            column += b.n[0] * b.n[1];
        }
        prop_assert_eq!(column, n_cols);

        // Each (m, n) a kept pair covers is one column, through the
        // column's own (m, n) or, off a diagonal block, its mirror; each
        // column one (m, n); nothing outside the support.
        let mut hits: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for c in 0..n_cols {
            let (m, n) = (rows[c] as usize, cols[c] as usize);
            *hits.entry((m, n)).or_default() += 1;
            if weights[c] == 2.0 {
                prop_assert_ne!(m, n, "an off-diagonal pair's blocks miss the diagonal");
                *hits.entry((n, m)).or_default() += 1;
            }
        }
        let s = support(packing);
        for m in 0..nao {
            for n in 0..nao {
                let want = usize::from(s[m * nao + n]);
                prop_assert_eq!(
                    hits.get(&(m, n)).copied().unwrap_or(0),
                    want,
                    "({}, {})",
                    m,
                    n
                );
            }
        }

        // Pack then unpack of a symmetric matrix: the matrix on the
        // support and zero off it, bitwise; and the weights make the
        // packed dot product the matrices' on the support.
        let a = symmetric(nao, n_cols as u64);
        let mut packed = vec![f64::NAN; n_cols];
        packing.pack(&a, &mut packed);
        let mut back = vec![f64::NAN; nao * nao];
        packing.unpack(&packed, &mut back);
        for i in 0..nao * nao {
            let want = if s[i] { a[i] } else { 0.0 };
            prop_assert_eq!(back[i].to_bits(), want.to_bits(), "element {}", i);
        }
        // Integer-valued, so both sums are exact and agree bitwise.
        let ints: Vec<f64> = a.iter().map(|v| v.floor()).collect();
        packing.pack(&ints, &mut packed);
        let weighted: f64 = packed.iter().zip(&weights).map(|(x, w)| w * x * x).sum();
        let full: f64 = (0..nao * nao)
            .filter(|&i| s[i])
            .map(|i| ints[i] * ints[i])
            .sum();
        prop_assert_eq!(weighted, full);
    }
    Ok(())
}

proptest! {
    /// With no bounds, or at threshold 0, every pair is kept, in the
    /// tables' order, and the map holds every property above.
    #[test]
    fn every_pair_kept_unscreened((basis, bounds, _) in arb_screened()) {
        let tables = record_tables(&basis);
        let n_pairs = tables.n_pairs();
        for map in [PairMap::new(&tables, None, 0.0), PairMap::new(&tables, Some(&bounds), 0.0)] {
            prop_assert_eq!(map.pairs().len(), n_pairs);
            prop_assert!(map.pairs().iter().enumerate().all(|(i, p)| p.pair == i));
            prop_assert_eq!(map.cart().nao(), basis.nao_cart() as usize);
            prop_assert_eq!(map.sphe().nao(), basis.nao_sphe() as usize);
            for (i, p) in map.pairs().iter().enumerate() {
                for (k, &shell) in p.shells.iter().enumerate() {
                    let (cart, sphe) = (map.cart().blocks()[i], map.sphe().blocks()[i]);
                    prop_assert_eq!(cart.ao[k], basis.ao_offset_cart(shell) as usize);
                    prop_assert_eq!(sphe.ao[k], basis.ao_offset_sphe(shell) as usize);
                }
            }
            check_map(&map, n_pairs)?;
        }
    }

    /// A screened map keeps exactly the pairs whose bound is not below the
    /// threshold, holds every property above, and a higher threshold keeps
    /// a subset of a lower one's pairs.
    #[test]
    fn screened((basis, bounds, threshold) in arb_screened(), raise in 0.0_f64..0.5) {
        let tables = record_tables(&basis);
        let map = PairMap::new(&tables, Some(&bounds), threshold);
        let kept: Vec<usize> = map.pairs().iter().map(|p| p.pair).collect();
        let want: Vec<usize> = (0..bounds.len()).filter(|&i| bounds[i] >= threshold).collect();
        prop_assert_eq!(&kept, &want);
        check_map(&map, tables.n_pairs())?;

        let higher = PairMap::new(&tables, Some(&bounds), threshold + raise);
        prop_assert!(higher.pairs().iter().all(|p| kept.contains(&p.pair)));
    }
}

/// The map of one small basis, a p, an s and a d shell over two centres
/// (the p first, so the (p, s) pair is swapped) with the (s, d) pair
/// screened: every kept pair's shells and blocks, and each packing's
/// tables, as text.
#[test]
fn map_golden() {
    let at = |z| Double3 { x: 0.0, y: 0.0, z };
    let basis = BasisSet::new(vec![
        Shell::zipped(1, &[1.2], &[1.0], at(0.0)),
        Shell::zipped(0, &[3.0], &[1.0], at(0.0)),
        Shell::zipped(2, &[0.8], &[1.0], at(1.4)),
    ]);
    let tables = record_tables(&basis);
    // The pairs in the tables' order: classes (0,0) (0,1) (0,2) (1,1)
    // (1,2) (2,2).
    let mut bounds = vec![1.0; tables.n_pairs()];
    let order: Vec<[u32; 2]> = tables
        .classes()
        .iter()
        .flat_map(|c| c.bra.iter().zip(&c.ket).map(|(&a, &b)| [a, b]))
        .collect();
    for (i, s) in order.iter().enumerate() {
        if *s == [1, 2] {
            bounds[i] = 1e-12;
        }
    }
    let map = PairMap::new(&tables, Some(&bounds), 1e-10);

    let mut text = String::new();
    writeln!(
        text,
        "pairs of the tables {}, kept {}",
        tables.n_pairs(),
        map.pairs().len()
    )
    .unwrap();
    for c in map.classes() {
        writeln!(text, "class ({}, {}): pairs {:?}", c.la, c.lb, c.pairs).unwrap();
    }
    for (i, p) in map.pairs().iter().enumerate() {
        let (cart, sphe) = (map.cart().blocks()[i], map.sphe().blocks()[i]);
        writeln!(
            text,
            "pair {} shells {:?}: cart ao {:?} n {:?} column {}; sphe ao {:?} n {:?} column {}",
            p.pair, p.shells, cart.ao, cart.n, cart.column, sphe.ao, sphe.n, sphe.column
        )
        .unwrap();
    }
    for (name, packing) in [("cart", map.cart()), ("sphe", map.sphe())] {
        writeln!(
            text,
            "{name}: nao {} columns {}",
            packing.nao(),
            packing.n_columns()
        )
        .unwrap();
        let (rows, cols, weights) = index_tables(packing);
        for c in 0..packing.n_columns() {
            writeln!(text, "  {c}: ({}, {}) w {}", rows[c], cols[c], weights[c]).unwrap();
        }
    }
    golden(env!("CARGO_MANIFEST_DIR"), "map_psd.txt", &text);
}
