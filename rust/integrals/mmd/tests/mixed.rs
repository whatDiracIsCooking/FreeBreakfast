//! Two bases' records on a card (tier 4, device 0) against one basis's,
//! where the two coincide: water in cc-pVDZ as both `rows` and `cols`
//! (`tests/common/mod.rs`'s geometry).
//!
//! Over one basis the ordered pairs are the `i <= j` pairs and their
//! mirrors, so every two-basis class is known from the one-basis records:
//!
//! - `la < lb`: the one-basis class `(la, lb)` itself;
//! - `la > lb`: the one-basis class `(lb, la)` swapped (P − A and P − B,
//!   α and β, the AO offsets, the ranks and the two dressed bounds
//!   exchanged; ζ, P, `s` and the bound kept);
//! - `la = lb`: the one-basis class, and its records of two distinct
//!   shells (offsets that differ) swapped.
//!
//! The bar, one run: every field bitwise, but for a record of equal
//! exponents (α = β), whose P, P − A, P − B and dressed bounds may round
//! differently ([`TIE_FIELDS`]). The same kernels form each record from
//! the same shells' values; `s` and the bound are symmetric in the pair,
//! and so is P while the exponents differ, since the records kernel fuses
//! the larger-exponent term, whichever is the bra; at a tie it fuses the
//! bra's, which the mirror swaps. The two sets enumerate their pairs in
//! different orders (the ordered pairs `rows`-major, the `i <= j` pairs in
//! sweep order), so each class is matched record by record on its
//! primitive pair (offsets, ranks, exponents); and each set is held to its
//! own enumeration (`common::enumeration`).

mod common;

use common::{Class, HostClass, check_enumeration, enumeration};
use fb_core_planner::Dag;
use fb_integrals_mmd::{add_mixed_records, add_records};

/// The fields a record of equal exponents (α = β) may round differently
/// swapped: P, P − A and P − B (1..10), and the two dressed bounds formed
/// from them (14, 15).
const TIE_FIELDS: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15];
/// Their bar for a position: the records kernel fuses P's larger-exponent
/// term, and at α = β it fuses the bra's, so P = (αA + βB)/ζ of a pair and
/// of its mirror differ by one rounding of a product term, at most an ulp
/// of max(|A|, |B|) after the division, and P − A and P − B carry that and
/// their own rounding: 4 ulps of the record's largest position, or of the
/// centres', whichever is larger.
const POSITION_ULPS: f64 = 4.0;
/// Their bar for a dressed bound, relative: a few operations over inputs
/// an ulp apart, the same-basis test's bar for every field it cannot hold
/// bitwise (`tests/records.rs`).
const BOUND_RELATIVE: f64 = 1e-13;

/// One record: the 16 doubles' bits, then the 4 integers.
type Record = ([u64; 16], [i32; 4]);

/// The class's records, in record order.
fn records(c: &HostClass) -> Vec<Record> {
    (0..c.n_rec as usize)
        .map(|r| {
            (
                std::array::from_fn(|f| c.f64s[f][r].to_bits()),
                std::array::from_fn(|f| c.i32s[f][r]),
            )
        })
        .collect()
}

/// A record with its bra and ket exchanged.
fn swapped((f, i): Record) -> Record {
    let mut g = f;
    // P - A (4..7) with P - B (7..10), α (11) with β (12), the dressed
    // bounds (14, 15).
    for k in 0..3 {
        g.swap(4 + k, 7 + k);
    }
    g.swap(11, 12);
    g.swap(14, 15);
    (g, [i[1], i[0], i[3], i[2]])
}

/// Water in cc-pVDZ against itself, both record sets formed in one plan
/// and run once. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn two_bases_that_coincide_give_one_basis_records() {
    let [(_, basis), _] = common::cases();
    // The largest centre coordinate, the scale of a position's rounding.
    let centre_scale = basis
        .shells()
        .iter()
        .map(|s| {
            let c = s.center();
            c.x.abs().max(c.y.abs()).max(c.z.abs())
        })
        .fold(0.0_f64, f64::max);
    let mut dag = Dag::new();
    let same = add_records(&mut dag, 0, &basis).unwrap();
    let mixed = add_mixed_records(&mut dag, 0, &basis, &basis).unwrap();
    assert!(
        mixed
            .iter()
            .all(|c| c.n_cart() == [basis.nao_cart() as usize; 2])
    );
    let n_same = same.len();
    let classes: Vec<Class> = same
        .iter()
        .map(Class::from)
        .chain(mixed.iter().map(Class::from))
        .collect();
    let host = common::run(dag, classes);
    let (same, mixed) = host.split_at(n_same);
    let mut misses = Vec::new();
    check_enumeration("one basis", same, &enumeration(&basis, None), &mut misses);
    check_enumeration(
        "two bases",
        mixed,
        &enumeration(&basis, Some(&basis)),
        &mut misses,
    );

    let one = |la: i32, lb: i32| {
        same.iter()
            .find(|c| (c.la, c.lb) == (la, lb))
            .map(records)
            .unwrap_or_default()
    };
    let mut want_keys = Vec::new();
    for c in same {
        want_keys.push((c.la, c.lb));
        if c.la != c.lb {
            want_keys.push((c.lb, c.la));
        }
    }
    want_keys.sort_unstable();
    let got_keys: Vec<_> = mixed.iter().map(|c| (c.la, c.lb)).collect();
    assert_eq!(got_keys, want_keys, "the two-basis classes, ascending");

    for c in mixed {
        let (la, lb) = (c.la, c.lb);
        let want: Vec<Record> = match la.cmp(&lb) {
            std::cmp::Ordering::Less => one(la, lb),
            std::cmp::Ordering::Greater => one(lb, la).into_iter().map(swapped).collect(),
            std::cmp::Ordering::Equal => {
                let r = one(la, lb);
                let mirrors: Vec<Record> = r
                    .iter()
                    .filter(|(_, i)| i[0] != i[1])
                    .map(|&x| swapped(x))
                    .collect();
                r.into_iter().chain(mirrors).collect()
            }
        };
        let got = records(c);
        if got.len() != want.len() {
            misses.push(format!(
                "({la},{lb}): {} records against {}",
                got.len(),
                want.len()
            ));
            continue;
        }
        // A record is its primitive pair: the offsets, ranks and the two
        // exponents name it.
        let key = |(f, i): &Record| (*i, f[11], f[12]);
        let want: std::collections::HashMap<_, Record> =
            want.into_iter().map(|r| (key(&r), r)).collect();
        let (mut unmatched, mut off_bits, mut worst_tie) = (0, 0, 0.0_f64);
        for g in &got {
            let Some(w) = want.get(&key(g)) else {
                unmatched += 1;
                continue;
            };
            let tie = g.0[11] == g.0[12];
            for k in 0..16 {
                if g.0[k] == w.0[k] {
                    continue;
                }
                let (a, b) = (f64::from_bits(g.0[k]), f64::from_bits(w.0[k]));
                if tie && TIE_FIELDS.contains(&k) {
                    let scale = if k < 10 {
                        g.0[1..10]
                            .iter()
                            .chain(&w.0[1..10])
                            .map(|&x| f64::from_bits(x).abs())
                            .fold(centre_scale, f64::max)
                    } else {
                        a.abs().max(b.abs())
                    };
                    let bar = if k < 10 {
                        POSITION_ULPS * f64::EPSILON
                    } else {
                        BOUND_RELATIVE
                    };
                    let err = (a - b).abs() / scale;
                    worst_tie = worst_tie.max(err);
                    if err > bar {
                        off_bits += 1;
                    }
                } else {
                    off_bits += 1;
                }
            }
        }
        println!(
            "({la},{lb}): {} records, {unmatched} unmatched, {off_bits} fields past the bar, \
             largest scaled difference where alpha = beta {worst_tie:e}",
            got.len()
        );
        if unmatched > 0 || off_bits > 0 {
            misses.push(format!(
                "({la},{lb}): {unmatched} records unmatched, {off_bits} fields past the bar"
            ));
        }
    }
    assert!(misses.is_empty(), "past the bar:\n{}", misses.join("\n"));
}
