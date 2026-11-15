//! The records on a card (tier 4, device 0) against the C++ tree's, dumped
//! on CUDA by `tools/mmd_records` (`tests/data/README.md`): one basis's for
//! H2O/cc-pVDZ and CO/cc-pVTZ, and two bases' for H2O/cc-pVDZ against
//! H2O/STO-3G at a displaced geometry (`common::two_bases`), which
//! `mixed_primitive_pair_classes` forms in the C++ tree and
//! `add_mixed_records` here.
//!
//! The records here are in enumeration order, and the C++ tree's in
//! descending ζ, equal ζ in enumeration order (its stable sort, which this
//! fork leaves out: `fb_integrals_mmd`'s crate doc). So each class is first
//! held to the enumeration, every record's offsets and exponents in sweep
//! order exactly (`common::enumeration`, written from the bases), and then
//! sorted stably by descending ζ, which is the C++ tree's order exactly,
//! since ζ is bitwise the same (below), and compared field by field.
//!
//! The bars are fixed by analysis, one run, the same for both:
//!
//! - The class list, each `n_rec`, and every integer field (`off_*`,
//!   `rank_*`) exactly: they are the same sweep of the same bases.
//! - `zeta`, `ea` and `eb` bitwise: exponents read from the same JSON, and
//!   ζ their sum, one rounding on both sides. So the stable sort by ζ
//!   reproduces the C++ order, and these fields being equal checks it.
//! - Every other field within 1e-13 relative: the coefficients come out of
//!   each tree's host normalization (`pow`, `sqrt`), whose libm rounding may
//!   differ by an ulp, carried through a few device operations (`s`, and
//!   the dressed bounds through it). A field the reference holds as exactly
//!   0 must be exactly 0.

mod common;

use common::{Class, HostClass, check_enumeration, enumeration};
use fb_core_planner::Dag;
use fb_integrals_mmd::{add_mixed_records, add_records};

/// The double fields, in the reference file's order.
const F64_FIELDS: [&str; 16] = [
    "zeta", "p_x", "p_y", "p_z", "pa_x", "pa_y", "pa_z", "pb_x", "pb_y", "pb_z", "s", "ea", "eb",
    "bound", "bound_ab", "bound_ba",
];
/// The integer fields, after them.
const I32_FIELDS: [&str; 4] = ["off_a", "off_b", "rank_a", "rank_b"];
/// The fields compared bitwise: sums of exponents, and the exponents.
const BITWISE: [&str; 3] = ["zeta", "ea", "eb"];
/// Every other double's relative bar.
const RELATIVE: f64 = 1e-13;

/// The reference file `tests/data/<stem>.bin` (its README has the format).
fn reference(stem: &str) -> Vec<HostClass> {
    let path = format!("{}/tests/data/{stem}.bin", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut at = 0;
    let mut take = |n: usize| {
        let s = &bytes[at..at + n];
        at += n;
        s
    };
    assert_eq!(take(8), b"fbmmd v1", "{path}: not a records file");
    let word = |s: &[u8]| <[u8; 4]>::try_from(s).unwrap();
    let n_classes = u32::from_le_bytes(word(take(4)));
    let mut classes = Vec::new();
    for _ in 0..n_classes {
        let mut head = [0_i32; 3];
        head.iter_mut()
            .for_each(|h| *h = i32::from_le_bytes(word(take(4))));
        let [la, lb, n_rec] = head;
        let n = n_rec as usize;
        let f64s = (0..F64_FIELDS.len())
            .map(|_| {
                take(8 * n)
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| f64::from_le_bytes(*c))
                    .collect()
            })
            .collect();
        let i32s = (0..I32_FIELDS.len())
            .map(|_| {
                take(4 * n)
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| i32::from_le_bytes(*c))
                    .collect()
            })
            .collect();
        classes.push(HostClass {
            la,
            lb,
            n_rec,
            f64s,
            i32s,
        });
    }
    assert_eq!(at, bytes.len(), "{path}: trailing bytes");
    classes
}

/// `c`'s records in the C++ tree's order: sorted stably by descending ζ.
fn sorted_by_zeta(c: &HostClass) -> HostClass {
    let mut order: Vec<usize> = (0..c.n_rec as usize).collect();
    order.sort_by(|&a, &b| c.f64s[0][b].total_cmp(&c.f64s[0][a]));
    let permute = |v: &Vec<f64>| order.iter().map(|&r| v[r]).collect();
    let permute_i = |v: &Vec<i32>| order.iter().map(|&r| v[r]).collect();
    HostClass {
        f64s: c.f64s.iter().map(permute).collect(),
        i32s: c.i32s.iter().map(permute_i).collect(),
        ..c.clone()
    }
}

/// `got`, the records of case `stem` sorted as the C++ tree's are, against
/// the C++ tree's, at the bars
/// above: every field's largest error printed, and every miss pushed onto
/// `misses`.
fn compare(stem: &str, got: &[HostClass], misses: &mut Vec<String>) {
    let want = reference(stem);
    let list = |c: &[HostClass]| c.iter().map(|c| (c.la, c.lb, c.n_rec)).collect::<Vec<_>>();
    assert_eq!(
        list(got),
        list(&want),
        "{stem}: the classes and their n_rec"
    );
    for (g, w) in got.iter().zip(&want) {
        let class = format!("{stem} ({},{})", w.la, w.lb);
        for (f, name) in I32_FIELDS.iter().enumerate() {
            let off = g.i32s[f]
                .iter()
                .zip(&w.i32s[f])
                .filter(|(a, b)| a != b)
                .count();
            println!("{class} {name}: {off} of {} differ", w.n_rec);
            if off > 0 {
                misses.push(format!("{class} {name}: {off} records differ"));
            }
        }
        for (f, name) in F64_FIELDS.iter().enumerate() {
            let pairs = g.f64s[f].iter().zip(&w.f64s[f]);
            if BITWISE.contains(name) {
                let off = pairs.filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                println!("{class} {name}: {off} of {} differ bitwise", w.n_rec);
                if off > 0 {
                    misses.push(format!("{class} {name}: {off} records not bitwise"));
                }
            } else {
                // The largest relative error, infinite for a nonzero value
                // where the reference holds exactly 0.
                let worst = pairs
                    .map(|(&a, &b)| {
                        if a == b {
                            0.0
                        } else if b == 0.0 {
                            f64::INFINITY
                        } else {
                            ((a - b) / b).abs()
                        }
                    })
                    .fold(0.0_f64, f64::max);
                println!("{class} {name}: largest relative error {worst:e}");
                if worst > RELATIVE {
                    misses.push(format!("{class} {name}: relative error {worst:e}"));
                }
            }
        }
    }
}

/// Each molecule's records planned, run once on device 0, held to the
/// enumeration, and compared, sorted, with the C++ tree's against the bars
/// above. Every field's largest error is
/// printed, and every miss named before the test fails. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn records_match_the_cpp_tree() {
    let mut misses = Vec::new();
    for (stem, basis) in common::cases() {
        let mut dag = Dag::new();
        let classes = add_records(&mut dag, 0, &basis).unwrap();
        let got = common::run(dag, classes.iter().map(Class::from).collect());
        check_enumeration(stem, &got, &enumeration(&basis, None), &mut misses);
        let got: Vec<HostClass> = got.iter().map(sorted_by_zeta).collect();
        compare(stem, &got, &mut misses);
    }
    assert!(misses.is_empty(), "past the bars:\n{}", misses.join("\n"));
}

/// Two bases' records, water in cc-pVDZ as `rows` and water in STO-3G at
/// a displaced geometry as `cols`, planned, run once on device 0, held to
/// the ordered pairs' enumeration, and compared, sorted, with the C++
/// tree's `mixed_primitive_pair_classes` against the same bars. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn mixed_records_match_the_cpp_tree() {
    let (rows, cols) = common::two_bases();
    let mut dag = Dag::new();
    let classes = add_mixed_records(&mut dag, 0, &rows, &cols).unwrap();
    let got = common::run(dag, classes.iter().map(Class::from).collect());
    let mut misses = Vec::new();
    let case = "h2o_ccpvdz_sto3g_mixed";
    check_enumeration(case, &got, &enumeration(&rows, Some(&cols)), &mut misses);
    let got: Vec<HostClass> = got.iter().map(sorted_by_zeta).collect();
    compare(case, &got, &mut misses);
    assert!(misses.is_empty(), "past the bars:\n{}", misses.join("\n"));
}
