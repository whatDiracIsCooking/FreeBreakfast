//! The host tables from a hand-built basis (tier 3): the shell table, each
//! class's members in sweep order with the swap into `la <= lb`, and their
//! record prefixes, exactly.

use fb_basis::{BasisSet, Shell};
use fb_core_types::Double3;

use fb_integrals_sys::{ClassPairs, ShellTable};

use super::tables::tables;

/// Four shells on two centres, a p before the s shells so the swap is
/// needed: 0 p on A (2 primitives), 1 s on B (3), 2 s on A (1), 3 d on B (1).
fn basis() -> BasisSet {
    let a = Double3 {
        x: 0.0,
        y: 0.5,
        z: -1.0,
    };
    let b = Double3 {
        x: 1.25,
        y: 0.0,
        z: 2.0,
    };
    BasisSet::new(vec![
        Shell::zipped(1, &[3.0, 0.5], &[0.4, 0.6], a),
        Shell::zipped(0, &[10.0, 2.0, 0.25], &[0.1, 0.2, 0.3], b),
        Shell::zipped(0, &[0.75], &[1.0], a),
        Shell::zipped(2, &[1.5], &[1.0], b),
    ])
}

fn class(la: i32, lb: i32, bra: &[u32], ket: &[u32], rec_begin: &[i32]) -> ClassPairs {
    ClassPairs {
        la,
        lb,
        bra: bra.to_vec(),
        ket: ket.to_vec(),
        rec_begin: rec_begin.to_vec(),
    }
}

/// The sweep `i <= j`, each pair swapped when its first shell has the
/// higher momentum: (0,1) and (0,2) land in (s|p) with the s shell as bra,
/// (0,3) stays (p|d). Each prefix sums `K_a K_b` in sweep order.
#[test]
fn tables_of_a_hand_built_basis() {
    let t = tables(&basis());
    assert_eq!(
        *t.shells(),
        ShellTable {
            prim_begin: vec![0, 2, 5, 6, 7],
            exponent: vec![3.0, 0.5, 10.0, 2.0, 0.25, 0.75, 1.5],
            coefficient: vec![0.4, 0.6, 0.1, 0.2, 0.3, 1.0, 1.0],
            c_x: vec![0.0, 1.25, 0.0, 1.25],
            c_y: vec![0.5, 0.0, 0.5, 0.0],
            c_z: vec![-1.0, 2.0, -1.0, 2.0],
            // p 3, s 1, s 1, d 6 Cartesians.
            ao_offset: vec![0, 3, 4, 5],
            // The first p, the first s, the second s, the first d.
            rank: vec![0, 0, 1, 0],
        }
    );
    assert_eq!(
        t.classes(),
        vec![
            // (1,1) K 9, (1,2) K 3, (2,2) K 1.
            class(0, 0, &[1, 1, 2], &[1, 2, 2], &[0, 9, 12, 13]),
            // (0,1) swapped K 6, (0,2) swapped K 2.
            class(0, 1, &[1, 2], &[0, 0], &[0, 6, 8]),
            // (1,3) K 3, (2,3) K 1.
            class(0, 2, &[1, 2], &[3, 3], &[0, 3, 4]),
            // (0,0) K 4.
            class(1, 1, &[0], &[0], &[0, 4]),
            // (0,3) K 2.
            class(1, 2, &[0], &[3], &[0, 2]),
            // (3,3) K 1.
            class(2, 2, &[3], &[3], &[0, 1]),
        ]
    );
    assert_eq!(t.n_cart(), 11);
    assert_eq!(t.n_pairs(), 10);
    assert_eq!(t.n_max(), 13);
}
