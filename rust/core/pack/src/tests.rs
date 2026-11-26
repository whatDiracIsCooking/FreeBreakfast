//! The packer against reachability found by a plain search, and against
//! the dummy DAG `DUMMY` of `experimental/dag_transitive_closure/model.py`.

use super::{Region, closure, lower_bound, ordered, pack};
use proptest::prelude::*;

/// Whether a path leads from `from` to `to` in the DAG of `preds`, by a
/// depth-first search over the successors: independent of the bitsets.
fn reaches(preds: &[Vec<usize>], from: usize, to: usize) -> bool {
    let n = preds.len();
    let mut succs = vec![Vec::new(); n];
    for (i, of) in preds.iter().enumerate() {
        for &p in of {
            succs[p].push(i);
        }
    }
    let mut seen = vec![false; n];
    let mut stack = succs[from].clone();
    while let Some(s) = stack.pop() {
        if s == to {
            return true;
        }
        if !seen[s] {
            seen[s] = true;
            stack.extend(&succs[s]);
        }
    }
    false
}

/// A DAG of up to 40 steps, each reading some earlier ones, and up to one
/// region per step, some of them empty.
fn arb_dag() -> impl Strategy<Value = (Vec<Vec<usize>>, Vec<Region>)> {
    (1usize..40).prop_flat_map(|n| {
        let preds = (0..n)
            .map(|i| proptest::collection::vec(0..i.max(1), 0..=i.min(3)))
            .collect::<Vec<_>>();
        let sizes = proptest::collection::vec(proptest::option::of(0u64..1000), n);
        (preds, sizes).prop_map(|(preds, sizes)| {
            let preds: Vec<Vec<usize>> = preds
                .into_iter()
                .enumerate()
                .map(|(i, mut of)| {
                    of.retain(|&p| p < i);
                    of.sort_unstable();
                    of.dedup();
                    of
                })
                .collect();
            let regions = sizes
                .into_iter()
                .enumerate()
                .filter_map(|(step, bytes)| {
                    bytes.map(|bytes| Region {
                        step,
                        name: Box::leak(format!("s{step:02}").into_boxed_str()),
                        bytes,
                    })
                })
                .collect();
            (preds, regions)
        })
    })
}

proptest! {
    /// Over random DAGs: the closure is reachability, two regions sharing
    /// a byte belong to steps the DAG orders, the bound's set is an
    /// antichain weighing the bound, and the peak lies between the bound
    /// and the sum.
    #[test]
    fn random_packings_overlap_only_ordered_steps((preds, regions) in arb_dag()) {
        let reach = closure(&preds);
        for (a, of) in reach.iter().enumerate() {
            for b in 0..preds.len() {
                prop_assert_eq!(of.has(b), reaches(&preds, a, b), "{} -> {}", a, b);
            }
        }
        let packed = pack(&reach, &regions);
        for (i, ri) in regions.iter().enumerate() {
            let (bi, ei) = (packed.offsets[i], packed.offsets[i] + ri.bytes);
            prop_assert!(ei <= packed.peak);
            for (j, rj) in regions.iter().enumerate().skip(i + 1) {
                let (bj, ej) = (packed.offsets[j], packed.offsets[j] + rj.bytes);
                if bi < ej && bj < ei {
                    prop_assert!(
                        reaches(&preds, ri.step, rj.step) || reaches(&preds, rj.step, ri.step),
                        "{} [{}, {}) and {} [{}, {}) overlap unordered",
                        ri.name, bi, ei, rj.name, bj, ej
                    );
                }
            }
        }
        let set = &packed.bound_set;
        for (k, &a) in set.iter().enumerate() {
            for &b in &set[k + 1..] {
                prop_assert!(!ordered(&reach, regions[a].step, regions[b].step));
            }
        }
        prop_assert_eq!(set.iter().map(|&r| regions[r].bytes).sum::<u64>(), packed.bound);
        prop_assert!(packed.bound <= packed.peak && packed.peak <= packed.sum);
        prop_assert_eq!(pack(&reach, &regions), packed);
    }
}

const MIB: u64 = 1 << 20;

/// The dummy DAG packs at its lower bound: the chain from `metric` to
/// `nevpt2` shares the bottom 2040 MiB, S, T, V and X go above `eri`, and S
/// shares with X, which it precedes; the peak is the heaviest antichain,
/// T, V, X and `eri`, at 2049 MiB, against a plain sum of 3318.
#[test]
fn the_dummy_dag_packs_at_its_bound() {
    // (name, reads, scratch MiB), as model.py's DUMMY.
    let steps: [(&'static str, &[&str], u64); 12] = [
        ("S", &[], 1),
        ("T", &[], 1),
        ("V", &[], 2),
        ("guess", &[], 0),
        ("metric", &[], 20),
        ("X", &["S"], 6),
        ("H", &["T", "V"], 0),
        ("factor", &["metric"], 8),
        ("eri", &["factor"], 2040),
        ("scf", &["H", "X", "eri", "guess"], 40),
        ("casscf", &["scf", "eri"], 300),
        ("nevpt2", &["casscf", "eri"], 900),
    ];
    let index = |name: &str| steps.iter().position(|s| s.0 == name).unwrap();
    let preds: Vec<Vec<usize>> = steps
        .iter()
        .map(|s| s.1.iter().map(|r| index(r)).collect())
        .collect();
    let regions: Vec<Region> = steps
        .iter()
        .enumerate()
        .filter(|(_, s)| s.2 > 0)
        .map(|(step, s)| Region {
            step,
            name: s.0,
            bytes: s.2 * MIB,
        })
        .collect();
    let reach = closure(&preds);
    let packed = pack(&reach, &regions);
    let at = |name: &str| packed.offsets[regions.iter().position(|r| r.name == name).unwrap()];
    for name in ["metric", "factor", "eri", "scf", "casscf", "nevpt2"] {
        assert_eq!(at(name), 0, "{name}");
    }
    assert_eq!(at("X"), 2040 * MIB);
    assert_eq!(at("S"), 2040 * MIB);
    assert_eq!(at("V"), 2046 * MIB);
    assert_eq!(at("T"), 2048 * MIB);
    assert_eq!(packed.peak, 2049 * MIB);
    assert_eq!(packed.bound, packed.peak);
    assert_eq!(packed.sum, 3318 * MIB);
    let mut set: Vec<&str> = packed.bound_set.iter().map(|&r| regions[r].name).collect();
    set.sort_unstable();
    assert_eq!(set, ["T", "V", "X", "eri"]);
    assert_eq!(
        lower_bound(&reach, &regions),
        (packed.bound, packed.bound_set)
    );
}

/// A cycle is refused once the order is built.
#[test]
#[should_panic(expected = "the step graph has a cycle")]
fn a_cycle_panics() {
    let _ = closure(&[vec![1], vec![0]]);
}
