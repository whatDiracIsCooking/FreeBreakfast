//! The arena's device-free tests: the assigner, the table, admission and the
//! memory golden.

use proptest::prelude::*;

use super::*;
use crate::core::error_handling::ErrorKind;
use crate::core::test_support::{golden, panic_message};

fn span(first: usize, last: usize) -> Life {
    Life::Span { first, last }
}

fn row(bytes: usize, align: usize, life: Life) -> Row {
    Row { bytes, align, life }
}

/// The greedy layout by hand: disjoint lifetimes share bytes and
/// overlapping ones do not, best fit takes the smallest gap rather than the
/// first, a plan-long row is never shared, and alignment is honoured.
#[test]
fn assign_by_hand() {
    // a and b overlap at 1; c starts after a ends and takes a's bytes.
    let a = assign(&[
        row(100, 1, span(0, 1)),
        row(50, 1, span(1, 2)),
        row(100, 1, span(2, 3)),
    ]);
    assert_eq!(a.offsets, vec![0, 100, 0]);
    assert_eq!(a.total, 150);

    // At 3 only the two 10-byte rows are live, at 300 and 410, so the free
    // gaps are [0, 300) and [310, 410). An 80-byte row takes the smaller
    // although the larger comes first.
    let a = assign(&[
        row(300, 1, span(0, 2)),
        row(10, 1, span(0, 5)),
        row(100, 1, span(1, 2)),
        row(10, 1, span(1, 5)),
        row(80, 1, span(3, 4)),
    ]);
    assert_eq!(
        a.offsets,
        vec![0, 300, 310, 410, 310],
        "best fit, not first fit"
    );
    assert_eq!(a.total, 420);

    // A plan-long row conflicts with every other, however short.
    let a = assign(&[row(64, 1, Life::Plan), row(64, 1, span(5, 5))]);
    assert_eq!(a.offsets, vec![0, 64]);

    // Alignment: the larger row goes first, at 0, and the 1-byte row is
    // pushed to the next multiple of 256, not to 8.
    let a = assign(&[row(1, 256, span(0, 0)), row(8, 256, span(0, 0))]);
    assert_eq!(a.offsets, vec![256, 0]);
    assert_eq!(a.total, 257);

    // A zero-byte row is placed at 0 and takes no room.
    let a = assign(&[row(0, 256, span(0, 9)), row(8, 1, span(0, 9))]);
    assert_eq!(a.offsets, vec![0, 0]);
    assert_eq!(a.total, 8);

    assert_eq!(
        assign(&[]),
        Assignment {
            offsets: vec![],
            total: 0
        }
    );
}

fn arb_rows() -> impl Strategy<Value = Vec<Row>> {
    prop::collection::vec(
        (
            0usize..5000,
            0u32..10,
            prop::bool::weighted(0.1),
            0usize..20,
            0usize..10,
        )
            .prop_map(|(bytes, align_log, plan, first, len)| Row {
                bytes,
                align: 1 << align_log,
                life: if plan {
                    Life::Plan
                } else {
                    span(first, first + len)
                },
            }),
        0..40,
    )
}

proptest! {
    /// The assigner's invariants over random rows: no two conflicting rows
    /// share a byte, every offset honours its alignment, the total covers
    /// every row and is no more than the rows laid end to end with their
    /// worst padding, and the same input lays out the same way.
    #[test]
    fn assign_invariants(rows in arb_rows()) {
        let a = assign(&rows);
        for (i, r) in rows.iter().enumerate() {
            let o = a.offsets[i];
            prop_assert_eq!(o % r.align, 0, "row {} at {} is not {}-aligned", i, o, r.align);
            if r.bytes > 0 {
                prop_assert!(o + r.bytes <= a.total);
            }
            for (j, s) in rows.iter().enumerate().skip(i + 1) {
                if r.conflicts(s) {
                    let p = a.offsets[j];
                    prop_assert!(
                        o + r.bytes <= p || p + s.bytes <= o,
                        "rows {} [{}, {}) and {} [{}, {}) conflict and overlap",
                        i, o, o + r.bytes, j, p, p + s.bytes
                    );
                }
            }
        }
        let bound: usize = rows
            .iter()
            .filter(|r| r.bytes > 0)
            .map(|r| r.bytes + r.align - 1)
            .sum();
        prop_assert!(a.total <= bound);
        prop_assert_eq!(assign(&rows), a);
    }
}

/// A row as the handoff property draws it: its producer, a mask of later
/// nodes that read it, its bytes, and whether it is scratch.
type DrawnRow = (usize, u16, usize, bool);

/// Kahn's algorithm, picking among the ready nodes by a xorshift seeded
/// with `seed`: one random linear extension of `edges`, as the execution
/// times of the nodes.
fn random_order(n: usize, edges: &[(usize, usize)], mut seed: u64) -> Vec<usize> {
    let mut indegree = vec![0; n];
    let mut succs = vec![Vec::new(); n];
    for &(u, v) in edges {
        indegree[v] += 1;
        succs[u].push(v);
    }
    let mut ready: Vec<usize> = (0..n).filter(|&v| indegree[v] == 0).collect();
    let mut time = vec![usize::MAX; n];
    seed |= 1;
    for t in 0..n {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let v = ready.swap_remove((seed % ready.len() as u64) as usize);
        time[v] = t;
        for &s in &succs[v] {
            indegree[s] -= 1;
            if indegree[s] == 0 {
                ready.push(s);
            }
        }
    }
    time
}

proptest! {
    /// The handoffs are enough: in a plan with an edge from every output's
    /// producer to its readers, plus some edges of its own, plus the
    /// handoffs, any order the plan may run in keeps two rows that share
    /// bytes apart in time. This is the property that makes packing
    /// against the ranks safe when the run does not follow them. Also: a
    /// handoff points forward in rank, so it can be added as a dependency
    /// when the plan is built, and the peak is within the total.
    #[test]
    fn handoffs_order_every_reuse(
        n in 2usize..12,
        drawn in prop::collection::vec((0usize..64, any::<u16>(), 0usize..1024, any::<bool>()), 0..24),
        extra in prop::collection::vec((0usize..64, 0usize..64), 0..16),
        seed in any::<u64>(),
    ) {
        let mut b = TableBuilder::<Device>::new();
        let mut edges = Vec::new();
        let mut ids = Vec::new();
        for &(p, mask, len, scratch) in &drawn as &[DrawnRow] {
            let p = p % n;
            if scratch {
                ids.push(b.scratch::<u8>(p, "s", len));
                continue;
            }
            let id = b.output::<u8>(p, "o", len);
            for r in (p + 1)..n {
                if mask & (1 << (r % 16)) != 0 {
                    b.read(r, id);
                    edges.push((p, r));
                }
            }
            ids.push(id);
        }
        let table = b.build();
        prop_assert!(table.peak().bytes <= table.total());
        let handoffs = table.handoffs();
        for &(u, v) in &handoffs {
            prop_assert!(u < v, "handoff {}->{} points backward", u, v);
        }
        edges.extend(extra.iter().map(|&(u, v)| (u % n, v % n)).filter(|(u, v)| u < v));
        edges.extend_from_slice(&handoffs);
        let time = random_order(n, &edges, seed);

        let interval = |i: usize| {
            let users = &table.rows[i].users;
            let ts = users.iter().map(|&(p, _)| time[p]);
            (ts.clone().min().unwrap(), ts.max().unwrap())
        };
        for i in 0..table.rows.len() {
            for j in (i + 1)..table.rows.len() {
                let (a, c) = (&table.rows[i], &table.rows[j]);
                let (oa, oc) = (table.offsets[i], table.offsets[j]);
                if a.bytes == 0 || c.bytes == 0 || oa >= oc + c.bytes || oc >= oa + a.bytes {
                    continue;
                }
                let ((a0, a1), (c0, c1)) = (interval(i), interval(j));
                prop_assert!(
                    a1 < c0 || c1 < a0,
                    "rows {} and {} share bytes and run at times {}..{} and {}..{}",
                    i, j, a0, a1, c0, c1
                );
            }
        }
    }
}

/// An SCF iteration's table at `nbf`: S, H, X, D, F and C as outputs, J and
/// the eigensolver's workspace as scratch, the solver's `info` flag as the
/// worker's.
fn scf_fixture(nbf: usize) -> Table<Device> {
    let n2 = nbf * nbf;
    let lwork = 1 + 6 * nbf + 2 * n2;
    let mut b = TableBuilder::<Device>::new();
    let _info = b.worker::<i32>("solver info", 1);
    let s = b.output::<f64>(0, "S", n2);
    let h = b.output::<f64>(1, "H", n2);
    b.read(2, s);
    let x = b.output::<f64>(2, "X", n2);
    let _w = b.scratch::<f64>(2, "syevd work (X)", lwork);
    b.read(3, h);
    b.read(3, x);
    let d = b.output::<f64>(3, "D", n2);
    b.read(4, h);
    b.read(4, d);
    let _j = b.scratch::<f64>(4, "J", n2);
    let f = b.output::<f64>(4, "F", n2);
    b.read(5, f);
    b.read(5, x);
    let c = b.output::<f64>(5, "C", n2);
    let _w = b.scratch::<f64>(5, "syevd work (F)", lwork);
    b.read(6, c);
    let _d = b.output::<f64>(6, "D next", n2);
    b.build()
}

/// The memory golden: the layout of a fixed plan is a committed number, so
/// a change to the assigner or to how rows are declared is a diff in review
/// (`docs/testing.md`, "Performance"). `FB_BLESS=1` rewrites it.
#[test]
fn scf_fixture_golden() {
    golden("arena_scf_fixture.txt", &scf_fixture(24).to_string());
}

/// `Table::reads` is every (producer, reader) pair, and a node's `Layout`
/// declares what the builder's positional calls do.
#[test]
fn reads_and_layouts() {
    let table = scf_fixture(24);
    assert_eq!(
        table.reads(),
        vec![
            (0, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 5),
            (3, 4),
            (4, 5),
            (5, 6)
        ]
    );
    let mut b = TableBuilder::<Device>::new();
    let mut l0 = b.node(0);
    let s = l0.output::<f64>("S", 8);
    let _w = l0.scratch::<f64>("W", 8);
    assert_eq!((l0.position(), l0.used()), (0, 2));
    let mut l1 = b.node(1);
    assert_eq!(l1.read(s).index, s.index);
    let _x = l1.output::<f64>("X", 8);
    assert_eq!(l1.used(), 2);
    assert_eq!(b.build().reads(), vec![(0, 1)]);
}
/// Admission is exact: the total fits a cap of the total, one byte less is
/// an `Allocation` error naming the worker, the kind, the peak node and the
/// rows live at it.
#[test]
fn admit_is_exact_and_names_the_peak() {
    let table = scf_fixture(24);
    let total = table.total();
    assert_eq!(admit(&table, total, "stream0"), Ok(()));
    let e = admit(&table, total - 1, "stream0").unwrap_err();
    assert_eq!(e.kind, ErrorKind::Allocation);
    let peak = table.peak();
    let node = format!("at node {}", peak.position.unwrap());
    for needle in ["stream0", "device block", "1 over", node.as_str()] {
        assert!(
            e.message.contains(needle),
            "{needle:?} not in {}",
            e.message
        );
    }
    for (name, _) in &peak.rows {
        assert!(e.message.contains(name), "{name} not in {}", e.message);
    }
}

/// The builder refuses a read the plan could not order: before the
/// producer, at it, twice, of scratch, or of another table's row.
#[test]
fn builder_refuses_misordered_reads() {
    assert!(
        panic_message(|| {
            let mut b = TableBuilder::<Device>::new();
            let r = b.output::<f64>(3, "A", 1);
            b.read(3, r);
        })
        .contains("node 3 reads it, but node 3 produces it")
    );
    assert!(
        panic_message(|| {
            let mut b = TableBuilder::<Device>::new();
            let r = b.output::<f64>(0, "A", 1);
            b.read(1, r);
            b.read(1, r);
        })
        .contains("already uses it")
    );
    assert!(
        panic_message(|| {
            let mut b = TableBuilder::<Device>::new();
            let r = b.scratch::<f64>(0, "A", 1);
            b.read(1, r);
        })
        .contains("only an output")
    );
    assert!(
        panic_message(|| {
            let mut a = TableBuilder::<Device>::new();
            let r = a.output::<f64>(0, "A", 1);
            let t = TableBuilder::<Device>::new().build();
            let _ = t.offset(r);
        })
        .contains("another table")
    );
}

use super::kind::Heap;

/// A view gives a node its own rows and refuses the rest: a value one node
/// writes another reads back; a row outside the node's layout, a read-only
/// row taken to write, a row taken twice, a row another live view holds
/// in a conflicting mode, and another table's row all panic; and a view's
/// borrows are returned when it drops. Also: the block is filled with
/// `0xFF` before anyone writes.
#[test]
fn views_give_a_node_its_rows_and_no_others() {
    let mut b = TableBuilder::<Heap>::new();
    let a = b.output::<f64>(0, "A", 4);
    let s = b.scratch::<u64>(0, "S", 3);
    b.read(1, a);
    b.read(2, a);
    let w = b.worker::<i32>("info", 1);
    let block = Block::new(0, b.build());

    {
        let v0 = block.partition(0);
        assert!(v0.get(s).iter().all(|&x| x == u64::MAX), "the 0xFF fill");
        let a0 = v0.take_mut(a);
        assert!(a0.iter().all(|x| x.is_nan()), "the 0xFF fill");
        a0.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        v0.take_mut(w)[0] = 7;

        let m = panic_message(|| {
            v0.take_mut(a);
        });
        assert!(m.contains("A is already borrowed by this view"), "{m}");
        let v1 = block.partition(1);
        let m = panic_message(|| {
            v1.get(a);
        });
        assert!(m.contains("A is held by another node's view"), "{m}");
    }

    // Readers share, once the writer's view has dropped.
    let (v1, v2) = (block.partition(1), block.partition(2));
    assert_eq!(v1.get(a), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(v2.get(a), v1.get(a));
    assert_eq!(v1.take_mut(w), &[7]);

    let m = panic_message(|| {
        v1.get(s);
    });
    assert!(m.contains("node 1: S is not in its layout"), "{m}");
    let m = panic_message(|| {
        v1.take_mut(a);
    });
    assert!(m.contains("node 1: A is read-only to it"), "{m}");
    let v3 = block.partition(3);
    let m = panic_message(|| {
        v3.take_mut(w);
    });
    assert!(m.contains("info is held by another node's view"), "{m}");
    let other = TableBuilder::<Heap>::new().output::<f64>(0, "X", 1);
    let m = panic_message(|| {
        v1.get(other);
    });
    assert!(m.contains("another table"), "{m}");
}

/// A table of zero bytes allocates nothing and hands out empty rows.
#[test]
fn an_empty_block_allocates_nothing() {
    let mut b = TableBuilder::<Heap>::new();
    let e = b.output::<f64>(0, "E", 0);
    let block = Block::new(0, b.build());
    assert!(block.base.is_null());
    assert!(block.partition(0).take_mut(e).is_empty());
}
