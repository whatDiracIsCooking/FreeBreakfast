//! The arena's device-free tests: the assigner, the table, admission and the
//! memory golden.

use proptest::prelude::*;

use super::*;
use fb_core_error_handling::ErrorKind;
use fb_test_support::{golden, panic_message};

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

/// The orders the assigner tries, two where a single order loses: the
/// SCF loop fixture's parent with the orthogonalizer's workspace in it
/// (#1511), which start order with the larger first lays out at 44168
/// bytes, the larger first at 38400 and the longer-lived first at 38280,
/// against a bound of 38148; and [`scf_fixture`], at 29184 and 29064 in
/// the two start orders and 24576 larger first, against 24324. Start order
/// with the larger first is the smallest of the three on the DIIS golden.
#[test]
fn assign_keeps_the_smallest_order() {
    let n2 = 24 * 24 * 8;
    let rows = [
        row(4, DEVICE_ALIGNMENT, Life::Plan),
        row(n2, DEVICE_ALIGNMENT, span(0, 8)),
        row(n2, DEVICE_ALIGNMENT, span(1, 8)),
        row(n2, DEVICE_ALIGNMENT, span(2, 8)),
        row(10376, DEVICE_ALIGNMENT, span(2, 2)),
        row(n2, DEVICE_ALIGNMENT, span(3, 8)),
        row(19592, DEVICE_ALIGNMENT, span(4, 8)),
    ];
    assert_eq!((assign(&rows).total, live_bound(&rows)), (38280, 38148));
    assert_eq!(scf_fixture(24).total(), 24576);
}

/// The block's lower bound for `rows`: at each position, the live rows'
/// bytes each padded to its alignment, less the largest padding, since
/// the last row needs none; the largest over the positions. Rows live at
/// one position conflict pairwise, so no layout fits them in less.
fn live_bound(rows: &[Row]) -> usize {
    let at = |p: Option<usize>| {
        let live = rows.iter().filter(|r| {
            r.bytes > 0
                && match (r.life, p) {
                    (Life::Plan, _) => true,
                    (Life::Span { first, last }, Some(p)) => first <= p && p <= last,
                    (Life::Span { .. }, None) => false,
                }
        });
        let padded: usize = live
            .clone()
            .map(|r| r.bytes.next_multiple_of(r.align))
            .sum();
        let slack = live
            .map(|r| r.bytes.next_multiple_of(r.align) - r.bytes)
            .max()
            .unwrap_or(0);
        padded - slack
    };
    rows.iter()
        .filter_map(|r| match r.life {
            Life::Span { first, last } => Some(first..=last),
            Life::Plan => None,
        })
        .flatten()
        .map(|p| at(Some(p)))
        .max()
        .unwrap_or_else(|| at(None))
}

/// One table of a committed golden, as its text gives it: the block's
/// total, the rows outside any loop's body, and each loop's region with
/// its body's rows.
struct GoldenTable {
    name: String,
    total: usize,
    rows: Vec<Row>,
    regions: Vec<(Row, Vec<Row>)>,
}

/// Every table in a golden's text: a `table:` line, then its row lines
/// (`offset bytes owner life name`) and region lines, to `handoffs:`.
fn golden_tables(file: &str, text: &str) -> Vec<GoldenTable> {
    let mut tables: Vec<GoldenTable> = Vec::new();
    let mut open = false;
    for line in text.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        if let Some(at) = words.iter().position(|&w| w == "table:") {
            let total = words[at + 3].parse().expect("a table's total");
            tables.push(GoldenTable {
                name: format!("{file} {}", words[at - 1]),
                total,
                rows: Vec::new(),
                regions: Vec::new(),
            });
            open = true;
            continue;
        }
        let Some(t) = tables.last_mut().filter(|_| open) else {
            continue;
        };
        if line.starts_with("handoffs:") {
            open = false;
        } else if let Some(span) = line.strip_prefix("loop ").and_then(|l| l.split(':').next()) {
            let (first, last) = span.split_once('-').expect("a loop's span");
            let bytes = words[4].parse().expect("a region's bytes");
            let life = Life::Span {
                first: first.parse().expect("a loop's first position"),
                last: last.parse().expect("a loop's last position"),
            };
            t.regions
                .push((row(bytes, DEVICE_ALIGNMENT, life), Vec::new()));
        } else if let [_, bytes, "task" | "edge" | "worker", life, ..] = words[..] {
            let Ok(bytes) = bytes.parse() else { continue };
            let life = match life.split_once('-') {
                Some((a, b)) => span(
                    a.parse().expect("a row's first"),
                    b.parse().expect("a row's last"),
                ),
                None => Life::Plan,
            };
            let r = row(bytes, DEVICE_ALIGNMENT, life);
            match t.regions.last_mut() {
                Some((_, body)) => body.push(r),
                None => t.rows.push(r),
            }
        }
    }
    tables
}

/// Every `tests/golden/*.txt` under `dir`, skipping build trees.
fn golden_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("a readable directory") {
        let path = entry.expect("a directory entry").path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if path.is_dir() {
            if name != "target" && !name.starts_with("build") && !name.starts_with('.') {
                golden_files(&path, out);
            }
        } else if name.ends_with(".txt")
            && path.parent().is_some_and(|p| p.ends_with("tests/golden"))
        {
            out.push(path);
        }
    }
}

/// The assigner's quality, measured: every committed golden's block, and
/// each loop's region, is within 3% of [`live_bound`], a region counted as
/// one row of its parent (`docs/memory.md`, "Lifetimes"). The goldens'
/// own tests hold their text to the code; this holds the text to the
/// bound, so an assigner or planning change that wastes bytes fails here
/// and not only as a diff.
#[test]
fn goldens_are_within_the_bound() {
    const BOUND: f64 = 1.03;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut files = Vec::new();
    golden_files(&root, &mut files);
    files.sort();
    let mut seen = Vec::new();
    let mut over = Vec::new();
    for path in &files {
        let file = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .display()
            .to_string();
        let text = std::fs::read_to_string(path).expect("a golden");
        for t in golden_tables(&file, &text) {
            let mut parent = t.rows.clone();
            let mut check = |name: String, total: usize, bound: usize| {
                if total as f64 > BOUND * bound as f64 {
                    over.push(format!("{name}: {total} bytes, bound {bound}"));
                }
            };
            for (i, (region, body)) in t.regions.iter().enumerate() {
                check(
                    format!("{} loop {i}", t.name),
                    region.bytes,
                    live_bound(body),
                );
                parent.push(*region);
            }
            check(t.name.clone(), t.total, live_bound(&parent));
            seen.push(t.name);
        }
    }
    for name in [
        "core/arena/tests/golden/arena_scf_loop.txt device",
        "pipeline/config/tests/golden/plan_rhf.txt pinned",
    ] {
        assert!(seen.iter().any(|s| s == name), "{name} not read: {seen:?}");
    }
    assert!(over.is_empty(), "over {BOUND} times the bound: {over:#?}");
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

/// A row as the loop property draws it: its producer, a mask of later
/// nodes that read it, its bytes, what it is (0 scratch, 1 output, 2 an
/// output the loop carries when produced before it), and a pick of the
/// carrying writer.
type DrawnLoopRow = (usize, u16, usize, u8, usize);

proptest! {
    /// A loop's body in its parent's table: every body row lies inside the
    /// loop's region, and none shares a byte with a parent row live across
    /// the loop. And every reuse is ordered across iterations: with the
    /// body unrolled `iterations` times, each iteration after the decision
    /// of the one before and before its own, and the table's reads and
    /// handoffs (plus some edges of the plan's own) as edges, any order the
    /// run may take keeps two rows that share bytes apart in time. A body
    /// row is live once per iteration; any other row from its first use to
    /// its last over every iteration. No pair runs from a body node to a
    /// node after the loop but from the decision, which the executor's
    /// contract asks.
    #[test]
    fn loop_bodies_lie_in_their_regions(
        n in 2usize..12,
        start in 0usize..12,
        len in 0usize..12,
        drawn in prop::collection::vec(
            (0usize..64, any::<u16>(), 0usize..1024, 0u8..3, 0usize..12),
            0..24,
        ),
        extra in prop::collection::vec((0usize..64, 0usize..64), 0..16),
        iterations in 1usize..4,
        seed in any::<u64>(),
    ) {
        let first = start % n;
        let last = first + len % (n - first);
        let mut b = TableBuilder::<Device>::new();
        for &(p, mask, bytes, what, pick) in &drawn as &[DrawnLoopRow] {
            let p = p % n;
            if what == 0 {
                let _ = b.scratch::<u8>(p, "s", bytes);
                continue;
            }
            let carried = what == 2 && p < first;
            let writer = first + pick % (last - first + 1);
            let id = b.output::<u8>(p, "o", bytes);
            for r in (p + 1)..n {
                let after_writer = carried && writer <= r && r <= last;
                if mask & (1 << (r % 16)) != 0 && !after_writer {
                    b.read(r, id);
                }
            }
            if carried {
                b.write(writer, id);
            }
        }
        b.add_loop(first..=last);
        let table = b.build();
        let in_loop = |p: usize| first <= p && p <= last;

        let g = &table.regions[0];
        for i in 0..table.rows.len() {
            if table.body[i].is_none() || table.rows[i].bytes == 0 {
                continue;
            }
            let (o, bytes) = (table.offsets[i], table.rows[i].bytes);
            prop_assert!(g.offset <= o && o + bytes <= g.offset + g.bytes);
            for j in 0..table.rows.len() {
                let across = matches!(table.lives[j], Life::Span { first: f, last: l } if f <= last && first <= l);
                if table.body[j].is_some() || !across || table.rows[j].bytes == 0 {
                    continue;
                }
                let p = table.offsets[j];
                prop_assert!(
                    o + bytes <= p || p + table.rows[j].bytes <= o,
                    "body row {} overlaps row {}, live across the loop", i, j
                );
            }
        }

        let mut deps = table.reads();
        deps.extend(table.handoffs());
        for &(u, v) in &deps {
            prop_assert!(u < v, "{}->{} points backward", u, v);
            prop_assert!(
                !(in_loop(u) && v > last) || u == last,
                "{}->{} leaves the body from a body node", u, v
            );
        }
        deps.extend(
            extra
                .iter()
                .map(|&(u, v)| (u % n, v % n))
                .filter(|(u, v)| u < v && !(in_loop(*u) && *v > last && *u != last)),
        );

        // The unrolled plan: the nodes before the loop, the body once per
        // iteration, the nodes after it.
        let pre = first;
        let id = |p: usize, it: usize| {
            if p < first {
                p
            } else if p <= last {
                pre + it * (last - first + 1) + (p - first)
            } else {
                pre + iterations * (last - first + 1) + (p - last - 1)
            }
        };
        let total_nodes = n + (iterations - 1) * (last - first + 1);
        let mut edges = Vec::new();
        for &(u, v) in &deps {
            match (in_loop(u), in_loop(v)) {
                (true, true) => edges.extend((0..iterations).map(|it| (id(u, it), id(v, it)))),
                (false, true) => edges.push((id(u, 0), id(v, 0))),
                (true, false) => edges.push((id(u, iterations - 1), id(v, 0))),
                (false, false) => edges.push((id(u, 0), id(v, 0))),
            }
        }
        for it in 0..iterations {
            for p in first..last {
                edges.push((id(p, it), id(last, it)));
                if it + 1 < iterations {
                    edges.push((id(last, it), id(p, it + 1)));
                }
            }
            if it + 1 < iterations {
                edges.push((id(last, it), id(last, it + 1)));
            }
        }
        let time = &random_order(total_nodes, &edges, seed);

        // Each row's intervals in time: a body row's per iteration, any
        // other's over every iteration.
        let intervals = |i: usize| -> Vec<(usize, usize)> {
            let users = &table.rows[i].users;
            let span = |ts: Vec<usize>| (*ts.iter().min().unwrap(), *ts.iter().max().unwrap());
            if table.body[i].is_some() {
                (0..iterations)
                    .map(|it| span(users.iter().map(|&(p, _)| time[id(p, it)]).collect()))
                    .collect()
            } else {
                let ts = users.iter().flat_map(|&(p, _)| {
                    let its = if in_loop(p) { 0..iterations } else { 0..1 };
                    its.map(move |it| time[id(p, it)])
                });
                vec![span(ts.collect())]
            }
        };
        for i in 0..table.rows.len() {
            for j in (i + 1)..table.rows.len() {
                let (a, c) = (&table.rows[i], &table.rows[j]);
                let (oa, oc) = (table.offsets[i], table.offsets[j]);
                if a.bytes == 0 || c.bytes == 0 || oa >= oc + c.bytes || oc >= oa + a.bytes {
                    continue;
                }
                for &(a0, a1) in &intervals(i) {
                    for &(c0, c1) in &intervals(j) {
                        prop_assert!(
                            a1 < c0 || c1 < a0,
                            "rows {} and {} share bytes and run at times {}..{} and {}..{}",
                            i, j, a0, a1, c0, c1
                        );
                    }
                }
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
    golden(
        env!("CARGO_MANIFEST_DIR"),
        "arena_scf_fixture.txt",
        &scf_fixture(24).to_string(),
    );
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
/// The SCF iteration of [`scf_fixture`] as a loop's body (positions 4 to 8,
/// the decision last, holding no device row), inside a parent holding S, H
/// and X, the solver's flag, and the guess's D, which the body carries: the
/// Fock build and the residual read it, the density writes it. The residual
/// reads S, so all three of the parent's matrices are live across the loop.
/// Returns the table and the body's rows' bytes, laid out alone.
///
/// The orthogonalizer's eigensolver workspace is left out of the parent, so
/// the parent's rows are exactly those live across the loop;
/// [`assign_keeps_the_smallest_order`] lays it out with it.
fn scf_loop_fixture(nbf: usize) -> (Table<Device>, usize) {
    let n2 = nbf * nbf;
    let lwork = 1 + 6 * nbf + 2 * n2;
    let mut b = TableBuilder::<Device>::new();
    let _info = b.worker::<i32>("solver info", 1);
    let s = b.output::<f64>(0, "S", n2);
    let h = b.output::<f64>(1, "H", n2);
    b.read(2, s);
    let x = b.output::<f64>(2, "X", n2);
    b.read(3, h);
    b.read(3, x);
    let d = b.output::<f64>(3, "D", n2);
    // The body: Fock, residual, diagonalize, density; then the decision.
    b.read(4, h);
    b.read(4, d);
    let _j = b.scratch::<f64>(4, "J", n2);
    let f = b.output::<f64>(4, "F", n2);
    b.read(5, f);
    b.read(5, d);
    b.read(5, s);
    b.read(5, x);
    let _e = b.output::<f64>(5, "e", n2);
    b.read(6, f);
    b.read(6, x);
    let c = b.output::<f64>(6, "C", n2);
    let _w = b.scratch::<f64>(6, "syevd work (F)", lwork);
    b.read(7, c);
    b.write(7, d);
    b.add_loop(4..=8);
    // Laid out alone: F, C and the eigensolver's workspace at 6, the
    // body's peak, with no gap (n² doubles is a multiple of 256 at 24).
    let body = 2 * n2 * 8 + lwork * 8;
    (b.build(), body)
}

/// A loop's body in its parent's block (`docs/memory.md`, "Nested
/// plans"): the block is the parent's rows live across the loop plus the
/// body's peak, from the rows' sizes; every body row lies in the region,
/// apart from those rows; the carried D is live across the loop and
/// ordered within it, its readers before its writer; and admission counts
/// the region once, by name. The text is the committed golden.
#[test]
fn scf_loop_golden() {
    let nbf = 24;
    let n2 = nbf * nbf;
    let (table, body) = scf_loop_fixture(nbf);
    // The flag, padded to the alignment, and S, H, X and D.
    let across = DEVICE_ALIGNMENT + 4 * n2 * 8;
    assert_eq!(table.regions.len(), 1);
    let region = &table.regions[0];
    assert_eq!(region.bytes, body, "the region is the body's peak, once");
    assert_eq!(table.total(), across + body);
    let peak = table.peak();
    assert_eq!(peak.position, Some(4));
    assert_eq!(peak.bytes, 4 + 4 * n2 * 8 + body);
    assert!(peak.rows.contains(&("loop 4-8", body)), "{peak:?}");

    let names = |b: Option<usize>| -> Vec<&str> {
        (0..table.rows.len())
            .filter(|&i| table.body[i] == b)
            .map(|i| table.rows[i].name.as_str())
            .collect()
    };
    assert_eq!(names(Some(0)), ["J", "F", "e", "C", "syevd work (F)"]);
    for i in (0..table.rows.len()).filter(|&i| table.body[i] == Some(0)) {
        let o = table.offsets[i];
        assert!(region.offset <= o && o + table.rows[i].bytes <= region.offset + region.bytes);
    }
    let d = table.rows.iter().position(|r| r.name == "D").unwrap();
    assert_eq!(table.lives[d], span(3, 8));
    let reads = table.reads();
    for pair in [(3, 4), (3, 5), (3, 7), (4, 7), (5, 7)] {
        assert!(reads.contains(&pair), "{pair:?} not in {reads:?}");
    }

    let e = admit(&table, table.total() - 1, "stream0").unwrap_err();
    assert!(
        e.message.contains(&format!("loop 4-8 ({body} B)")),
        "{}",
        e.message
    );
    golden(
        env!("CARGO_MANIFEST_DIR"),
        "arena_scf_loop.txt",
        &table.to_string(),
    );
}

/// A carried row is refused where a loop cannot carry it: written outside
/// any loop, produced inside the loop that writes it, or read in the loop
/// after its writer, which would read the iteration's own value in one
/// iteration and the last one's in the next.
#[test]
#[cfg(debug_assertions)] // the contracts are compiled out of release
fn a_carried_row_is_refused_where_no_loop_carries_it() {
    let build = |produce: usize, read: usize, write: usize| {
        panic_message(move || {
            let mut b = TableBuilder::<Device>::new();
            let r = b.output::<f64>(produce, "D", 1);
            b.read(read, r);
            b.write(write, r);
            b.add_loop(2..=4);
            let _ = b.build();
        })
    };
    let m = build(0, 1, 5);
    assert!(
        m.contains("a row is written again only inside a loop"),
        "{m}"
    );
    let m = build(2, 3, 4);
    assert!(
        m.contains("a carried row is produced before its loop"),
        "{m}"
    );
    let m = build(0, 4, 3);
    assert!(
        m.contains("a carried row's writer is its last user in the loop"),
        "{m}"
    );
    let m = panic_message(|| {
        let mut b = TableBuilder::<Device>::new();
        let r = b.scratch::<f64>(0, "W", 1);
        b.write(1, r);
    });
    assert!(m.contains("only an output is carried"), "{m}");
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
    assert!(!e.message.contains("alone"), "{}", e.message);
}

/// A cap smaller than a row names that row, with its bytes, among the rows
/// over the cap on their own, whether or not it is live at the peak, and
/// names no row that fits: at 24 the `n²` matrices fit a cap of their own
/// size exactly and the two `syevd` workspaces do not.
#[test]
fn admit_names_every_row_over_the_cap_alone() {
    let table = scf_fixture(24);
    let matrix = 24 * 24 * 8;
    let work = (1 + 6 * 24 + 2 * 24 * 24) * 8;
    let e = admit(&table, matrix, "stream0").unwrap_err();
    assert_eq!(e.kind, ErrorKind::Allocation);
    let alone = format!(
        "over the cap alone: syevd work (X) ({work} B), syevd work (F) ({work} B); its peak"
    );
    assert!(e.message.contains(&alone), "{alone:?} not in {}", e.message);
}

/// The builder refuses a read the plan could not order: before the
/// producer, at it, twice, of scratch, or of another table's row. Each is a
/// contract of [`TableBuilder::read`], and through it of [`Layout::read`].
#[test]
#[cfg(debug_assertions)] // the contracts are compiled out of release
fn builder_refuses_misordered_reads() {
    let refused = |f: fn(&mut TableBuilder<Device>)| {
        panic_message(move || f(&mut TableBuilder::<Device>::new()))
    };
    let m = refused(|b| {
        let r = b.output::<f64>(3, "A", 1);
        b.read(2, r);
    });
    assert!(m.contains("a row is read after its producer"), "{m}");
    let m = refused(|b| {
        let r = b.output::<f64>(3, "A", 1);
        b.read(3, r);
    });
    assert!(m.contains("a row is read after its producer"), "{m}");
    let m = refused(|b| {
        let r = b.output::<f64>(0, "A", 1);
        b.read(1, r);
        b.read(1, r);
    });
    assert!(m.contains("a node uses a row once"), "{m}");
    let m = refused(|b| {
        let r = b.output::<f64>(0, "A", 1);
        b.node(1).read(r);
        b.node(1).read(r);
    });
    assert!(m.contains("a node uses a row once"), "{m}");
    let m = refused(|b| {
        let r = b.scratch::<f64>(0, "A", 1);
        b.read(1, r);
    });
    assert!(m.contains("only an output is read by another node"), "{m}");
    let m = refused(|b| {
        let r = TableBuilder::<Device>::new().output::<f64>(0, "A", 1);
        b.read(1, r);
    });
    assert!(m.contains("a row of this table"), "{m}");
}

/// A laid-out table refuses another table's row, in every build: a row's
/// offset is what a view's pointer is made from.
#[test]
fn a_table_refuses_another_tables_row() {
    let m = panic_message(|| {
        let r = TableBuilder::<Device>::new().output::<f64>(0, "A", 1);
        let t = TableBuilder::<Device>::new().build();
        let _ = t.offset(r);
    });
    assert!(m.contains("a row of another table"), "{m}");
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

/// The host copies move a node's data in and out of its rows, and refuse a
/// source of another length.
#[test]
fn host_copies_round_trip_and_check_lengths() {
    let mut b = TableBuilder::<Heap>::new();
    let a = b.output::<f64>(0, "A", 3);
    b.read(1, a);
    let block = Block::new(0, b.build());
    memcpy_h2p(&block.partition(0), a, &[1.0, 2.0, 3.0]);
    assert_eq!(memcpy_p2h(&block.partition(1), a), vec![1.0, 2.0, 3.0]);

    let m = panic_message(|| memcpy_h2p(&block.partition(0), a, &[1.0]));
    assert!(m.contains("memcpy_h2p: the lengths differ"), "{m}");
}
