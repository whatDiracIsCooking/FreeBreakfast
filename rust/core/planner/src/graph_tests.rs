//! [`Graph`]'s tests (tier 3): the SCF loop's DOT a golden, the structure
//! every graph keeps, and a run's times joined on.

use std::sync::Arc;

use super::tests::{Decl, Declares, Stops, fixed_point_dag, scf_loop_dag};
use super::*;
use fb_core_arena::PinnedStandIn;
use fb_core_thread_pool::HostOnly;
use fb_test_support::golden;

/// The lines of `dot` that declare node `pos`.
fn declarations(dot: &str, pos: usize) -> usize {
    let head = format!("n{pos} [");
    dot.lines()
        .filter(|l| l.trim_start().starts_with(&head))
        .count()
}

/// Whether the drawn edges of `dot` lead from `u` to `v`.
fn drawn_path(dot: &str, u: usize, v: usize) -> bool {
    let drawn: Vec<(usize, usize)> = dot
        .lines()
        .filter_map(|l| {
            let (a, rest) = l.trim_start().strip_prefix('n')?.split_once(" -> n")?;
            let b = rest.split_once(' ')?.0;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .filter(|(a, b)| a < b)
        .collect();
    let mut reached = vec![u];
    let mut i = 0;
    while i < reached.len() {
        let at = reached[i];
        for &(a, b) in &drawn {
            if a == at && !reached.contains(&b) {
                reached.push(b);
            }
        }
        i += 1;
    }
    reached.contains(&v)
}

/// The lines of `dot` that are the edge `u -> v`.
fn edges(dot: &str, u: usize, v: usize) -> Vec<&str> {
    let head = format!("n{u} -> n{v} [");
    dot.lines()
        .filter(|l| l.trim_start().starts_with(&head))
        .collect()
}

/// The SCF loop's graph is a golden: each task's lane and memory, the
/// device table's peak bold at the Fock build, each edge's rows and the
/// loop's cluster and back edge.
#[test]
fn scf_loop_graph_golden() {
    let planned = scf_loop_dag(24).plan();
    let dot = planned.graph().dot(None);
    golden(env!("CARGO_MANIFEST_DIR"), "planner_scf_loop.dot", &dot);
    // S, 24² doubles, from the overlap to the orthogonalizer.
    assert_eq!(edges(&dot, 0, 2), ["  n0 -> n2 [label=\"S 4.50 KiB\"];"]);
    // No row is behind an edge from the residual: nothing reads its e, so
    // the diagonalization follows it for e's bytes alone, drawn dashed;
    // the decision follows it for closing the body, which the path
    // through the diagonalization already implies, so that is not drawn.
    assert!(edges(&dot, 5, 6)[0].contains("style=dashed"), "{dot}");
    assert!(edges(&dot, 5, 8).is_empty(), "{dot}");
    assert!(dot.contains("orderings with no row, implied by other paths, not drawn"));
}

/// Every task is declared once, in the innermost loop's cluster, a counted
/// loop's nested in its decision loop's; every dependency is at most one
/// edge, one with a row behind it always, any other left out only where
/// the drawn edges still lead there; a name with a quote is escaped; and
/// an untimed graph says nothing of a run.
#[test]
fn a_graph_declares_each_task_and_dependency_once() {
    use Decl::*;
    let mut dag = Dag::<PinnedStandIn, ()>::new();
    let a = dag.add_gpu("a \"quoted\"", 0, Declares(vec![Out("A", 4)]))[0];
    let body = dag.len();
    let b = dag.add_gpu("b", 0, Declares(vec![Read(a), Out("B", 4)]))[0];
    let inner = dag.len();
    dag.add_gpu("c", 0, Declares(vec![Read(b), Scratch("C", 4)]));
    dag.add_counted(inner, 3, 2);
    dag.add_loop("d", 0, body, 5, Stops);
    let planned = dag.plan();
    let dot = planned.graph().dot(None);
    for pos in 0..planned.len() {
        assert_eq!(declarations(&dot, pos), 1, "node {pos}: {dot}");
        for &u in planned.predecessors(pos) {
            let drawn = edges(&dot, u, pos);
            assert!(drawn.len() <= 1, "{u} -> {pos}: {dot}");
            assert!(drawn_path(&dot, u, pos), "{u} -> {pos} implied: {dot}");
        }
    }
    assert!(dot.contains("0 a \\\"quoted\\\""), "{dot}");
    // A's reader and B's reader each keep their edge, 4 doubles each.
    assert!(edges(&dot, 0, 1)[0].contains("A 32 B"), "{dot}");
    assert!(edges(&dot, 1, 2)[0].contains("B 32 B"), "{dot}");
    let outer = dot.find("subgraph cluster_loop_1_3 {").expect(&dot);
    let nested = dot.find("    subgraph cluster_loop_2_2 {").expect(&dot);
    assert!(outer < nested, "{dot}");
    assert!(dot.contains("counted loop 2-2: 3 trips, depth 2"), "{dot}");
    assert!(dot.contains("loop 1-3: at most 5 iterations"), "{dot}");
    assert!(!dot.contains("host "), "{dot}");
}

/// A run's record joined on by rank: the loop's 41 iterations on its
/// cluster and back edge, each body task's count and time, the edges out
/// of the loop through its decision carrying `x`, and the run's wall time
/// in the label.
#[test]
fn a_record_puts_the_times_on_the_graph() {
    let out = Arc::new(std::sync::Mutex::new(0.0));
    let planned = fixed_point_dag(&out).plan();
    let graph = planned.graph();
    let mut instance = planned.instantiate(&[0]);
    let record = instance.run(&mut HostOnly::new(1), 2);
    record.result.as_ref().unwrap();
    let dot = graph.dot(Some(&record));
    assert!(
        dot.contains("loop 1-3: at most 100 iterations, ran 41"),
        "{dot}"
    );
    assert!(
        dot.contains("n3 -> n1 [style=dotted, constraint=false, label=\"x41\"]"),
        "{dot}"
    );
    let step = dot.lines().find(|l| l.contains("1 step")).expect(&dot);
    assert!(step.contains("x41 host "), "{step}");
    let report = dot.lines().find(|l| l.contains("4 report")).expect(&dot);
    assert!(
        report.contains("\\nhost ") && !report.contains("x1 "),
        "{report}"
    );
    assert_eq!(
        edges(&dot, 3, 4),
        ["  n3 -> n4 [label=\"x 8 B\"];"],
        "{dot}"
    );
    assert!(dot.contains("run: wall "), "{dot}");
    assert!(!dot.contains("did not run"), "{dot}");
}
