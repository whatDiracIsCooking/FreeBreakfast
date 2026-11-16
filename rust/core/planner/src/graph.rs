//! A plan's graph, detached from its work: what [`Graph::dot`] writes as
//! Graphviz, before a run and, with its record, after it.

use std::fmt::Write as _;
use std::ops::RangeInclusive;
use std::time::Duration;

use super::dag::{Plan, Work};
use fb_core_arena::{Access, Device, Heap, Kind, Owner, StagingKind, Table};
use fb_core_thread_pool::{Placement, Record};

/// A [`Plan`]'s tasks, edges, loops and memory, owned, so it outlives
/// [`Plan::instantiate`]: [`Plan::graph`] makes it, [`Graph::dot`] writes
/// it as Graphviz, joining a run's [`Record`] onto it by node rank when
/// given one.
pub struct Graph {
    nodes: Vec<GraphNode>,
    /// One per predecessor `u` of every `v`, by `v` then `u`, but an
    /// ordering with no row behind it that another path already implies.
    edges: Vec<Edge>,
    /// The orderings left out of `edges` as implied.
    implied: usize,
    loops: Vec<GraphLoop>,
    /// Each block, by name and bytes: each worker's device and host, then
    /// the heap.
    blocks: Vec<(String, usize)>,
    n_workers: usize,
}

/// A dependency `u -> v`: the rows `u` (or, out of a decision, its loop's
/// body) writes and `v` uses, each by name and bytes; none for an ordering
/// alone.
struct Edge {
    u: usize,
    v: usize,
    rows: Vec<(String, usize)>,
}

struct GraphNode {
    name: String,
    /// `gpu<w> stream`, `gpu<w> stream<l>` or `cpu`, as the plan's text
    /// names the lane.
    at: String,
    sort: Sort,
    dims: Vec<(&'static str, usize)>,
    /// One line per kind of a table the node's worker holds and the heap's,
    /// where the node makes something or something is live: the bytes it
    /// produces and the bytes live while it runs.
    memory: Vec<String>,
    /// Whether some table's peak is at it.
    peak: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Sort {
    Gpu,
    Cpu,
    Decide,
}

struct GraphLoop {
    first: usize,
    last: usize,
    /// `Some((trips, depth))` for a counted loop; `None` for a decision
    /// loop.
    counted: Option<(usize, usize)>,
    /// A decision loop's bound.
    max_iterations: usize,
    /// The innermost loop strictly holding it, by index.
    parent: Option<usize>,
}

impl<H: StagingKind, S> Plan<'_, H, S> {
    /// The plan's [`Graph`]: each task's name, lane, dimensions and memory,
    /// each dependency with the rows behind it, each loop and each block.
    /// Pure, and cheap beside planning: it copies names and sizes.
    pub fn graph(&self) -> Graph {
        let tables = |w: usize| (&self.tables[w].0, &self.tables[w].1);
        let peaks: Vec<Option<usize>> = self
            .tables
            .iter()
            .flat_map(|(d, h)| [d.peak().position, h.peak().position])
            .chain([self.heap.peak().position])
            .collect();
        let nodes = self
            .nodes
            .iter()
            .enumerate()
            .map(|(pos, node)| {
                let at = match node.placement() {
                    Placement::Stream(at) if self.lanes[at.worker] > 1 => {
                        format!("gpu{} stream{}", at.worker, at.lane)
                    }
                    Placement::Stream(at) => format!("gpu{} stream", at.worker),
                    Placement::Cpu => "cpu".to_string(),
                };
                let sort = match node.work {
                    Work::Gpu(_) => Sort::Gpu,
                    Work::Cpu(_) => Sort::Cpu,
                    Work::Decide { .. } => Sort::Decide,
                };
                let (device, host) = tables(node.at.worker);
                let memory = [
                    memory_line(device, pos),
                    memory_line(host, pos),
                    memory_line(&self.heap, pos),
                ]
                .into_iter()
                .flatten()
                .collect();
                GraphNode {
                    name: node.name.clone(),
                    at,
                    sort,
                    dims: node.dims.clone(),
                    memory,
                    peak: peaks.contains(&Some(pos)),
                }
            })
            .collect();
        let mut loops: Vec<GraphLoop> = self
            .loops
            .iter()
            .map(|l| GraphLoop {
                first: l.first,
                last: l.last,
                counted: l.counted,
                max_iterations: match self.nodes[l.last].work {
                    Work::Decide { max_iterations, .. } => max_iterations,
                    _ => 0,
                },
                parent: None,
            })
            .collect();
        for i in 0..loops.len() {
            loops[i].parent = (0..loops.len())
                .filter(|&j| {
                    j != i
                        && loops[j].first <= loops[i].first
                        && loops[i].last <= loops[j].last
                        && (loops[j].first, loops[j].last) != (loops[i].first, loops[i].last)
                })
                .min_by_key(|&j| loops[j].last - loops[j].first);
        }
        let mut edges = Vec::new();
        for (v, deps) in self.deps.iter().enumerate() {
            let (device, host) = tables(self.nodes[v].at.worker);
            for &u in deps {
                // Out of a decision to a node after its loop, the rows come
                // from the loop's body (the tables' `after_loop`).
                let writers = match self
                    .loops
                    .iter()
                    .find(|l| l.last == u && l.counted.is_none())
                {
                    Some(l) if v > u => l.first..=u,
                    _ => u..=u,
                };
                let mut rows = carried(device, &writers, v);
                rows.extend(carried(host, &writers, v));
                rows.extend(carried(&self.heap, &writers, v));
                edges.push(Edge { u, v, rows });
            }
        }
        // Every dependency points forward in rank, so each node's
        // descendants are its successors and theirs, swept from the last.
        let n = self.nodes.len();
        let mut below = vec![vec![0u64; n.div_ceil(64)]; n];
        for u in (0..n).rev() {
            for v in (u + 1..n).filter(|&v| self.deps[v].contains(&u)) {
                below[u][v / 64] |= 1 << (v % 64);
                let (lo, hi) = below.split_at_mut(v);
                for (a, b) in lo[u].iter_mut().zip(&hi[0]) {
                    *a |= b;
                }
            }
        }
        let reaches = |u: usize, p: usize| below[u][p / 64] & (1 << (p % 64)) != 0;
        let before = edges.len();
        edges.retain(|e| {
            !e.rows.is_empty() || !self.deps[e.v].iter().any(|&p| p != e.u && reaches(e.u, p))
        });
        let implied = before - edges.len();
        let mut blocks = Vec::new();
        for (w, (device, host)) in self.tables.iter().enumerate() {
            blocks.push((format!("gpu{w} {}", Device::NAME), device.total()));
            blocks.push((format!("gpu{w} {}", H::NAME), host.total()));
        }
        blocks.push((Heap::NAME.to_string(), self.heap.total()));
        Graph {
            nodes,
            edges,
            implied,
            loops,
            blocks,
            n_workers: self.tables.len(),
        }
    }
}

/// Node `pos`'s line for `table`, or `None` if the table is empty or the
/// node makes nothing in it and nothing is live there: the bytes of the
/// rows it produces, outputs and scratch, and the bytes live while it
/// runs, `(peak)` when the table's peak is there.
fn memory_line<K: Kind>(table: &Table<K>, pos: usize) -> Option<String> {
    if table.total() == 0 {
        return None;
    }
    let made = |owner: Owner| -> usize {
        table
            .rows()
            .iter()
            .filter(|r| r.owner() == owner && r.users().first().is_some_and(|&(p, _)| p == pos))
            .map(|r| r.bytes())
            .sum()
    };
    let (out, scratch, live) = (made(Owner::Edge), made(Owner::Task), table.live_at(pos));
    if out + scratch + live == 0 {
        return None;
    }
    let mut line = format!("{}:", K::NAME);
    if out > 0 {
        let _ = write!(line, " out {}", bytes(out));
    }
    if scratch > 0 {
        let _ = write!(line, " scratch {}", bytes(scratch));
    }
    let _ = write!(line, " live {}", bytes(live));
    if table.peak().position == Some(pos) {
        line.push_str(" (peak)");
    }
    Some(line)
}

/// The rows of `table` that a node in `writers` writes and `v` uses, by
/// name and bytes.
fn carried<K: Kind>(
    table: &Table<K>,
    writers: &RangeInclusive<usize>,
    v: usize,
) -> Vec<(String, usize)> {
    table
        .rows()
        .iter()
        .filter(|r| r.owner() == Owner::Edge)
        .filter(|r| {
            let users = r.users();
            users
                .iter()
                .any(|&(p, a)| writers.contains(&p) && a == Access::Write && p != v)
                && users.iter().any(|&(p, _)| p == v)
        })
        .map(|r| (r.name().to_string(), r.bytes()))
        .collect()
}

/// A run's times of one node: its entries' count and their summed host
/// and device times, `device` `None` when no entry had one.
#[derive(Clone, Copy, Default)]
struct Times {
    runs: usize,
    host: Duration,
    device: Option<Duration>,
}

impl Times {
    /// What the heat of a node is read from: its device time, else its
    /// host time.
    fn weight(self) -> Duration {
        self.device.unwrap_or(self.host)
    }
}

impl Graph {
    /// The graph as Graphviz: a `digraph` whose nodes carry each task's
    /// rank and name, its lane, its dimensions and per kind the bytes it
    /// produces and the bytes live while it runs, a table's peak drawn
    /// bold; GPU tasks are boxes, CPU tasks ellipses and decisions
    /// diamonds. An edge is labelled with the rows behind it, each by name
    /// and size, and dashed when none are (a handoff of bytes, or a
    /// decision after its body); such an ordering is drawn only when no
    /// other path implies it, and the graph's label counts those left out.
    /// Each loop is a cluster, a counted loop in a decision loop nested in
    /// it, and a decision loop has a dotted edge back to its body's first
    /// task.
    ///
    /// With `record`, the run's times are joined on by node rank: each
    /// node's host and device time, summed over its iterations, with its
    /// count when it ran more than once, filled by its share of the most
    /// any node took (its device time, else its host time); a node that
    /// did not run is grey and dashed. Each decision loop's cluster names
    /// the iterations it ran, and the graph's label the run's wall time,
    /// its profile and its error, if it failed.
    pub fn dot(&self, record: Option<&Record>) -> String {
        let times: Option<Vec<Times>> = record.map(|record| {
            let mut times = vec![Times::default(); self.nodes.len()];
            for t in &record.tasks {
                let Some(at) = times.get_mut(t.node.index()) else {
                    continue;
                };
                at.runs += 1;
                at.host += t.host;
                if let Some(d) = t.device {
                    at.device = Some(at.device.unwrap_or_default() + d);
                }
            }
            times
        });
        let hottest = times
            .iter()
            .flatten()
            .map(|t| t.weight())
            .max()
            .unwrap_or_default();

        let mut out = String::new();
        out.push_str("digraph plan {\n");
        let mut label = vec![format!(
            "plan: {} tasks, GPU workers: {}",
            self.nodes.len(),
            self.n_workers
        )];
        let blocks: Vec<String> = self
            .blocks
            .iter()
            .filter(|(_, b)| *b > 0)
            .map(|(name, b)| format!("{name} {}", bytes(*b)))
            .collect();
        if !blocks.is_empty() {
            label.push(format!("blocks: {}", blocks.join(", ")));
        }
        if self.implied > 0 {
            label.push(format!(
                "{} orderings with no row, implied by other paths, not drawn",
                self.implied
            ));
        }
        if let Some(record) = record {
            label.push(format!(
                "run: wall {}, {}",
                duration(record.wall),
                record.profile
            ));
            if let Err(e) = &record.result {
                label.push(format!("failed: {:?}: {}", e.kind, e.message));
            }
        }
        let _ = writeln!(out, "  label=\"{}\";", escape_lines(&label));
        out.push_str("  labelloc=t;\n  fontname=\"monospace\";\n");
        out.push_str("  node [fontname=\"monospace\", style=filled, fillcolor=white];\n");
        out.push_str("  edge [fontname=\"monospace\", fontsize=10];\n");

        // Each node in its innermost loop's cluster.
        let innermost = |pos: usize| {
            (0..self.loops.len())
                .filter(|&i| self.loops[i].first <= pos && pos <= self.loops[i].last)
                .min_by_key(|&i| self.loops[i].last - self.loops[i].first)
        };
        let node_line = |pos: usize, indent: &str| -> String {
            let n = &self.nodes[pos];
            let mut lines = vec![format!("{pos} {}", n.name), n.at.clone()];
            if !n.dims.is_empty() {
                let dims: Vec<String> = n.dims.iter().map(|(k, v)| format!("{k}={v}")).collect();
                lines.push(dims.join(" "));
            }
            lines.extend(n.memory.iter().cloned());
            let mut attrs = vec![format!(
                "shape={}",
                match n.sort {
                    Sort::Gpu => "box",
                    Sort::Cpu => "ellipse",
                    Sort::Decide => "diamond",
                }
            )];
            if n.peak {
                attrs.push("penwidth=2".to_string());
            }
            match times.as_ref().map(|t| t[pos]) {
                Some(t) if t.runs == 0 => {
                    lines.push("did not run".to_string());
                    attrs.push(
                        "style=\"filled,dashed\", fillcolor=gray90, fontcolor=gray40".to_string(),
                    );
                }
                Some(t) => {
                    let mut line = String::new();
                    if t.runs > 1 {
                        let _ = write!(line, "x{} ", t.runs);
                    }
                    let _ = write!(line, "host {}", duration(t.host));
                    if let Some(d) = t.device {
                        let _ = write!(line, " device {}", duration(d));
                    }
                    lines.push(line);
                    let share = if hottest.is_zero() {
                        0.0
                    } else {
                        t.weight().as_secs_f64() / hottest.as_secs_f64()
                    };
                    attrs.push(format!(
                        "fillcolor=\"0.000 {:.3} 1.000\"",
                        0.05 + 0.6 * share
                    ));
                }
                None => {}
            }
            format!(
                "{indent}n{pos} [label=\"{}\", {}];\n",
                escape_lines(&lines),
                attrs.join(", ")
            )
        };
        let iterations = |l: &GraphLoop| {
            record.and_then(|r| {
                r.iterations
                    .iter()
                    .find(|(d, _)| d.index() == l.last)
                    .map(|(_, &n)| n)
            })
        };
        let draw = Draw {
            graph: self,
            node_line: &node_line,
            innermost: &innermost,
            iterations: &iterations,
        };
        for pos in (0..self.nodes.len()).filter(|&p| innermost(p).is_none()) {
            out.push_str(&node_line(pos, "  "));
        }
        for i in (0..self.loops.len()).filter(|&i| self.loops[i].parent.is_none()) {
            draw.cluster(i, 1, &mut out);
        }

        for Edge { u, v, rows } in &self.edges {
            if rows.is_empty() {
                let _ = writeln!(out, "  n{u} -> n{v} [style=dashed, color=gray50];");
            } else {
                let rows: Vec<String> = rows
                    .iter()
                    .map(|(name, b)| format!("{name} {}", bytes(*b)))
                    .collect();
                let _ = writeln!(out, "  n{u} -> n{v} [label=\"{}\"];", escape_lines(&rows));
            }
        }
        for l in self.loops.iter().filter(|l| l.counted.is_none()) {
            let label = iterations(l).map_or("next iteration".to_string(), |n| format!("x{n}"));
            let _ = writeln!(
                out,
                "  n{} -> n{} [style=dotted, constraint=false, label=\"{label}\"];",
                l.last, l.first
            );
        }
        out.push_str("}\n");
        out
    }
}

/// What drawing a loop's cluster needs from [`Graph::dot`]: the graph, a
/// node's line, each node's innermost loop and a loop's iterations run.
struct Draw<'d> {
    graph: &'d Graph,
    node_line: &'d dyn Fn(usize, &str) -> String,
    innermost: &'d dyn Fn(usize) -> Option<usize>,
    iterations: &'d dyn Fn(&GraphLoop) -> Option<usize>,
}

impl Draw<'_> {
    /// Loop `i`'s cluster at `depth`: its label, the clusters of the loops
    /// it holds, then the nodes whose innermost loop it is.
    fn cluster(&self, i: usize, depth: usize, out: &mut String) {
        let l = &self.graph.loops[i];
        let indent = "  ".repeat(depth);
        let mut label = match l.counted {
            Some((trips, depth)) => format!(
                "counted loop {}-{}: {trips} trips, depth {depth}",
                l.first, l.last
            ),
            None => format!(
                "loop {}-{}: at most {} iterations",
                l.first, l.last, l.max_iterations
            ),
        };
        if let Some(n) = (self.iterations)(l) {
            let _ = write!(label, ", ran {n}");
        }
        let _ = writeln!(
            out,
            "{indent}subgraph cluster_loop_{}_{} {{",
            l.first, l.last
        );
        let _ = writeln!(
            out,
            "{indent}  label=\"{}\"; style=rounded; color=gray50;",
            escape(&label)
        );
        let loops = &self.graph.loops;
        for j in (0..loops.len()).filter(|&j| loops[j].parent == Some(i)) {
            self.cluster(j, depth + 1, out);
        }
        for pos in (l.first..=l.last).filter(|&p| (self.innermost)(p) == Some(i)) {
            out.push_str(&(self.node_line)(pos, &format!("{indent}  ")));
        }
        let _ = writeln!(out, "{indent}}}");
    }
}

/// `text` escaped for a quoted DOT string.
fn escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

/// `lines`, each escaped, joined by DOT's centred line break.
fn escape_lines(lines: &[String]) -> String {
    lines
        .iter()
        .map(|l| escape(l))
        .collect::<Vec<_>>()
        .join("\\n")
}

/// `n` bytes in binary units, three significant digits from KiB up.
fn bytes(n: usize) -> String {
    const UNITS: [&str; 4] = ["KiB", "MiB", "GiB", "TiB"];
    if n < 1024 {
        return format!("{n} B");
    }
    let mut value = n as f64 / 1024.0;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{} {}", three_digits(value), UNITS[unit])
}

/// `d` in the largest unit under it, three significant digits.
fn duration(d: Duration) -> String {
    let ns = d.as_nanos() as f64;
    let (value, unit) = if ns < 1e3 {
        (ns, "ns")
    } else if ns < 1e6 {
        (ns / 1e3, "us")
    } else if ns < 1e9 {
        (ns / 1e6, "ms")
    } else {
        (ns / 1e9, "s")
    };
    format!("{} {unit}", three_digits(value))
}

/// `value`, at least 1, to three significant digits.
fn three_digits(value: f64) -> String {
    if value < 10.0 {
        format!("{value:.2}")
    } else if value < 100.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.0}")
    }
}
