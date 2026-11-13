//! The records on a card (tier 4, device 0) against the C++ tree's, dumped
//! on CUDA by `tools/mmd_records` (`tests/data/README.md`), for
//! H2O/cc-pVDZ and CO/cc-pVTZ.
//!
//! The bars are fixed by analysis, one run:
//!
//! - The class list, each `n_rec`, and every integer field (`off_*`,
//!   `rank_*`) exactly: they are the same sweep of the same basis.
//! - `zeta`, `ea` and `eb` bitwise: exponents read from the same JSON, and
//!   ζ their sum, one rounding on both sides. Equal ζ keep the enumeration
//!   order, so this is the record order too.
//! - Every other field within 1e-13 relative: the coefficients come out of
//!   each tree's host normalization (`pow`, `sqrt`), whose libm rounding may
//!   differ by an ulp, carried through a few device operations (`s`, and
//!   the dressed bounds through it). A field the reference holds as exactly
//!   0 must be exactly 0.

mod common;

use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{self as thread_pool, DeviceStream, Lane};
use fb_core_wwr::gpu_status;
use fb_core_wwr::handle::DeviceHandleStream;
use fb_integrals_mmd::{ClassRows, add_records};
use wwr_sys::wwr_memcpy_kind;

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

/// One class's records on the host, every field in record order.
#[derive(Debug, Default)]
struct HostClass {
    la: i32,
    lb: i32,
    n_rec: i32,
    f64s: Vec<Vec<f64>>,
    i32s: Vec<Vec<i32>>,
}

/// The class's double rows, in the reference file's order.
fn f64_fields(c: &ClassRows) -> [RowId<Device, f64>; 16] {
    [
        c.zeta, c.p_x, c.p_y, c.p_z, c.pa_x, c.pa_y, c.pa_z, c.pb_x, c.pb_y, c.pb_z, c.s, c.ea,
        c.eb, c.bound, c.bound_ab, c.bound_ba,
    ]
}

/// Its integer rows, likewise.
fn i32_fields(c: &ClassRows) -> [RowId<Device, i32>; 4] {
    [c.off_a, c.off_b, c.rank_a, c.rank_b]
}

/// A class's pinned copies.
#[derive(Clone)]
struct PinnedClass {
    f64s: Vec<RowId<Pinned, f64>>,
    i32s: Vec<RowId<Pinned, i32>>,
}

/// GPU: every field of every class copied down into pinned rows on the
/// stream.
struct Download(Vec<ClassRows>);

impl Task for Download {
    type Rows = Vec<PinnedClass>;
    const GPU: bool = true;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        self.0
            .iter()
            .map(|c| {
                let n = c.n_rec as usize;
                let (f, i) = (f64_fields(c), i32_fields(c));
                PinnedClass {
                    f64s: f
                        .iter()
                        .map(|&r| d.read(r))
                        .map(|_| h.output("down", n))
                        .collect(),
                    i32s: i
                        .iter()
                        .map(|&r| d.read(r))
                        .map(|_| h.output("down", n))
                        .collect(),
                }
            })
            .collect()
    }

    fn run(
        &self,
        rows: &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        for (c, down) in self.0.iter().zip(rows) {
            let (f, i) = (f64_fields(c), i32_fields(c));
            for (&src, &dst) in f.iter().zip(&down.f64s) {
                let (src, dst) = (d.get(src), h.take_mut(dst));
                // SAFETY: two rows of n_rec doubles, on the task's stream; a
                // stream task hands the pinned row to the copy and no more.
                gpu_status(unsafe {
                    wwr_sys::wwr_memcpy_async(
                        dst.as_mut_ptr().cast(),
                        src.as_ptr().cast(),
                        size_of_val(dst),
                        wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_HOST,
                        stream.stream(),
                    )
                })?;
            }
            for (&src, &dst) in i.iter().zip(&down.i32s) {
                let (src, dst) = (d.get(src), h.take_mut(dst));
                // SAFETY: as above, n_rec int32s.
                gpu_status(unsafe {
                    wwr_sys::wwr_memcpy_async(
                        dst.as_mut_ptr().cast(),
                        src.as_ptr().cast(),
                        size_of_val(dst),
                        wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_HOST,
                        stream.stream(),
                    )
                })?;
            }
        }
        Ok(())
    }
}

/// CPU: the pinned copies read into `out`, after the executor waited for
/// the download.
struct Collect {
    classes: Vec<ClassRows>,
    down: Vec<PinnedClass>,
    out: Arc<Mutex<Vec<HostClass>>>,
}

impl Task for Collect {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) {
        for c in &self.down {
            c.f64s.iter().for_each(|&r| {
                h.read(r);
            });
            c.i32s.iter().for_each(|&r| {
                h.read(r);
            });
        }
    }

    fn run(
        &self,
        _: &(),
        _: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        let mut out = self.out.lock().unwrap();
        *out = self
            .classes
            .iter()
            .zip(&self.down)
            .map(|(c, down)| HostClass {
                la: c.la,
                lb: c.lb,
                n_rec: c.n_rec,
                f64s: down.f64s.iter().map(|&r| h.get(r).to_vec()).collect(),
                i32s: down.i32s.iter().map(|&r| h.get(r).to_vec()).collect(),
            })
            .collect();
        Ok(())
    }
}

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

/// Each molecule's records planned, run once on device 0 and compared with
/// the C++ tree's against the bars above. Every field's largest error is
/// printed, and every miss named before the test fails. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn records_match_the_cpp_tree() {
    let mut misses = Vec::new();
    for (stem, basis) in common::cases() {
        let out = Arc::new(Mutex::new(Vec::new()));
        let mut dag = Dag::new();
        let classes = add_records(&mut dag, 0, &basis).unwrap();
        let down = dag.add("download", 0, Download(classes.clone()));
        dag.add(
            "collect",
            0,
            Collect {
                classes,
                down,
                out: Arc::clone(&out),
            },
        );
        let instance = dag.plan().instantiate(&[0]);
        let backend = thread_pool::Device::new(&[0], instance.plan());
        instance.run(&backend, 1).result.unwrap();
        let got = std::mem::take(&mut *out.lock().unwrap());
        let want = reference(stem);

        let list = |c: &[HostClass]| c.iter().map(|c| (c.la, c.lb, c.n_rec)).collect::<Vec<_>>();
        assert_eq!(
            list(&got),
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
                    // The largest relative error, infinite for a nonzero
                    // value where the reference holds exactly 0.
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
    assert!(misses.is_empty(), "past the bars:\n{}", misses.join("\n"));
}
