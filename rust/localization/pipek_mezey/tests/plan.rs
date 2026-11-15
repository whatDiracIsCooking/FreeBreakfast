//! The plan of water's localization in cc-pVDZ (`n_ao = 24`, three atoms,
//! `n_occ = 5`, so the schedule is padded to 6), tier 3: S and C from a
//! task that declares them and is never run, then `pm tables`, its two
//! uploads and `pm start`, `pm before`, the loop of `pm sweep` and `pm
//! converged`, and `pm rotate` after it. The plan text, each worker's device
//! and pinned table and the heap table among it, is the committed golden
//! `tests/golden/plan_h2o_cc-pvdz.txt`, `FB_BLESS=1` rewriting it
//! (`docs/testing.md`, "Performance"). Planning touches no device; it does
//! load `libfb_localization_kernels.so`, whose load check planning runs.

use std::path::Path;

use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_localization_pipek_mezey::{PmOptions, add_pm_localize};
use fb_test_support::golden;

/// GPU: S and C, declared and never computed.
struct Inputs {
    n: usize,
}

impl GpuTask for Inputs {
    type Rows = (RowId<Device, f64>, RowId<Device, f64>);

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        (
            device.output("S", self.n * self.n),
            device.output("C", self.n * self.n),
        )
    }

    fn run(
        &self,
        _: &Self::Rows,
        _: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        _: &DeviceStream,
    ) -> Status {
        unreachable!("the plan is never run")
    }
}

#[test]
fn plan_golden() {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/h2o.xyz");
    let molecule = fb_io_xyz::read_xyz(&data, 0, 1).unwrap();
    let basis = fb_io_basis::resolve_basis("cc-pvdz")
        .unwrap()
        .primitive_normalized()
        .place(&molecule)
        .unwrap();
    let n = basis.nao_sphe() as usize;
    assert_eq!(n, 24);
    let mut dag = Dag::new();
    let (s, c) = dag.add_gpu("inputs", 0, Inputs { n });
    let (info, [c_local, u]) =
        add_pm_localize(&mut dag, 0, &PmOptions::default(), &basis, 5, s, c).unwrap();
    let planned = dag.plan();
    assert_eq!(planned.heap_table().len(info), 3);
    assert_eq!(planned.device_table(0).len(c_local), n * 5);
    assert_eq!(planned.device_table(0).len(u), 25);
    // The table's text ends an empty handoffs line in a space, which the
    // trailing-whitespace hook strips from the committed file.
    let text: String = planned
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect();
    golden(env!("CARGO_MANIFEST_DIR"), "plan_h2o_cc-pvdz.txt", &text);
}
