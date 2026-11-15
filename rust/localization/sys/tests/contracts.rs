//! [`add_pm_start`]'s always-on contracts (tier 3): each refuses its
//! arguments at planning, before anything is added. Planning touches no
//! device.

use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_localization_sys::{FB_PM_MAX_OCCUPIED, add_pm_start};

/// GPU: an input row of `len` doubles, declared and never computed.
struct Input(&'static str, usize);

impl GpuTask for Input {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.output(self.0, self.1)
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

/// `add_pm_start` over `offsets`, `n_ao` and `n_occ`, with S and C rows
/// sized by `n_ao` and `n_occ`, capped so a huge count declares no huge
/// row.
fn start(offsets: &[usize], n_ao: usize, n_occ: usize) {
    let mut dag = Dag::new();
    let (ao, occ) = (n_ao.min(8), n_occ.min(8));
    let s = dag.add_gpu("S", 0, Input("S", ao * ao));
    let c = dag.add_gpu("C", 0, Input("C", ao * occ.max(1)));
    let _ = add_pm_start(&mut dag, 0, offsets, n_ao, n_occ, 2, s, c);
}

#[test]
#[should_panic(expected = "Pre-condition of add_pm_start violated: n_ao fits the kernels' i32")]
fn an_n_ao_past_i32_is_refused() {
    let n_ao = i32::MAX as usize + 1;
    start(&[0, n_ao], n_ao, 2);
}

#[test]
#[should_panic(
    expected = "Pre-condition of add_pm_start violated: the offsets bound one atom or more"
)]
fn a_single_offset_is_refused() {
    start(&[0], 4, 2);
}

#[test]
#[should_panic(expected = "Pre-condition of add_pm_start violated: the offsets do not decrease")]
fn decreasing_offsets_are_refused() {
    start(&[0, 3, 2, 4], 4, 2);
}

#[test]
#[should_panic(expected = "Pre-condition of add_pm_start violated: the offsets end within n_ao")]
fn offsets_past_n_ao_are_refused() {
    start(&[0, 2, 5], 4, 2);
}

#[test]
#[should_panic(expected = "Pre-condition of add_pm_start violated: one occupied orbital or more")]
fn no_occupied_orbital_is_refused() {
    start(&[0, 2, 4], 4, 0);
}

#[test]
#[should_panic(expected = "Pre-condition of add_pm_start violated: n_occ is within n_ao")]
fn an_n_occ_past_n_ao_is_refused() {
    start(&[0, 2, 4], 4, 5);
}

#[test]
#[should_panic(
    expected = "Pre-condition of add_pm_start violated: n_occ is within FB_PM_MAX_OCCUPIED"
)]
fn an_n_occ_past_the_kernels_cap_is_refused() {
    let n_occ = FB_PM_MAX_OCCUPIED as usize + 1;
    start(&[0, n_occ], n_occ, n_occ);
}
