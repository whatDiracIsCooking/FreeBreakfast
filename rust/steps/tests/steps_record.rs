//! The four host steps under the allocation record (tier 3): each builds
//! its triangle in its pinned staging, so no `run` allocates one of its own
//! (`docs/memory.md`, "The budget"), past 1 MiB from about 360 AOs.
//!
//! A binary of its own: the record's peak heap growth is one count for the
//! process, which an execute here would move under `steps_host`'s case
//! reading it.

#[allow(
    dead_code,
    reason = "the shared module's orthogonalizer check is the other binaries'"
)]
mod common;

use fb_core_budget::Backing;
use fb_core_flow::{Bus, Dag, Executor};
use fb_steps::{CoreHamiltonianStep, GuessStep, MainBus, MetricStep, OverlapStep};

fn fast_tests() -> bool {
    std::env::var("FREE_BREAKFAST_FAST_TESTS").is_ok_and(|v| !v.is_empty() && !v.starts_with('0'))
}

/// `xyz`'s four host steps in cc-pVDZ / cc-pVDZ-JKFIT, budgeted and
/// executed once on a host-backed plan under the allocation record: no
/// step's `run` makes an allocation past its threshold on the thread that
/// runs it. The record does not watch the pool's threads, so each host
/// integral is then run again in the pool into a buffer of the test's own,
/// watched on the thread that `install` runs it on, where its unshared
/// work is: none past the threshold either.
#[allow(
    clippy::disallowed_macros,
    reason = "a test's buffer for the host integrals, outside any step's run"
)]
fn host_steps_pass_the_record(xyz: &str) {
    let config = common::config(xyz, "cc-pvdz", Some("cc-pvdz-jkfit"));
    let mut dag = Dag::new();
    dag.add(OverlapStep)
        .add(CoreHamiltonianStep)
        .add(GuessStep)
        .add(MetricStep);
    let plan = dag
        .build(MainBus::setup(config, Backing::Host).unwrap())
        .unwrap();
    let executor = Executor::new(plan.bus().shared().config.resources.threads);
    plan.budget(&executor).unwrap();
    plan.execute(&executor).unwrap();

    let shared = plan.bus().shared();
    let (n, q) = (shared.dims.n_ao, shared.dims.n_aux);
    let aux = shared.aux.as_ref().unwrap();
    let charges = fb_integrals_v_potential_en_host::point_charges(&shared.molecule);
    let mut s = vec![0.0; n * (n + 1) / 2];
    let mut m = vec![0.0; q * (q + 1) / 2];
    let ((), largest) = shared.host_pool().install(|| {
        fb_core_budget::AllocationRecord::watch(|| {
            fb_integrals_s_overlap_host::overlap_into(&shared.basis, &mut s);
            fb_integrals_t_kinetic_host::kinetic_into(&shared.basis, &mut s);
            fb_integrals_v_potential_en_host::potential_en_into(&shared.basis, &charges, &mut s);
            fb_integrals_df_metric::metric_into(aux, &mut m);
        })
    });
    assert_eq!(largest, None, "{xyz}: a host integral's largest allocation");
    println!(
        "{xyz}: n_ao {n}, n_aux {q}: the host steps and their integrals pass the record (packed S {:.2} MiB, (P|Q) {:.2} MiB)",
        (s.len() * 8) as f64 / f64::from(1 << 20),
        (m.len() * 8) as f64 / f64::from(1 << 20)
    );
}

/// Benzene, 114 AOs: the family's representative, its triangles under the
/// record's threshold even had they been returned.
#[test]
fn benzenes_host_steps_pass_the_allocation_record() {
    host_steps_pass_the_record("benzene.xyz");
}

/// gly10, 734 AOs and 3606 fitting functions: every triangle past the
/// threshold (S's 2.06 MiB, (P|Q)'s 49.6 MiB), so a step that returned
/// one from its library would fail. Slow: the metric's host build.
#[test]
fn gly10s_host_steps_pass_the_allocation_record() {
    if fast_tests() {
        return;
    }
    host_steps_pass_the_record("gly10.xyz");
}
