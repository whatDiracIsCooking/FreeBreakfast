//! The localization step on a card (tier 4): water's DF-RHF in cc-pVDZ
//! over cc-pVDZ-JKFIT, its occupied orbitals localized at both exponents
//! under the NaN fill and the allocation record, written and read back.
//! The sweep body against PySCF's maximum is the fixture case in
//! `src/steps/localization.rs`; this is the step on the chain: its order,
//! its products, and that what it wrote is orthonormal localized orbitals.
//!
//! **The bars, by analysis, one run.**
//!
//! - `c_local.npy` is the host copy's bits: no arithmetic between them.
//! - **The order.** On the host, from the written C and the orbital
//!   energies of `scf.json`, `F = S C diag(ε) Cᵀ S` (C diagonalizes the
//!   Fock matrix its last Roothaan step solved), with S the host's
//!   overlap, and `f_i = c_iᵀ F c_i` for each written column. The step's
//!   key is `Σ_k ε_k U_ki²`, the same number by another association, each
//!   off by round-off in sums of a few dozen terms of at most `max|ε|`
//!   (about 20 Eh for water's O 1s) and by `Cᵀ S C - I`, of order 1e-14:
//!   1e-12 Eh. Each `f_i` is at least `f_{i-1} - 1e-10` Eh, a margin of
//!   two decades, so a near-tie may swap within it and nothing else.
//! - `C_localᵀ S C_local = I` to 1e-10, which catches a column copied
//!   twice or left unwritten (NaN under the fill): the SCF's commutator
//!   tolerance of 1e-7 bounds nothing here, since the rotation is exact
//!   within the occupied space, and S's host and device builds agree to
//!   1e-12 of the largest element.
//! - L after is at least L before, and the run took a sweep.

#![allow(
    clippy::disallowed_macros,
    reason = "a test's paths and host reads, outside any step's run"
)]

#[allow(
    dead_code,
    reason = "the shared module's orthogonalizer check is the other binaries'"
)]
mod common;

use fb_core_budget::Backing;
use fb_core_flow::Bus;
use fb_steps::{LocalizationStep, LocalizationTable, ScfStep, ScfTable, run};
use serde_json::Value;

/// The full symmetric matrix of the packed lower triangle `packed`.
fn unpack(packed: &[f64], n: usize) -> Vec<f64> {
    let mut full = vec![0.0; n * n];
    let mut at = 0;
    for p in 0..n {
        for q in 0..=p {
            full[p * n + q] = packed[at];
            full[q * n + p] = packed[at];
            at += 1;
        }
    }
    full
}

/// `a b`, `a` `r × k` and `b` `k × c`, column-major.
fn mul(a: &[f64], b: &[f64], r: usize, k: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0; r * c];
    for j in 0..c {
        for l in 0..k {
            let blj = b[l + k * j];
            for i in 0..r {
                out[i + r * j] += a[i + r * l] * blj;
            }
        }
    }
    out
}

/// `aᵀ`, `a` `r × c`.
fn transpose(a: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0; r * c];
    for j in 0..c {
        for i in 0..r {
            out[j + c * i] = a[i + r * j];
        }
    }
    out
}

/// Water localized at `exponent`, written into a directory of its own,
/// and its products checked against the module's bars.
fn water_at(exponent: i32) {
    let mut config = common::config("h2o.xyz", "cc-pvdz", Some("cc-pvdz-jkfit"));
    config.scf = Some(ScfTable::default());
    config.localization = Some(LocalizationTable {
        exponent,
        ..LocalizationTable::default()
    });
    config.resources.nan_fill = true;
    let run = run(config, Backing::Real).expect("the chain runs");
    let dir = std::env::temp_dir().join(format!(
        "fb-steps-localization-device-{}-{exponent}",
        std::process::id()
    ));
    run.write(&dir).unwrap();

    let shared = run.plan.bus().shared();
    let (n, m) = (shared.dims.n_ao, shared.dims.n_occ);
    let local = run.plan.output(LocalizationStep);
    let array = fb_io_npy::read_npy(&dir.join("c_local.npy")).unwrap();
    assert_eq!(array.shape, [n, m]);
    assert!(
        array
            .data
            .iter()
            .zip(local.c())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "c_local.npy is not the host copy"
    );
    let c_local = array.data;
    let c = fb_io_npy::read_npy(&dir.join("c.npy")).unwrap().data;
    let scf: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("scf.json")).unwrap()).unwrap();
    let epsilon: Vec<f64> = scf["epsilon"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e.as_f64().unwrap())
        .collect();
    let s = unpack(&fb_integrals_s_overlap_host::overlap(&shared.basis), n);

    // F = (S C) diag(ε) (S C)ᵀ, then f_i = c_iᵀ F c_i.
    let sc = mul(&s, &c, n, n, n);
    let mut sc_eps = sc.clone();
    for (j, e) in epsilon.iter().enumerate() {
        sc_eps[n * j..n * (j + 1)].iter_mut().for_each(|v| *v *= e);
    }
    let f = mul(&sc_eps, &transpose(&sc, n, n), n, n, n);
    let fc = mul(&f, &c_local, n, n, m);
    let key: Vec<f64> = (0..m)
        .map(|i| (0..n).map(|r| c_local[r + n * i] * fc[r + n * i]).sum())
        .collect();
    println!("water p={exponent}: <i|F|i> = {key:.6?}");
    for i in 1..m {
        assert!(
            key[i] >= key[i - 1] - 1e-10,
            "p={exponent}: column {i}'s <F> {} below column {}'s {}",
            key[i],
            i - 1,
            key[i - 1]
        );
    }

    let gram = mul(
        &transpose(&c_local, n, m),
        &mul(&s, &c_local, n, n, m),
        m,
        n,
        m,
    );
    let off = (0..m * m)
        .map(|k| (gram[k] - f64::from(u8::from(k % (m + 1) == 0))).abs())
        .fold(0.0_f64, |a, b| if b.is_nan() { b } else { a.max(b) });
    assert!(
        off <= 1e-10,
        "p={exponent}: |C_localᵀ S C_local - I| = {off:e}"
    );

    let json: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("localization.json")).unwrap())
            .unwrap();
    let (before, after) = (
        json["functional_before"].as_f64().unwrap(),
        json["functional_after"].as_f64().unwrap(),
    );
    assert_eq!(before.to_bits(), local.functional_before().to_bits());
    assert_eq!(after.to_bits(), local.functional_after().to_bits());
    assert_eq!(json["sweeps"], local.sweeps());
    assert_eq!(json["exponent"], exponent);
    assert!(
        after >= before && local.sweeps() >= 1,
        "p={exponent}: L {before} -> {after} in {} sweeps",
        local.sweeps()
    );
    assert!(run.plan.output(ScfStep).converged());
    println!(
        "water p={exponent}: L {before:.12} -> {after:.12} in {} sweeps, |C_localᵀ S C_local - I| {off:.1e}",
        local.sweeps()
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Water at `L_2` and `L_4`: the family's representative, the chain once
/// per exponent. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn waters_localized_orbitals_are_written_in_fock_order() {
    water_at(2);
    water_at(4);
}
