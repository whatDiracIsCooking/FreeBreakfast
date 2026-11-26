//! `libfb_localization_kernels.so`'s C ABI: the launch wrappers rust/ has
//! forked from `src/localization`, as raw FFI (`docs/gpu_layer.md`,
//! "Layering"), and the Pipek-Mezey launches that are their safe calls.
//!
//! The bindings are generated from `localization/kernels/fb_localization.h`
//! by `devtools/rust-bindings.py` and committed; `fb-device-check`
//! regenerates and diffs them. Every function is `unsafe`: the C side trusts
//! its device pointers and lengths, and three of its launches trust indices
//! held in device memory (the atoms' AO offsets, a round's orbital pairs and
//! the partners the pairs give), which no host check can read.
//!
//! | C++ (`src/localization/pipek_mezey`) | Here |
//! |---|---|
//! | `device::populations(stream, n_atoms, n_occ, ld, c, w, atom_ao_offsets, pops)` | [`fb_pm_populations`], counts below 1 and null pointers refused; the safe call is [`pm_populations`] |
//! | `device::pair_angles(stream, exponent, …)`, any exponent but 4 taken as 2 | [`fb_pm_pair_angles`], an exponent other than 2 or 4 refused |
//! | `device::rotate_populations`, `device::rotate_columns` | [`fb_pm_rotate_populations`], [`fb_pm_rotate_columns`] |
//! | the driver's one sweep: `n_occ - 1` rounds of the three launches above | [`pm_sweep`] |
//! | `device::functional`, `device::identity` | [`fb_pm_functional`], [`fb_pm_identity`]; the safe calls are [`pm_functional`] and [`pm_identity`] |
//! | `device::kMaxOccupied` | [`FB_PM_MAX_OCCUPIED`] |
//! | `pm_fill_round_robin(m, pairs)`, `pm_schedule_size(n_occ)` | [`pm_schedule`], which writes the same tournament into its host end and uploads it into a [`Schedule`], and [`pm_schedule_len`] |
//! | `pm_atom_ao_offsets(basis)` | the caller's: [`pm_populations`] takes the offsets and checks them |
//! | `wwrError_t` | [`fb_launch_status`]: 0, a positive `wwr_status`, or a negative refusal ([`FB_LAUNCH_INVALID_ARGUMENT`]) |
//! | nothing: the C++ tree compiles its kernels into the binary | [`check_library`], the library's [`fb_localization_abi_version`] and [`fb_localization_backend`] against the bindings' |
//! | `localization/ibo`'s IAO populations and the rest of `src/localization` | not forked yet |
//!
//! **The safe layer** is the v0.0.4 layer's, for `fb_steps::LocalizationStep`,
//! over `fb-core-device`'s stream and slices; the module `sweep`'s doc
//! comment has why each launch may trust the indices it follows
//! (`docs/gpu_layer.md`, "The safe wrapper").

#![allow(non_camel_case_types, reason = "the C header's names, kept as written")]

#[rustfmt::skip]
mod bindings;
mod sweep;

pub use bindings::*;
pub use sweep::{
    Angles, Schedule, pm_functional, pm_identity, pm_populations, pm_schedule, pm_schedule_len,
    pm_sweep,
};

use fb_kernel_abi::Backend;

/// Checks that the `libfb_localization_kernels.so` this process loaded is
/// the one these bindings describe: its [`fb_localization_abi_version`]
/// against [`FB_LOCALIZATION_ABI_VERSION`], and its
/// [`fb_localization_backend`] against the backend of the CMake build this
/// crate linked (`docs/gpu_layer.md`, "What the port changes"). Call it
/// once, before the first launch. Needs no device. The comparison is
/// [`fb_kernel_abi::check`], tested where no library is built.
///
/// # Errors
///
/// A mismatch, with a message naming the found and the expected values.
pub fn check_library() -> Result<(), String> {
    // SAFETY: a host function with no arguments, which calls no runtime.
    let version = unsafe { fb_localization_abi_version() };
    // SAFETY: as above.
    let backend = unsafe { fb_localization_backend() };
    let built_for = const { Backend::named(env!("FB_KERNELS_BACKEND")) };
    fb_kernel_abi::check(
        "localization",
        (version, backend),
        (FB_LOCALIZATION_ABI_VERSION, built_for),
    )
}

/// The schedule's index count `m`: `n_occ`, padded to even with one dummy
/// index, `n_occ` itself, which becomes the pivot.
fn padded(n_occ: usize) -> usize {
    n_occ + n_occ % 2
}

/// Fill `pairs`, `(m - 1) m` entries, with a round-robin ("circle method")
/// tournament over `m` indices, `m` even and at least 2: `m - 1` rounds of
/// `m / 2` disjoint pairs, round `r` at `r m` as consecutive (first,
/// second) entries, index `m - 1` the fixed pivot and the rest rotating
/// past it. The C++ `pm_fill_round_robin`, the same entries in the same
/// order.
fn fill_round_robin(m: usize, pairs: &mut [i32]) {
    let pivot = m - 1;
    for r in 0..pivot {
        let round = &mut pairs[r * m..(r + 1) * m];
        // Pair 0 draws the pivot against the round number; the rest fan out
        // around it, one step further apart per pair.
        round[0] = pivot as i32;
        round[1] = r as i32;
        for i in 1..m / 2 {
            round[2 * i] = ((r + i) % pivot) as i32;
            round[2 * i + 1] = ((r + pivot - i) % pivot) as i32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every round of the tournament over m = 2 to 12 pairs each index below
    /// m exactly once, and the m - 1 rounds together pair every two indices
    /// exactly once. What the kernels trust of the schedule: catches a pair
    /// repeated or an orbital left out of a round, which would leave its
    /// partner unwritten.
    #[test]
    fn each_round_pairs_every_index_once_and_a_sweep_every_pair() {
        for m in (2..=12).step_by(2) {
            let mut pairs = vec![-1; (m - 1) * m];
            fill_round_robin(m, &mut pairs);
            let mut met = vec![0; m * m];
            for round in pairs.chunks(m) {
                let mut seen = vec![0; m];
                for pair in round.chunks(2) {
                    let (p, q) = (pair[0] as usize, pair[1] as usize);
                    seen[p] += 1;
                    seen[q] += 1;
                    met[p.min(q) * m + p.max(q)] += 1;
                }
                assert!(seen.iter().all(|&n| n == 1), "m = {m}: {round:?}");
            }
            for p in 0..m {
                for q in p + 1..m {
                    assert_eq!(met[p * m + q], 1, "m = {m}: ({p}, {q})");
                }
            }
        }
    }

    /// The padding: an odd count's one dummy index is the pivot.
    #[test]
    fn an_odd_count_is_padded_by_one() {
        assert_eq!(padded(5), 6);
        assert_eq!(padded(1), 2);
        assert_eq!(padded(8), 8);
    }
}
