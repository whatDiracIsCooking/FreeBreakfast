//! The Pipek-Mezey launches on the v0.0.4 layer, over `fb-core-device`'s
//! stream and slices: [`pm_populations`], [`pm_schedule`] and
//! [`pm_sweep`], [`pm_functional`] and [`pm_identity`]
//! (`docs/gpu_layer.md`, "The v0.0.4 runtime", "The safe wrapper").
//!
//! **Why a launch may trust its indices.** Three launches follow indices
//! held in device memory. The atoms' AO offsets come in on the host and
//! are checked and uploaded by [`pm_populations`] itself, in the call that
//! launches over them, as `fb-integrals-sys`'s three-center call does with
//! its cells. The round-robin schedule is written by [`pm_schedule`] into
//! the host end it is handed and uploaded into a device slice it then
//! keeps, by value, in a [`Schedule`] whose fields are private: nothing but
//! this crate can name those device bytes until the `Schedule` drops, and
//! [`pm_sweep`] reads them only through it. The partners the rotations
//! follow are written by the pair kernel inside [`pm_sweep`], into the
//! caller's slice, just before the rotations on the same stream.

use std::ops::DerefMut;

use contracts::requires;
use fb_core_device::{
    DeviceSlice, DeviceSliceMut, HostSliceMut, Pending, ScopedDevice, Stream, part_mut,
};
use fb_core_error_handling::{Result, Status};
use fb_core_log::{eq, fits_i32, le};

use crate::{
    FB_PM_MAX_OCCUPIED, fb_pm_functional, fb_pm_identity, fb_pm_pair_angles, fb_pm_populations,
    fb_pm_rotate_columns, fb_pm_rotate_populations, fill_round_robin, launched, padded,
};

/// The schedule's length for `n_occ` orbitals: `m - 1` rounds of `m`
/// indices, `m` the count padded to even.
#[must_use]
pub fn pm_schedule_len(n_occ: usize) -> usize {
    let m = padded(n_occ);
    (m - 1) * m
}

/// The round-robin schedule of `n_occ` orbitals on the device, as
/// [`pm_schedule`] wrote and uploaded it: the device slice is held here by
/// value, so no one else writes it while [`pm_sweep`] trusts it.
pub struct Schedule<D> {
    dev: D,
    n_occ: usize,
}

impl<D> Schedule<D> {
    /// The orbitals it schedules.
    #[must_use]
    pub fn n_occ(&self) -> usize {
        self.n_occ
    }
}

/// The round-robin tournament over `n_occ` orbitals (`m - 1` rounds of
/// `m / 2` disjoint pairs, `m` the count padded to even, the padding index
/// `n_occ` sitting out) written into `host` and uploaded into the head of
/// `dev` on `stream`; the [`Schedule`] keeps `dev`, and the upload's token
/// holds `host` until a sync of `stream` has covered the copy.
///
/// # Errors
/// The upload's failure.
#[requires(
    le!(1, n_occ) && le!(n_occ, FB_PM_MAX_OCCUPIED as usize),
    "pm_schedule: n_occ is within 1..=FB_PM_MAX_OCCUPIED"
)]
#[requires(
    le!(pm_schedule_len(n_occ), host.len()),
    "pm_schedule: host holds the schedule"
)]
#[requires(
    le!(pm_schedule_len(n_occ), dev.len()),
    "pm_schedule: dev holds the schedule"
)]
#[track_caller]
pub fn pm_schedule<'s, H, D>(
    stream: &'s Stream,
    n_occ: usize,
    mut host: H,
    mut dev: D,
) -> Result<(Schedule<D>, Pending<'s, H>)>
where
    H: HostSliceMut<i32> + DerefMut<Target = [i32]>,
    D: DeviceSliceMut<i32>,
{
    let len = pm_schedule_len(n_occ);
    fill_round_robin(padded(n_occ), &mut host[..len]);
    let uploaded = stream.upload(&mut part_mut(&mut dev, 0, len), host)?;
    Ok((Schedule { dev, n_occ }, uploaded))
}

/// The symmetrized Mulliken populations `Q^A_ij` of the `n_occ` columns of
/// `c` (leading dimension `ld`) and `w = S C` into `pops`, atom-fastest
/// ([`fb_pm_populations`]), on `stream`. Atom `A` owns the AOs from
/// `offsets[A]` up to `offsets[A + 1]`: `offsets` is checked here, then
/// uploaded into the head of `offsets_dev` just before the launch; the
/// upload's token holds it until a sync of `stream` has covered the copy.
///
/// # Errors
/// The upload's failure, or the launch's.
///
/// # Panics
/// If `offsets` decrease or pass `ld`, naming the entry, before anything
/// is enqueued.
#[allow(clippy::too_many_arguments, reason = "the kernel's arguments")]
#[requires(
    le!(1, n_occ) && le!(n_occ, FB_PM_MAX_OCCUPIED as usize),
    "pm_populations: n_occ is within 1..=FB_PM_MAX_OCCUPIED"
)]
#[requires(
    le!(1, ld) && fits_i32!(ld.checked_mul(n_occ)),
    "pm_populations: ld × n_occ is a positive i32"
)]
#[requires(le!(ld * n_occ, c.len()), "pm_populations: c holds ld × n_occ")]
#[requires(le!(ld * n_occ, w.len()), "pm_populations: w holds ld × n_occ")]
#[requires(le!(2, offsets.len()), "pm_populations: the offsets bound an atom")]
#[requires(
    le!(offsets.len(), offsets_dev.len()),
    "pm_populations: offsets_dev holds the offsets"
)]
#[requires(
    fits_i32!(offsets.len().checked_mul(n_occ * n_occ)),
    "pm_populations: n_atoms × n_occ² fits an i32"
)]
#[requires(
    le!((offsets.len() - 1) * n_occ * n_occ, pops.len()),
    "pm_populations: pops holds n_atoms × n_occ²"
)]
#[track_caller]
pub fn pm_populations<'s, H>(
    stream: &'s Stream,
    n_occ: usize,
    ld: usize,
    c: &impl DeviceSlice<f64>,
    w: &impl DeviceSlice<f64>,
    offsets: H,
    offsets_dev: &mut impl DeviceSliceMut<i32>,
    pops: &mut impl DeviceSliceMut<f64>,
) -> Result<Pending<'s, H>>
where
    H: HostSliceMut<i32> + DerefMut<Target = [i32]>,
{
    let mut below = 0;
    for (a, &o) in offsets.iter().enumerate() {
        assert!(
            below <= o && usize::try_from(o).is_ok_and(|o| o <= ld),
            "pm_populations: offset {a} is {o}, below {below} or past ld {ld}"
        );
        below = o;
    }
    let n_atoms = offsets.len() - 1;
    let uploaded = stream.upload(&mut part_mut(offsets_dev, 0, n_atoms + 1), offsets)?;
    let _current = ScopedDevice::new(stream.physical());
    // SAFETY: c and w hold the ld × n_occ blocks and pops n_atoms n_occ²
    // doubles (the contracts, each product an i32, so the casts are
    // exact); the offsets the kernel follows were checked above to be
    // non-decreasing and within ld, and uploaded just before on this
    // stream from host memory the upload's token holds unwritten until a
    // sync covers the copy; pops and offsets_dev have no other user until
    // the enqueued work finishes (DeviceSliceMut's promise); the launch
    // only enqueues.
    let status = unsafe {
        fb_pm_populations(
            stream.raw(),
            n_atoms as i32,
            n_occ as i32,
            ld as i32,
            c.as_ptr(),
            w.as_ptr(),
            offsets_dev.as_ptr(),
            pops.as_mut_ptr(),
        )
    };
    launched("fb_pm_populations", status)?;
    Ok(uploaded)
}

/// The rotation arrays of one round: the cosines, the sines and the
/// partners, `n_occ` each, which [`pm_sweep`] writes and reads.
pub struct Angles<'a, C, S, P> {
    /// Each orbital's cosine.
    pub cos: &'a mut C,
    /// Each orbital's sine.
    pub sin: &'a mut S,
    /// Each orbital's partner.
    pub partner: &'a mut P,
}

/// One Pipek-Mezey sweep on `stream`: each of the schedule's rounds is the
/// round's angles ([`fb_pm_pair_angles`], a pair that gains no more than
/// `gain_threshold` left alone), `Q ← Rᵀ Q R`
/// ([`fb_pm_rotate_populations`]) and `U ← U R` ([`fb_pm_rotate_columns`]),
/// each out of place between the carried slice and its pong. The rounds
/// are odd in number, so the last lands in the pongs and is copied back:
/// `pops` and `u` hold the sweep's result.
///
/// # Errors
/// A launch's failure, or a copy's.
#[allow(clippy::too_many_arguments, reason = "the kernels' arguments")]
#[requires(le!(1, n_atoms), "pm_sweep: one atom or more")]
#[requires(
    fits_i32!(n_atoms.checked_mul(schedule.n_occ * schedule.n_occ)),
    "pm_sweep: n_atoms × n_occ² fits an i32"
)]
#[requires(
    le!(n_atoms * schedule.n_occ * schedule.n_occ, pops.len())
        && le!(n_atoms * schedule.n_occ * schedule.n_occ, pops_pong.len()),
    "pm_sweep: pops and its pong hold n_atoms × n_occ²"
)]
#[requires(
    le!(schedule.n_occ * schedule.n_occ, u.len())
        && le!(schedule.n_occ * schedule.n_occ, u_pong.len()),
    "pm_sweep: u and its pong hold n_occ²"
)]
#[requires(
    le!(schedule.n_occ, angles.cos.len())
        && le!(schedule.n_occ, angles.sin.len())
        && le!(schedule.n_occ, angles.partner.len()),
    "pm_sweep: the angles hold n_occ each"
)]
// RELEASE: the launcher refuses another exponent, enqueueing nothing.
#[debug_requires(exponent == 2 || exponent == 4, "the exponent is 2 or 4")]
#[debug_requires(!gain_threshold.is_nan(), "the gain threshold is a number")]
#[track_caller]
pub fn pm_sweep<D: DeviceSliceMut<i32>>(
    stream: &Stream,
    exponent: i32,
    n_atoms: usize,
    schedule: &Schedule<D>,
    gain_threshold: f64,
    (pops, pops_pong): (&mut impl DeviceSliceMut<f64>, &mut impl DeviceSliceMut<f64>),
    (u, u_pong): (&mut impl DeviceSliceMut<f64>, &mut impl DeviceSliceMut<f64>),
    angles: Angles<
        '_,
        impl DeviceSliceMut<f64>,
        impl DeviceSliceMut<f64>,
        impl DeviceSliceMut<i32>,
    >,
) -> Status {
    let n_occ = schedule.n_occ;
    let m = padded(n_occ);
    let (n_atoms_i, n_occ_i) = (n_atoms as i32, n_occ as i32);
    let (mut q_in, mut q_out) = (pops.as_mut_ptr(), pops_pong.as_mut_ptr());
    let (mut u_in, mut u_out) = (u.as_mut_ptr(), u_pong.as_mut_ptr());
    let (cos, sin, partner) = (
        angles.cos.as_mut_ptr(),
        angles.sin.as_mut_ptr(),
        angles.partner.as_mut_ptr(),
    );
    let _current = ScopedDevice::new(stream.physical());
    for round in 0..m - 1 {
        let pairs = schedule.dev.as_ptr().wrapping_add(round * m);
        // SAFETY: the tensors are n_atoms n_occ² doubles and the angles
        // n_occ each (the contracts, the product an i32); pairs is round
        // `round` of the schedule, m entries inside its (m - 1) m, which
        // pm_schedule wrote as a tournament and uploaded into the device
        // bytes the Schedule alone holds, so every orbital is in exactly
        // one pair and every index is below n_occ or the padding the kernel
        // sits out; the launch only enqueues.
        let status = unsafe {
            fb_pm_pair_angles(
                stream.raw(),
                exponent,
                n_atoms_i,
                n_occ_i,
                q_in,
                pairs,
                (m / 2) as i32,
                gain_threshold,
                cos,
                sin,
                partner,
            )
        };
        launched("fb_pm_pair_angles", status)?;
        // SAFETY: the launch above wrote every orbital's partner, each
        // below n_occ, before this one on the stream; the two tensors are
        // distinct slices of n_atoms n_occ² doubles.
        let status = unsafe {
            fb_pm_rotate_populations(
                stream.raw(),
                n_atoms_i,
                n_occ_i,
                q_in,
                cos,
                sin,
                partner,
                q_out,
            )
        };
        launched("fb_pm_rotate_populations", status)?;
        // SAFETY: as above, over the two n_occ² slices of U.
        let status =
            unsafe { fb_pm_rotate_columns(stream.raw(), n_occ_i, u_in, cos, sin, partner, u_out) };
        launched("fb_pm_rotate_columns", status)?;
        std::mem::swap(&mut q_in, &mut q_out);
        std::mem::swap(&mut u_in, &mut u_out);
    }
    // m is even, so the rounds are odd in number and the last wrote the
    // pongs.
    stream.copy_device(pops, &*pops_pong)?;
    stream.copy_device(u, &*u_pong)
}

/// `L = Σ_i Σ_A (Q^A_ii)^exponent` of `pops` into `out[0]`, on `stream`
/// ([`fb_pm_functional`]): one fold in a fixed order.
///
/// # Errors
/// The launch's failure.
#[requires(
    le!(1, n_atoms) && le!(1, n_occ) && le!(n_occ, FB_PM_MAX_OCCUPIED as usize),
    "pm_functional: an atom and an orbital or more, within FB_PM_MAX_OCCUPIED"
)]
#[requires(
    fits_i32!(n_atoms.checked_mul(n_occ * n_occ)),
    "pm_functional: n_atoms × n_occ² fits an i32"
)]
#[requires(
    le!(n_atoms * n_occ * n_occ, pops.len()),
    "pm_functional: pops holds n_atoms × n_occ²"
)]
#[requires(le!(1, out.len()), "pm_functional: out holds one double")]
// RELEASE: the launcher refuses another exponent, enqueueing nothing.
#[debug_requires(exponent == 2 || exponent == 4, "the exponent is 2 or 4")]
#[track_caller]
pub fn pm_functional(
    stream: &Stream,
    exponent: i32,
    n_atoms: usize,
    n_occ: usize,
    pops: &impl DeviceSlice<f64>,
    out: &mut impl DeviceSliceMut<f64>,
) -> Status {
    let _current = ScopedDevice::new(stream.physical());
    // SAFETY: pops holds n_atoms n_occ² doubles and out one (the
    // contracts, the product an i32); the launch only enqueues.
    let status = unsafe {
        fb_pm_functional(
            stream.raw(),
            exponent,
            n_atoms as i32,
            n_occ as i32,
            pops.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    launched("fb_pm_functional", status)
}

/// The `n × n` identity into `u`, column-major, on `stream`
/// ([`fb_pm_identity`]).
///
/// # Errors
/// The launch's failure.
#[requires(
    le!(1, n) && le!(n, FB_PM_MAX_OCCUPIED as usize),
    "pm_identity: n is within 1..=FB_PM_MAX_OCCUPIED"
)]
#[requires(eq!(n * n, u.len()), "pm_identity: u is n × n")]
#[track_caller]
pub fn pm_identity(stream: &Stream, n: usize, u: &mut impl DeviceSliceMut<f64>) -> Status {
    let _current = ScopedDevice::new(stream.physical());
    // SAFETY: u is n² doubles (the contracts); the launch only enqueues.
    let status = unsafe { fb_pm_identity(stream.raw(), n as i32, u.as_mut_ptr()) };
    launched("fb_pm_identity", status)
}

#[cfg(test)]
mod tests {
    use super::pm_schedule_len;

    /// The schedule's length: an odd count padded by one.
    #[test]
    fn the_schedules_length_pads_an_odd_count() {
        assert_eq!(pm_schedule_len(5), 30);
        assert_eq!(pm_schedule_len(4), 12);
        assert_eq!(pm_schedule_len(1), 2);
    }
}
