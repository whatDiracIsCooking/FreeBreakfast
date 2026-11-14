//! The density-fitting Coulomb metric
//!
//! ```text
//! (P|Q) = ∫∫ P(r₁) Q(r₂) / r₁₂ dr₁ dr₂
//! ```
//!
//! of an auxiliary (fitting) [`BasisSet`], over its real solid harmonics, as
//! tasks of the model (`docs/programming_model.md`, "A task declares"), by
//! McMurchie–Davidson over the `metric<lP, lQ>` kernels FelizJueves emits
//! (`deps/FelizJueves/docs/implementation/metric.md`). The C++
//! `src/integrals/eri/density_fitting/metric` is the reference design; its
//! README has the build and why the weights need no further normalization.
//!
//! [`add_metric`] adds, after the kernel library's load check at planning:
//!
//! - **`metric tables`**, on the CPU, and **`metric`**, on the worker's
//!   stream ([`fb_integrals_sys::add_metric_tasks`]): one record per aux
//!   primitive, its centre, exponent α and weight `s = c (π/α)^{3/2}`, `c`
//!   the coefficient as the basis holds it, and one entry per aux shell pair
//!   P ≤ Q in its class `(lP, lQ)`, `lP ≤ lQ`, staged into pinned rows and
//!   uploaded; then one launch per class into the Cartesian metric, every
//!   element a plain store by the one pair it belongs to, so nothing is
//!   zeroed and the Cartesian metric is exactly symmetric.
//! - **`harmonics`** (only when the caller's slot is empty) and **`spherical
//!   transform`** ([`add_cartesian_to_spherical`]): `Uᵀ M U`, U's blocks
//!   general in l, here through the kernels' i.
//! - **`metric average`**, on the stream: `Uᵀ M U` is two GEMMs, which round
//!   `(P, Q)` and `(Q, P)` in different orders, so the spherical metric is
//!   copied into its own row and replaced by `(M + Mᵀ)/2`
//!   ([`fb_recipes_symmetrize::symmetrize`]), whose `(P, Q)` and `(Q, P)`
//!   are the same bits.
//!
//! The aux basis is an ordinary [`BasisSet`], placed as the orbital one is:
//! `fb_io_basis::resolve_aux_basis(name)?.primitive_normalized().place(&molecule)`.
//! Its contractions are not renormalized: a contracted fitting function's
//! norm is its published contraction's, which rescales that function and
//! leaves the fit `(mn|P) M⁻¹ (P|ls)` unchanged when `(mn|P)` is built over
//! the same basis.
//!
//! | C++ | Here |
//! |---|---|
//! | `metric(res, aux, m)`, into a `MatrixAOPair` made for `aux` | [`add_metric`]`(dag, worker, aux, u)`, returning M's row; U's slot passed in, as every transform takes it |
//! | the records and pairs staged on the host from `res.device_workspace()`, the call synchronizing once after the upload | `metric tables` into pinned rows and `metric`'s device scratch, all in the plan's arena; no synchronization |
//! | `metric_class` per class (`metric_bridge.h`) | [`fb_integrals_sys::add_metric_tasks`] over an [`AuxTable`], whose pairs the safe layer forms itself |
//! | `m.cartesian_to_spherical(res)` | [`add_cartesian_to_spherical`] |
//! | `symmetrize(res, ns, m.spherical().data(), ns)`, in place | `metric average`: the spherical metric copied into M's row, then [`fb_recipes_symmetrize::symmetrize`] there, since a task writes only its own rows |
//! | `InvalidConfig` for a shell above i, or `nao_cart²` past `INT32_MAX` | `Unsupported` at planning, adding nothing |
//! | `InvalidConfig` for a device or an order mismatch of `m` | nothing: M is a row the call declares, on the worker's device |
//! | success with nothing written for an empty `aux` | a contract: an aux basis of at least one shell |
//! | `metric_footprint(aux)` | nothing: the layout declares every row, and the arena's size is the plan's golden |
//!
//! [`AuxTable`]: fb_integrals_sys::AuxTable

#![forbid(unsafe_code)]

use contracts::debug_requires;
use fb_basis::BasisSet;
use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2d};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_integrals_sys::{AuxTable, FB_METRIC_MAX_MOMENTUM, add_metric_tasks};
use fb_recipes_matrix_ao::add_cartesian_to_spherical;
use fb_recipes_symmetrize::symmetrize;

/// Add the spherical Coulomb metric (P|Q) of the auxiliary basis `aux` to
/// `dag` on GPU worker `worker`, and return M: an output row of
/// `nao_sphe²` doubles, column-major, exactly symmetric.
///
/// `metric tables` (CPU) and `metric` (stream: every class into the
/// Cartesian metric), `harmonics` (CPU: U into a pinned row) only when `u`
/// is `None`, `spherical transform` (stream: `Uᵀ M U`) and `metric
/// average` (stream: `(M + Mᵀ)/2` into M's row). `u` is the plan's slot for
/// U over `aux` on `worker`, shared with every other transform over `aux`.
///
/// First, before anything is added, the kernel library's load check
/// ([`fb_integrals_sys::check_library`]), and then what the kernels can
/// take, both at planning.
///
/// # Errors
/// A `GpuRuntime` error with the check's message, if the loaded
/// `libfb_integrals_kernels.so` is not the one the bindings describe; an
/// `Unsupported` one naming the shell, if a shell of `aux` is above the
/// kernels' table ([`FB_METRIC_MAX_MOMENTUM`], i), or naming the count, if
/// `nao_cart²` does not fit the kernels' 32-bit index. Nothing is added to
/// `dag` on an error.
///
/// # Panics
/// If `u` holds a row of another worker's table ([`Dag::add`]).
#[debug_requires(aux.nshells() > 0, "an aux basis of at least one shell")]
#[track_caller]
pub fn add_metric(
    dag: &mut Dag<'_>,
    worker: usize,
    aux: &BasisSet,
    u: &mut Option<RowId<Pinned, f64>>,
) -> Result<RowId<Device, f64>> {
    // Not `map_err(Error::gpu)`: through a function pointer the error's
    // origin would be core's, not this call's caller.
    if let Err(e) = fb_integrals_sys::check_library() {
        return Err(Error::gpu(e));
    }
    if let Some((s, shell)) = (0..)
        .zip(aux.shells())
        .find(|(_, shell)| shell.angular_momentum() > FB_METRIC_MAX_MOMENTUM)
    {
        return Err(Error::unsupported(format!(
            "add_metric: aux shell {s} has momentum {}, but the metric kernels cover l <= \
             {FB_METRIC_MAX_MOMENTUM}",
            shell.angular_momentum()
        )));
    }
    let n_cart = aux.nao_cart() as usize;
    if i32::try_from(n_cart * n_cart).is_err() {
        return Err(Error::unsupported(format!(
            "add_metric: {n_cart} Cartesian fitting functions are too many for the kernels' \
             32-bit index"
        )));
    }

    let mut table = AuxTable::new(n_cart);
    for (s, shell) in (0..).zip(aux.shells()) {
        let c = shell.center();
        table.push_shell(
            shell.angular_momentum(),
            [c.x, c.y, c.z],
            aux.ao_offset_cart(s) as usize,
            shell
                .primitives()
                .iter()
                .map(|p| (p.exponent, p.coefficient)),
        );
    }
    let cartesian = add_metric_tasks(dag, worker, &table);
    let spherical = add_cartesian_to_spherical(dag, worker, aux, u, "M spherical", cartesian);
    let n = aux.nao_sphe() as usize;
    Ok(dag.add("metric average", worker, Average { spherical, n }))
}

/// GPU: the spherical metric, `n²`, copied into M's row and averaged with
/// its transpose there.
struct Average {
    spherical: RowId<Device, f64>,
    n: usize,
}

impl Task for Average {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.spherical);
        device.output("M", self.n * self.n)
    }

    fn run(
        &self,
        m: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let mut m = device.take_mut(*m);
        memcpy_d2d(&m, device.get(self.spherical), stream)?;
        symmetrize(stream, self.n, &mut m, self.n)
    }
}
