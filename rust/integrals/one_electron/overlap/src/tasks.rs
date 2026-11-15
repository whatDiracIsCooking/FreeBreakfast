//! The Cartesian overlap on a stream, after the records, and its spherical
//! transform ([`fb_recipes_matrix_ao`]).

use contracts::debug_requires;
use fb_basis::BasisSet;
use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::memset;
use fb_integrals_sys::{ClassRows, Fields, overlap_class};
use fb_recipes_matrix_ao::add_cartesian_to_spherical;

/// Add the spherical AO overlap of `basis` to `dag` on GPU worker
/// `worker`, and return S: an output row of `nao_sphe²` doubles,
/// column-major, symmetric up to rounding.
///
/// `overlap` (stream: the Cartesian matrix zeroed and one
/// [`overlap_class`] launch per class, in order, or each on a fork of the
/// stream joined back before the task returns when the classes opt in
/// ([`ClassRows::fork`], `DeviceStream::fork_join`), reading the
/// records as an edge),
/// then the spherical transform of [`add_cartesian_to_spherical`]:
/// `harmonics` and `upload U` (U into a pinned row, then its
/// device row), only when `u` is `None`, and
/// `spherical transform` (stream: `S = Uᵀ S_cart U`). `classes` are the
/// records of `basis`, added on `worker` by
/// [`fb_integrals_mmd::add_records`] earlier in `dag`, so one build serves
/// every reader of the plan; `u` is the plan's slot for U over `basis` on
/// `worker`, shared with every other matrix the plan transforms.
///
/// # Panics
/// If `nao_cart²` does not fit an `i32` (the overlap kernels index the
/// matrix in 32 bits), or `classes` are rows of another worker's or another
/// plan's table ([`Dag::add_gpu`]).
#[debug_requires(basis.nshells() > 0, "a basis of at least one shell")]
#[track_caller]
pub fn add_overlap(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &BasisSet,
    classes: &[ClassRows],
    u: &mut Option<RowId<Device, f64>>,
) -> RowId<Device, f64> {
    let n_cart = basis.nao_cart() as usize;
    assert!(
        i32::try_from(n_cart * n_cart).is_ok(),
        "add_overlap: {n_cart}² Cartesian AOs do not fit the kernels' 32-bit index"
    );
    let s_cart = dag.add_gpu(
        "overlap",
        worker,
        Overlap {
            classes: classes.to_vec(),
            n_cart,
        },
    );
    add_cartesian_to_spherical(dag, worker, basis, u, "S", s_cart)
}

/// GPU: the Cartesian overlap, `nao_cart²`, zeroed and then summed over
/// the classes, in order on the stream or, when they opt in
/// ([`ClassRows::fork`]), a fork per class ([`DeviceStream::fork_join`]),
/// every class's contributions added atomically at both mirror positions.
struct Overlap {
    classes: Vec<ClassRows>,
    n_cart: usize,
}

impl GpuTask for Overlap {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        for c in &self.classes {
            c.read(device, Fields::OVERLAP);
        }
        device.output("S cartesian", self.n_cart * self.n_cart)
    }

    fn forks(&self) -> usize {
        if self.classes.iter().any(|c| c.fork) {
            self.classes.len()
        } else {
            0
        }
    }

    fn run(
        &self,
        s: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        let mut s = device.take_mut(*s);
        memset::zero(&mut s, stream)?;
        if self.classes.iter().any(|c| c.fork) {
            stream.fork_join(self.classes.len(), |c, fork| {
                overlap_class(fork, device, &self.classes[c], &mut s)
            })
        } else {
            self.classes
                .iter()
                .try_for_each(|c| overlap_class(stream, device, c, &mut s))
        }
    }
}
