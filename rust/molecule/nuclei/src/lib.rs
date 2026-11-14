//! `src/molecule`, its device half: a [`Molecule`]'s nuclei on the device,
//! as tasks of the model (`docs/programming_model.md`, "A task declares").
//!
//! [`add_nuclei`] adds three nodes to a [`Dag`]:
//!
//! - **`nuclei`**, on the CPU: each nucleus's charge `Z` and its position,
//!   in atom order, formed from the molecule and written into two pinned
//!   rows, its outputs: `n` charges and `3 n` coordinates (x, y, z per
//!   nucleus), as `f64`, the C++ `upload`'s loop.
//! - **`upload nuclei z`** and **`upload nuclei c`**, the planner's copy
//!   nodes ([`Dag::upload`]) on the worker's stream: each row copied into a
//!   device row of its name, together [`NucleiRows`], which a reader takes
//!   as an edge, as it takes the records' `ClassRows`.
//!
//! Every size is the molecule's, fixed at planning. The C++ `Molecule`
//! uploads its nuclei at construction, from buffers it allocates itself;
//! here only the arena allocates, so the upload is a plan's, run whenever
//! the plan runs, and the `Molecule` stays host data
//! (`fb_molecule`'s crate doc).
//!
//! **Why a crate of its own.** `fb-molecule` stays link-free, so its tests
//! are tier 3 and every crate that names a `Molecule` (the basis, the XYZ
//! reader, the SAD guess) builds with no wwr crate in its graph. The tasks
//! need the planner and a stream, which link wwr, so they sit here, beside
//! it, as the records sit beside the basis. A feature on `fb-molecule`
//! would put the planner in its graph even with the feature off, and so
//! every planner change would select the crates above the molecule.
//!
//! | C++ (`src/molecule`) | Here |
//! |---|---|
//! | `Molecule`'s nuclei, uploaded at construction (`upload`: a carve of `res`'s pinned workspace, two device buffers, two copies on `res.stream()` and a sync) | [`add_nuclei`]`(dag, worker, molecule)`: the `nuclei` task's pinned rows and the two upload nodes' device rows and copies, on the worker's one stream; the sync is the executor's edge |
//! | `NucleiView` (`nuclei_view.h`), from `view()` | [`NucleiRows`]: `n_nuc` and the two device rows, declared by a reader with [`NucleiRows::read`]; the reader forms the launch wrappers' `fb_nuclei_view` (`fb_integrals_sys`) from its rows' pointers, since this crate links no kernel library |
//! | `device_stream()`, the stream the nuclei were uploaded on | nothing: a reader runs after the upload by its edge |
//! | `molecule_footprint(natoms)` | the plan's tables ([`Planned`](fb_core_planner::Planned)): the footprint is the layout |

#![forbid(unsafe_code)]

use fb_core_arena::{Bufs, Device, Heap, Layout, RowId, StagingKind};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag};
use fb_molecule::{Molecule, nuclear_charge};

#[cfg(test)]
mod tests;

/// A molecule's nuclei on the device, outputs of the two upload nodes:
/// `NucleiView`'s fields, each array a row. A reader declares both rows
/// with [`NucleiRows::read`] and hands its launch wrapper an
/// `fb_nuclei_view` of `n_nuc` and the rows' pointers.
#[derive(Clone, Copy, Debug)]
pub struct NucleiRows {
    /// The number of nuclei, positive.
    pub n_nuc: i32,
    /// `Z_C`, each nucleus's charge, `n_nuc` long.
    pub z: RowId<Device, f64>,
    /// `C`, each nucleus's position in bohr, `3 n_nuc` long: x, y, z per
    /// nucleus.
    pub c: RowId<Device, f64>,
}

impl NucleiRows {
    /// Declare, in a reader's layout, that it reads both rows.
    pub fn read(&self, device: &mut Layout<'_, Device>) {
        device.read(self.z);
        device.read(self.c);
    }
}

/// Add the nuclei of `molecule` to `dag` on GPU worker `worker`: a CPU
/// task, `nuclei`, that writes each nucleus's charge and position into two
/// pinned rows, and two copy nodes, `upload nuclei z` and `upload nuclei c`
/// ([`Dag::upload`]), that copy them into the device rows it returns.
/// Every size is the molecule's, fixed here at planning; a `Molecule` has
/// at least one atom and at most `i32::MAX`.
pub fn add_nuclei(dag: &mut Dag<'_>, worker: usize, molecule: &Molecule) -> NucleiRows {
    let (z, c) = dag.add_cpu(
        "nuclei",
        worker,
        Stage {
            molecule: molecule.clone(),
        },
    );
    let n = molecule.natoms() as usize;
    NucleiRows {
        // In range: a `Molecule` refuses more than an `i32` counts.
        n_nuc: n as i32,
        z: dag.upload(worker, z, "nuclei z", n),
        c: dag.upload(worker, c, "nuclei c", 3 * n),
    }
}

/// CPU: the charges and the positions written into two rows of the host
/// kind `H`, its outputs. Pinned in a plan; any host kind and stream, so the
/// tests run it on `PinnedStandIn` rows and `HostOnly` workers.
struct Stage {
    molecule: Molecule,
}

impl<H: StagingKind> CpuTask<H> for Stage {
    type Rows = (RowId<H, f64>, RowId<H, f64>);

    fn layout(&self, host: &mut Layout<'_, H>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        let n = self.molecule.atoms().len();
        (host.output("nuclei z", n), host.output("nuclei c", 3 * n))
    }

    fn run(&self, rows: &Self::Rows, host: &Bufs<'_, H>, _: &Bufs<'_, Heap>) -> Status {
        let z = host.take_mut(rows.0);
        let c = host.take_mut(rows.1);
        for (a, atom) in self.molecule.atoms().iter().enumerate() {
            z[a] = f64::from(nuclear_charge(atom));
            c[3 * a] = atom.position.x;
            c[3 * a + 1] = atom.position.y;
            c[3 * a + 2] = atom.position.z;
        }
        Ok(())
    }
}
