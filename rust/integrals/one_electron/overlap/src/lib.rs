//! The AO overlap matrix S, the first whole run through the model
//! (`docs/programming_model.md`, "A task declares"): an XYZ file and a basis
//! name, declared as tasks with layouts, laid out in one arena per kind,
//! planned once, run on one stream and timed. The C++
//! `src/integrals/one_electron/overlap` and the pipeline's Overlap step
//! (`MatrixAOPair::cartesian_to_spherical`) are the reference design.
//!
//! [`add_overlap`] adds S to a [`Dag`](fb_core_planner::Dag) after the
//! records ([`fb_integrals_mmd::add_records`]):
//!
//! - **`overlap`**, on the worker's stream: the Cartesian matrix zeroed,
//!   then one `fb_overlap_class` launch per class, each reading its class's
//!   records as an edge ([`ClassRows::read`](fb_integrals_mmd::ClassRows::read)).
//! - **`harmonics`**, on the CPU: U, the Cartesian-to-spherical transform
//!   ([`fb_basis::spherical_transform_matrix`]), written into a pinned row.
//! - **`spherical transform`**, on the worker's stream: U copied into device
//!   scratch, then `S = Uᵀ S_cart U` as two GEMMs through `wwr-sys`'s BLAS,
//!   the intermediate `S_cart U` scratch too.
//!
//! [`Run`] is the whole run: the basis read and placed, those tasks plus a
//! download and a collect, the plan, and one run on a device returning S
//! and the timing record.
//!
//! S is not renormalized: an AO's norm is its published contraction's,
//! since the basis normalizes each primitive and never a contraction
//! ([`fb_basis::ShellStamp::primitive_normalized`]), so diag(S) is 1 only to
//! the coefficients' printed digits.
//!
//! | C++ | Here |
//! |---|---|
//! | `overlap(res, classes, s.cartesian())` | the `overlap` task |
//! | `upload_spherical_transform_matrix` (nothing for an identity U) | the `harmonics` task's pinned row and the transform's device scratch, always |
//! | `MatrixAOPair::cartesian_to_spherical` (`gemmwich`; a copy for an identity U) | the `spherical transform` task's two GEMMs, always |
//! | `res.blas()`, the `DeviceResources`' handle | the transform's own handle, made at its first run |
//! | `OverlapStep::run`, `write_overlap_npy` | [`Run`], S returned to the caller |

mod run;
mod tasks;

pub use run::Run;
pub use tasks::add_overlap;
