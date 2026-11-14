//! The AO overlap matrix S, the first whole run through the model
//! (`docs/programming_model.md`, "A task declares"): an XYZ file and a basis
//! name, declared as tasks with layouts, laid out in one arena per kind,
//! planned once, run on one stream and timed. The C++
//! `src/integrals/one_electron/overlap` and the pipeline's Overlap step
//! (`MatrixAOPair::cartesian_to_spherical`) are the reference design.
//!
//! [`add_overlap`] adds S to a [`Dag`](fb_core_planner::Dag) after the
//! records ([`fb_integrals_mmd::add_records`]), which the caller adds and
//! passes in, so one build serves every reader of a plan:
//!
//! - **`overlap`**, on the worker's stream: the Cartesian matrix zeroed,
//!   then one [`overlap_class`](fb_integrals_sys::overlap_class) launch
//!   per class, each reading its class's records as an edge
//!   ([`ClassRows::read`](fb_integrals_sys::ClassRows::read)).
//! - then the spherical transform, `S = Uᵀ S_cart U`
//!   ([`fb_recipes_matrix_ao::add_cartesian_to_spherical`]): `harmonics` on
//!   the CPU (U into a pinned row, unless an earlier matrix of the plan
//!   staged it) and `spherical transform` on the worker's stream.
//!
//! [`Run`] is the whole run: the basis read and placed, the records and
//! those tasks plus a download and a collect, the plan, and one run on a
//! device returning S and the timing record.
//!
//! S is not renormalized: an AO's norm is its published contraction's,
//! since the basis normalizes each primitive and never a contraction
//! ([`fb_basis::ShellStamp::primitive_normalized`]), so diag(S) is 1 only to
//! the coefficients' printed digits.
//!
//! | C++ | Here |
//! |---|---|
//! | `overlap(res, classes, s.cartesian())` | the `overlap` task, the records' [`ClassRows`](fb_integrals_sys::ClassRows) passed in by the caller |
//! | `upload_spherical_transform_matrix`, `MatrixAOPair::cartesian_to_spherical` | [`fb_recipes_matrix_ao::add_cartesian_to_spherical`], U's slot passed in by the caller |
//! | `OverlapStep::run`, `write_overlap_npy` | [`Run`], S returned to the caller |

#![forbid(unsafe_code)]

mod run;
mod tasks;

pub use run::Run;
pub use tasks::add_overlap;
