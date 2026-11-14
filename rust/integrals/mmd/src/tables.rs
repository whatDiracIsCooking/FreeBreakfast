//! The host's half: the shell table and each class's shell pairs with their
//! record prefixes, the only things formed on the host. The device forms
//! every record from them; the host never writes one (the C++
//! `src/integrals/mmd` README's design).

use std::collections::BTreeMap;

use fb_basis::{BasisSet, Shell};
use fb_core_error_handling::{Error, check_or_abort};
use fb_integrals_sys::{ClassPairs, RecordTables, ShellTable};

/// The sweep of `basis`: every shell pair `i <= j`, swapped when `i` has
/// the higher momentum, listed in its class with its first record, and the
/// shell table, ascending in `(la, lb)`, none empty. A class past
/// `i32::MAX` records, or a basis past `i32::MAX` primitives, aborts, as
/// the C++ does: the kernels index both in `int32_t`. `BasisSet` already
/// bounds the AO offsets, which [`RecordTables::new`] checks again.
///
/// # Panics
/// As [`RecordTables::new`], which a basis's own sweep never trips.
#[track_caller]
pub(crate) fn tables(basis: &BasisSet) -> RecordTables {
    let shells = basis.shells();
    let mut t = ShellTable {
        prim_begin: vec![0],
        ..ShellTable::default()
    };
    push_shells(&mut t, basis);

    let mut sweep = Sweep::new();
    let n = basis.nshells();
    for i in 0..n {
        for j in i..n {
            let (si, sj) = (&shells[i as usize], &shells[j as usize]);
            let (bra, ket) = if si.angular_momentum() > sj.angular_momentum() {
                (j, i)
            } else {
                (i, j)
            };
            add_pair(
                &mut sweep,
                &shells[bra as usize],
                &shells[ket as usize],
                bra,
                ket,
            );
        }
    }
    RecordTables::new(t, classes(sweep), basis.nao_cart() as usize)
}

/// The sweep of two bases, `rows` and `cols`, as the C++
/// `mixed_primitive_pair_classes` forms it: every ordered shell pair (`i`
/// of `rows`, `j` of `cols`), never swapped, in its class `(l_i, l_j)`,
/// ascending, none empty; the shell table `rows`' shells and then `cols'`,
/// so `j` is `n_rows + j` there, each shell's offset into its own basis's
/// Cartesian AOs and its rank among its own basis's shells. Aborts as
/// [`tables`] does, the primitives of both bases counted together.
///
/// # Panics
/// As [`RecordTables::mixed`], which two bases' own sweep never trips.
#[track_caller]
pub(crate) fn mixed_tables(rows: &BasisSet, cols: &BasisSet) -> RecordTables {
    let mut t = ShellTable {
        prim_begin: vec![0],
        ..ShellTable::default()
    };
    push_shells(&mut t, rows);
    push_shells(&mut t, cols);
    let mut sweep = Sweep::new();
    let n_rows = rows.nshells();
    for (i, si) in (0_u32..).zip(rows.shells()) {
        for (j, sj) in (0_u32..).zip(cols.shells()) {
            add_pair(&mut sweep, si, sj, i, n_rows + j);
        }
    }
    RecordTables::mixed(
        t,
        classes(sweep),
        n_rows as usize,
        [rows.nao_cart() as usize, cols.nao_cart() as usize],
    )
}

/// Each class's bra shells, ket shells and record prefix (in `i64`, so a
/// class past `i32::MAX` is caught before it is sized), keyed by
/// `(la, lb)` so the classes come out ascending.
type Sweep = BTreeMap<(i32, i32), (Vec<u32>, Vec<u32>, Vec<i64>)>;

/// The pair (`bra`, `ket`) of shells `sb` and `sk`, at those indices of the
/// shell table, listed in its class with its first record.
fn add_pair(sweep: &mut Sweep, sb: &Shell, sk: &Shell, bra: u32, ket: u32) {
    let (b, k, r) = sweep
        .entry((sb.angular_momentum(), sk.angular_momentum()))
        .or_insert_with(|| (Vec::new(), Vec::new(), vec![0]));
    b.push(bra);
    k.push(ket);
    let last = *r.last().expect("a prefix starts at 0");
    r.push(last + i64::from(sb.nprimitives()) * i64::from(sk.nprimitives()));
}

/// `basis`'s shells appended to `t`: their primitives, centres, first
/// Cartesian AOs (into `basis`'s own) and ranks (among `basis`'s own
/// shells of each momentum). Aborts past `i32::MAX` primitives in the
/// table.
#[track_caller]
fn push_shells(t: &mut ShellTable, basis: &BasisSet) {
    let mut next_rank = BTreeMap::<i32, i32>::new();
    let mut n_prim = i64::from(*t.prim_begin.last().expect("a prefix starts at 0"));
    for (i, shell) in (0_u32..).zip(basis.shells()) {
        n_prim += i64::from(shell.nprimitives());
        t.prim_begin
            .push(check_or_abort(i32::try_from(n_prim).map_err(|_| {
                Error::config(format!(
                    "primitive pair records: the basis has more than {} primitives",
                    i32::MAX
                ))
            })));
        for p in shell.primitives() {
            t.exponent.push(p.exponent);
            t.coefficient.push(p.coefficient);
        }
        let c = shell.center();
        t.c_x.push(c.x);
        t.c_y.push(c.y);
        t.c_z.push(c.z);
        // BasisSet bounds nao_cart() to i32::MAX.
        t.ao_offset.push(basis.ao_offset_cart(i) as i32);
        let rank = next_rank.entry(shell.angular_momentum()).or_insert(0);
        t.rank.push(*rank);
        *rank += 1;
    }
}

/// The sweep's classes, ascending, each prefix narrowed to `i32`. A class
/// past `i32::MAX` records aborts.
#[track_caller]
fn classes(sweep: Sweep) -> Vec<ClassPairs> {
    sweep
        .into_iter()
        .map(|((la, lb), (bra, ket, rec_begin))| {
            let n_rec = *rec_begin.last().expect("a prefix starts at 0");
            check_or_abort(i32::try_from(n_rec).map_err(|_| {
                Error::config(format!(
                    "primitive pair records: class ({la}, {lb}) has {n_rec} records, past {}",
                    i32::MAX
                ))
            }));
            ClassPairs {
                la,
                lb,
                bra,
                ket,
                // Every entry is at most the last, checked above.
                rec_begin: rec_begin.into_iter().map(|r| r as i32).collect(),
            }
        })
        .collect()
}
