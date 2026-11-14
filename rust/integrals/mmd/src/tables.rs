//! The host's half: the shell table and each class's shell pairs with their
//! record prefixes, the only things formed on the host. The device forms
//! every record from them; the host never writes one (the C++
//! `src/integrals/mmd` README's design).

use std::collections::BTreeMap;

use fb_basis::BasisSet;
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
    let mut next_rank = BTreeMap::<i32, i32>::new();
    let mut n_prim = 0_i64;
    for (i, shell) in (0_u32..).zip(shells) {
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

    // Keyed by (la, lb), so the classes come out ascending; prefixes in
    // i64, so a class past i32::MAX is caught before it is sized.
    let mut sweep = BTreeMap::<(i32, i32), (Vec<u32>, Vec<u32>, Vec<i64>)>::new();
    let n = basis.nshells();
    for i in 0..n {
        for j in i..n {
            let (si, sj) = (&shells[i as usize], &shells[j as usize]);
            let (bra, ket) = if si.angular_momentum() > sj.angular_momentum() {
                (j, i)
            } else {
                (i, j)
            };
            let (sb, sk) = (&shells[bra as usize], &shells[ket as usize]);
            let (b, k, r) = sweep
                .entry((sb.angular_momentum(), sk.angular_momentum()))
                .or_insert_with(|| (Vec::new(), Vec::new(), vec![0]));
            b.push(bra);
            k.push(ket);
            let last = *r.last().expect("a prefix starts at 0");
            r.push(last + i64::from(sb.nprimitives()) * i64::from(sk.nprimitives()));
        }
    }
    let classes = sweep
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
        .collect();
    RecordTables::new(t, classes, basis.nao_cart() as usize)
}
