//! The host's half: the shell table and each class's shell pairs with their
//! record prefixes, the only things formed on the host. The device forms
//! every record from them; the host never writes one (the C++
//! `src/integrals/mmd` README's design).

use std::collections::BTreeMap;

use fb_basis::BasisSet;
use fb_core_error_handling::{Error, check_or_abort};

/// The basis's shells and primitives, a field per array, as the device
/// reads them (`fb_primitive_pair_source`).
#[derive(Debug, Default, PartialEq)]
pub(crate) struct ShellTable {
    /// Each shell's first primitive, `n_shells + 1` long.
    pub(crate) prim_begin: Vec<i32>,
    /// Every primitive's exponent, shell by shell.
    pub(crate) exponent: Vec<f64>,
    /// Every primitive's coefficient, as the basis gives it.
    pub(crate) coefficient: Vec<f64>,
    /// Each shell's centre: x.
    pub(crate) c_x: Vec<f64>,
    /// y.
    pub(crate) c_y: Vec<f64>,
    /// z.
    pub(crate) c_z: Vec<f64>,
    /// Each shell's first Cartesian AO.
    pub(crate) ao_offset: Vec<i32>,
    /// Each shell's index among the basis's shells of its momentum, in
    /// basis order.
    pub(crate) rank: Vec<i32>,
}

/// One class's shell pairs, in sweep order, each swapped into `la <= lb`,
/// and each one's first record.
#[derive(Debug, PartialEq)]
pub(crate) struct ClassPairs {
    /// The bra shells' momentum.
    pub(crate) la: i32,
    /// The ket shells'.
    pub(crate) lb: i32,
    /// Each pair's bra shell.
    pub(crate) bra: Vec<u32>,
    /// Each pair's ket shell.
    pub(crate) ket: Vec<u32>,
    /// The prefix of the pairs' primitive products, `n_pairs + 1` long: its
    /// last entry is the class's record count.
    pub(crate) rec_begin: Vec<i32>,
}

impl ClassPairs {
    /// The class's records: every primitive product of every pair, none
    /// screened.
    pub(crate) fn n_rec(&self) -> i32 {
        *self.rec_begin.last().expect("a prefix starts at 0")
    }
}

/// The shell table and every reached class, ascending in `(la, lb)`, none
/// empty.
#[derive(Debug, PartialEq)]
pub(crate) struct Tables {
    pub(crate) shells: ShellTable,
    pub(crate) classes: Vec<ClassPairs>,
}

impl Tables {
    /// The sweep of `basis`: every shell pair `i <= j`, swapped when `i` has
    /// the higher momentum, listed in its class with its first record, and
    /// the shell table. A class past `i32::MAX` records, or a basis past
    /// `i32::MAX` primitives, aborts, as the C++ does: the kernels index both
    /// in `int32_t`. `BasisSet` already bounds the AO offsets.
    #[track_caller]
    pub(crate) fn new(basis: &BasisSet) -> Tables {
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
        Tables { shells: t, classes }
    }

    /// Every class's shell pairs.
    pub(crate) fn n_pairs(&self) -> usize {
        self.classes.iter().map(|c| c.bra.len()).sum()
    }

    /// The largest class's records.
    pub(crate) fn n_max(&self) -> i32 {
        self.classes
            .iter()
            .map(ClassPairs::n_rec)
            .max()
            .unwrap_or(0)
    }
}
