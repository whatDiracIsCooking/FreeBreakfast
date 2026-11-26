//! The v0.0.4 layer's records call, `fb_integrals_sys::mmd_records`, on a
//! card (tier 4, device 0) against the old layer's `add_records` on water
//! in cc-pVDZ: every class formed from the same `record_tables`, each
//! table handed to the call on the host, and every field of every record
//! compared **bitwise**. The two run the same kernel on the same inputs,
//! one thread per record, so there is no tolerance to set.

mod common;

use common::Class;
use fb_core_budget::{Backing, Caps, Ledger};
use fb_core_device::Stream;
use fb_core_planner::Dag;
use fb_integrals_mmd::{add_records, record_tables};
use fb_integrals_sys::{RECORD_F64_FIELDS, RECORD_I32_FIELDS, mmd_records};

#[test]
#[ignore = "tier 4: needs a GPU"]
fn equals_the_old_layers_records_on_water_bitwise() {
    let [(_, basis), _] = common::cases();
    let mut dag = Dag::new();
    let classes = add_records(&mut dag, 0, &basis).unwrap();
    let want = common::run(dag, classes.iter().map(Class::from).collect());

    let tables = record_tables(&basis);
    let s = tables.shells();
    let (n_shell, n_prim) = (s.ao_offset.len(), s.exponent.len());
    let caps = Caps::read(&[0], &[None], None, Backing::Real).unwrap();
    let ledger = Ledger::new(caps, Backing::Real, vec![0], false);
    let budget = ledger.budget("test");
    let stream = Stream::new(0);

    // The shell table's doubles, uploaded once.
    let doubles: Vec<f64> = [&s.exponent, &s.coefficient, &s.c_x, &s.c_y, &s.c_z]
        .into_iter()
        .flatten()
        .copied()
        .collect();
    let mut staged = budget.pinned::<f64>(0, doubles.len());
    staged.copy_from_slice(&doubles);
    let mut shells = budget.device::<f64>(0, doubles.len());
    drop(stream.upload(&mut shells, staged).unwrap().sync().unwrap());

    assert_eq!(want.len(), tables.classes().len(), "one class each");
    let mut misses = Vec::new();
    for (c, w) in tables.classes().iter().zip(&want) {
        let (n_pairs, n_rec) = (c.bra.len(), c.n_rec() as usize);
        let ints: Vec<i32> = s
            .prim_begin
            .iter()
            .chain(&s.ao_offset)
            .chain(&s.rank)
            .chain(&c.rec_begin)
            .copied()
            .chain(c.bra.iter().chain(&c.ket).map(|&x| x as i32))
            .collect();
        let mut host = budget.pinned::<i32>(0, ints.len());
        host.copy_from_slice(&ints);
        let mut tables_dev = budget.device::<i32>(0, ints.len());
        let mut f64s = budget.device::<f64>(0, RECORD_F64_FIELDS * n_rec);
        let mut i32s = budget.device::<i32>(0, RECORD_I32_FIELDS * n_rec);
        let host = mmd_records(
            &stream,
            (n_shell, n_prim),
            &shells,
            (n_pairs, n_rec),
            host,
            &mut tables_dev,
            &mut f64s,
            &mut i32s,
        )
        .unwrap();
        assert_eq!(&host.sync().unwrap()[..], &ints[..], "the tables come back");
        let got_f = stream
            .download(budget.pinned::<f64>(0, RECORD_F64_FIELDS * n_rec), &f64s)
            .unwrap()
            .sync()
            .unwrap();
        let got_i = stream
            .download(budget.pinned::<i32>(0, RECORD_I32_FIELDS * n_rec), &i32s)
            .unwrap()
            .sync()
            .unwrap();
        assert_eq!((c.la, c.lb, c.n_rec()), (w.la, w.lb, w.n_rec), "the class");
        for k in 0..RECORD_F64_FIELDS {
            let got = &got_f[k * n_rec..(k + 1) * n_rec];
            let off = got
                .iter()
                .zip(&w.f64s[k])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            if off > 0 {
                misses.push(format!(
                    "({},{}) double field {k}: {off} records",
                    c.la, c.lb
                ));
            }
        }
        for k in 0..RECORD_I32_FIELDS {
            if got_i[k * n_rec..(k + 1) * n_rec] != w.i32s[k][..] {
                misses.push(format!("({},{}) integer field {k}", c.la, c.lb));
            }
        }
    }
    assert!(misses.is_empty(), "not bitwise:\n{}", misses.join("\n"));
}
