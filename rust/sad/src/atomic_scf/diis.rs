//! `:diis`: Pulay DIIS on the host, over flat vectors. `src/recipes/diis`
//! is the device-resident one; this one holds a few hundred doubles per
//! iterate. The elimination, the B matrix and the extrapolation run in the
//! C++ order.

use std::collections::VecDeque;

/// Solve `A x = b` by Gaussian elimination with partial pivoting:
/// `bordered_solve`. `None` if a pivot vanishes (the DIIS matrix goes
/// singular as its residuals align).
fn bordered_solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut pivot = col;
        for r in col + 1..n {
            if a[r + (n * col)].abs() > a[pivot + (n * col)].abs() {
                pivot = r;
            }
        }
        if a[pivot + (n * col)].abs() < 1e-14 {
            return None;
        }
        if pivot != col {
            for c in 0..n {
                a.swap(col + (n * c), pivot + (n * c));
            }
            b.swap(col, pivot);
        }
        for r in col + 1..n {
            let f = a[r + (n * col)] / a[col + (n * col)];
            for c in col..n {
                a[r + (n * c)] -= f * a[col + (n * c)];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut sum = b[r];
        for c in r + 1..n {
            sum -= a[r + (n * c)] * x[c];
        }
        x[r] = sum / a[r + (n * r)];
    }
    Some(x)
}

/// Pulay DIIS, `AtomicDiis`: keeps the last few (Fock, error) pairs and
/// extrapolates the Fock vector whose error combination is smallest.
pub(super) struct AtomicDiis {
    depth: usize,
    fock: VecDeque<Vec<f64>>,
    error: VecDeque<Vec<f64>>,
}

impl AtomicDiis {
    /// A DIIS keeping `depth` pairs; the oldest is dropped past it.
    pub(super) fn new(depth: usize) -> AtomicDiis {
        AtomicDiis {
            depth,
            fock: VecDeque::new(),
            error: VecDeque::new(),
        }
    }

    /// Record `fock` and its `error`, then return the extrapolated Fock
    /// vector over everything held: `extrapolate`.
    ///
    /// When the DIIS matrix is singular (the residuals have aligned) the
    /// oldest pairs are dropped until it is not; with one pair left that
    /// pair's Fock vector is returned as is.
    pub(super) fn extrapolate(&mut self, fock: Vec<f64>, error: Vec<f64>) -> Vec<f64> {
        self.fock.push_back(fock);
        self.error.push_back(error);
        if self.fock.len() > self.depth {
            self.fock.pop_front();
            self.error.pop_front();
        }
        loop {
            let depth = self.fock.len();
            let dim = depth + 1;
            // [B -1; -1 0] [w; lambda] = [0; -1], B_ij = e_i . e_j.
            let mut b = vec![0.0; dim * dim];
            let mut rhs = vec![0.0; dim];
            for i in 0..depth {
                for j in 0..depth {
                    // std::inner_product's order: one running sum from 0.0.
                    let mut dot = 0.0;
                    for (x, y) in self.error[i].iter().zip(&self.error[j]) {
                        dot += x * y;
                    }
                    b[i + (dim * j)] = dot;
                }
                b[i + (dim * depth)] = -1.0;
                b[depth + (dim * i)] = -1.0;
            }
            rhs[depth] = -1.0;
            match bordered_solve(b, rhs, dim) {
                None if depth > 1 => {
                    self.fock.pop_front();
                    self.error.pop_front();
                }
                None => return self.fock[depth - 1].clone(),
                Some(w) => {
                    let mut out = vec![0.0; self.fock[depth - 1].len()];
                    for (i, f) in self.fock.iter().enumerate() {
                        for (o, x) in out.iter_mut().zip(f) {
                            *o += w[i] * x;
                        }
                    }
                    return out;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two pairs whose errors are `+e` and `−e` extrapolate to their mean,
    /// where the combined error vanishes; one pair is returned as is.
    #[test]
    fn extrapolates_to_the_zero_error_combination() {
        let mut diis = AtomicDiis::new(8);
        assert_eq!(diis.extrapolate(vec![1.0, 2.0], vec![0.5, 0.0]), [1.0, 2.0]);
        let out = diis.extrapolate(vec![3.0, 4.0], vec![-0.5, 0.0]);
        assert!((out[0] - 2.0).abs() < 1e-15 && (out[1] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn a_vanishing_pivot_is_none() {
        assert!(bordered_solve(vec![1.0, 1.0, 1.0, 1.0], vec![1.0, 1.0], 2).is_none());
    }
}
