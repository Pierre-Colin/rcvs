extern crate nalgebra as na;

use std::iter::repeat;

type Vector = na::DVector<f64>;
type Matrix = na::DMatrix<f64>;
type MatrixSlice<'a> = na::DMatrixSlice<'a, f64>;

fn make_a(constraints: &Matrix) -> Matrix {
    let (m, n) = constraints.shape();
    let mut a = Matrix::from_element(m, m + n, 0f64);
    (0..m).for_each(|i| a[(i, i)] = 1f64);
    for i in 0..m {
        for j in 0..n { a[(i, j + m)] = constraints[(i, j)]; }
    }
    a
}

fn feasible_basic_vector(ab: &MatrixSlice, b: &Vector) -> Vector {
    let lu = ab.clone().lu();
    let xb = lu.solve(b).unwrap();
    if xb.iter().all(|x| *x >= 0f64) {
        xb
    } else {
        // FIXME: implement actual phase 1 from NR
        if b.iter().all(|x| *x >= 0f64) {
            Vector::from_element(xb.len(), 1f64 / (3f64 * xb.len() as f64))
        } else if b.iter().all(|x| *x <= 0f64) {
            Vector::from_element(xb.len(), 1f64 / (xb.len() as f64))
        } else {
            panic!("Vector doesn't have a constant sign: {}", b.transpose());
        }
    }
}

fn make_indices(b: &Vector, m: usize, n: usize) -> Vec<Option<usize>> {
    if b.iter().all(|x| *x >= 0f64) {
        repeat(None).take(m).chain((0..n).map(|i| Some(i))).collect()
    } else if b.iter().all(|x| *x <= 0f64) {
        (0..n).map(|i| Some(i)).chain(repeat(None).take(m)).collect()
    } else {
        panic!("Vector doesn't have a constant sign: {}", b.transpose());
    }
}

// NOTE: prints are for debugging purposes and should be removed eventually
pub fn simplex(constraints: &Matrix, cost: &Vector, b: &Vector) -> Vector {
    let (m, n) = constraints.shape();
    let mut a = make_a(constraints);
    // 1. Find a feasible basis
    let mut xb = feasible_basic_vector(&a.columns(0, m).clone(), b);
    let mut c = cost.clone();
    let mut ind = make_indices(b, m, n);
    //println!("b = {}", b.transpose());
    for _pass in 1.. {
        //println!("========== Pass {} ==========", _pass);
        //println!("A = {}", a);
        //println!("c = {}", c.transpose());
        //println!("xB = {}", xb.transpose());
        //println!("ind = {:?}", ind);
        // 2. Compute reduced costs for all xk not in basis
        //println!("[[[ Phase 2 ]]]");
        let y = a.columns(0, m).transpose().lu().solve(&c.rows(0, m)).unwrap();
        let u = c.rows(m, n) - a.columns(m, n).clone().transpose() * y;
        //println!("u = {}", u.transpose());
        // 3. If all uk >= 0, break; otherwise choose most negative uk
        //println!("[[[ Phase 3 ]]]");
        if u.iter().all(|x| *x >= 0f64) { break; }
        let k = u.into_iter().enumerate()
                 .min_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
                 .unwrap().0 + m;
        //println!("k = {}", k);
        // 4. Minimum ratio test
        //println!("[[[ Phase 4 ]]]");
        let ablu = a.columns(0, m).clone().lu();
        xb = ablu.solve(b).unwrap();
        //println!("New xB value: {}", xb.transpose());
        let w = ablu.solve(&a.column(k)).unwrap();
        //println!("w = {}", w.transpose());
        assert!(w.iter().any(|e| *e > 0f64), "Objective function is unbounded");
        if let Some((i, _)) = w.into_iter().zip(xb.iter()).enumerate()
                               .filter(|(_, (wi, _))| **wi > 0f64)
                               .min_by(|(_, (wi, xi)), (_, (wj, xj))|
                                       (*xi / *wi).partial_cmp(&(*xj / *wj))
                                                  .unwrap()) {
            // 5. Swap columns i and k
            //println!("Swapping columns k = {} and i = {}", k, i);
            a.swap_columns(i, k);
            c.swap_rows(i, k);
            let temp = ind[i];
            ind[i] = ind[k];
            ind[k] = temp;
        } else {
            break;
        }
    }
    xb = a.columns(0, m).clone().lu().solve(b).unwrap();
    let x = Vector::from_iterator(
        n,
        (0..n).map(|i|
            if let Some((e, _)) = xb.into_iter().zip(ind.iter())
                                    .find(|(_, n)| **n == Some(i)) {
                *e
            } else {
                0f64
            }
        )
    );
    x
}

pub fn vector_to_lottery(x: Vector) -> Vec<f64> {
    let v: f64 = x.iter().sum();
    x.into_iter().map(|e| e / v).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wiki() {
        let mut m = Matrix::zeros(2, 3);
        m[(0, 0)] = 3f64;
        m[(0, 1)] = 2f64;
        m[(0, 2)] = 1f64;
        m[(1, 0)] = 2f64;
        m[(1, 1)] = 5f64;
        m[(1, 2)] = 3f64;
        let b = Vector::from_iterator(2, vec![10f64, 15f64].into_iter());
        let c = Vector::from_iterator(5, vec![0f64, 0f64, -2f64, -3f64, -4f64].into_iter());
        let x = simplex(&m, &c, &b);
        assert!(x.iter().take(2).all(|x| *x < 0.000001f64));
        assert!(f64::abs(x.iter().last().unwrap() - 5f64) < 0.000001f64);
    }
}
