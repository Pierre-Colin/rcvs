extern crate nalgebra as na;

use std::{
    error::Error,
    fmt,
    iter::repeat,
};

type Vector = na::DVector<f64>;
type Matrix = na::DMatrix<f64>;

#[derive(Debug)]
pub enum SimplexError {
    Unsolvable(Matrix, Vector),
    Filtering,
    Unbounded,
    Loop,
    Unfeasible,
    NanVector(Vector),
    Whatever,
}

impl fmt::Display for SimplexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SimplexError::Unsolvable(a, b) =>
                writeln!(f, "System is unsolvable:\n{}\n{}", a, b),
            SimplexError::Filtering => writeln!(f, "Iterator filtering failed"),
            SimplexError::Unbounded => writeln!(f, "Objective function is unbounded"),
            SimplexError::Loop => writeln!(f, "Iteration limit reached"),
            SimplexError::Unfeasible => writeln!(f, "Program is unfeasible"),
            SimplexError::NanVector(v) =>
                writeln!(f, "Vector contains NaN: {}", v.transpose()),
            SimplexError::Whatever => writeln!(f, "Unknown error"),
        }
    }
}

impl Error for SimplexError {
    fn description(&self) -> &str {
        match self {
            SimplexError::Unsolvable(_, _) =>
                "System contained in the error is unsolvable",
            SimplexError::Filtering => "An iterator was filtered empty",
            SimplexError::Unbounded =>
                "Objective function is unbounded on given domain",
            SimplexError::Loop => "Iteration limit reached",
            SimplexError::Unfeasible => "Feasible region is empty",
            SimplexError::NanVector(_) => "Vector contains NaN",
            SimplexError::Whatever => "Unknown error",
        }
    }

    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

fn make_a(constraints: &Matrix) -> Matrix {
    let (m, n) = constraints.shape();
    let mut a = Matrix::from_element(m, m + n, 0f64);
    (0..m).for_each(|i| a[(i, i)] = 1f64);
    for i in 0..m {
        for j in 0..n { a[(i, j + m)] = constraints[(i, j)]; }
    }
    a
}

fn make_c(cost: &Vector, m: usize) -> Vector {
    Vector::from_iterator(
        m + cost.len(),
        repeat(0f64).take(m).chain(cost.iter().cloned())
    )
}

fn make_indices(m: usize, n: usize) -> Vec<Option<usize>> {
    repeat(None).take(m).chain((0..n).map(|i| Some(i))).collect()
}

fn choose_pivot(a: &Matrix, c: &Vector) -> Result<Option<usize>, SimplexError> {
    // A has shape (m, m + n)
    let (m, mn) = a.shape();
    let n = mn - m;
    //println!("a = {}", a);
    //println!("c = {}", c.transpose());
    //println!("(m, n) = ({}, {})", m, n);
    // Compute reduced costs for all xk not in basis
    let abt = a.columns(0, m).transpose();
    let cb = c.rows(0, m);
    let y = abt.clone_owned().lu().solve(&cb).ok_or_else(||
        SimplexError::Unsolvable(abt, cb.clone_owned())
    )?;
    //println!("y = {}", y.transpose());
    let u = c.rows(m, n) - a.columns(m, n).clone().transpose() * y;
    //println!("u = {}", u.transpose());
    if u.iter().any(|x| x.is_nan()) { return Err(SimplexError::NanVector(u)); }
    match u.into_iter().enumerate()
                       .min_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
    {
        Some((k, &x)) => Ok(if x >= 0f64 { None } else { Some(k + m) }),
        None => Err(SimplexError::Filtering),
    }
}

fn make_auxiliary_objective(xb: &Vector, l: usize) -> Vector {
    Vector::from_iterator(l, xb.iter().map(|x|
        if *x < 0f64 { -1f64 } else { 0f64 }
    ).chain(repeat(0f64).take(l - xb.len())))
}

fn feasible_basic_vector(a: &mut Matrix,
                         b: &Vector,
                         c: &mut Vector,
                         ind: &mut [Option<usize>])
    -> Result<(), SimplexError>
{
    let (m, mn) = a.shape();
    let ab = a.columns(0, m).clone_owned();
    let mut xb = ab.clone().lu().solve(b).ok_or_else(||
        SimplexError::Unsolvable(ab, b.clone())
    )?;
    let mut v = make_auxiliary_objective(&xb, c.len());
    if v.iter().all(|x| *x == 0f64) { return Ok(()); }
    for _pass in 1.. {
        if _pass > 10 * mn { return Err(SimplexError::Loop); }
        //println!("Pass {}", _pass);
        let ab = a.columns(0, m).clone_owned();
        let ablu = ab.clone().lu();
        xb = ablu.solve(b).ok_or_else(||
            SimplexError::Unsolvable(ab.clone(), b.clone())
        )?;
        if xb.iter().all(|x| *x >= 0f64) { break; }
        //println!("Feasible basic xb = {}", xb.transpose());
        let k = choose_pivot(a, &v)?.ok_or(SimplexError::Unfeasible)?;
        //println!("k = {}", k);
        let w = ablu.solve(&a.column(k)).ok_or_else(||
            SimplexError::Unsolvable(ab, a.column(k).clone_owned())
        )?;
        //println!("w = {}", w.transpose());
        if let Some((i, _)) = w.into_iter().zip(xb.iter()).enumerate()
                               .filter(|(_, (_, xi))| **xi < 0f64)
                               .min_by(|(_, (wi, xi)), (_, (wj, xj))|
                                       (*xi / *wi).partial_cmp(&(*xj / *wj))
                                                  .unwrap()) {
            // 5. Swap columns i and k
            //println!("Swapping columns k = {} and i = {}", k, i);
            a.swap_columns(i, k);
            c.swap_rows(i, k);
            v.swap_rows(i, k);
            ind.swap(i, k);
        } else {
            break;
        }
    }
    //println!("Finished phase 1 with xb = {}", xb.transpose());
    Ok(())
}

// NOTE: prints are for debugging purposes and should be removed eventually
pub fn simplex(constraints: &Matrix, cost: &Vector, b: &Vector)
    -> Result<Vector, SimplexError>
{
    let (m, n) = constraints.shape();
    let mut a = make_a(constraints);
    // 1. Find a feasible basis
    let mut c = make_c(&cost, m);
    let mut ind = make_indices(m, n);
    let mut xb;
    feasible_basic_vector(&mut a, b, &mut c, &mut ind)?;
    //println!("b = {}", b.transpose());
    for _pass in 1.. {
        if _pass > 10 * (m + n) { return Err(SimplexError::Loop); }
        //println!("========== Pass {} ==========", _pass);
        //println!("A = {}", a);
        //println!("c = {}", c.transpose());
        //println!("xB = {}", xb.transpose());
        //println!("ind = {:?}", ind);
        // 2. Compute reduced costs for all xk not in basis
        //println!("[[[ Phase 2 ]]]");
        //println!("u = {}", u.transpose());
        // 3. If all uk >= 0, break; otherwise choose most negative uk
        //println!("[[[ Phase 3 ]]]");
        let k: usize;
        match choose_pivot(&a, &c)? {
            None => break,
            Some(kk) => k = kk,
        };
        //println!("k = {}", k);
        // 4. Minimum ratio test
        //println!("[[[ Phase 4 ]]]");
        let ab = a.columns(0, m).clone_owned();
        let ablu = ab.clone_owned().lu();
        xb = ablu.solve(b).ok_or_else(||
            SimplexError::Unsolvable(ab.clone(), b.clone())
        )?;
        //println!("New xB value: {}", xb.transpose());
        let w = ablu.solve(&a.column(k)).ok_or_else(||
            SimplexError::Unsolvable(ab, a.column(k).clone_owned())
        )?;
        //println!("w = {}", w.transpose());
        if w.iter().all(|e| *e <= 0f64) {
            return Err(SimplexError::Unbounded);
        }
        // TODO: change this epsilon to something better
        if let Some((i, _)) = w.into_iter().zip(xb.iter()).enumerate()
                               .filter(|(_, (wi, _))| **wi > 1e-6)
                               .min_by(|(_, (wi, xi)), (_, (wj, xj))|
                                       (*xi / *wi).partial_cmp(&(*xj / *wj))
                                                  .unwrap()) {
            // 5. Swap columns i and k
            //println!("Swapping columns k = {} and i = {}", k, i);
            a.swap_columns(i, k);
            c.swap_rows(i, k);
            ind.swap(i, k);
        } else {
            return Err(SimplexError::Whatever);
        }
    }
    //println!("Almost finished phase 2 with xb = {}", xb.transpose());
    let ab = a.columns(0, m).clone_owned();
    xb = ab.clone().lu().solve(b).ok_or_else(||
        SimplexError::Unsolvable(ab, b.clone())
    )?;
    //println!("Almost finished phase 2 with xb = {}", xb.transpose());
    //println!("with indices {:?}", ind);
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
    //println!("Finished phase 2 with x = {}", x.transpose());
    Ok(x)
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
        let c = Vector::from_iterator(3, vec![-2f64, -3f64, -4f64].into_iter());
        let x = simplex(&m, &c, &b).unwrap();
        assert!(x.iter().take(2).all(|x| *x < 0.000001f64));
        assert!(f64::abs(x.iter().last().unwrap() - 5f64) < 0.000001f64);
    }

    // https://cbom.atozmath.com/CBOM/Simplex.aspx?q=sm&q1=6%606%60MAX%60Z%60x1%2cx2%2cx3%2cx4%2cx5%2cx6%601%2c1%2c1%2c1%2c1%2c1%602%2c1%2c1%2c1%2c3%2c1%3b3%2c2%2c1%2c3%2c1%2c3%3b3%2c3%2c2%2c1%2c3%2c1%3b3%2c1%2c3%2c2%2c1%2c1%3b1%2c3%2c1%2c3%2c2%2c3%3b3%2c1%2c3%2c3%2c1%2c2%60%3c%3d%2c%3c%3d%2c%3c%3d%2c%3c%3d%2c%3c%3d%2c%3c%3d%601%2c1%2c1%2c1%2c1%2c1%60%60A%60false%60true%60false%60true%60false%60false%60true&do=1#PrevPart
    #[test]
    fn screw_up() {
        let m = Matrix::from_iterator(6, 6,
            vec![
                2f64, 1f64, 1f64, 1f64, 3f64, 1f64,
                3f64, 2f64, 1f64, 3f64, 1f64, 3f64,
                3f64, 3f64, 2f64, 1f64, 3f64, 1f64,
                3f64, 1f64, 3f64, 2f64, 1f64, 1f64,
                1f64, 3f64, 1f64, 3f64, 2f64, 3f64,
                3f64, 1f64, 3f64, 3f64, 1f64, 2f64,
            ].into_iter()
        ).transpose();
        let b = Vector::from_element(6, 1f64);
        let c = Vector::from_element(6, -1f64);
        println!("Running simplex with\nM = {}\nb = {}\nc = {}", m, b.transpose(), c.transpose());
        let x = simplex(&m, &c, &b).unwrap();
        let sixth = 1f64 / 6f64;
        assert!(
            x.iter().zip([0f64, 0f64, sixth, 0f64, sixth, sixth].into_iter()).all(|(x, y)| (x - y).abs() < 1e-6),
            "x = {}\nexpected (0, 0, 1/6, 0, 1/6, 1/6)", x.transpose()
        );
    }
}
