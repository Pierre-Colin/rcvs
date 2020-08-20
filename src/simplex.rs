extern crate nalgebra as na;

use std::{error::Error, fmt, iter::repeat};

type Vector = na::DVector<f64>;
type Matrix = na::DMatrix<f64>;
type LuDecomp = na::LU<f64, na::Dynamic, na::Dynamic>;

/// Gives possible reasons why the simplex algorithm failed to solve a linear
/// program.
#[derive(Debug)]
pub enum SimplexError {
    /// The system described by the arguments is unsolvable.
    Unsolvable(Matrix, Vector),

    /// A vector was filtered empty while it should not.
    Filtering,

    /// The objective function is unbounded for the given feasible region.
    Unbounded,

    /// The algorithm exceeded the iteration limit.
    Loop,

    /// The linear program is unfeasible.
    Unfeasible,

    /// A vector ended up containing `NaN`.
    NanVector(Vector),

    /// Unknown error.
    Whatever,
}

impl fmt::Display for SimplexError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SimplexError::Unsolvable(a, b) => writeln!(f, "System is unsolvable:\n{}\n{}", a, b),
            SimplexError::Filtering => writeln!(f, "Iterator filtering failed"),
            SimplexError::Unbounded => writeln!(f, "Objective function is unbounded"),
            SimplexError::Loop => writeln!(f, "Iteration limit reached"),
            SimplexError::Unfeasible => writeln!(f, "Program is unfeasible"),
            SimplexError::NanVector(v) => writeln!(f, "Vector contains NaN: {}", v.transpose()),
            SimplexError::Whatever => writeln!(f, "Unknown error"),
        }
    }
}

impl Error for SimplexError {
    fn description(&self) -> &str {
        match self {
            SimplexError::Unsolvable(_, _) => "System contained in the error is unsolvable",
            SimplexError::Filtering => "An iterator was filtered empty",
            SimplexError::Unbounded => "Objective function is unbounded on given domain",
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
        for j in 0..n {
            a[(i, j + m)] = constraints[(i, j)];
        }
    }
    a
}

fn make_c(cost: &Vector, m: usize) -> Vector {
    Vector::from_iterator(
        m + cost.len(),
        repeat(0f64).take(m).chain(cost.iter().cloned()),
    )
}

fn make_indices(m: usize, n: usize) -> Vec<Option<usize>> {
    repeat(None)
        .take(m)
        .chain((0..n).map(|i| Some(i)))
        .collect()
}

fn manhattan_norm(x: &Vector) -> f64 {
    let mut a: Vec<f64> = x.iter().map(|x| x.abs()).collect();
    a.sort_unstable_by(|x, y| x.partial_cmp(&y).unwrap());
    a.into_iter().sum()
}

fn choose_pivot(a: &Matrix, c: &Vector, norm: f64) -> Result<Option<usize>, SimplexError> {
    // A has shape (m, m + n)
    let (m, mn) = a.shape();
    let n = mn - m;
    //println!("a = {}", a);
    //println!("c = {}", c.transpose());
    //println!("(m, n) = ({}, {})", m, n);
    // Compute reduced costs for all xk not in basis
    let abt = a.columns(0, m).transpose();
    let cb = c.rows(0, m);
    let y = abt
        .clone_owned()
        .lu()
        .solve(&cb)
        .ok_or_else(|| SimplexError::Unsolvable(abt, cb.clone_owned()))?;
    //println!("y = {}", y.transpose());
    let u = c.rows(m, n) - a.columns(m, n).clone().transpose() * y;
    //println!("u = {}", u.transpose());
    if u.iter().any(|x| x.is_nan()) {
        return Err(SimplexError::NanVector(u));
    }
    // TODO: find a better epsilon if possible
    match u
        .into_iter()
        .enumerate()
        .min_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
    {
        Some((k, &x)) => Ok(if x >= -norm * 1e-10 {
            None
        } else {
            Some(k + m)
        }),
        None => Err(SimplexError::Filtering),
    }
}

fn make_auxiliary_objective(xb: &Vector, l: usize) -> Vector {
    Vector::from_iterator(
        l,
        xb.iter()
            .map(|x| if *x < 0f64 { -1f64 } else { 0f64 })
            .chain(repeat(0f64).take(l - xb.len())),
    )
}

struct LinearProgram {
    constraints: Matrix,
    costs: Vector,
    limits: Vector,
    a: Matrix,
    indices: Vec<Option<usize>>,
    feasible_basis: Vector,
    a_basis: Matrix,
    a_basis_lu: LuDecomp,
}

impl LinearProgram {
    pub fn new(constraints: &Matrix, costs: &Vector, limits: &Vector) -> Self {
        let (m, n) = constraints.shape();
        let a = make_a(constraints);
        let a_basis = a.columns(0, a.nrows()).clone_owned();
        let a_basis_lu = a_basis.clone().lu();

        Self {
            constraints: constraints.to_owned(),
            costs: make_c(costs, m),
            limits: limits.to_owned(),
            a: a,
            indices: make_indices(m, n),
            feasible_basis: unsafe { Vector::new_uninitialized(0) },
            a_basis: a_basis,
            a_basis_lu: a_basis_lu,
        }
    }

    fn check_pass_too_high(&self, pass: usize) -> Result<(), SimplexError> {
        if pass <= 10 * self.a.ncols() {
            Ok(())
        } else {
            Err(SimplexError::Loop)
        }
    }

    fn extract_a_basis(&self) -> Matrix {
        self.a.columns(0, self.a.nrows()).clone_owned()
    }

    fn make_feasible_basic_vector_from_ab(&mut self) -> Result<(), SimplexError> {
        self.feasible_basis = self
            .a_basis_lu
            .solve(&self.limits)
            .ok_or_else(|| SimplexError::Unsolvable(self.a_basis.clone(), self.limits.clone()))?;
        Ok(())
    }

    fn choose_pivot_b(&self, objective: &Vector) -> Result<Option<usize>, SimplexError> {
        choose_pivot(&self.a, objective, manhattan_norm(&self.feasible_basis))
    }

    fn make_w(&self, k: usize) -> Result<Vector, SimplexError> {
        self.a_basis_lu.solve(&self.a.column(k)).ok_or_else(|| {
            SimplexError::Unsolvable(self.a_basis.clone(), self.a.column(k).clone_owned())
        })
    }

    fn get_w_min_by_filter(
        &self,
        w: Vector,
        predicate: impl FnMut(&(usize, (&f64, &f64))) -> bool,
    ) -> Option<usize> {
        w.into_iter()
            .zip(self.feasible_basis.iter())
            .enumerate()
            .filter(predicate)
            .min_by(|(_, (wi, xi)), (_, (wj, xj))| (*xi / *wi).partial_cmp(&(*xj / *wj)).unwrap())
            .map(|(i, _)| i)
    }

    fn get_w_min_phase_0(&self, w: Vector) -> Option<usize> {
        self.get_w_min_by_filter(w, |(_, (_, xi))| **xi < 0f64)
    }

    fn get_w_min_main_loop(&self, w: Vector) -> Option<usize> {
        // TODO: find a better epsilon if possible
        self.get_w_min_by_filter(w, |(_, (wi, _))| **wi > 1e-6)
    }

    fn swap_columns(&mut self, i: usize, j: usize) {
        self.a.swap_columns(i, j);
        self.costs.swap_rows(i, j);
        self.indices.swap(i, j);
    }

    fn decompose_a_basis(&mut self) {
        self.a_basis = self.extract_a_basis();
        self.a_basis_lu = self.a_basis.clone().lu();
    }

    fn find_feasible_basic_vector(&mut self) -> Result<(), SimplexError> {
        self.decompose_a_basis();
        self.make_feasible_basic_vector_from_ab()?;
        let mut v = make_auxiliary_objective(&self.feasible_basis, self.costs.len());
        if v.iter().all(|x| *x == 0f64) {
            return Ok(());
        }
        for pass in 1.. {
            self.check_pass_too_high(pass)?;
            self.decompose_a_basis();
            self.make_feasible_basic_vector_from_ab()?;
            if self.feasible_basis.iter().all(|x| *x >= 0f64) {
                break;
            }
            let pivot = self.choose_pivot_b(&v)?.ok_or(SimplexError::Unfeasible)?;
            let w = self.make_w(pivot)?;
            if let Some(i) = self.get_w_min_phase_0(w) {
                self.swap_columns(i, pivot);
                v.swap_rows(i, pivot);
            } else {
                break;
            }
        }
        Ok(())
    }

    fn make_solution_from_basic_vector(&self) -> Vector {
        let n = self.constraints.ncols();
        Vector::from_iterator(
            n,
            (0..n).map(|i| {
                self.feasible_basis
                    .into_iter()
                    .zip(self.indices.iter())
                    .find(|(_, n)| **n == Some(i))
                    .map(|(e, _)| *e)
                    .unwrap_or(0f64)
            }),
        )
    }

    fn try_swap_main_loop(&mut self, w: Vector, pivot: usize) -> Result<(), SimplexError> {
        if let Some(i) = self.get_w_min_main_loop(w) {
            self.swap_columns(i, pivot);
            Ok(())
        } else {
            Err(SimplexError::Whatever)
        }
    }

    pub fn solve(mut self) -> Result<Vector, SimplexError> {
        // 1. Find feasible basis
        self.find_feasible_basic_vector()?;
        //println!("b = {}", b.transpose());
        for pass in 1.. {
            self.check_pass_too_high(pass)?;
            // 2. Compute reduced costs for all xk not in basis
            self.decompose_a_basis();
            self.make_feasible_basic_vector_from_ab()?;
            // 3. If all uk >= 0, break; otherwise choose most negative uk
            let pivot = match self.choose_pivot_b(&self.costs)? {
                None => break,
                Some(k) => k,
            };
            // 4. Minimum ratio test
            let w = self.make_w(pivot)?;
            if w.iter().all(|e| *e <= 0f64) {
                return Err(SimplexError::Unbounded);
            }
            self.try_swap_main_loop(w, pivot)?;
        }
        //println!("Almost finished phase 2 with xb = {}", xb.transpose());
        self.decompose_a_basis();
        self.make_feasible_basic_vector_from_ab()?;
        let x = self.make_solution_from_basic_vector();
        Ok(x)
    }
}

pub fn simplex(constraints: &Matrix, cost: &Vector, b: &Vector) -> Result<Vector, SimplexError> {
    let program = LinearProgram::new(constraints, cost, b);
    program.solve()
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

    // https://cbom.atozmath.com/CBOM/Simplex.aspx?q=rsm&q1=6%606%60MAX%60Z%60x1%2Cx2%2Cx3%2Cx4%2Cx5%2Cx6%601%2C1%2C1%2C1%2C1%2C1%602%2C1%2C3%2C1%2C1%2C1%3B3%2C2%2C1%2C2%2C3%2C3%3B1%2C3%2C2%2C2%2C3%2C2%3B3%2C2%2C2%2C2%2C1%2C2%3B3%2C1%2C1%2C3%2C2%2C1%3B3%2C1%2C2%2C2%2C3%2C2%60%3C%3D%2C%3C%3D%2C%3C%3D%2C%3C%3D%2C%3C%3D%2C%3C%3D%601%2C1%2C1%2C1%2C1%2C1%60%60A%60false%60true%60false%60true%60false%60false%60true&do=1#tblSolution
    #[test]
    fn failed_alpha() {
        let m = Matrix::from_iterator(
            6,
            6,
            vec![
                2f64, 1f64, 3f64, 1f64, 1f64, 1f64, 3f64, 2f64, 1f64, 2f64, 3f64, 3f64, 1f64, 3f64,
                2f64, 2f64, 3f64, 2f64, 3f64, 2f64, 2f64, 2f64, 1f64, 2f64, 3f64, 1f64, 1f64, 3f64,
                2f64, 1f64, 3f64, 1f64, 2f64, 2f64, 3f64, 2f64,
            ]
            .into_iter(),
        )
        .transpose();
        let b = Vector::from_element(6, 1f64);
        let c = Vector::from_element(6, -1f64);
        println!(
            "Running simplex with\nM = {}\nb = {}\nc = {}",
            m,
            b.transpose(),
            c.transpose()
        );
        let _x = simplex(&m, &c, &b).expect("Simplex algorithm failed");
    }
}
