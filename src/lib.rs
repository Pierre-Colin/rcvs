//! # Randomized Condorcet Voting System
//!
//! The crate `rcvs` implements the Randomized Condorcet Voting System, a
//! strategy-proof voting system using game theory to generalize the original
//! Condorcet method.
//!
//! ## Condorcet method
//!
//! The Condorcet method consists of building a directed graph called the _duel
//! graph_ of the election. Its vertices are the alternatives to vote among,
//! and an arrow between two alternatives _A_ and _B_ means _A_ is preferred
//! over _B_ more often than the opposite. The Condorcet criterion states that
//! if the duel graph has a unique source, then the alternative it corresponds
//! to never loses in a duel against another alternative and therefore must be
//! elected.
//!
//! ## Randomization
//!
//! If no source or several exist, then the Condorcet criterion is not
//! applicable and something else must be used. As surprising as it seems,
//! randomly picking the winner usually yields very good properties in voting
//! systems, but in order to maximize the electors' utility (or rather minimize
//! the number of electors who end up wishing another alternative won), the
//! probability distribution used to pick the winner (called strategy) is not
//! necessarily uniform. Computing the optimal strategy requires some knowledge
//! of game theory and linear programming, and the resulting voting system has
//! excellent strategic properties.
//!
//! ## Implementation
//!
//! This crate provides structures to carry out elections using the Randomized
//! Condorcet Voting System in Rust. It uses the crate
//! [nalgebra](https://crates.io/crates/nalgebra) to solve linear programs
//! and compute the optimal strategy, and [rand](https://crates.io/crates/rand)
//! to generate pseudo-random numbers which are used both for picking winners
//! randomly and for more efficient internal numerical algorithms.
//!
//! It is never mentioned in this documentation, but whenever a method takes an
//! argument implementing `rand::Rng`, it means it will make use of
//! pseudo-random numbers and the programmer will need to provide one,
//! `rand::thread_rng()` being a quick-and-dirty default which is used in this
//! crate's unit tests.
extern crate nalgebra as na;
extern crate rand;

mod ballot;
mod simplex;
mod strategies;
pub mod util;

use std::{
    clone::Clone,
    cmp::{Eq, Ordering},
    collections::{HashMap, HashSet},
    error::Error,
    fmt,
    hash::Hash,
};

pub use ballot::Ballot;
pub use ballot::Rank;
pub use simplex::SimplexError;
pub use strategies::Strategy;

type Adjacency = na::DMatrix<bool>;
type Matrix = na::DMatrix<f64>;
type Vector = na::DVector<f64>;

#[derive(Clone, Debug, Hash)]
struct Arrow<A>(A, A);

impl<A: Eq> PartialEq for Arrow<A> {
    fn eq(&self, other: &Arrow<A>) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl<A: Eq> Eq for Arrow<A> {}

/// Implements the duel graph of an election.
#[derive(Clone, Debug)]
pub struct DuelGraph<A: fmt::Debug> {
    v: Vec<A>,
    a: Adjacency,
}

/// Implements errors in the election process. Interfaces with simplex errors.
#[derive(Debug)]
pub enum ElectionError {
    /// The simplex algorithm failed to compute both the minimax and maximin
    /// strategies; the underlying errors are contained in the arguments.
    BothFailed(simplex::SimplexError, simplex::SimplexError),

    /// The simplex algorithm failed to compute the strategy; the underlying
    /// error is contained in the argument.
    SimplexFailed(simplex::SimplexError),

    /// The operation failed because the election is already closed.
    ElectionClosed,

    /// The operation failed because the election is still open.
    ElectionOpen,
}

impl fmt::Display for ElectionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ElectionError::BothFailed(a, b) => {
                writeln!(f, "Both methods failed:")?;
                writeln!(f, " * minimax: {}", a)?;
                writeln!(f, " * maximin: {}", b)
            }
            ElectionError::SimplexFailed(e) => write!(f, "Simplex algorithm failed: {}", e),
            ElectionError::ElectionClosed => write!(f, "Election is closed"),
            ElectionError::ElectionOpen => write!(f, "Election is open"),
        }
    }
}

impl From<simplex::SimplexError> for ElectionError {
    fn from(error: simplex::SimplexError) -> Self {
        ElectionError::SimplexFailed(error)
    }
}

impl Error for ElectionError {
    fn description(&self) -> &str {
        match self {
            ElectionError::BothFailed(_, _) => {
                "Both minimax and maximin strategies failed to be solved"
            }
            ElectionError::SimplexFailed(_) => {
                "The simplex algorithm failed to compute the strategy"
            }
            ElectionError::ElectionClosed => "Election is already closed",
            ElectionError::ElectionOpen => "Election is still open",
        }
    }

    fn source(&self) -> Option<&(dyn Error + 'static)> {
        // in case of multiple cause, no other choice but to return itself
        match self {
            ElectionError::BothFailed(_, _) => Some(self),
            ElectionError::SimplexFailed(e) => Some(e),
            _ => None,
        }
    }
}

impl<A: fmt::Debug> fmt::Display for DuelGraph<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Graph {{")?;
        writeln!(f, "Alternatives: {:?}", self.v)?;
        writeln!(f, "{}", self.a)?;
        write!(f, "}}")
    }
}

impl<A: Clone + Eq + Hash + fmt::Debug> DuelGraph<A> {
    fn get_special_node(&self, f: impl Fn(usize, usize) -> (usize, usize)) -> Option<A> {
        let mut n: Option<A> = None;
        for i in 0..self.v.len() {
            if (0..self.v.len()).all(|j| !self.a[f(i, j)]) {
                match n {
                    Some(_) => return None,
                    None => n = Some(self.v[i].clone()),
                }
            }
        }
        n
    }

    /// Returns the source of the graph if it is unique, `None` otherwise.
    pub fn get_source(&self) -> Option<A> {
        self.get_special_node(|i, j| (j, i))
    }

    /// Returns the sink of the graph if it is unique, `None` otherwise.
    pub fn get_sink(&self) -> Option<A> {
        self.get_special_node(|i, j| (i, j))
    }

    fn adjacency_to_matrix(a: &Adjacency) -> Matrix {
        let (n, nn) = a.shape();
        assert_eq!(n, nn);
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..i {
                if a[(i, j)] {
                    m[(i, j)] = 1f64;
                    m[(j, i)] = -1f64;
                } else if a[(j, i)] {
                    m[(j, i)] = 1f64;
                    m[(i, j)] = -1f64;
                }
            }
        }
        m
    }

    fn compute_strategy(
        &self,
        m: &Matrix,
        bval: f64,
        cval: f64,
    ) -> Result<Strategy<A>, simplex::SimplexError> {
        let n = self.v.len();
        let b = Vector::from_element(n, bval);
        let c = Vector::from_element(n, cval);
        let x = simplex::simplex(m, &c, &b)?;
        let mut mixed_data: Vec<(A, f64)> = self
            .v
            .iter()
            .cloned()
            .zip(x.into_iter().map(|&x| x))
            .collect();
        mixed_data.sort_unstable_by(|(_, p), (_, q)| p.partial_cmp(&q).unwrap());
        let sum: f64 = mixed_data.iter().map(|(_, p)| p).sum();
        for (_, p) in mixed_data.iter_mut() {
            *p /= sum;
        }
        Ok(Strategy::Mixed(mixed_data))
    }

    /// Returns the minimax strategy of the duel graph.
    ///
    /// # Errors
    ///
    /// If the simplex algorithm fails, returns an error describing the reason
    /// why.
    pub fn get_minimax_strategy(&self) -> Result<Strategy<A>, simplex::SimplexError> {
        let mut m = Self::adjacency_to_matrix(&self.a);
        m.iter_mut().for_each(|e| *e += 2f64);
        self.compute_strategy(&m, 1f64, -1f64)
    }

    /// Returns the maximin strategy of the duel graph.
    ///
    /// # Errors
    ///
    /// If the simplex algorithm fails, returns an error describing the reason
    /// why.
    pub fn get_maximin_strategy(&self) -> Result<Strategy<A>, simplex::SimplexError> {
        let mut m = Self::adjacency_to_matrix(&self.a).transpose();
        m.iter_mut().for_each(|e| *e = -(*e + 2f64));
        self.compute_strategy(&m, -1f64, 1f64)
    }

    /// Returns an optimal strategy for the duel graph.
    /// * If the graph has a source, returns a pure strategy electing it.
    /// * If the simplex algorithm manages to compute both the minimax and
    /// maximin strategies, floating-point operations might cause one to score
    /// slightly higher. Returns the higher-scoring one.
    /// * If the simplex algorithm only manages to compute one of minimax and
    /// maximin, returns said strategy.
    ///
    /// # Errors
    ///
    /// If the simplex algorithm fails to compute both strategies, returns an
    /// error giving both reasons.
    pub fn get_optimal_strategy(&self) -> Result<Strategy<A>, ElectionError> {
        match self.get_source() {
            Some(x) => Ok(Strategy::Pure(x)),
            None => match (self.get_minimax_strategy(), self.get_maximin_strategy()) {
                (Ok(minimax), Ok(maximin)) => {
                    Ok(match self.compare_strategies(&minimax, &maximin) {
                        Ordering::Less => maximin,
                        _ => minimax,
                    })
                }
                (Err(_), Ok(maximin)) => Ok(maximin),
                (Ok(minimax), Err(_)) => Ok(minimax),
                (Err(e), Err(f)) => Err(ElectionError::BothFailed(e, f)),
            },
        }
    }

    fn strategy_vector(&self, p: &Strategy<A>) -> Vector {
        match p {
            Strategy::Pure(x) => Vector::from_iterator(
                self.v.len(),
                self.v.iter().map(|e| if e == x { 1f64 } else { 0f64 }),
            ),
            Strategy::Mixed(u) => Vector::from_iterator(
                self.v.len(),
                self.v
                    .iter()
                    .map(|x| match u.iter().find(|(y, _)| *y == *x) {
                        None => panic!("Alternative not found"),
                        Some((_, p)) => p.clone(),
                    }),
            ),
        }
    }

    /// Returns a comparating number between two strategies `x` and `y`. If
    /// negative, then `x` performs worse than `y` for the graph `self`. If
    /// positive, then `x` performs better than `y` for the graph `self`.
    pub fn confront_strategies(&self, x: &Strategy<A>, y: &Strategy<A>) -> f64 {
        let m = Self::adjacency_to_matrix(&self.a);
        let p = self.strategy_vector(x);
        let q = self.strategy_vector(y);
        (p.transpose() * m * q)[(0, 0)]
    }

    // NOTE: This is numerically unstable
    /// Compares two strategies for the given graph to determine which one
    /// scores the better.
    ///
    /// Floating-point operations can make this method unsuitable for some
    /// uses. Consider using `confront_strategies()` with an epsilon instead.
    pub fn compare_strategies(&self, x: &Strategy<A>, y: &Strategy<A>) -> std::cmp::Ordering {
        self.confront_strategies(x, y).partial_cmp(&0f64).unwrap()
    }
}

/// Implements an election using the Randomized Condorcet Voting System.
#[derive(Clone)]
pub struct Election<A: Clone + Eq + Hash> {
    alternatives: HashSet<A>,
    duels: HashMap<Arrow<A>, u64>,
    open: bool,
}

impl<A: Clone + Eq + Hash + fmt::Debug> Election<A> {
    /// Creates a new empty election.
    pub fn new() -> Election<A> {
        Election::<A> {
            alternatives: HashSet::new(),
            duels: HashMap::new(),
            open: true,
        }
    }

    fn get(&self, x: &A, y: &A) -> Option<u64> {
        self.duels
            .get(&Arrow::<A>(x.to_owned(), y.to_owned()))
            .cloned()
    }

    /// Closes the election, preventing the casting of ballots.
    pub fn close(&mut self) {
        self.open = false;
    }

    /// Attemps to cast a ballot. Returns `true` if the casting was successful
    /// and `false` if it was not (which only happens if the election is
    /// closed).
    ///
    /// Casting an alternative that is not in the set of the alternatives of
    /// the election will add it to the set; if the electors are not supposed
    /// to be able to add their own alternatives, enforcing this rule is at the
    /// responsibility of the programmer using the structure.
    pub fn cast(&mut self, ballot: Ballot<A>) -> bool {
        if !self.open {
            return false;
        }
        for x in ballot.iter() {
            let (a, r) = x;
            self.alternatives.insert(a.to_owned());
            for y in ballot.iter() {
                let (b, s) = y;
                self.alternatives.insert(b.to_owned());
                if r > s {
                    let n = self.get(a, b).unwrap_or(0) + 1;
                    self.duels.insert(Arrow::<A>(a.to_owned(), b.to_owned()), n);
                }
            }
        }
        true
    }

    /// Attempts to agregate an election `sub` into the main election `self`,
    /// merging their lists of alternatives and duels. Returns `true` if the
    /// merging was possible, or `false` if it failed.
    ///
    /// Agregating `sub` into `self` requires `sub` to be closed and `self` to
    /// be open.
    pub fn agregate(&mut self, sub: Election<A>) -> bool {
        if !self.open || sub.open {
            return false;
        }
        for x in sub.alternatives.into_iter() {
            self.alternatives.insert(x);
        }
        for (Arrow::<A>(x, y), m) in sub.duels.into_iter() {
            let n = m + self.get(&x, &y).unwrap_or(0);
            self.duels.insert(Arrow::<A>(x, y), n);
        }
        true
    }

    /// Attempts to normalize an election. If the election is still open, this
    /// method does nothing. Normalizing means setting the election's internal
    /// state so that it reflects what the duel graph would be. In other
    /// words, if the election counted that `a` electors prefer `A` over `B`
    /// and `b` electors prefer `B` over `A`, then:
    /// * if `a > b`, then it will be as if it only counted one elector
    /// prefering `A` over `B`;
    /// * if `b > a`, then it will be as if it only counted one elector
    /// prefering `B` over `A`;
    /// * if `a == b`, then it will be as if no elector ever compared `A` to
    /// `B`.
    ///
    /// Since this method requires the election to be closed, it cannot be
    /// used to mess with a direct election. This method is intended to be used
    /// with `agregate()` to carry out elections working like the American
    /// Electoral College.
    ///
    /// Normalizing an election before computing its duel graph is not
    /// necessary.
    ///
    /// # Example
    ///
    /// ```
    /// # use rcvs::Election;
    /// let mut sub_a = Election::new();
    /// // -- carry out election sub_a --
    /// # sub_a.add_alternative(&"Alpha");
    /// sub_a.close();
    ///
    /// let mut sub_b = Election::new();
    /// // -- carry out election sub_b --
    /// # sub_b.add_alternative(&"Alpha");
    /// sub_b.close();
    ///
    /// /*
    ///  * normalize both elections so that the main election treats them
    ///  * equally
    ///  */
    /// sub_a.normalize();
    /// sub_b.normalize();
    ///
    /// // agregate both elections into a main election
    /// let mut e = Election::new();
    /// e.agregate(sub_a);
    /// e.agregate(sub_b);
    /// e.close();
    /// ```
    pub fn normalize(&mut self) {
        if self.open {
            return;
        }
        for x in self.alternatives.iter() {
            for y in self.alternatives.iter() {
                let xy = Arrow::<A>(x.clone(), y.clone());
                let yx = Arrow::<A>(y.clone(), x.clone());
                // Dirty workaround for the fact `if let` borrows self.duels
                let m;
                if let Some(k) = self.duels.get(&xy) {
                    m = k.clone();
                } else {
                    continue;
                }
                let n;
                if let Some(k) = self.duels.get(&yx) {
                    n = k.clone();
                } else {
                    continue;
                }
                match m.cmp(&n) {
                    Ordering::Less => {
                        self.duels.remove(&xy);
                        self.duels.insert(yx, 1);
                    }
                    Ordering::Equal => {
                        self.duels.remove(&xy);
                        self.duels.remove(&yx);
                    }
                    Ordering::Greater => {
                        self.duels.insert(xy, 1);
                        self.duels.remove(&yx);
                    }
                }
            }
        }
    }

    /// Adds an alternative to the set of alternatives without casting any
    /// vote. Returns `true` if the addition was successful, and `false` if the
    /// election is closed or if the alternative was already present.
    pub fn add_alternative(&mut self, v: &A) -> bool {
        if !self.open {
            return false;
        }
        self.alternatives.insert(v.to_owned())
    }

    /// Returns the duel graph of an election. A duel graph may be computed
    /// before the election is closed, giving information on a partial result
    /// of the election.
    pub fn get_duel_graph(&self) -> DuelGraph<A> {
        let v: Vec<A> = self.alternatives.iter().cloned().collect();
        let n = v.len();
        let mut a = Adjacency::from_element(n, n, false);
        for (i, x) in v.iter().enumerate() {
            for (j, y) in v.iter().enumerate() {
                match (self.get(x, y), self.get(y, x)) {
                    (Some(m), Some(n)) if m > n => a[(i, j)] = true,
                    (Some(_), None) => a[(i, j)] = true,
                    _ => (),
                }
            }
        }
        DuelGraph { v: v, a: a }
    }

    /// Decides if `x` is already in the set of alternatives known to the
    /// election. For an alternative to be there, at least one ballot involving
    /// it must have been cast, or it must have been manually added with the
    /// method `add_alternative()`.
    pub fn has_alternative(&self, x: &A) -> bool {
        self.alternatives.contains(x)
    }

    /// Returns the Condorcet winner of the election if it exists, `None`
    /// otherwise.
    ///
    /// Internally, this method computes the duel graph of the election.
    /// Instead of calling several methods that do it in the same scope,
    /// consider computing the duel graph separately and operating on it.
    pub fn get_condorcet_winner(&self) -> Option<A> {
        self.get_duel_graph().get_source()
    }

    /// Returns the Condorcet loser of the election if it exists, `None`
    /// otherwise.
    ///
    /// Internally, this method computes the duel graph of the election.
    /// Instead of calling several methods that do it in the same scope,
    /// consider computing the duel graph separately and operating on it.
    pub fn get_condorcet_loser(&self) -> Option<A> {
        self.get_duel_graph().get_sink()
    }

    /// Returns the minimax strategy of the election.
    ///
    /// Internally, this method computes the duel graph of the election.
    /// Instead of calling several methods that do it in the same scope,
    /// consider computing the duel graph separately and operating on it.
    ///
    /// # Errors
    ///
    /// If the simplex algorithm fails to compute the strategy, an error
    /// describing the reason why is returned.
    pub fn get_minimax_strategy(&self) -> Result<Strategy<A>, simplex::SimplexError> {
        self.get_duel_graph().get_minimax_strategy()
    }

    /// Returns the maximin strategy of the election.
    ///
    /// Internally, this method computes the duel graph of the election.
    /// Instead of calling several methods that do it in the same scope,
    /// consider computing the duel graph separately and operating on it.
    ///
    /// # Errors
    ///
    /// If the simplex algorithm fails to compute the strategy, an error
    /// describing the reason why is returned.
    pub fn get_maximin_strategy(&self) -> Result<Strategy<A>, simplex::SimplexError> {
        self.get_duel_graph().get_maximin_strategy()
    }

    /// Returns the optimal strategy of the election.
    ///
    /// Internally, this method computes the duel graph of the election.
    /// Instead of calling several methods that do it in the same scope,
    /// consider computing the duel graph separately and operating on it.
    ///
    /// # Errors
    ///
    /// If the election has no Condorcet winner and the simplex algorithm fails
    /// to compute both the minimax and maximin strategies, an error describing
    /// both failures is returned.
    pub fn get_optimal_strategy(&self) -> Result<Strategy<A>, ElectionError> {
        self.get_duel_graph().get_optimal_strategy()
    }

    /// Elects the winner of the election using the optimal strategy.
    ///
    /// Internally, this method computes the duel graph of the election.
    /// Instead of calling several methods that do it in the same scope,
    /// consider computing the duel graph separately and operating on it.
    ///
    /// # Errors
    ///
    /// If the election has no Condorcet winner and the simplex algorithm fails
    /// to compute both the minimax and maximin strategies, an error describing
    /// both failures is returned.
    pub fn get_randomized_winner(
        &self,
        rng: &mut impl rand::Rng,
    ) -> Result<Option<A>, ElectionError> {
        Ok(self.get_optimal_strategy()?.play(rng))
    }
}

impl<A: Clone + Eq + Hash + fmt::Display> fmt::Display for Election<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Election {{")?;
        for x in self.duels.iter() {
            let (Arrow::<A>(a, b), n) = x;
            writeln!(f, "    {} beats {} {} times", a, b, n)?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_graph(names: &[String]) -> DuelGraph<String> {
        let n = rand::random::<usize>() % names.len() + 1;
        let v = names.iter().take(n).cloned().collect();
        let mut a = Adjacency::from_element(n, n, false);
        for i in 1..n {
            for j in 0..i {
                if rand::random::<f64>() < 0.5f64 {
                    a[(i, j)] = true;
                } else if rand::random::<f64>() < 0.5f64 {
                    a[(j, i)] = true;
                }
            }
        }
        DuelGraph { v: v, a: a }
    }

    #[test]
    fn source_strategy() {
        let names = string_vec!["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"];
        for n in 1..=names.len() {
            for _ in 0..100 {
                let mut m = Adjacency::from_element(n, n, false);
                (0..n).for_each(|i| {
                    (0..i).for_each(|j| {
                        if rand::random::<f64>() < 0.5f64 {
                            m[(i, j)] = true;
                        } else {
                            m[(j, i)] = true;
                        }
                    })
                });
                let s = rand::random::<usize>() % n;
                (0..n).filter(|i| *i != s).for_each(|i| {
                    m[(s, i)] = true;
                    m[(i, s)] = false;
                });
                let g = DuelGraph {
                    v: names.iter().cloned().take(n).collect(),
                    a: m,
                };
                let w;
                match g.get_source() {
                    Some(x) => w = x,
                    None => panic!("No source in graph {}", g),
                }
                assert!(
                    g.get_minimax_strategy()
                        .unwrap()
                        .almost_chooses(&w.to_string(), 1e-6),
                    "Minimax doesn't choose {}",
                    w
                );
                assert!(
                    g.get_maximin_strategy()
                        .unwrap()
                        .almost_chooses(&w.to_string(), 1e-6),
                    "Minimax doesn't choose {}",
                    w
                );
                assert!(
                    g.get_optimal_strategy().unwrap().is_pure(),
                    "Optimal strategy is mixed"
                );
            }
        }
    }

    #[test]
    fn condorcet_paradox() {
        let mut e = Election::<String>::new();
        let mut b = vec![
            Ballot::<String>::new(),
            Ballot::<String>::new(),
            Ballot::<String>::new(),
        ];
        let names = string_vec!["Alpha", "Bravo", "Charlie"];
        for (i, b) in b.iter_mut().enumerate() {
            for j in 0u64..3u64 {
                assert!(
                    b.insert(names[(i + (j as usize)) % 3].to_owned(), j, j),
                    "add_entry failed"
                );
            }
        }
        for b in b.iter().cloned() {
            e.cast(b);
        }
        let g = e.get_duel_graph();
        assert_eq!(g.get_source(), None);
        assert_eq!(g.get_sink(), None);
        assert!(
            g.get_optimal_strategy().unwrap().is_uniform(&names, 1e-6),
            "Non uniform strategy for Condorcet paradox"
        );
    }

    // Last name commented out for convenience (doubles testing time)
    #[test]
    fn tournament() {
        let names = string_vec!["Alpha", "Bravo", "Charlie", "Delta", "Echo" /*, "Foxtrot"*/];
        for n in 1..=names.len() {
            println!("Size {}", n);
            let v: Vec<String> = names.iter().take(n).cloned().collect();
            let mut a = Adjacency::from_element(n, n, false);
            (0..(n - 1)).for_each(|i| ((i + 1)..n).for_each(|j| a[(i, j)] = true));
            loop {
                // Test graph
                let g = DuelGraph::<String> {
                    v: v.clone(),
                    a: a.clone(),
                };
                match (g.get_minimax_strategy(), g.get_maximin_strategy()) {
                    (Ok(minimax), Ok(maximin)) => {
                        for _ in 0..100 {
                            let p = Strategy::random_mixed(&v, &mut rand::thread_rng());
                            let vminimax = g.confront_strategies(&minimax, &p);
                            let vmaximin = g.confront_strategies(&maximin, &p);
                            if vminimax < -1e-6 && vmaximin < -1e-6 {
                                panic!(
                                    "{:?} beats both:\n * minimax by {}\n{:?}\n * maximin by {}\n{:?}",
                                    p,
                                    vminimax,
                                    minimax,
                                    vmaximin,
                                    maximin
                                );
                            }
                        }
                    }
                    (Err(e), Ok(maximin)) => {
                        println!("{}\nMinimax failed: {}", g, e);
                        for _ in 0..100 {
                            let p = Strategy::random_mixed(&v, &mut rand::thread_rng());
                            let v = g.confront_strategies(&maximin, &p);
                            if v < -1e-6 {
                                panic!("{:?} beats maximin by {}\n{:?}", p, v, maximin);
                            }
                        }
                    }
                    (Ok(minimax), Err(e)) => {
                        println!("{}\nMaximin failed: {}", g, e);
                        for _ in 0..100 {
                            let p = Strategy::random_mixed(&v, &mut rand::thread_rng());
                            let v = g.confront_strategies(&minimax, &p);
                            if v < -1e-6 {
                                panic!("{:?} beats minimax by {}\n{:?}", p, v, minimax);
                            }
                        }
                    }
                    (Err(e), Err(f)) => {
                        panic!("{}\nBoth failed:\n * minimax: {}\n * maximin: {}", g, e, f)
                    }
                };
                // Next graph
                let mut carry = true;
                for i in 1..n {
                    for j in 0..i {
                        if !carry {
                            break;
                        }
                        if a[(i, j)] {
                            a[(i, j)] = false;
                            a[(j, i)] = true;
                        } else {
                            a[(i, j)] = true;
                            a[(j, i)] = false;
                            carry = false;
                        }
                    }
                }
                // Stop test
                if (1..n).all(|i| (0..i).all(|j| !a[(i, j)])) {
                    break;
                }
            }
        }
    }

    /*
     * NOTE:
     * Wasn't observed to fail anymore after fixing simplex; keep an eye on it
     * anyway...
     */
    #[test]
    fn optimal_strategy() {
        let names = string_vec!["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"];
        for _pass in 0..1000 {
            println!("Pass {}", _pass);
            let g = random_graph(&names);
            println!("{}", g);
            match (g.get_minimax_strategy(), g.get_maximin_strategy()) {
                (Ok(minimax), Ok(maximin)) => {
                    let opt = g.get_optimal_strategy().unwrap();
                    assert!(
                        g.confront_strategies(&opt, &minimax) > -1e-6,
                        "Minimax beats optimal strategy"
                    );
                    assert!(
                        g.confront_strategies(&opt, &maximin) > -1e-6,
                        "Maximin beats optimal strategy"
                    );
                }
                (Ok(minimax), Err(e)) => {
                    println!("Maximin failed: {}", e);
                    let opt = g.get_optimal_strategy().unwrap();
                    assert!(
                        g.confront_strategies(&opt, &minimax) > -1e-6,
                        "Minimax beats optimal strategy"
                    );
                }
                (Err(e), Ok(maximin)) => {
                    println!("Minimax failed: {}", e);
                    let opt = g.get_optimal_strategy().unwrap();
                    assert!(
                        g.confront_strategies(&opt, &maximin) > -1e-6,
                        "Maximin beats optimal strategy"
                    );
                }
                (Err(e), Err(f)) => panic!("Both failed: {}\n{}", e, f),
            }
        }
    }

    fn random_ballot(v: &[String]) -> Ballot<String> {
        let mut b = Ballot::<String>::new();
        for x in v.iter() {
            let s = rand::random::<u64>();
            let r = rand::random::<u64>() % (s + 1);
            assert!(
                b.insert(x.to_string(), r, s),
                "Insert ({}, {}) failed",
                r,
                s
            );
        }
        b
    }

    #[test]
    fn agregate() {
        let names = string_vec!["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"];
        for _ in 0..50 {
            let mut e = Election::<String>::new();
            let mut sum = Election::<String>::new();
            let num_district = rand::random::<u64>() % 49 + 2;
            for _ in 0..num_district {
                let mut f = Election::<String>::new();
                let num_ballot = rand::random::<u64>() % 100;
                for _ in 0..num_ballot {
                    let b = random_ballot(&names);
                    e.cast(b.clone());
                    f.cast(b);
                }
                f.close();
                sum.agregate(f);
            }
            // e and sum must be identical
            assert_eq!(e.alternatives, sum.alternatives, "Alternative lists differ");
            for (a, n) in e.duels.into_iter() {
                match sum.duels.get(&a) {
                    Some(m) => assert_eq!(*m, n, "{:?} is {} in e but {} in sum", a, n, m),
                    None => panic!("{:?} isn't in sum", a),
                }
            }
        }
    }

    #[test]
    fn normalize() {
        let names = string_vec!["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"];
        for _pass in 0..100 {
            let mut e = Election::<String>::new();
            for _ in 0..500 {
                e.cast(random_ballot(&names));
            }
            let mut n = e.clone();
            n.close();
            n.normalize();
            for x in n.alternatives.iter() {
                let xx = Arrow::<String>(x.to_string(), x.to_string());
                assert_eq!(n.duels.get(&xx), None, "{} wins over itself", x);
                for y in n.alternatives.iter() {
                    let xy = Arrow::<String>(x.to_string(), y.to_string());
                    let yx = Arrow::<String>(y.to_string(), x.to_string());
                    if let Some(m) = n.duels.get(&xy) {
                        assert_eq!(n.duels.get(&yx), None, "{} and {} loop", x, y);
                        assert_eq!(*m, 1, "Normalized election has {}", m);
                        if let Some(n) = e.duels.get(&yx) {
                            assert!(e.duels.get(&xy).unwrap() > n, "Backward normalization");
                        }
                    }
                }
            }
        }
    }
}
