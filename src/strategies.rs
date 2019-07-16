//! # Strategies
//!
//! A strategy is a rule on how to pick the winner of an election. It can be
//! either _pure_ or _mixed_.
//!
//! ## Pure strategies
//!
//! A pure strategy deterministically always elect the same alternative. It is
//! intended to be used in the case where a Condorcet winner exists in the
//! election.
//!
//! ## Mixed strategies
//!
//! A mixed strategy randomly elects an alternative according to a probability
//! distribution that is not necessarily uniform and is intended to be
//! computed in a way that minimizes the number of electors that end up
//! wishing another alternative was chosen.
use std::{
    collections::HashSet,
    fmt,
    hash::Hash,
};

use util::quick_sort;

/// Implements a strategy that may be either pure or mixed.
#[derive(Clone, Debug)]
pub enum Strategy<A: Clone + Eq + Hash> {
    /// Pure strategy, always picking the same winner.
    Pure(A),

    /// Mixed strategy, picking the winner randomly. The vector it contains
    /// bears tuples which associate a probability to each alternative.
    /// `std::collections::HashMap` was not used because some algorithms
    /// require sorting the vector. In general, no guarantee is made as to how
    /// the elements of the vector are ordered.
    Mixed(Vec<(A, f64)>),
}

impl <A: Clone + Eq + Hash + fmt::Display> fmt::Display for Strategy<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Strategy::Pure(x) => write!(f, "Pure {}", x),
            Strategy::Mixed(v) => {
                writeln!(f, "Mixed {{")?;
                for (x, p) in v.iter() {
                    writeln!(f, "    {}% {}", 100f64 * p, x)?;
                }
                write!(f, "}}")
            },
        }
    }
}

impl <A: Clone + Eq + Hash> Strategy<A> {
    /// Generates a random mixed strategy over a set of alternatives. This
    /// function is intended for testing the optimality of the minimax and
    /// maximin strategies.
    pub fn random_mixed(v: &[A], rng: &mut impl rand::Rng) -> Self {
        let mut u: Vec<(A, f64)> =
            v.iter().map(|x| (x.to_owned(), rng.gen::<f64>())).collect();
        let sum: f64 = quick_sort(
            u.clone(),
            |(_, p), (_, q)| p.partial_cmp(&q).unwrap(),
            &mut rand::thread_rng()
        ).into_iter().map(|(_, p)| p).sum();
        u.iter_mut().for_each(|(_, p)| *p /= sum);
        Strategy::Mixed(u)
    }

    /// Plays a strategy to elect the winner. On a pure strategy, will
    /// deterministically elect the one described. On a mixed strategy, will
    /// follow the described probability distribution to randomly elect the
    /// winner. Returns `None` if `self` is an empty mixed strategy.
    pub fn play(&self, rng: &mut impl rand::Rng) -> Option<A> {
        match self {
            Strategy::Pure(x) => Some(x.to_owned()),
            Strategy::Mixed(v) => {
                if v.is_empty() {
                    None
                } else {
                    let sorted = quick_sort(
                        v.to_vec(),
                        |(_, x), (_, y)| x.partial_cmp(&y).unwrap(),
                        &mut rand::thread_rng()
                    );
                    let r = rng.gen::<f64>();
                    let mut acc = 0f64;
                    for (x, p) in sorted.iter() {
                        acc += p;
                        if acc > r { return Some(x.to_owned()); }
                    }
                    Some(sorted.iter().last().unwrap().0.to_owned())
                }
            },
        }
    }

    /// Returns `true` if `self` is a pure strategy, and `false` if it is
    /// mixed.
    ///
    /// # Example
    ///
    /// ```
    /// let mut e = Election::new();
    /// let mut b = Ballot::new();
    /// b.insert("Rock", 1, 1);
    /// b.insert("Paper", 0, 0);
    /// b.insert("Scissors", 0, 0);
    /// e.cast(b);
    ///
    /// assert!(e.get_optimal_strategy(&mut rand::thread_rng()).is_pure());
    /// ```
    pub fn is_pure(&self) -> bool {
        if let Strategy::Pure(_) = self { true } else { false }
    }

    /// Returns `true` if `self` is a mixed strategy, and `false` if it is
    /// pure. `p.is_mixed()` is equivalent to `!p.is_pure()`.
    pub fn is_mixed(&self) -> bool {
        !self.is_pure()
    }

    /// Converts a pure strategy into a mixed strategy which always elects the
    /// candidate described.
    ///
    /// Converting a pure strategy to a mixed one will slow down the `play()`
    /// method.
    pub fn to_mixed(&self) -> Strategy<A> {
        match self {
            Strategy::Pure(x) => Strategy::Mixed(vec![(x.to_owned(), 1f64)]),
            _ => self.to_owned(),
        }
    }

    fn unwrap_mixed(&self) -> Vec<(A, f64)> {
        match self {
            Strategy::Pure(_) => panic!("Strategy is not mixed"),
            Strategy::Mixed(v) => v.to_owned(),
        }
    }

    /// Returns the Manhattan distance between two strategies.
    pub fn distance(&self, other: &Strategy<A>) -> f64 {
        let sm = self.to_mixed().unwrap_mixed();
        let om = other.to_mixed().unwrap_mixed();
        // Alternative lists may differ; find the union
        let mut alternatives = HashSet::new();
        for (x, _) in sm.iter().chain(om.iter()) {
            alternatives.insert(x);
        }
        let mut diff = Vec::new();
        for x in alternatives.into_iter() {
            match (sm.iter().find(|(y, _)| y == x),
                   om.iter().find(|(y, _)| y == x))
            {
                (Some((_, p)), Some((_, q))) => diff.push((p - q).abs()),
                (Some((_, p)), None) => diff.push(*p),
                (None, Some((_, p))) => diff.push(*p),
                (None, None) => (),
            }
        }
        // Sum in increasing order for better numerical stability
        quick_sort(
            diff,
            |x, y| x.partial_cmp(&y).unwrap(),
            &mut rand::thread_rng()
        ).into_iter().sum()
    }

    /// Decides if, up to a chosen `epsilon`, a strategy always elects a given
    /// alternative.
    ///
    /// # Panics
    ///
    /// This method asserts that `epsilon > 0f64`.
    pub fn almost_chooses(&self, x: &A, epsilon: f64) -> bool {
        assert!(epsilon > 0f64);
        self.distance(&Strategy::Pure(x.to_owned())) < epsilon
    }

    /// Returns a vector of all the alternatives the strategy contains. For
    /// mixed strategies, alternatives with probability 0 to be elected will
    /// still be included.
    pub fn support(&self) -> Vec<A> {
        match self {
            Strategy::Pure(x) => vec![x.to_owned()],
            Strategy::Mixed(v) => v.iter().map(|(x, _)| x.to_owned()).collect(),
        }
    }

    /// Returns the number of alternatives the strategy involves. Always `1`
    /// for pure strategies.
    pub fn len(&self) -> usize {
        match self {
            Strategy::Pure(_) => 1usize,
            Strategy::Mixed(v) => v.len(),
        }
    }

    /// Decides if, up to a chosen `epsilon`, a strategy is uniform. Returns
    /// `false` if `v` differs, even merely in order, to the alternatives in
    /// `self`'s description. For pure strategies, simply checks if `self`
    /// elects `v`'s first element.
    ///
    /// # Panics
    ///
    /// This method asserts that `epsilon > 0f64`.
    pub fn is_uniform(&self, v: &[A], epsilon: f64) -> bool {
        assert!(epsilon > 0f64);
        match self {
            Strategy::Pure(x) => v.get(0) == Some(x),
            Strategy::Mixed(u) => {
                if u.len() != v.len() { return false; }
                for x in v.iter() {
                    if u.iter().find(|(y, _)| y == x) == None { return false; }
                }
                let p = 1f64 / u.len() as f64;
                let uni = Strategy::Mixed(
                    v.iter().map(|x| (x.to_owned(), p)).collect()
                );
                self.distance(&uni) < epsilon
            },
        }
    }
}

#[cfg(tests)]
mod tests {
    #[test]
    fn quickie() {
        let mut e = Election::new();
        let mut b = Ballot::new();
        b.insert("Rock", 2, 2);
        b.insert("Paper", 1, 1);
        b.insert("Scissors", 0, 0);
        e.cast(b);
        
        assert!(e.get_optimal_strategy(&mut rand::thread_rng()).is_pure());
    }
}
