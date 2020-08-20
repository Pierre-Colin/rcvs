//! # Ballots
//!
//! Theoretically, ballots in the RCVS may be any kind of oriented graph whose
//! vertices corresponds to alternatives. However, this would leave room for
//! cyclical preferences which are not desirable as they can be exploited for
//! easy scams as described in the following code. Instead, the ballot has to
//! be a _directed acyclic graph_, but this is still a bit too complicated for
//! practical use.
//!
//! The settled-upon ballot model is instead a _pseudo-ranking ballot_ system
//! where a range of ranks may be attributed to alternatives, and one
//! alternative is said to be preferred over another if its range is above the
//! other's with no overlapping. An alternative may have a single range or no
//! range at all, meaning it is unranked.
use std::{
    cmp,
    collections::{hash_map, HashMap},
    fmt,
    hash::Hash,
};

/// Implements a range of ranks
#[derive(Clone, Debug)]
pub struct Rank(u64, u64);

impl Rank {
    /// Constructs a range of ranks from two integers and returns it if it is
    /// valid, `None` otherwise.
    ///
    /// A range `[a, b]` is invalid if and only if `b > a`.
    pub fn new(a: u64, b: u64) -> Option<Rank> {
        if a <= b {
            Some(Rank(a, b))
        } else {
            None
        }
    }

    /// Returns the lower bound of the range.
    pub fn low(&self) -> u64 {
        self.0
    }

    /// Returns the upper bound of the range.
    pub fn high(&self) -> u64 {
        self.1
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Rank(a, b) = self;
        if a == b {
            write!(f, "{}", a)
        } else {
            write!(f, "[{}, {}]", a, b)
        }
    }
}

impl cmp::PartialEq for Rank {
    fn eq(&self, _: &Rank) -> bool {
        false
    }
}

impl cmp::PartialOrd for Rank {
    fn partial_cmp(&self, other: &Rank) -> Option<cmp::Ordering> {
        match (self, other) {
            (&Rank(_, b), &Rank(a, _)) if b < a => Some(cmp::Ordering::Less),
            (&Rank(a, _), &Rank(_, b)) if b < a => Some(cmp::Ordering::Greater),
            _ => None,
        }
    }
}

/// Implements a pre-order ballot to be cast into an election.
#[derive(Clone)]
pub struct Ballot<A: Hash> {
    m: HashMap<A, Rank>,
}

impl<A: Hash + Eq> Ballot<A> {
    /// Creates a new empty ballot.
    pub fn new() -> Ballot<A> {
        Ballot::<A> {
            m: HashMap::<A, Rank>::new(),
        }
    }

    /// Attempts to insert rank `(a, b)` into a ballot. Returns `true` if the
    /// rank is well-formed, and otherwise returns `false` without inserting
    /// it. This same method can be used to re-rank an alternative that is
    /// already in the ballot.
    pub fn insert(&mut self, x: A, a: u64, b: u64) -> bool {
        let or = Rank::new(a, b);
        match or {
            Some(r) => {
                self.m.insert(x, r);
                true
            }
            None => false,
        }
    }

    /// Removes an alternative from a ballot, making it unranked, and returns
    /// `true` if it was in the ballot and `false` otherwise.
    pub fn remove(&mut self, x: &A) -> bool {
        self.m.remove(x) != None
    }

    /// Returns a non-mutable hash map iterator. The keys are the alternatives,
    /// and the values are their ranks.
    pub fn iter(&self) -> hash_map::Iter<A, Rank> {
        self.m.iter()
    }
}

impl<'a, A: 'a + Hash> IntoIterator for Ballot<A> {
    type Item = (A, Rank);
    type IntoIter = hash_map::IntoIter<A, Rank>;

    /// Consumes the ballot and returns a non-mutable hash map iterator. The
    /// keys are the alternatives, and the values are their ranks.
    fn into_iter(self) -> Self::IntoIter {
        self.m.into_iter()
    }
}

impl<A: fmt::Display + Hash> fmt::Display for Ballot<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Ballot {{")?;
        for (a, r) in self.m.iter() {
            writeln!(f, "    {} ranks {}", a, r)?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate rand;
    use self::rand::{
        distributions::{Distribution, Standard},
        Rng,
    };

    impl Distribution<Rank> for Standard {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Rank {
            let b = rng.gen();
            Rank(rng.gen_range(0, b + 1), b)
        }
    }

    #[test]
    fn new() {
        for _ in 1..=10000 {
            let a = rand::random::<u64>();
            let b = rand::random::<u64>();
            match Rank::new(a, b) {
                Some(_) => assert!(a <= b, "a > b but rank was instanciated"),
                None => assert!(a > b, "a <= b but rank wasn't instanciated"),
            }
        }
    }

    #[test]
    fn rng() {
        for _ in 1..=10000 {
            let Rank(a, b) = rand::random::<Rank>();
            assert!(a <= b, "a > b");
        }
    }

    #[test]
    fn never_equals() {
        for _ in 1..=10000 {
            let x = rand::random::<Rank>();
            let y = rand::random::<Rank>();
            assert_ne!(x, x);
            assert_ne!(y, y);
            assert_ne!(x, y);
            assert_ne!(y, x);
        }
    }

    #[test]
    fn compare() {
        for _ in 1..=10000 {
            let x = rand::random::<Rank>();
            let y = rand::random::<Rank>();
            let Rank(ax, bx) = x;
            let Rank(ay, by) = y;
            if bx < ay {
                assert!(x < y);
            } else if by < ax {
                assert!(y < x);
            } else {
                assert!(!(x < y) && !(y < x));
            }
        }
    }
}
