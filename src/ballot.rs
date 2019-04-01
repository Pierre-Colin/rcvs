use std::{fmt, cmp};

#[derive(Clone, Debug)]
pub struct Rank(u64, u64);

impl Rank {
    pub fn new(a: u64, b: u64) -> Option<Rank> {
        if a <= b {
            Some(Rank(a, b))
        } else {
            None
        }
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Rank(a, b) = self;
        if a == b { write!(f, "{}", a) }
        else { write!(f, "[{}, {}]", a, b) }
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
                (&Rank(_, b), &Rank(a, _)) if b < a
                    => Some(cmp::Ordering::Less),
                (&Rank(a, _), &Rank(_, b)) if b < a
                    => Some(cmp::Ordering::Greater),
                _ => None,
            }
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

