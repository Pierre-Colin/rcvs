use std::{
    collections::HashSet,
    fmt,
    hash::Hash,
};

use util::quick_sort;

#[derive(Clone, Debug)]
pub enum Strategy<A: Clone + Eq + Hash> {
    Pure(A),
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
    pub fn random_mixed(v: &[A]) -> Self {
        let mut u: Vec<(A, f64)> =
            v.iter().map(|x| (x.to_owned(), rand::random::<f64>())).collect();
        let sum: f64 = quick_sort(
            u.clone(),
            |(_, p), (_, q)| p.partial_cmp(&q).unwrap()
        ).into_iter().map(|(_, p)| p).sum();
        u.iter_mut().for_each(|(_, p)| *p /= sum);
        Strategy::Mixed(u)
    }

    pub fn play(&self) -> Option<A> {
        match self {
            Strategy::Pure(x) => Some(x.to_owned()),
            Strategy::Mixed(v) => {
                if v.is_empty() {
                    None
                } else {
                    let sorted = quick_sort(
                        v.to_vec(),
                        |(_, x), (_, y)| x.partial_cmp(&y).unwrap()
                    );
                    let r = rand::random::<f64>();
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

    pub fn is_pure(&self) -> bool {
        if let Strategy::Pure(_) = self { true } else { false }
    }

    pub fn is_mixed(&self) -> bool {
        !self.is_pure()
    }

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
        quick_sort(diff, |x, y| x.partial_cmp(&y).unwrap()).into_iter().sum()
    }

    pub fn almost_chooses(&self, x: &A, epsilon: f64) -> bool {
        self.distance(&Strategy::Pure(x.to_owned())) < epsilon
    }

    pub fn support(&self) -> Vec<A> {
        match self {
            Strategy::Pure(x) => vec![x.to_owned()],
            Strategy::Mixed(v) => v.iter().map(|(x, _)| x.to_owned()).collect(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Strategy::Pure(_) => 1usize,
            Strategy::Mixed(v) => v.len(),
        }
    }

    pub fn is_uniform(&self, v: &[A], epsilon: f64) -> bool {
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

