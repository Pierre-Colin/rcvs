extern crate nalgebra as na;
extern crate rand;

mod ballot;
mod simplex;
pub mod util;

use std::{
    clone::Clone,
    cmp::{Eq, Ordering},
    collections::{HashMap, hash_map, HashSet},
    error::Error,
    fmt,
    hash::Hash,
};

use util::quick_sort;

type Adjacency = na::DMatrix<bool>;
type Matrix = na::DMatrix<f64>;
type Vector = na::DVector<f64>;

#[derive(Clone)]
pub struct Ballot<A: Hash> {
    m: HashMap<A, ballot::Rank>
}

impl <A: Hash + Eq> Ballot<A> {
    pub fn new() -> Ballot<A> {
        Ballot::<A>{ m: HashMap::<A, ballot::Rank>::new() }
    }

    pub fn insert(&mut self, x: A, a: u64, b: u64) -> bool {
        let or = ballot::Rank::new(a, b);
        match or {
            Some(r) => { self.m.insert(x, r); true },
            None => false,
        }
    }

    pub fn remove(&mut self, x: &A) -> bool {
        self.m.remove(x) != None
    }

    pub fn iter(&self) -> hash_map::Iter<A, ballot::Rank> {
        self.m.iter()
    }
}

impl <A: fmt::Display + Hash> fmt::Display for Ballot<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Ballot {{")?;
        for (a, r) in self.m.iter() {
            writeln!(f ,"    {} ranks {}", a, r)?;
        }
        write!(f, "}}")
    }
}

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
                    let sorted = util::quick_sort(
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

#[derive(Clone, Debug, Hash)]
struct Arrow<A>(A, A);

impl <A: Eq> PartialEq for Arrow<A> {
    fn eq(&self, other: &Arrow<A>) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}

impl <A: Eq> Eq for Arrow<A> {}

pub struct DuelGraph<A: fmt::Debug> {
    v: Vec<A>,
    a: Adjacency,
}

#[derive(Debug)]
pub enum ElectionError {
    BothFailed(simplex::SimplexError, simplex::SimplexError),
    SimplexFailed(simplex::SimplexError),
    ElectionClosed,
    ElectionOpen,
}

impl fmt::Display for ElectionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ElectionError::BothFailed(a, b) => {
                writeln!(f, "Both methods failed:")?;
                writeln!(f, " * minimax: {}", a)?;
                writeln!(f, " * maximin: {}", b)
            },
            ElectionError::SimplexFailed(e) =>
                write!(f, "Simplex algorithm failed: {}", e),
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
            ElectionError::BothFailed(_, _) =>
                "Both minimax and maximin strategies failed to be solved",
            ElectionError::SimplexFailed(_) =>
                "The simplex algorithm failed to compute the strategy",
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

impl <A: fmt::Debug> fmt::Display for DuelGraph<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Graph {{")?;
        writeln!(f, "Alternatives: {:?}", self.v)?;
        writeln!(f, "{}", self.a)?;
        write!(f, "}}")
    }
}

impl <A: Clone + Eq + Hash + fmt::Debug> DuelGraph<A> {
    fn get_special_node(&self, f: impl Fn(usize, usize) -> (usize, usize))
        -> Option<A>
    {
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

    pub fn get_source(&self) -> Option<A> {
        self.get_special_node(|i, j| (j, i))
    }

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

    fn compute_strategy(&self, m: &Matrix, bval: f64, cval: f64)
        -> Result<Strategy<A>, simplex::SimplexError>
    {
        let n = self.v.len();
        let b = Vector::from_element(n, bval);
        let c = Vector::from_element(n, cval);
        let x = simplex::simplex(m, &c, &b)?;
        let p = simplex::vector_to_lottery(x);
        Ok(Strategy::Mixed(self.v.iter().cloned().zip(p.into_iter()).collect()))
    }

    pub fn get_minimax_strategy(&self)
        -> Result<Strategy<A>, simplex::SimplexError>
    {
        let mut m = Self::adjacency_to_matrix(&self.a);
        m.iter_mut().for_each(|e| *e += 2f64);
        self.compute_strategy(&m, 1f64, -1f64)
    }

    pub fn get_maximin_strategy(&self)
        -> Result<Strategy<A>, simplex::SimplexError>
    {
        let mut m = Self::adjacency_to_matrix(&self.a).transpose();
        m.iter_mut().for_each(|e| *e = -(*e + 2f64));
        self.compute_strategy(&m, -1f64, 1f64)
    }

    pub fn get_optimal_strategy(&self) -> Result<Strategy<A>, ElectionError> {
        match self.get_source() {
            Some(x) => Ok(Strategy::Pure(x)),
            None => {
                match (self.get_minimax_strategy(), self.get_maximin_strategy())
                {
                    (Ok(minimax), Ok(maximin)) => {
                        Ok(match self.compare_strategies(&minimax, &maximin) {
                            Ordering::Less => maximin,
                            _ => minimax,
                        })
                    },
                    (Err(_), Ok(maximin)) => Ok(maximin),
                    (Ok(minimax), Err(_)) => Ok(minimax),
                    (Err(e), Err(f)) => Err(ElectionError::BothFailed(e, f)),
                }
            },
        }
    }

    fn strategy_vector(&self, p: &Strategy<A>) -> Vector {
        match p {
            Strategy::Pure(x) => Vector::from_iterator(
                self.v.len(),
                self.v.iter().map(|e| if e == x { 1f64 } else { 0f64 })
            ),
            Strategy::Mixed(u) => Vector::from_iterator(
                self.v.len(),
                self.v.iter().map(|x|
                    match u.iter().find(|(y, _)| *y == *x) {
                        None => panic!("Alternative not found"),
                        Some((_, p)) => p.clone(),
                    }
                )
            ),
        }
    }

    pub fn confront_strategies(&self, x: &Strategy<A>, y: &Strategy<A>) -> f64 {
        let m = Self::adjacency_to_matrix(&self.a);
        let p = self.strategy_vector(x);
        let q = self.strategy_vector(y);
        (p.transpose() * m * q)[(0, 0)]
    }

    // NOTE: This is numerically unstable
    pub fn compare_strategies(&self, x: &Strategy<A>, y: &Strategy<A>)
        -> std::cmp::Ordering
    {
        self.confront_strategies(x, y).partial_cmp(&0f64).unwrap()
    }
}

#[derive(Clone)]
pub struct Election<A: Clone + Eq + Hash> {
    alternatives: HashSet<A>,
    duels: HashMap<Arrow<A>, u64>,
    open: bool,
}

impl <A: Clone + Eq + Hash + fmt::Debug> Election<A> {
    pub fn new() -> Election<A> {
        Election::<A> {
            alternatives: HashSet::new(),
            duels: HashMap::new(),
            open: true,
        }
    }

    fn get(&self, x: &A, y: &A) -> Option<u64> {
        self.duels.get(&Arrow::<A>(x.to_owned(), y.to_owned())).cloned()
    }

    pub fn close(&mut self) {
        self.open = false;
    }

    pub fn cast(&mut self, ballot: Ballot<A>) -> bool {
        if !self.open { return false; }
        for x in ballot.iter() {
            let (a, r) = x;
            self.alternatives.insert(a.to_owned());
            for y in ballot.iter() {
                let (b, s) = y;
                self.alternatives.insert(b.to_owned());
                if r > s {
                    let n = self.get(a, b).unwrap_or(0) + 1;
                    self.duels.insert(Arrow::<A>(a.to_owned(), b.to_owned()),
                                      n);
                }
            }
        }
        true
    }

    pub fn agregate(&mut self, sub: Election<A>) -> bool {
        if !self.open || sub.open { return false; }
        for x in sub.alternatives.into_iter() {
            self.alternatives.insert(x);
        }
        for (Arrow::<A>(x, y), m) in sub.duels.into_iter() {
            let n = m + self.get(&x, &y).unwrap_or(0);
            self.duels.insert(Arrow::<A>(x, y), n);
        }
        true
    }

    pub fn normalize(&mut self) {
        if self.open { return; }
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
                    },
                    Ordering::Equal => {
                        self.duels.remove(&xy);
                        self.duels.remove(&yx);
                    },
                    Ordering::Greater => {
                        self.duels.insert(xy, 1);
                        self.duels.remove(&yx);
                    },
                }
            }
        }
    }

    pub fn add_alternative(&mut self, v: &A) -> bool {
        if !self.open { return false; }
        self.alternatives.insert(v.to_owned())
    }

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
        DuelGraph{ v: v, a: a }
    }

    pub fn has_alternative(&self, x: &A) -> bool {
        self.alternatives.contains(x)
    }

    pub fn get_condorcet_winner(&self) -> Option<A> {
        self.get_duel_graph().get_source()
    }

    pub fn get_condorcet_loser(&self) -> Option<A> {
        self.get_duel_graph().get_sink()
    }

    pub fn get_minimax_strategy(&self)
        -> Result<Strategy<A>, simplex::SimplexError>
    {
        self.get_duel_graph().get_minimax_strategy()
    }

    pub fn get_maximin_strategy(&self)
        -> Result<Strategy<A>, simplex::SimplexError>
    {
        self.get_duel_graph().get_maximin_strategy()
    }

    pub fn get_optimal_strategy(&self) -> Result<Strategy<A>, ElectionError> {
        self.get_duel_graph().get_optimal_strategy()
    }

    pub fn get_randomized_winner(&self) -> Result<Option<A>, ElectionError> {
        Ok(self.get_optimal_strategy()?.play())
        //let p = self.get_duel_graph().get_minimax_strategy()?;
        //Ok(play_strategy(&p).clone())
    }
}

impl <A: Clone + Eq + Hash + fmt::Display> fmt::Display for Election<A> {
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

    fn random_graph(names: &Vec<String>) -> DuelGraph<String> {
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
        DuelGraph {
            v: v,
            a: a,
        }
    }

    #[test]
    fn source_strategy() {
        let names = string_vec!["Alpha", "Bravo", "Charlie",
                                "Delta", "Echo", "Foxtrot"];
        for n in 1..=names.len() {
            for _ in 0..100 {
                let mut m = Adjacency::from_element(n, n, false);
                (0..n).for_each(|i|
                    (0..i).for_each(|j| if rand::random::<f64>()< 0.5f64 {
                        m[(i, j)] = true;
                    } else {
                        m[(j, i)] = true;
                    }
                ));
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
                    g.get_minimax_strategy().unwrap()
                     .almost_chooses(&w.to_string(), 1e-6),
                    "Minimax doesn't choose {}", w
                );
                assert!(
                    g.get_maximin_strategy().unwrap()
                     .almost_chooses(&w.to_string(), 1e-6),
                    "Minimax doesn't choose {}", w
                );
                assert!(g.get_optimal_strategy().unwrap().is_pure(),
                        "Optimal strategy is mixed");
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
                assert!(b.insert(names[(i + (j as usize)) % 3].to_owned(),
                                 j, j),
                        "add_entry failed");
            }
        }
        for b in b.iter().cloned() { e.cast(b); }
        let g = e.get_duel_graph();
        assert_eq!(g.get_source(), None);
        assert_eq!(g.get_sink(), None);
        assert!(g.get_optimal_strategy().unwrap().is_uniform(&names, 1e-6),
                "Non uniform strategy for Condorcet paradox");
    }

    // Last name commented out for convenience (doubles testing time)
    #[test]
    fn tournament() {
        let names = string_vec!["Alpha", "Bravo", "Charlie",
                                "Delta", "Echo"/*, "Foxtrot"*/];
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
                            let p = Strategy::random_mixed(&v);
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
                    },
                    (Err(e), Ok(maximin)) => {
                        println!("{}\nMinimax failed: {}", g, e);
                        for _ in 0..100 {
                            let p = Strategy::random_mixed(&v);
                            let v = g.confront_strategies(&maximin, &p);
                            if v < -1e-6 {
                                panic!(
                                    "{:?} beats maximin by {}\n{:?}",
                                    p,
                                    v,
                                    maximin
                                );
                            }
                        }
                    },
                    (Ok(minimax), Err(e)) => {
                        println!("{}\nMaximin failed: {}", g, e);
                        for _ in 0..100 {
                            let p = Strategy::random_mixed(&v);
                            let v = g.confront_strategies(&minimax, &p);
                            if v < -1e-6 {
                                panic!(
                                    "{:?} beats minimax by {}\n{:?}",
                                    p,
                                    v,
                                    minimax
                                );
                            }
                        }
                    },
                    (Err(e), Err(f)) =>
                        panic!(
                            "{}\nBoth failed:\n * minimax: {}\n * maximin: {}",
                            g,
                            e,
                            f
                        ),
                };
                // Next graph
                let mut carry = true;
                for i in 1..n {
                    for j in 0..i {
                        if !carry { break; }
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
                if (1..n).all(|i| (0..i).all(|j| !a[(i, j)])) { break; }
            }
        }
    }

    // FIXME: Fails in rare cases due to unsolvable system
    #[test]
    fn optimal_strategy() {
        let names = string_vec!["Alpha", "Bravo", "Charlie",
                                "Delta", "Echo", "Foxtrot"];
        for _pass in 0..1000 {
            println!("Pass {}", _pass);
            let g = random_graph(&names);
            println!("{}", g);
            match (g.get_minimax_strategy(), g.get_maximin_strategy()) {
                (Ok(minimax), Ok(maximin)) => {
                    let opt = g.get_optimal_strategy().unwrap();
                    assert!(g.confront_strategies(&opt, &minimax) > -1e-6,
                            "Minimax beats optimal strategy");
                    assert!(g.confront_strategies(&opt, &maximin) > -1e-6,
                            "Maximin beats optimal strategy");
                },
                (Ok(minimax), Err(_)) => {
                    let opt = g.get_optimal_strategy().unwrap();
                    assert!(g.confront_strategies(&opt, &minimax) > -1e-6,
                            "Minimax beats optimal strategy");
                },
                (Err(_), Ok(maximin)) => {
                    let opt = g.get_optimal_strategy().unwrap();
                    assert!(g.confront_strategies(&opt, &maximin) > -1e-6,
                            "Maximin beats optimal strategy");
                },
                (Err(e), Err(f)) => panic!("Both failed: {}\n{}", e, f),
            }
        }
    }

    fn random_ballot(v: &Vec<String>) -> Ballot<String> {
        let mut b = Ballot::<String>::new();
        for x in v.iter() {
            let s = rand::random::<u64>();
            let r = rand::random::<u64>() % (s + 1);
            assert!(b.insert(x.to_string(), r, s),
                    "Insert ({}, {}) failed", r, s);
        }
        b
    }

    #[test]
    fn agregate() {
        let names = string_vec!["Alpha", "Bravo", "Charlie",
                                "Delta", "Echo", "Foxtrot"];
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
            assert_eq!(e.alternatives, sum.alternatives,
                       "Alternative lists differ");
            for (a, n) in e.duels.into_iter() {
                match sum.duels.get(&a) {
                    Some(m) => assert_eq!(*m, n,
                                          "{:?} is {} in e but {} in sum",
                                          a, n, m),
                    None => panic!("{:?} isn't in sum", a),
                }
            }
        }
    }

    #[test]
    fn normalize() {
        let names = string_vec!["Alpha", "Bravo", "Charlie",
                                "Delta", "Echo", "Foxtrot"];
        for _pass in 0..100 {
            let mut e = Election::<String>::new();
            for _ in 0..500 { e.cast(random_ballot(&names)); }
            let mut n = e.clone();
            n.close();
            n.normalize();
            for x in n.alternatives.iter() {
                let xx = Arrow::<String>(x.to_string(), x.to_string());
                assert_eq!(n.duels.get(&xx), None,
                           "{} wins over itself", x);
                for y in n.alternatives.iter() {
                    let xy = Arrow::<String>(x.to_string(), y.to_string());
                    let yx = Arrow::<String>(y.to_string(), x.to_string());
                    if let Some(m) = n.duels.get(&xy) {
                        assert_eq!(n.duels.get(&yx), None,
                                   "{} and {} loop", x, y);
                        assert_eq!(*m, 1, "Normalized election has {}", m);
                        if let Some(n) = e.duels.get(&yx) {
                            assert!(e.duels.get(&xy).unwrap() > n,
                                    "Backward normalization");
                        }
                    }
                }
            }
        }
    }
}
