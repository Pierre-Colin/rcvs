extern crate nalgebra as na;
extern crate rand;

mod ballot;
mod simplex;
pub mod util;

use std::{
    fmt,
    clone::Clone,
    collections::{HashMap, hash_map, HashSet},
    cmp::{Eq, Ordering},
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

impl <A> fmt::Display for DuelGraph<A>
    where A: fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Graph {{")?;
        writeln!(f, "Alternatives: {:?}", self.v)?;
        writeln!(f, "{}", self.a)?;
        write!(f, "}}")
    }
}

impl <A> DuelGraph<A>
    where A: fmt::Debug + Clone + std::cmp::Eq
{
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
        -> Result<Vec<(A, f64)>, simplex::SimplexError>
    {
        let n = self.v.len();
        let b = Vector::from_element(n, bval);
        let c = Vector::from_element(n, cval);
        let x = simplex::simplex(m, &c, &b)?;
        let p = simplex::vector_to_lottery(x);
        Ok(self.v.iter().cloned().zip(p.into_iter()).collect())
    }

    pub fn get_minimax_strategy(&self)
        -> Result<Vec<(A, f64)>, simplex::SimplexError>
    {
        let mut m = Self::adjacency_to_matrix(&self.a);
        m.iter_mut().for_each(|e| *e += 2f64);
        self.compute_strategy(&m, 1f64, -1f64)
    }

    pub fn get_maximin_strategy(&self)
        -> Result<Vec<(A, f64)>, simplex::SimplexError>
    {
        let mut m = Self::adjacency_to_matrix(&self.a).transpose();
        m.iter_mut().for_each(|e| *e = -(*e + 2f64));
        self.compute_strategy(&m, -1f64, 1f64)
    }

    // TODO: make an error type
    pub fn get_optimal_strategy(&self) -> Option<Vec<(A, f64)>> {
        match (self.get_minimax_strategy(), self.get_maximin_strategy()) {
            (Ok(minimax), Ok(maximin)) => {
                Some(match self.compare_strategies(&minimax, &maximin) {
                    Ordering::Less => maximin,
                    Ordering::Equal => minimax,
                    Ordering::Greater => minimax,
                })
            },
            (Err(_), Ok(maximin)) => Some(maximin),
            (Ok(minimax), Err(_)) => Some(minimax),
            (Err(_), Err(_)) => None,
        }
    }

    fn strategy_vector(&self, p: &Vec<(A, f64)>) -> Vector {
        Vector::from_iterator(self.v.len(), self.v.iter().map(|e|
            match p.iter().find(|(u, _)| *u == *e) {
                None => panic!("Alternative not found"),
                Some((_, p)) => p.clone(),
            }
        ))
    }

    pub fn confront_strategies(&self, x: &Vec<(A, f64)>, y: &Vec<(A, f64)>)
        -> f64
    {
        let m = Self::adjacency_to_matrix(&self.a);
        let p = self.strategy_vector(x);
        let q = self.strategy_vector(y);
        (p.transpose() * m * q)[(0, 0)]
    }

    pub fn compare_strategies(&self, x: &Vec<(A, f64)>, y: &Vec<(A, f64)>)
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
                    let n = match self.get(a, b) {
                        Some(m) => m + 1,
                        None => 1,
                    };
                    self.duels.insert(Arrow::<A>(a.to_owned(), b.to_owned()), n);
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
            let n = match self.get(&x, &y) {
                Some(k) => m + k,
                None => m,
            };
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

    pub fn get_minimax_lottery(&self)
        -> Result<Vec<(A, f64)>, simplex::SimplexError>
    {
        self.get_duel_graph().get_minimax_strategy()
    }

    pub fn get_randomized_winner(&self) -> Result<A, simplex::SimplexError> {
        let p = self.get_duel_graph().get_minimax_strategy()?;
        Ok(play_strategy(&p).clone())
    }
}

impl <A> fmt::Display for Election<A>
    where A: Clone + Eq + Hash + fmt::Display
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Election {{")?;
        for x in self.duels.iter() {
            let (Arrow::<A>(a, b), n) = x;
            writeln!(f, "    {} beats {} {} times", a, b, n)?;
        }
        write!(f, "}}")
    }
}

pub fn play_strategy<A>(p: &Vec<(A, f64)>) -> A
    where A: std::fmt::Debug + Clone
{
    // Sort the array for better numerical accuracy
    let a = quick_sort(p.clone().to_vec(),
                       |(_, x), (_, y)| x.partial_cmp(y).unwrap().reverse());
    let mut x = rand::random::<f64>();
    for (v, p) in a.into_iter() {
        x -= p;
        if x <= 0f64 { return v; }
    }
    panic!("Strategy is ill-formed");
}

#[cfg(test)]
mod tests {
    use super::*;

    use util::random_strategy;

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

    fn strategy_chooses(p: Vec<(String, f64)>, w: String) -> bool {
        p.into_iter().all(|(v, x)|
            if v == w { 1f64 - x < 0.000001f64 } else { x < 0.000001f64 }
        )
    }

    #[test]
    fn source_strategy() {
        let names: Vec<String> = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
            .into_iter().map(|x| x.to_string()).collect();
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
                let w = g.get_source();
                assert_ne!(w, None, "No source in graph {}", g);
                assert!(strategy_chooses(g.get_minimax_strategy().unwrap(),
                                         w.unwrap().to_string()));
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
        let names = ["Alpha".to_string(),
                     "Bravo".to_string(),
                     "Charlie".to_string()];
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
        println!("Minimax {:?}", g.get_minimax_strategy().unwrap());
        println!("Maximin {:?}", g.get_maximin_strategy().unwrap());
        assert!(g.get_minimax_strategy().unwrap()
                 .into_iter()
                 .all(|(_, x)| f64::abs(x - 0.333f64) < 0.001f64),
                "Strategy for Condorcet paradox isn't uniform");
    }

    // Last name commented out for convenience (doubles testing time)
    #[test]
    fn tournament() {
        let names = vec!["Alpha", "Bravo", "Charlie",
                         "Delta", "Echo" /*, "Foxtrot"*/];
        for n in 1..=names.len() {
            println!("Size {}", n);
            let v: Vec<String> = names.iter().take(n).map(|x| x.to_string()).collect();
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
                            let p = random_strategy(&v);
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
                            let p = random_strategy(&v);
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
                            let p = random_strategy(&v);
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

    #[test]
    fn optimal_strategy() {
        let names = vec!["Alpha".to_string(), "Bravo".to_string(),
                         "Charlie".to_string(), "Delta".to_string(),
                         "Echo".to_string(), "Foxtrot".to_string()];
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
        let names = vec!["Alpha".to_string(), "Bravo".to_string(),
                         "Charlie".to_string(), "Delta".to_string(),
                         "Echo".to_string(), "Foxtrot".to_string()];
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
        let names = vec!["Alpha".to_string(), "Bravo".to_string(),
                         "Charlie".to_string(), "Delta".to_string(),
                         "Echo".to_string(), "Foxtrot".to_string()];
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
