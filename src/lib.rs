extern crate nalgebra as na;
extern crate rand;

use std::{
    fmt,
    clone::Clone,
    collections::{HashMap, hash_map, HashSet},
    cmp::Eq,
    hash::Hash,
};
mod ballot;
mod simplex;

type Adjacency = na::DMatrix<bool>;
type Matrix = na::DMatrix<f64>;

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

pub fn print_ballot<A: fmt::Display + Eq + Hash>(b: &Ballot<A>) {
    println!("Ballot {{");
    for (a, r) in b.m.iter() {
        println!("    {} ranks {}", a, r);
    }
    println!("}}")
}

#[derive(Hash)]
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
        writeln!(f, "Graph {{");
        writeln!(f, "Alternatives: {:?}", self.v);
        writeln!(f, "{}", self.a);
        write!(f, "}}")
    }
}

impl <A> DuelGraph<A>
where A: fmt::Debug + Clone
{
    pub fn get_source(&self) -> Option<A> {
        let mut n: Option<A> = None;
        for i in 0..self.v.len() {
            if (0..self.v.len()).all(|j| !self.a[(j, i)]) {
                match n {
                    Some(_) => return None,
                    None => n = Some(self.v[i].clone()),
                }
            }
        }
        n
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

    pub fn get_minimax_strategy(&self) -> Vec<(A, f64)> {
        let n = self.v.len();
        let mut m = Self::adjacency_to_matrix(&self.a);
        m.iter_mut().for_each(|e| *e += 2f64);
        let b = na::DVector::from_element(n, 1f64);
        let c = na::DVector::from_iterator(2 * n,
                                           std::iter::repeat(0f64).take(n)
                                                                  .chain(std::iter::repeat(-1f64).take(n)));
        let x = simplex::simplex(&m, &c, &b);
        let p = simplex::vector_to_lottery(x);
        self.v.iter().cloned().zip(p.into_iter()).collect()
    }

    pub fn get_maximin_strategy(&self) -> Vec<(A, f64)> {
        let n = self.v.len();
        let mut m = Self::adjacency_to_matrix(&self.a);
        m.iter_mut().for_each(|e| *e = -(*e + 2f64));
        let b = na::DVector::from_element(n, -1f64);
        let c = na::DVector::from_iterator(2 * n,
                                           std::iter::repeat(0f64).take(n)
                                                                  .chain(std::iter::repeat(1f64).take(n)));
        let x = simplex::simplex(&m, &c, &b);
        let p = simplex::vector_to_lottery(x);
        self.v.iter().cloned().zip(p.into_iter()).collect()
    }
}

pub struct Election<A>
where A: Eq + Hash + Clone + fmt::Display
{
    alternatives: HashSet<A>,
    duels: HashMap<Arrow<A>, u64>,
    open: bool,
}

impl <A> Election<A>
where A: fmt::Display + Eq + Hash + Clone + fmt::Debug
{
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

    pub fn add_alternative(&mut self, v: &A) -> bool {
        if !self.open { return false; }
        self.alternatives.insert(v.to_owned())
    }

    //fn pseudocast(&mut self, ballot: &Ballot<A>) -> bool {
    //    if !self.open {
    //        return false;
    //    }
    //    for x in ballot.iter() {
    //        let (a, r) = x;
    //        self.alternatives.insert(a.to_owned());
    //        for y in ballot.iter() {
    //            let (b, s) = y;
    //            self.alternatives.insert(b.to_owned());
    //            if r > s {
    //                let n = match self.get(a, b) {
    //                    Some(m) => m + 1,
    //                    None => 1,
    //                };
    //                self.duels.insert(Arrow::<A>(a.to_owned(), b.to_owned()), n);
    //            }
    //        }
    //    }
    //    true
    //}

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

    pub fn get_minimax_lottery(&self) -> Vec<(A, f64)> {
        self.get_duel_graph().get_minimax_strategy()
    }

    pub fn get_randomized_winner(&self) -> A {
        let p = self.get_duel_graph().get_minimax_strategy();
        play_strategy(&p).clone()
    }
}

impl <A> fmt::Display for Election<A>
    where A: Eq + Hash + Clone + fmt::Display
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Election {{");
        for x in self.duels.iter() {
            let (Arrow::<A>(a, b), n) = x;
            writeln!(f, "    {} beats {} {} times", a, b, n);
        }
        write!(f, "}}")
    }
}

fn insertion_sort<A, F>(a: &mut Vec<A>, b: usize, e: usize, compare: &F)
    where F: Fn(&A, &A) -> std::cmp::Ordering
{
    for i in (b + 1)..e {
        let mut j = i;
        while j > 0 && compare(&a[j], &a[j - 1]) == std::cmp::Ordering::Less {
            a.swap(j - 1, j);
        }
    }
}

fn quick_sort<A, F>(mut a: Vec<A>, compare: F) -> Vec<A> 
    where F: Fn(&A, &A) -> std::cmp::Ordering
{
    let mut stack = vec![(0usize, a.len())];
    while !stack.is_empty() {
        let last = stack.len() - 1;
        let (b, size) = stack.remove(last);
        if size <= 7 {
            insertion_sort(&mut a, b, b + size, &compare);
        } else {
            a.swap(rand::random::<usize>() % size, b);
            let mut i = b;
            let mut j = i;
            let mut k = b + size - 1;
            // Invariant: [i, j) only contains copies of the pivot
            while j < k {
                match compare(&a[j], &a[i]) {
                    std::cmp::Ordering::Less => {
                        a.swap(i, j);
                        i += 1;
                        j += 1;
                    },
                    std::cmp::Ordering::Greater => {
                        a.swap(j, k);
                        k -= 1;
                    },
                    std::cmp::Ordering::Equal => j += 1,
                }
            }
            stack.push((b, i - b));
            stack.push((j, size - j));
        }
    }
    a
}

pub fn play_strategy<A>(p: &Vec<(A, f64)>) -> A
where A: std::fmt::Debug + Clone
{
    // Sort the array for better numerical accuracy
    println!("Before: {:?}", p);
    let a = quick_sort(p.clone().to_vec(),
                       |(_, x), (_, y)| x.partial_cmp(y).unwrap().reverse());
    println!("Sorted: {:?}", a);
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
                assert!(strategy_chooses(g.get_minimax_strategy(),
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
        println!("Minimax {:?}", g.get_minimax_strategy());
        println!("Maximin {:?}", g.get_maximin_strategy());
        assert!(g.get_minimax_strategy()
                 .into_iter()
                 .all(|(_, x)| f64::abs(x - 0.333f64) < 0.001f64),
                "Strategy for Condorcet paradox isn't uniform");
    }
}
