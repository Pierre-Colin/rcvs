extern crate rcvs;

extern crate rand;

use std::cmp::Ordering;

fn ovo_ballot(e: &mut rcvs::Election<String>, name: &str, other: &str) {
    let mut b = rcvs::Ballot::<String>::new();
    assert!(b.insert(name.to_string(), 1, 1), "Ballot building error");
    assert!(b.insert(other.to_string(), 0, 0), "Ballot building error");
    e.cast(b);
}

fn ovo_wins(p: Vec<(String, f64)>, w: Option<String>) -> bool {
    match w {
        None => p.into_iter().all(|(_, x)| f64::abs(x - 0.5f64) < 0.000001f64),
        Some(x) => {
            if let Some((n, _)) = p.iter()
                                   .enumerate()
                                   .find(|(_, (e, _))| e == x.as_str()) {
                1f64 - p[n].1 < 0.000001f64 && p[1usize - n].1 < 0.000001f64
            } else {
                false
            }
        },
    }
}

#[test]
fn one_versus_one() {
    for _ in 1..=100 {
        let mut e = rcvs::Election::<String>::new();
        let na = rand::random::<u64>() % 1000;
        let nb = rand::random::<u64>() % 1000;
        for _ in 1..=na { ovo_ballot(&mut e, "Alpha", "Bravo"); }
        for _ in 1..=nb { ovo_ballot(&mut e, "Bravo", "Alpha"); }
        let g = e.get_duel_graph();
        let s = g.get_source();
        let p = g.get_minimax_strategy();
        match na.cmp(&nb) {
            Ordering::Less => assert_eq!(s, Some("Bravo".to_string())),
            Ordering::Equal => assert_eq!(s, None),
            Ordering::Greater => assert_eq!(s, Some("Alpha".to_string())),
        }
        assert!(ovo_wins(p, s), "Minimax strategy doesn't elect winner");
    }
}

fn cp_ballot(e: &mut rcvs::Election<String>, names: Vec<String>) {
    let mut b = rcvs::Ballot::<String>::new();
    let l = names.len();
    names.iter()
         .enumerate()
         .for_each(|(i, n)| {
             let r = (l - i) as u64;
             assert!(b.insert(n.to_string(), r, r), "Ballot building error");
         });
    e.cast(b);
}

fn is_uniform(p: Vec<(String, f64)>) -> bool {
    let y = 1f64 / (p.len() as f64);
    p.into_iter().all(|(_, x)| f64::abs(x - y) < 0.001f64)
}

#[test]
fn condorcet_paradox() {
    let names = ["Alpha", "Bravo", "Charlie"];
    let mut e = rcvs::Election::<String>::new();
    for i in 0..3 {
        let v: Vec<String> = (0..3).map(|j| names[(i + j) % 3].to_string())
                                   .collect();
        cp_ballot(&mut e, v);
    }
    let g = e.get_duel_graph();
    assert_eq!(g.get_source(), None);
    let p = g.get_minimax_strategy();
    println!("{:?}", p);
    assert!(is_uniform(p));
    let p = g.get_maximin_strategy();
    println!("{:?}", p);
    assert!(is_uniform(p));
}

fn strategy_chooses(p: Vec<(String, f64)>, w: &String) -> bool {
    p.into_iter().all(|(v, x)|
        if v == *w { 1f64 - x < 0.000001f64 } else { x < 0.000001f64 }
    )
}

#[test]
fn condorcet_winner() {
    let names = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"];
    for n in 2..=names.len() {
        let v: Vec<String> = names.iter()
                                  .take(n)
                                  .map(|x| x.to_string())
                                  .collect();
        let s = rand::random::<usize>() % n;
        for _ in 0..10 {
            let mut e = rcvs::Election::<String>::new();
            for _ in 0..100 {
                let mut b = rcvs::Ballot::<String>::new();
                v.iter().enumerate().for_each(|(i, x)|
                    if i == s {
                        let r = n as u64;
                        assert!(b.insert(v[s].clone(), r, r),
                                "Ballot building error");
                    } else {
                        let rsup = rand::random::<u64>() % (n as u64);
                        let rinf = rand::random::<u64>() % (rsup + 1);
                        assert!(b.insert(x.to_string(), rinf, rsup),
                                "Ballot building error");
                    }
                );
                assert!(e.cast(b), "Election was closed");
            }
            assert_eq!(e.get_condorcet_winner(), Some(v[s].clone()));
            assert!(strategy_chooses(e.get_minimax_lottery(), &v[s]));
        }
    }
}

fn single_ballot(e: &mut rcvs::Election<String>, x: &str, y: &str) {
    let mut b = rcvs::Ballot::<String>::new();
    assert!(b.insert(x.to_string(), 1, 1), "Failed to insert {}", x);
    assert!(b.insert(y.to_string(), 0, 0), "Failed to insert {}", y);
    e.cast(b);
}

#[test]
fn five_non_uniform() {
    let names = vec!["Alpha", "Bravo", "Charlie", "Delta", "Echo"];
    let mut e = rcvs::Election::<String>::new();
    single_ballot(&mut e, names[0], names[1]);
    single_ballot(&mut e, names[0], names[2]);
    single_ballot(&mut e, names[0], names[3]);
    single_ballot(&mut e, names[1], names[2]);
    single_ballot(&mut e, names[1], names[4]);
    single_ballot(&mut e, names[2], names[3]);
    single_ballot(&mut e, names[2], names[4]);
    single_ballot(&mut e, names[3], names[1]);
    single_ballot(&mut e, names[3], names[4]);
    single_ballot(&mut e, names[4], names[0]);
    assert_eq!(e.get_condorcet_winner(), None);
    let g = e.get_duel_graph();
    let minimax = g.get_minimax_strategy();
    let maximin = g.get_maximin_strategy();
    println!("{:?}\n{:?}", minimax, maximin);
    // Panic to print during tests (dirty but too lazy to do it the right way)
    panic!("Nope");
}
