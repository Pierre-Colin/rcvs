extern crate rcvs;

extern crate rand;

use std::cmp::Ordering;

use rand::seq::SliceRandom;

use rcvs::string_vec;

fn ovo_ballot(e: &mut rcvs::Election<String>, name: &str, other: &str) {
    let mut b = rcvs::Ballot::<String>::new();
    assert!(b.insert(name.to_string(), 1, 1), "Ballot building error");
    assert!(b.insert(other.to_string(), 0, 0), "Ballot building error");
    e.cast(b);
}

fn ovo_wins(p: rcvs::Strategy<String>, w: &Option<String>) -> bool {
    if let Some(x) = w {
        p.almost_chooses(&x, 1e-8)
    } else {
        // If graph is not weakly connected, maximal strategy is not unique
        true
    }
}

#[test]
fn one_versus_one() {
    for _ in 1..=100 {
        let mut e = rcvs::Election::<String>::new();
        let na = rand::random::<u64>() % 100;
        let nb = rand::random::<u64>() % 100;
        for _ in 1..=na { ovo_ballot(&mut e, "Alpha", "Bravo"); }
        for _ in 1..=nb { ovo_ballot(&mut e, "Bravo", "Alpha"); }
        let g = e.get_duel_graph();
        let s = g.get_source();
        match na.cmp(&nb) {
            Ordering::Less => assert_eq!(s, Some("Bravo".to_string())),
            Ordering::Equal => assert_eq!(s, None),
            Ordering::Greater => assert_eq!(s, Some("Alpha".to_string())),
        }
        let minimax = g.get_minimax_strategy(&mut rand::thread_rng()).unwrap();
        let maximin = g.get_maximin_strategy(&mut rand::thread_rng()).unwrap();
        println!("na = {} and nb = {}", na, nb);
        println!("{}", g);
        println!("Source is {:?}", s);
        println!("Minimax is {}", minimax);
        println!("Maximin is {}", maximin);
        assert!(ovo_wins(minimax, &s),
                "Minimax strategy doesn't elect winner");
        assert!(ovo_wins(maximin, &s),
                "Maximin strategy doesn't elect winner");
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

#[test]
fn condorcet_paradox() {
    let names = string_vec!["Alpha", "Bravo", "Charlie"];
    let mut e = rcvs::Election::<String>::new();
    for i in 0..3 {
        let v: Vec<String> = (0..3).map(|j| names[(i + j) % 3].to_string())
                                   .collect();
        cp_ballot(&mut e, v);
    }
    let g = e.get_duel_graph();
    assert_eq!(g.get_source(), None);
    let p = g.get_minimax_strategy(&mut rand::thread_rng()).unwrap();
    println!("{:?}", p);
    assert!(p.is_uniform(&names, 1e-6));
    let p = g.get_maximin_strategy(&mut rand::thread_rng()).unwrap();
    println!("{:?}", p);
    assert!(p.is_uniform(&names, 1e-6));
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
            assert!(e.get_minimax_strategy(&mut rand::thread_rng())
                     .unwrap()
                     .almost_chooses(&v[s], 1e-6));
            assert!(e.get_maximin_strategy(&mut rand::thread_rng())
                     .unwrap()
                     .almost_chooses(&v[s], 1e-6));
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
    let minimax = g.get_minimax_strategy(&mut rand::thread_rng()).unwrap();
    let maximin = g.get_maximin_strategy(&mut rand::thread_rng()).unwrap();
    println!("{:?}\n{:?}", minimax, maximin);
    let result = rcvs::Strategy::Mixed(vec![
        ("Alpha".to_string(), 1f64 / 3f64),
        ("Bravo".to_string(), 1f64 / 9f64),
        ("Charlie".to_string(), 1f64 / 9f64),
        ("Delta".to_string(), 1f64 / 9f64),
        ("Echo".to_string(), 1f64 / 3f64)
    ]);
    assert!(minimax.distance(&result) < 1e-6);
    assert!(maximin.distance(&result) < 1e-6);
}

#[test]
fn simulate_election() {
    let names = ["Alpha", "Bravo", "Charlie"];
    let mut e = rcvs::Election::<String>::new();
    for i in 0..3 {
        let v: Vec<String> = (0..3).map(|j| names[(i + j) % 3].to_string())
                                   .collect();
        cp_ballot(&mut e, v);
    }
    for _ in 0..10 {
        println!("Winner: {}",
                 e.get_randomized_winner(&mut rand::thread_rng())
                  .unwrap()
                  .unwrap());
    }
}

/*
 * NOTE:
 * Has been observed to fail in extremely rare cases with minimax and maximin
 * strategies often being completely different; record if it happens again.
 */
#[test]
fn condorcet_strategies_optimal() {
    let names: Vec<String> = string_vec!["Alpha", "Bravo", "Charlie",
                                         "Delta", "Echo", "Foxtrot"];
    let num_elections = 200;
    let num_strategies = 500;
    let mut failed = 0u64;
    let mut rng = rand::thread_rng();
    for _enum in 0..num_elections {
        println!("Election #{}", _enum);
        let mut e = rcvs::Election::<String>::new();
        for _ in 0..500 {
            let rank = {
                let mut x = (0..(names.len() as u64)).collect::<Vec<u64>>();
                x.shuffle(&mut rng);
                x
            };
            let mut b = rcvs::Ballot::<String>::new();
            names.iter().zip(rank.into_iter()).for_each(|(v, r)|
                assert!(b.insert(v.to_string(), r, r), "Ill-formed ballot")
            );
            e.cast(b);
        }
        let g = e.get_duel_graph();
        if let Some(x) = g.get_source() { println!("Winner: {}", x); }
        if let Some(x) = g.get_sink() { println!("Loser: {}", x); }
        match (g.get_minimax_strategy(&mut rand::thread_rng()),
               g.get_maximin_strategy(&mut rand::thread_rng()))
        {
            (Ok(minimax), Ok(maximin)) => {
                println!("Both worked with {}", minimax.distance(&maximin));
                for _ in 0..num_strategies {
                    let p = rcvs::Strategy::random_mixed(
                        &names,
                        &mut rand::thread_rng()
                    );
                    if g.confront_strategies(&minimax, &p) < -1e-8
                        && g.confront_strategies(&maximin, &p) < -1e-8 {
                        println!("{}", g);
                        println!("Minimax: {:?}", minimax);
                        println!("Maximin: {:?}", maximin);
                        println!("{:?} beats both minimax and maximin", p);
                        panic!("NNNNNNNNNNNNNNOPE!!!");
                        failed += 1;
                    }
                }
            },
            (Ok(minimax), Err(e)) => {
                println!("Maximin failed: {}", e);
                for _ in 0..1000 {
                    let p = rcvs::Strategy::random_mixed(
                        &names,
                        &mut rand::thread_rng()
                    );
                    if g.confront_strategies(&minimax, &p) < -1e-8 {
                        println!("{}", g);
                        println!("Minimax: {:?}", minimax);
                        println!("{:?} beats minimax", p);
                        panic!("NNNNNNNNNNNNNNOPE!!!");
                        failed += 1;
                    }
                }
            },
            (Err(e), Ok(maximin)) => {
                println!("Minimax failed: {}", e);
                for _ in 0..1000 {
                    let p = rcvs::Strategy::random_mixed(
                        &names,
                        &mut rand::thread_rng()
                    );
                    if g.confront_strategies(&maximin, &p) < -1e-8 {
                        println!("{}", g);
                        println!("Maximin: {:?}", maximin);
                        println!("{:?} beats maximin", p);
                        panic!("NNNNNNNNNNNNNNOPE!!!");
                        failed += 1;
                    }
                }
            },
            (Err(e), Err(f)) => panic!("Both failed:\n{}\n{}", e, f),
        };
    }
    let failure_rate = failed as f64 / (num_elections * num_strategies) as f64;
    println!("Failure rate: {}", failure_rate);
    assert!(failure_rate < 1e-6, "Failure rate was too big");
}
