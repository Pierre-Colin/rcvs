# Randomized Condorcet Voting System
This is a Rust implementation of the Randomized Condorcet Voting System (RCVS).

## Introduction to the RCVS
The RCVS is a generalization of the [Condorcet method](https://en.wikipedia.org/wiki/Condorcet_method) which offers a fair way to pick a winner in the absence of Condorcet winners. It came out of the arcanes of mathematical research when French YouTuber [Lê Nguyên Hoang](https://scholar.google.ch/citations?user=0ZADKSkAAAAJ&hl=en) wrote [a paper summarizing the decisive theorems proving its nice properties](https://link.springer.com/article/10.1007/s00355-017-1031-2) and presented it in [his series of videos on democracy and game theory](https://youtu.be/wKimU8jy2a8) (subtitles available and auto-translatable by YouTube).

### The Condorcet method
The Condorcet method, developed by 18th century French mathematician Nicolas de Condorcet, consists of aggregating pairwise comparisons between alternatives (a.k.a. candidates) into a *duel graph* where the vertices are alternatives and the arrows means one alternative is preferred over another more often than the opposite. The *Condorcet criterion* states that if there is only one source in the duel graph, the alternative it corresponds to must be elected, for it will never lose against another alternative it can be compared to.

### Nondeterminism
Unfortunately, it can happen, albeit not that often in practice, that there is no source or that there are several. In that case, the Condorcet method cannot choose any, and another method needs to be used. The RCVS relies on mathematical work dating back mostly to the 1990s. It sees the duel graph as a generalized rock-paper-scissors game and computes the _best_ mixed strategy, meaning the probability distribution which will maximize elector satisfaction were the winner to be picked by it. The best mixed strategy is described quantitatively by the minimax theorem and can be computed by means of linear programming. It is not always uniform, and if the duel graph is not a tournament, it might not be unique.

This voting system has a few advantages. Most notably, it incentivizes honest voting even when there are more than two alternatives to choose from, a property most voting systems used in politics lack. More on the theoretical aspects of the voting system can be found in the file `doc/report.tex`.

## Implementation
This Rust implementation is fairly basic, for I am still a beginner in Rust. In particular, both the ballots and the election structure clone everything instead of working with references and it is not concurrent yet (although the title of the report suggests it will eventually be).

## Contribution guidelines
I welcome all help to improve it, be it with Rust or with the algorithms at hand, as long as you explain your work and don't bring in unnecessary dependencies that will make building it more of a hassle ([nalgebra](https://www.nalgebra.org/) is already a big one that I wish I could realistically do without). Please don't propose to replace the simplex algorithm with a wrapper of a pre-existing LP solver. Most of them would interface poorly with the rest of my code and are way too complex for what I am trying to do here. In particular, they tend to be optimized for huge sparse matrices whereas here we're dealing with small dense ones.

### To-dos
* complete the simplex algorithm for maximin strategies (≥ constraints);
* more tests;
* make it concurrent, if possible in a more efficient way than just wrapping Election with a mutex;
* optimize resource management if possible (ballots having ownership of everything, etc.);
* remove all those unwraps in the simplex algorithm, and replace them with proper error management.

## Dependencies
The root dependencies are:
* [rand-0.6.4](https://crates.io/crates/rand)
* [nalgebra-0.16.0](https://crates.io/crates/nalgebra)

These are not up to date right now. As it turns out, `rand` was updated recently, and `nalgebra` did not compile on my (obsolete) version of `rustc`. Nevertheless, I have access to all the features I need right now, so it is not urgent to change those. Note that these dependencies in turn bring other dependencies into the picture.

## Building
As easy as:
```
cargo build
```
I don't really know what to do with the compiled files though, as I'm only testing with `cargo test`. Like I said, I'm still a beginner in Rust.
