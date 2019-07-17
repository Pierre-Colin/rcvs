/*
 * NOTE:
 * This file is intended to provide functions that may be used in various
 * places in the project. Most of these functions are public, but are not
 * guaranteed to be supported in the future. They should only be used inside
 * the crate and in integration tests.
 */
//! # Utilities
//!
//! `rcvs::util` is a module providing a few tools that are useful for both
//! the crate and integration tests. *It is not intended for external use, and
//! the elements of this module are not guaranteed to be supported in the
//! future.*
use std::cmp::Ordering;

fn insertion_sort<A, F>(a: &mut [A], b: usize, e: usize, compare: &F)
    where F: Fn(&A, &A) -> Ordering
{
    for i in (b + 1)..e {
        for j in (1..=i).rev() {
            if compare(&a[j], &a[j - 1]) != Ordering::Less { break; }
            a.swap(j - 1, j);
        }
    }
}

/// Consumes a vector of items ordered by a comparating function, and returns
/// a sorted version of it. The argument `rng` is a `rand::Rng` used to
/// randomize the pivot for optimal average-case time complexity. This
/// implementation falls back to insertion sort when the size of a slice to
/// sort drops below 8 elements.
///
/// # Example
///
/// ```
/// let a = vec![6, 4, 8, 3];
/// let s = quick_sort(a, i32::cmp, &mut rand::thread_rng());
///
/// assert_eq!(vec![3, 4, 6, 8], s);
/// ```
pub fn quick_sort<A, F, R>(mut a: Vec<A>, compare: F, rng: &mut R) -> Vec<A> 
    where F: Fn(&A, &A) -> Ordering,
          R: rand::Rng,
{
    let mut stack = vec![(0usize, a.len())];
    while !stack.is_empty() {
        let (b, size) = stack.pop().unwrap();
        if size <= 7 {
            insertion_sort(&mut a, b, b + size, &compare);
        } else {
            a.swap(b + rng.gen::<usize>() % size, b);
            let mut i = b;
            let mut j = i + 1;
            let mut k = b + size - 1;
            // Invariant: [i, j) only contains copies of the pivot
            while j <= k {
                match compare(&a[j], &a[i]) {
                    Ordering::Less => {
                        a.swap(i, j);
                        i += 1;
                        j += 1;
                    },
                    Ordering::Greater => {
                        a.swap(j, k);
                        k -= 1;
                    },
                    Ordering::Equal => j += 1,
                }
            }
            if i > b { stack.push((b, i - b))};
            if j < b + size { stack.push((j, b + size - j)) };
        }
    }
    a
}

/*
 * Credit:
 * https://stackoverflow.com/questions/38183551/concisely-initializing-a-vector-of-strings
 */
/// Works like the standard `vec!` macro, but calls `.to_string()` on all the
/// items. This is used to create `Vec<String>` conveniently.
///
/// # Example
///
/// ```
/// let names = string_vec!["Stan", "Kyle", "Eric", "Kenny"];
/// ```
#[macro_export]
macro_rules! string_vec {
    ($($x: expr), *) => (vec![$($x.to_string()), *]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_nondecreasing<A, F>(a: &[A], compare: F) -> bool
        where F: Fn(&A, &A) -> Ordering
    {
        if !a.is_empty() {
            let mut last = &a[0];
            for x in a.iter().skip(1) {
                if compare(x, last) == Ordering::Less { return false; }
                last = x;
            }
        }
        true
    }

    #[test]
    fn quick_sort_test() {
        for n in 0..=20 {
            let num_tries = if n < 2 { 1 } else { 1000 };
            for _ in 0..num_tries {
                let a: Vec<f64> = (0..n).map(|_| 
                    rand::random::<f64>() - 0.5f64
                ).collect();
                let s = quick_sort(a,
                                   |x, y| x.partial_cmp(&y).unwrap(),
                                   &mut rand::thread_rng());
                assert!(
                    is_nondecreasing(&s, |x, y| x.partial_cmp(&y).unwrap()),
                    "{:?} isn't sorted",
                    s
                );
            }
        }
    }
}
