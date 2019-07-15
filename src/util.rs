/*
 * NOTE:
 * This file is intended to provide functions that may be used in various
 * places in the project. Most of these functions are public, but are not
 * guaranteed to be supported in the future. They should only be used inside
 * the crate and in integration tests.
 */
fn insertion_sort<A, F>(a: &mut [A], b: usize, e: usize, compare: &F)
    where F: Fn(&A, &A) -> std::cmp::Ordering
{
    for i in (b + 1)..e {
        let mut j = i;
        while j > 0 && compare(&a[j], &a[j - 1]) == std::cmp::Ordering::Less {
            a.swap(j - 1, j);
        }
    }
}

pub fn quick_sort<A, F>(mut a: Vec<A>, compare: F) -> Vec<A> 
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

/*
 * Credit:
 * https://stackoverflow.com/questions/38183551/concisely-initializing-a-vector-of-strings
 */
#[macro_export]
macro_rules! string_vec {
    ($($x: expr), *) => (vec![$($x.to_string()), *]);
}
