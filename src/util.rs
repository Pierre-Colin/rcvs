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
/// # use rcvs::string_vec;
/// let names = string_vec!["Stan", "Kyle", "Eric", "Kenny"];
/// ```
#[macro_export]
macro_rules! string_vec {
    ($($x: expr), *) => (vec![$($x.to_string()), *]);
}
