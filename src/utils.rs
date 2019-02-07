//! Collections of some util functions.
//!
use ndarray::{ArrayBase, Dimension, Data};
use num::traits::Signed;

/// Compute the max absolute difference of two array.
///
/// In most situations, it is used in unit tests.
/// # Example:
/// ```
/// use num::traits::Signed;
///
/// let a = ndarray::arr1(&[0.12, 0.24, 0.35]);
/// let b = ndarray::arr1(&[0.125, 0.22, 0.34]);
/// let diff = simplecv::utils::max_diff(&a, &b);
/// // due to float computation, the diff may be 0.0199999...
/// assert!((diff - 0.02).abs() < 1e-5);
/// ```
pub fn max_diff<A, D, S>(a:&ArrayBase<S, D>, b:&ArrayBase<S, D>) -> A
    where A:Signed + PartialOrd + Copy, D:Dimension, S: Data<Elem=A>
{
    a.iter()
     .zip(b.iter())
     .map(|(v1, v2)| *v1-*v2)
     .map(|v| v.abs())
     .fold(A::zero(), |m, x| if m < x {x} else {m})
}
/// Transform a float to u8 and multiply it by 255.
/// 
/// It is used in transform `image` output to f64 array used by `simplecv`.
/// ```
/// assert_eq!(simplecv::utils::f2u(0.9), 230);
/// ```
pub fn f2u(v: f64) -> u8 {
    (v * 255.0 + 0.5) as u8
}
