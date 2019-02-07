//! 2D filters for image processing.

use ndarray::prelude::*;
use ndarray::{Data, DataMut};

/// Representing the border type for filters.
///
/// Following border types are supported:
/// * Constant(v): a constant, i.e., vvvv|abcdefgh|vvvv
/// * Reflect: reflect the image, i.e., edcb|abcdefgh|gfed
/// * Replicate: copy the value at the border, i.e., aaaa|abcdefgh|hhhh
#[derive(Copy, Clone)]
pub enum BorderType {
    Constant(f64),
    Reflect,
    Replicate
}

/// Compute the source location of the outside point.
///
/// This function is used by `filter`. 
/// For example, when the border type is `Reflect`, 
/// ```
/// use simplecv::filter::*;
/// let nx = border_interpolate(-2, 10, BorderType::Reflect).unwrap();
/// assert_eq!(nx, 2);
/// ```
/// 
/// The function return None when border type is Constant.
pub fn border_interpolate(p:i32, len:usize, border: BorderType) -> Option<usize> {
    fn abs(a: i32) -> i32{
        if a < 0 {
            -a
        } else {
            a
        }
    }
    match border{
        BorderType::Constant(_) => None,
        BorderType::Reflect => Some(abs(p % (len as i32)) as usize),
        BorderType::Replicate => {
            if p < 0 {
                Some(0usize)
            } else {
                Some(len - 1)
            }
        }
    }
}

/// Get the value of an image at a location which may be outside the image.
///
/// This function is used by `filter`.
fn access_img_border<S>(src: &ArrayBase<S, Ix2>, x:i32, y:i32, border: BorderType) -> f64 
    where S:Data<Elem=f64> 
{
    if x >= 0 && y >= 0 && x < src.shape()[0] as i32 && y < src.shape()[1] as i32 {
        src[[x as usize, y as usize]]
    }
    else {
        match border {
            BorderType::Constant(v) => v,
            BorderType::Reflect|BorderType::Replicate => {
                let nx = border_interpolate(x, src.shape()[0], border).unwrap();
                let ny = border_interpolate(y, src.shape()[1], border).unwrap();
                src[[nx, ny]]
            }
        }
    }
}

/// Apply a linear filter to the source image.
///
/// `out` must have the exactly same shape of `src`. Both the `src` and `kernel` 
/// should be 2D array. For more channels you may need to write a wrapper by yourself.
///
/// The method of dealing with border situation is selected by `border`. By setting
/// `border=Reflect`, you will get the default result of OpenCV.
///
pub fn filter_<S, T, K>(src: &ArrayBase<S, Ix2>, kernel: &ArrayBase<K, Ix2>, border: BorderType, 
               out:&mut ArrayBase<T, Ix2>)
    where S: Data<Elem=f64>, T: DataMut<Elem=f64>, K: Data<Elem=f64>
{
    let kh = kernel.shape()[0];
    let kw = kernel.shape()[1];
    let kcx = (kh / 2) as i32; // kernel center x
    let kcy = (kw / 2) as i32; // kernel center x
    let height = src.shape()[0];
    let width = src.shape()[1];
    for i in 0..height {
        for j in 0..width {
            let mut val = 0.0f64;
            for ki in 0..kh {
                for kj in 0..kw {
                    let sx = i as i32 + ki as i32 - kcx;
                    let sy = j as i32 + kj as i32 - kcy;
                    let sval = access_img_border(src, sx, sy, border);
                    val = val + sval * kernel[[ki, kj]];
                }
            }
            out[[i, j]] = val;
        }
    }
}

/// Apply a linear filter to the source image.
///
/// The method of dealing with border situation is selected by `border`. By setting
/// `border=Reflect`, you will get the default result of OpenCV.
/// 
/// # Example
/// ```
///  use simplecv::filter::*;
///  use ndarray::arr2;
///  let A = arr2(&[[0.0, 0.0, 0.0, 0.0, 0.0],
///                 [0.0, 1.0, 1.0, 1.0, 0.0],
///                 [0.0, 1.0, 1.0, 1.0, 0.0],
///                 [0.0, 1.0, 1.0, 1.0, 0.0],
///                 [0.0, 0.0, 0.0, 0.0, 0.0]]);
///  let kernel = arr2(&[[1.0, 2.0, 1.0],
///                      [2.0, 4.0, 2.0],
///                      [1.0, 2.0, 1.0]]);
///  let target = arr2(&[[1.0, 3.0, 4.0, 3.0, 1.0],
///                      [3.0, 9.0, 12.0, 9.0, 3.0],
///                      [4.0, 12.0, 16.0, 12.0, 4.0],
///                      [3.0, 9.0, 12.0, 9.0, 3.0],
///                      [1.0, 3.0, 4.0, 3.0, 1.0]]);
///  let output = filter(&A, &kernel, BorderType::Constant(0.0));
///  assert_eq!(target, output);
/// ```
///
pub fn filter<S, K>(src: &ArrayBase<S, Ix2>, kernel: &ArrayBase<K, Ix2>, border: BorderType) -> Array<f64, Ix2>
    where S: Data<Elem=f64>, K: Data<Elem=f64>
{
    let shape = src.shape();
    let height = shape[0] as usize;
    let width = shape[1] as usize;
    let mut buffer = Array::zeros((height, width));
    filter_(src, kernel, border, &mut buffer);
    buffer
}

/// Generate a Gaussian kernel with the simplest method.
pub fn gaussian_kernel_generator(ksize: usize) -> Array<f64, Ix2>{
    fn sqr_dis(dx:i32, dy:i32) -> i32{
        dx * dx + dy * dy
    }
    let cx = (ksize / 2) as i32;
    let cy = (ksize / 2) as i32;
    let mut kernel = Array::zeros((ksize, ksize));
    for x in 0..ksize {
        for y in 0..ksize{
            let dist = sqr_dis(cx - x as i32, cy - y as i32);
            kernel[[x, y]] = -dist as f64 / 2.0;
        }
    }
    kernel.map_inplace(|x| *x = x.exp());
    let normalized_constant = kernel.fold(0.0, |acc, x| acc + x);
    kernel /= normalized_constant;
    kernel
}

/// Smooth the image with a gaussian kernel.
///
/// The output buffer should be allcated by users.
/// * `ksize`: is the kernel size. 
/// * `border`: how to deal with the border.
pub fn gaussian_smooth_<S, T>(src: &ArrayBase<S, Ix2>, ksize:usize, border: BorderType, out:&mut ArrayBase<T, Ix2>) 
    where S: Data<Elem=f64>, T:DataMut<Elem=f64>
{
    let kernel = gaussian_kernel_generator(ksize);
    filter_(src, &kernel, border, out);
}

/// Smooth the image with a gaussian kernel.
///
/// * `ksize`: is the kernel size. 
/// * `border`: how to deal with the border.
pub fn gaussian_smooth<S>(src: &ArrayBase<S, Ix2>, ksize:usize, border: BorderType) -> Array<f64, Ix2>
    where S: Data<Elem=f64>
{
    let kernel = gaussian_kernel_generator(ksize);
    filter(src, &kernel, border)
}

/// Smooth the image with a mean kernel.
///
/// The output buffer should be allcated by users.
/// * `ksize`: is the kernel size. 
/// * `border`: how to deal with the border.
pub fn mean_smooth_<S, T>(src: &ArrayBase<S, Ix2>, ksize:usize, border:BorderType, out: &mut ArrayBase<T, Ix2>)
    where S: Data<Elem=f64>, T:DataMut<Elem=f64>
{
    let kernel = Array::ones((ksize, ksize)) / ((ksize * ksize) as f64);
    filter_(src, &kernel, border, out);
}

/// Smooth the image with a mean kernel.
///
/// * `ksize`: is the kernel size. 
/// * `border`: how to deal with the border.
///
/// # Example
/// ```
/// let a = ndarray::arr2(&[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]);
/// let smoothed = simplecv::filter::mean_smooth(&a, 3,
///                     simplecv::filter::BorderType::Constant(0.0));
/// let diff = simplecv::utils::max_diff(&smoothed, &(ndarray::Array::ones((3, 3)) / 9.0));
/// assert!(diff < 1e-4);
/// ```
///
pub fn mean_smooth<S>(src: &ArrayBase<S, Ix2>, ksize:usize, border:BorderType) -> Array<f64, Ix2>
    where S: Data<Elem=f64>
{
    let kernel = Array::ones((ksize, ksize)) / ((ksize * ksize) as f64);
    filter(src, &kernel, border)
}
