//! Color transformation and enhancement.
//!
use ndarray::prelude::*;
use ndarray::{Data, DataMut};
use super::utils::f2u;

/// Transform an RGB image to grayscale image.
///
/// The output buffer is allocated by users.
/// The weights are 0.299, 0.587 and 0.114 for red, green and blue respectively.
pub fn rgb2gray_<A, B>(img: &ArrayBase<A, Ix3>, out: &mut ArrayBase<B, Ix2>)
    where A: Data<Elem=f64>, B: DataMut<Elem=f64>
{
    let shape = img.shape();
    let h = shape[0];
    let w = shape[1];
    let c = shape[2];
    let rgb_weights = [0.299, 0.587, 0.114];
    assert_eq!(c, 3);
    let output_shape = out.shape();
    assert_eq!(h, output_shape[0]);
    assert_eq!(w, output_shape[1]);
    for i in 0..h {
        for j in 0..w {
            let pixel = img.slice(s![i as usize, j as usize, ..]);
            let gray = pixel.into_iter()
                            .zip(&rgb_weights)
                            .map(|(p, w)|(*p) * (*w))
                            .fold(0.0, |acc, x| acc + x);
            out[[i as usize, j as usize]] = gray;
        }
    }
}

/// Transform an RGB image to grayscale image.
/// 
/// The weights are 0.299, 0.587 and 0.114 for red, green and blue respectively.
/// # Example:
/// ```
/// let img_color = ndarray::arr3(&[[[0.0588, 1.0000, 0.4902], [0.0784, 0.9412, 0.4314]]]);
/// let gray = simplecv::color::rgb2gray(&img_color);
/// let max_diff_val = simplecv::utils::max_diff(&gray, &ndarray::arr2(&[[0.6605, 0.6251]]));
/// assert!(max_diff_val < 1e-3);
/// ```
pub fn rgb2gray<A>(img: &ArrayBase<A, Ix3>) -> Array<f64, Ix2> 
    where A:Data<Elem=f64>
{
    let shape = img.shape();
    let h = shape[0];
    let w = shape[1];
    let c = shape[2];
    assert_eq!(c, 3);
    let mut buffer = Array::zeros((h as usize, w as usize));
    rgb2gray_(img, &mut buffer);
    buffer
}

/// Histogram equalization of a grayscale image.
///
/// The output buffer is allocated by users. Implementated following the
/// OpenCV toturial on [histogram equalization](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)
pub fn histeq_<A, B>(img: &ArrayBase<A, Ix2>, out:&mut ArrayBase<B, Ix2>)
    where A: Data<Elem=f64>, B:DataMut<Elem=f64>
{
    let shape = img.shape();
    let h = shape[0] as usize;
    let w = shape[1] as usize;
    fn hist256cdf<A>(img: &ArrayBase<A, Ix2>) -> [f64; 256]
        where A: Data<Elem=f64>
    {
        let mut hist = [0f64; 256];
        for v in img.iter(){
            hist[f2u(*v) as usize] += 1.0;
        }
        for i in 1usize..256usize{
            hist[i] = hist[i-1] + hist[i];
        }
        let maxval = hist[255] as f64;
        for i in 0usize..256usize{
            hist[i] = hist[i] / maxval
        }
        hist
    }
    let img_hist = hist256cdf(img);
    for i in 0usize..h {
        for j in 0usize..w {
            out[[i, j]] = img_hist[(img[[i, j]] * 255.0 + 0.5) as usize];
        }
    }
}

/// Histogram equalization of a grayscale image.
///
/// Implementated following the OpenCV toturial on 
/// [histogram equalization](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)
pub fn histeq<A>(img: &ArrayBase<A, Ix2>) -> Array<f64, Ix2> 
    where A: Data<Elem=f64>
{
    let shape = img.shape();
    let h = shape[0] as usize;
    let w = shape[1] as usize;
    let mut buffer = Array::zeros((h, w));
    histeq_(img, &mut buffer);
    buffer
}
