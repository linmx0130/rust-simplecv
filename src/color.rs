use ndarray::prelude::{Array, Ix2, Ix3};

/// Transform an RGB image to grayscale image.
/// 
/// The weights are 0.299, 0.587 and 0.114 for red, green and blue respectively.
/// # Example:
/// ```
/// let img_color = ndarray::arr3(&[[[15, 255, 125], [20, 240, 110]]]);
/// let gray = simplecv::color::rgb2gray(&img_color);
/// assert_eq!(gray, ndarray::arr2(&[[168, 159]]));
/// ```
pub fn rgb2gray(img: &Array<u8, Ix3>) -> Array<u8, Ix2> {
    let shape = img.shape();
    let h = shape[0];
    let w = shape[1];
    let c = shape[2];
    let rgb_weights = [0.299, 0.587, 0.114];
    assert_eq!(c, 3);
    let mut buffer = Array::zeros((h as usize, w as usize));
    for i in 0..h {
        for j in 0..w {
            let pixel = img.slice(s![i as usize, j as usize, ..]);
            let gray = pixel.into_iter()
                            .zip(&rgb_weights)
                            .map(|(p, w)|(*p) as f64 * (*w))
                            .fold(0.0, |acc, x| acc + x);
            buffer[[i as usize, j as usize]] = (gray + 0.5) as u8;
        }
    }
    buffer
}

