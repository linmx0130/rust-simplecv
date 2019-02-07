//! Some wrapper functions for image IO operations. 
//!
//! Since `simplecv` is based on `ndarray`, these functions use `image` 
//! crate to read images and store data in `ndarray::Array`.
use image::*;
use ndarray::prelude::*;
use super::utils::f2u;

/// Read an image file into an array.
/// 
/// The return value is a 3D array of f64, in which all values are between 0 to 1.
pub fn imread(filename: &str) -> ndarray::Array<f64, Ix3> {
    let img = image::open(filename).expect("Read image failed!");
    let (img_height, img_width) = img.dimensions();
    let mut buffer = Array::zeros((img_height as usize, img_width as usize, 3));
    for u in img.pixels() {
        let (x, y, color) = u;
        for c in 0..3 {            
            buffer[[x as usize, y as usize, c as usize]] = color.data[c] as f64 / 255.0
        }
    }
    buffer
}
/// Save an RGB image to an file.
///
/// The argument must be a 3D array, in which all values must be in \[0.0, 1.0\].
pub fn imsave(img: &Array<f64, Ix3>, filename: &str) {
    let shape = img.shape();
    let height = shape[0] as u32;
    let width = shape[1] as u32;
    let mut buffer = image::ImageBuffer::new(height, width);
    for (x, y, pixel) in buffer.enumerate_pixels_mut() {
        let val = img.slice(s![x as usize, y as usize, ..]);
        assert_eq!(val.len(), 3);
        *pixel = image::Rgb([f2u(val[0]), f2u(val[1]), f2u(val[2])]);
    }
    buffer.save(filename).expect("Error in saving image!");
}

/// Save an grayscale image to an file.
///
/// The argument must be a 2D array, in which all values must be in \[0.0, 1.0\].
pub fn imsave_gray(img: &Array<f64, Ix2>, filename: &str) {
    let shape = img.shape();
    let height = shape[0] as u32;
    let width = shape[1] as u32;
    let mut buffer = image::ImageBuffer::new(height, width);
    for (x, y, pixel) in buffer.enumerate_pixels_mut() {
        let val = f2u(img[[x as usize, y as usize]]);
        *pixel = image::Rgb([val, val, val]);
    }
    buffer.save(filename).expect("Error in saving image!");
}
