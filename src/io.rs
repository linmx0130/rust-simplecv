//! Some wrapper functions for image IO operations. 
//!
//! Since `simplecv` is based on `ndarray`, these functions use `image` 
//! crate to read images and store data in `ndarray::Array`.
use image::*;
use ndarray::prelude::*;

/// Read an image file into an array.
/// 
/// The return value is a 3D array of u8.
pub fn imread(filename: &str) -> ndarray::Array<u8, Ix3> {
    let img = image::open(filename).expect("Read image failed!");
    let (img_height, img_width) = img.dimensions();
    let mut buffer = Array::zeros((img_height as usize, img_width as usize, 3));
    for u in img.pixels() {
        let (x, y, color) = u;
        for c in 0..3 {            
            buffer[[x as usize, y as usize, c as usize]] = color.data[c]
        }
    }
    buffer
}
/// Save an RGB image to an file.
///
/// The argument must be a 3D array.
pub fn imsave(img: &Array<u8, Ix3>, filename: &str) {
    let shape = img.shape();
    let height = shape[0] as u32;
    let width = shape[1] as u32;
    let mut buffer = image::ImageBuffer::new(height, width);
    for (x, y, pixel) in buffer.enumerate_pixels_mut() {
        let val = img.slice(s![x as usize, y as usize, ..]);
        assert_eq!(val.len(), 3);
        *pixel = image::Rgb([val[0], val[1], val[2]]);
    }
    buffer.save(filename).expect("Error in saving image!");
}

/// Save an grayscale image to an file.
///
/// The argument must be a 2D array.
pub fn imsave_gray(img: &Array<u8, Ix2>, filename: &str) {
    let shape = img.shape();
    let height = shape[0] as u32;
    let width = shape[1] as u32;
    let mut buffer = image::ImageBuffer::new(height, width);
    for (x, y, pixel) in buffer.enumerate_pixels_mut() {
        let val = img[[x as usize, y as usize]];
        *pixel = image::Rgb([val, val, val]);
    }
    buffer.save(filename).expect("Error in saving image!");
}
