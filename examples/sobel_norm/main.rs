#[macro_use]
extern crate ndarray;
extern crate simplecv;

use ndarray::prelude::*;
use simplecv::io::*;
use simplecv::color::*;
use simplecv::filter::*;

fn main() {
    let lenna = imread("lenna.png");
    let lenna = rgb2gray(&lenna);
    let lenna = gaussian_smooth(&lenna, 7, BorderType::Reflect);
    let gnorm = sobel_norm(&lenna, 3, -1, BorderType:: Reflect);
    imsave_gray(&gnorm, "sobel_norm.png");
}
