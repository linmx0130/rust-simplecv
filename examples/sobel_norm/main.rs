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
    let gnorm = sobel_norm(&lenna, 3, -1, BorderType:: Replicate);
    imsave_gray(&gnorm, "sobel_norm.png");
}
