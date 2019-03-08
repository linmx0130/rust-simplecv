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
    let edge = canny_edge(&lenna, 0.5, 0.05, BorderType:: Reflect);
    imsave_gray(&edge, "canny.png");
}
