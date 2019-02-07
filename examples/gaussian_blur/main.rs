#[macro_use]
extern crate ndarray;
extern crate simplecv;

use ndarray::prelude::*;
use simplecv::io::*;
use simplecv::color::*;
use simplecv::filter::*;

fn main() {
    let lenna = imread("lenna.png");
    let mut buffer = Array::zeros((512, 512, 3));
    for c in 0..3 {
        gaussian_smooth_(&lenna.slice(s![..,..,c as usize]), 7, BorderType::Reflect, &mut buffer.slice_mut(s![..,..,c as usize]));
    }
    imsave(&buffer, "blur.png");
}
