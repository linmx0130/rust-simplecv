//! `simple-cv` is a pure-Rust computer vision toolkit with a simple interface
//! similar to MATLAB or OpenCV. All code are based on `ndarray` package in 
//! order to make computer vision simpler in Rust!
//!
#[macro_use]
extern crate ndarray;

pub mod utils;
pub mod io;
pub mod color;
pub mod filter;
