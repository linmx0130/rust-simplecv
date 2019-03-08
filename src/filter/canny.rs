//! Canny's edge detector

use ndarray::prelude::*;
use ndarray::{Data, DataMut};

use super::{filter_, BorderType, sobel};
use std::collections::VecDeque;

/// The default 5x5 Gaussian kernel for smoothing.
fn get_gaussian_filter() -> Array<f64, Ix2> {
    arr2(&[[2.0, 4.0, 5.0, 4.0, 2.0],
           [4.0, 9.0, 12.0, 9.0, 4.0],
           [5.0, 12.0, 15.0, 12.0, 5.0],
           [4.0, 9.0, 12.0, 9.0, 4.0],
           [2.0, 4.0, 5.0, 4.0, 2.0]]) / 159.0
}

/// Obtaining direction array for NMS. The arguments are Gx and Gy.
///
/// Making decision with tan(\theta). The direction definition:
/// * 0: -90 ~ -67.5 and  67.5 ~ 90 (vertical)
/// * 1: -67.5 ~ -22.5 (bottom part) 
/// * 2: -22.5 ~ 22.5 (middle part) 
/// * 3: 22.5 ~ 67.5 (top part)
fn obtain_direction<S>(gx: &ArrayBase<S, Ix2>, gy: &ArrayBase<S, Ix2>) -> Array<i32, Ix2> 
    where S: Data<Elem=f64>
{
    let height: usize = gx.shape()[0];
    let width: usize = gy.shape()[1];
    let tan_225 = 0.414213562373095;
    let tan_675 = 2.414213562373095;
    let eps = 1e-7;
    let mut dir = Array::zeros((gx.shape()[0], gx.shape()[1]));
    for i in 0usize..height {
        for j in 0usize..width{
            let tan_theta = gy[[i, j]] / (gx[[i, j]] + eps);
            if tan_theta < 0.0 {
                if tan_theta < - tan_675 {
                    dir[[i, j]] = 0;
                } else {
                    if tan_theta < -tan_225 {
                        dir[[i, j]] = 1;
                    } else {
                        dir[[i, j]] = 2;
                    }
                }
            } else {
                if tan_theta > tan_675 {
                    dir[[i, j]] = 0;
                } else {
                    if tan_theta < tan_225 {
                        dir[[i, j]] = 2;
                    } else {
                        dir[[i, j]] = 3;
                    }
                }
            }
        }
    }
    dir
}
fn clip_pixel(v:f64) -> f64 {
    f64::min(f64::max(v, 0.0), 1.0)
}
// Perform NMS for edges.
fn edge_nms<S, T>(dir: &ArrayBase<S, Ix2>, out: &mut ArrayBase<T, Ix2>)
    where S: Data<Elem=i32>, T:DataMut<Elem=f64> 
{
    let height: usize = dir.shape()[0];
    let width: usize = dir.shape()[1];
    for i in 0usize..height {
        for j in 0usize..width {
            let v = out[[i, j]];
            out[[i, j]] = clip_pixel(
                match dir[[i, j]] {
                    0 => {
                        // vertical
                        if (i==0) || (i == height-1) {
                            0.0
                        } else {
                            if v > out[[i-1, j]] && v > out[[i+1, j]] {
                                v
                            } else {
                                0.0
                            }
                        }
                    }
                    1 => {
                        //down part
                        if (i==0) || (i == height-1) || (j==0) || (j == width-1) {
                            v
                        } else {
                            if v > out[[i-1, j-1]] && v > out[[i+1, j+1]] {
                                v
                            } else {
                                0.0
                            }
                        }
                    }
                    2 => {
                        //middle part
                        if (j==0) || (j == width-1) {
                            0.0
                        } else {
                            if v > out[[i, j-1]] && v > out[[i, j+1]] {
                                v
                            } else {
                                0.0
                            }
                        }
                    }
                    3 => {
                        //top part
                        if (i==0) || (i == height-1) || (j==0) || (j == width-1) {
                            0.0
                        } else {
                            if v > out[[i-1, j+1]] && v > out[[i, j-1]] {
                                v
                            } else {
                                0.0
                            }
                        }
                    }
                    _ => panic!("Unexpected direction in Canny edge detector!")
                }
            );  
        }
    }
}
/// Suppress all values lower than `min_val` and keep left values which are larger 
/// than `max_val` or connected to large values.
fn max_min_suppression<S>(max_val: f64, min_val: f64, out: &mut ArrayBase<S, Ix2>)
    where S: DataMut<Elem=f64>
{
    assert!(max_val >= min_val);
    let height: usize = out.shape()[0];
    let width: usize = out.shape()[1];
    let mut connected_check_buffer = Array::zeros((height, width));
    let mut queue: VecDeque<usize> = VecDeque::new();
        // check connectivity for all values >= max_val
    for i in 0usize..height {
        for j in 0usize..width {
            if connected_check_buffer[[i, j]] == 0 {
                if out[[i, j]] >= max_val {
                    queue.push_back( i * width + j);
                    connected_check_buffer[[i, j]] = 1;
                    loop {
                        match queue.pop_front() {
                            Some(f) => {
                                let x = (f / width) as i32;
                                let y = (f % width) as i32;
                                for dx in -1..1 {
                                    for dy in -1..1 {
                                        let nx = x + dx;
                                        let ny = y + dy;
                                        if nx < 0 || nx >= (height as i32) || ny < 0 || ny >= (width as i32) {
                                            continue;
                                        }
                                        let nx = nx as usize;
                                        let ny = ny as usize;
                                        if out[[nx, ny]] >= min_val && connected_check_buffer[[nx, ny]] == 0{
                                            connected_check_buffer[[nx, ny]] = 1;
                                            queue.push_back(nx * width + ny);
                                        }
                                    }
                                }
                            }
                            None => {break;}
                        }
                    }
                }
            }
        }
    }
    for i in 0usize..height {
        for j in 0usize..width {
            if connected_check_buffer[[i, j]] == 0{
                out[[i, j]] = 0.0;
            }
        }
    }
}

/// Get histogram of a given bin size. All values in `src` are required to be in [0, 1]
fn get_histogram<S>(src: &ArrayBase<S, Ix2>, bin_size: usize) -> Vec<f64> 
    where S: Data<Elem=f64>
{
    let mut bins = vec![0f64; bin_size+1];
    let mut sum = 0.0;
    let height = src.shape()[0];
    let width = src.shape()[1];
    for i in 0usize..height {
        for j in 0usize..width {
            assert!(src[[i, j]] >= 0.0 && src[[i, j]] <=1.0);
            let v = src[[i, j]] * (bin_size as f64);
            let idx = v as usize;
            bins[idx] = bins[idx] + 1.0;
            sum = sum + 1.0;
        }
    }
    bins[bin_size - 2] = bins[bin_size - 1] + bins[bin_size - 2];
    bins.truncate(bin_size);
    for i in 0usize..bin_size {
        bins[i] = bins[i] / sum;
    }
    bins
}
/// Simply Canny's edge detector. Output buffer is allocated by users.
///
/// * `src`: input image 
/// * `max_val_percent`: the ratio of strong edge.
/// * `min_val_percent`: the ratio of noise (smaller than weak edge).
/// * `out`: output buffer.
///
/// Refered to [canny_edge()](./fn.canny_edge.html) for more details.
pub fn canny_edge_<S, T>(src: &ArrayBase<S, Ix2>, max_val_percent: f64, min_val_percent: f64, border:BorderType, out: &mut ArrayBase<T, Ix2>)
    where S: Data<Elem=f64>, T:DataMut<Elem=f64>
{
    // smooth the image, use out as the buffer
    filter_(src, &get_gaussian_filter(), border, out);
    // obtain gradients
    let gx = sobel(&out, 3, 1, 0, border);
    let gy = sobel(&out, 3, 0, 1, border);
    let gnorm = gx.mapv(|x| x.powi(2)) + gy.mapv(|x| x.powi(2));
    // put norm of gradient to out, which is almost the final result
    out.assign(&gnorm.mapv(f64::sqrt));
    let dir = obtain_direction(&gx, &gy);
    //non-maximum suppression
    edge_nms(&dir, out);
    //estimate min/max val
    let edge_hist = get_histogram(src, 100);
    let mut max_val_left = max_val_percent;
    let mut min_val_left = min_val_percent;
    let mut max_val = 1.0;
    let mut min_val = 0.0;
    for i in 0usize..99 {
        max_val_left -= edge_hist[99usize - i];
        max_val -= 0.01;
        if max_val_left <=0.0 { break; }
    }
    for i in 0usize..99 {
        min_val_left -= edge_hist[i];
        min_val += 0.01;
        if min_val_left <=0.0 { break; }
    }
    if min_val > max_val{
        let k = min_val;
        min_val = max_val;
        max_val = k;
    }
    //suppress weak edges
    max_min_suppression(max_val, min_val, out);
    // binarization
    let height = src.shape()[0];
    let width = src.shape()[1];
    for i in 0usize..height {
        for j in 0usize..width {
            if out[[i, j]] > 0.0 {
                out[[i, j]] = 1.0;
            }
        }
    }

}

/// Simply Canny's edge detector.
///
/// * `src`: input image 
/// * `max_val_percent`: the ratio of strong edge.
/// * `min_val_percent`: the ratio of noise (smaller than weak edge).
///
/// This functions requires that `max_val_percent` + `min_val_percent` <= 1.0. 
/// The implementation follows [Canny edge
/// detector](https://en.wikipedia.org/wiki/Canny_edge_detector), while 
/// [fast-edge](https://code.google.com/archive/p/fast-edge/) is also referred.
pub fn canny_edge<S>(src: &ArrayBase<S, Ix2>, max_val_percent:f64, min_val_percent:f64, border:BorderType) -> Array<f64, Ix2>
    where S: Data<Elem=f64>
{
    let height: usize = src.shape()[0];
    let width: usize = src.shape()[1];
    let mut out = Array::zeros((height, width));
    canny_edge_(src, max_val_percent, min_val_percent, border, &mut out);
    out
}    
