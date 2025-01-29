#![allow(unused)]

// This is like Numpy equivalent - just wrappers using ndarray crate.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use ndarray::prelude::*;
use ndarray::{stack, Array, Array1, Array2, Axis};

use rand::distributions::WeightedIndex;
use rand::prelude::*;

use num_traits::{Bounded, Num, ToPrimitive};

use crate::utils::rand_normal_distrib;

pub fn arange(start: f64, stop: f64, step: f64) -> Array1<f64> {
    Array::range(start, stop, step)
}

pub fn max_array1(xs: &Array1<f64>) -> f64 {
    xs.iter().cloned().fold(f64::MIN, |a, b| a.max(b))
}

pub fn min_array1(xs: &Array1<f64>) -> f64 {
    xs.iter().cloned().fold(f64::MAX, |a, b| a.min(b))
}

pub fn zeros_float(num_rows: usize, num_cols: usize) -> Array2<f64> {
    Array2::zeros((num_rows, num_cols))
}

pub fn zeros_int(num_rows: usize, num_cols: usize) -> Array2<i64> {
    Array2::zeros((num_rows, num_cols))
}

// pub fn multinomial_distrib(num_samples: usize, input: &ArrayView1<f64>, rng: &mut ThreadRng) -> Array1<usize> {
pub fn multinomial_distrib(
    num_samples: usize,
    input: &Array1<f64>,
    rng: &mut ThreadRng,
) -> Array1<usize> {
    // let input = Array1::from_vec(vec![0.2, 0.5, 0.3]);
    // let num_samples = 10;

    // pass the weights
    let dist = WeightedIndex::new(input.to_vec()).unwrap();

    let mut samples = vec![0_usize; num_samples];
    for i in 0..num_samples {
        samples[i] = dist.sample(rng);
    }

    Array1::from(samples)

    // let mut rng = thread_rng();

    // Sample from the multinomial distribution
    // let mut counts = vec![0usize; input.len() ];
    // for _ in 0..num_samples {
    //     let sample = dist.sample(rng);
    //     counts[sample] += 1;
    // }

    // println!("{:?}", counts);

    // let xs = Array1::from(counts);
    // xs
}

// pub fn multinomial_char_index(input: &Array1<f64>) -> usize {
//     let samples = multinomial_distrib(1, input);
//     let idx = (0usize..).zip(samples.iter()).find(|(idx, &value)|value == 1).unwrap().0;
//     idx
// }

pub fn probability_distrib<T>(data: &Array2<T>, row_num: usize) -> Array1<f64>
where
    T: Copy + Num + Bounded + ToPrimitive + PartialOrd,
{
    let row_n = data.row(row_num);
    let row_n_f = row_n.mapv(|e| (e.to_f64()).unwrap());
    let total = row_n_f.sum();

    let probabilities = row_n_f.mapv(|e| e / total);
    probabilities
}

pub fn probability_distrib_matrix<T>(data: &Array2<T>) -> Vec<Array1<f64>>
where
    T: Copy + Num + Bounded + ToPrimitive + PartialOrd,
{
    let data = data.mapv(|e| (e.to_f64()).unwrap());

    let mut pd_vec = Vec::new();

    for row in data.rows() {
        let total = row.sum();
        let probs = row.mapv(|e| e / total);
        pd_vec.push(probs);
    }

    // let xs = Array2::from_shape_vec(
    //     (data.nrows(), data.ncols()),
    //     pd_vec.into_iter().flatten().collect(),
    // ).unwrap();

    // let mut res = Vec::new();

    pd_vec
}

pub fn ones_distrib(num_cols: usize) -> Array1<f64> {
    // let x: Array1<i32> = Array1::<i32>::ones(27);
    // let ones = Array::<f64, _>::ones((1, num_cols)); // dim is 2
    // let ones = ones.row(0);
    let ones: Array1<f64> = Array1::<f64>::ones(num_cols);
    let d = num_cols as f64;
    let p = ones.mapv(|n| n / d);
    p
}

// Return type could have been u8 too!
pub fn one_hot_vec(xs: &[usize], num_classes: usize) -> Vec<Vec<f64>> {
    let mut res = Vec::new();
    for &x in xs {
        let mut v = vec![0.0_f64; num_classes];
        v[x] = 1.0;
        res.push(v);
    }
    res
}

pub fn one_hot(xs: &[usize], num_classes: usize) -> Array2<f64> {
    let v = one_hot_vec(xs, num_classes);
    vec_to_ndarr2(&v, v.len(), v[0].len())
}

pub fn vec_to_ndarr2<T>(xs: &Vec<Vec<T>>, num_rows: usize, num_cols: usize) -> Array2<T>
where
    T: Copy + Num + Bounded + ToPrimitive + PartialOrd,
{
    let v: Vec<T> = xs.iter().cloned().flatten().collect();
    // let v1: Vec<i64> = v.iter().map(|e| (e.to_i64()).unwrap()).collect();
    Array2::from_shape_vec((num_rows, num_cols), v).unwrap()
}

pub fn randn(num_rows: usize, num_cols: usize) -> Array2<f64> {
    let total_elems = num_rows * num_cols;
    let xs = rand_normal_distrib(total_elems);
    Array2::from_shape_vec((num_rows, num_cols), xs).unwrap()
}

pub fn probabiity_row_wise(a: &Array2<f64>) -> Array2<f64> {
    let mut xs = Vec::new();
    for row in a.rows() {
        let sum = row.sum();
        let new_row = row.mapv(|e| e / sum);
        xs.push(new_row);
    }
    let xs: Vec<_> = xs.iter().map(|r| r.view()).collect();
    let a1 = stack(Axis(0), &xs).unwrap();
    a1
}

fn explore1() {
    let mut rng = thread_rng();
    let input = Array1::from_vec(vec![0.2, 0.5, 0.3]);
    let res = multinomial_distrib(100, &input, &mut rng);
    dbg!(&res);
}

fn explore2() {
    let array = Array2::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6]).unwrap();

    let new_array = Array2::from_shape_vec(
        (2, 3),
        array
            .axis_iter(Axis(0))
            .flat_map(|row| row.mapv(|x| x * 2))
            .collect(),
    )
    .unwrap();

    dbg!(new_array);
}

fn explore3() {
    let array = Array2::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6]).unwrap();
    let shape = array.shape();
    let (num_rows, num_cols) = (shape[0], shape[1]);
    println!("row: {}, cols: {}", num_rows, num_cols);
}

pub fn main() {
    println!("NumRust explore");
    explore3();
}
