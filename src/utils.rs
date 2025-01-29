#![allow(unused)]

use ndarray::Array2;
use std::collections::{HashMap, HashSet};
use std::fs;

use rand::distributions::Uniform;
use rand::Rng;

use crate::numr::*;
use crate::rust_mllib::{IntTensor2d, TensorInt};

pub fn get_names() -> Vec<String> {
    let names = fs::read_to_string("names.txt").unwrap();
    let names: Vec<_> = names.lines().map(|e| e.to_string()).collect();
    names
}

fn old_ctoi() -> HashMap<char, usize> {
    let names = get_names();
    let mut chs: HashSet<char> = names.join("").chars().collect();
    let mut chs = Vec::from_iter(chs);
    chs.sort();
    chs.push('<');
    chs.push('>');
    let xs: HashMap<char, usize> = (0..).zip(chs).map(|e| (e.1, e.0)).collect();
    xs
}

pub fn ctoi() -> HashMap<char, usize> {
    let names = get_names();
    let mut chs: HashSet<char> = names.join("").chars().collect();
    let mut chs = Vec::from_iter(chs);
    chs.sort();
    let mut chs1 = vec!['.']; // we need this to be at index 0
    chs1.extend_from_slice(&chs);
    let xs: HashMap<char, usize> = (0..).zip(chs1).map(|e| (e.1, e.0)).collect();
    xs
}

pub fn itoc(ctoi: &HashMap<char, usize>) -> HashMap<usize, char> {
    let h = ctoi.iter().map(|(&k, &v)| (v, k)).collect();
    h
}

// Returns an NdArray object not Burn Tensor
// pub fn create_char_matrix_ndarr() -> Array2<i64> {
//     let ctoi = ctoi();
//     let num_chars = ctoi.len();
//     let mut n = zeros_int(num_chars, num_chars);
//     let names = get_names();
//     for w in &names {
//         let w = format!(".{}.", w);
//         for (ch1, ch2) in w.chars().zip(w.chars().skip(1)) {
//             let ix1 = ctoi[&ch1];
//             let ix2 = ctoi[&ch2];
//             n[[ix1, ix2]] += 1;
//         }
//     }
//     n
// }

pub fn create_char_matrix_ndarr() -> Array2<i64> {
    let matrix_vec = create_char_matrix_vec();
    let num_rows = matrix_vec.len();
    let num_cols = matrix_vec[0].len();
    let raw_data: Vec<i64> = matrix_vec.into_iter().flatten().collect();
    Array2::from_shape_vec((num_rows, num_cols), raw_data).unwrap()
}

pub fn create_char_matrix_tensor() -> TensorInt<2> {
    let matrix_vec = create_char_matrix_vec();
    let num_rows = matrix_vec.len();
    let num_cols = matrix_vec[0].len();
    let raw_data: Vec<i64> = matrix_vec.into_iter().flatten().collect();
    let t = TensorInt::<2>::new(&raw_data, &[num_rows, num_cols]);
    t
}

pub fn create_char_matrix_vec() -> Vec<Vec<i64>> {
    let ctoi = ctoi();
    let num_chars = ctoi.len();
    let words = get_names();

    // create z zeroed 2d vector
    let mut matrix: Vec<Vec<i64>> = vec![vec![0; num_chars]; num_chars];

    for w in &words {
        let w = format!(".{}.", w);
        for (ch1, ch2) in w.chars().zip(w.chars().skip(1)) {
            let ix1 = ctoi[&ch1];
            let ix2 = ctoi[&ch2];
            matrix[ix1][ix2] += 1;
        }
    }

    matrix
}

pub fn rand_uniform_nums(n: usize, low: f64, high: f64) -> Vec<f64> {
    let mut rng = rand::thread_rng();

    let uniform_range = Uniform::new(low, high);

    // 0..n is fine too :-). Just that I like this more. Sounds natural!
    (1..=n).map(|_| rng.sample(uniform_range)).collect()
}

pub fn rand_normal_distrib(n: usize) -> Vec<f64> {
    rand_uniform_nums(n, -3.1, 3.1)
}
