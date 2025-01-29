#![allow(unused)]

use std::collections::HashSet;
use std::f64::consts::E;
use std::ops::Range;
use std::{collections::HashMap, fs};

use ndarray::{Array1, Array2};
use rand::thread_rng;

use std::sync::OnceLock;

use crate::utils::{create_char_matrix_ndarr, create_char_matrix_tensor, ctoi, get_names, itoc};
use crate::rust_mllib::*;
use crate::numr::*;
use crate::plot::*;

// Ignore this notebook
// We were planning to rewrite all the cells using Burn Tensor rather than ndarray crate
// But for fast rewrite we will continue till probably we reach auto diff position
// Go to notebook 5

fn ex1() {
    let words = get_names();
    let ctoi = ctoi();
    let itoc = itoc(&ctoi);

    let mut xs: Vec<f64> = vec![]; // input char
    let mut ys: Vec<f64> = vec![]; // desired char

    for w in &words[..1] {
        let chs = format!(".{}.", w);
        for (ch1, ch2) in chs.chars().zip(chs.chars().skip(1)) {
            let ix1 = ctoi[&ch1];
            let ix2 = ctoi[&ch2];
            println!("{} {}", ch1, ch2);
            xs.push(ix1 as f64);
            ys.push(ix2 as f64);
        }
    }

    let xs = tensor1df(&xs);
    let ys = tensor1df(&ys);
    println!("{}", &xs);
    let l = xs.dims();
    println!("{}", l[0]);

    println!("length of xs: {}", len1df(&xs));

    let onehot = one_hot_tensor(&xs, 27);
    println!("{}", onehot);
}

fn ex2() {
    // Use the struct TensorInt and TensorFloat rather than the functions - much easier
    let matrix = create_char_matrix_tensor();
    // println!("{}", matrix);
    let tensor = matrix.tensor();
    let t1 = tensor.clone().slice([0..1, 0..27]);
    // println!("{}", t1);
    let t2 = tensor.clone().slice([0..1, 1..2]);
    println!("{}", t2);
}

pub fn main() {
    println!("Notebook 004");
    ex2();

}

// 53:38