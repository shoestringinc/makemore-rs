#![allow(unused)]

use std::collections::HashSet;
use std::f64::consts::E;
use std::ops::Range;
use std::{collections::HashMap, fs};

use ndarray::{Array1, Array2};
use rand::thread_rng;

use std::sync::OnceLock;

use crate::utils::{create_char_matrix_ndarr, ctoi, get_names, itoc};
use crate::rust_mllib::*;
use crate::numr::*;
use crate::plot::*;

static WORDS: OnceLock<Vec<String>> = OnceLock::new();
static DATA: OnceLock<Array2<i64>> = OnceLock::new();
static PD: OnceLock<Vec<Array1<f64>>> = OnceLock::new();
static CTOI: OnceLock<HashMap<char, usize>> = OnceLock::new();
static ITOC: OnceLock<HashMap<usize, char>> = OnceLock::new();


fn init_core_data() {
    WORDS.get_or_init(|| get_names());
    DATA.get_or_init(|| create_char_matrix_ndarr());
    let data = DATA.get().unwrap();
    PD.get_or_init(|| probability_distrib_matrix(data));
    CTOI.get_or_init(||ctoi());
    let ctoi = CTOI.get().unwrap();
    ITOC.get_or_init(||itoc(ctoi));
}

fn ex1() {
    let words = WORDS.get().unwrap();
    let data = DATA.get().unwrap();
    let pd = PD.get().unwrap();
    let ctoi = CTOI.get().unwrap();
    let itoc = ITOC.get().unwrap();
}

fn ex2() {
    let words = WORDS.get().unwrap();
    let data = DATA.get().unwrap();
    let pd = PD.get().unwrap();
    let ctoi = CTOI.get().unwrap();
    let itoc = ITOC.get().unwrap();

    let mut log_likelihood = 0.0;
    let mut n = 0_usize;

    for w in &words[..3] {
        let chs = format!(".{}.", w);
        for (ch1, ch2) in chs.chars().zip(chs.chars().skip(1)) {
            let ix1 = ctoi[&ch1];
            let ix2 = ctoi[&ch2];
            let prob = &pd[ix1][ix2];
            let logprob = prob.log(E); // natural log - base `e`
            log_likelihood += logprob;
            n += 1;
            // println!("{}{}: {:.4} {:.4}", ch1, ch2, prob, logprob);
        }
    }
    // maximize the probs which are model params
    // we maximize the log_likelihood (log is monotonic - lease negative)
    // equivalent to minimizing the nll
    // equivalent to minimizing the avg nll - lowest 0
    println!("{}", log_likelihood); // log likelihood is 0 when prob 1. Otherwise grows to negative inf as prob decreases.
    let nll = -log_likelihood; // higher it is bad. near 0 best
    println!("{nll}");
    let avg_nll = nll / (n as f64); // lower is better
    println!("{}", avg_nll);
}

fn ex3() {
    let words = WORDS.get().unwrap();
    let data = DATA.get().unwrap();
    let pd = PD.get().unwrap();
    let ctoi = CTOI.get().unwrap();
    let itoc = ITOC.get().unwrap();

    let mut log_likelihood = 0.0;
    let mut n = 0_usize;

    // just adding q - make the loss function `inf`
    for w in ["andrejq"] {
        let chs = format!(".{}.", w);
        for (ch1, ch2) in chs.chars().zip(chs.chars().skip(1)) {
            let ix1 = ctoi[&ch1];
            let ix2 = ctoi[&ch2];
            let prob = &pd[ix1][ix2];
            let logprob = prob.log(E); // natural log - base `e`
            log_likelihood += logprob;
            n += 1;
            println!("{}{}: {:.4} {:.4}", ch1, ch2, prob, logprob);
        }
    }
    println!("{}", log_likelihood); // log likelihood is 0 when prob 1. Otherwise grows to negative inf as prob decreases.
    let nll = -log_likelihood; // higher it is bad. near 0 best
    println!("{nll}");
    let avg_nll = nll / (n as f64); // lower is better
    println!("{}", avg_nll);

}

fn ex3_1() {
    let data = DATA.get().unwrap();
    plot_data(data, "data_ndarr.png");

}

fn ex4() {
    let words = WORDS.get().unwrap();
    let data = DATA.get().unwrap();
    let pd = PD.get().unwrap();
    let ctoi = CTOI.get().unwrap();
    let itoc = ITOC.get().unwrap();

    // println!("{:?}", data[[0,1]]);
    // println!("{:?}", data.row(0));

    // model smoothing
    // so that in our avg_nll we don't get infinity
    // respectable bigram character level langaueg model
    let data = data.mapv(|e| e+ 1); 
    let pd = probability_distrib_matrix(&data);

    let mut log_likelihood = 0.0;
    let mut n = 0_usize;

    // just adding q - make the loss function `inf`
    for w in ["andrejq"] {
        let chs = format!(".{}.", w);
        for (ch1, ch2) in chs.chars().zip(chs.chars().skip(1)) {
            let ix1 = ctoi[&ch1];
            let ix2 = ctoi[&ch2];
            let prob = &pd[ix1][ix2];
            let logprob = prob.log(E); // natural log - base `e`
            log_likelihood += logprob;
            n += 1;
            println!("{}{}: {:.4} {:.4}", ch1, ch2, prob, logprob);
        }
    }
    println!("{}", log_likelihood); // log likelihood is 0 when prob 1. Otherwise grows to negative inf as prob decreases.
    let nll = -log_likelihood; // higher it is bad. near 0 best
    println!("{nll}");
    let avg_nll = nll / (n as f64); // lower is better
    println!("{}", avg_nll);


}

pub fn main() {
    println!("Notebook 002");
    init_core_data();
    ex3_1();
}

// 56:17