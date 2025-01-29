#![allow(unused)]

use std::collections::HashSet;
use std::ops::Range;
use std::{collections::HashMap, fs};

use ndarray::Array1;
use rand::thread_rng;

use crate::utils::{create_char_matrix_ndarr, ctoi, get_names, itoc};
use crate::rust_mllib::*;
use crate::numr::*;
use crate::plot::*;


fn ex1() {
    // Andrej switches to dot replacing <S> and <E> we used chars '<', '>'
    let ctoi = ctoi();
    dbg!(&ctoi);
}

fn ex2() {
    let data = create_char_matrix_ndarr();
    plot_data(&data, "data3.png");

    // first row will be all the starting characters - ".a", ".b", etc.
    // first column are all the ending characters - "a.", "b."
}

fn ex3() {
    // let us begin with starting chars i.e. row 0
    let data = create_char_matrix_ndarr();
    
    let row0 = data.row(0);
    // println!("{:?}", row0);
    
    // convert to floats
    let row0f = row0.mapv(|e|e as f64);
    // println!("{:?}", row0f);

    // probability distribution
    let total = row0f.sum();
    let p = row0f.mapv(|e| e/ total);
    // println!("{:?}", p);

    // let samples = multinomial_distrib(100, &p);
    // dbg!(&samples);
    // println!("{}", samples.len());

    // let samples = multinomial_distrib(1, &p);
    // let ctoi = ctoi();
    // let itoc = itoc(&ctoi);
    // // find the first index having 1
    // let x = (0..).zip(samples.iter()).find(|(idx, &value)|value == 1).unwrap().0;
    // println!("{:?}", samples);
    // println!("{}", x);
    // println!("{}", itoc[&x]);

}

fn ex4() {
    let data = create_char_matrix_ndarr();
    let ctoi = ctoi();
    let itoc = itoc(&ctoi);

    let probs = probability_distrib(&data, 0);
    let mut rng = thread_rng();
    let res = multinomial_distrib(1, &probs, &mut rng);
    dbg!(&res);
    let idx = res[0];
    println!("{}", itoc[&idx]);

    // let idx = multinomial_char_index(&probs);
    // println!("char - {}", itoc[&idx]);
}

fn ex5() {
    let data = create_char_matrix_ndarr();
    let ctoi = ctoi();
    let itoc = itoc(&ctoi);
    let mut rng = thread_rng();

    let mut row_num = 0_usize;
    let mut chs = Vec::new();
    loop {
        let probs = probability_distrib(&data, row_num);
        row_num = multinomial_distrib(1, &probs, &mut rng)[0];// remember we get Array so extract
        let ch = itoc[&row_num];
        if row_num == 0 {
            break;
        }
        chs.push(ch);
    }

    let name: String = chs.iter().collect();

    println!("{}", name);

}

fn ex6() {
    // Bigram generation was not that good but better than uniform distribution shown here

    let data = create_char_matrix_ndarr();
    let ctoi = ctoi();
    let itoc = itoc(&ctoi);
    let mut rng = thread_rng();

    let mut row_num = 0_usize;
    let mut chs = Vec::new();
    let probs = ones_distrib(27);
    loop {
        row_num = multinomial_distrib(1, &probs, &mut rng)[0];// remember we get Array so extract
        let ch = itoc[&row_num];
        if row_num == 0 {
            break;
        }
        chs.push(ch);
    }

    let name: String = chs.iter().collect();

    println!("{}", name);

}

fn ex7() {
    let data = create_char_matrix_ndarr();
    let pd = probability_distrib_matrix(&data);
    dbg!(&pd[0]);
}

fn ex8() {
    let data = create_char_matrix_ndarr();
    let pd = probability_distrib_matrix(&data);
    let ctoi = ctoi();
    let itoc = itoc(&ctoi);
    let mut rng = thread_rng();

    let mut row_num = 0_usize;
    let mut chs = Vec::new();
    loop {
        row_num = multinomial_distrib(1, &pd[row_num], &mut rng)[0];// remember we get Array so extract
        let ch = itoc[&row_num];
        if row_num == 0 {
            break;
        }
        chs.push(ch);
    }

    let name: String = chs.iter().collect();

    println!("{}", name);


}

fn ex9() {
    let data = create_char_matrix_ndarr();
    let pd = probability_distrib_matrix(&data);
    println!("{}", &pd[1].sum());
    println!("{:?}", &pd[0]);

}


pub fn main() {
    println!("Notebook 002");
    ex9();
}

// 1:06:26