#![allow(unused)]

use std::collections::HashSet;
use std::ops::Range;
use std::{collections::HashMap, fs};

use crate::numr::*;
use crate::plot::*;
use crate::rust_mllib::*;
use crate::utils::{create_char_matrix_ndarr, ctoi, get_names};

fn ex1() {
    let names = fs::read_to_string("names.txt").unwrap();
    let names: Vec<_> = names.lines().collect();
    // NOTE: we have a get_names() function in utils module to save some typing in cells.
    // println!("{:#?}", &names);

    let xs = &names[0..10]; // first 10 names
                            // dbg!(&xs);

    println!("Total names: {}", names.len());

    // shortest word
    let shortest = names.iter().map(|e| e.len()).min().unwrap();
    println!("shortest length: {}", shortest);

    // longest word
    let longest = names.iter().map(|e| e.len()).max().unwrap();
    println!("longest length: {}", longest);
}

fn ex2() {
    let names = get_names();
    // println!("{:#?}", &names);

    for w in &names[0..3] {
        println!("{w}");
        for (ch1, ch2) in w.chars().zip(w.chars().skip(1)) {
            println!("{ch1} {ch2}");
        }
        println!("===");
    }
}

fn ex3() {
    let names = get_names();

    let mut b: HashMap<(char, char), usize> = HashMap::new();
    // for w in &names[0..3] {
    for w in &names {
        let w = format!("<{}>", w);
        for (ch1, ch2) in w.chars().zip(w.chars().skip(1)) {
            let bigram = (ch1, ch2);
            b.entry(bigram).and_modify(|e| *e += 1).or_insert(1);
            // println!("{ch1} {ch2}");
        }
    }

    // println!("{:#?}", &b);

    let mut sorted_entries: Vec<_> = b.iter().collect();
    sorted_entries.sort_by(|a, b| b.1.cmp(a.1));
    sorted_entries
        .iter()
        .take(10)
        .for_each(|e| println!("{:?}", e));
}

fn ex4() {
    // Rough work: Testing Rust stuff
    let a = 1;
    let b = 2;
    println!("{:?}", a.cmp(&b));
    println!("{:?}", b.cmp(&a));
    let mut xs = vec![a, b];
    xs.sort_by(|x, y| x.cmp(y));
    println!("{:?}", xs);
    let mut xs = vec![a, b];
    xs.sort_by(|x, y| y.cmp(x));
    println!("{:?}", xs);
}

fn ex5() {
    let mut a = zeros_int(3, 5);
    dbg!(&a);
    a[[1, 3]] = 1;
    dbg!(&a);
    a[[1, 3]] += 1;
    dbg!(&a);
    a[[0, 0]] = 5;
    dbg!(&a);
}

fn ex6() {
    let names = get_names();
    let mut chs: HashSet<char> = names.join("").chars().collect();
    // println!("{chs:?}");
    // println!("{}", chs.len());
    let mut chs = Vec::from_iter(chs);
    chs.sort();
    chs.push('<');
    chs.push('>');
    println!("{chs:?}");
    let xs: HashMap<char, usize> = (0..).zip(chs).map(|e| (e.1, e.0)).collect();
    dbg!(&xs);

    // we have refactored this into a function calle ctoi in utils

    // let a = zeros_int(28, 28);
}

fn ex7() {
    let ctoi = ctoi();
    let mut n = zeros_int(28, 28);
    let names = get_names();
    for w in &names {
        let w = format!("<{}>", w);
        for (ch1, ch2) in w.chars().zip(w.chars().skip(1)) {
            let ix1 = ctoi[&ch1];
            let ix2 = ctoi[&ch2];
            n[[ix1, ix2]] += 1;
        }
    }
    // dbg!(&n);
    plot_heat_map(&n, "heat.png");
}

fn ex8() {
    let ctoi = ctoi();
    let itoc: HashMap<usize, char> = ctoi.iter().map(|(&k, &v)| (v, k)).collect();
    println!("{:?}", itoc);
}

fn ex9() {
    let data = create_char_matrix_ndarr();
    plot_heat_map(&data, "heat7_1.png");
}

fn ex10() {
    let data = create_char_matrix_ndarr();
    plot_data(&data, "data.png");
}

fn ex11() {
    let data = create_char_matrix_ndarr();
    let (v, min, max) = ndarray_to_image_data(&data);
    dbg!(&v);
}

pub fn main() {
    println!("Notebook 001");
    ex9();
}
