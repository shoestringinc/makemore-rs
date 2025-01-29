#![allow(unused)]

// This is for plotting graphs like matplotlib.
// We use plotters crate for actual rendering.

use std::fmt::Debug;
use std::fmt::Display;
use std::i64;
use std::rc::Rc;

use crate::numr::*;
use full_palette::DEEPPURPLE;
use full_palette::GREY;
use ndarray::{Array, Array1, Array2, Axis};
use plotters::coord::types::RangedCoordusize;
use plotters::prelude::*;
use plotters::style::colors::colormaps;
use plotters::style::text_anchor::HPos;
use plotters::style::text_anchor::Pos;
use plotters::style::text_anchor::VPos;

// use ndarray::prelude::*;
// use ndarray::Array;
use crate::utils::*;
use num_traits::{Bounded, Num, ToPrimitive};
use std::collections::HashMap;

// plotting points
pub fn plot_to_file(xs: &Array1<f64>, ys: &Array1<f64>, file_name: &str) {
    let xmin = min_array1(xs);
    let xmax = max_array1(xs) + 0.1;
    let xrange = xmin..xmax;

    let ymin = min_array1(ys);
    let ymax = max_array1(ys) + 0.1;
    let yrange = ymin..ymax;

    let root = BitMapBackend::new(file_name, (640, 480));
    let draw = root.into_drawing_area();
    draw.fill(&WHITE);

    let mut chart = ChartBuilder::on(&draw)
        .caption("Simple Plot", ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(xrange, yrange)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new(
            xs.iter().zip(ys.iter()).map(|(&x, &y)| (x, y)),
            &RED,
        ))
        .unwrap();
}

// fn ndarray_to_normalized_image_data<T>(data: &Array2<T>) -> (Vec<Vec<f64>>, f64, f64)
// where
//     T: Copy + Num + Bounded + ToPrimitive + PartialOrd,
// {
//     let shape = data.shape();
//     let (num_rows, num_cols) = (shape[0], shape[1]);
//     let mut image_data = Vec::new();
//     let mut min_value = T::max_value();
//     let mut max_value = T::min_value();
//     for i in 0..num_rows {
//         let mut row_data = Vec::new();
//         for j in 0..num_cols {
//             let v = data[[i,j]];
//             if v > max_value {
//                 max_value = v;
//             }
//             if v < min_value {
//                 min_value = v;
//             }
//             let v = v.to_f64();
//             row_data.push(v.unwrap()); // [row, col]
//         }
//         image_data.push(row_data);
//     }

//     let min = min_value.to_f64().unwrap();
//     let max = max_value.to_f64().unwrap();
//     let range = max - min;

//     // Normalize the data
//     let image_data_normalized: Vec<Vec<f64>> = image_data.iter()
//         .map(|row|{
//             row.iter()
//                 .map(|&value|{
//                     (value - min) / range
//                 }).collect()
//         }).collect();

//     (image_data_normalized, min, max)

// }

fn normalize_image_data<T>(image_data: &Vec<Vec<T>>, min: T, max: T) -> Vec<Vec<f64>>
where
    T: Copy + Num + Bounded + ToPrimitive + PartialOrd,
{
    let min = min.to_f64().unwrap();
    let max = max.to_f64().unwrap();
    let range = max - min;
    let xs: Vec<Vec<f64>> = image_data
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| {
                    let v = value.to_f64().unwrap();
                    (v - min) / range
                })
                .collect()
        })
        .collect();
    xs
}

pub fn ndarray_to_image_data<T>(data: &Array2<T>) -> (Vec<Vec<T>>, T, T)
where
    T: Copy + Num + Bounded + ToPrimitive + PartialOrd,
{
    let shape = data.shape();
    let (num_rows, num_cols) = (shape[0], shape[1]);
    let mut image_data: Vec<Vec<T>> = Vec::new();

    let mut min_value = T::max_value();
    let mut max_value = T::min_value();

    for i in 0..num_rows {
        let mut row_data = Vec::new();
        for j in 0..num_cols {
            let v = data[[i, j]]; // [row, col]
            if v > max_value {
                max_value = v;
            }
            if v < min_value {
                min_value = v;
            }
            row_data.push(v);
        }
        image_data.push(row_data);
    }
    (image_data, min_value, max_value)
}

enum ChartType {
    HeatMap,
    Text,
}

type RenderContext<'a> =
    ChartContext<'a, BitMapBackend<'a>, Cartesian2d<RangedCoordusize, RangedCoordusize>>;

fn prepare_chart_data<'a, 'b: 'a, T>(
    image_data: &Vec<Vec<T>>,
    file_name: &'b str,
    chart_type: ChartType,
) -> RenderContext<'a>
where
    T: Copy + Num + Bounded + ToPrimitive + PartialOrd,
{
    let ifactor: usize = match chart_type {
        ChartType::HeatMap => 10,
        ChartType::Text => 40,
    };

    let width = (image_data[0].len() * ifactor) as u32;
    let height = (image_data.len() * ifactor) as u32;

    let root = BitMapBackend::new(file_name, (width, height)).into_drawing_area();
    root.fill(&WHITE);

    let caption: &str = match chart_type {
        ChartType::HeatMap => "Heat Map",
        ChartType::Text => "Data Plot",
    };

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20).into_font())
        .margin(20)
        .build_cartesian_2d(0..image_data[0].len(), (image_data.len()..0))
        .unwrap();

    chart.configure_mesh().draw().unwrap();
    chart
}

pub fn plot_heat_map<T>(data: &Array2<T>, file_name: &str)
where
    T: Copy + Num + Bounded + ToPrimitive + PartialOrd,
{
    let (image_data, min, max) = ndarray_to_image_data(&data);
    let mut chart = prepare_chart_data(&image_data, file_name, ChartType::HeatMap);

    // Since this is heat map we have to normalize the value.
    let image_data = normalize_image_data(&image_data, min, max);

    chart
        .draw_series(image_data.iter().enumerate().flat_map(|(y, row)| {
            row.iter().enumerate().map(move |(x, &value)| {
                let color = colormaps::ViridisRGB::get_color(value);
                Rectangle::new([(x, y), (x + 1, y + 1)], color.filled())
            })
        }))
        .unwrap();
}

fn blue_colormap<T>(value: T, max: T) -> RGBColor
where
    T: Copy + Num + Bounded + ToPrimitive + PartialOrd + Debug + Display,
{
    let value = value.to_f64().unwrap();
    let max = max.to_f64().unwrap();

    // clamping
    let t = value / max;

    // from light blue to dark blue
    let r = (173.0 * (1.0 - t)) as u8;
    let g = (216.0 * (1.0 - t)) as u8;
    let b = (230.0 * (1.0 - t) + 139.0 * t) as u8;

    // let blue = (normal_value * 255.0) as u8;
    RGBColor(r, g, b)
}

pub fn plot_data<T>(data: &Array2<T>, file_name: &str)
where
    T: Copy + Num + Bounded + ToPrimitive + PartialOrd + Debug + Display,
{
    let (image_data, min, max) = ndarray_to_image_data(&data);
    let mut chart = prepare_chart_data(&image_data, file_name, ChartType::Text);

    // dbg!(&image_data);

    let ctoi = ctoi();
    let itoc: HashMap<usize, char> = ctoi.iter().map(|(&k, &v)| (v, k)).collect();

    // Not required - much better option which works.
    // let pos_label = Pos::new(HPos::Center, VPos::Top);
    // let pos_val = Pos::new(HPos::Center, VPos::Center);

    let root = chart.plotting_area();
    let (width_px, height_px) = root.dim_in_pixel();
    let rect_width = (width_px as f64) / (image_data[0].len() as f64);
    let rect_height = (height_px as f64) / (image_data.len() as f64);

    let x_loc = ((rect_width / 2.0) - 10.0) as i32;
    let y_val = ((rect_height / 2.0) - 10.0) as i32;
    let y_lbl = ((rect_height / 2.0) - 4.0) as i32;

    let text_el_val = |v: T| {
        Text::new(
            format!("{}", v),
            (x_loc, y_val), // 15, 15 when we hard coded for testing
            ("sans-serif", 10).into_font().color(&BLACK), // .pos(pos_val)
        )
    };

    let text_el_label = |v: String| {
        Text::new(
            format!("{}", v),
            (x_loc, y_lbl),                                    // 15, 25
            ("sans-serif", 15).into_font().color(&DEEPPURPLE), // .pos(pos_val)
        )
    };

    // Draw the background
    for (y, row) in image_data.iter().enumerate() {
        for (x, &value) in row.iter().enumerate() {
            let color = blue_colormap(value, max);
            chart
                .draw_series(std::iter::once(Rectangle::new(
                    [(x, y), (x + 1, y + 1)],
                    color.filled(),
                )))
                .unwrap();
        }
    }

    // Overlay the text
    for (y, row) in image_data.iter().enumerate() {
        for (x, &value) in row.iter().enumerate() {
            let chstr = format!("{}{}", itoc[&y], itoc[&x]);
            let e1 = EmptyElement::at((x, y)) + text_el_val(value);
            let e2 = EmptyElement::at((x, y)) + text_el_label(chstr);
            chart.draw_series(vec![e1, e2]).unwrap();
        }
    }
}

// pub fn plot_data_3<T>(data: &Array2<T>, file_name: &str)
// where
//     T: Copy + Num + Bounded + ToPrimitive + PartialOrd + Debug + Display,
// {
//     let (image_data, min, max) = ndarray_to_image_data(&data);
//     let mut chart = prepare_chart_data(&image_data, file_name, ChartType::Text);

//     // dbg!(&image_data);

//     // I know we use little bit of advanced Rust
//     // that is because we are using th maps inside closure
//     // with move semantics and we don't want expensive cloning!
//     let ctoi = Rc::new(ctoi());
//     let itoc: HashMap<usize, char> = ctoi.iter().map(|(&k, &v)| (v, k)).collect();
//     let itoc = Rc::new(itoc);

//     let pos_label = Pos::new(HPos::Center, VPos::Top);
//     let pos_val = Pos::new(HPos::Center, VPos::Bottom);

//     let max_row = image_data.len() - 1;

//     chart.draw_series(
//         image_data.iter().enumerate().flat_map(|(y, row)| {
//             let y = max_row - y;
//             println!("{y}");
//             let ctoi = ctoi.clone();
//             let itoc = itoc.clone();
//             row.iter().enumerate().flat_map(move |(x, &value)| {
//                 let val_str = value.to_string();
//                 let chstr = format!("{}{}", itoc[&y], itoc[&x]);

//                 let tcell1 = Text::new(chstr, (x * TEXT_SCALE_FACTOR, y * TEXT_SCALE_FACTOR),
//                     ("sans-serif", 15).into_font()
//                         .color(&GREY)
//                         .pos(pos_label)
//                 );
//                 let tcell2 = Text::new(val_str,
//                     (x * TEXT_SCALE_FACTOR, y * TEXT_SCALE_FACTOR),
//                     ("sans-serif", 10).into_font()
//                     .color(&BLACK)
//                     .pos(pos_val)
//                 );
//                 vec![tcell1, tcell2]

//                 // let color = colormaps::ViridisRGB::get_color(value);
//                 // Rectangle::new([(x, y), (x+1, y+1)], color.filled())

//             })
//         }),
//     ).unwrap();
// }

// fn plot_data_old<T>(data: &Array2<T>, file_name: &str)
// where
//     T: Copy + Num + Bounded + ToPrimitive + PartialOrd,
// {
//     let (image_data, min, max) = ndarray_to_normalized_image_data(data);

//     let width = (image_data[0].len() * 10) as u32;
//     let height = (image_data.len() * 10) as u32;

//     let root =
//         BitMapBackend::new(file_name, (width, height)).into_drawing_area();
//     root.fill(&WHITE);

//     let mut chart = ChartBuilder::on(&root)
//         .caption("Heat Map", ("sans-serif", 20).into_font())
//         .margin(20)
//         .build_cartesian_2d(0..image_data[0].len(), 0..image_data.len()).unwrap();
//     // TODO: Think how to make the cartesian x & y range to be more flexible.

//     chart.configure_mesh().draw().unwrap();

//     chart.draw_series(
//         image_data.iter().rev().enumerate().flat_map(|(y, row)| {
//             row.iter().enumerate().map(move|(x, &value)| {
//                 let color = colormaps::ViridisRGB::get_color(value);
//                 Rectangle::new([(x, y), (x+1, y+1)], color.filled())
//             })
//         }),
//     ).unwrap();

// }

// fn plot_heat_map_old(data: &Array2<i64>, file_name: &str) {
//     let shape = data.shape();
//     let (num_rows, num_cols) = (shape[0], shape[1]);
//     let mut image_data = Vec::new();
//     let mut min_value = i64::MAX;
//     let mut max_value = i64::MIN;
//     for i in 0..num_rows {
//         let mut row_data = Vec::new();
//         for j in 0..num_cols {
//             let v = data[[i,j]];
//             if v > max_value {
//                 max_value = v;
//             }
//             if v < min_value {
//                 min_value = v;
//             }
//             row_data.push(v as f64); // [row, col]
//         }
//         image_data.push(row_data);
//     }
//     // dbg!(&image_data);
//     println!("min {}", min_value);
//     println!("max {}", max_value);

//     let min = min_value as f64;
//     let max = max_value as f64;
//     let range = max - min;

//     // Normalize the data
//     let image_data_normalized: Vec<Vec<f64>> = image_data.iter()
//         .map(|row|{
//             row.iter()
//                 .map(|&value|{
//                     (value - min) / range
//                 }).collect()
//         }).collect();

//     let width = (image_data[0].len() * 10) as u32;
//     let height = (image_data.len() * 10) as u32;
//     println!("{}, {}", width, height);

//     let root =
//         BitMapBackend::new(file_name, (width, height)).into_drawing_area();
//     root.fill(&WHITE);

//     let mut chart = ChartBuilder::on(&root)
//         .caption("Heat Map", ("sans-serif", 20).into_font())
//         .margin(20)
//         .build_cartesian_2d(0..image_data[0].len(), 0..image_data.len()).unwrap();

//     chart.configure_mesh().draw().unwrap();

//     // we start from the last row upwards ! Exact heatmap as Andrej did using
//     // matplotlib.pyplot.imshow()
//     chart.draw_series(
//         image_data_normalized.iter().rev().enumerate().flat_map(|(y, row)| {
//             row.iter().enumerate().map(move|(x, &value)| {
//                 let color = colormaps::ViridisRGB::get_color(value);
//                 Rectangle::new([(x, y), (x+1, y+1)], color.filled())
//             })
//         }),
//     ).unwrap();

// }
