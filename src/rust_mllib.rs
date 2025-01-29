#![allow(unused)]

// These are wrappers over Rust Burn library to look more like PyTorch
// that Andrej used in his examples.

// use burn::backend::ndarray::NdArrayDevice;
// use burn::tensor::{Shape, Tensor, TensorData};

use std::cell::RefCell;
use std::rc::Rc;

use burn::backend::autodiff::grads::Gradients;
use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use burn::prelude::*;
use burn::tensor::backend::Backend;

use ndarray::Array2;

use num_traits::{Bounded, Num, ToPrimitive};

use crate::utils::create_char_matrix_ndarr;

const DEVICE: NdArrayDevice = NdArrayDevice::Cpu;

type BckAutoDiff<T> = Autodiff<NdArray<T>>; // this is for f32 and f64 only
                                            // we choose f64
type BckAutoDiffFloat = BckAutoDiff<f64>;

pub type FloatTensor2d = Tensor<BckAutoDiffFloat, 2>;
pub type FloatTensor1d = Tensor<BckAutoDiffFloat, 1>;

pub type IntTensor2d = Tensor<NdArray, 2, Int>;
pub type IntTensor1d = Tensor<NdArray, 1, Int>;

pub fn tensor1df(data: &[f64]) -> FloatTensor1d {
    Tensor::<BckAutoDiffFloat, 1>::from_data(data, &DEVICE)
}

pub fn tensor2df(data: &[f64], num_rows: usize, num_cols: usize) -> FloatTensor2d {
    let shape = &[num_rows, num_cols];
    let tensor_data = TensorData::new(data.to_vec(), shape); // unfortunately vec is reqd.
    Tensor::<BckAutoDiffFloat, 2>::from_data(tensor_data, &DEVICE)
}

pub fn ndarr2_to_tensor(arr: &Array2<f64>) -> FloatTensor2d {
    let shape = arr.shape();
    let raw_vec: Vec<f64> = arr.iter().cloned().collect();
    // let raw_vec: Vec<f64> = arr.into_raw_vec_and_offset();
    let tensor_data = TensorData::new(raw_vec, shape);
    // This is assuming a leaf tensor!
    Tensor::<BckAutoDiffFloat, 2>::from_data(tensor_data, &DEVICE).require_grad()
}

pub fn len1df(tensor: &FloatTensor1d) -> usize {
    tensor.dims()[0]
}

pub fn zeros2d(num_rows: usize, num_cols: usize) -> FloatTensor2d {
    let shape = Shape::new([num_rows, num_cols]);
    let mut tensor = Tensor::<BckAutoDiffFloat, 2>::zeros(shape, &DEVICE);
    tensor
}

pub fn tensor2d_to_nested_vec(tensor: &FloatTensor2d) -> Vec<Vec<f64>> {
    let shape = tensor.dims();
    let num_cols = shape[1];
    let data = tensor.to_data();
    let xs: Vec<f64> = data.to_vec().unwrap();
    let ys: Vec<Vec<f64>> = xs.chunks(num_cols).map(|row| row.to_vec()).collect();
    ys
}

pub fn zeros1d(num_elems: usize) -> FloatTensor1d {
    let shape = Shape::new([num_elems]);
    Tensor::<BckAutoDiffFloat, 1>::zeros(shape, &DEVICE)
}

pub fn ndarray_matrix_to_tensor<T>(matrix: &Array2<T>) -> FloatTensor2d
where
    T: Copy + Num + Bounded + ToPrimitive + PartialOrd,
{
    let v: Vec<f64> = matrix.iter().map(|&e| (e.to_f64()).unwrap()).collect();
    let shape = &[matrix.nrows(), matrix.ncols()];
    let tensor_data = TensorData::new(v, shape);
    Tensor::<BckAutoDiffFloat, 2>::from_data(tensor_data, &DEVICE)
}

pub fn one_hot_tensor(indices: &FloatTensor1d, num_classes: usize) -> FloatTensor2d {
    let depth = len1df(indices);
    // Tensor2d dim = depth x num_classes
    let mut zero_tensor = zeros2d(depth, num_classes);

    let indices: Vec<f64> = indices.clone().into_data().to_vec().unwrap();

    let xs: [usize; 2] = zero_tensor.shape().dims();
    println!("DIMS: {:?}", &xs);

    // zero_tensor.iter_dim(1).for_each(|e|);
    // zero_tensor.select_assign(dim, indices, values)

    zero_tensor
}

pub fn get_row(tensor: &FloatTensor2d, i: usize) -> FloatTensor2d {
    let num_cols = tensor.dims()[1];
    get_tensor(tensor, i, i + 1, 0, num_cols)
}

pub fn get_elem(tensor: &FloatTensor2d, i: usize) -> FloatTensor2d {
    // Assume there is only one row. We are just using 2d tensors instead of 1d tensor.
    get_tensor(tensor, 0, 1, i, i + 1)
}

fn get_tensor(tensor: &FloatTensor2d, a1: usize, a2: usize, b1: usize, b2: usize) -> FloatTensor2d {
    tensor.clone().slice([a1..a2, b1..b2])
}

pub fn create_empty_2d_tensor(num_rows: usize, num_cols: usize) -> FloatTensor2d {
    let shape = Shape::new([num_rows, 1]);
    Tensor::<BckAutoDiffFloat, 2>::empty(shape, &DEVICE)
}

pub fn null_grad(num_rows: usize, num_cols: usize) -> Tensor<NdArray<f64>, 2> {
    let shape = Shape::new([num_rows, num_cols]);
    let t = Tensor::<NdArray<f64>, 2>::zeros(shape, &DEVICE);
    t
}

pub fn vec_to_tensor1d<T>(xs: &Vec<T>) -> FloatTensor1d
where
    T: Copy + Num + Bounded + ToPrimitive + PartialOrd,
{
    let shape = Shape::new([xs.len()]);
    let xs: Vec<f64> = xs.iter().map(|&e| (e.to_f64()).unwrap()).collect();
    let tensor_data = TensorData::new(xs, shape);
    Tensor::<BckAutoDiffFloat, 1>::from_data(tensor_data, &DEVICE)
}

pub fn convert_nd_to_autodiff(
    tensor: &Tensor<NdArray<f64>, 2>,
) -> Tensor<Autodiff<NdArray<f64>>, 2> {
    let tensor_data = tensor.to_data();
    Tensor::<BckAutoDiffFloat, 2>::from_data(tensor_data, &DEVICE)
}

pub fn tensor2d_to_ndarr2(tensor: &FloatTensor2d) -> Array2<f64> {
    let dims = tensor.dims();
    let data: Vec<f64> = tensor.to_data().to_vec().unwrap();
    Array2::from_shape_vec((dims[0], dims[1]), data).unwrap()
}

fn ex1() {
    let data = create_char_matrix_ndarr();
    let v: Vec<i64> = data.iter().cloned().collect();
    v.iter().take(10).for_each(|e| println!("{e}"));
    let shape = data.shape();
    dbg!(&shape);
    println!("rows: {}", data.nrows());
    println!("cols: {}", data.ncols());
}

fn ex2() {
    // Since tensors of Burn libs are all floats we convert our word matrix too into floats
    let data = create_char_matrix_ndarr();
    let v: Vec<f64> = data.iter().map(|&e| e as f64).collect();

    let t1 = tensor2df(&v, data.nrows(), data.ncols());
}

fn ex3() {
    let narr = Array2::from_shape_vec((2, 3), vec![1, 2, 3, 4, 5, 6]).unwrap();

    let tensor = ndarray_matrix_to_tensor(&narr);
    let t2 = tensor.clone() + 1.;
    dbg!(&t2);
}

fn ex4() {
    let indices = tensor1df(&[1.0, 2.0, 3.0]);
    let num_classes = 27;
    let depth = len1df(&indices);
    let mut zero_tensor = zeros2d(depth, num_classes);

    let indices: Vec<f64> = indices.clone().into_data().to_vec().unwrap();

    let xs: [usize; 2] = zero_tensor.shape().dims();
    println!("DIMS: {:?}", &xs);

    // zero_tensor.select_assign(1, , values)

    // Tensor<B, 1, Int>
}

trait MLTensor {}

#[derive(Debug)]
pub struct TensorInt<const D: usize> {
    tensor: Tensor<NdArray, D, Int>,
}

impl<const D: usize> TensorInt<D> {
    pub fn new(xs: &[i64], shape: &[usize]) -> Self {
        if D == 1 {
            Self::create1d(xs)
        } else {
            assert!(shape.len() == 2);
            let num_rows = shape[0];
            let num_cols = shape[1];
            let total_elems = num_rows * num_cols;
            assert_eq!(xs.len(), total_elems);
            Self::create2d(xs, num_rows, num_cols)
        }
    }
    fn create1d(xs: &[i64]) -> Self {
        let xs = xs.to_vec();
        let shape = Shape::new([xs.len()]);
        let tensor_data = TensorData::new(xs, shape);
        let tensor = Tensor::<NdArray, D, Int>::from_data(tensor_data, &DEVICE);
        Self { tensor }
    }

    fn create2d(xs: &[i64], num_rows: usize, num_cols: usize) -> Self {
        let xs = xs.to_vec();
        let shape = Shape::new([num_rows, num_cols]);
        let tensor_data = TensorData::new(xs, shape);
        let tensor = Tensor::<NdArray, D, Int>::from_data(tensor_data, &DEVICE);
        Self { tensor }
    }

    pub fn tensor(&self) -> &Tensor<NdArray, D, Int> {
        &self.tensor
    }
}

impl<const D: usize> MLTensor for TensorInt<D> {}

impl<const D: usize> std::fmt::Display for TensorInt<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.tensor.fmt(f)
    }
}

// This is an autograd tensor.
// For time and simplicity purpose from user perspective there is some duplication.
pub struct TensorFloat<const D: usize> {
    tensor: Tensor<Autodiff<NdArray<f64>>, D>,
}

impl<const D: usize> TensorFloat<D> {
    pub fn new(xs: &[f64], shape: &[usize]) -> Self {
        if D == 1 {
            Self::create1d(xs)
        } else {
            assert!(shape.len() == 2);
            let num_rows = shape[0];
            let num_cols = shape[1];
            let total_elems = num_rows * num_cols;
            assert_eq!(xs.len(), total_elems);
            Self::create2d(xs, num_rows, num_cols)
        }
    }

    fn create1d(xs: &[f64]) -> Self {
        let xs = xs.to_vec();
        let shape = Shape::new([xs.len()]);
        let tensor_data = TensorData::new(xs, shape);
        let tensor = Tensor::<Autodiff<NdArray<f64>>, D>::from_data(tensor_data, &DEVICE);
        Self { tensor }
    }

    fn create2d(xs: &[f64], num_rows: usize, num_cols: usize) -> Self {
        let xs = xs.to_vec();
        let shape = Shape::new([num_rows, num_cols]);
        let tensor_data = TensorData::new(xs, shape);
        let tensor = Tensor::<Autodiff<NdArray<f64>>, D>::from_data(tensor_data, &DEVICE);
        Self { tensor }
    }

    pub fn tensor(&self) -> &Tensor<Autodiff<NdArray<f64>>, D> {
        &self.tensor
    }
}

impl<const D: usize> std::fmt::Display for TensorFloat<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.tensor.fmt(f)
    }
}

impl<const D: usize> MLTensor for TensorFloat<D> {}

// // Trying to wrap both the types of tensor into a common one
// // so that we can call the methods without duplication.
// enum BackendDType {
//     IntType,
//     FloatType
// }

// pub struct RTensor<const D: usize> {
//     tensor: Rc<RefCell<dyn MLTensor>>,
// }

// impl<const D: usize> RTensor<D> {
//     pub fn new(tensor_type: BackendDType) -> Self {
//         todo!()
//     }
// }

fn ex5() {
    // This backend has support for Int
    let l1: Tensor<NdArray, 1, Int> = Tensor::from_ints([1, 2, 3], &DEVICE);
    let l2: Tensor<NdArray, 1, Float> = Tensor::from_floats([1.0, 2.0], &DEVICE);
    let l2: Tensor<NdArray<f64>, 1> = Tensor::from_floats([1.0, 2.0], &DEVICE);

    // i64 is supported!
    let xs: Vec<u64> = vec![1, 2, 3]; // usize is not supported for trait bound Element
    let shape = Shape::new([xs.len()]);
    let tdata = TensorData::new(xs, shape);

    let l3 = Tensor::<NdArray, 1, Int>::from_data(tdata, &DEVICE);
    println!("{}", l3);

    let ys: Vec<f64> = vec![1., 2., 3.];
    let shape = Shape::new([ys.len()]);
    let tdata = TensorData::new(ys, shape);
    // dtype still will be f32. For f64 see below
    let l4 = Tensor::<NdArray, 1, Float>::from_data(tdata, &DEVICE);
    println!("{}", l4);

    let ys: Vec<f64> = vec![1., 2., 3.];
    let shape = Shape::new([ys.len()]);
    let tdata = TensorData::new(ys, shape);
    let l4 = Tensor::<NdArray<f64>, 1>::from_data(tdata, &DEVICE);
    println!("{}", l4);
}

fn ex6() {
    let t = TensorInt::<1>::new(&[1, 2, 3], &[]);
    println!("{}", t);
    let v = vec![1_i64, 2, 3, 4, 5, 6];
    let t1 = TensorInt::<2>::new(&v, &[2, 3]);
    println!("{}", t1);
}

fn ex7() {
    // let t: Tensor<dyn Backend, 1, Int> = Tensor::<NdArray, 1, Int>::from_ints([1,2,3], &DEVICE);
}

pub fn main() {
    println!("Exploring Burn API");
    ex6();
}
