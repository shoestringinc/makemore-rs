#![allow(unused)]

use burn::backend::autodiff::grads::Gradients;
use ndarray::{arr2, Array1, Array2};
use ndarray::{stack, Axis};
use rand::thread_rng;
use std::f64::consts::E;

use crate::numr::*;
use crate::plot::*;
use crate::rust_mllib::*;
use crate::utils::*;
// use crate::utils::{
//     create_char_matrix_ndarr, create_char_matrix_tensor, ctoi, get_names, itoc,
//     rand_normal_distrib, rand_uniform_nums,
// };

fn input_output_pairs(num_words: usize) -> (Vec<usize>, Vec<usize>) {
    let words = get_names();
    let ctoi = ctoi();
    let itoc = itoc(&ctoi);

    let mut xs: Vec<usize> = vec![]; // input char
    let mut ys: Vec<usize> = vec![]; // desired char

    for w in &words[..num_words] {
        let chs = format!(".{}.", w);
        for (ch1, ch2) in chs.chars().zip(chs.chars().skip(1)) {
            let ix1 = ctoi[&ch1];
            let ix2 = ctoi[&ch2];
            xs.push(ix1);
            ys.push(ix2);
        }
    }

    (xs, ys)
}

fn ex1() {
    let words = get_names();
    let ctoi = ctoi();
    let itoc = itoc(&ctoi);

    let mut xs: Vec<usize> = vec![]; // input char
    let mut ys: Vec<usize> = vec![]; // desired char

    for w in &words[..1] {
        let chs = format!(".{}.", w);
        for (ch1, ch2) in chs.chars().zip(chs.chars().skip(1)) {
            let ix1 = ctoi[&ch1];
            let ix2 = ctoi[&ch2];
            // println!("{} {}", ch1, ch2);
            xs.push(ix1);
            ys.push(ix2);
        }
    }

    // single word i.e. first will have 5 examples for the neural net
    // dbg!(&xs);
    // dbg!(&ys);

    let xenc = one_hot_vec(&xs, 27);
    // dbg!(&xenc);

    // Because plot_heat_map takes ndarray Array2
    let t1 = vec_to_ndarr2(&xenc, xenc.len(), xenc[0].len());

    // dbg!(&t1);
    plot_heat_map(&t1, "one_hot2.png");
}

fn ex2() {
    let (xs, ys) = input_output_pairs(1);
    let xenc = one_hot(&xs, 27);

    // Weights
    let weights = randn(27, 1);
    dbg!(&weights);

    // 5 x 1
    // Basically for each input what value did the neuron give!
    // Here we had only one neuron
    let xs_x_ws = xenc.dot(&weights);
    dbg!(&xs_x_ws);
}

fn ex3() {
    // Dot product check
    // 2 x 3
    let inputs = vec![1, 2, 3, 4, 5, 6];
    let t1 = Array2::from_shape_vec((2, 3), inputs).unwrap();
    dbg!(&t1);
    // 3 x 1
    let weights = vec![10, 20, 30];
    let t2 = Array2::from_shape_vec((3, 1), weights).unwrap();
    dbg!(&t2);
    // 2 x 1
    let t3 = t1.dot(&t2);
    dbg!(&t3);
}

fn ex4() {
    let (xs, ys) = input_output_pairs(1);
    let inputs = one_hot(&xs, 27);
    // weights for 27 neurons
    let weights = randn(27, 27);
    // dbg!(&weights);

    // for each input what is the value of each of the 27 neurons
    // 5 x 27
    let res = inputs.dot(&weights);
    // println!("{}", res);

    // First input and the second neuron
    let x = res[[0, 1]];
    // println!("{x}");

    // 3rd input on the 13th neuron
    // println!("13th neuron firing on 3rd input {}", res[[2, 12]]); // 2.8925935942128285

    /*
    [[-2.6896996588544386, -1.5949525349934195, -0.36076418044950387, 0.4091364954293706, 2.6120071232583846, 1.2314528278303034, -2.838831036125167, 0.33008849484841907, 3.0133161273678764, -0.35825709187816734, -0.6420595146743118, 1.5875684977386153, 1.8350184501541702, -0.6456139093298119, -1.9579950436862585, 2.648992573329092, 1.3222773043017235, -1.3252280993555992, -2.001235221408179, -2.9887649454088847, -1.1458983501403546, 2.257846216908319, 1.7772263189951798, 0.1501913083161388, -0.9796481956789109, -2.677147109388177, 0.8744016396511647],
     [1.14150296134278, 1.1684028926131984, -1.9888881302044372, -2.4273821098120187, -2.6507106299391148, 0.16882557016191546, -1.1275460307685656, -2.9991717770551842, -2.0690872682826056, 1.1784448747673388, 0.42905521502608757, -1.1296264777749223, -1.7768623223259352, 0.8230961606019918, -2.9185420465990592, -2.0427854171285342, -1.7164251125736245, 0.820068555661086, 1.3812426363022312, 0.7198595207154219, 1.3916527284260618, 0.7719923798457673, -1.9528237901573842, -1.6580223728891172, 1.381601839069209, -3.0598736900607317, 2.70972898766277],
     [2.7364450625162093, -1.4804293758899, 2.3449644944575945, -1.8241268282369218, 2.295022887019592, 2.634656372868091, -2.357621156287847, -2.3759311533525147, -2.8726209958495392, -0.030764101098648133, 1.7477583825770275, 0.6661904676565276, 2.8925935942128285, 0.5016242514520788, -0.8393768333410856, 2.8318586096005025, 0.9476680056897604, -2.0984665285721937, -0.6364859649935615, -2.4372038136858114, 0.41510535351636557, -1.819521334684037, 2.9592667579412413, 2.089531878287825, -1.2108739131442412, 0.35087157482443354, -0.6282696840285347],
     [2.7364450625162093, -1.4804293758899, 2.3449644944575945, -1.8241268282369218, 2.295022887019592, 2.634656372868091, -2.357621156287847, -2.3759311533525147, -2.8726209958495392, -0.030764101098648133, 1.7477583825770275, 0.6661904676565276, 2.8925935942128285, 0.5016242514520788, -0.8393768333410856, 2.8318586096005025, 0.9476680056897604, -2.0984665285721937, -0.6364859649935615, -2.4372038136858114, 0.41510535351636557, -1.819521334684037, 2.9592667579412413, 2.089531878287825, -1.2108739131442412, 0.35087157482443354, -0.6282696840285347],
     [-0.4796817763134116, 2.550442308247694, -1.3461200878793176, 1.4538877412171343, -2.3848771152149033, -3.037387056610661, 2.8421633473301573, 2.4433997195850634, -2.436300151093138, 1.018084680537577, -1.542138822300317, 0.6312516894379772, 1.3019996289959805, -2.6872518711520126, -3.0214084136512707, 1.4270891982515468, -3.086351429775178, -1.4526508069631214, -0.5338361782536127, -1.6680061995542634, 1.2349638869443553, 1.96173543854704, 1.194760487087747, 1.2998929911940684, 0.35068435171745493, 2.705259270207009, -1.7281041665883825]]

    */
    let third_input = inputs.row(2);
    let thirteenth_neuron = weights.column(12);
    // println!("Third input: \n{}", third_input);
    // println!("Thirteenth neuron: \n{}", thirteenth_neuron);
    let neuron_val = third_input.dot(&thirteenth_neuron);
    println!("13th neuron firing on 3rd input {}", res[[2, 12]]);
    println!("Neuron value verified by dot product: {}", neuron_val);
}

fn ex5() {
    let (xs, ys) = input_output_pairs(1);

    // Set of inputs to be fed into the neural net.
    let inputs = one_hot(&xs, 27);

    // weights for each of the 27 neurons.
    let weights = randn(27, 27);

    // for each input what is the value of each of the 27 neurons
    // 5 x 27
    // Interpret: X dot W, as log counts or logits.
    let logits = inputs.dot(&weights);
    // println!("{}", res)

    // If we exponentiate then we can interpret them as counts
    // So this matrix equivalent to our 2d char matrix that we create using create_char_matrix
    let counts = logits.exp();
    // dbg!(&counts);

    // Now to get the probabilities like in our char matrix
    let prob = probabiity_row_wise(&counts);
    // dbg!(&prob);
    println!("{}", prob.row(0).sum());
}

fn ex6() {
    // ndarray api explorations

    // 2 x 3 matrix
    let a = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
    let mut xs = Vec::new();
    for row in a.rows() {
        let sum = row.sum();
        let new_row = row.mapv(|e| e / sum);
        xs.push(new_row);
    }
    let xs: Vec<_> = xs.iter().map(|r| r.view()).collect();
    let a1 = stack(Axis(0), &xs).unwrap();
    dbg!(&a1);
}

fn ex7() {
    // Rough calc
    let xs = [
        9.489309669402608,
        0.13125712870354542,
        2.925142047537846,
        1.0059390165304172,
        9.278091884400373,
        0.0754846633095178,
        1.296102016519838,
        4.786387470159512,
        2.577201356724872,
        1.382037567031469,
        0.2654712204372306,
        0.3615618334072086,
        0.2199932013834451,
        5.502815071258198,
        9.44850250687406,
        0.2690937339965065,
        0.13325850812268905,
        0.200961135799917,
        4.517666126512665,
        0.10964306268307805,
        7.0953902334609635,
        3.8108544650483362,
        0.15610612772999363,
        5.614646996336443,
        0.11316566217358716,
        0.11959632979313718,
        0.9425025089400019,
    ];
    let sum: f64 = xs.iter().sum();
    println!("{}", xs[0] / sum);
    println!("{}", xs[1] / sum);
}

fn ex8() {
    let (xs, ys) = input_output_pairs(1);

    let num_classes = 27_usize;

    let inputs = one_hot(&xs, num_classes);
    let weights = randn(num_classes, num_classes);

    // forward pass of neural net starts here
    // Interpret: X dot W, as log counts or logits.
    let logits = inputs.dot(&weights);
    let counts = logits.exp();

    let probs = probabiity_row_wise(&counts);
    // forward pass - everything is differentiable so we can do backpropagation!

    // println!("{:?}", probs.shape());

    // Because we are starting on back propagation we will use Burn Tensors now!
    let inputs_t = ndarr2_to_tensor(&inputs);
    let weights_t = ndarr2_to_tensor(&weights);

    // WARNING: Do not set gradient on computed i.e. non-leaf tensor
    let logits_t = inputs_t.matmul(weights_t);
    let counts_t = logits_t.exp();

    // Probability calculation Tensor style (now thatis Gangname style)
    // Burn supports auto broadcasting!
    let row_sums = counts_t.clone().sum_dim(1);
    let probs_t = counts_t.clone() / row_sums;
}

fn ex9() {
    let ctoi = ctoi();
    let itoc = itoc(&ctoi);

    let (xs, ys) = input_output_pairs(1);

    // We have already written this using ndarray, so we will convert these to tensors
    let inputs = one_hot(&xs, 27);
    let weights = randn(27, 27);

    // Now we have the tensors for each of the above
    let inputs_t = ndarr2_to_tensor(&inputs);
    let weights_t = ndarr2_to_tensor(&weights);

    let logits_t = inputs_t.matmul(weights_t);
    let counts_t = logits_t.exp();

    let row_sums = counts_t.clone().sum_dim(1);
    let probs_t = counts_t.clone() / row_sums;

    let mut nlls = create_empty_2d_tensor(xs.len(), 1);

    let num_cols = probs_t.dims()[1];

    // dbg!(&probs_t);
    for i in 0..5 {
        // ith-bigram
        let x = xs[i]; // input character index
        let y = ys[i]; // label character index
        println!("-------------");
        println!(
            "bigram example {}: {}{} (indexes {},{})",
            i + 1,
            itoc[&x],
            itoc[&y],
            x,
            y
        );
        println!("input to the neural net: {}", x);
        let prob_i = get_row(&probs_t, i);
        println!("output probabilities from the neural net: {:?}", &prob_i);
        println!("index (actual next character): {}", y);
        let p = get_elem(&prob_i, y);
        println!(
            "probability assigned by the neural net to the correct character: {}",
            p.clone().into_scalar()
        );
        // Remember log likelihood goes negative if probability goes towards zero
        let logp = p.log();
        println!("log likelihood: {}", logp.clone().into_scalar());
        let nll = logp.neg();
        println!("negative log likelihood: {}", nll.clone().into_scalar());
        nlls = nlls.slice_assign([i..i + 1, 0..1], nll);
    }
    println!("{}", nlls.clone());
    println!("==========");
    let avg_nll = nlls.mean();
    println!("average negative log likelihood, i.e. loss = {}", avg_nll);
}

fn ex10() {
    let ctoi = ctoi();
    let itoc = itoc(&ctoi);

    let (xs, ys) = input_output_pairs(1);

    let inputs = one_hot(&xs, 27);
    let weights = randn(27, 27);

    let inputs_t = ndarr2_to_tensor(&inputs);
    let mut weights_t = ndarr2_to_tensor(&weights);

    // let null_grad = weights_t.zeros_like();

    let logits_t = inputs_t.matmul(weights_t.clone());
    let counts_t = logits_t.exp();

    let row_sums = counts_t.clone().sum_dim(1);
    let probs_t = counts_t.clone() / row_sums;

    let mut tgt_probs = create_empty_2d_tensor(xs.len(), 1);

    for i in 0..(xs.len()) {
        let row = get_row(&probs_t, i);
        let tgt_index = ys[i];
        let p = get_elem(&row, tgt_index);
        tgt_probs = tgt_probs.slice_assign([i..i + 1, 0..1], p);
    }

    // println!("{}", tgt_probs);
    let loss = tgt_probs.log().neg().mean();
    println!("Loss: {}", loss);

    // let node_ref = loss

    // backward pass
    // let mut grads = Gradients::new(root_node, root_tensor)
    // weights_t = weights_t.set_require_grad(true);
    let mut grads = loss.backward();
    let g = weights_t.grad(&grads).unwrap();
    dbg!(&g);
    let ng = g.zeros_like();
    weights_t.grad_replace(&mut grads, ng);
    let g = weights_t.grad(&grads).unwrap();
    dbg!(&g);
}

fn ex11() {
    // Showtime

    let names = get_names();
    let total_words = names.len();
    println!("total words: {}", total_words);

    // Prepare data
    // Putting total words gives a huge number of input pairs 2lakh plus
    // and my machine crawls. Time for update.
    // So spiffy speed we put 20
    // NOte, I am not using GPU backend in Burn.
    // Plain NdArray
    let (xs, ys) = input_output_pairs(20);
    // let total_inputs = xs.len();
    // println!("total inputs: {}", total_inputs);

    let inputs = one_hot(&xs, 27);
    let weights = randn(27, 27);

    let inputs_t = ndarr2_to_tensor(&inputs);

    // this is going to change
    let mut weights_t = ndarr2_to_tensor(&weights);

    for k in 0..100 {
        let inputs_t = inputs_t.clone();
        let logits_t = inputs_t.matmul(weights_t.clone());
        let counts_t = logits_t.exp();

        let row_sums = counts_t.clone().sum_dim(1);
        let probs_t = counts_t.clone() / row_sums;

        let mut tgt_probs = create_empty_2d_tensor(xs.len(), 1);

        // We have to find the probability assigned to bigram pairs.
        // Probability has to be high or nll should be least.
        // Actually we will look at mean nll
        for i in 0..(xs.len()) {
            let row = get_row(&probs_t, i);
            let tgt_index = ys[i];
            let p = get_elem(&row, tgt_index);
            tgt_probs = tgt_probs.slice_assign([i..i + 1, 0..1], p);
        }

        // println!("{}", tgt_probs);
        let loss = tgt_probs.log().neg().mean();
        println!("loss: {}", loss.clone().into_scalar());
        let mut grads = loss.backward();
        let wg = weights_t.grad(&grads);
        if let Some(wg) = wg {
            let nudge = (wg.clone() * -10.0);
            let nudge = convert_nd_to_autodiff(&nudge);
            weights_t = weights_t.clone() + nudge;
            let ng = wg.zeros_like();
            weights_t.grad_replace(&mut grads, ng);
        } else {
            // println!("no gradient. Weights remain same");
            weights_t = weights_t.clone();
        }
    }

    // WORD GENERATION (based on our trained weights)

    // Now we will take the weight_t as it comes from the training
    // We used bigrams from just 20 words and did 100 passes to get the weights
    // But we will now use those weights for probability distribution of all the bigrams! Fidgety :-)
    // If you have a powerful machine then you can use all the bigram pairs to train and
    // use even more passes to get lowest loss!

    let (xs, _) = input_output_pairs(total_words);
    // println!("Total bigram pairs: {}", xs.len());

    let inputs = one_hot(&xs, 27);
    let inputs_t = ndarr2_to_tensor(&inputs);

    let logits_t = inputs_t.matmul(weights_t.clone());
    let counts_t = logits_t.exp();

    let row_sums = counts_t.clone().sum_dim(1);

    let probs_t = counts_t.clone() / row_sums;
    // println!("Trained probability dim: {:?}", probs_t.dims());

    // I have already written a multinomial in numr module using NdArray
    // so I am going to reuse it. Later we will create a tensor version.
    // which is really not required.
    // Once you have trained the neural net and have the calculation then
    // all we need is the trained probabilities in any structure.
    let probs_nd = tensor2d_to_ndarr2(&probs_t);
    let pd = probability_distrib_matrix(&probs_nd);

    let ctoi = ctoi();
    let itoc = itoc(&ctoi);
    let mut rng = thread_rng();

    // The moment we were waiting for!
    for i in 0..5 {
        let mut row_num = 0_usize;
        let mut chs = Vec::new();

        loop {
            row_num = multinomial_distrib(1, &pd[row_num], &mut rng)[0]; // remember we get Array so extract
            let ch = itoc[&row_num];
            if row_num == 0 {
                break;
            }
            chs.push(ch);
        }

        let name: String = chs.iter().collect();

        println!("{}", name);
    }
}

fn rough_calc() {}

pub fn main() {
    println!("Notebook 005");
    ex11();
    // rough_calc()
}

// 31:20
