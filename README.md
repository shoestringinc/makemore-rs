# makemore-rs

### Objective

This is a rewrite of Andrej Karpathy's character level language model which he codes in the following video:

<https://www.youtube.com/watch?v=PaCmpygFfXo>

[Rust](https://www.rust-lang.org/) is the only requirement.

### Prerequisites

You must have Rust installed. Follow instructions here:
<https://www.rust-lang.org/tools/install>

### Installation

Clone this repo in any folder.

Inside the folder, just do:

```bash
cargo run
```

### Code layout

The code has been written as a single binary crate with modules for libs as well as *notebook* explorations.

It can be refactored into a library crate with multiple binaries for each of the notebook later.

The *notebook* files are prefixed with `nb`. For e.g. `nb001`, `nb002`, etc.

Each of the *cell* inside a *notebook* file is written as a Rust function prefixed with `ex`.
For e.g. `ex1`, `ex2`, etc.

### Usage

Whenever you want to execute cells in a particular notebook, do the following:
- go to `main.rs`.
- In the `main` function, select the notebook (for e.g. `nb005`) by:
```rust
nb005::main();
```
- Each notebook file has its own `main` function.
- Inside the `nb005.rs` file, execute any cell (for e.g. `ex11` by executing the `ex11()` function in `nb005` `main` function.

### Highlights

Andrej was using matplotlib and PyTorch. Many functions like plotting the heat map, bigram matrix, multinomial sampling, PyTorch one hot encoding. etc. were written easily using other Rust crates.

We used [ndarray crate](https://crates.io/crates/ndarray) for majority of the notebook examples till we came to the point where gradients had to be calculated. We switched to [burn crate](https://crates.io/crates/burn) thereafter.

### Code examples

The *notebooks* of importance are everything except `nb004`. Just ignore that.

Andrej this time did not write any library unlike [microgradr](https://github.com/shoestringinc/microgradr).

The entire `makemore` language model was coded in a Python notebook and we likewise coded in `nb001` to `nb005`.

We kept on adding functions for notebook users in the following files:
- `rust_mllib` which has functions that use `Burn lib`. This is our `PyTorch` equivalent.
- `numr` which has functions that usse `ndarray` crate. This is our `numpy` equivalent.
- `plot` has all the plotting functions. This is our `matplotlib` equivalent.

The library files are little messy as we got ambitious creating super convenient interface for notebook users. But we wanted to finish the project so some structs, etc. are not used and not complete.

But nevertheless, the lib modules have tons of tips for using various Rust AI crates. You will find lots of useful nuggets.

For the finale, read the cell `ex11` in `nb005` notebook.
It trains the neural net and then generates the words using multinomial distribution from the probabilities calculated from the final weights.


### Important imports for notebooks

```rust
use crate::numr::*;
use crate::plot::*;
use crate::rust_mllib::*;
use crate::utils::*;
```

There are others too which are just exposing the lower level crates. Maybe in future we will refactor them to use only the above crates.

### Plotting

Andrej uses matplotlib for plotting.

Check the `plot` module.

The main functions are:
- `plot_to_file` (for plotting points. We can rename it :-)
- `plot_heat_map`
- `plot_data` (used for plotting bigram matrix counts)

Open the files in any rendering app. We use `Preview` on `osx`. `Preview` automatically re-renders as the graph changes.


### Final notes

As usual, lots of reorgnization and refactoring can be done including separate test folder, lib and binaries etc.

### License

MIT

See the LICENSE file for more info.
