//! Neural architecture search on **real MNIST**: `AutoNas` searches a stack of
//! convolutional cells (op, width, skip, depth) plus learning rate and dropout,
//! driven by the evolutionary sampler, to maximize validation accuracy.
//!
//! This is the larger-dataset counterpart to `mnist_search` (which tunes an
//! MLP): here the *architecture itself* is searched on genuine image data.
//! Running it **downloads MNIST** on first use and trains CNNs on the CPU, so it
//! takes several minutes; enable the `wgpu` feature to run on a GPU instead.
//!
//! ```bash
//! cargo run -p automl-burn --release --example mnist_nas
//! # on a GPU:
//! cargo run -p automl-burn --release --features wgpu --example mnist_nas
//! ```

use automl_burn::AutoNas;
use automl_core::prelude::*;
use burn::data::dataset::vision::MnistDataset;
use burn::data::dataset::Dataset;

/// Load the first `n` MNIST items as flat, [0,1]-scaled `1x28x28` pixel vectors.
fn load(dataset: &MnistDataset, n: usize) -> (Vec<Vec<f32>>, Vec<i64>) {
    let count = dataset.len().min(n);
    let mut images = Vec::with_capacity(count);
    let mut labels = Vec::with_capacity(count);
    for i in 0..count {
        if let Some(item) = dataset.get(i) {
            let flat: Vec<f32> = item
                .image
                .iter()
                .flat_map(|row| row.iter().map(|&p| p / 255.0))
                .collect();
            images.push(flat);
            labels.push(item.label as i64);
        }
    }
    (images, labels)
}

fn main() -> Result<()> {
    println!("loading MNIST (downloads on first run)...");
    let (images, labels) = load(&MnistDataset::train(), 2000);

    println!(
        "searching CNN architectures on MNIST ({} images)...",
        images.len()
    );
    let study = AutoNas::new(images, labels, 1, 28, 28)
        .num_classes(10)
        .epochs(4)
        .trials(8)
        .seed(7)
        .fit()?;

    let best = study.best_trial()?.expect("a completed trial");
    println!(
        "\nbest validation accuracy: {:.2}%",
        best.final_value("accuracy").unwrap_or(0.0)
    );
    println!(
        "best learning rate:       {:.5}",
        best.params.float("lr").unwrap_or(0.0)
    );
    println!(
        "architecture depth:       {}",
        best.params.int("depth").unwrap_or(0)
    );
    Ok(())
}
