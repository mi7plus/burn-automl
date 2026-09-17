//! Transformer sequence classification with `AutoTransformer`: searches heads,
//! per-head width, depth and FFN ratio over a synthetic positional task (is the
//! spike in the first half or the second half of the sequence?).
//!
//! ```bash
//! cargo run -p automl-burn --release --example transformer
//! ```

use automl_burn::AutoTransformer;
use rand::{Rng, SeedableRng};

fn main() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let (mut seqs, mut labels) = (Vec::new(), Vec::new());
    for i in 0..160 {
        let first_half = i % 2 == 0;
        let spike = if first_half {
            rng.gen_range(0..4)
        } else {
            rng.gen_range(4..8)
        };
        let seq: Vec<Vec<f32>> = (0..8)
            .map(|t| vec![if t == spike { 1.0 } else { 0.0 } + rng.gen_range(-0.05..0.05)])
            .collect();
        seqs.push(seq);
        labels.push(if first_half { 0 } else { 1 });
    }
    let study = AutoTransformer::new(seqs, labels)
        .num_classes(2)
        .epochs(6)
        .trials(3)
        .seed(1)
        .fit()
        .unwrap();
    let best = study.best_trial().unwrap().unwrap();
    println!(
        "best validation accuracy: {:.1}%",
        best.final_value("accuracy").unwrap_or(0.0)
    );
}
