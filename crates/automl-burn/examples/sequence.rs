//! Sequence classification with `AutoSequence`: an LSTM/GRU search over
//! synthetic rising vs. falling multivariate sequences.
//!
//! ```bash
//! cargo run -p automl-burn --release --example sequence
//! ```

use automl_burn::AutoSequence;
use rand::{Rng, SeedableRng};

fn main() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let (mut seqs, mut labels) = (Vec::new(), Vec::new());
    for i in 0..160 {
        let rising = i % 2 == 0;
        let seq: Vec<Vec<f32>> = (0..8)
            .map(|t| {
                let base = if rising { t as f32 } else { 8.0 - t as f32 };
                vec![base + rng.gen_range(-0.3..0.3), rng.gen_range(-0.3..0.3)]
            })
            .collect();
        seqs.push(seq);
        labels.push(if rising { 0 } else { 1 });
    }
    let study = AutoSequence::new(seqs, labels)
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
