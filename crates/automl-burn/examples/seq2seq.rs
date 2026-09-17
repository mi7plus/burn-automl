//! Sequence-to-sequence with `AutoSeq2Seq`: an LSTM encoder-decoder that learns
//! to reverse short symbol sequences.
//!
//! ```bash
//! cargo run -p automl-burn --release --example seq2seq
//! ```

use automl_burn::AutoSeq2Seq;
use rand::{Rng, SeedableRng};

fn main() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let (mut src, mut tgt) = (Vec::new(), Vec::new());
    for _ in 0..320 {
        let s: Vec<usize> = (0..4).map(|_| rng.gen_range(1..=5)).collect();
        let mut r = s.clone();
        r.reverse();
        src.push(s);
        tgt.push(r);
    }
    let study = AutoSeq2Seq::new(src, tgt, 6) // vocab 6 = 5 symbols + BOS
        .epochs(20)
        .trials(3)
        .seed(1)
        .fit()
        .unwrap();
    let best = study.best_trial().unwrap().unwrap();
    println!(
        "best token accuracy: {:.1}%",
        best.final_value("token_acc").unwrap_or(0.0)
    );
}
