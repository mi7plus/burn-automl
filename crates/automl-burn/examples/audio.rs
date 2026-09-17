//! Speech/audio classification with `AutoAudio`: two classes of tones at
//! distinct frequencies. A spectrogram front-end feeds the recurrent classifier.
//!
//! ```bash
//! cargo run -p automl-burn --release --example audio
//! ```

use automl_burn::AutoAudio;
use rand::{Rng, SeedableRng};
use std::f32::consts::PI;

fn main() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let (mut waves, mut labels) = (Vec::new(), Vec::new());
    for i in 0..120 {
        let low = i % 2 == 0;
        let f = if low { 3.0 } else { 9.0 };
        let phase: f32 = rng.gen_range(0.0..PI);
        let wave: Vec<f32> = (0..160)
            .map(|t| (2.0 * PI * f * t as f32 / 32.0 + phase).sin() + rng.gen_range(-0.05..0.05))
            .collect();
        waves.push(wave);
        labels.push(if low { 0 } else { 1 });
    }
    let study = AutoAudio::new(waves, labels)
        .num_classes(2)
        .epochs(6)
        .trials(2)
        .seed(1)
        .fit()
        .unwrap();
    println!(
        "best validation accuracy: {:.1}%",
        study
            .best_trial()
            .unwrap()
            .unwrap()
            .final_value("accuracy")
            .unwrap_or(0.0)
    );
}
