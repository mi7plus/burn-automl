//! Multimodal fusion primitives: a `MultimodalSpace` searches each modality's
//! encoder and projection dimension plus a fusion strategy, and validates
//! dimension compatibility *before* scheduling so an incompatible configuration
//! never reaches training.
//!
//! ```bash
//! cargo run -p automl-core --release --example multimodal
//! ```

use automl_core::prelude::*;

fn main() {
    let space = MultimodalSpace::new(vec![
        Modality::new("image", ["cnn", "vit"], 4, 8),
        Modality::new("text", ["rnn", "transformer"], 4, 8),
    ]);
    let sp = space.to_search_space();
    let history = TrialHistory::default();

    // Sample a few configurations; validate each before it would be scheduled.
    for i in 0..6 {
        let mut sampler = RandomSampler::new(i);
        let plan = space.decode(&sampler.suggest(&sp, &history)).unwrap();
        let dims: Vec<usize> = plan.encoders.iter().map(|m| m.dim).collect();
        match plan.validate() {
            Ok(()) => println!(
                "{:?} + {:?}  fused_dim={}",
                plan.encoders
                    .iter()
                    .map(|m| m.encoder.as_str())
                    .collect::<Vec<_>>(),
                plan.fusion,
                plan.fused_dim(),
            ),
            Err(e) => println!(
                "rejected before scheduling ({dims:?}, {:?}): {e}",
                plan.fusion
            ),
        }
    }

    // Fusing per-modality feature vectors directly.
    let fused = fuse(&[vec![1.0, 2.0], vec![3.0, 4.0]], Fusion::Concat).unwrap();
    println!("concat fuse([1,2],[3,4]) = {fused:?}");
}
