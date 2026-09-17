//! Image classification with `AutoVision`: a CNN search over synthetic 1x8x8
//! images where a bright block occupies one of four quadrants (the label).
//!
//! ```bash
//! cargo run -p automl-burn --release --example vision
//! ```

use automl_burn::AutoVision;
use rand::{Rng, SeedableRng};

fn main() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let (mut imgs, mut labels) = (Vec::new(), Vec::new());
    for i in 0..160 {
        let q = (i % 4) as i64;
        let (r0, c0) = ((q / 2) as usize * 4, (q % 2) as usize * 4);
        let mut img = vec![0f32; 64];
        for r in r0..r0 + 4 {
            for c in c0..c0 + 4 {
                img[r * 8 + c] = 1.0 + rng.gen_range(-0.1..0.1);
            }
        }
        imgs.push(img);
        labels.push(q);
    }
    let study = AutoVision::new(imgs, labels, 1, 8, 8)
        .num_classes(4)
        .epochs(5)
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
