//! Semantic segmentation with `AutoSegmentation`: synthetic 1x8x8 images with a
//! foreground square; the mask labels foreground pixels. Scored by mean IoU.
//!
//! ```bash
//! cargo run -p automl-burn --release --example segmentation
//! ```

use automl_burn::AutoSegmentation;
use rand::{Rng, SeedableRng};

fn main() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let (mut imgs, mut masks) = (Vec::new(), Vec::new());
    for _ in 0..96 {
        let (r0, c0) = (rng.gen_range(0..4), rng.gen_range(0..4));
        let mut img = vec![0f32; 64];
        let mut mask = vec![0i64; 64];
        for r in r0..r0 + 4 {
            for c in c0..c0 + 4 {
                img[r * 8 + c] = 1.0 + rng.gen_range(-0.1..0.1);
                mask[r * 8 + c] = 1;
            }
        }
        imgs.push(img);
        masks.push(mask);
    }
    let study = AutoSegmentation::new(imgs, masks, 1, 8, 8, 2)
        .epochs(8)
        .trials(3)
        .seed(1)
        .fit()
        .unwrap();
    println!(
        "best mean IoU: {:.1}%",
        study
            .best_trial()
            .unwrap()
            .unwrap()
            .final_value("iou")
            .unwrap_or(0.0)
    );
}
