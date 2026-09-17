//! Single-object detection with `AutoDetection`: synthetic 1x16x16 images with a
//! bright square; the target is its normalized bounding box. Scored by mean IoU.
//!
//! ```bash
//! cargo run -p automl-burn --release --example detection
//! ```

use automl_burn::AutoDetection;
use rand::{Rng, SeedableRng};

fn main() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let (n, sz) = (64usize, 16usize);
    let (mut imgs, mut boxes) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let side = 5usize;
        let (r0, c0) = (rng.gen_range(0..sz - side), rng.gen_range(0..sz - side));
        let mut img = vec![0f32; sz * sz];
        for r in r0..r0 + side {
            for c in c0..c0 + side {
                img[r * sz + c] = 1.0;
            }
        }
        imgs.push(img);
        boxes.push([
            c0 as f32 / sz as f32,
            r0 as f32 / sz as f32,
            (c0 + side) as f32 / sz as f32,
            (r0 + side) as f32 / sz as f32,
        ]);
    }
    let study = AutoDetection::new(imgs, boxes, 1, sz, sz)
        .epochs(8)
        .trials(2)
        .seed(1)
        .fit()
        .unwrap();
    println!(
        "best mean box IoU: {:.1}%",
        study
            .best_trial()
            .unwrap()
            .unwrap()
            .final_value("iou")
            .unwrap_or(0.0)
    );
}
