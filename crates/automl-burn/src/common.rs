//! Small helpers shared across the `automl-burn` adapters: a deterministic
//! train/validation split and image-tensor construction. Extracted so every
//! adapter uses one implementation rather than re-deriving it.

use burn::prelude::*;
use burn::tensor::TensorData;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Split `n` indices into `(train, val)` by shuffling with `seed` and taking a
/// `val_fraction` tail. At least one item lands in each side.
pub(crate) fn split(n: usize, val_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    idx.shuffle(&mut rng);
    let n_val = ((n as f64 * val_fraction).round() as usize).clamp(1, n.saturating_sub(1).max(1));
    let val = idx.split_off(n - n_val.min(n));
    (idx, val)
}

/// Build a `[n, channels, height, width]` image tensor from flat pixel rows.
pub(crate) fn image_tensor<B: Backend>(
    imgs: &[Vec<f32>],
    idx: &[usize],
    dims: (usize, usize, usize),
    device: &B::Device,
) -> Tensor<B, 4> {
    let (c, h, w) = dims;
    let n = idx.len();
    let flat: Vec<f32> = idx.iter().flat_map(|&i| imgs[i].iter().copied()).collect();
    Tensor::<B, 4>::from_data(
        TensorData::new(flat, [n, c, h, w]).convert::<B::FloatElem>(),
        device,
    )
}
