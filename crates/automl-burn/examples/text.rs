//! NLP text classification with `AutoText`: two topics with disjoint
//! vocabularies. Shows both the feature-hashing front-end and the learned
//! embedding table.
//!
//! ```bash
//! cargo run -p automl-burn --release --example text
//! ```

use automl_burn::AutoText;
use rand::{seq::SliceRandom, Rng, SeedableRng};

fn corpus(seed: u64) -> (Vec<String>, Vec<i64>) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let pos = ["great", "excellent", "love", "wonderful", "brilliant"];
    let neg = ["terrible", "awful", "hate", "boring", "worst"];
    let filler = ["the", "a", "this", "was", "it", "and"];
    let (mut texts, mut labels) = (Vec::new(), Vec::new());
    for i in 0..160 {
        let positive = i % 2 == 0;
        let topic = if positive { &pos } else { &neg };
        let mut words: Vec<&str> = Vec::new();
        for _ in 0..rng.gen_range(6..12) {
            words.push(if rng.gen_bool(0.5) {
                topic[rng.gen_range(0..topic.len())]
            } else {
                filler[rng.gen_range(0..filler.len())]
            });
        }
        words.shuffle(&mut rng);
        texts.push(words.join(" "));
        labels.push(if positive { 0 } else { 1 });
    }
    (texts, labels)
}

fn main() {
    let (texts, labels) = corpus(1);
    let hashed = AutoText::new(texts.clone(), labels.clone())
        .frontend(128, 16)
        .num_classes(2)
        .epochs(6)
        .trials(2)
        .seed(1)
        .fit()
        .unwrap();
    println!(
        "feature hashing:   best accuracy {:.1}%",
        hashed
            .best_trial()
            .unwrap()
            .unwrap()
            .final_value("accuracy")
            .unwrap_or(0.0)
    );

    let embedded = AutoText::new(texts, labels)
        .learned_embedding(64)
        .num_classes(2)
        .epochs(6)
        .trials(2)
        .seed(2)
        .fit()
        .unwrap();
    println!(
        "learned embedding: best accuracy {:.1}%",
        embedded
            .best_trial()
            .unwrap()
            .unwrap()
            .final_value("accuracy")
            .unwrap_or(0.0)
    );
}
