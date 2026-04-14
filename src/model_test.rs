use super::*;
use crate::bigram::Bigram;

#[test]
fn bigram_implements_model_trait() {
    let mut bigram = Bigram::new(4);
    let model: &mut dyn Model = &mut bigram;
    assert_eq!(model.vocab_size(), 4);
    let logits = model.forward(&[0]);
    assert_eq!(logits.len(), 4);
}

#[test]
fn train_step_reduces_loss() {
    let mut bigram = Bigram::new(8);
    let model: &mut dyn Model = &mut bigram;
    let loss_before = model.train_step(&[0], 1, 1.0);
    let loss_after = model.train_step(&[0], 1, 1.0);
    assert!(loss_after < loss_before);
}

#[test]
fn save_load_round_trip_preserves_forward() {
    let mut source = Bigram::new(3);
    let gradient = vec![1.0, -2.0, 0.5];
    source.update(0, &gradient, 1.0);
    let source_logits = source.forward(0);

    let mut buf: Vec<u8> = Vec::new();
    (&source as &dyn Model).save(&mut buf).unwrap();

    let mut restored = Bigram::new(3);
    (&mut restored as &mut dyn Model)
        .load(&mut buf.as_slice())
        .unwrap();

    assert_eq!(restored.forward(0), source_logits);
}
