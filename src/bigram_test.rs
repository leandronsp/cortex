use super::*;

#[test]
fn test_given_a_token_id_returns_its_logits() {
    let mut bigram = Bigram::new(3);

    bigram.set_weights(vec![
        vec![0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
        vec![0.7, 0.8, 0.9],
    ]);

    let first_logits = bigram.forward(0);
    assert_eq!(first_logits, vec![0.1, 0.2, 0.3]);
}

#[test]
fn test_forward_logits_into_probabilities() {
    let mut bigram = Bigram::new(3);

    bigram.set_weights(vec![
        vec![2.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.0, 2.0],
    ]);

    let mut logits = bigram.forward(0);
    let mut probs = crate::calc::softmax(&logits);

    // 2.0 > 1.0 > 0.0, so the probabilities should reflect that order
    assert!(probs[0] > probs[1] && probs[0] > probs[2] && probs[1] > probs[2]);

    logits = bigram.forward(1);
    probs = crate::calc::softmax(&logits);

    // All logits are the same, so probabilities should be equal
    // 0.0 == 0.0 == 0.0, so the probabilities should be equal
    assert!(probs[0] == probs[1] && probs[1] == probs[2]);

    logits = bigram.forward(2);
    probs = crate::calc::softmax(&logits);

    // [1.0, 0.0, 2.0] means the order of probabilities should be 2.0 > 1.0 > 0.0
    assert!(probs[2] > probs[0] && probs[2] > probs[1] && probs[0] > probs[1]);
}
