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
