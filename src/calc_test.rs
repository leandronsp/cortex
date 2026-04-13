use super::*;

#[test]
fn test_softmax_logits_into_probabilities() {
    let logits = vec![2.0, 1.0, 0.0];

    let probabilities = softmax(&logits);

    let expected = vec![0.6652409, 0.24472848, 0.09003057];

    for (p, e) in probabilities.iter().zip(expected.iter()) {
        assert!((p - e).abs() < 1e-6_f32);
    }
}

#[test]
fn test_cross_entropy_loss() {
    let predicted = vec![0.7, 0.2, 0.1];

    let mut loss = cross_entropy_loss(&predicted, 0);

    assert!((loss - 0.35667494).abs() < 1e-6_f32);

    loss = cross_entropy_loss(&predicted, 1);

    assert!((loss - 1.6094379).abs() < 1e-6_f32);
}

#[test]
fn test_cross_entropy_gradient() {
    let probs = vec![0.7, 0.2, 0.1];

    let gradient = cross_entropy_gradient(&probs, 0);

    // Target index 0: prob - 1 = 0.7 - 1 = -0.3
    // Other indices: prob unchanged = 0.2 and 0.1
    let expected = vec![-0.3, 0.2, 0.1];

    for (g, e) in gradient.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-6_f32);
    }
}
