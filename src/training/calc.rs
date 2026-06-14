pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Expoente da diferença do logit pro max_logit
    let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();

    let sum_exp_logits: f32 = exp_logits.iter().sum();

    exp_logits.iter().map(|&x| x / sum_exp_logits).collect()
}

pub fn cross_entropy_loss(predicted: &[f32], target_index: usize) -> f32 {
    -predicted[target_index].ln()
}

pub fn cross_entropy_gradient(probs: &[f32], target_index: usize) -> Vec<f32> {
    let mut gradient = probs.to_vec();
    gradient[target_index] -= 1.0; // For the correct class, subtract 1 from the probability
    gradient
}

pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub fn relu_vec(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| relu(x)).collect()
}

/// Deterministic xorshift PRNG. Same seed yields the same stream, so weight
/// initialization stays reproducible. Lives outside the model: callers inject
/// the weights, the model is just the engine.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform value in [-range, range).
    pub fn uniform(&mut self, range: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        (unit * 2.0 - 1.0) * range
    }
}

pub fn fill_uniform(matrix: &mut [Vec<f32>], range: f32, rng: &mut Rng) {
    for row in matrix {
        for value in row {
            *value = rng.uniform(range);
        }
    }
}

pub fn random_matrix(rows: usize, cols: usize, range: f32, rng: &mut Rng) -> Vec<Vec<f32>> {
    let mut matrix = vec![vec![0.0; cols]; rows];
    fill_uniform(&mut matrix, range, rng);
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_logits_into_probabilities() {
        let logits = vec![2.0, 1.0, 0.0];

        let probabilities = softmax(&logits);

        let expected = [0.6652409, 0.24472848, 0.09003057];

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

        assert!((loss - 1.609_438).abs() < 1e-6_f32);
    }

    #[test]
    fn test_cross_entropy_gradient() {
        let probs = vec![0.7, 0.2, 0.1];

        let gradient = cross_entropy_gradient(&probs, 0);

        // Target index 0: prob - 1 = 0.7 - 1 = -0.3
        // Other indices: prob unchanged = 0.2 and 0.1
        let expected = [-0.3, 0.2, 0.1];

        for (g, e) in gradient.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6_f32);
        }
    }

    #[test]
    fn test_relu_returns_zero_for_negative() {
        assert_eq!(relu(-2.0), 0.0);
        assert_eq!(relu(-0.5), 0.0);
    }

    #[test]
    fn test_relu_returns_input_for_positive() {
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(3.5), 3.5);
    }

    #[test]
    fn test_relu_vec_zeros_negatives_and_keeps_positives() {
        let input = vec![-2.0, 0.0, 3.0, -0.5, 1.5];
        let output = relu_vec(&input);

        assert_eq!(output, vec![0.0, 0.0, 3.0, 0.0, 1.5]);
    }

    #[test]
    fn test_rng_same_seed_yields_same_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);

        for _ in 0..16 {
            assert_eq!(a.uniform(0.1), b.uniform(0.1));
        }
    }

    #[test]
    fn test_rng_uniform_stays_within_range() {
        let mut rng = Rng::new(7);

        for _ in 0..256 {
            let value = rng.uniform(0.1);
            assert!((-0.1..0.1).contains(&value));
        }
    }
}
