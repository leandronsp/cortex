pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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

#[cfg(test)]
#[path = "calc_test.rs"]
mod tests;
