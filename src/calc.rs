pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp_logits: f32 = exp_logits.iter().sum();

    exp_logits.iter().map(|&x| x / sum_exp_logits).collect()
}

pub fn cross_entropy_loss(predicted: &[f32], target_index: usize) -> f32 {
    -predicted[target_index].ln()
}

#[cfg(test)]
#[path = "calc_test.rs"]
mod tests;
