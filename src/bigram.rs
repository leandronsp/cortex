pub struct Bigram {
    vocab_size: u16,
    weights: Vec<Vec<f32>>,
}

impl Bigram {
    pub fn new(vocab_size: u16) -> Self {
        Self {
            vocab_size: vocab_size,
            weights: vec![vec![0.0; vocab_size as usize]; vocab_size as usize],
        }
    }

    pub fn forward(&self, token_id: u16) -> Vec<f32> {
        self.weights[token_id as usize].clone()
    }

    pub fn set_weights(&mut self, weights: Vec<Vec<f32>>) {
        self.weights = weights;
    }
}

#[cfg(test)]
#[path = "bigram_test.rs"]
mod tests;
