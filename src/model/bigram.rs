use std::io::{Read, Write};

use super::Model;
use crate::training::calc;

pub struct Bigram {
    vocab_size: u16,
    weights: Vec<Vec<f32>>,
}

impl Bigram {
    pub fn new(vocab_size: u16) -> Self {
        Self {
            vocab_size,
            weights: vec![vec![0.0; vocab_size as usize]; vocab_size as usize],
        }
    }

    pub fn forward(&self, token_id: u16) -> Vec<f32> {
        self.weights[token_id as usize].clone()
    }

    pub fn set_weights(&mut self, weights: Vec<Vec<f32>>) {
        self.weights = weights;
    }

    pub fn update(&mut self, token_id: u16, gradient: &[f32], learning_rate: f32) {
        let row = &mut self.weights[token_id as usize];
        for (w, g) in row.iter_mut().zip(gradient.iter()) {
            *w -= learning_rate * g;
        }
    }
}

impl Model for Bigram {
    fn vocab_size(&self) -> u16 {
        self.vocab_size
    }

    fn forward(&self, context: &[u16]) -> Vec<f32> {
        let last = *context.last().expect("context must not be empty");
        Bigram::forward(self, last)
    }

    fn train_step(&mut self, context: &[u16], target: u16, learning_rate: f32) -> f32 {
        let input = *context.last().expect("context must not be empty");
        let logits = Bigram::forward(self, input);
        let probs = calc::softmax(&logits);
        let loss = calc::cross_entropy_loss(&probs, target as usize);
        let gradient = calc::cross_entropy_gradient(&probs, target as usize);
        self.update(input, &gradient, learning_rate);
        loss
    }

    fn save(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        writer.write_all(&self.vocab_size.to_le_bytes())?;
        for row in &self.weights {
            for &w in row {
                writer.write_all(&w.to_le_bytes())?;
            }
        }
        Ok(())
    }

    fn load(&mut self, reader: &mut dyn Read) -> std::io::Result<()> {
        let mut vs_buf = [0u8; 2];
        reader.read_exact(&mut vs_buf)?;
        let vocab_size = u16::from_le_bytes(vs_buf);
        let mut weights = vec![vec![0.0f32; vocab_size as usize]; vocab_size as usize];
        let mut f_buf = [0u8; 4];
        for row in weights.iter_mut() {
            for cell in row.iter_mut() {
                reader.read_exact(&mut f_buf)?;
                *cell = f32::from_le_bytes(f_buf);
            }
        }
        self.vocab_size = vocab_size;
        self.weights = weights;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenization::Bpe;

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
        let mut probs = crate::training::calc::softmax(&logits);

        // 2.0 > 1.0 > 0.0, so the probabilities should reflect that order
        assert!(probs[0] > probs[1] && probs[0] > probs[2] && probs[1] > probs[2]);

        logits = bigram.forward(1);
        probs = crate::training::calc::softmax(&logits);

        // All logits are the same, so probabilities should be equal
        // 0.0 == 0.0 == 0.0, so the probabilities should be equal
        assert!(probs[0] == probs[1] && probs[1] == probs[2]);

        logits = bigram.forward(2);
        probs = crate::training::calc::softmax(&logits);

        // [1.0, 0.0, 2.0] means the order of probabilities should be 2.0 > 1.0 > 0.0
        assert!(probs[2] > probs[0] && probs[2] > probs[1] && probs[0] > probs[1]);
    }

    #[test]
    fn test_pipeline_forward_loss() {
        let mut bigram = Bigram::new(3);

        bigram.set_weights(vec![
            vec![2.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 2.0],
        ]);

        let logits = bigram.forward(0);
        let probs = crate::training::calc::softmax(&logits);
        let loss = crate::training::calc::cross_entropy_loss(&probs, 0);

        // The correct class is index 0 with a probability of ~0.6652409, so the loss should be around -ln(0.6652409) ≈ 0.40760596
        assert!((loss - 0.40760596).abs() < 1e-6_f32);
    }

    #[test]
    pub fn test_update_adjust_weights_for_token_row() {
        let mut bigram = Bigram::new(3);

        bigram.set_weights(vec![
            vec![0.5, 0.5, 0.5],
            vec![0.1, 0.2, 0.3],
            vec![0.9, 0.9, 0.9],
        ]);

        let gradient = vec![-0.3, 0.2, 0.1]; // Example gradient for token_id 0

        bigram.update(0, &gradient, 0.1);

        let updated = bigram.forward(0);

        assert!((updated[0] - 0.53).abs() < 1e-6_f32);
        assert!((updated[1] - 0.48).abs() < 1e-6_f32);
        assert!((updated[2] - 0.49).abs() < 1e-6_f32);

        assert_eq!(bigram.forward(1), vec![0.1, 0.2, 0.3]);
        assert_eq!(bigram.forward(2), vec![0.9, 0.9, 0.9]);
    }

    #[test]
    fn test_train_then_predict_next_token() {
        let text = "ababab";
        let bpe = Bpe::new();
        let tokens = bpe.encode(text);
        // tokens: [97, 98, 97, 98, 97, 98]  ('a' = 97, 'b' = 98)

        let vocab_size = 256u16; // cover all possible bytes
        let mut bigram = Bigram::new(vocab_size);

        // Train: walk consecutive pairs, update weights
        let learning_rate = 1.0;
        let epochs = 100;

        for _ in 0..epochs {
            for window in tokens.windows(2) {
                let input = window[0];
                let target = window[1] as usize;

                let logits = bigram.forward(input);
                let probs = crate::training::calc::softmax(&logits);
                let gradient = crate::training::calc::cross_entropy_gradient(&probs, target);
                bigram.update(input, &gradient, learning_rate);
            }
        }

        // Predict: after 'a' (97), model should say 'b' (98)
        let probs = crate::training::calc::softmax(&bigram.forward(97));
        let predicted = argmax(&probs);
        assert_eq!(predicted, 98);

        // After 'b' (98), model should say 'a' (97)
        let probs = crate::training::calc::softmax(&bigram.forward(98));
        let predicted = argmax(&probs);
        assert_eq!(predicted, 97);
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }
}
