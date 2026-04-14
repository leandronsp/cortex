use std::io::{Read, Write};

use crate::calc;
use crate::model::Model;

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
        for i in 0..self.vocab_size as usize {
            self.weights[token_id as usize][i] -= learning_rate * gradient[i];
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
#[path = "bigram_test.rs"]
mod tests;
