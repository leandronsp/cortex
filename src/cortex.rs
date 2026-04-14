use std::io::{Read, Write};

use crate::bpe::Bpe;
use crate::calc;
use crate::model::Model;

pub struct Cortex {
    bpe: Bpe,
    model: Box<dyn Model>,
}

pub struct TrainReport {
    pub first_avg_loss: f32,
    pub last_avg_loss: f32,
    pub epochs: usize,
    pub token_count: usize,
}

impl Cortex {
    pub fn new(model: Box<dyn Model>) -> Self {
        Self { bpe: Bpe::new(), model }
    }

    pub fn train(&mut self, corpus: &str, epochs: usize, learning_rate: f32) -> TrainReport {
        let tokens = self.bpe.encode(corpus);
        let mut first_avg_loss = 0.0;
        let mut last_avg_loss = 0.0;
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for window in tokens.windows(2) {
                let target = window[1];
                total_loss += self.model.train_step(&window[..1], target, learning_rate);
            }
            let avg = total_loss / (tokens.len().saturating_sub(1)).max(1) as f32;
            if epoch == 0 {
                first_avg_loss = avg;
            }
            last_avg_loss = avg;
            if epoch % 20 == 0 {
                println!("Epoch {:3}: avg loss = {:.4}", epoch, avg);
            }
        }
        TrainReport {
            first_avg_loss,
            last_avg_loss,
            epochs,
            token_count: tokens.len(),
        }
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> String {
        let mut context = self.bpe.encode(prompt);
        if context.is_empty() {
            return String::new();
        }
        let mut produced: Vec<u8> = Vec::new();
        for _ in 0..max_tokens {
            let logits = self.model.forward(&context);
            let probs = calc::softmax(&logits);
            let next = argmax(&probs) as u16;
            let byte = next as u8;
            produced.push(byte);
            context.push(next);
            if byte == b'\n' {
                break;
            }
        }
        String::from_utf8_lossy(&produced).into_owned()
    }

    pub fn save(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        let merges = self.bpe.merges();
        let count = merges.len() as u32;
        writer.write_all(&count.to_le_bytes())?;
        for ((a, b), new_token) in merges {
            writer.write_all(&a.to_le_bytes())?;
            writer.write_all(&b.to_le_bytes())?;
            writer.write_all(&new_token.to_le_bytes())?;
        }
        self.model.save(writer)
    }

    pub fn load(&mut self, reader: &mut dyn Read) -> std::io::Result<()> {
        let mut count_buf = [0u8; 4];
        reader.read_exact(&mut count_buf)?;
        let count = u32::from_le_bytes(count_buf) as usize;
        let mut merges = Vec::with_capacity(count);
        let mut u16_buf = [0u8; 2];
        for _ in 0..count {
            reader.read_exact(&mut u16_buf)?;
            let a = u16::from_le_bytes(u16_buf);
            reader.read_exact(&mut u16_buf)?;
            let b = u16::from_le_bytes(u16_buf);
            reader.read_exact(&mut u16_buf)?;
            let new_token = u16::from_le_bytes(u16_buf);
            merges.push(((a, b), new_token));
        }
        self.bpe.set_merges(merges);
        self.model.load(reader)
    }
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

#[cfg(test)]
#[path = "cortex_test.rs"]
mod tests;
