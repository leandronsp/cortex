use std::io::{Read, Write};

pub trait Model {
    fn vocab_size(&self) -> u16;
    fn forward(&self, context: &[u16]) -> Vec<f32>;
    fn train_step(&mut self, context: &[u16], target: u16, learning_rate: f32) -> f32;
    fn save(&self, writer: &mut dyn Write) -> std::io::Result<()>;
    fn load(&mut self, reader: &mut dyn Read) -> std::io::Result<()>;
}

#[cfg(test)]
#[path = "model_test.rs"]
mod tests;
