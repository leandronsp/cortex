use std::io::{Read, Write};

use crate::model::layer::Layer;
use crate::training::calc;

use super::Model;

pub struct Mlp {
    vocab_size: u16,
    context_size: usize,
    embedding: Vec<Vec<f32>>,
    layers: Vec<Layer>,
}

impl Mlp {
    pub fn new(
        vocab_size: u16,
        context_size: usize,
        embedding_dim: usize,
        hidden_dim: usize,
    ) -> Self {
        let vs = vocab_size as usize;
        let input_size = context_size * embedding_dim;

        Self {
            vocab_size,
            context_size,
            embedding: vec![vec![0.0; embedding_dim]; vs],
            layers: vec![
                Layer::new(input_size, hidden_dim),
                Layer::new(hidden_dim, vs),
            ],
        }
    }
}

impl Model for Mlp {
    fn vocab_size(&self) -> u16 {
        self.vocab_size
    }

    fn context_size(&self) -> usize {
        self.context_size
    }

    fn forward(&self, context: &[u16]) -> Vec<f32> {
        let embedded: Vec<f32> = context
            .iter()
            .flat_map(|&t| self.embedding[t as usize].clone())
            .collect();

        let mut activation = self.layers[0].forward(&embedded);
        activation = calc::relu_vec(&activation);
        self.layers[1].forward(&activation)
    }

    fn train_step(&mut self, _context: &[u16], _target: u16, _learning_rate: f32) -> f32 {
        todo!()
    }

    fn save(&self, _writer: &mut dyn Write) -> std::io::Result<()> {
        todo!()
    }

    fn load(&mut self, _reader: &mut dyn Read) -> std::io::Result<()> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_forward_with_known_weights() {
        let mut mlp = Mlp::new(4, 1, 2, 3);

        mlp.embedding[0] = vec![1.0, 0.0];
        mlp.embedding[1] = vec![0.0, 1.0];

        mlp.layers[0].weights = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        mlp.layers[1].weights = vec![
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, -1.0],
            vec![0.0, 0.0, 0.0],
        ];

        let logits = mlp.forward(&[0]);

        assert_eq!(logits.len(), 4);
    }
}