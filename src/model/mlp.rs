use std::io::{Read, Write};

use crate::model::layer::{Activation, Layer};

use super::Model;

#[derive(Clone)]
pub struct MlpConfig {
    pub vocab_size: u16,
    pub context_size: usize,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
}

pub struct Mlp {
    config: MlpConfig,
    embedding: Vec<Vec<f32>>,
    hidden_layers: Vec<Layer>,
    output_layer: Layer,
}

impl Mlp {
    pub fn new(config: MlpConfig) -> Self {
        let vs = config.vocab_size as usize;
        let embedding_dim = config.embedding_dim;
        let input_size = config.context_size * config.embedding_dim;
        let hidden_dim = config.hidden_dim;

        Self {
            config,
            embedding: vec![vec![0.0; embedding_dim]; vs],
            hidden_layers: vec![
                Layer::new(input_size, hidden_dim, Activation::ReLU),
            ],
            output_layer: Layer::new(hidden_dim, vs, Activation::None),
        }
    }
}

impl Model for Mlp {
    fn vocab_size(&self) -> u16 {
        self.config.vocab_size
    }

    fn context_size(&self) -> usize {
        self.config.context_size
    }

    fn forward(&self, context: &[u16]) -> Vec<f32> {
        let mut neurons: Vec<f32> = context
            .iter()
            .flat_map(|&t| self.embedding[t as usize].clone())
            .collect();

        for layer in &self.hidden_layers {
            neurons = layer.forward(&neurons);
        }

        self.output_layer.forward(&neurons)
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
        let mut mlp = Mlp::new(MlpConfig {
            vocab_size: 4,
            context_size: 1,
            embedding_dim: 2,
            hidden_dim: 3,
        });

        mlp.embedding = vec![
            vec![1.0, 0.0], // Token 0
            vec![0.0, 1.0], // Token 1
        ];

        // Hidden layer: 3 neurons, each with 2 inputs, ReLU
        mlp.hidden_layers[0].weights = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];

        // Output layer: 4 neurons, each with 3 inputs, no activation
        mlp.output_layer.weights = vec![
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, -1.0],
            vec![0.0, 0.0, 0.0],
        ];

        let logits = mlp.forward(&[0]);

        assert_eq!(logits, vec![2.0, 0.0, 0.0, 0.0]);
    }
}

