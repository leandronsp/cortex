use std::io::{Read, Write};

use crate::{
    model::layer::{Activation, Layer},
    training::calc,
};

use super::Model;

#[derive(Clone)]
pub struct MlpConfig {
    pub vocab_size: u16,
    pub context_size: usize,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub num_hidden_layers: usize,
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
        let num_hidden_layers = config.num_hidden_layers;

        let mut hidden_layers: Vec<Layer> = Vec::new();

        for i in 0..num_hidden_layers {
            if i == 0 {
                // input layer
                hidden_layers.push(Layer::new(input_size, hidden_dim, Activation::ReLU));
            } else {
                // remaining layers
                hidden_layers.push(Layer::new(hidden_dim, hidden_dim, Activation::ReLU));
            }
        }

        Self {
            config,
            embedding: vec![vec![0.0; embedding_dim]; vs],
            hidden_layers,
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

    fn train_step(&mut self, context: &[u16], target: u16, learning_rate: f32) -> f32 {
        // =====================================================================
        // BLOCK 1: FORWARD PASS THROUGH HIDDEN LAYERS (with caching)
        // =====================================================================
        // Start by concatenating embeddings for every token in the context.
        //
        // TRACE: context = [0], embedding[0] = [1.0, 0.0]
        //        neurons = [1.0, 0.0]
        let mut neurons: Vec<f32> = context
            .iter()
            .flat_map(|&t| self.embedding[t as usize].clone())
            .collect();

        // Cache what goes IN and OUT of each hidden layer.
        // We need these later for backprop.
        //
        // TRACE (after hidden layer 0):
        //   hidden_inputs[0]  = [1.0, 0.0]
        //   hidden_outputs[0] = [1.0, 0.0, 1.0]   (ReLU applied)
        let mut hidden_inputs: Vec<Vec<f32>> = Vec::new();
        let mut hidden_outputs: Vec<Vec<f32>> = Vec::new();

        for layer in &self.hidden_layers {
            hidden_inputs.push(neurons.clone());
            neurons = layer.forward(&neurons);
            hidden_outputs.push(neurons.clone());
        }

        // =====================================================================
        // BLOCK 2: FORWARD PASS THROUGH OUTPUT LAYER
        // =====================================================================
        // Output layer has no activation (linear).
        // Cache the input because backprop needs it to compute weight grads.
        //
        // TRACE: output_input = [1.0, 0.0, 1.0]
        //        logits = [2.0, 0.0, 0.0, 0.0]
        let output_input = neurons.clone();
        let logits = self.output_layer.forward(&output_input);

        // =====================================================================
        // BLOCK 3: SOFTMAX + CROSS-ENTROPY LOSS
        // =====================================================================
        // Softmax turns logits into probabilities that sum to 1.0.
        // Cross-entropy measures how far the prediction is from the target.
        //
        // TRACE: probs  ≈ [0.878, 0.041, 0.041, 0.041]
        //        loss   ≈ 3.199   (for target = 1)
        let probs = calc::softmax(&logits);
        let loss = calc::cross_entropy_loss(&probs, target as usize);

        // =====================================================================
        // BLOCK 4: GRADIENT OF THE LOSS W.R.T. LOGITS
        // =====================================================================
        // cross_entropy_gradient returns: d_logits = probs - one_hot(target)
        //
        // TRACE: d_logits ≈ [0.878, -0.959, 0.041, 0.041]
        let d_logits = calc::cross_entropy_gradient(&probs, target as usize);

        // =====================================================================
        // BLOCK 5: BACKPROP THROUGH OUTPUT LAYER (no activation)
        // =====================================================================
        // Propagate the error upward by multiplying with the transposed weights.
        //
        // formula: d_above[k] = sum_j( W_output[j][k] * d_logits[j] )
        //
        // TRACE: d_above ≈ [0.919, -0.918, 0.837]
        //        size = hidden_dim (3)
        let mut d_above: Vec<f32> = vec![0.0; output_input.len()];
        for (j, neuron_weights) in self.output_layer.weights.iter().enumerate() {
            for (k, &w) in neuron_weights.iter().enumerate() {
                d_above[k] += w * d_logits[j];
            }
        }

        // =====================================================================
        // BLOCK 6: BACKPROP THROUGH HIDDEN LAYERS (reverse order)
        // =====================================================================
        // Walk backward from the last hidden layer to the first.
        // For each ReLU layer:
        //   1. Build a mask: 1.0 where output > 0, 0.0 otherwise
        //   2. Kill gradients of neurons that were zeroed by ReLU
        //   3. Propagate upward: d_next = W^T * d_hidden
        //
        // TRACE (layer 0):
        //   relu_mask = [1.0, 0.0, 1.0]
        //   d_hidden  = [0.919, 0.0, 0.837]
        //   d_next (to embedding) = [1.756, 0.837]
        let mut d_hidden_grads: Vec<Vec<f32>> = Vec::new();

        for i in (0..self.hidden_layers.len()).rev() {
            let h_out = &hidden_outputs[i];

            // ReLU derivative: 1.0 if value > 0, else 0.0
            let relu_mask: Vec<f32> = h_out
                .iter()
                .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
                .collect();

            // Zero out gradients where ReLU killed the neuron
            let d_hidden: Vec<f32> = d_above
                .iter()
                .zip(&relu_mask)
                .map(|(&g, &m)| g * m)
                .collect();
            d_hidden_grads.push(d_hidden.clone());

            // Propagate gradient upward to the previous layer (or embedding)
            let h_in = &hidden_inputs[i];
            let mut d_next: Vec<f32> = vec![0.0; h_in.len()];
            for (j, neuron_weights) in self.hidden_layers[i].weights.iter().enumerate() {
                for (k, &w) in neuron_weights.iter().enumerate() {
                    d_next[k] += w * d_hidden[j];
                }
            }
            d_above = d_next;
        }

        // Reverse so d_hidden_grads[i] aligns with hidden layer i
        d_hidden_grads.reverse();

        // =====================================================================
        // BLOCK 7: EMBEDDING GRADIENT
        // =====================================================================
        // d_above now has size = context_size * embedding_dim.
        // Split into chunks of embedding_dim, one chunk per context token.
        //
        // TRACE: d_embedding[0] = [1.756, 0.837]
        //        d_embedding[1..3] = [0.0, 0.0]
        let mut d_embedding: Vec<Vec<f32>> =
            vec![vec![0.0; self.config.embedding_dim]; self.config.vocab_size as usize];

        for (pos, &token) in context.iter().enumerate() {
            let start = pos * self.config.embedding_dim;
            for k in 0..self.config.embedding_dim {
                d_embedding[token as usize][k] += d_above[start + k];
            }
        }

        // =====================================================================
        // BLOCK 8: SGD UPDATE (weight -= lr * grad)
        // =====================================================================

        // --- Output layer weights ---
        // grad_W[j][k] = d_logits[j] * output_input[k]
        for (j, neuron_weights) in self.output_layer.weights.iter_mut().enumerate() {
            for (k, w) in neuron_weights.iter_mut().enumerate() {
                *w -= learning_rate * d_logits[j] * output_input[k];
            }
        }

        // --- Hidden layer weights ---
        // grad_W[j][k] = d_hidden[j] * hidden_input[k]
        for (i, layer) in self.hidden_layers.iter_mut().enumerate() {
            let h_in = &hidden_inputs[i];
            let d_hidden = &d_hidden_grads[i];
            for (j, neuron_weights) in layer.weights.iter_mut().enumerate() {
                for (k, w) in neuron_weights.iter_mut().enumerate() {
                    *w -= learning_rate * d_hidden[j] * h_in[k];
                }
            }
        }

        // --- Embeddings ---
        // grad = d_embedding[token][k]
        // avoid re-visiting tokens by only updating each unique token once.
        for token in context
            .iter()
            .copied()
            .collect::<std::collections::HashSet<u16>>()
        {
            for (embedding, grad) in self.embedding[token as usize]
                .iter_mut()
                .zip(&d_embedding[token as usize])
            {
                *embedding -= learning_rate * grad;
            }
        }

        // =====================================================================
        // BLOCK 9: RETURN LOSS FOR THIS EXAMPLE
        // =====================================================================
        loss
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
            num_hidden_layers: 1,
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

    #[test]
    fn test_mlp_forward_with_two_hidden_layers() {
        let mut mlp = Mlp::new(MlpConfig {
            vocab_size: 4,
            context_size: 1,
            embedding_dim: 2,
            hidden_dim: 3,
            num_hidden_layers: 2,
        });

        mlp.embedding = vec![
            vec![1.0, 0.0], // Token 0
            vec![0.0, 1.0], // Token 1
        ];

        mlp.hidden_layers[0].weights = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        mlp.hidden_layers[1].weights = vec![
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, -1.0],
        ];

        mlp.output_layer.weights = vec![
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, -1.0],
            vec![0.0, 0.0, 0.0],
        ];

        let logits = mlp.forward(&[1]);

        assert_eq!(logits, vec![1.0, 1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_mlp_train_step_reduces_loss_with_known_weights() {
        let mut mlp = Mlp::new(MlpConfig {
            vocab_size: 4,
            context_size: 1,
            embedding_dim: 2,
            hidden_dim: 3,
            num_hidden_layers: 1,
        });

        mlp.embedding = vec![
            vec![1.0, 0.0], // Token 0
            vec![0.0, 1.0], // Token 1
        ];

        mlp.hidden_layers[0].weights = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        mlp.output_layer.weights = vec![
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, -1.0],
            vec![0.0, 0.0, 0.0],
        ];

        let loss_before = mlp.train_step(&[0], 1, 1.0);
        let loss_after = mlp.train_step(&[0], 1, 1.0);

        println!(
            "Loss before: {:.4}, Loss after: {:.4}",
            loss_before, loss_after
        );
        assert!(loss_after < loss_before);
    }

    #[test]
    fn test_mlp_new_chains_hidden_layer_input_dims() {
        let mlp = Mlp::new(MlpConfig {
            vocab_size: 4,
            context_size: 1,
            embedding_dim: 2,
            hidden_dim: 3,
            num_hidden_layers: 2,
        });

        // Layer 1 receives layer 0's output (hidden_dim), not the embedding input.
        assert_eq!(mlp.hidden_layers[1].weights[0].len(), 3);
    }

    #[test]
    fn test_mlp_train_step_repeated_token_updates_embedding_once() {
        let mut mlp = Mlp::new(MlpConfig {
            vocab_size: 2,
            context_size: 2,
            embedding_dim: 1,
            hidden_dim: 1,
            num_hidden_layers: 1,
        });

        mlp.embedding = vec![vec![1.0], vec![1.0]];
        mlp.hidden_layers[0].weights = vec![vec![1.0, 1.0]];
        mlp.output_layer.weights = vec![vec![1.0], vec![0.0]];

        mlp.train_step(&[0, 0], 1, 1.0);

        eprintln!("DEBUG emb[0][0] = {}", mlp.embedding[0][0]);
        assert!((mlp.embedding[0][0] - (-0.7616)).abs() < 1e-3);
    }
}
