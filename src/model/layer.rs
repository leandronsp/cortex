pub enum Activation {
    ReLU,
    None,
}

pub struct Layer {
    pub weights: Vec<Vec<f32>>,
    pub activation: Activation,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        Self {
            weights: vec![vec![0.0; input_size]; output_size],
            activation,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let raw: Vec<f32> = self.weights
            .iter()
            .map(|neurons| {
                neurons
                    .iter()
                    .zip(input.iter())
                    .map(|(&neuron, &input_val)| neuron * input_val)
                    .sum::<f32>()
            })
            .collect();

        match self.activation {
            Activation::ReLU => raw.iter().map(|&x| x.max(0.0)).collect(),
            Activation::None => raw,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_forward() {
        let mut layer = Layer::new(2, 3, Activation::None);

        layer.weights = vec![vec![1.0, 2.0], vec![0.0, -1.0], vec![3.0, 0.5]];

        let input = vec![1.0, 3.0];
        let output = layer.forward(&input);

        assert_eq!(output, [7.0, -3.0, 4.5]);
    }

    #[test]
    fn test_layer_forward_with_relu() {
        let mut layer = Layer::new(2, 3, Activation::ReLU);

        layer.weights = vec![vec![1.0, 2.0], vec![0.0, -1.0], vec![3.0, 0.5]];

        let input = vec![1.0, 3.0];
        let output = layer.forward(&input);

        assert_eq!(output, [7.0, 0.0, 4.5]);
    }
}
