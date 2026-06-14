use crate::model::bigram::Bigram;
use crate::model::mlp::{Mlp, MlpConfig};
use crate::config::ModelSection;
use crate::training::calc;
use super::Model;

const AVAILABLE: &[&str] = &["bigram", "mlp"];

// Fixed seed keeps initialization reproducible across runs.
const INIT_SEED: u64 = 0x5EED;

pub fn create_model(section: &ModelSection) -> Result<Box<dyn Model>, String> {
    match section.name.as_str() {
        "bigram" => Ok(Box::new(Bigram::new(section.vocab_size))),
        "mlp" => {
            let mut mlp = Mlp::new(MlpConfig {
                vocab_size: section.vocab_size,
                context_size: required(section.context_size, "context_size")?,
                embedding_dim: required(section.embedding_dim, "embedding_dim")?,
                hidden_dim: required(section.hidden_dim, "hidden_dim")?,
                num_hidden_layers: required(section.num_hidden_layers, "num_hidden_layers")?,
            });
            mlp.init_weights(&mut calc::Rng::new(INIT_SEED));
            Ok(Box::new(mlp))
        }
        other => Err(format!(
            "unknown model {:?}. available: {:?}",
            other, AVAILABLE
        )),
    }
}

fn required(value: Option<usize>, field: &str) -> Result<usize, String> {
    value.ok_or_else(|| format!("model \"mlp\" requires {field}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelSection;

    #[test]
    fn mlp_from_registry_learns_after_training() {
        let section = ModelSection {
            name: "mlp".to_string(),
            vocab_size: 8,
            context_size: Some(2),
            embedding_dim: Some(4),
            hidden_dim: Some(8),
            num_hidden_layers: Some(1),
        };

        let mut model = create_model(&section).unwrap();
        for _ in 0..200 {
            model.train_step(&[1, 2], 3, 0.5);
        }

        let logits = model.forward(&[1, 2]);
        let predicted = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(predicted, 3);
    }

    #[test]
    fn creates_bigram_by_name() {
        let section = ModelSection {
            name: "bigram".to_string(),
            vocab_size: 64,
            ..Default::default()
        };
        let model = create_model(&section).unwrap();
        assert_eq!(model.vocab_size(), 64);
    }

    #[test]
    fn creates_mlp_by_name() {
        let section = ModelSection {
            name: "mlp".to_string(),
            vocab_size: 64,
            context_size: Some(3),
            embedding_dim: Some(16),
            hidden_dim: Some(32),
            num_hidden_layers: Some(2),
        };
        let model = create_model(&section).unwrap();
        assert_eq!(model.vocab_size(), 64);
        assert_eq!(model.context_size(), 3);
    }

    #[test]
    fn mlp_missing_field_errors_naming_field() {
        let section = ModelSection {
            name: "mlp".to_string(),
            vocab_size: 64,
            context_size: Some(3),
            embedding_dim: None,
            hidden_dim: Some(32),
            num_hidden_layers: Some(2),
        };
        let err = match create_model(&section) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(err.contains("embedding_dim"));
    }

    #[test]
    fn unknown_model_errors_with_available_names() {
        let section = ModelSection {
            name: "transformer".to_string(),
            vocab_size: 256,
            ..Default::default()
        };
        let err = match create_model(&section) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(err.contains("transformer"));
        assert!(err.contains("bigram"));
    }
}
