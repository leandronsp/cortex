use crate::model::bigram::Bigram;
use crate::model::mlp::{Mlp, MlpConfig};
use crate::config::ModelSection;
use super::Model;

const AVAILABLE: &[&str] = &["bigram", "mlp"];

pub fn create_model(section: &ModelSection) -> Result<Box<dyn Model>, String> {
    match section.name.as_str() {
        "bigram" => Ok(Box::new(Bigram::new(section.vocab_size))),
        "mlp" => Ok(Box::new(Mlp::new(MlpConfig {
            vocab_size: section.vocab_size,
            context_size: required(section.context_size, "context_size")?,
            embedding_dim: required(section.embedding_dim, "embedding_dim")?,
            hidden_dim: required(section.hidden_dim, "hidden_dim")?,
            num_hidden_layers: required(section.num_hidden_layers, "num_hidden_layers")?,
        }))),
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
