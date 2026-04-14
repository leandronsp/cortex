use super::*;
use crate::config::ModelSection;

#[test]
fn creates_bigram_by_name() {
    let section = ModelSection {
        name: "bigram".to_string(),
        vocab_size: 64,
    };
    let model = create_model(&section).unwrap();
    assert_eq!(model.vocab_size(), 64);
}

#[test]
fn unknown_model_errors_with_available_names() {
    let section = ModelSection {
        name: "transformer".to_string(),
        vocab_size: 256,
    };
    let err = match create_model(&section) {
        Ok(_) => panic!("expected error"),
        Err(e) => e,
    };
    assert!(err.contains("transformer"));
    assert!(err.contains("bigram"));
}
