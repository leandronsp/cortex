use super::*;

#[test]
fn parses_valid_config() {
    let input = r#"
[model]
name = "bigram"
vocab_size = 256

[training]
corpus = "data/corpus.txt"
epochs = 200
learning_rate = 1.0

[weights]
path = "target/bigram.bin"
"#;
    let cfg = Config::from_str(input).unwrap();
    assert_eq!(cfg.model.name, "bigram");
    assert_eq!(cfg.model.vocab_size, 256);
    assert_eq!(cfg.training.corpus, "data/corpus.txt");
    assert_eq!(cfg.training.epochs, 200);
    assert_eq!(cfg.training.learning_rate, 1.0);
    assert_eq!(cfg.weights.path, "target/bigram.bin");
}

#[test]
fn malformed_toml_returns_error_with_message() {
    let input = "this is = not = valid toml [[[";
    let err = Config::from_str(input).unwrap_err();
    assert!(!err.is_empty());
}

#[test]
fn missing_required_field_returns_error() {
    let input = r#"
[model]
name = "bigram"
"#;
    let err = Config::from_str(input).unwrap_err();
    assert!(err.contains("vocab_size") || err.contains("training") || err.contains("missing"));
}

#[test]
fn missing_file_returns_error_naming_path() {
    let err = Config::from_path("/tmp/cortex-does-not-exist-xyz.toml").unwrap_err();
    assert!(err.contains("/tmp/cortex-does-not-exist-xyz.toml"));
}
