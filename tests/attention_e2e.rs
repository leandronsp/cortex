use cortex::config::Config;
use cortex::model::registry::create_model;
use cortex::Cortex;

/// Fast smoke: proves the pipeline wires up (BPE, train, generate) without
/// committing to exact output strings or full training budgets.
#[test]
#[ignore = "integration: run with --ignored"]
fn attention_trains_end_to_end_and_generates() {
    let config = Config::from_path("configs/attention.toml").expect("config");
    let model = create_model(&config.model).expect("model");
    let corpus = std::fs::read_to_string(&config.training.corpus).expect("corpus");

    let mut cortex = Cortex::new(model);
    // Few epochs: check that the loss drops from its initial ~ln(vocab) value.
    let report = cortex.train(&corpus, 5, config.training.learning_rate);
    assert!(report.last_avg_loss < 8.0, "loss should drop below 8.0, got {}", report.last_avg_loss);

    // Output should be non-empty (model learned something, even if not fully converged).
    let out = cortex.generate("the quick brown fox jumps over the", 12, 1, 1.0);
    assert!(!out.is_empty(), "generation should not be empty");

    let short_out = cortex.generate("lazy", 8, 1, 1.0);
    assert!(!short_out.is_empty(), "short-prompt generation should not be empty");

    let med_out = cortex.generate("all the", 20, 1, 1.0);
    assert!(!med_out.is_empty(), "medium-prompt generation should not be empty");
}

/// Verifies save+load round-trip preserves forward output for the attention model.
#[test]
#[ignore = "integration: run with --ignored"]
fn attention_save_load_preserves_forward() {
    use cortex::model::Model;
    use cortex::model::attention::{Attention, AttentionConfig};
    use cortex::training::calc;

    let config = AttentionConfig {
        vocab_size: 8,
        context_size: 3,
        embedding_dim: 4,
        ffn_hidden: 16,
        num_blocks: 2,
    };

    let mut source = Attention::new(config.clone());
    source.init_weights(&mut calc::Rng::new(0xC0FFEE));
    let expected = source.forward(&[1, 2, 3]);

    let mut buf: Vec<u8> = Vec::new();
    Model::save(&source, &mut buf).unwrap();

    let mut restored = Attention::new(config);
    Model::load(&mut restored, &mut buf.as_slice()).unwrap();

    assert_eq!(restored.forward(&[1, 2, 3]), expected);
}
