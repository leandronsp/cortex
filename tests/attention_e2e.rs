use cortex::config::Config;
use cortex::model::registry::create_model;
use cortex::Cortex;

#[test]
fn attention_trains_end_to_end_and_generates() {
    let config = Config::from_path("configs/attention.toml").expect("config");
    let model = create_model(&config.model).expect("model");
    let corpus = std::fs::read_to_string(&config.training.corpus).expect("corpus");

    let mut cortex = Cortex::new(model);
    // Fewer epochs than production training: the goal is to prove the pipeline
    // wires up and learns, not to reproduce the full budget. Keeps the suite fast.
    let report = cortex.train(&corpus, 50, config.training.learning_rate);

    // The pipeline learns: loss drops from the uniform baseline.
    assert!(
        report.last_avg_loss < report.first_avg_loss,
        "loss should decrease: first={} last={}",
        report.first_avg_loss,
        report.last_avg_loss
    );

    // Generation produces non-empty output for various prompt lengths.
    let out = cortex.generate("the quick brown fox jumps over the", 12, 1, 1.0);
    assert!(!out.is_empty(), "long prompt should produce output");

    let short_out = cortex.generate("lazy", 12, 1, 1.0);
    assert!(!short_out.is_empty(), "short prompt should produce output");

    let shakespeare_out = cortex.generate("all the", 30, 1, 1.0);
    assert!(!shakespeare_out.is_empty(), "shakespeare prompt should produce output");
}
