use cortex::config::Config;
use cortex::model::registry::create_model;
use cortex::Cortex;

#[test]
fn attention_trains_end_to_end_and_generates() {
    let config = Config::from_path("configs/attention.toml").expect("config");
    let model = create_model(&config.model).expect("model");
    let corpus = std::fs::read_to_string(&config.training.corpus).expect("corpus");

    let mut cortex = Cortex::new(model);
    let report = cortex.train(&corpus, 120, config.training.learning_rate);
    // Attention alone plateaus around ~1.2 on this corpus; the full block
    // (attention + FFN + residual) has the capacity to memorize it.
    assert!(report.last_avg_loss < 0.5);

    // The MLP (window 3) saw only "the " here and wrongly continued with
    // "question". Attention (window 8) can look back to "over the" and should
    // stay on the dog sentence.
    let out = cortex.generate("the quick brown fox jumps over the", 12);
    assert!(!out.is_empty());
    eprintln!("ATTENTION GENERATED: {out:?}");
}
