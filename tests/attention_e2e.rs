use cortex::config::Config;
use cortex::model::registry::create_model;
use cortex::Cortex;

#[test]
fn attention_trains_and_loss_drops() {
    let config = Config::from_path("configs/attention.toml").expect("config");
    let model = create_model(&config.model).expect("model");
    let corpus = std::fs::read_to_string(&config.training.corpus).expect("corpus");

    // Fewer epochs than production: prove the pipeline wires up and learns,
    // not reproduce the full training budget.
    let mut cortex = Cortex::new(model);
    let report = cortex.train(&corpus, 10, config.training.learning_rate);

    // Loss drops from the uniform baseline.
    assert!(report.last_avg_loss < report.first_avg_loss);

    let out = cortex.generate("the ", 20, 1, 1.0);
    assert!(!out.is_empty());
}
