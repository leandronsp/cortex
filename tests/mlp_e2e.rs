use cortex::config::Config;
use cortex::model::registry::create_model;
use cortex::Cortex;

#[test]
#[ignore]
fn mlp_trains_end_to_end_and_generates() {
    let config = Config::from_path("configs/mlp.toml").expect("config");
    let model = create_model(&config.model).expect("model");
    let corpus = std::fs::read_to_string(&config.training.corpus).expect("corpus");

    // Fewer epochs than the config: this proves the pipeline wires up and
    // learns, not the production training budget. Keeps the suite fast.
    let mut cortex = Cortex::new(model);
    let report = cortex.train(&corpus, 50, config.training.learning_rate);

    // The pipeline learns: loss drops from the uniform baseline.
    assert!(report.last_avg_loss < report.first_avg_loss);

    let out = cortex.generate("the ", 40, 1, 1.0);
    assert!(!out.is_empty());
}
