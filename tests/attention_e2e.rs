use cortex::config::Config;
use cortex::model::registry::create_model;
use cortex::Cortex;

#[test]
fn attention_trains_and_loss_decreases() {
    let config = Config::from_path("configs/attention.toml").expect("config");
    let model = create_model(&config.model).expect("model");
    let corpus = std::fs::read_to_string(&config.training.corpus).expect("corpus");

    let mut cortex = Cortex::new(model);
    let report = cortex.train(&corpus, 10, config.training.learning_rate);
    assert!(
        report.last_avg_loss < report.first_avg_loss,
        "loss should decrease over 10 epochs, first={} last={}",
        report.first_avg_loss,
        report.last_avg_loss,
    );
}

#[test]
fn attention_generates_non_empty_output() {
    let config = Config::from_path("configs/attention.toml").expect("config");
    let model = create_model(&config.model).expect("model");
    let corpus = std::fs::read_to_string(&config.training.corpus).expect("corpus");

    let mut cortex = Cortex::new(model);
    cortex.train(&corpus, 10, config.training.learning_rate);

    let out = cortex.generate("the", 8, 1, 1.0);
    assert!(!out.is_empty(), "model should produce output after training");
}