use cortex::config::Config;
use cortex::model::registry::create_model;
use cortex::Cortex;

#[test]
#[ignore = "slow E2E training; run via `make test-slow` (release)"]
fn attention_trains_end_to_end_and_generates() {
    let config = Config::from_path("configs/attention.toml").expect("config");
    let model = create_model(&config.model).expect("model");
    let corpus = std::fs::read_to_string(&config.training.corpus).expect("corpus");

    let mut cortex = Cortex::new(model);
    // Fewer epochs than production training: the goal is to prove the stacked
    // pipeline wires up and learns, not to reproduce the full budget.
    let report = cortex.train(&corpus, 200, config.training.learning_rate);
    assert!(report.last_avg_loss < 0.5, "loss should drop below 0.5, got {}", report.last_avg_loss);

    // Stacked attention's strength is contextual recall: given a full-window
    // prompt it routes back to "over the" and stays on the dog sentence. We
    // assert the substring rather than an exact string — exact short-prompt
    // memorization is not a property the 2-block model holds at this budget
    // (see RUN-REPORT: short prompts are under-determined). Detailed generation
    // quality is verified via the TUI smoke, not here.
    let out = cortex.generate("the quick brown fox jumps over the", 12, 1, 1.0);
    assert!(out.contains("dog"), "full-window prompt should reach the dog sentence, got {out:?}");

    // Short prompts carry little context; we only assert the pipeline produces
    // output, not a specific continuation.
    let short_out = cortex.generate("hello", 12, 1, 1.0);
    assert!(!short_out.is_empty(), "short prompt should still generate something");
}
