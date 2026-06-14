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
    // wires up and memorizes, not to reproduce the full budget.
    let report = cortex.train(&corpus, 200, config.training.learning_rate);
    assert!(report.last_avg_loss < 0.5, "loss should drop below 0.5, got {}", report.last_avg_loss);

    // The MLP (window 3) saw only "the " here and wrongly continued with
    // "question". Attention (window 8) can look back to "over the" and should
    // stay on the dog sentence.
    let out = cortex.generate("the quick brown fox jumps over the", 12);
    assert_eq!(out, " lazy dog\n");

    // Short prompts used to be left-padded with the unseen token 0, which made
    // the model hallucinate. With BOS padding the model should continue from
    // the prompt using a learned start-of-sequence signal.
    let lazy_out = cortex.generate("lazy", 12);
    assert_eq!(lazy_out, " dog\n");

    // The corpus now has a Shakespeare line. A two-word prompt should give
    // enough context to stay on that sentence.
    let shakespeare_out = cortex.generate("all the", 30);
    assert!(
        shakespeare_out.contains("players") || shakespeare_out.contains("men and women"),
        "should continue the shakespeare line, got {shakespeare_out:?}"
    );
}
