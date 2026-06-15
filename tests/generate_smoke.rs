use std::fs::File;

use cortex::config::Config;
use cortex::model::registry::create_model;
use cortex::Cortex;

// Validation-only smoke test. Requires target/attention.weights (run
// `make train CONFIG=configs/attention.toml` first). Loads the trained
// model and prints greedy outputs for a mix of prompt sizes so the actual
// generation can be inspected without driving the TUI.
#[test]
#[ignore]
fn generate_smoke_with_various_prompts() {
    let config = Config::from_path("configs/attention.toml").expect("config");
    let model = create_model(&config.model).expect("model");

    let mut cortex = Cortex::new(model);
    let mut file = File::open(&config.weights.path).expect("open weights");
    cortex.load(&mut file).expect("load weights");

    // Mix of one-word, two-word, and long prompts sampled from the corpus.
    // Greedy (top_k=1, temperature=1.0) for deterministic, repeatable output.
    let prompts = [
        ("hello (one word, not in corpus)", "hello"),
        ("lazy (one word, end of dog sentence)", "lazy"),
        ("the quick brown (trigram, dog sentence)", "the quick brown"),
        ("to be (start of question sentence)", "to be"),
        ("the quick brown fox jumps over the (most of dog sentence)", "the quick brown fox jumps over the"),
        ("all the (start of shakespeare line)", "all the"),
    ];

    for (label, prompt) in &prompts {
        let out = cortex.generate(prompt, 30, 1, 1.0);
        eprintln!("PROMPT  : {label:?}  input={prompt:?}");
        eprintln!("STR     : {out:?}");
        eprintln!("BYTES   : {:?}", out.as_bytes());
        eprintln!();
    }
}

