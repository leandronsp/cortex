use super::*;
use crate::bigram::Bigram;

fn make_cortex() -> Cortex {
    Cortex::new(Box::new(Bigram::new(256)))
}

#[test]
fn train_reduces_average_loss() {
    let mut cortex = make_cortex();
    let report = cortex.train("hello world hello world hello world", 10, 1.0);
    assert!(report.first_avg_loss > report.last_avg_loss);
}

#[test]
fn generate_caps_at_max_tokens() {
    let mut cortex = make_cortex();
    cortex.train("abcabcabc", 20, 1.0);
    let out = cortex.generate("a", 5);
    assert!(out.chars().count() <= 5);
}

#[test]
fn generate_stops_on_newline() {
    let mut cortex = make_cortex();
    cortex.train("hi\n", 100, 1.0);
    let out = cortex.generate("h", 100);
    assert!(
        out.ends_with('\n') || out.chars().count() == 100,
        "got: {:?}",
        out
    );
    if out.ends_with('\n') {
        assert!(out.chars().count() < 100);
    }
}

#[test]
fn save_load_round_trip_preserves_generation() {
    let mut source = make_cortex();
    source.train("the quick brown fox", 30, 1.0);
    let expected = source.generate("t", 8);

    let mut buf: Vec<u8> = Vec::new();
    source.save(&mut buf).unwrap();

    let mut restored = Cortex::new(Box::new(Bigram::new(256)));
    restored.load(&mut buf.as_slice()).unwrap();

    assert_eq!(restored.generate("t", 8), expected);
}
