use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn bin_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("target");
    p.push("debug");
    p.push(name);
    p
}

fn tmp_dir(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "cortex-train-{}-{}",
        tag,
        std::process::id()
    ));
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn train_writes_weights_from_config() {
    let dir = tmp_dir("happy");
    let corpus_path = dir.join("corpus.txt");
    let weights_path = dir.join("weights.bin");
    let config_path = dir.join("run.toml");

    fs::write(&corpus_path, "abcabcabc").unwrap();
    let toml = format!(
        r#"
[model]
name = "bigram"
vocab_size = 256

[training]
corpus = "{}"
epochs = 5
learning_rate = 1.0

[weights]
path = "{}"
"#,
        corpus_path.display(),
        weights_path.display()
    );
    fs::write(&config_path, toml).unwrap();

    let status = Command::new(bin_path("train"))
        .arg(&config_path)
        .status()
        .unwrap();
    assert!(status.success());
    assert!(weights_path.exists());
    assert!(fs::metadata(&weights_path).unwrap().len() > 0);
}

#[test]
fn train_missing_config_exits_nonzero() {
    let status = Command::new(bin_path("train"))
        .arg("/tmp/cortex-absent-config.toml")
        .status()
        .unwrap();
    assert!(!status.success());
}

#[test]
fn train_unknown_model_exits_nonzero() {
    let dir = tmp_dir("unknown-model");
    let config_path = dir.join("run.toml");
    fs::write(
        &config_path,
        r#"
[model]
name = "mlp"
vocab_size = 256

[training]
corpus = "/tmp/whatever"
epochs = 1
learning_rate = 1.0

[weights]
path = "/tmp/w.bin"
"#,
    )
    .unwrap();
    let output = Command::new(bin_path("train"))
        .arg(&config_path)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("mlp"));
    assert!(stderr.contains("bigram"));
}

#[test]
fn train_missing_corpus_exits_nonzero_naming_path() {
    let dir = tmp_dir("missing-corpus");
    let config_path = dir.join("run.toml");
    fs::write(
        &config_path,
        r#"
[model]
name = "bigram"
vocab_size = 256

[training]
corpus = "/tmp/cortex-absent-corpus.txt"
epochs = 1
learning_rate = 1.0

[weights]
path = "/tmp/w.bin"
"#,
    )
    .unwrap();
    let output = Command::new(bin_path("train"))
        .arg(&config_path)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("cortex-absent-corpus.txt"));
}
