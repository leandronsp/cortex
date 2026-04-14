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
    p.push(format!("cortex-chat-{}-{}", tag, std::process::id()));
    fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn chat_missing_config_exits_nonzero() {
    let status = Command::new(bin_path("chat"))
        .arg("/tmp/cortex-absent.toml")
        .status()
        .unwrap();
    assert!(!status.success());
}

#[test]
fn chat_missing_weights_tells_user_to_train_first() {
    let dir = tmp_dir("missing-weights");
    let config_path = dir.join("run.toml");
    fs::write(
        &config_path,
        r#"
[model]
name = "bigram"
vocab_size = 256

[training]
corpus = "/tmp/whatever"
epochs = 1
learning_rate = 1.0

[weights]
path = "/tmp/cortex-absent-weights-xyz.bin"
"#,
    )
    .unwrap();
    let output = Command::new(bin_path("chat"))
        .arg(&config_path)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("train"));
}
