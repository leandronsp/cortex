use std::env;
use std::fs::File;
use std::io::BufWriter;
use std::process::ExitCode;

use cortex::config::Config;
use cortex::registry::create_model;
use cortex::Cortex;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let Some(config_path) = args.get(1) else {
        eprintln!("usage: train <config.toml>");
        return ExitCode::from(2);
    };

    let config = match Config::from_path(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("config error: {}", e);
            return ExitCode::from(1);
        }
    };

    let model = match create_model(&config.model) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{}", e);
            return ExitCode::from(1);
        }
    };

    let corpus = match std::fs::read_to_string(&config.training.corpus) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "failed to read corpus {}: {}",
                config.training.corpus, e
            );
            return ExitCode::from(1);
        }
    };

    let mut cortex = Cortex::new(model);
    let report = cortex.train(
        &corpus,
        config.training.epochs,
        config.training.learning_rate,
    );
    println!(
        "Trained {} tokens over {} epochs. first_avg_loss={:.4} last_avg_loss={:.4}",
        report.token_count, report.epochs, report.first_avg_loss, report.last_avg_loss
    );

    let weights_file = match File::create(&config.weights.path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("failed to open weights {}: {}", config.weights.path, e);
            return ExitCode::from(1);
        }
    };
    let mut writer = BufWriter::new(weights_file);
    if let Err(e) = cortex.save(&mut writer) {
        eprintln!("failed to write weights: {}", e);
        return ExitCode::from(1);
    }
    println!("Wrote weights to {}", config.weights.path);
    ExitCode::SUCCESS
}
