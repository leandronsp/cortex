use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::process::ExitCode;

use cortex::Cortex;
use cortex::config::Config;
use cortex::model::registry::create_model;
use cortex::tui;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let Some(config_path) = args.get(1) else {
        eprintln!("usage: chat <config.toml>");
        return ExitCode::from(2);
    };

    let config = match Config::from_path(config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("config error: {}", e);
            return ExitCode::from(1);
        }
    };

    if !Path::new(&config.weights.path).exists() {
        eprintln!(
            "weights file {} not found. run `make train CONFIG={}` first.",
            config.weights.path, config_path
        );
        return ExitCode::from(1);
    }

    let model = match create_model(&config.model) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{}", e);
            return ExitCode::from(1);
        }
    };

    let mut cortex = Cortex::new(model);
    let weights_file = match File::open(&config.weights.path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("failed to open weights {}: {}", config.weights.path, e);
            return ExitCode::from(1);
        }
    };
    let mut reader = BufReader::new(weights_file);
    if let Err(e) = cortex.load(&mut reader) {
        eprintln!("failed to read weights: {}", e);
        return ExitCode::from(1);
    }

    if let Err(e) = tui::run(cortex) {
        eprintln!("tui error: {}", e);
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
