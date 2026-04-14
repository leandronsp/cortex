use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub model: ModelSection,
    pub training: TrainingSection,
    pub weights: WeightsSection,
}

#[derive(Debug, Deserialize)]
pub struct ModelSection {
    pub name: String,
    pub vocab_size: u16,
}

#[derive(Debug, Deserialize)]
pub struct TrainingSection {
    pub corpus: String,
    pub epochs: usize,
    pub learning_rate: f32,
}

#[derive(Debug, Deserialize)]
pub struct WeightsSection {
    pub path: String,
}

impl Config {
    pub fn parse(input: &str) -> Result<Self, String> {
        toml::from_str(input).map_err(|e| e.to_string())
    }

    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let bytes = std::fs::read_to_string(path_ref)
            .map_err(|e| format!("failed to read config {}: {}", path_ref.display(), e))?;
        Self::parse(&bytes)
    }
}

#[cfg(test)]
#[path = "config_test.rs"]
mod tests;
