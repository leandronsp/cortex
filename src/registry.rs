use crate::bigram::Bigram;
use crate::config::ModelSection;
use crate::model::Model;

const AVAILABLE: &[&str] = &["bigram"];

pub fn create_model(section: &ModelSection) -> Result<Box<dyn Model>, String> {
    match section.name.as_str() {
        "bigram" => Ok(Box::new(Bigram::new(section.vocab_size))),
        other => Err(format!(
            "unknown model {:?}. available: {:?}",
            other, AVAILABLE
        )),
    }
}

#[cfg(test)]
#[path = "registry_test.rs"]
mod tests;
