pub fn encode(input: &str) -> Vec<u16> {
    input.as_bytes().iter().map(|&b| b as u16).collect()
}

#[cfg(test)]
#[path = "tokenizer_test.rs"]
mod tests;
