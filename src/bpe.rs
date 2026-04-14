use std::collections::HashMap;

use crate::{merge, pairs, tokenizer};

pub struct Bpe {
    merges: Vec<((u16, u16), u16)>,
}

impl Bpe {
    pub fn new() -> Self {
        Self { merges: Vec::new() }
    }

    pub fn encode(&self, input: &str) -> Vec<u16> {
        let mut result = tokenizer::encode(input);

        for (pair, new_token) in &self.merges {
            result = self.merge(&result, *pair, *new_token);
        }

        result
    }

    pub fn count_pairs(&self, tokens: &[u16]) -> HashMap<(u16, u16), usize> {
        pairs::count(tokens)
    }

    pub fn most_frequent_pair(&self, counts: &HashMap<(u16, u16), usize>) -> Option<(u16, u16)> {
        pairs::most_frequent(counts)
    }

    pub fn merge(&self, tokens: &[u16], pair: (u16, u16), new_token: u16) -> Vec<u16> {
        merge::merge(tokens, pair, new_token)
    }

    pub fn merges(&self) -> &[((u16, u16), u16)] {
        &self.merges
    }

    pub fn set_merges(&mut self, merges: Vec<((u16, u16), u16)>) {
        self.merges = merges;
    }

    pub fn build_vocab(&mut self, tokens: &[u16], num_merges: usize) -> Vec<u16> {
        let mut next_token = 256;
        let mut result = tokens.to_vec();

        for _ in 1..=num_merges {
            let counts = self.count_pairs(&result);
            if let Some(pair) = self.most_frequent_pair(&counts) {
                result = self.merge(&result, pair, next_token);
                self.merges.push((pair, next_token));
                next_token += 1;
            }
        }

        result
    }
}

#[cfg(test)]
#[path = "bpe_test.rs"]
mod tests;
