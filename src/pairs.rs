use std::{cmp::Reverse, collections::HashMap};

pub fn count(tokens: &[u16]) -> HashMap<(u16, u16), usize> {
    let mut counts = HashMap::new();
    for pair in tokens.windows(2) {
        if let [a, b] = pair {
            *counts.entry((*a, *b)).or_insert(0) += 1;
        }
    }
    counts
}

pub fn most_frequent(counts: &HashMap<(u16, u16), usize>) -> Option<(u16, u16)> {
    counts
        .iter()
        .max_by_key(|&(pair, count)| (count, Reverse(pair)))
        .map(|(pair, _)| *pair)
}

#[cfg(test)]
#[path = "pairs_test.rs"]
mod tests;
