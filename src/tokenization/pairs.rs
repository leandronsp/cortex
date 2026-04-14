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
mod tests {
    use super::*;
    use crate::tokenization::tokenizer;

    #[test]
    fn count_pairs_in_tokens() {
        let tokens = tokenizer::encode("aaab");
        let counts = count(&tokens);
        assert_eq!(counts[&(97, 97)], 2);
        assert_eq!(counts[&(97, 98)], 1);
    }

    #[test]
    fn count_more_complex_pairs() {
        let tokens = tokenizer::encode("abacate");
        let counts = count(&tokens);
        assert_eq!(counts.len(), 6);
        assert_eq!(counts[&(97, 98)], 1);
        assert_eq!(counts[&(98, 97)], 1);
        assert_eq!(counts[&(97, 99)], 1);
        assert_eq!(counts[&(99, 97)], 1);
        assert_eq!(counts[&(97, 116)], 1);
        assert_eq!(counts[&(116, 101)], 1);
    }

    #[test]
    fn count_pairs_returns_empty_for_single_token() {
        let tokens = tokenizer::encode("a");
        let counts = count(&tokens);
        assert!(counts.is_empty());
    }

    #[test]
    fn count_pairs_returns_empty_for_empty_tokens() {
        let tokens = tokenizer::encode("");
        let counts = count(&tokens);
        assert!(counts.is_empty());
    }

    #[test]
    fn most_frequent_pair() {
        let tokens = tokenizer::encode("aaab");
        let counts = count(&tokens);
        assert_eq!(most_frequent(&counts), Some((97, 97)));
    }

    #[test]
    fn most_frequent_pair_returns_none_for_empty() {
        let counts = count(&[]);
        assert_eq!(most_frequent(&counts), None);
    }

    #[test]
    fn most_frequent_pair_single_pair() {
        let tokens = tokenizer::encode("ab");
        let counts = count(&tokens);
        assert_eq!(most_frequent(&counts), Some((97, 98)));
    }

    #[test]
    fn most_frequent_pair_all_equal() {
        let tokens = tokenizer::encode("aaaa");
        let counts = count(&tokens);
        assert_eq!(most_frequent(&counts), Some((97, 97)));
    }

    #[test]
    fn most_frequent_pair_in_aabbccaa() {
        let tokens = tokenizer::encode("aabbccaa");
        let counts = count(&tokens);
        assert_eq!(most_frequent(&counts), Some((97, 97)));
    }

    #[test]
    fn most_frequent_pair_in_leanndddro() {
        let tokens = tokenizer::encode("leanndddro");
        let counts = count(&tokens);
        assert_eq!(most_frequent(&counts), Some((100, 100)));
    }

    // Tied pairs: smallest pair lexicographically wins (minbpe convention)
    #[test]
    fn most_frequent_pair_tiebreak_smallest_pair() {
        let tokens = tokenizer::encode("abcd");
        let counts = count(&tokens);
        assert_eq!(most_frequent(&counts), Some((97, 98)));
    }

    #[test]
    fn most_frequent_pair_in_aspirina() {
        let tokens = tokenizer::encode("aspirina");
        let counts = count(&tokens);
        assert_eq!(most_frequent(&counts), Some((97, 115)));
    }
}
