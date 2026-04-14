pub fn merge(tokens: &[u16], pair: (u16, u16), new_token: u16) -> Vec<u16> {
    let mut result = Vec::new();
    let mut rest = tokens;

    while !rest.is_empty() {
        match rest {
            [a, b, ..] if (*a, *b) == pair => {
                result.push(new_token);
                rest = &rest[2..];
            }
            [a, ..] => {
                result.push(*a);
                rest = &rest[1..];
            }
            _ => {}
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_tokens() {
        let tokens = vec![97, 97, 98]; // "aab"
        let result = merge(&tokens, (97, 97), 256);
        assert_eq!(result, vec![256, 98]);
    }

    #[test]
    fn merge_tokens_with_multiple_occurrences() {
        let tokens = vec![97, 97, 97, 98]; // "aaab"
        let result = merge(&tokens, (97, 97), 256);
        assert_eq!(result, vec![256, 97, 98]);
    }

    #[test]
    fn merge_tokens_with_pair_in_the_end() {
        let tokens = vec![98, 97, 97];
        let result = merge(&tokens, (97, 97), 256);
        assert_eq!(result, vec![98, 256]);
    }

    #[test]
    fn merge_tokens_with_two_distinct_occurencies() {
        let tokens = vec![97, 97, 98, 97, 97];
        let result = merge(&tokens, (97, 97), 256);
        assert_eq!(result, vec![256, 98, 256]);
    }

    #[test]
    fn merge_tokens_with_no_occurrences() {
        let tokens = vec![97, 98, 99];
        let result = merge(&tokens, (97, 97), 256);
        assert_eq!(result, vec![97, 98, 99]);
    }

    #[test]
    fn merge_tokens_with_single_element() {
        let tokens = vec![97];
        let result = merge(&tokens, (97, 97), 256);
        assert_eq!(result, vec![97]);
    }

    #[test]
    fn merge_tokens_with_empty_tokens() {
        let tokens = vec![];
        let result = merge(&tokens, (97, 97), 256);
        assert_eq!(result, vec![]);
    }

    #[test]
    fn merge_tokens_with_repeated_pairs() {
        let tokens = vec![97, 97, 97, 97]; // "aaaa"
        let result = merge(&tokens, (97, 97), 256);
        assert_eq!(result, vec![256, 256]);
    }
}
