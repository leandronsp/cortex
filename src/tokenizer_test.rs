use super::*;

#[test]
fn encode_returns_bytes_of_input() {
    let tokens = encode("hi");
    assert_eq!(tokens, vec![104, 105]);
}

#[test]
fn encode_returns_empty_vector_for_empty_input() {
    let tokens = encode("");
    assert_eq!(tokens, vec![]);
}

#[test]
fn encode_multibyte_utf8() {
    let tokens = encode("café");
    // 'c'=99, 'a'=97, 'f'=102, 'é'=0xC3 0xA9 (2 bytes in UTF-8)
    assert_eq!(tokens, vec![99, 97, 102, 195, 169]);
}

#[test]
fn encode_with_spaces() {
    let tokens = encode("a b");
    assert_eq!(tokens, vec![97, 32, 98]);
}

#[test]
fn encode_only_spaces() {
    let tokens = encode("   ");
    assert_eq!(tokens, vec![32, 32, 32]);
}

#[test]
fn encode_with_newline() {
    let tokens = encode("a\nb");
    assert_eq!(tokens, vec![97, 10, 98]);
}

#[test]
fn encode_numbers_as_text() {
    let tokens = encode("123");
    assert_eq!(tokens, vec![49, 50, 51]);
}

#[test]
fn encode_hello_world() {
    let tokens = encode("hello world");
    assert_eq!(tokens.len(), 11);
    assert_eq!(tokens[0], 104); // 'h'
    assert_eq!(tokens[5], 32);  // ' '
    assert_eq!(tokens[6], 119); // 'w'
}
