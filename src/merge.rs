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
#[path = "merge_test.rs"]
mod tests;
