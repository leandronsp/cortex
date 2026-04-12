use super::*;

#[test]
fn forward_returns_row_for_token_id() {
    let mut embedding = Embedding::new();
    embedding.add_token(256, vec![0.1, 0.2, 0.3]);
    embedding.add_token(257, vec![0.4, 0.5, 0.6]);

    let result = embedding.forward(256);
    assert_eq!(result, Some(vec![0.1, 0.2, 0.3]));

    let result = embedding.forward(257);
    assert_eq!(result, Some(vec![0.4, 0.5, 0.6]));
}

#[test]
fn forward_returns_none_for_unknown_token_id() {
    let embedding = Embedding::new();
    let result = embedding.forward(999);
    assert_eq!(result, None);
}

#[test]
fn forward_sequence_returns_vectors_for_each_id() {
    let mut embedding = Embedding::new();
    embedding.add_token(256, vec![0.1, 0.2, 0.3]);
    embedding.add_token(257, vec![0.4, 0.5, 0.6]);

    let ids = vec![256, 257];
    let result: Vec<Option<Vec<f32>>> = ids.iter().map(|&id| embedding.forward(id)).collect();
    assert_eq!(
        result,
        vec![Some(vec![0.1, 0.2, 0.3]), Some(vec![0.4, 0.5, 0.6])]
    );

    let ids = vec![256, 257, 256];
    let result: Vec<Option<Vec<f32>>> = ids.iter().map(|&id| embedding.forward(id)).collect();
    assert_eq!(
        result,
        vec![
            Some(vec![0.1, 0.2, 0.3]),
            Some(vec![0.4, 0.5, 0.6]),
            Some(vec![0.1, 0.2, 0.3])
        ]
    );
}
