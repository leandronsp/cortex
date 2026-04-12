use super::*;

#[test]
fn bpe_encodes_and_finds_most_frequent_pair() {
    let bpe = Bpe::new();
    let tokens = bpe.encode("aaab");
    let counts = bpe.count_pairs(&tokens);
    let best = bpe.most_frequent_pair(&counts).unwrap();
    assert_eq!(best, (97, 97));
    let merged = bpe.merge(&tokens, best, 256);
    assert_eq!(merged, vec![256, 97, 98]);
}

#[test]
fn bpe_train_with_one_merge() {
    let mut bpe = Bpe::new();
    let tokens = bpe.encode("aaab");
    let trained = bpe.train(&tokens, 1);
    assert_eq!(trained, &[256, 97, 98]);
}

#[test]
fn bpe_train_with_two_merges() {
    let mut bpe = Bpe::new();
    let tokens = bpe.encode("aaab");
    let trained = bpe.train(&tokens, 2);
    assert_eq!(trained, &[256, 257]);
}

#[test]
fn bpe_train_three_merges_collapses_to_single_token() {
    let mut bpe = Bpe::new();
    let tokens = bpe.encode("aaab");
    let trained = bpe.train(&tokens, 3);
    assert_eq!(trained, &[258]);
}

#[test]
fn bpe_train_zero_merges_returns_original() {
    let mut bpe = Bpe::new();
    let tokens = bpe.encode("aaab");
    let trained = bpe.train(&tokens, 0);
    assert_eq!(trained, tokens);
}

#[test]
fn bpe_train_more_merges_than_possible() {
    let mut bpe = Bpe::new();
    let tokens = bpe.encode("aaab");
    let trained = bpe.train(&tokens, 10);
    assert_eq!(trained, &[258]);
}

#[test]
fn bpe_train_with_repeated_pattern() {
    let mut bpe = Bpe::new();
    let tokens = bpe.encode("abababab");
    let trained = bpe.train(&tokens, 1);
    assert_eq!(trained, &[256, 256, 256, 256]);
}

#[test]
fn bpe_train_repeated_pattern_two_merges() {
    let mut bpe = Bpe::new();
    let tokens = bpe.encode("abababab");
    let trained = bpe.train(&tokens, 2);
    assert_eq!(trained, &[257, 257]);
}

#[test]
fn bpe_train_repeated_pattern_three_merges() {
    let mut bpe = Bpe::new();
    let tokens = bpe.encode("aaab");
    let trained = bpe.train(&tokens, 3);
    assert_eq!(trained, &[258]);
}

#[test]
fn bpe_train_single_char_no_pairs() {
    let mut bpe = Bpe::new();
    let tokens = bpe.encode("a");
    let trained = bpe.train(&tokens, 5);
    assert_eq!(trained, &[97]);
}

#[test]
fn bpe_train_empty_input() {
    let mut bpe = Bpe::new();
    let tokens = bpe.encode("");
    let trained = bpe.train(&tokens, 5);
    assert_eq!(trained, &[] as &[u16]);
}

#[test]
fn bpe_train_hello_world() {
    let mut bpe = Bpe::new();
    let tokens = bpe.encode("hello world");
    let trained = bpe.train(&tokens, 1);
    assert_eq!(trained.len(), 10);
}

#[test]
fn bpe_train_the_cat_sat_on_the_mat() {
    let tokens = Bpe::new().encode("the cat sat on the mat");
    assert_eq!(tokens.len(), 22);

    let after_1 = Bpe::new().train(&tokens, 1);
    assert_eq!(after_1.len(), 19);

    let after_2 = Bpe::new().train(&tokens, 2);
    assert_eq!(after_2.len(), 17);

    let after_3 = Bpe::new().train(&tokens, 3);
    assert_eq!(after_3.len(), 15);

    let after_5 = Bpe::new().train(&tokens, 5);
    assert!(after_5.len() > 5);

    // After 15 merges, we should have a single token representing the entire input
    let after_15 = Bpe::new().train(&tokens, 15);
    assert_eq!(after_15, &[270]);

    // Even if we ask for more merges, we should still get the same single token
    let after_22 = Bpe::new().train(&tokens, 22);
    assert_eq!(after_22, &[270]);
}

#[test]
fn bpe_train_stores_merges() {
    let mut bpe = Bpe::new();
    let tokens = bpe.encode("aaab");
    bpe.train(&tokens, 1);
    assert_eq!(bpe.merges.len(), 1);
}

#[test]
fn bpe_train_stores_correct_merges() {
    let mut bpe = Bpe::new();
    let tokens = bpe.encode("aaab");
    bpe.train(&tokens, 4);
    assert_eq!(bpe.merges.len(), 3);
    assert_eq!(bpe.merges[0], ((97, 97), 256));
    assert_eq!(bpe.merges[1], ((97, 98), 257));
    assert_eq!(bpe.merges[2], ((256, 257), 258));
}

#[test]
fn bpe_applies_stored_merges_after_training() {
    let mut bpe = Bpe::new();
    let tokens = bpe.encode("aaab");
    assert_eq!(tokens, vec![97, 97, 97, 98]);

    bpe.train(&tokens, 1);

    assert_eq!(bpe.encode("aaab"), vec![256, 97, 98]);
}

#[test]
fn bpe_applies_stored_merges_after_training_twice() {
    let mut bpe = Bpe::new();
    let tokens = bpe.encode("aaab");
    assert_eq!(tokens, vec![97, 97, 97, 98]);

    bpe.train(&tokens, 2);

    assert_eq!(bpe.encode("aaab"), vec![256, 257]);
}
