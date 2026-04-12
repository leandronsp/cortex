mod bpe;
mod embedding;
mod merge;
mod pairs;
mod tokenizer;

use bpe::Bpe;

fn main() {
    let mut bpe = Bpe::new();
    let input = "the cat sat on the mat";
    let tokens = bpe.encode(input);
    println!("Input: {:?}", input);
    println!("Encoded: {:?} ({} tokens)", tokens, tokens.len());

    for n in 1..=21 {
        let trained = bpe.train(&tokens, n); // build vocabulary
        println!(
            "After {:2} merges: {:?} ({} tokens)",
            n,
            trained,
            trained.len()
        );
        if trained.len() == 1 {
            break;
        }
    }
}
