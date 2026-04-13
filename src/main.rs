#![allow(dead_code)]

mod bigram;
mod bpe;
mod calc;
mod embedding;
mod merge;
mod pairs;
mod tokenizer;

use std::io::{self, BufRead, Write};

use bigram::Bigram;
use bpe::Bpe;

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    print!("Training corpus: ");
    io::stdout().flush().unwrap();
    let corpus = lines.next().unwrap().unwrap();

    let bpe = Bpe::new();
    let tokens = bpe.encode(&corpus);

    let vocab_size = 256u16;
    let mut bigram = Bigram::new(vocab_size);

    let learning_rate = 1.0;
    let epochs = 200;

    for _ in 0..epochs {
        for window in tokens.windows(2) {
            let input = window[0];
            let target = window[1] as usize;
            let logits = bigram.forward(input);
            let probs = calc::softmax(&logits);
            let gradient = calc::cross_entropy_gradient(&probs, target);
            bigram.update(input, &gradient, learning_rate);
        }
    }

    println!(
        "Trained on {} tokens. Enter a prompt (Ctrl-D to exit):",
        tokens.len()
    );

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let line = match lines.next() {
            Some(Ok(l)) => l,
            _ => break,
        };

        if line.is_empty() {
            continue;
        }

        let input_byte = line.as_bytes()[0] as u16;
        let probs = calc::softmax(&bigram.forward(input_byte));
        let predicted = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        println!(
            "Input: {:?} ({}) -> Predicted: {:?} ({})",
            line.chars().next().unwrap(),
            input_byte,
            predicted as u8 as char,
            predicted
        );
    }
}
