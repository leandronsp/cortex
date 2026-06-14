use std::io::{Read, Write};

use crate::model::Model;
use crate::tokenization::Bpe;

/// Special token prepended to every sequence during training and used for
/// left-padding short prompts during generation. It lives just above the
/// byte range so it can never collide with raw text tokens.
const BOS: u16 = 256;

pub struct Cortex {
    bpe: Bpe,
    model: Box<dyn Model>,
}

pub struct TrainReport {
    pub first_avg_loss: f32,
    pub last_avg_loss: f32,
    pub epochs: usize,
    pub token_count: usize,
}

impl Cortex {
    pub fn new(model: Box<dyn Model>) -> Self {
        Self { bpe: Bpe::new(), model }
    }

    pub fn train(&mut self, corpus: &str, epochs: usize, learning_rate: f32) -> TrainReport {
        // Prepend BOS so the model learns what "beginning of text" looks like.
        // This lets short prompts align with a real training position instead
        // of falling into the OOD padding regime of the unseen byte 0.
        let encoded = self.bpe.encode(corpus);
        let mut tokens = Vec::with_capacity(encoded.len() + 1);
        tokens.push(BOS);
        tokens.extend(encoded);
        let context_size = self.model.context_size();
        let mut first_avg_loss = 0.0;
        let mut last_avg_loss = 0.0;
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for window in tokens.windows(context_size + 1) {
                let (context, target) = window.split_at(context_size);
                total_loss += self.model.train_step(context, target[0], learning_rate);
            }
            let avg = total_loss / (tokens.len().saturating_sub(context_size)).max(1) as f32;
            if epoch == 0 {
                first_avg_loss = avg;
            }
            last_avg_loss = avg;
            if epoch % 20 == 0 {
                println!("Epoch {:3}: avg loss = {:.4}", epoch, avg);
            }
        }
        TrainReport {
            first_avg_loss,
            last_avg_loss,
            epochs,
            token_count: tokens.len(),
        }
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> String {
        let mut context = self.bpe.encode(prompt);
        if context.is_empty() {
            return String::new();
        }
        let context_size = self.model.context_size();
        let mut produced: Vec<u8> = Vec::new();
        for _ in 0..max_tokens {
            let window = last_window(&context, context_size);
            let logits = self.model.forward(&window);
            let next = argmax(&logits) as u16;
            let byte = next as u8;
            produced.push(byte);
            context.push(next);
            if byte == b'\n' {
                break;
            }
        }
        String::from_utf8_lossy(&produced).into_owned()
    }

    pub fn save(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        let merges = self.bpe.merges();
        let count = merges.len() as u32;
        writer.write_all(&count.to_le_bytes())?;
        for ((a, b), new_token) in merges {
            writer.write_all(&a.to_le_bytes())?;
            writer.write_all(&b.to_le_bytes())?;
            writer.write_all(&new_token.to_le_bytes())?;
        }
        self.model.save(writer)
    }

    pub fn load(&mut self, reader: &mut dyn Read) -> std::io::Result<()> {
        let mut count_buf = [0u8; 4];
        reader.read_exact(&mut count_buf)?;
        let count = u32::from_le_bytes(count_buf) as usize;
        let mut merges = Vec::with_capacity(count);
        let mut u16_buf = [0u8; 2];
        for _ in 0..count {
            reader.read_exact(&mut u16_buf)?;
            let a = u16::from_le_bytes(u16_buf);
            reader.read_exact(&mut u16_buf)?;
            let b = u16::from_le_bytes(u16_buf);
            reader.read_exact(&mut u16_buf)?;
            let new_token = u16::from_le_bytes(u16_buf);
            merges.push(((a, b), new_token));
        }
        self.bpe.set_merges(merges);
        self.model.load(reader)
    }
}

/// The last `size` tokens of `context`, left-padded with BOS when the context
/// is shorter. The MLP needs exactly `context_size` tokens; the bigram (size 1)
/// just gets the most recent token.
fn last_window(context: &[u16], size: usize) -> Vec<u16> {
    let mut window = vec![BOS; size];
    let take = context.len().min(size);
    window[size - take..].copy_from_slice(&context[context.len() - take..]);
    window
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    use std::io::{Read, Write};

    /// Minimal Model double. Cortex is tested against the interface, never a
    /// concrete model: `forward` predicts `last + 1` for deterministic decode,
    /// and `train_step` returns the context length so the average reveals the
    /// window size Cortex sliced out.
    struct FakeModel {
        context_size: usize,
        steps: f32,
    }

    impl FakeModel {
        fn new(context_size: usize) -> Self {
            Self {
                context_size,
                steps: 0.0,
            }
        }
    }

    impl Model for FakeModel {
        fn vocab_size(&self) -> u16 {
            // Must include BOS, which lives above the byte range.
            BOS + 1
        }

        fn context_size(&self) -> usize {
            self.context_size
        }

        fn forward(&self, context: &[u16]) -> Vec<f32> {
            assert_eq!(
                context.len(),
                self.context_size,
                "engine must feed exactly context_size tokens"
            );
            let last = context.last().copied().unwrap_or(BOS);
            let next = (last.wrapping_add(1)) % self.vocab_size();
            let mut logits = vec![0.0; self.vocab_size() as usize];
            logits[next as usize] = 1.0;
            logits
        }

        fn train_step(&mut self, context: &[u16], _target: u16, _learning_rate: f32) -> f32 {
            self.steps += 1.0;
            context.len() as f32
        }

        fn save(&self, writer: &mut dyn Write) -> std::io::Result<()> {
            writer.write_all(&self.steps.to_le_bytes())
        }

        fn load(&mut self, reader: &mut dyn Read) -> std::io::Result<()> {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            self.steps = f32::from_le_bytes(buf);
            Ok(())
        }
    }

    #[test]
    fn train_feeds_context_size_window_to_model() {
        let mut cortex = Cortex::new(Box::new(FakeModel::new(3)));
        let report = cortex.train("hello world hello world", 1, 1.0);
        // Each window's loss is its context length, so the average equals the
        // window size Cortex fed the model.
        assert_eq!(report.first_avg_loss, 3.0);
    }

    #[test]
    fn generate_caps_at_max_tokens() {
        let cortex = Cortex::new(Box::new(FakeModel::new(1)));
        let out = cortex.generate("a", 5);
        assert_eq!(out.chars().count(), 5);
    }

    #[test]
    fn generate_stops_on_newline() {
        // '\t' (9) makes the model predict '\n' (10), which halts decoding.
        let cortex = Cortex::new(Box::new(FakeModel::new(1)));
        let out = cortex.generate("\t", 100);
        assert_eq!(out, "\n");
    }

    #[test]
    fn generate_pads_and_slides_context_window() {
        // cs=2 with a 1-token prompt: only passes if generate left-pads the
        // short prompt and slides a 2-token window (FakeModel asserts len == 2).
        let cortex = Cortex::new(Box::new(FakeModel::new(2)));
        let out = cortex.generate("a", 3);
        assert_eq!(out.chars().count(), 3);
    }

    #[test]
    fn save_load_round_trip_preserves_generation() {
        let mut source = Cortex::new(Box::new(FakeModel::new(1)));
        source.train("the quick brown fox", 3, 1.0);
        let expected = source.generate("t", 8);

        let mut buf: Vec<u8> = Vec::new();
        source.save(&mut buf).unwrap();

        let mut restored = Cortex::new(Box::new(FakeModel::new(1)));
        restored.load(&mut buf.as_slice()).unwrap();

        assert_eq!(restored.generate("t", 8), expected);
    }

    /// Minimal Model double that records every context it sees. Lets us test
    /// Cortex sequencing/padding decisions without depending on a real model.
    struct RecordingModel {
        context_size: usize,
        contexts: std::rc::Rc<std::cell::RefCell<Vec<Vec<u16>>>>,
    }

    impl RecordingModel {
        fn new(context_size: usize) -> (Self, std::rc::Rc<std::cell::RefCell<Vec<Vec<u16>>>>) {
            let contexts = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
            (
                Self {
                    context_size,
                    contexts: contexts.clone(),
                },
                contexts,
            )
        }
    }

    impl Model for RecordingModel {
        fn vocab_size(&self) -> u16 {
            BOS + 1
        }

        fn context_size(&self) -> usize {
            self.context_size
        }

        fn forward(&self, context: &[u16]) -> Vec<f32> {
            self.contexts.borrow_mut().push(context.to_vec());
            vec![0.0; self.vocab_size() as usize]
        }

        fn train_step(&mut self, context: &[u16], _target: u16, _learning_rate: f32) -> f32 {
            self.contexts.borrow_mut().push(context.to_vec());
            0.0
        }

        fn save(&self, writer: &mut dyn Write) -> std::io::Result<()> {
            writer.write_all(&0f32.to_le_bytes())
        }

        fn load(&mut self, reader: &mut dyn Read) -> std::io::Result<()> {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(())
        }
    }

    #[test]
    fn train_prefixes_corpus_with_bos_token() {
        let (model, contexts) = RecordingModel::new(2);
        let mut cortex = Cortex::new(Box::new(model));
        cortex.train("ab", 1, 1.0);

        let first = &contexts.borrow()[0];
        assert_eq!(first[0], BOS, "first training context must start with BOS");
    }

    #[test]
    fn generate_left_pads_short_prompt_with_bos() {
        let (model, contexts) = RecordingModel::new(4);
        let cortex = Cortex::new(Box::new(model));
        cortex.generate("ab", 1);

        let window = &contexts.borrow()[0];
        assert_eq!(
            window,
            &vec![BOS, BOS, 'a' as u16, 'b' as u16],
            "short prompt must be left-padded with BOS"
        );
    }

    #[test]
    fn generate_does_not_pad_when_prompt_fills_window() {
        let (model, contexts) = RecordingModel::new(2);
        let cortex = Cortex::new(Box::new(model));
        cortex.generate("ab", 1);

        let window = &contexts.borrow()[0];
        assert_eq!(
            window,
            &vec!['a' as u16, 'b' as u16],
            "full-length prompt must not be padded"
        );
    }

}
