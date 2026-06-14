use std::collections::HashSet;
use std::io::{Read, Write};

use crate::training::calc;

use super::Model;

// =============================================================================
// SINGLE TRANSFORMER BLOCK: causal self-attention + FFN + residuals
// =============================================================================
// Karpathy step 4 gave us attention (route information from past tokens). On its
// own it plateaus high: the only non-linearity is the attention softmax, so it
// can't actually *compute* much. Step 5 completes the block:
//
//   a = X[t] + attention(X)[t]      <- residual 1: keep the token's own signal
//   h = a   + FFN(a)                <- residual 2: FFN does the heavy non-linear work
//   logits = Wo . h
//
// FFN(a) = W2 . relu(W1 . a): an up-projection, a ReLU, a down-projection. This
// is where most of a transformer's capacity lives. The residuals let gradients
// flow straight through and let the current token reach the output directly.
// (No LayerNorm: its backward needs the full Jacobian and is fragile at this
// scale; the spike confirmed the model trains fine without it.)
//
// Shapes (d = embedding_dim, f = ffn_hidden, n = context_size, V = vocab_size):
//   token_embedding : V x d        positional_encoding : n x d (fixed sinusoidal)
//   wq, wk, wv      : d x d         w1 : f x d    w2 : d x f    wo : V x d
//
// As in step 4 we only compute the query at the LAST position: it's the only one
// whose output feeds the next-token loss, and being last it attends to everyone
// (causal mask is implicit).

#[derive(Clone)]
pub struct AttentionConfig {
    pub vocab_size: u16,
    pub context_size: usize,
    pub embedding_dim: usize,
    pub ffn_hidden: usize,
}

pub struct Attention {
    config: AttentionConfig,
    token_embedding: Vec<Vec<f32>>,
    positional_encoding: Vec<Vec<f32>>,
    wq: Vec<Vec<f32>>,
    wk: Vec<Vec<f32>>,
    wv: Vec<Vec<f32>>,
    w1: Vec<Vec<f32>>,
    w2: Vec<Vec<f32>>,
    wo: Vec<Vec<f32>>,
}

impl Attention {
    pub fn new(config: AttentionConfig) -> Self {
        let vocab = config.vocab_size as usize;
        let d = config.embedding_dim;
        let f = config.ffn_hidden;
        let n = config.context_size;

        // Fixed sinusoidal positional encoding (constant, never serialized).
        let mut positional_encoding = vec![vec![0.0; d]; n];
        for (pos, row) in positional_encoding.iter_mut().enumerate() {
            for (a, value) in row.iter_mut().enumerate() {
                let exponent = 2.0 * (a / 2) as f32 / d as f32;
                let angle = pos as f32 / 10000f32.powf(exponent);
                *value = if a % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        Self {
            config,
            token_embedding: vec![vec![0.0; d]; vocab],
            positional_encoding,
            wq: vec![vec![0.0; d]; d],
            wk: vec![vec![0.0; d]; d],
            wv: vec![vec![0.0; d]; d],
            w1: vec![vec![0.0; d]; f],
            w2: vec![vec![0.0; f]; d],
            wo: vec![vec![0.0; d]; vocab],
        }
    }

    /// Caller-injected initialization (the registry factory). Same contract as
    /// the MLP: the model is the engine, the caller supplies the seed. The fixed
    /// positional encoding is left untouched.
    pub fn init_weights(&mut self, rng: &mut calc::Rng) {
        const RANGE: f32 = 0.1;
        calc::fill_uniform(&mut self.token_embedding, RANGE, rng);
        calc::fill_uniform(&mut self.wq, RANGE, rng);
        calc::fill_uniform(&mut self.wk, RANGE, rng);
        calc::fill_uniform(&mut self.wv, RANGE, rng);
        calc::fill_uniform(&mut self.w1, RANGE, rng);
        calc::fill_uniform(&mut self.w2, RANGE, rng);
        calc::fill_uniform(&mut self.wo, RANGE, rng);
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn matvec(rows: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    rows.iter().map(|row| dot(row, x)).collect()
}

fn write_matrix(writer: &mut dyn Write, matrix: &[Vec<f32>]) -> std::io::Result<()> {
    for row in matrix {
        for &value in row {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    Ok(())
}

fn read_matrix(reader: &mut dyn Read, matrix: &mut [Vec<f32>]) -> std::io::Result<()> {
    let mut buf = [0u8; 4];
    for row in matrix.iter_mut() {
        for value in row.iter_mut() {
            reader.read_exact(&mut buf)?;
            *value = f32::from_le_bytes(buf);
        }
    }
    Ok(())
}

/// In-place SGD step over a weight matrix.
fn update(weights: &mut [Vec<f32>], grad: &[Vec<f32>], learning_rate: f32) {
    for (row, d_row) in weights.iter_mut().zip(grad) {
        for (w, g) in row.iter_mut().zip(d_row) {
            *w -= learning_rate * g;
        }
    }
}

impl Model for Attention {
    fn vocab_size(&self) -> u16 {
        self.config.vocab_size
    }

    fn context_size(&self) -> usize {
        self.config.context_size
    }

    fn forward(&self, context: &[u16]) -> Vec<f32> {
        let d = self.config.embedding_dim;
        let n = context.len();
        let t = n - 1; // last position is the query that predicts the next token

        // X[i] = token_embedding[ctx[i]] + positional_encoding[i]
        let x: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let emb = &self.token_embedding[context[i] as usize];
                let pe = &self.positional_encoding[i];
                (0..d).map(|b| emb[b] + pe[b]).collect()
            })
            .collect();

        // --- Self-attention (query at the last position only) ---
        let q = matvec(&self.wq, &x[t]);
        let k: Vec<Vec<f32>> = x.iter().map(|xi| matvec(&self.wk, xi)).collect();
        let v: Vec<Vec<f32>> = x.iter().map(|xi| matvec(&self.wv, xi)).collect();
        let scale = (d as f32).sqrt();
        let scores: Vec<f32> = k.iter().map(|kj| dot(&q, kj) / scale).collect();
        let weights = calc::softmax(&scores);
        let mut attended = vec![0.0; d];
        for (j, vj) in v.iter().enumerate() {
            for a in 0..d {
                attended[a] += weights[j] * vj[a];
            }
        }

        // --- Residual 1 + FFN + Residual 2 ---
        let a: Vec<f32> = (0..d).map(|c| x[t][c] + attended[c]).collect();
        let act = calc::relu_vec(&matvec(&self.w1, &a));
        let ffn_out = matvec(&self.w2, &act);
        let h: Vec<f32> = (0..d).map(|c| a[c] + ffn_out[c]).collect();

        matvec(&self.wo, &h)
    }

    fn train_step(&mut self, context: &[u16], target: u16, learning_rate: f32) -> f32 {
        let d = self.config.embedding_dim;
        let f = self.config.ffn_hidden;
        let vocab = self.config.vocab_size as usize;
        let n = context.len();
        let t = n - 1;
        let scale = (d as f32).sqrt();

        // ----------------------- FORWARD (cached) ----------------------------
        let x: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let emb = &self.token_embedding[context[i] as usize];
                let pe = &self.positional_encoding[i];
                (0..d).map(|b| emb[b] + pe[b]).collect()
            })
            .collect();
        let q = matvec(&self.wq, &x[t]);
        let k: Vec<Vec<f32>> = x.iter().map(|xi| matvec(&self.wk, xi)).collect();
        let v: Vec<Vec<f32>> = x.iter().map(|xi| matvec(&self.wv, xi)).collect();
        let scores: Vec<f32> = k.iter().map(|kj| dot(&q, kj) / scale).collect();
        let weights = calc::softmax(&scores);
        let mut attended = vec![0.0; d];
        for (j, vj) in v.iter().enumerate() {
            for a in 0..d {
                attended[a] += weights[j] * vj[a];
            }
        }
        let a: Vec<f32> = (0..d).map(|c| x[t][c] + attended[c]).collect();
        let pre = matvec(&self.w1, &a);
        let act = calc::relu_vec(&pre);
        let ffn_out = matvec(&self.w2, &act);
        let h: Vec<f32> = (0..d).map(|c| a[c] + ffn_out[c]).collect();
        let logits = matvec(&self.wo, &h);
        let probs = calc::softmax(&logits);
        let loss = calc::cross_entropy_loss(&probs, target as usize);

        // ----------------------- BACKWARD ------------------------------------
        let d_logits = calc::cross_entropy_gradient(&probs, target as usize);

        // logits = Wo . h
        let mut d_wo = vec![vec![0.0; d]; vocab];
        let mut d_h = vec![0.0; d];
        for (vi, row) in self.wo.iter().enumerate() {
            for c in 0..d {
                d_wo[vi][c] = d_logits[vi] * h[c];
                d_h[c] += row[c] * d_logits[vi];
            }
        }

        // h = a + ffn_out
        let mut d_a = d_h.clone();
        let d_ffn_out = d_h;

        // ffn_out = W2 . act
        let mut d_w2 = vec![vec![0.0; f]; d];
        let mut d_act = vec![0.0; f];
        for (c, row) in self.w2.iter().enumerate() {
            for m in 0..f {
                d_w2[c][m] = d_ffn_out[c] * act[m];
                d_act[m] += row[m] * d_ffn_out[c];
            }
        }
        // act = relu(pre)
        let d_pre: Vec<f32> = (0..f)
            .map(|m| if pre[m] > 0.0 { d_act[m] } else { 0.0 })
            .collect();
        // pre = W1 . a
        let mut d_w1 = vec![vec![0.0; d]; f];
        for (m, row) in self.w1.iter().enumerate() {
            for c in 0..d {
                d_w1[m][c] = d_pre[m] * a[c];
                d_a[c] += row[c] * d_pre[m];
            }
        }

        // a = X[t] + attended
        let mut d_x = vec![vec![0.0; d]; n];
        for c in 0..d {
            d_x[t][c] += d_a[c];
        }
        let d_attended = d_a;

        // attended = sum_j weights[j] * V[j]
        let mut d_weights = vec![0.0; n];
        let mut d_v = vec![vec![0.0; d]; n];
        for j in 0..n {
            for c in 0..d {
                d_weights[j] += d_attended[c] * v[j][c];
                d_v[j][c] = weights[j] * d_attended[c];
            }
        }

        // Softmax over scores
        let weighted_sum: f32 = (0..n).map(|kk| weights[kk] * d_weights[kk]).sum();
        let d_scores: Vec<f32> = (0..n)
            .map(|j| weights[j] * (d_weights[j] - weighted_sum))
            .collect();

        // scores[j] = (Q . K[j]) / sqrt(d)
        let mut d_q = vec![0.0; d];
        let mut d_k = vec![vec![0.0; d]; n];
        for j in 0..n {
            for c in 0..d {
                d_q[c] += d_scores[j] * k[j][c] / scale;
                d_k[j][c] = d_scores[j] * q[c] / scale;
            }
        }

        // Q = Wq . X[t]
        let mut d_wq = vec![vec![0.0; d]; d];
        for c in 0..d {
            for b in 0..d {
                d_wq[c][b] += d_q[c] * x[t][b];
                d_x[t][b] += self.wq[c][b] * d_q[c];
            }
        }
        // K[j] = Wk . X[j], V[j] = Wv . X[j]
        let mut d_wk = vec![vec![0.0; d]; d];
        let mut d_wv = vec![vec![0.0; d]; d];
        for j in 0..n {
            for c in 0..d {
                for b in 0..d {
                    d_wk[c][b] += d_k[j][c] * x[j][b];
                    d_x[j][b] += self.wk[c][b] * d_k[j][c];
                    d_wv[c][b] += d_v[j][c] * x[j][b];
                    d_x[j][b] += self.wv[c][b] * d_v[j][c];
                }
            }
        }

        // X[i] = token_embedding[ctx[i]] + PE[i]
        let mut d_embedding = vec![vec![0.0; d]; vocab];
        for (i, &token) in context.iter().enumerate() {
            for b in 0..d {
                d_embedding[token as usize][b] += d_x[i][b];
            }
        }

        // ----------------------- SGD UPDATE ----------------------------------
        update(&mut self.wq, &d_wq, learning_rate);
        update(&mut self.wk, &d_wk, learning_rate);
        update(&mut self.wv, &d_wv, learning_rate);
        update(&mut self.w1, &d_w1, learning_rate);
        update(&mut self.w2, &d_w2, learning_rate);
        update(&mut self.wo, &d_wo, learning_rate);
        for token in context.iter().copied().collect::<HashSet<u16>>() {
            for (w, g) in self.token_embedding[token as usize]
                .iter_mut()
                .zip(&d_embedding[token as usize])
            {
                *w -= learning_rate * g;
            }
        }

        loss
    }

    fn save(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        write_matrix(writer, &self.token_embedding)?;
        write_matrix(writer, &self.wq)?;
        write_matrix(writer, &self.wk)?;
        write_matrix(writer, &self.wv)?;
        write_matrix(writer, &self.w1)?;
        write_matrix(writer, &self.w2)?;
        write_matrix(writer, &self.wo)
    }

    fn load(&mut self, reader: &mut dyn Read) -> std::io::Result<()> {
        read_matrix(reader, &mut self.token_embedding)?;
        read_matrix(reader, &mut self.wq)?;
        read_matrix(reader, &mut self.wk)?;
        read_matrix(reader, &mut self.wv)?;
        read_matrix(reader, &mut self.w1)?;
        read_matrix(reader, &mut self.w2)?;
        read_matrix(reader, &mut self.wo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    #[test]
    fn test_attention_forward_with_known_weights() {
        let mut model = Attention::new(AttentionConfig {
            vocab_size: 4,
            context_size: 2,
            embedding_dim: 2,
            ffn_hidden: 4,
        });

        // Zero PE so the hand computation is clean. w1/w2 stay zero (from `new`),
        // so the FFN contributes nothing and we test attention + residual only.
        model.positional_encoding = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        model.token_embedding = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
        ];
        // wk = 0 -> uniform attention [0.5, 0.5]; wv = identity -> V[j] = X[j].
        // attended = avg([1,0], [0,1]) = [0.5, 0.5].
        // Residual 1: a = X[t] + attended = [0,1] + [0.5,0.5] = [0.5, 1.5].
        // FFN is zero, so h = a. logits = wo . [0.5, 1.5].
        model.wv = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        model.wo = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.0, 0.0],
        ];

        let logits = model.forward(&[1, 2]);
        assert_eq!(logits, vec![0.5, 1.5, 2.0, 0.0]);
    }

    #[test]
    fn test_attention_smoke_predicts_target_after_training() {
        let mut model = Attention::new(AttentionConfig {
            vocab_size: 8,
            context_size: 2,
            embedding_dim: 8,
            ffn_hidden: 32,
        });

        model.init_weights(&mut calc::Rng::new(0x5EED));

        for _ in 0..200 {
            model.train_step(&[1, 2], 3, 0.5);
        }

        assert_eq!(argmax(&model.forward(&[1, 2])), 3);
    }

    #[test]
    fn test_attention_save_load_round_trip_preserves_forward() {
        let config = AttentionConfig {
            vocab_size: 8,
            context_size: 3,
            embedding_dim: 4,
            ffn_hidden: 16,
        };

        let mut source = Attention::new(config.clone());
        source.init_weights(&mut calc::Rng::new(0xC0FFEE));
        let expected = source.forward(&[1, 2, 3]);

        let mut buf: Vec<u8> = Vec::new();
        source.save(&mut buf).unwrap();

        let mut restored = Attention::new(config);
        restored.load(&mut buf.as_slice()).unwrap();

        assert_eq!(restored.forward(&[1, 2, 3]), expected);
    }
}
