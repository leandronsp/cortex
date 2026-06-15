use std::collections::HashSet;
use std::io::{Read, Write};

use crate::training::calc;

use super::Model;

// =============================================================================
// STACKED TRANSFORMER BLOCKS
// =============================================================================
// Each block: causal self-attention + FFN + double residual.
//
//   Block forward (position i, attends 0..=i):
//     attended = attention(X[0..i], query at i)
//     a  = X[i] + attended            <- residual 1
//     h  = a + FFN(a)                 <- residual 2
//
// Blocks are stacked: the output of block k feeds block k+1. The final output
// projection Wo maps the last block's last-position output to vocabulary logits.
// (No LayerNorm: its full-Jacobian backward is fragile at this scale and the
// model trains fine without it.)
//
// Shapes (d = embedding_dim, f = ffn_hidden, n = context_size, V = vocab_size):
//   token_embedding : V x d        positional_encoding : n x d (fixed sinusoidal)
//   per block: wq,wk,wv : d x d    w1 : f x d    w2 : d x f
//   wo : V x d

#[derive(Clone)]
pub struct AttentionConfig {
    pub vocab_size: u16,
    pub context_size: usize,
    pub embedding_dim: usize,
    pub ffn_hidden: usize,
    pub num_blocks: usize,
}

struct Block {
    wq: Vec<Vec<f32>>,
    wk: Vec<Vec<f32>>,
    wv: Vec<Vec<f32>>,
    w1: Vec<Vec<f32>>,
    w2: Vec<Vec<f32>>,
}

pub struct Attention {
    config: AttentionConfig,
    token_embedding: Vec<Vec<f32>>,
    positional_encoding: Vec<Vec<f32>>,
    blocks: Vec<Block>,
    wo: Vec<Vec<f32>>,
}

impl Block {
    fn new(d: usize, f: usize) -> Self {
        Self {
            wq: vec![vec![0.0; d]; d],
            wk: vec![vec![0.0; d]; d],
            wv: vec![vec![0.0; d]; d],
            w1: vec![vec![0.0; d]; f],
            w2: vec![vec![0.0; f]; d],
        }
    }

    fn fill_uniform(&mut self, rng: &mut calc::Rng) {
        const RANGE: f32 = 0.1;
        calc::fill_uniform(&mut self.wq, RANGE, rng);
        calc::fill_uniform(&mut self.wk, RANGE, rng);
        calc::fill_uniform(&mut self.wv, RANGE, rng);
        calc::fill_uniform(&mut self.w1, RANGE, rng);
        calc::fill_uniform(&mut self.w2, RANGE, rng);
    }

    /// Forward pass through this block for all positions 0..n-1.
    /// x: input at each position [n][d]. Returns output at each position [n][d].
    fn forward_all(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = x.len();
        let d = self.wq[0].len();
        let scale = (d as f32).sqrt();
        let mut out = vec![vec![0.0; d]; n];

        for i in 0..n {
            let q = matvec(&self.wq, &x[i]);
            // causal: attend to positions 0..=i
            let k: Vec<Vec<f32>> = x[0..=i].iter().map(|xj| matvec(&self.wk, xj)).collect();
            let v: Vec<Vec<f32>> = x[0..=i].iter().map(|xj| matvec(&self.wv, xj)).collect();
            let scores: Vec<f32> = k.iter().map(|kj| dot(&q, kj) / scale).collect();
            let weights = calc::softmax(&scores);
            let mut attended = vec![0.0; d];
            for (j, vj) in v.iter().enumerate() {
                for a in 0..d {
                    attended[a] += weights[j] * vj[a];
                }
            }
            // residual 1
            let a: Vec<f32> = (0..d).map(|c| x[i][c] + attended[c]).collect();
            // FFN
            let act = calc::relu_vec(&matvec(&self.w1, &a));
            let ffn_out = matvec(&self.w2, &act);
            // residual 2
            out[i] = (0..d).map(|c| a[c] + ffn_out[c]).collect();
        }
        out
    }
}

/// Cached forward activations for one training step through a single block.
struct BlockFwd {
    x_in: Vec<Vec<f32>>,
    q: Vec<Vec<f32>>,
    k: Vec<Vec<Vec<f32>>>,
    v: Vec<Vec<Vec<f32>>>,
    #[allow(dead_code)]
    scores: Vec<Vec<f32>>,
    weights: Vec<Vec<f32>>,
    #[allow(dead_code)]
    attended: Vec<Vec<f32>>,
    a: Vec<Vec<f32>>,
    pre: Vec<Vec<f32>>,
    act: Vec<Vec<f32>>,
    #[allow(dead_code)]
    ffn_out: Vec<Vec<f32>>,
    x_out: Vec<Vec<f32>>,
}

struct BlockGrad {
    d_wq: Vec<Vec<f32>>,
    d_wk: Vec<Vec<f32>>,
    d_wv: Vec<Vec<f32>>,
    d_w1: Vec<Vec<f32>>,
    d_w2: Vec<Vec<f32>>,
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

        let blocks = (0..config.num_blocks).map(|_| Block::new(d, f)).collect();

        Self {
            config,
            token_embedding: vec![vec![0.0; d]; vocab],
            positional_encoding,
            blocks,
            wo: vec![vec![0.0; d]; vocab],
        }
    }

    /// Caller-injected initialization (the registry factory).
    pub fn init_weights(&mut self, rng: &mut calc::Rng) {
        const RANGE: f32 = 0.1;
        calc::fill_uniform(&mut self.token_embedding, RANGE, rng);
        for block in &mut self.blocks {
            block.fill_uniform(rng);
        }
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
        let t = n - 1;

        // X[i] = token_embedding[ctx[i]] + positional_encoding[i]
        let mut x: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let emb = &self.token_embedding[context[i] as usize];
                let pe = &self.positional_encoding[i];
                (0..d).map(|b| emb[b] + pe[b]).collect()
            })
            .collect();

        // Stacked blocks
        for block in &self.blocks {
            x = block.forward_all(&x);
        }

        // Output projection on last position
        matvec(&self.wo, &x[t])
    }

    fn train_step(&mut self, context: &[u16], target: u16, learning_rate: f32) -> f32 {
        let d = self.config.embedding_dim;
        let f = self.config.ffn_hidden;
        let vocab = self.config.vocab_size as usize;
        let n = context.len();
        let t = n - 1;
        let scale = (d as f32).sqrt();

        // --------------- FORWARD (cached) -----------------------------------
        let x: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let emb = &self.token_embedding[context[i] as usize];
                let pe = &self.positional_encoding[i];
                (0..d).map(|b| emb[b] + pe[b]).collect()
            })
            .collect();

        // Cache per-block forward activations.
        let mut block_fwds: Vec<BlockFwd> = Vec::with_capacity(self.blocks.len());
        let mut cur = x;
        for block in &self.blocks {
            let fwd = block_forward_cached(block, &cur, d, scale);
            let x_out = fwd.x_out.clone();
            block_fwds.push(fwd);
            cur = x_out;
        }

        // Output projection
        let h = &cur[t];
        let logits = matvec(&self.wo, h);
        let probs = calc::softmax(&logits);
        let loss = calc::cross_entropy_loss(&probs, target as usize);

        // --------------- BACKWARD -------------------------------------------
        let d_logits = calc::cross_entropy_gradient(&probs, target as usize);

        // logits = Wo . h
        let mut d_wo = vec![vec![0.0; d]; vocab];
        let mut d_cur: Vec<Vec<f32>> = vec![vec![0.0; d]; n];
        for (vi, row) in self.wo.iter().enumerate() {
            for c in 0..d {
                d_wo[vi][c] = d_logits[vi] * h[c];
                d_cur[t][c] += row[c] * d_logits[vi];
            }
        }

        // Backward through blocks in reverse order.
        let mut all_grads: Vec<BlockGrad> = Vec::with_capacity(self.blocks.len());
        for (fwd, block) in block_fwds.iter().zip(self.blocks.iter()).rev() {
            let (grad, d_in) = block_backward(block, fwd, &d_cur, d, f, scale);
            all_grads.push(grad);
            d_cur = d_in;
        }
        all_grads.reverse(); // restore forward order for SGD

        // Embedding gradient
        let mut d_embedding = vec![vec![0.0; d]; vocab];
        for (i, &token) in context.iter().enumerate() {
            for b in 0..d {
                d_embedding[token as usize][b] += d_cur[i][b];
            }
        }

        // --------------- SGD UPDATE -----------------------------------------
        for (block, grad) in self.blocks.iter_mut().zip(&all_grads) {
            update(&mut block.wq, &grad.d_wq, learning_rate);
            update(&mut block.wk, &grad.d_wk, learning_rate);
            update(&mut block.wv, &grad.d_wv, learning_rate);
            update(&mut block.w1, &grad.d_w1, learning_rate);
            update(&mut block.w2, &grad.d_w2, learning_rate);
        }
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
        for block in &self.blocks {
            write_matrix(writer, &block.wq)?;
            write_matrix(writer, &block.wk)?;
            write_matrix(writer, &block.wv)?;
            write_matrix(writer, &block.w1)?;
            write_matrix(writer, &block.w2)?;
        }
        write_matrix(writer, &self.wo)
    }

    fn load(&mut self, reader: &mut dyn Read) -> std::io::Result<()> {
        read_matrix(reader, &mut self.token_embedding)?;
        for block in &mut self.blocks {
            read_matrix(reader, &mut block.wq)?;
            read_matrix(reader, &mut block.wk)?;
            read_matrix(reader, &mut block.wv)?;
            read_matrix(reader, &mut block.w1)?;
            read_matrix(reader, &mut block.w2)?;
        }
        read_matrix(reader, &mut self.wo)
    }
}

// =============================================================================
// Block-level forward/backward (free functions to keep train_step readable)
// =============================================================================

fn block_forward_cached(block: &Block, x: &[Vec<f32>], d: usize, scale: f32) -> BlockFwd {
    let n = x.len();
    let mut q = vec![vec![0.0; d]; n];
    let mut k: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n);
    let mut v: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n);
    let mut scores: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut weights: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut attended = vec![vec![0.0; d]; n];
    let mut a = vec![vec![0.0; d]; n];
    let mut pre = vec![vec![0.0; d]; n]; // will be overwritten per position
    let mut act = vec![vec![0.0; d]; n];
    let mut ffn_out = vec![vec![0.0; d]; n];
    let mut x_out = vec![vec![0.0; d]; n];

    for i in 0..n {
        q[i] = matvec(&block.wq, &x[i]);
        let ki: Vec<Vec<f32>> = x[0..=i].iter().map(|xj| matvec(&block.wk, xj)).collect();
        let vi: Vec<Vec<f32>> = x[0..=i].iter().map(|xj| matvec(&block.wv, xj)).collect();
        let sc: Vec<f32> = ki.iter().map(|kj| dot(&q[i], kj) / scale).collect();
        let w = calc::softmax(&sc);
        let mut att = vec![0.0; d];
        for (j, vj) in vi.iter().enumerate() {
            for a in 0..d {
                att[a] += w[j] * vj[a];
            }
        }
        attended[i] = att.clone();
        a[i] = (0..d).map(|c| x[i][c] + att[c]).collect();
        pre[i] = matvec(&block.w1, &a[i]);
        act[i] = calc::relu_vec(&pre[i]);
        ffn_out[i] = matvec(&block.w2, &act[i]);
        x_out[i] = (0..d).map(|c| a[i][c] + ffn_out[i][c]).collect();

        k.push(ki);
        v.push(vi);
        scores.push(sc);
        weights.push(w);
    }

    BlockFwd {
        x_in: x.to_vec(),
        q,
        k,
        v,
        scores,
        weights,
        attended,
        a,
        pre,
        act,
        ffn_out,
        x_out,
    }
}

fn block_backward(
    block: &Block,
    fwd: &BlockFwd,
    d_out: &[Vec<f32>],
    d: usize,
    f: usize,
    scale: f32,
) -> (BlockGrad, Vec<Vec<f32>>) {
    let n = d_out.len();
    let mut d_wq = vec![vec![0.0; d]; d];
    let mut d_wk = vec![vec![0.0; d]; d];
    let mut d_wv = vec![vec![0.0; d]; d];
    let mut d_w1 = vec![vec![0.0; d]; f];
    let mut d_w2 = vec![vec![0.0; f]; d];
    let mut d_x_in = vec![vec![0.0; d]; n];

    for i in 0..n {
        // d_out[i] flows into both residual branches
        let mut d_a = d_out[i].clone();
        let d_ffn_out = d_out[i].clone();

        // ffn_out = W2 . act
        for (c, d_ffn) in d_ffn_out.iter().enumerate() {
            for (m, act_val) in fwd.act[i].iter().enumerate() {
                d_w2[c][m] += d_ffn * act_val;
            }
        }
        let mut d_act = vec![0.0; f];
        for (c, d_ffn) in d_ffn_out.iter().enumerate() {
            let w2_row = &block.w2[c];
            for (m, act_entry) in d_act.iter_mut().enumerate().take(f) {
                *act_entry += w2_row[m] * d_ffn;
            }
        }

        // act = relu(pre)
        let d_pre: Vec<f32> = (0..f)
            .map(|m| if fwd.pre[i][m] > 0.0 { d_act[m] } else { 0.0 })
            .collect();

        // pre = W1 . a
        for (m, row) in block.w1.iter().enumerate() {
            for c in 0..d {
                d_w1[m][c] += d_pre[m] * fwd.a[i][c];
                d_a[c] += row[c] * d_pre[m];
            }
        }

        // a = x[i] + attended
        d_x_in[i] = d_a.clone();
        let d_attended = d_a;

        // attended = sum_j weights[j] * v[j]
        let wi = &fwd.weights[i];
        let vi = &fwd.v[i];
        let mut d_weights = vec![0.0; i + 1];
        let mut d_v: Vec<Vec<f32>> = vec![vec![0.0; d]; i + 1];
        for j in 0..=i {
            for c in 0..d {
                d_weights[j] += d_attended[c] * vi[j][c];
                d_v[j][c] = wi[j] * d_attended[c];
            }
        }

        // softmax backward
        let weighted_sum: f32 = (0..=i).map(|j| wi[j] * d_weights[j]).sum();
        let d_scores: Vec<f32> = (0..=i)
            .map(|j| wi[j] * (d_weights[j] - weighted_sum))
            .collect();

        // scores[j] = (Q . K[j]) / scale
        let mut d_q = vec![0.0; d];
        let ki = &fwd.k[i];
        for j in 0..=i {
            for c in 0..d {
                d_q[c] += d_scores[j] * ki[j][c] / scale;
            }
        }

        // Q = Wq . x[i]
        for c in 0..d {
            for b in 0..d {
                d_wq[c][b] += d_q[c] * fwd.x_in[i][b];
                d_x_in[i][b] += block.wq[c][b] * d_q[c];
            }
        }

        // K[j] = Wk . x[j], V[j] = Wv . x[j]
        for j in 0..=i {
            let d_kj: Vec<f32> = (0..d)
                .map(|c| d_scores[j] * fwd.q[i][c] / scale)
                .collect();
            let d_vj = &d_v[j];
            for c in 0..d {
                for b in 0..d {
                    d_wk[c][b] += d_kj[c] * fwd.x_in[j][b];
                    d_x_in[j][b] += block.wk[c][b] * d_kj[c];
                    d_wv[c][b] += d_vj[c] * fwd.x_in[j][b];
                    d_x_in[j][b] += block.wv[c][b] * d_vj[c];
                }
            }
        }
    }

    (
        BlockGrad {
            d_wq,
            d_wk,
            d_wv,
            d_w1,
            d_w2,
        },
        d_x_in,
    )
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
            num_blocks: 1,
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
        model.blocks[0].wv = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
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
            num_blocks: 1,
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
            num_blocks: 2,
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

    #[test]
    fn test_attention_two_blocks_predicts_target_after_training() {
        let mut model = Attention::new(AttentionConfig {
            vocab_size: 8,
            context_size: 2,
            embedding_dim: 8,
            ffn_hidden: 32,
            num_blocks: 2,
        });

        model.init_weights(&mut calc::Rng::new(0x5EED));

        for _ in 0..200 {
            model.train_step(&[1, 2], 3, 0.5);
        }

        assert_eq!(argmax(&model.forward(&[1, 2])), 3);
    }
}
