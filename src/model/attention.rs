// Numerical code with multi-dim arrays. Explicit range loops (`for i in 0..d`)
// read more clearly than iterator chains when we index several tensors in
// parallel; suppress the lints that suggest rewriting them.
#![allow(clippy::needless_range_loop)]

use std::collections::HashSet;
use std::io::{Read, Write};

use crate::training::calc;

use super::Model;

// =============================================================================
// STACKED TRANSFORMER BLOCKS
// =============================================================================
// Each block = causal self-attention + FFN + residuals:
//   a = x + attn(x)              <- residual 1
//   h = a + FFN(a)               <- residual 2
// FFN(a) = W2 . relu(W1 . a)
//
// Multiple blocks are stacked: x_0 = embedding + PE, then
// x_{b+1} = block_b(x_b). Each block processes all positions with an
// explicit causal mask (position i attends to 0..=i). Only the last
// position's output feeds the next-token loss. Stacking is the lever
// for capacity beyond a single block; the all-positions forward is what
// makes stacking actually useful (otherwise block N would only see what
// block N-1 emitted at the last position).
//
// Shapes per block (d = embedding_dim, f = ffn_hidden, n = context_size, V = vocab_size):
//   wq, wk, wv      : d x d
//   w1 : f x d    w2 : d x f
// Shared across blocks:
//   token_embedding : V x d    positional_encoding : n x d (fixed sinusoidal)
//   wo : V x d (output projection, applied to the last position only)

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
}

pub struct Attention {
    config: AttentionConfig,
    token_embedding: Vec<Vec<f32>>,
    positional_encoding: Vec<Vec<f32>>,
    blocks: Vec<Block>,
    wo: Vec<Vec<f32>>,
}

impl Attention {
    pub fn new(config: AttentionConfig) -> Self {
        let vocab = config.vocab_size as usize;
        let d = config.embedding_dim;
        let f = config.ffn_hidden;
        let n = config.context_size;
        let num_blocks = config.num_blocks;

        // Fixed sinusoidal positional encoding (constant, never serialized).
        let mut positional_encoding = vec![vec![0.0; d]; n];
        for (pos, row) in positional_encoding.iter_mut().enumerate() {
            for (a, value) in row.iter_mut().enumerate() {
                let exponent = 2.0 * (a / 2) as f32 / d as f32;
                let angle = pos as f32 / 10000f32.powf(exponent);
                *value = if a % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(Block::new(d, f));
        }

        Self {
            config,
            token_embedding: vec![vec![0.0; d]; vocab],
            positional_encoding,
            blocks,
            wo: vec![vec![0.0; d]; vocab],
        }
    }

    /// Caller-injected initialization (the registry factory). Same contract as
    /// the MLP: the model is the engine, the caller supplies the seed. The
    /// fixed positional encoding is left untouched.
    pub fn init_weights(&mut self, rng: &mut calc::Rng) {
        const RANGE: f32 = 0.1;
        calc::fill_uniform(&mut self.token_embedding, RANGE, rng);
        for block in &mut self.blocks {
            calc::fill_uniform(&mut block.wq, RANGE, rng);
            calc::fill_uniform(&mut block.wk, RANGE, rng);
            calc::fill_uniform(&mut block.wv, RANGE, rng);
            calc::fill_uniform(&mut block.w1, RANGE, rng);
            calc::fill_uniform(&mut block.w2, RANGE, rng);
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

/// Gradients for one block: d_wq, d_wk, d_wv, d_w1, d_w2. All accumulated
/// across positions before the SGD step.
type BlockGrads = (
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
);

/// In-place SGD step over a weight matrix.
fn update(weights: &mut [Vec<f32>], grad: &[Vec<f32>], learning_rate: f32) {
    for (row, d_row) in weights.iter_mut().zip(grad) {
        for (w, g) in row.iter_mut().zip(d_row) {
            *w -= learning_rate * g;
        }
    }
}

/// Cached activations for one block's forward pass. Needed by `train_step` to
/// backprop without recomputing the forward.
struct BlockCache {
    x_in: Vec<Vec<f32>>,    // per position, d-dim
    q: Vec<Vec<f32>>,       // per position, d-dim
    k: Vec<Vec<f32>>,       // per position, d-dim
    v: Vec<Vec<f32>>,       // per position, d-dim
    weights: Vec<Vec<f32>>, // per position, weights[j] for j in 0..=i (length i+1)
    a: Vec<Vec<f32>>,       // per position, d-dim (residual 1 output)
    pre: Vec<Vec<f32>>,     // per position, f-dim (pre-activation of FFN)
    act: Vec<Vec<f32>>,     // per position, f-dim (post-activation of FFN)
}

/// One block's forward, all positions, causal mask. Returns the block output
/// (one d-dim vector per position) AND caches the intermediates needed for
/// backprop. `forward()` discards the cache; `train_step()` keeps it.
fn block_forward_cache(
    block: &Block,
    x_in: &[Vec<f32>],
    d: usize,
    n: usize,
    scale: f32,
) -> (Vec<Vec<f32>>, BlockCache) {
    let mut q_list = Vec::with_capacity(n);
    let mut k_list = Vec::with_capacity(n);
    let mut v_list = Vec::with_capacity(n);
    for i in 0..n {
        q_list.push(matvec(&block.wq, &x_in[i]));
        k_list.push(matvec(&block.wk, &x_in[i]));
        v_list.push(matvec(&block.wv, &x_in[i]));
    }

    let mut weights_list = Vec::with_capacity(n);
    let mut a_list = Vec::with_capacity(n);
    let mut pre_list = Vec::with_capacity(n);
    let mut act_list = Vec::with_capacity(n);
    let mut h_list = Vec::with_capacity(n);

    for i in 0..n {
        // Causal: position i attends to keys/values at 0..=i
        let scores: Vec<f32> = (0..=i).map(|j| dot(&q_list[i], &k_list[j]) / scale).collect();
        let weights = calc::softmax(&scores);
        let mut attended = vec![0.0; d];
        for j in 0..=i {
            for a in 0..d {
                attended[a] += weights[j] * v_list[j][a];
            }
        }
        let a_i: Vec<f32> = (0..d).map(|c| x_in[i][c] + attended[c]).collect();
        let pre = matvec(&block.w1, &a_i);
        let act = calc::relu_vec(&pre);
        let ffn_out = matvec(&block.w2, &act);
        let h_i: Vec<f32> = (0..d).map(|c| a_i[c] + ffn_out[c]).collect();

        weights_list.push(weights);
        a_list.push(a_i);
        pre_list.push(pre);
        act_list.push(act);
        h_list.push(h_i);
    }

    let cache = BlockCache {
        x_in: x_in.to_vec(),
        q: q_list,
        k: k_list,
        v: v_list,
        weights: weights_list,
        a: a_list,
        pre: pre_list,
        act: act_list,
    };
    (h_list, cache)
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
        let scale = (d as f32).sqrt();

        // X[i] = token_embedding[ctx[i]] + positional_encoding[i]
        let mut x: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let emb = &self.token_embedding[context[i] as usize];
                let pe = &self.positional_encoding[i];
                (0..d).map(|b| emb[b] + pe[b]).collect()
            })
            .collect();

        // Apply each block: all positions, causal mask.
        for block in &self.blocks {
            let (h, _cache) = block_forward_cache(block, &x, d, n, scale);
            x = h;
        }

        // Output projection on the last position only.
        let t = n - 1;
        matvec(&self.wo, &x[t])
    }

    fn train_step(&mut self, context: &[u16], target: u16, learning_rate: f32) -> f32 {
        let d = self.config.embedding_dim;
        let f = self.config.ffn_hidden;
        let vocab = self.config.vocab_size as usize;
        let n = context.len();
        let t = n - 1;
        let scale = (d as f32).sqrt();

        // ----------------------- FORWARD (cached per-block) -----------------
        let x0: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let emb = &self.token_embedding[context[i] as usize];
                let pe = &self.positional_encoding[i];
                (0..d).map(|b| emb[b] + pe[b]).collect()
            })
            .collect();

        let mut caches: Vec<BlockCache> = Vec::with_capacity(self.blocks.len());
        let mut current_x = x0;
        for block in &self.blocks {
            let (h, cache) = block_forward_cache(block, &current_x, d, n, scale);
            caches.push(cache);
            current_x = h;
        }

        // Final logits and loss (from the last block's output, h_{N-1}[t])
        let logits = matvec(&self.wo, &current_x[t]);
        let probs = calc::softmax(&logits);
        let loss = calc::cross_entropy_loss(&probs, target as usize);

        // ----------------------- BACKWARD ------------------------------------
        let d_logits = calc::cross_entropy_gradient(&probs, target as usize);

        // logits = wo . current_x[t]  ->  d_wo and d_h = d_loss/d_h_{N-1}[t]
        let mut d_wo = vec![vec![0.0; d]; vocab];
        let mut d_h = vec![0.0; d];
        for (vi, row) in self.wo.iter().enumerate() {
            for c in 0..d {
                d_wo[vi][c] = d_logits[vi] * current_x[t][c];
                d_h[c] += row[c] * d_logits[vi];
            }
        }

        // d_x: gradient w.r.t. the input of the last block.
        // Only position t has direct loss contribution; positions < t start at 0
        // and pick up indirect gradients as keys/values for higher positions.
        let mut d_x: Vec<Vec<f32>> = vec![vec![0.0; d]; n];
        d_x[t] = d_h;

        // Per-block gradients, accumulated across positions.
        let mut block_grads: Vec<BlockGrads> = (0..self.blocks.len())
            .map(|_| {
                (
                    vec![vec![0.0; d]; d], // d_wq
                    vec![vec![0.0; d]; d], // d_wk
                    vec![vec![0.0; d]; d], // d_wv
                    vec![vec![0.0; d]; f], // d_w1
                    vec![vec![0.0; f]; d], // d_w2
                )
            })
            .collect();

        // Backward through blocks in reverse.
        for (block_idx, cache) in caches.iter().enumerate().rev() {
            let block = &self.blocks[block_idx];
            let (d_wq, d_wk, d_wv, d_w1, d_w2) = &mut block_grads[block_idx];

            // Process positions in reverse so d_x[j] for j < i has already
            // accumulated k/v-source contributions from positions > j.
            for i in (0..n).rev() {
                let d_h_i = d_x[i].clone();

                // h_i = a_i + ffn_i
                let mut d_a = d_h_i.clone();
                let d_ffn = d_h_i;

                // FFN: ffn_i = W2 . act_i  (W2 is d x f)
                for c in 0..d {
                    for m in 0..f {
                        d_w2[c][m] += d_ffn[c] * cache.act[i][m];
                    }
                }
                // d_act = W2^T . d_ffn  (f-dim)
                let mut d_act = vec![0.0; f];
                for c in 0..d {
                    for m in 0..f {
                        d_act[m] += block.w2[c][m] * d_ffn[c];
                    }
                }
                // act_i = relu(pre_i)
                let d_pre: Vec<f32> = (0..f)
                    .map(|m| if cache.pre[i][m] > 0.0 { d_act[m] } else { 0.0 })
                    .collect();
                // pre_i = W1 . a_i  (W1 is f x d)
                for m in 0..f {
                    for c in 0..d {
                        d_w1[m][c] += d_pre[m] * cache.a[i][c];
                        d_a[c] += block.w1[m][c] * d_pre[m];
                    }
                }

                // a_i = x_in[i] + attended_i  ->  d_x_in[i] += d_a
                for c in 0..d {
                    d_x[i][c] += d_a[c];
                }
                let d_attended = d_a;

                // attended_i = sum_{j<=i} weights[j] * v[j]
                let mut d_weights = vec![0.0; i + 1];
                let mut d_v = vec![vec![0.0; d]; i + 1];
                for j in 0..=i {
                    for c in 0..d {
                        d_weights[j] += d_attended[c] * cache.v[j][c];
                        d_v[j][c] = cache.weights[i][j] * d_attended[c];
                    }
                }

                // Softmax backward over j in 0..=i
                let weighted_sum: f32 = (0..=i)
                    .map(|kk| cache.weights[i][kk] * d_weights[kk])
                    .sum();
                let d_scores: Vec<f32> = (0..=i)
                    .map(|j| cache.weights[i][j] * (d_weights[j] - weighted_sum))
                    .collect();

                // scores[j] = (q_i . k_j) / scale
                let mut d_q = vec![0.0; d];
                let mut d_k = vec![vec![0.0; d]; i + 1];
                for j in 0..=i {
                    for c in 0..d {
                        d_q[c] += d_scores[j] * cache.k[j][c] / scale;
                        d_k[j][c] = d_scores[j] * cache.q[i][c] / scale;
                    }
                }

                // q_i = Wq . x_in[i]  (Wq is d x d)
                for c in 0..d {
                    for b in 0..d {
                        d_wq[c][b] += d_q[c] * cache.x_in[i][b];
                        d_x[i][b] += block.wq[c][b] * d_q[c];
                    }
                }
                // k_j = Wk . x_in[j]  (j in 0..=i)
                for j in 0..=i {
                    for c in 0..d {
                        for b in 0..d {
                            d_wk[c][b] += d_k[j][c] * cache.x_in[j][b];
                            d_x[j][b] += block.wk[c][b] * d_k[j][c];
                        }
                    }
                }
                // v_j = Wv . x_in[j]  (j in 0..=i)
                for j in 0..=i {
                    for c in 0..d {
                        for b in 0..d {
                            d_wv[c][b] += d_v[j][c] * cache.x_in[j][b];
                            d_x[j][b] += block.wv[c][b] * d_v[j][c];
                        }
                    }
                }
            }
        }

        // X0[i] = token_embedding[ctx[i]] + PE[i]  ->  d_embedding
        let mut d_embedding = vec![vec![0.0; d]; vocab];
        for (i, &token) in context.iter().enumerate() {
            for b in 0..d {
                d_embedding[token as usize][b] += d_x[i][b];
            }
        }

        // ----------------------- SGD UPDATE ----------------------------------
        // Global gradient norm clipping keeps the all-positions transformer
        // stable on the small corpus: a single batch can otherwise push the
        // total gradient norm past the point of no return (NaN). We scale
        // every gradient matrix uniformly if their joint L2 norm exceeds
        // MAX_GRAD_NORM (PyTorch-style `clip_grad_norm_`).
        const MAX_GRAD_NORM: f32 = 1.0;
        let mut all_grads: Vec<&mut Vec<Vec<f32>>> = Vec::new();
        for grads in &mut block_grads {
            all_grads.push(&mut grads.0);
            all_grads.push(&mut grads.1);
            all_grads.push(&mut grads.2);
            all_grads.push(&mut grads.3);
            all_grads.push(&mut grads.4);
        }
        all_grads.push(&mut d_wo);
        all_grads.push(&mut d_embedding);
        let immutable: Vec<&Vec<Vec<f32>>> = all_grads.iter().map(|r| &**r).collect();
        let total_norm = calc::global_matrix_norm(&immutable);
        if total_norm > MAX_GRAD_NORM {
            let scale = MAX_GRAD_NORM / total_norm;
            calc::scale_matrices(&mut all_grads, scale);
        }

        for block_idx in 0..self.blocks.len() {
            let block = &mut self.blocks[block_idx];
            let (d_wq, d_wk, d_wv, d_w1, d_w2) = &block_grads[block_idx];
            update(&mut block.wq, d_wq, learning_rate);
            update(&mut block.wk, d_wk, learning_rate);
            update(&mut block.wv, d_wv, learning_rate);
            update(&mut block.w1, d_w1, learning_rate);
            update(&mut block.w2, d_w2, learning_rate);
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

    #[test]
    fn test_attention_save_load_round_trip_preserves_forward() {
        let config = AttentionConfig {
            vocab_size: 8,
            context_size: 3,
            embedding_dim: 4,
            ffn_hidden: 16,
            num_blocks: 1,
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
