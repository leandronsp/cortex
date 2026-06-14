use std::collections::HashSet;
use std::io::{Read, Write};

use crate::training::calc;

use super::Model;

// =============================================================================
// STACKED TRANSFORMER BLOCKS: causal self-attention + FFN + residuals
// =============================================================================
// Each block:
//   a[i] = x_in[i] + attention(x_in)[i]    <- residual 1
//   x_out[i] = a[i] + FFN(a[i])             <- residual 2
//
// where attention is causal: position i attends to positions 0..=i.
// Blocks are stacked: the output of block b feeds block b+1.
// After the final block, only the last position (n-1) feeds the output logits
// through Wo.
//
// (No LayerNorm: its backward needs the full Jacobian and is fragile at this
// scale; the spike confirmed the model trains fine without it.)
//
// Shapes (d = embedding_dim, f = ffn_hidden, n = context_size, V = vocab_size):
//   token_embedding : V x d        positional_encoding : n x d (fixed sinusoidal)
//   per block: wq, wk, wv : d x d    w1 : f x d    w2 : d x f
//   wo : V x d (output projection, outside blocks)

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

        let blocks = (0..config.num_blocks)
            .map(|_| Block {
                wq: vec![vec![0.0; d]; d],
                wk: vec![vec![0.0; d]; d],
                wv: vec![vec![0.0; d]; d],
                w1: vec![vec![0.0; d]; f],
                w2: vec![vec![0.0; f]; d],
            })
            .collect();

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
        let scale = (d as f32).sqrt();

        // X[i] = token_embedding[ctx[i]] + positional_encoding[i]
        let mut x: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let emb = &self.token_embedding[context[i] as usize];
                let pe = &self.positional_encoding[i];
                (0..d).map(|b| emb[b] + pe[b]).collect()
            })
            .collect();

        for block in &self.blocks {
            // Only the last position queries; all positions provide keys/values.
            let q = matvec(&block.wq, &x[t]);
            let k: Vec<Vec<f32>> = x.iter().map(|xi| matvec(&block.wk, xi)).collect();
            let v: Vec<Vec<f32>> = x.iter().map(|xi| matvec(&block.wv, xi)).collect();

            // Attention at the last position (attends to all 0..t, causal is implicit)
            let scores: Vec<f32> = k.iter().map(|kj| dot(&q, kj) / scale).collect();
            let weights = calc::softmax(&scores);
            let mut attended = vec![0.0; d];
            for (j, vj) in v.iter().enumerate() {
                for a in 0..d {
                    attended[a] += weights[j] * vj[a];
                }
            }

            // Residual 1 + FFN + Residual 2 (only at the last position)
            let a: Vec<f32> = (0..d).map(|c| x[t][c] + attended[c]).collect();
            let act = calc::relu_vec(&matvec(&block.w1, &a));
            let ffn_out = matvec(&block.w2, &act);
            x[t] = (0..d).map(|c| a[c] + ffn_out[c]).collect();
            // Non-last positions: keys/values come from the raw x (pass-through).
            // Their representations are unchanged so the next block sees the same
            // keys/values, which is the standard GPT KV-cache behaviour.
        }

        matvec(&self.wo, &x[t])
    }

    #[allow(clippy::needless_range_loop)]
    fn train_step(&mut self, context: &[u16], target: u16, learning_rate: f32) -> f32 {
        let d = self.config.embedding_dim;
        let f = self.config.ffn_hidden;
        let vocab = self.config.vocab_size as usize;
        let n = context.len();
        let t = n - 1;
        let scale = (d as f32).sqrt();
        let num_blocks = self.config.num_blocks;

        // ----------------------- FORWARD (cached per block) ------------------
        // Only the last position goes through attention+FFN; non-last positions
        // provide keys/values but are otherwise pass-through (GPT-style).
        let mut x_blocks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_blocks + 1);
        x_blocks.push(
            (0..n)
                .map(|i| {
                    let emb = &self.token_embedding[context[i] as usize];
                    let pe = &self.positional_encoding[i];
                    (0..d).map(|b| emb[b] + pe[b]).collect()
                })
                .collect(),
        );

        // Per-block caches: k, v (all positions), a (last), pre (last)
        let mut cache_k: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_blocks);
        let mut cache_v: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_blocks);
        let mut cache_a: Vec<Vec<f32>> = Vec::with_capacity(num_blocks);
        let mut cache_pre: Vec<Vec<f32>> = Vec::with_capacity(num_blocks);

        for b in 0..num_blocks {
            let block = &self.blocks[b];
            let x_in = &x_blocks[b];

            let k: Vec<Vec<f32>> = x_in.iter().map(|xi| matvec(&block.wk, xi)).collect();
            let v: Vec<Vec<f32>> = x_in.iter().map(|xi| matvec(&block.wv, xi)).collect();
            let q = matvec(&block.wq, &x_in[t]);

            // Attention at last position
            let scores: Vec<f32> = k.iter().map(|kj| dot(&q, kj) / scale).collect();
            let weights = calc::softmax(&scores);
            let mut attended = vec![0.0; d];
            for (j, vj) in v.iter().enumerate() {
                for a in 0..d {
                    attended[a] += weights[j] * vj[a];
                }
            }

            let a: Vec<f32> = (0..d).map(|c| x_in[t][c] + attended[c]).collect();
            let pre = matvec(&block.w1, &a);
            let act = calc::relu_vec(&pre);
            let ffn_out = matvec(&block.w2, &act);
            let h: Vec<f32> = (0..d).map(|c| a[c] + ffn_out[c]).collect();

            // Build next_x: non-last positions pass through, last position updated
            let mut next_x = x_in.clone();
            next_x[t] = h;

            cache_k.push(k);
            cache_v.push(v);
            cache_a.push(a);
            cache_pre.push(pre);
            x_blocks.push(next_x);
        }

        let h_final = &x_blocks[num_blocks][t];
        let logits = matvec(&self.wo, h_final);
        let probs = calc::softmax(&logits);
        let loss = calc::cross_entropy_loss(&probs, target as usize);

        // ----------------------- BACKWARD ------------------------------------
        let d_logits = calc::cross_entropy_gradient(&probs, target as usize);

        // logits = Wo . h
        let mut d_wo = vec![vec![0.0; d]; vocab];
        let mut d_h = vec![0.0; d];
        for (vi, row) in self.wo.iter().enumerate() {
            for c in 0..d {
                d_wo[vi][c] = d_logits[vi] * h_final[c];
                d_h[c] += row[c] * d_logits[vi];
            }
        }

        // d_h flows into the last position of the final block
        let mut d_x_out = vec![vec![0.0; d]; n];
        d_x_out[t] = d_h;

        let mut d_wq_blocks: Vec<Vec<Vec<f32>>> = (0..num_blocks)
            .map(|_| vec![vec![0.0; d]; d])
            .collect();
        let mut d_wk_blocks: Vec<Vec<Vec<f32>>> = (0..num_blocks)
            .map(|_| vec![vec![0.0; d]; d])
            .collect();
        let mut d_wv_blocks: Vec<Vec<Vec<f32>>> = (0..num_blocks)
            .map(|_| vec![vec![0.0; d]; d])
            .collect();
        let mut d_w1_blocks: Vec<Vec<Vec<f32>>> = (0..num_blocks)
            .map(|_| vec![vec![0.0; d]; f])
            .collect();
        let mut d_w2_blocks: Vec<Vec<Vec<f32>>> = (0..num_blocks)
            .map(|_| vec![vec![0.0; f]; d])
            .collect();

        // Backprop through blocks in reverse order
        for b in (0..num_blocks).rev() {
            let x_in = &x_blocks[b];
            let k = &cache_k[b];
            let v = &cache_v[b];
            let a_cache = &cache_a[b];
            let pre_cache = &cache_pre[b];
            let block = &self.blocks[b];

            // h = a + ffn_out  →  d_a = d_x_last, d_ffn_out = d_x_last
            let d_a_residual = d_x_out[t].clone();

            // ffn_out = W2 . act
            let act_i: Vec<f32> = pre_cache.iter().map(|&p| if p > 0.0 { p } else { 0.0 }).collect();
            for c in 0..d {
                for m in 0..f {
                    d_w2_blocks[b][c][m] += d_x_out[t][c] * act_i[m];
                }
            }
            let mut d_act = vec![0.0; f];
            for m in 0..f {
                for c in 0..d {
                    d_act[m] += block.w2[c][m] * d_x_out[t][c];
                }
            }
            // act = relu(pre)
            let d_pre: Vec<f32> = (0..f)
                .map(|m| if pre_cache[m] > 0.0 { d_act[m] } else { 0.0 })
                .collect();
            // pre = W1 . a
            for m in 0..f {
                for c in 0..d {
                    d_w1_blocks[b][m][c] += d_pre[m] * a_cache[c];
                }
            }
            let mut d_a_ffn = vec![0.0; d];
            for c in 0..d {
                for m in 0..f {
                    d_a_ffn[c] += block.w1[m][c] * d_pre[m];
                }
            }

            // Total d_a = residual path + FFN path
            let d_a_total: Vec<f32> = (0..d)
                .map(|c| d_a_residual[c] + d_a_ffn[c])
                .collect();

            // a = x_in[t] + attended  →  d_x_in[t] += d_a_total, d_attended = d_a_total
            let mut d_x_in_t = d_a_total.clone();
            let d_attended = &d_a_total;

            // Recompute attention forward for last position
            let q = matvec(&block.wq, &x_in[t]);
            let scores: Vec<f32> = k.iter().map(|kj| dot(&q, kj) / scale).collect();
            let weights = calc::softmax(&scores);

            // attended = Σ_j weights[j] * V[j]
            let mut d_weights = vec![0.0; n];
            for j in 0..n {
                for c in 0..d {
                    d_weights[j] += d_attended[c] * v[j][c];
                }
            }
            // Softmax backward
            let weighted_sum: f32 = (0..n).map(|kk| weights[kk] * d_weights[kk]).sum();
            let d_scores: Vec<f32> = (0..n)
                .map(|j| weights[j] * (d_weights[j] - weighted_sum))
                .collect();

            // scores[j] = (Q · K[j]) / sqrt(d)
            let mut d_q = vec![0.0; d];
            let mut d_k = vec![vec![0.0; d]; n];
            let mut d_v = vec![vec![0.0; d]; n];
            for j in 0..n {
                for c in 0..d {
                    d_q[c] += d_scores[j] * k[j][c] / scale;
                    d_k[j][c] = d_scores[j] * q[c] / scale;
                    d_v[j][c] = weights[j] * d_attended[c];
                }
            }

            // Q = Wq · x_in[t]
            for c in 0..d {
                for a in 0..d {
                    d_wq_blocks[b][c][a] += d_q[c] * x_in[t][a];
                    d_x_in_t[a] += block.wq[c][a] * d_q[c];
                }
            }

            // K_j = Wk · x_in[j], V_j = Wv · x_in[j]
            let mut d_x_in = vec![vec![0.0; d]; n];
            d_x_in[t] = d_x_in_t;
            for j in 0..n {
                for c in 0..d {
                    for a in 0..d {
                        d_wk_blocks[b][c][a] += d_k[j][c] * x_in[j][a];
                        d_x_in[j][a] += block.wk[c][a] * d_k[j][c];
                        d_wv_blocks[b][c][a] += d_v[j][c] * x_in[j][a];
                        d_x_in[j][a] += block.wv[c][a] * d_v[j][c];
                    }
                }
            }

            // d_x_out for previous block = d_x_in (all positions)
            d_x_out = d_x_in;
        }

        // d_x_out now holds gradients w.r.t. x_0 (embeddings + PE) for all positions
        let mut d_embedding = vec![vec![0.0; d]; vocab];
        for (i, &token) in context.iter().enumerate() {
            for a in 0..d {
                d_embedding[token as usize][a] += d_x_out[i][a];
            }
        }

        // ----------------------- SGD UPDATE ----------------------------------
        for b in 0..num_blocks {
            let block = &mut self.blocks[b];
            update(&mut block.wq, &d_wq_blocks[b], learning_rate);
            update(&mut block.wk, &d_wk_blocks[b], learning_rate);
            update(&mut block.wv, &d_wv_blocks[b], learning_rate);
            update(&mut block.w1, &d_w1_blocks[b], learning_rate);
            update(&mut block.w2, &d_w2_blocks[b], learning_rate);
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
    fn test_forward_consistent_after_training_two_blocks() {
        let config = AttentionConfig {
            vocab_size: 4,
            context_size: 3,
            embedding_dim: 2,
            ffn_hidden: 4,
            num_blocks: 2,
        };
        let mut model = Attention::new(config);
        model.init_weights(&mut calc::Rng::new(0xCAFE));
        let context = &[1u16, 2, 3];
        for t in 0..4u16 {
            model.train_step(context, t, 0.01);
        }
        // Manual forward matching new behaviour: only last position query+FFN
        let d = model.config.embedding_dim;
        let n = context.len();
        let last = n - 1;
        let scale = (d as f32).sqrt();
        let mut x: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let emb = &model.token_embedding[context[i] as usize];
                let pe = &model.positional_encoding[i];
                (0..d).map(|b| emb[b] + pe[b]).collect()
            })
            .collect();
        for block in &model.blocks {
            let q = matvec(&block.wq, &x[last]);
            let k: Vec<Vec<f32>> = x.iter().map(|xi| matvec(&block.wk, xi)).collect();
            let v: Vec<Vec<f32>> = x.iter().map(|xi| matvec(&block.wv, xi)).collect();
            let scores: Vec<f32> = k.iter().map(|kj| dot(&q, kj) / scale).collect();
            let weights = calc::softmax(&scores);
            let mut attended = vec![0.0; d];
            for (j, vj) in v.iter().enumerate() {
                for a in 0..d {
                    attended[a] += weights[j] * vj[a];
                }
            }
            let a: Vec<f32> = (0..d).map(|c| x[last][c] + attended[c]).collect();
            let act = calc::relu_vec(&matvec(&block.w1, &a));
            let ffn_out = matvec(&block.w2, &act);
            x[last] = (0..d).map(|c| a[c] + ffn_out[c]).collect();
        }
        let logits_manual = matvec(&model.wo, &x[last]);
        let logits_fwd = model.forward(context);
        assert_eq!(logits_fwd, logits_manual);
    }

    #[test]
    fn test_attention_two_blocks_all_gradients_numerical() {
        // Systematic gradient check for every weight matrix in a 2-block model.
        // Checks a representative element from each matrix.
        let config = AttentionConfig {
            vocab_size: 4,
            context_size: 3,
            embedding_dim: 2,
            ffn_hidden: 4,
            num_blocks: 2,
        };
        let context = &[1u16, 2, 3];
        let target: u16 = 0;
        let lr: f32 = 0.001;
        let eps: f32 = 1e-3;

        // Helper: analytical vs finite-difference for one weight element
        fn check_gradient(
            config: &AttentionConfig,
            context: &[u16],
            target: u16,
            lr: f32,
            eps: f32,
            label: &str,
            get: fn(&Attention) -> f32,
            set: fn(&mut Attention, f32),
        ) -> Result<(), String> {
            let mut model = Attention::new(config.clone());
            model.init_weights(&mut calc::Rng::new(0xCAFE));
            let before = get(&model);
            model.train_step(context, target, lr);
            let anal = (before - get(&model)) / lr;

            let mut model_fd = Attention::new(config.clone());
            model_fd.init_weights(&mut calc::Rng::new(0xCAFE));
            let logits_orig = model_fd.forward(context);
            let probs_orig = calc::softmax(&logits_orig);
            let loss_orig = calc::cross_entropy_loss(&probs_orig, target as usize);

            let current = get(&model_fd);
            set(&mut model_fd, current + eps);
            let logits_pert = model_fd.forward(context);
            let probs_pert = calc::softmax(&logits_pert);
            let loss_pert = calc::cross_entropy_loss(&probs_pert, target as usize);
            let fd = (loss_pert - loss_orig) / eps;

            let abs_err = (anal - fd).abs();
            if abs_err > 1e-3 {
                return Err(format!("{label}: anal={anal:.8}, fd={fd:.8}, abs_err={abs_err:.8}"));
            }
            Ok(())
        }

        // Block 0
        check_gradient(&config, context, target, lr, eps, "b0.wq[0][0]",
            |m| m.blocks[0].wq[0][0], |m, v| m.blocks[0].wq[0][0] = v).unwrap();
        check_gradient(&config, context, target, lr, eps, "b0.wk[0][0]",
            |m| m.blocks[0].wk[0][0], |m, v| m.blocks[0].wk[0][0] = v).unwrap();
        check_gradient(&config, context, target, lr, eps, "b0.wv[0][0]",
            |m| m.blocks[0].wv[0][0], |m, v| m.blocks[0].wv[0][0] = v).unwrap();
        check_gradient(&config, context, target, lr, eps, "b0.w1[0][0]",
            |m| m.blocks[0].w1[0][0], |m, v| m.blocks[0].w1[0][0] = v).unwrap();
        check_gradient(&config, context, target, lr, eps, "b0.w2[0][0]",
            |m| m.blocks[0].w2[0][0], |m, v| m.blocks[0].w2[0][0] = v).unwrap();

        // Block 1
        check_gradient(&config, context, target, lr, eps, "b1.wq[0][0]",
            |m| m.blocks[1].wq[0][0], |m, v| m.blocks[1].wq[0][0] = v).unwrap();
        check_gradient(&config, context, target, lr, eps, "b1.wk[0][0]",
            |m| m.blocks[1].wk[0][0], |m, v| m.blocks[1].wk[0][0] = v).unwrap();
        check_gradient(&config, context, target, lr, eps, "b1.wv[0][0]",
            |m| m.blocks[1].wv[0][0], |m, v| m.blocks[1].wv[0][0] = v).unwrap();
        check_gradient(&config, context, target, lr, eps, "b1.w1[0][0]",
            |m| m.blocks[1].w1[0][0], |m, v| m.blocks[1].w1[0][0] = v).unwrap();
        check_gradient(&config, context, target, lr, eps, "b1.w2[0][0]",
            |m| m.blocks[1].w2[0][0], |m, v| m.blocks[1].w2[0][0] = v).unwrap();

        // wo
        check_gradient(&config, context, target, lr, eps, "wo[0][0]",
            |m| m.wo[0][0], |m, v| m.wo[0][0] = v).unwrap();
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
