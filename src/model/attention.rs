use std::collections::HashSet;
use std::io::{Read, Write};

use crate::training::calc;

use super::Model;

// =============================================================================
// STACKED TRANSFORMER BLOCKS: causal self-attention + FFN + residuals
// =============================================================================
// Each block:
//   Q[i] = Wq . H[i]   K[i] = Wk . H[i]   V[i] = Wv . H[i]
//   attended[i] = sum_{j<=i} softmax(Q[i].K[j]/sqrt(d)) * V[j]   (causal mask)
//   a[i] = H[i] + attended[i]                                     (residual 1)
//   H'[i] = a[i] + W2 . relu(W1 . a[i])                           (FFN + residual 2)
//
// Blocks are stacked: H^{b+1} = block_b(H^b). Only the last position of the
// final block feeds the output logits: logits = Wo . H^last[t].
//
// Shapes (d = embedding_dim, f = ffn_hidden, n = context_size, V = vocab_size):
//   token_embedding : V x d        positional_encoding : n x d (fixed sinusoidal)
//   per block: wq, wk, wv : d x d   w1 : f x d   w2 : d x f
//   output: wo : V x d

#[derive(Clone)]
pub struct AttentionConfig {
    pub vocab_size: u16,
    pub context_size: usize,
    pub embedding_dim: usize,
    pub ffn_hidden: usize,
    pub num_blocks: usize,
}

#[derive(Clone)]
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
    blocks: Vec<Block>,
    token_embedding: Vec<Vec<f32>>,
    positional_encoding: Vec<Vec<f32>>,
    wo: Vec<Vec<f32>>,
}

impl Attention {
    pub fn new(config: AttentionConfig) -> Self {
        let vocab = config.vocab_size as usize;
        let d = config.embedding_dim;
        let f = config.ffn_hidden;
        let n = config.context_size;
        let nb = config.num_blocks.max(1);

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
            blocks: (0..nb).map(|_| Block::new(d, f)).collect(),
            token_embedding: vec![vec![0.0; d]; vocab],
            positional_encoding,
            wo: vec![vec![0.0; d]; vocab],
        }
    }

    /// Caller-injected initialization (the registry factory). Same contract as
    /// the MLP: the model is the engine, the caller supplies the seed.
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

        // Initial representations: X[i] = embedding[ctx[i]] + PE[i]
        let mut h: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let emb = &self.token_embedding[context[i] as usize];
                let pe = &self.positional_encoding[i];
                (0..d).map(|b| emb[b] + pe[b]).collect()
            })
            .collect();

        for block in &self.blocks {
            let q: Vec<Vec<f32>> = h.iter().map(|hi| matvec(&block.wq, hi)).collect();
            let k: Vec<Vec<f32>> = h.iter().map(|hi| matvec(&block.wk, hi)).collect();
            let v: Vec<Vec<f32>> = h.iter().map(|hi| matvec(&block.wv, hi)).collect();

            let mut new_h = vec![vec![0.0; d]; n];
            for i in 0..n {
                // Causal attention: position i attends to 0..=i
                let scores: Vec<f32> = (0..=i)
                    .map(|j| dot(&q[i], &k[j]) / scale)
                    .collect();
                let weights = calc::softmax(&scores);

                let mut attended = vec![0.0; d];
                for j in 0..=i {
                    for c in 0..d {
                        attended[c] += weights[j] * v[j][c];
                    }
                }

                // Residual 1 + FFN + Residual 2
                let a: Vec<f32> = (0..d).map(|c| h[i][c] + attended[c]).collect();
                let act = calc::relu_vec(&matvec(&block.w1, &a));
                let ffn_out = matvec(&block.w2, &act);
                for c in 0..d {
                    new_h[i][c] = a[c] + ffn_out[c];
                }
            }
            h = new_h;
        }

        matvec(&self.wo, &h[t])
    }

    fn train_step(&mut self, context: &[u16], target: u16, learning_rate: f32) -> f32 {
        let d = self.config.embedding_dim;
        let f = self.config.ffn_hidden;
        let vocab = self.config.vocab_size as usize;
        let n = context.len();
        let t = n - 1;
        let scale = (d as f32).sqrt();
        let nb = self.blocks.len();

        // ===================== FORWARD (with caching) ========================
        // h_ins[b] = input to block b; h_ins[nb] = output of last block
        let mut h_ins: Vec<Vec<Vec<f32>>> = Vec::with_capacity(nb + 1);

        let h0: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let emb = &self.token_embedding[context[i] as usize];
                let pe = &self.positional_encoding[i];
                (0..d).map(|b| emb[b] + pe[b]).collect()
            })
            .collect();
        h_ins.push(h0);

        // Per-block caches
        let mut qs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(nb);
        let mut ks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(nb);
        let mut vs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(nb);
        let mut ws: Vec<Vec<Vec<f32>>> = Vec::with_capacity(nb); // attention weights
        let mut as_: Vec<Vec<Vec<f32>>> = Vec::with_capacity(nb); // residual-1 outputs
        let mut pres: Vec<Vec<Vec<f32>>> = Vec::with_capacity(nb); // FFN pre-activations
        let mut acts: Vec<Vec<Vec<f32>>> = Vec::with_capacity(nb); // FFN activations

        for bi in 0..nb {
            let block = &self.blocks[bi];
            let h_in = &h_ins[bi];

            let q: Vec<Vec<f32>> = h_in.iter().map(|hi| matvec(&block.wq, hi)).collect();
            let k: Vec<Vec<f32>> = h_in.iter().map(|hi| matvec(&block.wk, hi)).collect();
            let v: Vec<Vec<f32>> = h_in.iter().map(|hi| matvec(&block.wv, hi)).collect();

            let mut block_w: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut block_a: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut block_pre: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut block_act: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut h_out = vec![vec![0.0; d]; n];

            for i in 0..n {
                let scores: Vec<f32> = (0..=i)
                    .map(|j| dot(&q[i], &k[j]) / scale)
                    .collect();
                let w = calc::softmax(&scores);

                let mut attended = vec![0.0; d];
                for j in 0..=i {
                    for c in 0..d {
                        attended[c] += w[j] * v[j][c];
                    }
                }

                let a: Vec<f32> = (0..d).map(|c| h_in[i][c] + attended[c]).collect();
                let pre = matvec(&block.w1, &a);
                let act = calc::relu_vec(&pre);
                let ffn_out = matvec(&block.w2, &act);
                for c in 0..d {
                    h_out[i][c] = a[c] + ffn_out[c];
                }

                block_w.push(w);
                block_a.push(a);
                block_pre.push(pre);
                block_act.push(act);
            }

            qs.push(q);
            ks.push(k);
            vs.push(v);
            ws.push(block_w);
            as_.push(block_a);
            pres.push(block_pre);
            acts.push(block_act);
            h_ins.push(h_out);
        }

        // Output logits from last position of last block
        let h_final = &h_ins[nb];
        let logits = matvec(&self.wo, &h_final[t]);
        let probs = calc::softmax(&logits);
        let loss = calc::cross_entropy_loss(&probs, target as usize);

        // ======================== BACKWARD ====================================
        let d_logits = calc::cross_entropy_gradient(&probs, target as usize);

        // logits = Wo . h_final[t]
        let mut d_wo = vec![vec![0.0; d]; vocab];
        let mut d_h_out = vec![vec![0.0; d]; n];
        for (vi, row) in self.wo.iter().enumerate() {
            for c in 0..d {
                d_wo[vi][c] = d_logits[vi] * h_final[t][c];
                d_h_out[t][c] += row[c] * d_logits[vi];
            }
        }

        // Per-block weight gradients (collected, applied after)
        let mut all_d_wq: Vec<Vec<Vec<f32>>> = Vec::with_capacity(nb);
        let mut all_d_wk: Vec<Vec<Vec<f32>>> = Vec::with_capacity(nb);
        let mut all_d_wv: Vec<Vec<Vec<f32>>> = Vec::with_capacity(nb);
        let mut all_d_w1: Vec<Vec<Vec<f32>>> = Vec::with_capacity(nb);
        let mut all_d_w2: Vec<Vec<Vec<f32>>> = Vec::with_capacity(nb);

        for bi in (0..nb).rev() {
            let block = &self.blocks[bi];
            let h_in = &h_ins[bi];
            let q = &qs[bi];
            let k = &ks[bi];
            let v = &vs[bi];
            let bw = &ws[bi];
            let ba = &as_[bi];
            let bpre = &pres[bi];
            let bact = &acts[bi];

            // h_out[i] = a[i] + ffn_out[i]
            // d_a starts as d_h_out (residual), d_ffn_out = d_h_out
            let d_ffn_out = d_h_out.clone();
            let mut d_a = d_h_out.clone();

            // ffn_out[i] = W2 . act[i]
            let mut d_w2 = vec![vec![0.0; f]; d];
            let mut d_act = vec![vec![0.0; f]; n];
            for i in 0..n {
                for (c, row) in block.w2.iter().enumerate() {
                    for m in 0..f {
                        d_w2[c][m] += d_ffn_out[i][c] * bact[i][m];
                        d_act[i][m] += row[m] * d_ffn_out[i][c];
                    }
                }
            }

            // act[i] = relu(pre[i])
            let mut d_pre = vec![vec![0.0; f]; n];
            for i in 0..n {
                for m in 0..f {
                    d_pre[i][m] = if bpre[i][m] > 0.0 { d_act[i][m] } else { 0.0 };
                }
            }

            // pre[i] = W1 . a[i]
            let mut d_w1 = vec![vec![0.0; d]; f];
            for i in 0..n {
                for (m, row) in block.w1.iter().enumerate() {
                    for c in 0..d {
                        d_w1[m][c] += d_pre[i][m] * ba[i][c];
                        d_a[i][c] += row[c] * d_pre[i][m];
                    }
                }
            }

            // a[i] = h_in[i] + attended[i]
            // d_attended = d_a, d_h_in starts as d_a (residual)
            let d_attended = d_a.clone();
            let mut d_h_in = d_a;

            // attended[i] = sum_{j<=i} w[i][j] * v[j]
            let mut d_v = vec![vec![0.0; d]; n];
            let mut d_weights: Vec<Vec<f32>> = Vec::with_capacity(n);
            for i in 0..n {
                let mut dw = vec![0.0; i + 1];
                for j in 0..=i {
                    for c in 0..d {
                        dw[j] += d_attended[i][c] * v[j][c];
                        d_v[j][c] += bw[i][j] * d_attended[i][c];
                    }
                }
                d_weights.push(dw);
            }

            // Softmax backward per position
            let mut d_scores: Vec<Vec<f32>> = Vec::with_capacity(n);
            for i in 0..n {
                let weighted_sum: f32 =
                    (0..=i).map(|j| bw[i][j] * d_weights[i][j]).sum();
                let ds: Vec<f32> = (0..=i)
                    .map(|j| bw[i][j] * (d_weights[i][j] - weighted_sum))
                    .collect();
                d_scores.push(ds);
            }

            // scores[i][j] = q[i] . k[j] / sqrt(d)
            let mut d_q = vec![vec![0.0; d]; n];
            let mut d_k = vec![vec![0.0; d]; n];
            for i in 0..n {
                for j in 0..=i {
                    for c in 0..d {
                        d_q[i][c] += d_scores[i][j] * k[j][c] / scale;
                        d_k[j][c] += d_scores[i][j] * q[i][c] / scale;
                    }
                }
            }

            // q[i] = Wq . h_in[i], k[i] = Wk . h_in[i], v[i] = Wv . h_in[i]
            let mut d_wq = vec![vec![0.0; d]; d];
            let mut d_wk = vec![vec![0.0; d]; d];
            let mut d_wv = vec![vec![0.0; d]; d];
            for i in 0..n {
                for c in 0..d {
                    for b in 0..d {
                        d_wq[c][b] += d_q[i][c] * h_in[i][b];
                        d_h_in[i][b] += block.wq[c][b] * d_q[i][c];
                        d_wk[c][b] += d_k[i][c] * h_in[i][b];
                        d_h_in[i][b] += block.wk[c][b] * d_k[i][c];
                        d_wv[c][b] += d_v[i][c] * h_in[i][b];
                        d_h_in[i][b] += block.wv[c][b] * d_v[i][c];
                    }
                }
            }

            all_d_wq.push(d_wq);
            all_d_wk.push(d_wk);
            all_d_wv.push(d_wv);
            all_d_w1.push(d_w1);
            all_d_w2.push(d_w2);

            // d_h_in is the gradient flowing to the previous block's output
            d_h_out = d_h_in;
        }

        // Embedding gradient (from block 0's input = initial X)
        let mut d_embedding = vec![vec![0.0; d]; vocab];
        for (i, &token) in context.iter().enumerate() {
            for b in 0..d {
                d_embedding[token as usize][b] += d_h_out[i][b];
            }
        }

        // ======================== SGD UPDATE ==================================
        // Blocks are stored in reverse order in the gradient vecs (last block first)
        for (bi, block) in self.blocks.iter_mut().enumerate() {
            let ri = nb - 1 - bi; // reverse index
            update(&mut block.wq, &all_d_wq[ri], learning_rate);
            update(&mut block.wk, &all_d_wk[ri], learning_rate);
            update(&mut block.wv, &all_d_wv[ri], learning_rate);
            update(&mut block.w1, &all_d_w1[ri], learning_rate);
            update(&mut block.w2, &all_d_w2[ri], learning_rate);
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
}
