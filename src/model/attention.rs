use std::collections::HashSet;
use std::io::{Read, Write};

use crate::training::calc;

use super::Model;

// =============================================================================
// TRANSFORMER BLOCK: causal self-attention + FFN + residuals
// =============================================================================
// Karpathy step 4 gave us attention (route information from past tokens). On its
// own it plateaus high: the only non-linearity is the attention softmax, so it
// can't actually *compute* much. Step 5 completes the block:
//
//   a = X[i] + attention(X)[i]      <- residual 1: keep the token's own signal
//   h = a   + FFN(a)                <- residual 2: FFN does the heavy non-linear work
//   logits = Wo . h[last]
//
// FFN(a) = W2 . relu(W1 . a): an up-projection, a ReLU, a down-projection. This
// is where most of a transformer's capacity lives. The residuals let gradients
// flow straight through and let the current token reach the output directly.
// (No LayerNorm: its backward needs the full Jacobian and is fragile at this
// scale; the spike confirmed the model trains fine without it.)
//
// A block maps a full sequence X (n x d) to H (n x d) under a causal mask
// (position i attends to 0..=i). Stacking blocks needs every position's output,
// because the next block's attention reads the whole sequence. Only the LAST
// position of the final block feeds the next-token loss.
//
// Shapes (d = embedding_dim, f = ffn_hidden, n = context_size, V = vocab_size):
//   token_embedding : V x d        positional_encoding : n x d (fixed sinusoidal)
//   wq, wk, wv      : d x d         w1 : f x d    w2 : d x f    wo : V x d

#[derive(Clone)]
pub struct AttentionConfig {
    pub vocab_size: u16,
    pub context_size: usize,
    pub embedding_dim: usize,
    pub ffn_hidden: usize,
    pub num_blocks: usize,
}

/// One transformer block's learnable weights. The token embedding, positional
/// encoding and output projection live on `Attention`, shared across blocks.
struct Block {
    wq: Vec<Vec<f32>>,
    wk: Vec<Vec<f32>>,
    wv: Vec<Vec<f32>>,
    w1: Vec<Vec<f32>>,
    w2: Vec<Vec<f32>>,
}

impl Block {
    fn zeros(d: usize, f: usize) -> Self {
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

        // Fixed sinusoidal positional encoding (constant, never serialized).
        let mut positional_encoding = vec![vec![0.0; d]; n];
        for (pos, row) in positional_encoding.iter_mut().enumerate() {
            for (a, value) in row.iter_mut().enumerate() {
                let exponent = 2.0 * (a / 2) as f32 / d as f32;
                let angle = pos as f32 / 10000f32.powf(exponent);
                *value = if a % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        let blocks = (0..config.num_blocks).map(|_| Block::zeros(d, f)).collect();

        Self {
            config,
            token_embedding: vec![vec![0.0; d]; vocab],
            positional_encoding,
            blocks,
            wo: vec![vec![0.0; d]; vocab],
        }
    }

    /// Caller-injected initialization (the registry factory). Same contract as
    /// the MLP: the model is the engine, the caller supplies the seed. The fixed
    /// positional encoding is left untouched.
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

    /// X[i] = token_embedding[ctx[i]] + positional_encoding[i].
    fn embed(&self, context: &[u16]) -> Vec<Vec<f32>> {
        let d = self.config.embedding_dim;
        (0..context.len())
            .map(|i| {
                let emb = &self.token_embedding[context[i] as usize];
                let pe = &self.positional_encoding[i];
                (0..d).map(|b| emb[b] + pe[b]).collect()
            })
            .collect()
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn matvec(rows: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    rows.iter().map(|row| dot(row, x)).collect()
}

/// Cached activations from a block's forward pass, needed by its backward.
struct BlockCache {
    x: Vec<Vec<f32>>,
    q: Vec<Vec<f32>>,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    weights: Vec<Vec<f32>>, // weights[i][j] for j in 0..=i, zero above the diagonal
    a: Vec<Vec<f32>>,
    pre: Vec<Vec<f32>>,
    act: Vec<Vec<f32>>,
}

/// Per-block weight gradients accumulated over all positions.
struct BlockGrads {
    d_wq: Vec<Vec<f32>>,
    d_wk: Vec<Vec<f32>>,
    d_wv: Vec<Vec<f32>>,
    d_w1: Vec<Vec<f32>>,
    d_w2: Vec<Vec<f32>>,
}

/// Forward a full sequence through one block under a causal mask. Returns the
/// output sequence H and the cache for backprop.
fn block_forward(block: &Block, x: &[Vec<f32>]) -> (Vec<Vec<f32>>, BlockCache) {
    let n = x.len();
    let d = x[0].len();
    let scale = (d as f32).sqrt();

    let k: Vec<Vec<f32>> = x.iter().map(|xi| matvec(&block.wk, xi)).collect();
    let v: Vec<Vec<f32>> = x.iter().map(|xi| matvec(&block.wv, xi)).collect();

    let mut q = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    let mut a = Vec::with_capacity(n);
    let mut pre = Vec::with_capacity(n);
    let mut act = Vec::with_capacity(n);
    let mut h = Vec::with_capacity(n);

    for i in 0..n {
        let qi = matvec(&block.wq, &x[i]);
        // Causal: position i attends only to 0..=i.
        let scores: Vec<f32> = (0..=i).map(|j| dot(&qi, &k[j]) / scale).collect();
        let w = calc::softmax(&scores);
        let mut attended = vec![0.0; d];
        for (j, wj) in w.iter().enumerate() {
            for c in 0..d {
                attended[c] += wj * v[j][c];
            }
        }
        let ai: Vec<f32> = (0..d).map(|c| x[i][c] + attended[c]).collect();
        let prei = matvec(&block.w1, &ai);
        let acti = calc::relu_vec(&prei);
        let ffn = matvec(&block.w2, &acti);
        let hi: Vec<f32> = (0..d).map(|c| ai[c] + ffn[c]).collect();

        let mut wfull = vec![0.0; n];
        wfull[..=i].copy_from_slice(&w);
        q.push(qi);
        weights.push(wfull);
        a.push(ai);
        pre.push(prei);
        act.push(acti);
        h.push(hi);
    }

    let cache = BlockCache { x: x.to_vec(), q, k, v, weights, a, pre, act };
    (h, cache)
}

/// Backprop one block: given the gradient on its output sequence, return the
/// gradient on its input sequence plus the block's weight gradients.
fn block_backward(block: &Block, cache: &BlockCache, d_h: &[Vec<f32>]) -> (Vec<Vec<f32>>, BlockGrads) {
    let n = cache.x.len();
    let d = cache.x[0].len();
    let f = block.w1.len();
    let scale = (d as f32).sqrt();

    let mut grads = BlockGrads {
        d_wq: vec![vec![0.0; d]; d],
        d_wk: vec![vec![0.0; d]; d],
        d_wv: vec![vec![0.0; d]; d],
        d_w1: vec![vec![0.0; d]; f],
        d_w2: vec![vec![0.0; f]; d],
    };
    let mut d_x = vec![vec![0.0; d]; n];
    let mut d_k = vec![vec![0.0; d]; n];
    let mut d_v = vec![vec![0.0; d]; n];

    for i in 0..n {
        // h = a + ffn
        let mut d_a = d_h[i].clone();
        let d_ffn = &d_h[i];
        // ffn = W2 . act
        let mut d_act = vec![0.0; f];
        for (c, row) in block.w2.iter().enumerate() {
            for m in 0..f {
                grads.d_w2[c][m] += d_ffn[c] * cache.act[i][m];
                d_act[m] += row[m] * d_ffn[c];
            }
        }
        // act = relu(pre)
        let d_pre: Vec<f32> = (0..f)
            .map(|m| if cache.pre[i][m] > 0.0 { d_act[m] } else { 0.0 })
            .collect();
        // pre = W1 . a
        for (m, row) in block.w1.iter().enumerate() {
            for c in 0..d {
                grads.d_w1[m][c] += d_pre[m] * cache.a[i][c];
                d_a[c] += row[c] * d_pre[m];
            }
        }
        // a = X[i] + attended
        for c in 0..d {
            d_x[i][c] += d_a[c];
        }
        let d_attended = d_a;
        // attended = sum_{j<=i} weights[i][j] * V[j]
        let mut d_weights = vec![0.0; n];
        for j in 0..=i {
            for c in 0..d {
                d_weights[j] += d_attended[c] * cache.v[j][c];
                d_v[j][c] += cache.weights[i][j] * d_attended[c];
            }
        }
        // softmax over scores (j = 0..=i)
        let w = &cache.weights[i];
        let weighted_sum: f32 = (0..=i).map(|j| w[j] * d_weights[j]).sum();
        let d_scores: Vec<f32> = (0..=i).map(|j| w[j] * (d_weights[j] - weighted_sum)).collect();
        // scores[j] = (Q[i] . K[j]) / sqrt(d)
        let qi = &cache.q[i];
        let mut d_q = vec![0.0; d];
        for j in 0..=i {
            for c in 0..d {
                d_q[c] += d_scores[j] * cache.k[j][c] / scale;
                d_k[j][c] += d_scores[j] * qi[c] / scale;
            }
        }
        // Q[i] = Wq . X[i]
        for (c, wq_row) in block.wq.iter().enumerate() {
            let dq_c = d_q[c];
            for (b, w) in wq_row.iter().enumerate() {
                grads.d_wq[c][b] += dq_c * cache.x[i][b];
                d_x[i][b] += w * dq_c;
            }
        }
    }

    // K[j] = Wk . X[j], V[j] = Wv . X[j]
    for j in 0..n {
        for (c, (wk_row, wv_row)) in block.wk.iter().zip(&block.wv).enumerate() {
            let dk = d_k[j][c];
            let dv = d_v[j][c];
            for (b, (wk, wv)) in wk_row.iter().zip(wv_row).enumerate() {
                grads.d_wk[c][b] += dk * cache.x[j][b];
                d_x[j][b] += wk * dk;
                grads.d_wv[c][b] += dv * cache.x[j][b];
                d_x[j][b] += wv * dv;
            }
        }
    }

    (d_x, grads)
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
        let t = context.len() - 1; // last position predicts the next token
        let mut seq = self.embed(context);
        for block in &self.blocks {
            seq = block_forward(block, &seq).0;
        }
        matvec(&self.wo, &seq[t])
    }

    fn train_step(&mut self, context: &[u16], target: u16, learning_rate: f32) -> f32 {
        let d = self.config.embedding_dim;
        let vocab = self.config.vocab_size as usize;
        let n = context.len();
        let t = n - 1;

        // ----------------------- FORWARD (cached per block) ------------------
        let mut caches = Vec::with_capacity(self.blocks.len());
        let mut seq = self.embed(context);
        for block in &self.blocks {
            let (h, cache) = block_forward(block, &seq);
            caches.push(cache);
            seq = h;
        }
        let logits = matvec(&self.wo, &seq[t]);
        let probs = calc::softmax(&logits);
        let loss = calc::cross_entropy_loss(&probs, target as usize);

        // ----------------------- BACKWARD ------------------------------------
        let d_logits = calc::cross_entropy_gradient(&probs, target as usize);
        // logits = Wo . seq[t] (last block's last position)
        let mut d_wo = vec![vec![0.0; d]; vocab];
        let mut d_ht = vec![0.0; d];
        for (vi, row) in self.wo.iter().enumerate() {
            for c in 0..d {
                d_wo[vi][c] = d_logits[vi] * seq[t][c];
                d_ht[c] += row[c] * d_logits[vi];
            }
        }
        // Only the last position feeds the loss.
        let mut d_seq = vec![vec![0.0; d]; n];
        d_seq[t] = d_ht;

        // Reverse fold: last block first. block_grads ends up last-to-first.
        let mut block_grads = Vec::with_capacity(self.blocks.len());
        for (block, cache) in self.blocks.iter().zip(&caches).rev() {
            let (d_in, grads) = block_backward(block, cache, &d_seq);
            block_grads.push(grads);
            d_seq = d_in;
        }

        // X[i] = token_embedding[ctx[i]] + PE[i]; d_seq is now the gradient on X.
        let mut d_embedding = vec![vec![0.0; d]; vocab];
        for (i, &token) in context.iter().enumerate() {
            for b in 0..d {
                d_embedding[token as usize][b] += d_seq[i][b];
            }
        }

        // ----------------------- SGD UPDATE ----------------------------------
        // block_grads is last-to-first, so pair it with the blocks reversed.
        for (block, grads) in self.blocks.iter_mut().rev().zip(block_grads) {
            update(&mut block.wq, &grads.d_wq, learning_rate);
            update(&mut block.wk, &grads.d_wk, learning_rate);
            update(&mut block.wv, &grads.d_wv, learning_rate);
            update(&mut block.w1, &grads.d_w1, learning_rate);
            update(&mut block.w2, &grads.d_w2, learning_rate);
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
    fn test_attention_two_blocks_forward_with_known_weights() {
        // Two identical blocks, each: wk=0 -> uniform causal attention,
        // wv=identity -> V[j]=X[j], w1/w2=0 -> FFN contributes nothing.
        // A block then maps H[i] = X[i] + mean(X[0..=i]).
        //
        // X = [[1,0],[0,1]] (zero PE, embeddings of tokens 1 and 2).
        // Block 1: H0 = [1,0]+mean([1,0]) = [2,0]
        //          H1 = [0,1]+mean([1,0],[0,1]) = [0.5,1.5]
        // Block 2: H1' = [0.5,1.5]+mean([2,0],[0.5,1.5]) = [0.5,1.5]+[1.25,0.75]
        //              = [1.75,2.25]
        // logits = wo . [1.75,2.25].
        let mut model = Attention::new(AttentionConfig {
            vocab_size: 4,
            context_size: 2,
            embedding_dim: 2,
            ffn_hidden: 4,
            num_blocks: 2,
        });

        model.positional_encoding = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        model.token_embedding = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
        ];
        for block in &mut model.blocks {
            block.wv = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        }
        model.wo = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.0, 0.0],
        ];

        let logits = model.forward(&[1, 2]);
        assert_eq!(logits, vec![1.75, 2.25, 4.0, 0.0]);
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
    fn test_attention_train_step_updates_every_block() {
        // Backprop must reach all stacked blocks. A single SGD step has to move
        // the last block's weights, not just the first. The previous single-block
        // train_step left blocks[1] untouched.
        let mut model = Attention::new(AttentionConfig {
            vocab_size: 8,
            context_size: 2,
            embedding_dim: 8,
            ffn_hidden: 32,
            num_blocks: 2,
        });
        model.init_weights(&mut calc::Rng::new(0x5EED));

        let before = model.blocks[1].wq.clone();
        model.train_step(&[1, 2], 3, 0.1);

        assert_ne!(model.blocks[1].wq, before, "last block must receive gradient");
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
            model.train_step(&[1, 2], 3, 0.1);
        }

        assert_eq!(argmax(&model.forward(&[1, 2])), 3);
    }

    #[test]
    fn test_attention_save_load_round_trip_two_blocks() {
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
