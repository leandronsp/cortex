use std::collections::HashSet;
use std::io::{Read, Write};

use crate::training::calc;

use super::Model;

// =============================================================================
// STACKED TRANSFORMER BLOCKS: causal self-attention + FFN + residuals
// =============================================================================

#[derive(Clone)]
pub struct AttentionConfig {
    pub vocab_size: u16,
    pub context_size: usize,
    pub embedding_dim: usize,
    pub ffn_hidden: usize,
    pub num_blocks: usize,
}

pub struct Block {
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

    fn init_weights(&mut self, rng: &mut calc::Rng) {
        const RANGE: f32 = 0.1;
        calc::fill_uniform(&mut self.wq, RANGE, rng);
        calc::fill_uniform(&mut self.wk, RANGE, rng);
        calc::fill_uniform(&mut self.wv, RANGE, rng);
        calc::fill_uniform(&mut self.w1, RANGE, rng);
        calc::fill_uniform(&mut self.w2, RANGE, rng);
    }
}

impl Attention {
    pub fn new(config: AttentionConfig) -> Self {
        let vocab = config.vocab_size as usize;
        let d = config.embedding_dim;
        let f = config.ffn_hidden;
        let n = config.context_size;
        let num_blocks = config.num_blocks.max(1);

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
            blocks: (0..num_blocks).map(|_| Block::new(d, f)).collect(),
            wo: vec![vec![0.0; d]; vocab],
        }
    }

    pub fn init_weights(&mut self, rng: &mut calc::Rng) {
        const RANGE: f32 = 0.1;
        calc::fill_uniform(&mut self.token_embedding, RANGE, rng);
        for block in &mut self.blocks {
            block.init_weights(rng);
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

fn add_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn relu_vec(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect()
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

struct BlockCache {
    x: Vec<Vec<f32>>,       // block input, n x d
    q: Vec<Vec<f32>>,       // n x d
    k: Vec<Vec<f32>>,       // n x d
    v: Vec<Vec<f32>>,       // n x d
    _scores: Vec<Vec<f32>>, // n x n (causal, padded with 0 for j > i)
    weights: Vec<Vec<f32>>, // n x n
    _attended: Vec<Vec<f32>>,// n x d
    a: Vec<Vec<f32>>,       // n x d (post residual)
    pre: Vec<Vec<f32>>,     // n x f (pre-relu)
    act: Vec<Vec<f32>>,     // n x f
    _ffn_out: Vec<Vec<f32>>,// n x d
    _h: Vec<Vec<f32>>,      // n x d (post residual)
}

/// Causal self-attention + FFN for one block. Input x has shape n x d.
fn block_forward(block: &Block, x: &[Vec<f32>], scale: f32) -> (Vec<Vec<f32>>, BlockCache) {
    let n = x.len();
    let d = x[0].len();
    let _f = block.w1.len();

    let q: Vec<Vec<f32>> = x.iter().map(|xi| matvec(&block.wq, xi)).collect();
    let k: Vec<Vec<f32>> = x.iter().map(|xi| matvec(&block.wk, xi)).collect();
    let v: Vec<Vec<f32>> = x.iter().map(|xi| matvec(&block.wv, xi)).collect();

    let mut scores = vec![vec![0.0; n]; n];
    let mut weights = vec![vec![0.0; n]; n];
    let mut attended = vec![vec![0.0; d]; n];

    for i in 0..n {
        let mut row = vec![0.0; i + 1];
        for j in 0..=i {
            row[j] = dot(&q[i], &k[j]) / scale;
        }
        let w = calc::softmax(&row);
        for j in 0..=i {
            scores[i][j] = row[j];
            weights[i][j] = w[j];
            for c in 0..d {
                attended[i][c] += w[j] * v[j][c];
            }
        }
    }

    let a: Vec<Vec<f32>> = (0..n).map(|i| add_vec(&x[i], &attended[i])).collect();
    let pre: Vec<Vec<f32>> = a.iter().map(|ai| matvec(&block.w1, ai)).collect();
    let act: Vec<Vec<f32>> = pre.iter().map(|p| relu_vec(p)).collect();
    let ffn_out: Vec<Vec<f32>> = act.iter().map(|r| matvec(&block.w2, r)).collect();
    let h: Vec<Vec<f32>> = (0..n).map(|i| add_vec(&a[i], &ffn_out[i])).collect();

    let cache = BlockCache {
        x: x.to_vec(),
        q,
        k,
        v,
        _scores: scores,
        weights,
        _attended: attended,
        a,
        pre,
        act,
        _ffn_out: ffn_out,
        _h: h.clone(),
    };
    (h, cache)
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

        let mut x: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let emb = &self.token_embedding[context[i] as usize];
                let pe = &self.positional_encoding[i];
                (0..d).map(|b| emb[b] + pe[b]).collect()
            })
            .collect();

        let scale = (d as f32).sqrt();
        for block in &self.blocks {
            x = block_forward(block, &x, scale).0;
        }

        matvec(&self.wo, &x[n - 1])
    }

    fn train_step(&mut self, context: &[u16], target: u16, learning_rate: f32) -> f32 {
        let d = self.config.embedding_dim;
        let _f = self.config.ffn_hidden;
        let vocab = self.config.vocab_size as usize;
        let n = context.len();
        let scale = (d as f32).sqrt();

        // ----------------------- FORWARD (cached) ----------------------------
        let mut x: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let emb = &self.token_embedding[context[i] as usize];
                let pe = &self.positional_encoding[i];
                (0..d).map(|b| emb[b] + pe[b]).collect()
            })
            .collect();

        let mut caches = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (h, cache) = block_forward(block, &x, scale);
            caches.push(cache);
            x = h;
        }

        let logits = matvec(&self.wo, &x[n - 1]);
        let probs = calc::softmax(&logits);
        let loss = calc::cross_entropy_loss(&probs, target as usize);

        // ----------------------- BACKWARD ------------------------------------
        let d_logits = calc::cross_entropy_gradient(&probs, target as usize);

        // logits = Wo . h_last
        let mut d_wo = vec![vec![0.0; d]; vocab];
        let mut d_h = vec![vec![0.0; d]; n];
        for (vi, row) in self.wo.iter().enumerate() {
            for c in 0..d {
                d_wo[vi][c] = d_logits[vi] * x[n - 1][c];
                d_h[n - 1][c] += row[c] * d_logits[vi];
            }
        }

        for (block_idx, block) in self.blocks.iter_mut().enumerate().rev() {
            let cache = &caches[block_idx];
            let (d_x, d_wq, d_wk, d_wv, d_w1, d_w2) = block_backward(block, cache, &d_h, scale);

            update(&mut block.wq, &d_wq, learning_rate);
            update(&mut block.wk, &d_wk, learning_rate);
            update(&mut block.wv, &d_wv, learning_rate);
            update(&mut block.w1, &d_w1, learning_rate);
            update(&mut block.w2, &d_w2, learning_rate);

            d_h = d_x;
        }

        update(&mut self.wo, &d_wo, learning_rate);

        // X[0] = token_embedding[ctx[0]] + PE[0]
        let mut d_embedding = vec![vec![0.0; d]; vocab];
        for (i, &token) in context.iter().enumerate() {
            for b in 0..d {
                d_embedding[token as usize][b] +=
 d_h[i][b];
            }
        }

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

/// Backward through one block. d_h_in is the gradient w.r.t. the block output h.
fn block_backward(
    block: &Block,
    cache: &BlockCache,
    d_h_in: &[Vec<f32>],
    scale: f32,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n = cache.x.len();
    let d = cache.x[0].len();
    let f = block.w1.len();

    // h = a + ffn_out
    let mut d_a = d_h_in.to_vec();

    // ffn_out = W2 . act
    let mut d_w2 = vec![vec![0.0; f]; d];
    let mut d_act = vec![vec![0.0; f]; n];
    for i in 0..n {
        for (c, row) in block.w2.iter().enumerate() {
            for m in 0..f {
                d_w2[c][m] += d_h_in[i][c] * cache.act[i][m];
                d_act[i][m] += row[m] * d_h_in[i][c];
            }
        }
    }

    // act = relu(pre)
    let mut d_pre = vec![vec![0.0; f]; n];
    for i in 0..n {
        for m in 0..f {
            d_pre[i][m] = if cache.pre[i][m] > 0.0 { d_act[i][m] } else { 0.0 };
        }
    }

    // pre = W1 . a
    let mut d_w1 = vec![vec![0.0; d]; f];
    for i in 0..n {
        for (m, row) in block.w1.iter().enumerate() {
            for c in 0..d {
                d_w1[m][c] += d_pre[i][m] * cache.a[i][c];
                d_a[i][c] += row[c] * d_pre[i][m];
            }
        }
    }

    // a = x + attended
    let mut d_x = d_a.clone();
    let d_attended = d_a;

    // attended[i] = sum_{j<=i} weights[i][j] * v[j]
    let mut d_weights = vec![vec![0.0; n]; n];
    let mut d_v = vec![vec![0.0; d]; n];
    for i in 0..n {
        for j in 0..=i {
            for c in 0..d {
                d_weights[i][j] += d_attended[i][c] * cache.v[j][c];
                d_v[j][c] += cache.weights[i][j] * d_attended[i][c];
            }
        }
    }

    // Softmax over scores[i][0..=i]
    let mut d_scores = vec![vec![0.0; n]; n];
    for i in 0..n {
        let weighted_sum: f32 = (0..=i).map(|k| cache.weights[i][k] * d_weights[i][k]).sum();
        for j in 0..=i {
            d_scores[i][j] = cache.weights[i][j] * (d_weights[i][j] - weighted_sum);
        }
    }

    // scores[i][j] = (q[i] . k[j]) / sqrt(d)
    let mut d_q = vec![vec![0.0; d]; n];
    let mut d_k = vec![vec![0.0; d]; n];
    for i in 0..n {
        for j in 0..=i {
            for c in 0..d {
                d_q[i][c] += d_scores[i][j] * cache.k[j][c] / scale;
                d_k[j][c] += d_scores[i][j] * cache.q[i][c] / scale;
            }
        }
    }

    // q = Wq . x, k = Wk . x, v = Wv . x
    let mut d_wq = vec![vec![0.0; d]; d];
    let mut d_wk = vec![vec![0.0; d]; d];
    let mut d_wv = vec![vec![0.0; d]; d];
    for i in 0..n {
        for c in 0..d {
            for b in 0..d {
                d_wq[c][b] += d_q[i][c] * cache.x[i][b];
                d_x[i][b] += block.wq[c][b] * d_q[i][c];

                d_wk[c][b] += d_k[i][c] * cache.x[i][b];
                d_x[i][b] += block.wk[c][b] * d_k[i][c];

                d_wv[c][b] += d_v[i][c] * cache.x[i][b];
                d_x[i][b] += block.wv[c][b] * d_v[i][c];
            }
        }
    }

    (d_x, d_wq, d_wk, d_wv, d_w1, d_w2)
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
        let block = &mut model.blocks[0];
        block.wv = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
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
            embedding_dim: 4
,
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
