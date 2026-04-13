# Cortex

A nano-LLM built from scratch in Rust with TDD. Following the Karpathy path: bigram baseline, MLP, then transformer.

Zero external dependencies. Every layer implemented by hand.

## Build

```
cargo build
```

## Test

```
cargo test
```

## Run

```
cargo run
```

Enter a training corpus, wait for training to finish, then type single characters to predict the next one.

## Roadmap

- [x] **Step 1: BPE tokenizer** — encode/decode, pairs, merge, vocab building. Embedding layer (token ID → vector)
- [x] **Step 2: Bigram model** — one matrix `vocab x vocab`. Forward (row lookup → logits), softmax, cross-entropy loss, SGD, greedy generate. First end-to-end milestone
- [ ] **Step 3: MLP with context window** — embedding + concat N tokens + hidden layers + ReLU + linear output. Manual backprop
- [ ] **Step 4: Self-attention + positional encoding** — single head, causal mask, sinusoidal PE
- [ ] **Step 5: FFN + residual** — completes the transformer block
- [ ] **Step 6: Multi-head + stack N blocks** — Q/K/V projections, multi-head attention
- [ ] **Step 7: AdamW + sampling** — optimizer with warmup, top-k/top-p
- [ ] **Step 8: Tiny Shakespeare** — first readable output

## Current state

56 tests. Full bigram pipeline working end-to-end:

```
text → BPE encode → tokens → forward → softmax → cross_entropy → gradient → update
                                 ↑                                              │
                                 └──── next epoch ─────────────────────────────┘
```

The model trains on consecutive token pairs and predicts the next token via argmax of softmax(forward(token)).

## Architecture

```
src/
  tokenizer.rs   → UTF-8 bytes → u16 tokens
  pairs.rs       → count pairs, find most frequent
  merge.rs       → replace pair with new token
  bpe.rs         → BPE encode/decode, build vocab
  embedding.rs   → token ID → vector (orphan until step 3)
  calc.rs        → softmax, cross_entropy_loss, cross_entropy_gradient
  bigram.rs      → one matrix, forward lookup, SGD update
  main.rs        → interactive train + predict loop
```

## License

MIT
