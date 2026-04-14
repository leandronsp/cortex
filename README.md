# Cortex

A nano-LLM built from scratch in Rust with TDD. Following the Karpathy path: bigram baseline, MLP, then transformer.

Model math is zero-dep. Every layer by hand. Infrastructure (TOML config, TUI, serde) uses the minimum libraries that make sense.

## Build

```
make build
```

## Test

```
make test
```

Runs the full suite (72 tests: 66 unit + 4 train-bin integration + 2 chat-bin integration).

## Lint

```
make lint
```

`cargo clippy -- -D warnings`.

## Train a model

```
make train CONFIG=configs/bigram.toml
```

Reads the TOML, trains, writes weights to the path declared in the config.

## Chat with a trained model

```
make chat CONFIG=configs/bigram.toml
```

Opens a Ratatui TUI. Type a prompt, press Enter, read the completion. Esc quits.

## Configuration

Everything a run needs lives in one TOML file:

```toml
[model]
name = "bigram"
vocab_size = 256

[training]
corpus = "configs/corpus.txt"
epochs = 200
learning_rate = 1.0

[weights]
path = "target/bigram.weights"
```

`configs/bigram.toml` + `configs/corpus.txt` ship in the repo. `make train CONFIG=configs/bigram.toml` works out of the box on a fresh clone.

## Roadmap

- [x] **Step 1: BPE tokenizer** — encode/decode, pairs, merge, vocab building
- [x] **Step 2: Bigram model** — one matrix `vocab x vocab`, softmax, cross-entropy, SGD, greedy generate
- [x] **Library refactor** — `Model` trait, `Cortex` wrapper, TOML-driven `train` + `chat` bins, Ratatui TUI
- [ ] **Step 3: MLP with context window** — embedding + concat N tokens + hidden layers + ReLU + linear output. Manual backprop
- [ ] **Step 4: Self-attention + positional encoding** — single head, causal mask, sinusoidal PE
- [ ] **Step 5: FFN + residual** — completes the transformer block
- [ ] **Step 6: Multi-head + stack N blocks** — Q/K/V projections, multi-head attention
- [ ] **Step 7: AdamW + sampling** — optimizer with warmup, top-k/top-p, repetition penalty
- [ ] **Step 8: Tiny Shakespeare** — first readable output

## Architecture

DDD layout. Each folder is a bounded context.

```
src/
├── lib.rs
├── config.rs              TOML config parsing
├── bin/
│   ├── train.rs           entry point: config → Cortex::train → save weights
│   └── chat.rs            entry point: config → load weights → tui::run
├── tokenization/          BPE context
│   ├── tokenizer.rs       UTF-8 bytes → u16 tokens
│   ├── pairs.rs           count pairs, find most frequent
│   ├── merge.rs           replace pair with new token
│   └── bpe.rs             encode, build_vocab
├── model/                 Modeling context
│   ├── mod.rs             Model trait (forward, train_step, save, load)
│   ├── bigram.rs          Bigram: Vec<Vec<f32>> lookup table
│   └── registry.rs        create_model(name, cfg) → Box<dyn Model>
├── training/              Training context
│   ├── calc.rs            softmax, cross_entropy_loss, cross_entropy_gradient
│   └── cortex.rs          Cortex wrapper: train, generate, save, load
└── tui/                   Chat UI presentation
    └── app.rs             Ratatui transcript + input + event loop
```

Adding a new model (MLP, transformer) means implementing the `Model` trait and adding one arm to `src/model/registry.rs`. Binaries, TUI, Cortex, Makefile stay untouched.

## Data flow

```
text → BPE encode → tokens → Model::forward → softmax → cross_entropy → gradient → Model::update
                                   ↑                                                       │
                                   └────────────────── next epoch ────────────────────────┘
```

## License

MIT
