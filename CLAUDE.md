# Cortex Project Rules

Project-specific guidance. Global rules live at `/Users/leandronsp/Documents/code/.claude/CLAUDE.md`.

## Identity

Nano-LLM from scratch in Rust. Karpathy path: bigram → MLP → transformer. Every layer by hand.

## Dependency policy

- **Model math is zero-dep.** Softmax, cross-entropy, gradients, matrix ops, BPE. Never add a crate to `training/calc.rs` or any `Model` implementation.
- **Infrastructure may use libraries when they earn their keep.** Allowed today: `serde`, `toml`, `ratatui`, `crossterm`. New deps must be argued for against a hand-rolled alternative.

## Layout (DDD)

Folders are bounded contexts. Do not flatten into a single `src/` dump.

- `tokenization/` — BPE (tokenizer, pairs, merge, bpe)
- `model/` — `Model` trait, model impls, registry
- `training/` — math primitives (calc) + `Cortex` wrapper
- `tui/` — Ratatui chat presentation
- `config.rs` — flat, single TOML-parsing concern
- `bin/{train,chat}.rs` — thin entry points, no business logic

Adding a new model: implement `Model` + add one arm in `src/model/registry.rs`. Do not edit bins, TUI, `Cortex`, or the Makefile to register it.

## Tests

- Chesswav-style: inline `#[cfg(test)] mod tests { ... }` at the bottom of the source file. No `*_test.rs` sidecars.
- Integration tests for bins in `tests/`.
- Every behavior starts with a failing test. `cargo test` is the source of truth.
- Scientific TDD: RED for the right reason, minimal GREEN, revert fix to verify the test catches regressions, commit.

## Make targets

- `make build | test | lint | train CONFIG=... | chat CONFIG=...`
- `make help` lists everything.

## Runtime guarantees

- Training and inference stay pure Rust. No async, no threads, no runtime.
- Greedy argmax decoding until sampling lands (roadmap step 7).
- Stop condition for `Cortex::generate`: newline byte or `max_tokens` cap.

## Serialization

- Bigram weights serialize as little-endian `u16` vocab_size + `f32` row-major weights. Hand-rolled, no bincode.
- `Cortex::save/load` bundles BPE merges + model weights in one stream.

## Error style

- TOML config errors: surface the parser message, name the path. See `src/config.rs`.
- Binary exit codes: 0 success, 1 runtime failure, 2 usage error.
- No `unwrap` in binary entry points for user-facing paths. `unwrap` is fine in model math where it proves an invariant.

## Commit discipline

- Small commits per RED-GREEN-REFACTOR cycle.
- Conventional prefixes: `feat(...)`, `refactor(...)`, `test(...)`, `docs(...)`, `fix(...)`.
- Never `git add -A`. Stage files explicitly.

## Handoffs

When pausing a session, write `.handoff/{YYYYMMDD-HHMM}-{slug}.md` capturing goal, state, decisions, open questions, and next steps. The next agent should not need to read the conversation.
