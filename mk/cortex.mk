# Cortex targets

.PHONY: train chat test-fast test-slow

train: ## Train a model (usage: make train CONFIG=configs/bigram.toml)
	@cargo run --release --bin train -- $(CONFIG)

chat: ## Open chat TUI (usage: make chat CONFIG=configs/bigram.toml)
	@cargo run --release --bin chat -- $(CONFIG)

test-fast: ## Run unit tests only (lib), target <2s
	@cargo test --lib

test-slow: ## Run slow E2E tests (#[ignore]) in release mode, target 10-20s
	@cargo test --release -- --ignored
