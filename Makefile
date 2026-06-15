.DEFAULT_GOAL := help
.PHONY: help test lint build

include mk/cortex.mk

help: ## Show available targets
	@grep -hE '^[a-zA-Z._%-]+:.*##' Makefile mk/*.mk | awk -F ':.*## ' '{printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2}'

build: ## Build library and binaries
	@cargo build

lint: ## Run clippy with warnings as errors
	@cargo clippy -- -D warnings

test: ## Run all tests (fast: skips ignored e2e tests)
	@cargo test

test-e2e: ## Run slow e2e tests (ignored by default)
	@cargo test -- --ignored

test-release: ## Run all tests in release mode
	@cargo test --release
