.DEFAULT_GOAL := help
.PHONY: help test test-slow lint build

include mk/cortex.mk

help: ## Show available targets
	@grep -hE '^[a-zA-Z._%-]+:.*##' Makefile mk/*.mk | awk -F ':.*## ' '{printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2}'

build: ## Build library and binaries
	@cargo build

lint: ## Run clippy with warnings as errors
	@cargo clippy -- -D warnings

test: ## Run fast tests (unit + fast integration; skips slow E2E)
	@cargo test

test-slow: ## Run the slow E2E training tests in release
	@cargo test --release -- --ignored
