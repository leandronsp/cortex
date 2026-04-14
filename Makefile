.DEFAULT_GOAL := help
.PHONY: help test lint build

include mk/cortex.mk

help: ## Show available targets
	@grep -hE '^[a-zA-Z._%-]+:.*##' Makefile mk/*.mk | awk -F ':.*## ' '{printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2}'

build: ## Build library and binaries
	@cargo build

lint: ## Run clippy with warnings as errors
	@cargo clippy -- -D warnings

test: ## Run all tests
	@cargo test
