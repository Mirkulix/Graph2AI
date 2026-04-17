.PHONY: build build-no-llvm test docker docker-qo run api qo repl install clean lint docs bench

# `qlang-cli` is shipped by the `qlang-compile` crate; `qo` by the workspace root.
BINARY := qlang-cli
QO_BINARY := qo
DOCKER_IMAGE := qlang
DOCKER_IMAGE_QO := qo
INSTALL_DIR := /usr/local/bin
# QO server production port per docs/SESSION_HANDOFF.md.
QO_PORT ?= 4646

build:
	cargo build --release

# JIT disabled — no LLVM 18 toolchain needed.
build-no-llvm:
	cargo build --release --no-default-features

test:
	cargo test --workspace

docker:
	docker build -t $(DOCKER_IMAGE) .

docker-qo:
	docker build -t $(DOCKER_IMAGE_QO) -f Dockerfile.qo .

run:
	cargo run --release --example full_pipeline

api:
	cargo run --release -p qlang-compile --bin $(BINARY) -- serve --port 8080

# Run the QO supervisor server (binds to QO_PORT, defaults to 4646).
qo:
	QO_PORT=$(QO_PORT) cargo run --release --bin $(QO_BINARY)

repl:
	cargo run --release -p qlang-compile --bin $(BINARY) -- repl

install: build
	cp target/release/$(BINARY) $(INSTALL_DIR)/$(BINARY)

clean:
	cargo clean

lint:
	cargo clippy --workspace -- -D warnings
	cargo fmt --all -- --check

docs:
	cargo doc --workspace --no-deps --open

bench:
	cargo run --release --example benchmark
