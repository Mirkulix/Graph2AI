.PHONY: build test docker run api repl install clean lint docs bench

BINARY := qlang-cli
DOCKER_IMAGE := qlang
INSTALL_DIR := /usr/local/bin

build:
	cargo build --release

test:
	cargo test --workspace

docker:
	docker build -t $(DOCKER_IMAGE) .

run:
	cargo run --release --example full_pipeline

api:
	cargo run --release --bin $(BINARY) -- serve --port 8080

repl:
	cargo run --release --bin $(BINARY) -- repl

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
