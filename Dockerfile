# Stage 1: Build
FROM rust:1.83-bookworm AS builder

# Install LLVM 18 and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    llvm-18-dev \
    libpolly-18-dev \
    libzstd-dev \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/qlang
COPY . .

# Build release binary
RUN cargo build --release

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libllvm18 \
    libzstd1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/qlang/target/release/qlang-cli /usr/local/bin/qlang-cli

EXPOSE 8080

ENTRYPOINT ["qlang-cli"]
CMD ["repl"]
