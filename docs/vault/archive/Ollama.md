# Ollama

QLANG integrates with local LLMs via the Ollama API. The client is implemented as raw HTTP over `std::net::TcpStream` -- no external HTTP crates.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `QLANG_OLLAMA_HOST` | `127.0.0.1` | Ollama server host |
| `QLANG_OLLAMA_PORT` | `11434` | Ollama server port |

## Rust API

```rust
use qlang_runtime::ollama::{OllamaClient, ChatMessage};

// Create client from env vars (or defaults)
let client = OllamaClient::from_env();

// Health check
let is_up = client.health()?;

// List locally available models
let models = client.list_models()?;

// One-shot generation
let reply = client.generate("llama3", "Explain IGQK", None)?;

// With system prompt
let reply = client.generate("llama3", "Hello", Some("Be concise"))?;

// Multi-turn chat
let messages = vec![
    ChatMessage::system("You are a QLANG expert."),
    ChatMessage::user("What is a density matrix?"),
];
let reply = client.chat("llama3", messages)?;
```

## Graph Operations

Two Ollama operations are available as graph nodes (see [[Language]]):

| Op | Graph Syntax | Description |
|----|-------------|-------------|
| `OllamaGenerate` | `ollama_generate(prompt, model="llama3")` | One-shot text generation |
| `OllamaChat` | `ollama_chat(messages, model="llama3")` | Multi-turn conversation |

These are ops 37 and 38 in the [[Protocol]] binary format.

## CLI Commands

```bash
# Check if Ollama is running
qlang-cli ollama health

# List available models
qlang-cli ollama models

# Generate text
qlang-cli ollama generate

# Chat completion
qlang-cli ollama chat
```

## AI-Designed Training

The `ai-train` command uses Ollama to have an LLM design a training pipeline:

```bash
qlang-cli ai-train --model llama3 --quick
```

The LLM designs the network architecture, and QLANG executes the training.

## Autonomous Loop

The `autonomous` command creates a feedback loop where an LLM iteratively improves a model:

```bash
qlang-cli autonomous --task "classify MNIST" --target 95 --model llama3
```

1. LLM designs initial architecture
2. QLANG trains the model
3. LLM evaluates results and suggests improvements
4. Repeat until target accuracy is met

See [[Agents]] for more on the autonomous agent system.

## Error Handling

| Error | Cause |
|-------|-------|
| `Connection` | Ollama server not running or unreachable |
| `Http { status, body }` | Server returned an error (404, 500, etc.) |
| `InvalidResponse` | Malformed HTTP response |
| `Json` | Response body is not valid JSON |
| `Timeout` | No response within 120 seconds |

## Supported Models

Any model available through Ollama works. Common choices:

| Model | Size | Use Case |
|-------|------|----------|
| `llama3` | 8B | General purpose |
| `llama3:70b` | 70B | Complex reasoning |
| `mistral:7b` | 7B | Fast inference |
| `codellama` | 7-34B | Code generation |
| `deepseek-coder` | 6-33B | Code understanding |

#ollama #llm #ai
