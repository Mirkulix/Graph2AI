# QLANG Web Dashboard

Real-time visualization dashboard for the QLANG graph-based AI-to-AI programming language.

## Quick Start

### Live Mode (with server)

1. Start the QLANG WebSocket server:

```bash
qlang-cli web --port 8081
```

2. Open `web/index.html` in your browser.

The dashboard connects to `ws://localhost:8081` and auto-reconnects on disconnect.

### Demo Mode (no server needed)

Open `web/demo.html` directly in your browser. It runs a self-contained simulation:

- A 9-node MNIST classifier graph executing node by node
- 30 epochs of training with decreasing loss and increasing accuracy
- Agent-to-agent communication messages
- IGQK ternary compression results (94.3 MB to 6.6 MB, 14.3x ratio)
- Model saves to the registry

No server, no build step, no dependencies.

## Dashboard Panels

### Header

System-wide stats: node count, total FLOPs, memory usage, uptime. Connection status indicator shows green when the WebSocket is connected.

### Left Panel -- Graph Visualizer

SVG-based visualization of the computation graph. Nodes are colored by type:

- **Blue** -- Input nodes (data entering the graph)
- **Green** -- Operation nodes (MatMul, Relu, Softmax, etc.)
- **Orange** -- Output nodes (results leaving the graph)
- **Purple** -- Quantum/IGQK nodes (Evolve, Measure, Entangle, etc.)

Nodes pulse with animation when executing. Click any node to see its tensor shape, execution time, and sample values in the info panel below the graph.

### Center Panel -- Activity Feed

Scrolling log of all system events, color-coded by category:

- **Green (graph)** -- Node executions, graph updates
- **Blue (agent)** -- Messages between AI agents
- **Yellow (training)** -- Epoch results, loss/accuracy updates
- **Gray (system)** -- Model saves, connections, status
- **Red (error)** -- Errors and warnings

Use the filter buttons (All, Graph, Agent, Training, System) to focus on specific event types.

### Right Panel -- Metrics

- **Training chart** -- Canvas 2D chart showing loss (red) and accuracy (green) over epochs
- **IGQK Compression** -- Original size, compressed size, compression ratio, accuracy delta
- **System Resources** -- Nodes executed, total FLOPs, peak memory
- **Model Registry** -- List of saved model checkpoints with sizes and timestamps

### Bottom Panel -- REPL

Interactive input for `.qlang` code. Type code and press Enter (or click Run) to send it to the server for execution. Results appear inline.

## WebSocket Message Format

All messages are JSON objects with a `type` field. The dashboard sends and receives the following message types:

### Server to Dashboard

#### `graph` -- Full graph definition

```json
{
  "type": "graph",
  "nodes": [
    {
      "id": "n0",
      "label": "x",
      "type": "input",
      "op": "Input",
      "shape": [1, 784],
      "values": [0.12, -0.03, 0.87]
    }
  ],
  "edges": [
    { "from": "n0", "to": "n2" }
  ]
}
```

Node `type` determines color: `"input"`, `"op"`, `"output"`, `"quantum"`.

#### `node_exec` -- Single node execution event

```json
{
  "type": "node_exec",
  "node_id": "n2",
  "name": "matmul",
  "op": "MatMul",
  "time_ms": 1.23
}
```

#### `training` -- Training epoch result

```json
{
  "type": "training",
  "epoch": 5,
  "loss": 0.4321,
  "accuracy": 0.872
}
```

#### `agent` -- Agent-to-agent message

```json
{
  "type": "agent",
  "from": "Optimizer",
  "to": "Compressor",
  "content": "Training converged. Ready for compression."
}
```

#### `compression` -- IGQK compression result

```json
{
  "type": "compression",
  "original_size": "94.3 MB",
  "compressed_size": "6.6 MB",
  "ratio": "14.3",
  "accuracy_delta": -0.8
}
```

#### `model_saved` -- Model saved to registry

```json
{
  "type": "model_saved",
  "name": "mnist_ternary_v1.qlang",
  "size": "6.6 MB"
}
```

#### `repl_result` -- REPL execution result

```json
{
  "type": "repl_result",
  "output": "Graph compiled: 5 nodes, 4 edges",
  "error": false
}
```

#### `system` -- System status update

```json
{
  "type": "system",
  "text": "Checkpoint saved",
  "resources": {
    "nodesExec": 1024,
    "flops": 5000000,
    "peakMem": 64
  }
}
```

#### `error` -- Error event

```json
{
  "type": "error",
  "text": "Node n3 failed: tensor shape mismatch"
}
```

### Dashboard to Server

#### `repl` -- Execute QLANG code

```json
{
  "type": "repl",
  "code": "graph G { x = input([784]) }"
}
```

## Technical Details

- Pure HTML/CSS/JS -- no frameworks, no npm, no build step
- CSS Grid layout with responsive breakpoint at 1024px
- Canvas 2D API for training charts (no chart library)
- SVG for graph visualization with click interaction
- WebSocket with automatic reconnection (3-second interval)
- Dark theme: background `#0d1117`, cards `#161b22`, accent `#58a6ff`

## Color Palette

| Token     | Hex       | Usage                        |
|-----------|-----------|------------------------------|
| bg        | `#0d1117` | Page background              |
| card      | `#161b22` | Panel backgrounds            |
| border    | `#30363d` | Borders, grid lines          |
| text      | `#c9d1d9` | Primary text                 |
| muted     | `#8b949e` | Secondary text, labels       |
| accent    | `#58a6ff` | Links, input nodes, highlights |
| success   | `#3fb950` | Op nodes, accuracy, OK       |
| warning   | `#d29922` | Training events              |
| error     | `#f85149` | Errors, loss line            |
| purple    | `#bc8cff` | Quantum nodes, REPL prompt   |
| orange    | `#f0883e` | Output nodes                 |
