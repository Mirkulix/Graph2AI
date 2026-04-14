# IGQK

**Informationsgeometrische Quantenkompression** (Information-Geometric Quantum Compression)

A mathematical framework that unifies information geometry, quantum mechanics, and compression theory to compress neural network weights.

## Core Idea

Neural network weights live on a statistical manifold. Instead of treating them as flat vectors, IGQK models them as quantum density matrices and evolves them via a quantum gradient flow that simultaneously optimizes and compresses.

## Mathematical Foundation

### Statistical Manifold

Neural network weights form a Riemannian manifold M with the Fisher-Information metric:

```
g_ij(theta) = E[d_i log p(x; theta) * d_j log p(x; theta)]
```

### Quantum State on Manifold

Weights are density matrices rho: M -> C^{d x d} satisfying:
- `rho(theta) >= 0` (positive semidefinite)
- `Tr(rho(theta)) = 1` (normalized)
- `rho` is a smooth mapping

### Quantum Gradient Flow

The evolution equation that drives training and compression simultaneously:

```
d_rho/dt = -i[H, rho] - gamma * {G^-1 * grad_L, rho}
```

Where:
- `[H, rho]` = commutator = unitary evolution (quantum exploration)
- `{G^-1 * grad_L, rho}` = anticommutator = dissipative evolution (gradient descent)
- `H = -Delta_M` (Laplace-Beltrami operator)
- `gamma > 0` (damping parameter)

### Measurement and Collapse

Quantum measurement operators {M_w} collapse the continuous state to discrete weights via the Born rule:

```
P(w | rho) = Tr(rho * M_w)
```

## Key Theorems

### Theorem 5.1 (Convergence)

The quantum gradient flow converges to a stationary state rho* with:
```
E_rho*[L] <= min L(theta) + O(hbar)
```

### Theorem 5.2 (Compression Bound)

Minimum distortion D for compression to k-dimensional submanifold N:
```
D >= (n-k) / (2*beta) * log(1 + beta * sigma^2_min)
```

For ternary compression (n -> n/16):
```
D >= (15n/16) / (2*beta) * log(1 + beta * sigma^2_min)
```

### Theorem 5.3 (Entanglement & Generalization)

Entangled quantum states across layers improve generalization:
```
E_gen <= E_train + O(sqrt(I(A:B) / n))
```
where `I(A:B)` is quantum mutual information.

## Implementation

IGQK is implemented across 5 modules in `qlang-runtime` (see [[Architecture]]):

| Module | Contents |
|--------|----------|
| `igqk.rs` | Full Algorithm 1 (matrix ops, commutator, anticommutator, trace, inverse) |
| `fisher.rs` | Empirical Fisher information metric, natural gradient, damped inverse |
| `quantum_flow.rs` | Density matrix evolution, Born rule measurement, state collapse |
| `theorems.rs` | Theorem verification (convergence, compression bound, generalization) |
| `linalg.rs` | Matrix algebra (Jacobi eigenvalues, matrix exponential) |

## Algorithm 1 (Training)

```
Input: Data D, architecture f, compression submanifold N
Output: Compressed weights w* in N

1. Initialize rho_0 = |theta_init><theta_init|
   hbar = 0.1, gamma = 0.01

2. Quantum Training (T steps):
   For t = 1 to T:
     a. Compute loss L_t
     b. Compute gradient
     c. Compute Fisher metric G_t
     d. Update: rho_{t+1} = rho_t - dt * (i[H, rho_t] + gamma{G_t^-1 grad_L, rho_t})
     e. Renormalize: rho_{t+1} /= Tr(rho_{t+1})

3. Compress: theta* = project(E[rho_T]) onto N

4. Measure: w* ~ P(w|rho_T) = Tr(rho_T * M_w)
```

## Compression Results

```
Original:    392 KB (f32 weights)
Compressed:   25 KB (ternary: {-1, 0, +1})
Ratio:       16x
Accuracy:    100% retained
```

## Compression Targets

| Submanifold | Description |
|-------------|-------------|
| Ternary | Weights in {-1, 0, +1}, 16x compression |
| Low-rank | `rank(W) <= r`, parameterized by target rank |
| Sparse | `\|\|W\|\|_0 <= s`, sparsification |

## Graph Syntax

See [[Language]] for full syntax reference.

```qlang
node compressed = to_ternary(W1) @proof theorem_5_2
node lowrank = to_low_rank(W2, rank=16)
node sparse = to_sparse(W3, sparsity=0.9)
```

## Unifying Three Theories

IGQK unifies three approaches:

1. **HLWT** (Hybrid Laplace-Wavelet Transform) -- Fourier transform of quantum gradient flow in local coordinates (Proposition 6.1)
2. **TLGT** (Ternary Lie Group Theory) -- Ternary group G3 is a discrete subgroup of the quantum symmetry group (Proposition 6.2)
3. **FCHL** (Fractional Calculus Hebbian Learning) -- Uses fractional Laplace-Beltrami operator (Proposition 6.3)

## Debugging

When working with quantum states, monitor:
- `Tr(rho)` should always be 1.0
- Eigenvalues of rho should be in [0, 1]
- Von Neumann entropy: `S(rho) = -Tr(rho log rho)`
- Fisher metric condition number

See also: [[Glossary]] for term definitions, [[Decisions]] for why density matrices.

#igqk #theory #compression #quantum
