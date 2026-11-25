# 2D Matrices in NumPy: Visual Guide for ML & Transformers

## Part 1: What a 2D Matrix Looks Like & Shape `(Rows, Cols)`

In NumPy, a **2D array (matrix)** is a grid of numbers arranged along **two axes**:
- `axis=0` -> **rows** (moving top-to-bottom)
- `axis=1` -> **columns** (moving left-to-right)

### Visual Representation

Think of a 2D matrix as a spreadsheet or a sequence of token embeddings `(T, d_model)`:

```text
                  axis=1 (columns / features: d_model = 4)
                  col 0    col 1    col 2    col 3
                +--------+--------+--------+--------+
row 0 (token 0) |  0.10  | -0.25  |  0.80  |  0.04  |  |
                +--------+--------+--------+--------+  | axis=0
row 1 (token 1) | -0.50  |  0.91  |  0.12  | -0.33  |  | (rows / tokens: T = 3)
                +--------+--------+--------+--------+  v
row 2 (token 2) |  0.44  |  0.15  | -0.72  |  0.60  |
                +--------+--------+--------+--------+

shape: (3, 4)   |   ndim: 2   |   size: 12
```

### Creating 2D Matrices in NumPy

```python
import numpy as np

# 3 tokens, each with a 4-dimensional embedding -> shape (3, 4)
X = np.array([
    [ 0.10, -0.25,  0.80,  0.04],
    [-0.50,  0.91,  0.12, -0.33],
    [ 0.44,  0.15, -0.72,  0.60],
], dtype=np.float32)

print("shape:", X.shape)   # (3, 4)
print("ndim:", X.ndim)     # 2

# Common 2D matrices in ML & Transformers
W_q = np.random.randn(4, 4) * 0.02     # Projection weight matrix (d_model, d_k)
causal_mask = np.triu(np.ones((3, 3)), k=1) * -1e9  # Upper-triangular mask (T, T)
eye = np.eye(4)                        # Identity matrix (4, 4)
```

### Why 2D Is the Backbone of ML & Transformers
- **Tabular dataset**: `(num_samples, num_features)`
- **Single sequence of embeddings**: `(T, d_model)` where `T` = sequence length
- **Embedding lookup table**: `(vocab_size, d_model)`
- **Linear layer weights**: `(d_in, d_out)`
- **Self-attention score matrix**: `(T, T)` (every token attending to every other token)

---

## Part 2: 2D Slicing, Axis Reductions & Matrix Multiplication

### 2D Slicing: `M[row_slice, col_slice]`

```text
M = [[10, 11, 12, 13],
     [20, 21, 22, 23],
     [30, 31, 32, 33]]   # shape: (3, 4)
```

- `M[0, :]` (or `M[0]`) -> `[10, 11, 12, 13]` (entire row 0 -> 1D shape `(4,)`)
- `M[:, 1]` -> `[11, 21, 31]` (entire column 1 -> 1D shape `(3,)`)
- `M[0:2, 1:3]` -> `[[11, 12], [21, 22]]` (2x2 sub-matrix -> 2D shape `(2, 2)`)
- `M[-1, :]` -> `[30, 31, 32, 33]` (last token's features in a sequence!)

### Reductions Along `axis=0` vs `axis=1`

Understanding `axis` is crucial for Softmax and LayerNorm:

```text
          axis=1 (across columns / features) --->
        +----+----+----+----+
        | 10 | 11 | 12 | 13 |  -> np.sum(M, axis=1, keepdims=True) -> [[46]]
        +----+----+----+----+
        | 20 | 21 | 22 | 23 |                                      -> [[86]]
        +----+----+----+----+
          |    |    |    |
          v    v    v    v     axis=0 (down rows / tokens)
        [[30, 32, 34, 36]]     <- np.sum(M, axis=0, keepdims=True)
```

- `axis=0` collapses **rows** (computes stats **per feature** across all tokens/samples).
- `axis=1` collapses **columns** (computes stats **per token/row** across all features — used in **Softmax** & **LayerNorm**!).
- Always pass `keepdims=True` when you plan to divide or subtract back into `M` via broadcasting!

### Transpose (`.T`) & Matrix Multiplication (`@`)

- **Transpose (`A.T`)**: Swaps rows and columns. Shape `(T, d_k)` becomes `(d_k, T)`.
- **Matrix Multiplication (`A @ B`)**: Inner dimensions must match:

```text
(m, k) @ (k, n)  --->  (m, n)
    ^     ^
    +-----+-- must match!
```

```python
X = np.random.randn(3, 4)   # (T=3, d_model=4)
W = np.random.randn(4, 2)   # (d_model=4, d_k=2)

out = X @ W                 # (3, 4) @ (4, 2) -> (3, 2)
```
