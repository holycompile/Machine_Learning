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
