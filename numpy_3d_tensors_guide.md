# 3D Tensors in NumPy: Visual Guide for ML & Transformers

## Part 1: What a 3D Tensor Looks Like & Shape `(B, T, D)`

In NumPy, a **3D array (tensor)** is a **stack of 2D matrices** along a third axis (`axis=0`).
In Transformers and sequence models, the standard 3D shape is **`(B, T, D)`**:
- `axis=0` (`B`): **Batch size** — how many independent sequences/sentences we process at once.
- `axis=1` (`T`): **Time / Sequence length** — how many tokens are in each sequence.
- `axis=2` (`D` / `d_model`): **Features / Embedding dimension** — how many numbers describe each token.

### Visual Representation

Think of `(B=2, T=3, D=4)` as **2 spreadsheets stacked behind each other**, each having 3 rows (tokens) and 4 columns (features):

```text
  Batch 1 (b=1) -->  +--------+--------+--------+--------+
                    /        /        /        /        /|
  Batch 0 (b=0) -> +--------+--------+--------+--------+ |
          tok 0    |  0.10  | -0.25  |  0.80  |  0.04  | +
                   +--------+--------+--------+--------+ |  axis=1 (T=3 tokens)
          tok 1    | -0.50  |  0.91  |  0.12  | -0.33  | +
                   +--------+--------+--------+--------+ |
          tok 2    |  0.44  |  0.15  | -0.72  |  0.60  |/
                   +--------+--------+--------+--------+
                      axis=2 (D=4 embedding dims) --->
   ^
   axis=0 (B=2 batch items, front to back)

   shape: (2, 3, 4)   |   ndim: 3   |   size: 24
```

### Creating 3D Tensors & Embedding Lookup in Transformers

When a batch of token IDs `(B, T)` is passed into an embedding table `E` of shape `(vocab_size, D)`, NumPy's **advanced indexing** automatically produces a **3D tensor `(B, T, D)`**!

```python
import numpy as np

B, T, D = 2, 3, 4
vocab_size = 100

# Embedding table (2D): (vocab_size, D)
E = np.random.randn(vocab_size, D).astype(np.float32) * 0.02

# Batch of token IDs (2D): (B, T)
token_ids = np.array([
    [12, 45, 7],   # Sequence 0
    [ 3, 88, 9],   # Sequence 1
])

# Lookup -> 3D Tensor: (B, T, D) = (2, 3, 4)
X = E[token_ids]
print("X.shape:", X.shape)  # (2, 3, 4)
```
