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

---

## Part 2: 3D Slicing, `axis=-1` Reductions & Batched Matmul (`@`)

### 3D Slicing: `X[batch_slice, time_slice, dim_slice]`

Suppose `X` has shape `(B=2, T=5, D=8)`:

- `X[0]` or `X[0, :, :]` -> shape `(5, 8)` (all tokens & features for **batch item 0**)
- `X[:, -1, :]` -> shape `(2, 8)` (**last token's** embedding for every sequence in the batch — used for next-token prediction!)
- `X[:, :-1, :]` -> shape `(2, 4, 8)` (all tokens except the last — autoregressive **inputs**)
- `X[:, 1:, :]` -> shape `(2, 4, 8)` (all tokens shifted by 1 — autoregressive **targets**)
- `X[:, :, :4]` -> shape `(2, 5, 4)` (first half of the feature dimensions)

### Reductions on 3D Tensors (`axis=-1`)

In Transformers, **Softmax** and **LayerNorm** almost always normalize across the **last axis (`axis=-1`)**:

- For `X` of shape `(B, T, D)`:
  - `np.mean(X, axis=-1, keepdims=True)` -> shape `(B, T, 1)` (mean of each token's `D` features).
- For attention scores `S` of shape `(B, T, T)`:
  - `softmax(S, axis=-1)` -> shape `(B, T, T)` (each query token's attention weights over all `T` key tokens sum to `1.0`).

### Batched Matrix Multiplication (`@`) & `transpose`

In NumPy, `@` on 3D arrays performs **batched matrix multiplication** over the last two axes `(T, D)`, while keeping the batch axis `B` aligned:

```text
(B, m, k) @ (B, k, n)  --->  (B, m, n)
```

Notice that `.T` on a 3D array reverses **all** axes `(B, T, D) -> (D, T, B)`, which is **NOT** what you want for `Q @ K^T`! Instead, swap only the last two axes:

```python
Q = np.random.randn(2, 5, 4)   # (B=2, T=5, d_k=4)
K = np.random.randn(2, 5, 4)   # (B=2, T=5, d_k=4)

# Swap only axes 1 and 2: (B, T, d_k) -> (B, d_k, T)
K_T = K.transpose(0, 2, 1)     # or np.swapaxes(K, -2, -1)

# Batched attention scores: (2, 5, 4) @ (2, 4, 5) -> (2, 5, 5)
scores = Q @ K_T
print("scores.shape:", scores.shape)  # (2, 5, 5)
```

---

## Part 3: 3D Tensors in Building a Transformer (Batched Attention & Multi-Head Split)

### 1. Projecting a 3D Tensor with a 2D Weight Matrix

In NumPy, a 3D activation `(B, T, d_model)` can be multiplied directly by a 2D weight matrix `(d_model, d_k)`:

```text
(B, T, d_model) @ (d_model, d_k)  --->  (B, T, d_k)
```

NumPy broadcasts the 2D weight matrix across all `B` batch items automatically!

```python
B, T, d_model, d_k = 2, 5, 8, 4
X = np.random.randn(B, T, d_model)
W_q = np.random.randn(d_model, d_k) * 0.1

Q = X @ W_q   # (2, 5, 8) @ (8, 4) -> (2, 5, 4)
```

### 2. Batched 3D Scaled Dot-Product Self-Attention

```python
def softmax_last_axis(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def batched_self_attention(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> np.ndarray:
    B, T, _ = X.shape
    d_k = W_q.shape[1]

    Q = X @ W_q                                        # (B, T, d_k)
    K = X @ W_k                                        # (B, T, d_k)
    V = X @ W_v                                        # (B, T, d_k)

    # (B, T, d_k) @ (B, d_k, T) -> (B, T, T)
    scores = (Q @ np.swapaxes(K, -2, -1)) / np.sqrt(d_k)

    # 2D causal mask (T, T) broadcasts across batch axis B -> (B, T, T)
    mask = np.triu(np.ones((T, T)), k=1) * -1e9
    scores = scores + mask

    weights = softmax_last_axis(scores)                # (B, T, T)
    return weights @ V                                 # (B, T, d_k)
```

### 3. Splitting & Merging Heads (`3D -> 4D -> 3D`)

To run **Multi-Head Attention** with `H` heads (where `d_model = H * d_head`), you reshape the 3D tensor `(B, T, d_model)` into 4D `(B, H, T, d_head)` and merge it back to 3D afterward:

```python
B, T, d_model = 2, 5, 8
H = 2
d_head = d_model // H   # 4

# 1. Split d_model into (H, d_head) and move H before T:
# (B, T, d_model) -> (B, T, H, d_head) -> (B, H, T, d_head)
X_heads = X.reshape(B, T, H, d_head).transpose(0, 2, 1, 3)

# 2. After attention per head, merge heads back to 3D:
# (B, H, T, d_head) -> (B, T, H, d_head) -> (B, T, d_model)
X_merged = X_heads.transpose(0, 2, 1, 3).reshape(B, T, d_model)
```

### Summary: 1D -> 2D -> 3D in a Transformer

| Dimension | Typical Shape | What It Represents |
| :--- | :--- | :--- |
| **1D** | `(d_model,)` or `(T,)` | Single token embedding, bias vector, or one sequence's token IDs |
| **2D** | `(T, d_model)` or `(d_in, d_out)` | One sequence of tokens, linear weight matrix, or `(T, T)` attention map |
| **3D** | `(B, T, d_model)` | A full **batch** of sequences flowing through Transformer blocks |
