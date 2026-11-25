# 1D Arrays (Vectors) in NumPy: Visual Guide for ML & Transformers

## Part 1: What a 1D Array Looks Like & Shape `(N,)`

In NumPy, a **1D array** (often called a **vector** in linear algebra and Machine Learning) is a single flat sequence of numbers along **one axis (`axis=0`)**.

### Visual Representation

Think of a 1D array as a single row of lockers, each holding one value at an index `0, 1, 2, ...`:

```text
Index:     0      1      2      3      4
        +------+------+------+------+------+
x =     | 0.12 | -0.4 | 0.88 | 0.05 | -0.9 |
        +------+------+------+------+------+
                  axis=0  --->
        shape: (5,)   |   ndim: 1   |   size: 5
```

Notice the trailing comma in `(5,)`:
- `(5,)` means **1 dimension** with 5 elements.
- `(1, 5)` or `(5, 1)` are **2D matrices** (a row matrix and a column matrix), NOT 1D vectors!

### Creating 1D Arrays in NumPy

```python
import numpy as np

# From a Python list (e.g., a single token's embedding or feature vector)
v = np.array([0.12, -0.40, 0.88, 0.05, -0.90], dtype=np.float32)

print("v:", v)
print("shape:", v.shape)   # (5,)
print("ndim:", v.ndim)     # 1
print("dtype:", v.dtype)   # float32

# Common ML initializations
zeros = np.zeros(4)                 # [0., 0., 0., 0.]  -> e.g., initial bias vector
ones = np.ones(4)                   # [1., 1., 1., 1.]  -> e.g., LayerNorm gamma init
positions = np.arange(0, 6)         # [0, 1, 2, 3, 4, 5] -> token positions in a sequence
token_ids = np.array([101, 2054, 2003, 102])  # integer token IDs from a tokenizer
```

### Why 1D Matters in ML & Transformers
- **Token IDs for one sentence**: `[101, 2054, 2003, 102]` has shape `(T,)` where `T` is sequence length.
- **Bias vectors (`b`)**: Added to every token's representation, shape `(d_model,)`.
- **Single token embedding**: One word/subword represented as a point in `d_model`-dimensional space, shape `(d_model,)`.

---

## Part 2: Indexing, Slicing & Basic 1D Operations

### Indexing & Slicing (`start:stop:step`)

```text
x = [10, 20, 30, 40, 50, 60]
      0   1   2   3   4   5    (positive indices)
     -6  -5  -4  -3  -2  -1    (negative indices)
```

- `x[0]` -> `10` (first element)
- `x[-1]` -> `60` (last element, e.g., last token logit in causal LM)
- `x[1:4]` -> `[20, 30, 40]` (`stop` index 4 is **excluded**)
- `x[:3]` -> `[10, 20, 30]` (first 3 elements, e.g., top-3 context tokens)
- `x[::2]` -> `[10, 30, 50]` (every 2nd element — used in sin/cos positional encoding!)
- `x[1::2]` -> `[20, 40, 60]` (odd indices — `cos` terms in Transformer positional encoding!)
- `x[::-1]` -> `[60, 50, 40, 30, 20, 10]` (reversed)

> **Important:** Slicing a NumPy array returns a **view**, not a copy! Modifying `sub = x[1:4]; sub[0] = 999` also mutates `x[1]`. Use `.copy()` if you need an independent array.

### Element-Wise Operations & Reductions

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

# Element-wise math (same shape required, or scalar broadcasting)
add = a + b          # [5.0, 7.0, 9.0]  -> e.g., residual connection: x + sublayer_out
mul = a * b          # [4.0, 10.0, 18.0] -> e.g., gating (SwiGLU / LayerNorm scale)
scaled = a * 0.5     # [0.5, 1.0, 1.5]  -> scalar broadcasting

# Reductions (collapse 1D -> scalar)
s = np.sum(a)        # 6.0
m = np.mean(a)       # 2.0  (used in LayerNorm!)
std = np.std(a)      # 0.8165 (used in LayerNorm!)
idx = np.argmax(b)   # 2    (greedy decoding: pick highest-logit token!)
```

### The Dot Product (`np.dot` / `@`)

The **dot product** measures similarity/alignment between two 1D vectors of equal length:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=0}^{d-1} a_i b_i$$

```python
q = np.array([1.0, 0.5, -1.0])   # Query vector for 1 token
k = np.array([0.8, 0.5, -0.5])   # Key vector for 1 token

score = np.dot(q, k)   # or q @ k -> 1.0*0.8 + 0.5*0.5 + (-1.0)*(-0.5) = 1.55
```

- Large positive dot product -> vectors point in a similar direction (high attention).
- Zero dot product -> orthogonal (unrelated).
- Negative dot product -> opposite directions.

---

## Part 3: 1D Vectors in Machine Learning & Building a Transformer

In a Transformer, 1D arrays appear at key moments:

1. **Token IDs** for a single prompt: `(T,)`
2. **Logits** for the next-token prediction: `(vocab_size,)`
3. **A single token's embedding**: `(d_model,)`
4. **Learnable LayerNorm parameters (`gamma`, `beta`)** and **Linear layer biases (`b`)**: `(d_model,)`

### 1. Numerically Stable Softmax on a 1D Logit Vector

When a Transformer outputs raw scores (logits) for the next token over a vocabulary of size `V`, we convert that 1D vector into probabilities:

```python
def softmax_1d(logits: np.ndarray) -> np.ndarray:
    # Subtract max for numerical stability (prevents exp() overflow)
    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)

logits = np.array([2.0, 1.0, 0.1, -1.5])  # shape: (4,)
probs = softmax_1d(logits)
print("Probabilities:", np.round(probs, 4))  # sums to 1.0
print("Predicted token ID:", np.argmax(probs))
```

### 2. Layer Normalization on a Single Token Vector `(d_model,)`

LayerNorm normalizes a single token's feature vector across its `d_model` dimensions to have mean 0 and variance 1, then scales (`gamma`) and shifts (`beta`):

```python
def layer_norm_1d(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x)
    var = np.var(x)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

x = np.array([1.2, -0.5, 3.1, 0.2])       # 1 token, d_model=4
gamma = np.ones(4)
beta = np.zeros(4)
print("LayerNorm output:", np.round(layer_norm_1d(x, gamma, beta), 4))
```

### 3. Positional Encoding Trick Using 1D Slicing (`::2` and `1::2`)

In *"Attention Is All You Need"*, even indices use `sin` and odd indices use `cos`:

```python
d_model = 8
pos = 3
pe = np.zeros(d_model)  # shape: (8,)

div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
pe[0::2] = np.sin(pos * div_term)   # even indices: 0, 2, 4, 6
pe[1::2] = np.cos(pos * div_term)   # odd indices:  1, 3, 5, 7
```

### 4. Stepping Up from 1D to 2D (`reshape` / `np.newaxis`)

To feed a 1D vector into matrix multiplication, you often promote `(N,)` to `(1, N)` or `(N, 1)`:

```python
v = np.array([1, 2, 3, 4])   # shape: (4,)
row = v[np.newaxis, :]       # shape: (1, 4) -> 1 token with 4 features
col = v[:, np.newaxis]       # shape: (4, 1) -> column vector
```
