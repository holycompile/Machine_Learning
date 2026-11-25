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
