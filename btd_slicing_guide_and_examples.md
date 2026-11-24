# Understanding & Slicing `(B, T, D)` Tensors

In deep learning and sequence modeling (NumPy, PyTorch, TensorFlow), sequential data is typically represented as a 3D array with shape **`(B, T, D)`**:
* **`B` (Batch):** Number of independent sequences or examples (e.g., stocks, users, sentences).
* **`T` (Time):** Number of time steps or sequence length (e.g., days, hours, words).
* **`D` (Dimension / Features):** Number of features recorded at each time step (e.g., price, temperature, embedding size).

---

## 1. Real-World Examples of `(B, T, D)` Data

### Example A: Weather Sensors — Shape `(3, 5, 2)`
* **`B = 3`** (Station A, Station B, Station C)
* **`T = 5`** (5 consecutive hours)
* **`D = 2`** (`[Temperature °C, Humidity %]`)

```python
weather = [
    # Batch 0: Station A (5 hours x 2 features)
    [ [22, 60], [23, 58], [25, 55], [24, 57], [21, 62] ],
    # Batch 1: Station B (5 hours x 2 features)
    [ [18, 70], [19, 68], [20, 65], [19, 67], [17, 72] ],
    # Batch 2: Station C (5 hours x 2 features)
    [ [30, 40], [31, 38], [33, 35], [32, 36], [29, 42] ]
]
```

### Example B: NLP Word Embeddings — Shape `(2, 4, 3)`
* **`B = 2`** (2 sentences)
* **`T = 4`** (4 words per sentence)
* **`D = 3`** (3 embedding dimensions per word)

```python
sentences = [
    # Batch 0: "I love deep learning"
    [
        [0.1, 0.5, 0.2],  # "I"
        [0.9, 0.1, 0.4],  # "love"
        [0.3, 0.8, 0.7],  # "deep"
        [0.4, 0.9, 0.6]   # "learning"
    ],
    # Batch 1: "AI is very cool"
    [
        [0.5, 0.8, 0.9],  # "AI"
        [0.1, 0.2, 0.1],  # "is"
        [0.2, 0.3, 0.2],  # "very"
        [0.8, 0.7, 0.3]   # "cool"
    ]
]
```

---

## 2. Slicing Walkthrough (Stock Market Dataset)

Consider a dataset tracking **2 stocks** over **4 days**, recording **3 metrics** (`[Open, Close, Volume]`).

```python
import numpy as np

# Shape: (2, 4, 3) -> (Stocks, Days, Metrics)
stocks = np.array([
    # Batch 0: Apple
    [
        [150.0, 152.0, 1000],  # Day 0 (Mon)
        [152.5, 151.0, 1200],  # Day 1 (Tue)
        [151.0, 155.0, 1500],  # Day 2 (Wed)
        [155.0, 158.0, 1100]   # Day 3 (Thu)
    ],
    # Batch 1: Google
    [
        [2800.0, 2820.0, 500], # Day 0 (Mon)
        [2825.0, 2810.0, 600], # Day 1 (Tue)
        [2815.0, 2850.0, 750], # Day 2 (Wed)
        [2850.0, 2890.0, 550]  # Day 3 (Thu)
    ]
])
```

### 1. Extract all metrics for one stock on a single day
* **Goal:** Google (`B=1`), Wednesday (`T=2`), all metrics (`D=:`).
* **Code:** `stocks[1, 2, :]`
* **Output:**
  ```python
  array([2815., 2850.,  750.])
  ```
* **Resulting Shape:** `(3,)` — Both Batch and Time axes collapse because integers (`1` and `2`) were used.

---

### 2. Extract a single feature column across all batches and time steps
* **Goal:** Closing price (`D=1`) for all stocks across all 4 days.
* **Code:** `stocks[:, :, 1]`
* **Output:**
  ```python
  array([[ 152.,  151.,  155.,  158.],   # Apple Closes
         [2820., 2810., 2850., 2890.]])  # Google Closes
  ```
* **Resulting Shape:** `(2, 4)` — The Feature axis (`D`) is removed.

---

### 3. Extract the final time step's prices for all stocks
* **Goal:** Thursday (`T=-1`), `[Open, Close]` (`D=0:2`), all stocks (`B=:`).
* **Code:** `stocks[:, -1, 0:2]`
* **Output:**
  ```python
  array([[ 155.,  158.],   # Apple Thursday [Open, Close]
         [2850., 2890.]])  # Google Thursday [Open, Close]
  ```
* **Resulting Shape:** `(2, 2)` — Time collapses via `-1`, while `0:2` keeps the Feature axis with length 2.

---

### 4. Extract one sequence while preserving the 3D `(B, T, D)` shape
* **Goal:** Isolate Apple (`B=0:1`) without losing the batch dimension.
* **Code:** `stocks[0:1, :, :]` *(or `stocks[:1]`)*
* **Output:**
  ```python
  array([[[ 150. ,  152. , 1000. ],
          [ 152.5,  151. , 1200. ],
          [ 151. ,  155. , 1500. ],
          [ 155. ,  158. , 1100. ]]])
  ```
* **Resulting Shape:** `(1, 4, 3)` — Using a slice (`0:1`) instead of an integer (`0`) preserves the axis.

---

### 5. Extract a time window across all stocks and metrics
* **Goal:** Tuesday and Wednesday (`T=1:3`) for every stock and metric.
* **Code:** `stocks[:, 1:3, :]`
* **Output:**
  ```python
  array([[[ 152.5,  151. , 1200. ],
          [ 151. ,  155. , 1500. ]],

         [[2825. , 2810. ,  600. ],
          [2815. , 2850. ,  750. ]]])
  ```
* **Resulting Shape:** `(2, 2, 3)` — All axes use slices, keeping all 3 dimensions intact.

---

### 6. Extract a single scalar value
* **Goal:** Google (`B=1`), Wednesday (`T=2`), Volume (`D=2`).
* **Code:** `stocks[1, 2, 2]`
* **Output:**
  ```python
  750.0
  ```
* **Resulting Shape:** `()` — Indexing every axis with an integer collapses all dimensions into a single scalar.

---

## 3. Quick Reference Cheat Sheet

| Indexing Rule | Effect on Axis | Example on `(2, 4, 3)` | Resulting Shape |
| :--- | :--- | :--- | :--- |
| **Integer** (`1`) | Removes (collapses) axis | `stocks[0]` | `(4, 3)` |
| **Slice** (`0:1` or `:`) | Preserves axis | `stocks[0:1]` | `(1, 4, 3)` |
| **Last step** (`-1`) | Collapses Time to final step | `stocks[:, -1, :]` | `(2, 3)` |
| **Feature slice** (`:, :, k`) | Extracts feature `k` across batch & time | `stocks[:, :, 2]` | `(2, 4)` |
| **Sub-window** (`a:b`) | Crops axis without removing it | `stocks[1, 1:3, 0:2]` | `(2, 2)` |