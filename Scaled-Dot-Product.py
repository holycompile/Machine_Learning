
import numpy as np


# Query matrix
Q = np.array([
    [1, 0],
    [0, 1],
    [1, 1]
])

# Key matrix
K = np.array([
    [1, 0],
    [0, 1],
    [1, 1]
])

# Value matrix
V = np.array([
    [10, 0],
    [0, 10],
    [5, 5]
])


K_transposed=np.transpose(K)

print(K_transposed)

scores = np.dot(Q,K_transposed)

print(f"Scores:\n\n{scores}")

#------------------------------------
d_k=2 # for this particular program implementation

scaled = scores / np.sqrt(d_k)

print(f"Scaled matrix: \n \n {scaled}")
print("----")


def softmax(scaled):
    rows_max=np.max(scaled, axis=1, keepdims=True)
    exp_values=np.exp(scaled-rows_max)
    attention_weights = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return attention_weights

weights=softmax(scaled)

print(f"Weights: \n\n {weights}")

# Final Output
output= np.dot(weights,V)

print("FInal Output is here : \n\n", output)


