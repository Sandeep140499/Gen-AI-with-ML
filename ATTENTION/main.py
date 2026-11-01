
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example sentence
tokens = ["The", "cat", "sat", "on", "the", "mat"]

# Example attention matrix
attention = np.array([
    [0.30, 0.20, 0.10, 0.10, 0.15, 0.15],
    [0.10, 0.35, 0.20, 0.10, 0.10, 0.15],
    [0.05, 0.20, 0.40, 0.15, 0.10, 0.10],
    [0.10, 0.10, 0.20, 0.40, 0.10, 0.10],
    [0.20, 0.10, 0.10, 0.10, 0.40, 0.10],
    [0.10, 0.15, 0.15, 0.10, 0.20, 0.30]
])

plt.figure(figsize=(8, 6))

sns.heatmap(
    attention,
    annot=True,
    cmap="Blues",
    xticklabels=tokens,
    yticklabels=tokens
)

plt.title("Transformer Attention Visualization")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")

plt.tight_layout()
plt.show()
