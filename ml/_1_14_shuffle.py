import numpy as np

# 1. Generate example data
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 1, 0, 1, 0])  # Label data

print("--- 1. Original Data (X, y) ---")
print("X:\n", X)
print("y:", y)
print("-" * 35)

# 2. Create and shuffle indices
# Use indices to shuffle X and y simultaneously
indices = np.arange(len(X))
np.random.shuffle(indices)

# 3. Rearrange data using the shuffled indices
X_shuffled = X[indices]
y_shuffled = y[indices]

print("--- 2. Data After Shuffling (X_shuffled, y_shuffled) ---")
print("X_shuffled:\n", X_shuffled)
print("y_shuffled:", y_shuffled)
