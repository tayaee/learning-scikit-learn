import numpy as np
from sklearn.preprocessing import StandardScaler

# Example data (Same as the previous Scaling example)
data = np.array(
    [
        [160],
        [175],
        [185],
        [190],
        [2000],  # Outlier
    ],
    dtype=np.float64,
)

# Create a StandardScaler object
standard_scaler = StandardScaler()

# Fit and transform the data
data_normalized = standard_scaler.fit_transform(data)

print("--- Original Data ---")
print(data.flatten())
print("\n--- Data After Standardization (Mean 0, Std Dev 1) ---")
print(data_normalized.flatten())
