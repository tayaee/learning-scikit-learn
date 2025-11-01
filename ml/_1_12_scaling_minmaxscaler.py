import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Example data: Height (cm) data. Assuming 2000cm is an outlier to check the effect of large values.
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

# Create a MinMaxScaler object
minmax_scaler = MinMaxScaler()

# Fit and transform the data
data_scaled = minmax_scaler.fit_transform(data)

print("--- Original Data ---")
print(data.flatten())
print("\n--- Data After MinMax Scaling (0 to 1 Range) ---")
print(data_scaled.flatten())
