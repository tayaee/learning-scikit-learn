from collections import Counter

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification  # Function for generating imbalanced data

# 1. Generate an imbalanced dataset
# n_samples: Total number of samples, n_features: Number of features
# n_informative: Number of informative features, n_redundant: Number of redundant features
# n_classes: Number of classes (2), weights: Class ratio (90%:10%)
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],
    flip_y=0,
    random_state=42,
)

# Convert to a DataFrame (for easier viewing)
df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(10)])
df["Target"] = y

print("--- 1. Class Distribution of the Original Data ---")
print(Counter(y))
# Result: Counter({0: 900, 1: 100})
print("-" * 40)


# Create a SMOTE object (by default, it matches the minority class count to the majority class count)
smote = SMOTE(random_state=42)

# Apply SMOTE to the data to get the new X and y
X_resampled, y_resampled = smote.fit_resample(X, y)

print("--- 2. Class Distribution After SMOTE Application ---")
print(Counter(y_resampled))
# Result: Counter({0: 900, 1: 900}) - The count of the minority class (1) now equals the majority class (0)
print("-" * 40)

# 3. Compare dataset sizes
print(f"Number of rows in the original data: {len(X)}")
print(f"Number of rows after SMOTE application: {len(X_resampled)}")
