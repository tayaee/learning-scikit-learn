"""
Feature Preprocessing and Engineering

Various techniques to process data into a form suitable for model training.
    sklearn.preprocessing.OneHotEncoder: Converts categorical data into a format machine learning models can understand using one-hot encoding. (e.g., 'Seoul', 'Busan' -> [1, 0], [0, 1])
    sklearn.preprocessing.MinMaxScaler: Scales data to values between 0 and 1. This is another important scaling method besides StandardScaler.
    sklearn.feature_extraction.text: Functionality for extracting features from text data. CountVectorizer and TfidfVectorizer are representative examples.

MinMaxScaler:
    MinMaxScaler is another important preprocessing tool for data scaling.
    This transformer converts all feature values into a specific range, typically between 0 and 1.
    Unlike StandardScaler, MinMaxScaler does not change the data's distribution but simply readjusts the range of values to unify the data scale.
"""

# 1. Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 2. Generate example data
# Generate a dataset where the two features have different scales using np.random.normal().
# - loc: Mean of the normal distribution (Feature 1: 50, Feature 2: 5)
# - scale: Standard deviation of the normal distribution (Feature 1: 10, Feature 2: 2)
# - size: Shape of the data to generate (100 samples, 2 features)
mean = [50, 5]
std = [10, 2]
X = np.random.normal(loc=mean, scale=std, size=(100, 2))

# 3. Data Splitting
# Split the data into training and test sets.
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# ---
# 4. Apply MinMax Scaler
# Create an instance of the MinMax Scaler estimator (transformer)
scaler = MinMaxScaler()

# fit_transform: Calculates (fit) the minimum and maximum values for each feature in the training data (X_train),
#                and then scales (transform) the data between 0 and 1 based on these values.
#                Consequently, the minimum value for each feature in X_train_scaled becomes 0, and the maximum becomes 1.
X_train_scaled = scaler.fit_transform(X_train)

# transform: The test data is only scaled using the minimum and maximum values from the training data.
X_test_scaled = scaler.transform(X_test)

# ---
# 5. Compare Data Before and After Scaling
print("Minimum values of the training data before scaling:")
print(f"Feature 1: {X_train[:, 0].min():.2f}, Feature 2: {X_train[:, 1].min():.2f}")
print("\nMinimum values of the training data after scaling:")
print(f"Feature 1: {X_train_scaled[:, 0].min():.2f}, Feature 2: {X_train_scaled[:, 1].min():.2f}")

print("\nMaximum values of the training data before scaling:")
print(f"Feature 1: {X_train[:, 0].max():.2f}, Feature 2: {X_train[:, 1].max():.2f}")
print("\nMaximum values of the training data after scaling:")
print(f"Feature 1: {X_train_scaled[:, 0].max():.2f}, Feature 2: {X_train_scaled[:, 1].max():.2f}")


# 6. Before and After Comparison via Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_train[:, 0], X_train[:, 1])
axes[0].set_title("Original Data")
axes[0].set_xlabel("Feature 1 (Large Scale)")
axes[0].set_ylabel("Feature 2 (Small Scale)")
axes[0].set_xlim(-10, 100)
axes[0].set_ylim(-10, 10)
axes[0].grid(True)

axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1])
axes[1].set_title("MinMaxScaler Data")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].set_xlim(-0.1, 1.1)
axes[1].set_ylim(-0.1, 1.1)
axes[1].grid(True)

plt.tight_layout()
plt.show()

"""
Execution Example:

Minimum values of the training data before scaling:
Feature 1: 29.21, Feature 2: -0.66

Minimum values of the training data after scaling:
Feature 1: 0.00, Feature 2: 0.00

Maximum values of the training data before scaling:
Feature 1: 76.78, Feature 2: 10.91

Maximum values of the training data after scaling:
Feature 1: 1.00, Feature 2: 1.00
"""

"""
Review

Q1. What is the main role of MinMaxScaler, and how does it differ from StandardScaler?

    - MinMaxScaler's role is to transform all feature values into a specific range, typically between 0 and 1.

    - Difference from StandardScaler:
        StandardScaler: Adjusts each feature's mean to 0 and standard deviation to 1. It preserves the shape of the data distribution while adjusting the scale.
        MinMaxScaler: Adjusts each feature's minimum value to 0 and maximum value to 1, re-ranging the values. The shape of the data distribution is preserved.

Q2. Explain why the data distribution is not changed by MinMaxScaler, but only the range of values is readjusted.

    - The transformation formula for MinMaxScaler is (x - min) / (max - min). 
      This formula is a linear transformation. 
      That is, it maintains the relative spacing and the shape of the distribution (e.g., normal distribution, skewed distribution) that the original data had, 
      while only compressing the overall range of values to fit between 0 and 1.

Q3. What kind of problem can occur when using MinMaxScaler if the data contains outliers?

    - MinMaxScaler is very sensitive to outliers because it scales the data based on the minimum and maximum values.
    - If the data contains an extremely large outlier, this outlier will be set as the maximum value (max). 
      Consequently, most of the remaining data will be transformed into very small values close to 0, making it difficult for the features to be properly distinguished.

Q4. What criteria should be used to decide whether to choose MinMaxScaler or StandardScaler?

    StandardScaler:
        It is less sensitive to outliers, providing more stable performance when outliers exist in the data.
        It is generally more suitable for algorithms that assume a Gaussian distribution, such as models using gradient descent (e.g., Logistic Regression, SVM, Neural Networks) or Principal Component Analysis (PCA).

    MinMaxScaler:
        It is used when the exact minimum and maximum values of the data are critical, or when all features must be strictly confined to the same range (e.g., 0 and 1).
        It can be useful in applications like image processing, where data is already defined within a specific range (e.g., 0-255).

    It is generally recommended to try StandardScaler first, and then consider MinMaxScaler if performance is not satisfactory or depending on the specific model's characteristics.
"""
