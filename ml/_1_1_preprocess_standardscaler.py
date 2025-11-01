"""
Preprocessing:
    Transforms the data into a suitable format for the model to learn. This includes handling missing values, data scaling, and feature extraction.
    Example: sklearn.preprocessing.StandardScaler, sklearn.impute.SimpleImputer

StandardScaler is the most commonly used Transformer in the data preprocessing stage.
It standardizes the data by adjusting each feature's mean to 0 and its standard deviation to 1, unifying the scale of the data.
This is crucial in datasets where features have different scales, preventing the model from being biased toward a specific feature.
"""

# 1. Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression  # For comparing model training
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.preprocessing import StandardScaler  # StandardScaler transformer

# 2. Generate example data
# Artificially create a 'scale difference between features' often seen in real data.
# Assume we are building a model to predict apartment prices.
# - 100 data samples (100 apartments)
# - 2 features (Number of rooms, House area)
# X[:, 0]: 'Number of rooms' (small scale, between 0 and 10)
# X[:, 1]: 'House area (m²)' (large scale, between 0 and 1000)
X = np.random.rand(100, 2) * np.array([10, 1000])

# Set the target (y) data simply as the sum of X plus some noise
y = X[:, 0] + X[:, 1] + np.random.randn(100) * 10

# 3. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---
# 4. Apply Standard Scaler
# Create an instance of the Standard Scaler estimator (transformer)
scaler = StandardScaler()

# fit_transform: Calculates (fit) the mean and standard deviation for the training data and then standardizes (transform) the data.
# This step is the most crucial.
X_train_scaled = scaler.fit_transform(X_train)

# transform: The test data is only standardized using the statistics (mean, std dev) from the training data.
# This simulates encountering new data and prevents data leakage.
X_test_scaled = scaler.transform(X_test)

# ---
# 5. Compare Data Before and After Scaling
print("Mean of the training data before scaling:")
print(f"Feature 1: {X_train[:, 0].mean():.2f}, Feature 2: {X_train[:, 1].mean():.2f}")
print("\nMean of the training data after scaling:")
print(f"Feature 1: {X_train_scaled[:, 0].mean():.2f}, Feature 2: {X_train_scaled[:, 1].mean():.2f}")

print("\nStandard deviation of the training data before scaling:")
print(f"Feature 1: {X_train[:, 0].std():.2f}, Feature 2: {X_train[:, 1].std():.2f}")
print("\nStandard deviation of the training data after scaling:")
print(f"Feature 1: {X_train_scaled[:, 0].std():.2f}, Feature 2: {X_train_scaled[:, 1].std():.2f}")

# 6. Before and After Comparison via Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_train[:, 0], X_train[:, 1])
axes[0].set_title("Original Data")
axes[0].set_xlabel("Feature 1 (Small Scale)")
axes[0].set_ylabel("Feature 2 (Large Scale)")
axes[0].axis("equal")

axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1])
axes[1].set_title("StandardScaled Data")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")
axes[1].axis("equal")

plt.tight_layout()
plt.show()

# 7. (Bonus) Comparison of Impact on Model Training
model_raw = LinearRegression().fit(X_train, y_train)
model_scaled = LinearRegression().fit(X_train_scaled, y_train)

# Compare the model coefficients (slopes)
print("\nCoefficients of the model before scaling:", model_raw.coef_)
print("Coefficients of the model after scaling:", model_scaled.coef_)

"""
Execution Result Analysis

Mean of the training data before scaling:
Feature 1: 5.13, Feature 2: 506.97

Mean of the training data after scaling:
Feature 1: -0.00, Feature 2: 0.00

Standard deviation of the training data before scaling:
Feature 1: 2.78, Feature 2: 277.18

Standard deviation of the training data after scaling:
Feature 1: 1.00, Feature 2: 1.00

Coefficients of the model before scaling: [1.36303872 1.00043582]
Coefficients of the model after scaling: [3.79430107 277.30533981]

# Observing the coefficients learned by the model, the coefficient for the 'standardized house area' (277.30) is overwhelmingly larger than the coefficient for the 'standardized number of rooms' (3.79).
# This intuitively shows that "house area" is a much more important feature than "number of rooms" for predicting house prices.
# If scaling were not performed, the importance of the more critical house area could be distorted, leading the model to make poor predictions.
# Therefore, scaling techniques like StandardScaler play a vital role in improving model performance.
"""

"""
Code Review Questions

Q1. What is the main role of StandardScaler, and why is it important to scale data before training a machine learning model?

    - StandardScaler standardizes the data by adjusting each feature's mean to 0 and its standard deviation to 1, unifying the scale of the data.

    - Scaling is important because many machine learning algorithms (especially those based on gradient descent, SVM, PCA, etc.) operate based on the distance between features.
      If features have vastly different scales, the model may become overly sensitive to the larger-scale feature, causing training to be ineffective or performance to degrade.

Q2. Why are the fit_transform() and transform() methods used specifically on the training data and test data, respectively, in StandardScaler?

    - Reason for using fit_transform() on the training data (X_train):
      To **calculate (fit)** the mean and standard deviation of the training data, and then **standardize (transform)** the data using those statistics.
      The model will be trained based on these statistics.
    
    - Reason for using only transform() on the test data (X_test):
      The test data must simulate unknown data, so it must be standardized using the exact same statistics (mean, std dev) obtained from the training data.
      If the statistics of the test data were calculated separately (fit), **Data Leakage** would occur, where information from the test data contaminates the training process, leading to an **overestimation** of model performance (measuring it as better than its true performance).

Q3. In the code, the regression model coefficients (model.coef_) are different before and after scaling. What does this signify?

    - This means that the relative magnitude and values of the coef_ change depending on the scale of the data the model is trained on.

    - Before scaling, the coefficient value for the large-scale feature (Feature 2) can be relatively small.
      Conversely, after scaling, since all features share the same scale, the relative magnitude of each feature's influence on the target value can be compared more **intuitively** using the coefficient values.

Q4. Preprocessing transformers like StandardScaler also have fit() and transform() methods. How can this be explained from the perspective of the Estimator API?

    - All objects in scikit-learn adhere to a consistent API called the **Estimator** API.
      The core of this API includes methods like fit(), predict(), or transform().
    
    - StandardScaler faithfully implements the Estimator API by **learning (fit)** the data's statistics (mean, standard deviation) instead of training a predictive model, and then **transforming** the data using those learned statistics.
      It uses transform() instead of predict() to maintain consistency for the preprocessing step required before model training.
"""
