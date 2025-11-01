"""
Unsupervised Learning:
    Clustering: Used to find the structure or pattern in data and group similar data points together.
        Examples: sklearn.cluster.KMeans, sklearn.cluster.DBSCAN
    Dimensionality Reduction: Reduces the dimension of data to facilitate analysis and visualization.
        Examples: sklearn.decomposition.PCA, sklearn.manifold.TSNE

PCA:
    PCA (Principal Component Analysis) is the most representative unsupervised learning algorithm for dimensionality reduction.
    It is used to reduce dimensions by finding new, **uncorrelated axes (Principal Components)** that maximize the original **Variance** in the data, and then projecting the data onto these axes.
"""

# 1. Import necessary libraries
import time

import matplotlib.pyplot as plt  # Visualization library
import pandas as pd
from sklearn.datasets import load_iris  # Example dataset (Iris)
from sklearn.decomposition import PCA  # PCA dimensionality reduction model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # Data scaling

# 2. Load example data and preprocess
# PCA is sensitive to the scale of features, so it's important to standardize the data.
iris = load_iris()
X = iris.data
y = iris.target  # PCA is unsupervised, but 'y' (target) is used for visualizing the results.

# Data splitting: Data is split first to evaluate the model's generalization performance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ==============================================================================
# Section 2-1: Original Data Visualization (For Reference)
# Since 4D data cannot be directly visualized on a 2D plane,
# we select two representative feature pairs (Sepal/Petal Length/Width) for visualization.
# This helps intuitively understand why PCA is needed.
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Sepal Length vs. Sepal Width
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="viridis", edgecolor="k", s=50)
axes[0].set_title("Original Data (Sepal Features)")
axes[0].set_xlabel(iris.feature_names[0])
axes[0].set_ylabel(iris.feature_names[1])
axes[0].grid(True)

# Petal Length vs. Petal Width
axes[1].scatter(X_train[:, 2], X_train[:, 3], c=y_train, cmap="viridis", edgecolor="k", s=50)
axes[1].set_title("Original Data (Petal Features)")
axes[1].set_xlabel(iris.feature_names[2])
axes[1].set_ylabel(iris.feature_names[3])
axes[1].grid(True)

plt.suptitle("Visualization of Original 4D Iris Data using Feature Pairs", fontsize=16)
plt.show()

# Why PCA is needed
# When visualizing feature pairs of Petal (Length, Width) and Sepal (Length, Width), i.e., 4D data,
# the data often overlaps or has a complex distribution, making it difficult to identify clusters or patterns.
# While 'setosa' (purple) is well-separated, 'versicolor' and 'virginica' overlap.
# By reducing this data to 2 dimensions using PCA, the different classes (Iris species) can be separated more clearly.


# Data Standardization: Scale the data to have a mean of 0 and a standard deviation of 1.
# IMPORTANT: The Scaler must only be fit on the training data (X_train).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform is applied to the test data

# 3. Create Estimator Instance
# Create the PCA model.
# n_components: Specifies the number of dimensions to reduce to (number of principal components).
# Although the Iris data is 4D, we reduce it to 2D for easy visualization.
pca = PCA(n_components=2)

# 4. Model Training and Transformation (fit_transform)
# The model.fit_transform(X) method is used to train the model and transform the data.
# IMPORTANT: PCA must also only be fit on the training data (X_train_scaled).
print("Starting PCA dimensionality reduction...")
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)  # Only transform is applied to the test data
print("PCA dimensionality reduction complete!")
print("\nTraining data dimension (Original):", X_train_scaled.shape)
print("Training data dimension (PCA):", X_train_pca.shape)
print("Test data dimension (PCA):", X_test_pca.shape)

# 5. Visualize Results
# Visualize the data reduced to 2 dimensions.
# By using the original target (y) labels for visualization, we can see how well PCA preserved the variance information.
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap="viridis", edgecolor="k", s=50)
plt.title("PCA Result of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Class")
plt.show()

# 6. Output Explained Variance Ratio
# Check the proportion of variance explained by each principal component.
explained_variance_ratio = pca.explained_variance_ratio_
print(f"Explained variance ratio of each principal component: {explained_variance_ratio}")
print(f"Total explained variance: {sum(explained_variance_ratio):.2f}")

# 7. Check the Relationship between Principal Components and Original Features
# The pca.components_ attribute shows the linear combination of original features that makes up each principal component.
# Rows: Principal Components (PC1, PC2)
# Columns: Original Features (sepal length, sepal width, petal length, petal width)
print("\n--- Relationship between Principal Components and Original Features ---")
components_df = pd.DataFrame(
    pca.components_,
    columns=iris.feature_names,
    index=["Principal Component 1", "Principal Component 2"],
)
print("Contribution (Weight) of original features per principal component:")
print(components_df)
(
    "Interpretation: For example, Principal Component 1 has a positive contribution from all features, "
    "with 'petal length' and 'petal width' having a particularly large influence."
    "Principal Component 2 is relatively more influenced by 'sepal length' and 'sepal width'."
)

# ==============================================================================
# Section 8: Practical Application Example of PCA (Model Performance and Speed Comparison)
# Demonstrate how PCA-reduced data is used in actual model training.
# Train Logistic Regression models using both the original data and the PCA data,
# and compare the accuracy and training time.
# ==============================================================================

print("\n--- [Application Example] Model Performance and Speed Comparison ---")

# 1. Train and predict model with original data (4D)
start_time = time.time()
model_original = LogisticRegression(random_state=42)
model_original.fit(X_train_scaled, y_train)
duration_original = time.time() - start_time

# Predict and evaluate with test data
y_pred_original = model_original.predict(X_test_scaled)
accuracy_original = model_original.score(X_test_scaled, y_test)
print(f"Original Data (4D) - Training Time: {duration_original:.6f} seconds")
print(f"Original Data (4D) - Test Accuracy: {accuracy_original:.4f}")

# 2. Train and predict model with PCA data (2D)
start_time = time.time()
model_pca = LogisticRegression(random_state=42)
model_pca.fit(X_train_pca, y_train)
duration_pca = time.time() - start_time

# Predict and evaluate with test data
y_pred_pca = model_pca.predict(X_test_pca)
accuracy_pca = model_pca.score(X_test_pca, y_test)
print(f"\nPCA Data (2D) - Training Time: {duration_pca:.6f} seconds")
print(f"PCA Data (2D) - Test Accuracy: {accuracy_pca:.4f}")

print(
    "\nConclusion: Although the difference is minimal because the Iris dataset is small, by reducing the number of features by half with PCA,\n    accuracy can be nearly maintained while improving training speed. (Effect is maximized with large-scale data)"
)
print("-" * 50)

# ==============================================================================
# Section 9: Prediction Process for New Data
# The process of predicting a single new data point using the trained Scaler, PCA, and Model.
# ==============================================================================
print("\n--- [Application Example] Prediction for a single new data point ---")

# New data sample (4D)
new_sample = [[5.9, 3.0, 5.1, 1.8]]  # Data point close to the Virginica species
print(f"1. New Original Data: {new_sample} (Data point close to the Virginica species)")

# 2. Transform using the trained Scaler
new_sample_scaled = scaler.transform(new_sample)
print(f"2. Scaled Data: {new_sample_scaled}")

# 3. Transform using the trained PCA
new_sample_pca = pca.transform(new_sample_scaled)
print(f"3. PCA Transformed Data: {new_sample_pca}")

# 4. Predict with the final models
prediction_original = model_original.predict(new_sample_scaled)
prediction_pca = model_pca.predict(new_sample_pca)

print(f"\n5. Original Data Model Prediction Result: {iris.target_names[prediction_original[0]]}")
print(f"   PCA Data Model Prediction Result: {iris.target_names[prediction_pca[0]]}")

"""
PCA dimensionality reduction start...
PCA dimensionality reduction complete!

Training data dimension (Original): (105, 4)
Training data dimension (PCA): (105, 2)
Test data dimension (PCA): (45, 2)
Explained variance ratio of each principal component: [0.7264421  0.23378786]
Total explained variance: 0.96

--- Relationship between Principal Components and Original Features ---
Contribution (Weight) of original features per principal component:
                       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
Principal Component 1           0.530568          -0.240617           0.582340          0.566993
Principal Component 2           0.337784           0.939166           0.024723          0.057082

--- [Application Example] Model Performance and Speed Comparison ---
Original Data (4D) - Training Time: 0.017752 seconds
Original Data (4D) - Test Accuracy: 0.9111

PCA Data (2D) - Training Time: 0.011999 seconds
PCA Data (2D) - Test Accuracy: 0.8889

Conclusion: Although the difference is minimal because the Iris dataset is small, by reducing the number of features by half with PCA,
     accuracy can be nearly maintained while improving training speed. (Effect is maximized with large-scale data)
--------------------------------------------------

--- [Application Example] Prediction for a single new data point ---
1. New Original Data: [[5.9, 3.0, 5.1, 1.8]] (Data point close to the Virginica species)
2. Scaled Data: [[ 0.0310503  -0.12139684  0.74075533  0.76797223]]
3. PCA Transformed Data: [[ 0.91249102 -0.0413724 ]]

5. Original Data Model Prediction Result: virginica
   PCA Data Model Prediction Result: versicolor  <== Incorrect
"""

"""
Review

Q1. Why does PCA belong to unsupervised learning? How is this related to not using 'y' in the fit_transform() method?

    PCA is an algorithm that reduces dimensionality by analyzing the **intrinsic structure (variance)** of the data.
    The true labels (y) are entirely unnecessary for this process.
    Therefore, 'y' is not used in the fit_transform() method, and PCA is classified as unsupervised learning.

Q2. Why is StandardScaler used before applying PCA? What problem can arise if this step is skipped?

    PCA finds principal components based on the **variance** of the data.
    If the features have significantly different scales, the feature with larger variance might be disproportionately considered more important than features with smaller variance.

    By using **StandardScaler** to unify the scale of all features, PCA can fairly capture the true variance structure of the data without being biased toward a specific feature.
    Skipping this step can distort the results.

Q3. What is the role of the n_components parameter in PCA?

    **n_components** specifies the number of dimensions to reduce to, or the number of new **principal components** to create, via PCA.
    For example, setting n_components=2 transforms the original N-dimensional data into 2-dimensional data containing the 2 principal components with the largest variance.

Q4. What does the pca.explained_variance_ratio_ attribute mean, and why is this value important?

    **explained_variance_ratio_** indicates the proportion of the total data variance accounted for by each principal component.
    For example, if the first principal component (Principal Component 1) has a value of 0.95, it means this component explains 95% of the total variance.

    This value is crucial for assessing how well the reduced dimension retains the information of the original data.
    By summing this ratio, one can perform dimensionality reduction while maintaining a desired level of information loss.

Q5. What were the results when Logistic Regression models were trained using the original data and the PCA data, respectively?

    The results of training Logistic Regression models using the original and PCA-transformed data were as follows:
    - The test accuracy of the **Original Data (4D)** model was approximately **0.9111**.
    - The test accuracy of the **PCA Data (2D)** model was approximately **0.8889**.
    Despite the dimensionality reduction via PCA, the accuracy was nearly maintained.

Q6. After applying PCA, how does the prediction process for new data proceed?

    After applying PCA, the prediction process for new data proceeds as follows:
    1. The new data is input in its original scale.
    2. The data is **scaled** using the StandardScaler trained on the training data.
    3. The data is **dimensionally reduced** using the PCA model trained on the training data.
    4. The prediction is performed using the final trained model.

    In the example above, for the new data [[5.9, 3.0, 5.1, 1.8]],
    the Original Data Model predicted 'virginica', while the PCA Data Model predicted 'versicolor'.
    The reason the PCA Data Model made a different prediction is that the data structure was slightly altered due to the PCA transformation.

Q7. What factors would lead an ML researcher to consider using PCA?

    **Solving the Curse of Dimensionality**:
        When the number of features is overwhelmingly large compared to the data size, PCA reduces the number of features the model must learn by eliminating redundant or irrelevant information.
    **Improving Model Training Speed and Memory Efficiency**:
        Using all numerous features in large-scale datasets can lead to very long training times and massive memory consumption.
        Reducing dimensions via PCA shrinks the size of the training data, allowing the model to be trained much faster.
        PCA is sometimes applied to the initial layers of deep learning models to reduce the number of features.
    **Addressing Multicollinearity**:
        For example, if 'number of rooms' and 'number of bathrooms' show high correlation when predicting house prices, PCA can integrate them into a new principal component.
    **Data Visualization and Pattern Discovery**:
        The human eye can only perceive 2D or 3D space. It is impossible to visualize data with hundreds or thousands of features, like genetic or customer behavior data.
        PCA can reduce such high-dimensional data into the 2-3 most important principal components for visualization, allowing for an intuitive understanding of the data's **Clustering** structure or the presence of **Outliers**.
        This is extremely useful in the initial stages of data analysis.
    **Noise Reduction**:
        PCA focuses on directions with high variance to find principal components.
        Principal components with very low variance often represent noise or minor fluctuations in the data.
        By discarding these low-variance components and using only the high-variance ones, noise can be effectively removed while preserving the important patterns in the data.
    **Data Compression and Storage Efficiency**:
        When storing or transmitting large-scale high-dimensional data,
        PCA can significantly reduce the data size by reducing its dimensionality.
        This helps lower storage costs in real-time systems or cloud-based services.

Q8. What are the drawbacks of PCA?

    **Information Loss**: As shown in the Iris example, subtle information important for classification or regression can be lost during dimensionality reduction.
    Researchers must check the `explained_variance_ratio_` to determine what level of information loss is acceptable.

    **Difficulty in Interpretation**: Since the new principal components are linear combinations of multiple original features,
    it is difficult to intuitively interpret "What does Principal Component 1 mean?".
    When the model results need to be explained in business terms, PCA can be a disadvantage.
"""
