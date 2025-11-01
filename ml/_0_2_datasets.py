import matplotlib.pyplot as plt
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    make_blobs,
    make_circles,
    make_classification,
    make_moons,
    make_regression,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# Generate 100 samples for a 2-class classification with 2 informative features
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42,
)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k")
plt.title("make_classification Example")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ---------------------------------------------------------
# Generate 100 samples for regression with 1 feature
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, edgecolor="k")
plt.title("make_regression Example")
plt.xlabel("Feature")
plt.ylabel("Target Value")
plt.show()


# ---------------------------------------------------------
# Generate 200 samples with 3 distinct clusters
X, y = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k")
plt.title("make_blobs Example")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ---------------------------------------------------------
# Generate 200 samples in a crescent moon shape
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k")
plt.title("make_moons Example")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ---------------------------------------------------------
# Generate 200 samples in a concentric circles shape
X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k")
plt.title("make_circles Example")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# ---------------------------------------------------------
# Description: The most representative dataset for classifying iris species.
#              It consists of 3 classes (Setosa, Versicolour, Virginica) and 4 features (sepal/petal length/width).
# Usage: Used as a basic example for multi-class classification algorithms.
iris = load_iris()

# 2. Check data
X = iris.data
y = iris.target

print("--- Iris Dataset ---")
print(f"Data shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature names: {iris.feature_names}")
print(f"Target names: {iris.target_names}")  # type: ignore
# print("\nDataset Description:\n", iris.DESCR) # Uncomment to see the full description
print("-" * 50)


# ---------------------------------------------------------
# Description: Handwritten digit images (8x8 pixels) from 0 to 9, converted into 64-dimensional vectors.
# Usage: Widely used for image classification and testing dimensionality reduction algorithms (PCA, t-SNE).
digits = load_digits()

# 2. Check data
X = digits.data
y = digits.target

print("--- Digits Dataset ---")
print(f"Data shape: {X.shape}")
print(f"Target shape: {y.shape}")

# 3. Visualize the first image data
plt.figure(figsize=(2, 2))
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation="nearest")
plt.title(f"Label: {digits.target[0]}")
plt.show()
print("-" * 50)


# ---------------------------------------------------------
# Description: Dataset for classifying breast cancer diagnosis results into malignant and benign. Includes 30 various medical features.
# Usage: Often used as an example for evaluating binary classification model performance and hyperparameter tuning.
cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

print("--- Breast Cancer Dataset ---")
print(f"Data shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature names (partial): {cancer.feature_names[:5]}")
print(f"Target names: {cancer.target_names}")  # type: ignore
print("-" * 50)


# ---------------------------------------------------------
# Description: Dataset for predicting disease progression one year later based on 10 features (age, sex, BMI, blood pressure, etc.) of diabetic patients.
# Usage: Used to test various regression models such as Linear Regression, Ridge, and Lasso.
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

print("--- Diabetes Dataset ---")
print(f"Data shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature names: {diabetes.feature_names}")
print("-" * 50)


# ---------------------------------------------------------
# Setting as_frame=True allows convenient loading of data directly into a pandas DataFrame.
housing = fetch_california_housing(as_frame=True)

# The housing object includes data (frame), target (target), feature names (feature_names), etc.
X = housing.data
y = housing.target

print("--- California Housing Dataset ---")
print(f"Data shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature names: {housing.feature_names}")
# print("\nDataset Description:\n", housing.DESCR) # Uncomment to see the full description

# Check data as a DataFrame (top 5 rows)
print("\n--- DataFrame Check (Top 5 Rows) ---")
# Since it was loaded with as_frame=True, X is already a DataFrame.
# Combine with the target variable (y) to view the entire data.
df = X.copy()
df["MedHouseVal"] = y
print(df.head())
print("-" * 50)


# 4. Simple Regression Model Training and Evaluation
print("\n--- Linear Regression Model Training and Evaluation Example ---")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
print("Linear Regression Model training complete!")

# Perform prediction on test data
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = model.score(X_test, y_test)

print(f"\nMean Squared Error (MSE) on test data: {mse:.4f}")
print(f"Coefficient of Determination (R-squared) on test data: {r2:.4f}")
print("-" * 50)
